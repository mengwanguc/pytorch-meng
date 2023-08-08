r""""Contains definitions of the methods used by the _BaseDataLoaderIter workers.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import torch
import random
import os
from dataclasses import dataclass
from torch._six import queue
from torch._utils import ExceptionWrapper
from typing import Union, List
from . import signal_handling, MP_STATUS_CHECK_INTERVAL, IS_WINDOWS

if IS_WINDOWS:
    import ctypes
    from ctypes.wintypes import DWORD, BOOL, HANDLE

    # On Windows, the parent ID of the worker process remains unchanged when the manager process
    # is gone, and the only way to check it through OS is to let the worker have a process handle
    # of the manager and ask if the process status has changed.
    class ManagerWatchdog(object):
        def __init__(self):
            self.manager_pid = os.getppid()

            # mypy cannot detect this code is windows only
            self.kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)  # type: ignore
            self.kernel32.OpenProcess.argtypes = (DWORD, BOOL, DWORD)
            self.kernel32.OpenProcess.restype = HANDLE
            self.kernel32.WaitForSingleObject.argtypes = (HANDLE, DWORD)
            self.kernel32.WaitForSingleObject.restype = DWORD

            # Value obtained from https://msdn.microsoft.com/en-us/library/ms684880.aspx
            SYNCHRONIZE = 0x00100000
            self.manager_handle = self.kernel32.OpenProcess(SYNCHRONIZE, 0, self.manager_pid)

            if not self.manager_handle:
                raise ctypes.WinError(ctypes.get_last_error())  # type: ignore

            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                # Value obtained from https://msdn.microsoft.com/en-us/library/windows/desktop/ms687032.aspx
                self.manager_dead = self.kernel32.WaitForSingleObject(self.manager_handle, 0) == 0
            return not self.manager_dead
else:
    class ManagerWatchdog(object):  # type: ignore[no-redef]
        def __init__(self):
            self.manager_pid = os.getppid()
            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                self.manager_dead = os.getppid() != self.manager_pid
            return not self.manager_dead

_worker_info = None


class WorkerInfo(object):
    __initialized = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__keys = tuple(kwargs.keys())
        self.__initialized = True

    def __setattr__(self, key, val):
        if self.__initialized:
            raise RuntimeError("Cannot assign attributes to {} objects".format(self.__class__.__name__))
        return super(WorkerInfo, self).__setattr__(key, val)

    def __repr__(self):
        items = []
        for k in self.__keys:
            items.append('{}={}'.format(k, getattr(self, k)))
        return '{}({})'.format(self.__class__.__name__, ', '.join(items))


def get_worker_info():
    r"""Returns the information about the current
    :class:`~torch.utils.data.DataLoader` iterator worker process.

    When called in a worker, this returns an object guaranteed to have the
    following attributes:

    * :attr:`id`: the current worker id.
    * :attr:`num_workers`: the total number of workers.
    * :attr:`seed`: the random seed set for the current worker. This value is
      determined by main process RNG and the worker id. See
      :class:`~torch.utils.data.DataLoader`'s documentation for more details.
    * :attr:`dataset`: the copy of the dataset object in **this** process. Note
      that this will be a different object in a different process than the one
      in the main process.

    When called in the main process, this returns ``None``.

    .. note::
       When used in a :attr:`worker_init_fn` passed over to
       :class:`~torch.utils.data.DataLoader`, this method can be useful to
       set up each worker process differently, for instance, using ``worker_id``
       to configure the ``dataset`` object to only read a specific fraction of a
       sharded dataset, or use ``seed`` to seed other libraries used in dataset
       code (e.g., NumPy).
    """
    return _worker_info


r"""Dummy class used to signal the end of an IterableDataset"""
@dataclass(frozen=True)
class _IterableDatasetStopIteration(object):
    worker_id: int

r"""Dummy class used to resume the fetching when worker reuse is enabled"""
@dataclass(frozen=True)
class _ResumeIteration(object):
    pass

def _worker_loop(dataset_kind, dataset, index_queue, data_queue, done_event,
                 auto_collation, collate_fn, drop_last, seed, init_fn, worker_id,
                 num_workers, persistent_workers, prefetch_factor, super_batch):
    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.

    try:
        # Initialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
        # module's handlers are executed after Python returns from C low-level
        # handlers, likely when the same fatal signal had already happened
        # again.
        # https://docs.python.org/3/library/signal.html#execution-of-python-signal-handlers
        signal_handling._set_worker_signal_handlers()

        torch.set_num_threads(1)
        random.seed(seed)
        torch.manual_seed(seed)

        global _worker_info
        _worker_info = WorkerInfo(id=worker_id, num_workers=num_workers,
                                  seed=seed, dataset=dataset)

        from torch.utils.data import _DatasetKind

        init_exception = None

        try:
            if init_fn is not None:
                init_fn(worker_id)

            fetcher = _DatasetKind.create_fetcher(dataset_kind, worker_id, dataset, auto_collation, collate_fn, drop_last)
        except Exception:
            init_exception = ExceptionWrapper(
                where="in DataLoader worker process {}".format(worker_id))

        # When using Iterable mode, some worker can exit earlier than others due
        # to the IterableDataset behaving differently for different workers.
        # When such things happen, an `_IterableDatasetStopIteration` object is
        # sent over to the main process with the ID of this worker, so that the
        # main process won't send more tasks to this worker, and will send
        # `None` to this worker to properly exit it.
        #
        # Note that we cannot set `done_event` from a worker as it is shared
        # among all processes. Instead, we set the `iteration_end` flag to
        # signify that the iterator is exhausted. When either `done_event` or
        # `iteration_end` is set, we skip all processing step and just wait for
        # `None`.
        iteration_end = False

        watchdog = ManagerWatchdog()

        final_signal = False
        while watchdog.is_alive() and not final_signal:                                                         # REWORK
            print("Worker loop (pid {}); final_signal = {}".format(os.getpid(), final_signal))

            to_load = []

            # Fetch up to SUPER_BATCH batched indices from the queue
            try:
                for _ in range(super_batch):
                    indices = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
                    to_load.append(indices)

                    if indices == None:
                        print("Saw a None!")
                        break
            except queue.Empty:
                if len(to_load) == 0:
                    continue
            
            # Combine everything so we can give the loader one big batch.
            all_idx = [] # index of each batch
            all_index = [] # batch indices

            # Handle special cases. If this is the resume iteration, create the
            # fetcher and remove the resume iteration from the to_load list. If
            # this is the final iteration, cleanly shut down.
            for i, indices in enumerate(to_load):
                print("(pid {}) i = {}, indices = {}".format(os.getpid(), i, indices))

                if isinstance(indices, _ResumeIteration):
                    # Acknowledge the main process
                    data_queue.put((indices, None))
                    iteration_end = False
                    # Recreate the fetcher for worker-reuse policy
                    fetcher = _DatasetKind.create_fetcher(                                                      # DELETE?
                        dataset_kind,
                        worker_id,
                        dataset,
                        auto_collation,
                        collate_fn,
                        drop_last
                    )
                    continue
                elif indices is None:
                    print("Parsed a None!")

                    # Received the final signal
                    assert done_event.is_set() or iteration_end
                    final_signal = True
                    break
                elif done_event.is_set() or iteration_end:
                    # `done_event` is set. But I haven't received the final signal
                    # (None) yet. I will keep continuing until get it, and skip the
                    # processing steps.
                    continue

                # If they're OK, add to the "all-" lists for loading
                all_idx.append(indices[0])
                all_index.append(indices[1])

            if len(all_index) == 0:
                continue

            all_data: List[Union[_IterableDatasetStopIteration, ExceptionWrapper]]
            if init_exception is not None:
                all_data = [init_exception]
                init_exception = None
            else:
                all_data, e = fetcher.fetch(all_index)
                if e != None:
                    if isinstance(e, StopIteration) and dataset_kind == _DatasetKind.Iterable:
                        all_data[-1] = _IterableDatasetStopIteration(worker_id)
                        # Set `iteration_end`
                        #   (1) to save future `next(...)` calls, and
                        #   (2) to avoid sending multiple `_IterableDatasetStopIteration`s.
                        iteration_end = True
                    else:
                        # It is important that we don't store exc_info in a variable.
                        # `ExceptionWrapper` does the correct thing.
                        # See NOTE [ Python Traceback Reference Cycle Problem ]
                        all_data[-1] = ExceptionWrapper(
                            where="in DataLoader worker process {}".format(worker_id))
                        
            for idx, data in zip(all_idx, all_data):
                data_queue.put((idx, data))

            # del all_data, data, all_idx, all_index, idx # save memory
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass
    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()
