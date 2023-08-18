r""""Contains definitions of the methods used by the _BaseDataLoaderIter workers.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import torch
import random
import os
import time
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
                 num_workers, persistent_workers, super_batch_size, process_raw,
                 internal_buffer, output_status, timing_file, timing_lock):
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
        final_signal = False

        watchdog = ManagerWatchdog()

        timing = {
            'data_request':[],      # Requests.
            'data_readback':[],       # Readback.
            'internal_to_output':[],
            'index_queue_get':[],
            'worked':[],
            'worker_load_preload':[],
        }


        preloaded = False
        all_idx = None
        all_index = None
        worked = False
        worked_start = time.time()
        while watchdog.is_alive() and (not final_signal or internal_buffer.qsize() > 0):                                                                          # REWORK
            if worked:
                timing["worked"].append((worked_start, time.time() - worked_start))
                worked = False
            worked_start = time.time()

            # Always keep 1 processed data ready to go in the result queue.
            status = output_status[worker_id].value
            qsize = internal_buffer.qsize()
            if status and qsize > 0: # _output_status[i] checks whether this worker is allowed to insert into the output queue
                # Take an item from the internal buffer, process it, and put
                # it into the output buffer.

                worked = True

                internal_to_output_start = time.time()
                idx, buffered = internal_buffer.get()
                processed = [process_raw(dataset, raw_data, target) for target, raw_data in buffered]
                data_queue.put((worker_id, (idx, collate_fn(processed))))
                with output_status[worker_id].get_lock():
                    output_status[worker_id].value = False

                timing['internal_to_output'].append((internal_to_output_start, time.time() - internal_to_output_start))
            
            # Check if the queue is empty and we've got a new superbatch preloaded
            if internal_buffer.qsize() == 0 and preloaded:
                worked = True

                t = time.time()
                preloaded = False
                data_readback_start = time.time()
                all_unprocessed_data = fetcher.readback(all_index)
                timing['data_readback'].append((data_readback_start, time.time() - data_readback_start))
                for idx, unprocessed_data in zip(all_idx, all_unprocessed_data):
                    # Tuple(idx, Tuple(target, data))
                    internal_buffer.put((idx, unprocessed_data))
                timing["worker_load_preload"].append((t, time.time() - t))
                continue

            # Check if we need to start the next preload.
            if preloaded or final_signal:
                continue
            worked = True

            # Get a list of <= SUPER_BATCH_SIZE batches.
            all_idx = [] # Indices of the batches themselves.
            all_index = [] # Batched indices of the files to be loaded.
            for i in range(super_batch_size):
                try:
                    t = time.time()
                    r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
                    timing['index_queue_get'].append((t, time.time() - t))

                    # Perform the regular checks.
                    if isinstance(r, _ResumeIteration):
                        # Acknowledge the main process
                        data_queue.put((None, (r, None)))
                        iteration_end = False

                        # Re-create fetcher for worker re-use policy.
                        fetcher = _DatasetKind.create_fetcher(
                            dataset_kind,
                            worker_id,
                            dataset,
                            auto_collation,
                            collate_fn,
                            drop_last
                        )

                        # Continue filling superbatch.
                        continue
                    elif r is None:
                        # Received final signal. Verify conditions.
                        assert done_event.is_set() or iteration_end
                        final_signal = True
                        break
                    elif done_event.is_set() or iteration_end or r == -1:
                        # Continue to wait for the final signal.
                        break
                    else:
                        # If it wasn't a special case, append to be loaded.
                        idx, index = r
                        if len(index) > 0:
                            all_idx.append(idx)
                            all_index.append(index)
                except queue.Empty:
                    print("index_queue.get() timeout")
                    pass

            # Don't handle this yet...
            if init_exception is not None:
                assert False

            if not all_index:
                continue
            
            # In the form of List[Tuple(target, data)]
            data_request_start = time.time()
            fetcher.request(all_index)
            timing['data_request'].append((data_request_start, time.time() - data_request_start))
            preloaded = True

    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass

    # Write output to our file
    timing_lock.acquire()
    for key in timing:
        for val in timing[key]:
            timing_file.write("{},{},{},{}\n".format(
                worker_id,
                key,    # label
                val[0], # time
                val[1]  # duration
            ))
    timing_file.close()
    timing_lock.release()

    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()
