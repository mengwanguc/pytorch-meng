r""""Contains definitions of the methods used by the _BaseDataLoaderIter to put
fetched tensors into pinned memory.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import torch
from torch._six import queue, container_abcs, string_classes
from . import MP_STATUS_CHECK_INTERVAL
from torch._utils import ExceptionWrapper

import time
import sys
import mlock


def _pin_memory_loop(in_queue, out_queue, device_id, done_event, max_output_length, output_status):
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)

    torch.cuda.set_device(device_id)

    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    while not done_event.is_set():
        if out_queue.qsize() >= max_output_length:
            continue

        try:
            worker_id, r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            if worker_id == None:
                continue
            with output_status[worker_id].get_lock():
                output_status[worker_id].value = True
        except queue.Empty:
            continue
        idx, data = r
        if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
            try:
                data = pin_memory(data)
            except Exception:
                data = ExceptionWrapper(
                    where="in pin memory thread for device {}".format(device_id))
            r = (idx, data)
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue
        del r  # save memory

def _emulate_pin_memory_loop(in_queue, out_queue, device_id, done_event, estimated_pin_mem_time, balloons, max_output_length, output_status, timing_file, timing_file_lock):
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)


    timing = {
        "pin_memory":[],
        "pin_memory_whole":[],
    }

    since_last_pin = time.time()
    while not done_event.is_set():
        if out_queue.qsize() >= max_output_length:
            continue

        try:
            worker_id, r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            if worker_id == None:
                continue
            with output_status[worker_id].get_lock():
                output_status[worker_id].value = True
        except queue.Empty:
            continue
        
        pin_memory_time_start = time.time()

        idx, data = r
        if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
            try:
                elapsed_time = 0
                pin_start = time.time()

                # Note: in the future we may want to actually copy the memory in
                # addition to just allocating it and simulating the copy time.
                # However, the current method seems fine for the moment.

                # balloons is a dict, with PyBalloons organized by size.
                for elem in data:
                    size = elem.nelement() * elem.element_size()
                    balloon = None

                    has_balloon = False
                    if size in balloons:

                        # Look through the existing balloons and try to claim
                        # one of the same size.
                        for balloon in balloons[size]:
                            if not balloon.get_used():
                                balloon.set_used(True)
                                has_balloon = True
                                break
                        
                        # If we didn't find a free balloon in the list, create a
                        # new one and add it to the list.
                        if not has_balloon:
                            balloon = mlock.PyBalloon(size)
                            balloon.set_used(True)
                            balloons[size].append(balloon)

                    else:
                        # If there's no key yet, create a balloon and make it
                        # the first element in the list for that size.

                        balloon = mlock.PyBalloon(size)
                        balloon.set_used(True)
                        balloons[size] = [balloon]

                data = [None for _ in data]
                alloc_end = time.time()
                # print("time to alloc: {} ms".format((alloc_end - pin_start) * 1000))
                while elapsed_time < estimated_pin_mem_time:
                    elapsed_time = time.time() - pin_start
            except Exception:
                data = ExceptionWrapper(
                    where="in emulation pin memory thread")
            r = (idx, data)
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue
        
        cur_time = time.time()
        timing["pin_memory"].append((pin_memory_time_start, cur_time - pin_memory_time_start))
        timing["pin_memory_whole"].append((since_last_pin, cur_time - since_last_pin))
        since_last_pin = time.time()
        
        del r  # save memory
    
    timing_file_lock.acquire()
    for key in timing:
        for start, duration in timing[key]:
            timing_file.write("{},{},{},{}\n".format(-1, "pin_memory", start, duration))
    timing_file_lock.release()




def pin_memory(data):
    if isinstance(data, torch.Tensor):
        return data.pin_memory()
    elif isinstance(data, string_classes):
        return data
    elif isinstance(data, container_abcs.Mapping):
        return {k: pin_memory(sample) for k, sample in data.items()}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return type(data)(*(pin_memory(sample) for sample in data))
    elif isinstance(data, container_abcs.Sequence):
        return [pin_memory(sample) for sample in data]
    elif hasattr(data, "pin_memory"):
        return data.pin_memory()
    else:
        return data
