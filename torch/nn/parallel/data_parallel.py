import operator
import torch
import warnings
from itertools import chain
from ..modules import Module
from .scatter_gather import scatter_kwargs, gather
from .replicate import replicate
from .parallel_apply import parallel_apply
from torch._utils import (
    _get_all_device_indices,
    _get_available_device_type,
    _get_device_index,
    _get_devices_properties
)
import time
def _check_balance(device_ids):
    imbalance_warn = """
    There is an imbalance between your GPUs. You may want to exclude GPU {} which
    has less than 75% of the memory or cores of GPU {}. You can do so by setting
    the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES
    environment variable."""
    device_ids = [_get_device_index(x, True) for x in device_ids]
    dev_props = _get_devices_properties(device_ids)

    def warn_imbalance(get_prop):
        values = [get_prop(props) for props in dev_props]
        min_pos, min_val = min(enumerate(values), key=operator.itemgetter(1))
        max_pos, max_val = max(enumerate(values), key=operator.itemgetter(1))
        if min_val / max_val < 0.75:
            warnings.warn(imbalance_warn.format(device_ids[min_pos], device_ids[max_pos]))
            return True
        return False

    if warn_imbalance(lambda props: props.total_memory):
        return
    if warn_imbalance(lambda props: props.multi_processor_count):
        return


class DataParallel(Module):
    r"""Implements data parallelism at the module level.

    This container parallelizes the application of the given :attr:`module` by
    splitting the input across the specified devices by chunking in the batch
    dimension (other objects will be copied once per device). In the forward
    pass, the module is replicated on each device, and each replica handles a
    portion of the input. During the backwards pass, gradients from each replica
    are summed into the original module.

    The batch size should be larger than the number of GPUs used.

    .. warning::
        It is recommended to use :class:`~torch.nn.parallel.DistributedDataParallel`,
        instead of this class, to do multi-GPU training, even if there is only a single
        node. See: :ref:`cuda-nn-ddp-instead` and :ref:`ddp`.

    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel but some types are specially handled. tensors will be
    **scattered** on dim specified (default 0). tuple, list and dict types will
    be shallow copied. The other types will be shared among different threads
    and can be corrupted if written to in the model's forward pass.

    The parallelized :attr:`module` must have its parameters and buffers on
    ``device_ids[0]`` before running this :class:`~torch.nn.DataParallel`
    module.

    .. warning::
        In each forward, :attr:`module` is **replicated** on each device, so any
        updates to the running module in ``forward`` will be lost. For example,
        if :attr:`module` has a counter attribute that is incremented in each
        ``forward``, it will always stay at the initial value because the update
        is done on the replicas which are destroyed after ``forward``. However,
        :class:`~torch.nn.DataParallel` guarantees that the replica on
        ``device[0]`` will have its parameters and buffers sharing storage with
        the base parallelized :attr:`module`. So **in-place** updates to the
        parameters or buffers on ``device[0]`` will be recorded. E.g.,
        :class:`~torch.nn.BatchNorm2d` and :func:`~torch.nn.utils.spectral_norm`
        rely on this behavior to update the buffers.

    .. warning::
        Forward and backward hooks defined on :attr:`module` and its submodules
        will be invoked ``len(device_ids)`` times, each with inputs located on
        a particular device. Particularly, the hooks are only guaranteed to be
        executed in correct order with respect to operations on corresponding
        devices. For example, it is not guaranteed that hooks set via
        :meth:`~torch.nn.Module.register_forward_pre_hook` be executed before
        `all` ``len(device_ids)`` :meth:`~torch.nn.Module.forward` calls, but
        that each such hook be executed before the corresponding
        :meth:`~torch.nn.Module.forward` call of that device.

    .. warning::
        When :attr:`module` returns a scalar (i.e., 0-dimensional tensor) in
        :func:`forward`, this wrapper will return a vector of length equal to
        number of devices used in data parallelism, containing the result from
        each device.

    .. note::
        There is a subtlety in using the
        ``pack sequence -> recurrent network -> unpack sequence`` pattern in a
        :class:`~torch.nn.Module` wrapped in :class:`~torch.nn.DataParallel`.
        See :ref:`pack-rnn-unpack-with-data-parallelism` section in FAQ for
        details.


    Args:
        module (Module): module to be parallelized
        device_ids (list of int or torch.device): CUDA devices (default: all devices)
        output_device (int or torch.device): device location of output (default: device_ids[0])

    Attributes:
        module (Module): the module to be parallelized

    Example::

        >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)  # input_var can be on any device, including CPU
    """

    # TODO: update notes/cuda.rst when this class handles 8+ GPUs well

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__()

        device_type = _get_available_device_type()
        if device_type is None:
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = _get_all_device_indices()

        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.device_ids = [_get_device_index(x, True) for x in device_ids]
        print("torch/nn/parallel/data_parallel.py: device_ids: {}".format(self.device_ids))
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device(device_type, self.device_ids[0])

        _check_balance(self.device_ids)

        if len(self.device_ids) == 1:
            self.module.to(self.src_device_obj)

    def forward(self, *inputs, **kwargs):
        print("I got into forward funciton of DataParallel! ! !, the ids are {}".format(self.device_ids))
        print("Note that the images are not divided here yet. ")
        print("The length of inputs is {}, and the length of kwargs is {}".format(len(inputs),len(kwargs)))
        print("Check if the image is on CUDA: {}".format(inputs[0].is_cuda))
        # print("This is inputs: {}".format(inputs))
        # if there is no devices. 
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        # used to rise an error
        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        torch.cuda.synchronize()
        time_bef_scat=time.time()
        torch.cuda.synchronize()

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)

        torch.cuda.synchronize()
        time_aft_scat=time.time()
        torch.cuda.synchronize()
        time_ins_scat=time_aft_scat-time_bef_scat
        print("The total time spent on scatter is: {}".format(time_ins_scat))
        # for forward function without any inputs, empty list and dict will be created
        # so the module can be executed on one device which is the first one in device_ids
        if not inputs and not kwargs:
            print("This is when both inputs and kwargs are None")
            inputs = ((),)
            kwargs = ({},)
        # once got here, neither inputs nor kwargs are none
        print("After scatter")
        print("The length of inputs is {}, and the length of kwargs is {}".format(len(inputs),len(kwargs)))

        print("Two inputs, the first one is in CUDA: {}".format(inputs[0][0].is_cuda))
        print("Two inputs, the second one is in CUDA: {}".format(inputs[1][0].is_cuda))
        # print below shows that kwargs is empty, even though it's length is 2. 
        # print("After scatter, this is kwargs: {}".format(kwargs))
        # if there is only 1 device, then just treat it like there is only 1. 
        if len(self.device_ids) == 1:
            # 1 GPU doesn't come here. 
            return self.module(*inputs[0], **kwargs[0])
        
        # the following case if when there are two devices
        print("If I have 2 devices I come here. I want to know the time of the following. ")
        torch.cuda.synchronize()
        time_bef_rep=time.time()
        torch.cuda.synchronize()

        # self.module is the Resnet architecture. 
        # It shows self.device_ids[:len(inputs)] = [0, 1]

        # It seems to me that resnet model cannot just be copied by .copy(), so PyTorch
        # had to enumerate through self.module. At the end, they give the same # of replicas 
        # as the length of the # of devices. 
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        # The len of replicas is 2
        # self.module==replica[0]==replica[1]
        # two replicas are not the same. 
        # self.module's text is the same as replicas[0] and replicas[1] by using text comparator. 
        # every time the model's update is put together, then we use th eupdated module to send to GPUs. 
        torch.cuda.synchronize()
        time_bef_par=time.time()
        torch.cuda.synchronize()
        print("The time took for replicate is: {}".format(time_bef_par-time_bef_rep))

        print("len of inputs: {}, kwargs are: {}".format(len(inputs), kwargs))
        # kwargs is empty
        # given 2 modules' replica and two inputs to assign 2 threads to each GPU. 
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        # len(outputs) = 2
        # check out what are the outputs, what are they? 

        torch.cuda.synchronize()
        time_bef_gat=time.time()
        torch.cuda.synchronize()
        print("The time took for parallel_apply is: {}".format(time_bef_gat-time_bef_par))

        # what does gather do: take tensors from different GPUs onto a specified device. 
        # Not sure which device yet. Notice that outputs is plural. 2 outputs. 
        # The self.output_device is: 0
        # The shape of the outputs is: [0]: torch.Size([128, 1000]), [1]: torch.Size([128, 1000])
        print(outputs)
        temp = self.gather(outputs, self.output_device)
        # print("The output of gather is: {}".format(temp))
        # output is on the first GPU. 
        torch.cuda.synchronize()
        time_aft_gat=time.time()
        torch.cuda.synchronize()
        print("The time took for gather is: {}".format(time_aft_gat-time_bef_gat))
    
        print("The time took extra here for 2 GPUs is: {}".format(time_aft_gat-time_bef_rep))
        return temp

    def replicate(self, module, device_ids):
        return replicate(module, device_ids, not torch.is_grad_enabled())

    def scatter(self, inputs, kwargs, device_ids):
        print("I got into function scatter in data_parallel.py. ")
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


def data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None):
    r"""Evaluates module(input) in parallel across the GPUs given in device_ids.

    This is the functional version of the DataParallel module.

    Args:
        module (Module): the module to evaluate in parallel
        inputs (Tensor): inputs to the module
        device_ids (list of int or torch.device): GPU ids on which to replicate module
        output_device (list of int or torch.device): GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Tensor containing the result of module(input) located on
        output_device
    """
    # print("Wonder if it gets here: data_parallel. ")
    # No it did not come here. 
    if not isinstance(inputs, tuple):
        inputs = (inputs,) if inputs is not None else ()

    device_type = _get_available_device_type()

    if device_ids is None:
        device_ids = _get_all_device_indices()

    if output_device is None:
        output_device = device_ids[0]

    device_ids = [_get_device_index(x, True) for x in device_ids]
    output_device = _get_device_index(output_device, True)
    src_device_obj = torch.device(device_type, device_ids[0])

    for t in chain(module.parameters(), module.buffers()):
        if t.device != src_device_obj:
            raise RuntimeError("module must have its parameters and buffers "
                               "on device {} (device_ids[0]) but found one of "
                               "them on device: {}".format(src_device_obj, t.device))

    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    # for module without any inputs, empty list and dict will be created
    # so the module can be executed on one device which is the first one in device_ids
    if not inputs and not module_kwargs:
        inputs = ((),)
        module_kwargs = ({},)

    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[:len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    return gather(outputs, output_device, dim)
