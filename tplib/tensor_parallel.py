from collections import namedtuple
from typing import Any, Dict, List, Optional, Union
import torch; from torch.distributed import ProcessGroup


_TP_GROUP = _DEVICE_WORLD_GROUP = _CPU_WORLD_GROUP = None
_LOCAL_RANK = -1

TensorMetadata = namedtuple("TensorMetadata", ["dtype", "size"])


def get_local_rank():
    global _LOCAL_RANK
    return _LOCAL_RANK

def create_group_and_assign(ranks, backend, global_var):
    group = torch.distributed.new_group(ranks, backend=backend)
    globals()[global_var] = group

def init_dist_env(world_size: int = -1, rank: int = -1, distributed_init_method: str = "env://", local_rank: int = -1, backend: str = "nccl"):
    if not torch.distributed.is_initialized():
        assert distributed_init_method is not None, "distributed_init_method must be provided when initializing distributed environment."
        torch.distributed.init_process_group(backend=backend, init_method=distributed_init_method, world_size=world_size, rank=rank)
        ranks = list(range(torch.distributed.get_world_size()))
        create_group_and_assign(ranks, "gloo", "_CPU_WORLD_GROUP")
        global _LOCAL_RANK; _LOCAL_RANK = local_rank

def init_model_parallel(tp_size: int = 1, backend: Optional[str] = None):
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend()
    if world_size != tp_size: raise RuntimeError(f"world_size ({world_size}) not equal to TP size ({tp_size})")
    num_tp_groups = world_size // tp_size
    assert _TP_GROUP is None, "tensor model parallel group is already initialized"
    for i in range(num_tp_groups):
        ranks = range(i * tp_size, (i + 1) * tp_size)
        create_group_and_assign(ranks, backend, "_TP_GROUP")

def ensure_init(tp_size: int, backend: Optional[str] = None) -> None:
    backend = backend or torch.distributed.get_backend()
    if not model_parallel_is_initialized(): init_model_parallel(tp_size, backend)
    else: assert get_tp_world_size() == tp_size, ("TP group already initialized, but of unexpected size: "
                                                    f"{get_tp_world_size()=} vs. {tp_size=}")

def model_parallel_is_initialized():
    return _TP_GROUP is not None

def get_cpu_world_group():
    return _CPU_WORLD_GROUP if _CPU_WORLD_GROUP is not None else (_CPU_WORLD_GROUP, "CPU world group is not initialized")[0]

def get_tp_group():
    return _TP_GROUP is not None and _TP_GROUP or (_TP_GROUP, "tensor model parallel group is not initialized")[0]

def get_tp_world_size():
    return torch.distributed.get_world_size(group=get_tp_group())

def get_tp_rank():
    return torch.distributed.get_rank(group=get_tp_group())

def get_tp_src_rank():
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tp_world_size()
    return (global_rank // local_world_size) * local_world_size

def destroy_model_parallel():
        global _TP_GROUP
        if _TP_GROUP: torch.distributed.destroy_process_group(_TP_GROUP); _TP_GROUP = None

def tp_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    if get_tp_world_size() == 1: return input_
    else: torch.distributed.all_reduce(input_, group=get_tp_group())
    return input_

def tp_all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    world_size = get_tp_world_size()
    if world_size == 1: return input_
    assert -input_.dim() <= dim < input_.dim(), f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
    if dim < 0: dim += input_.dim()
    input_size = input_.size()
    output_tensor = torch.empty((world_size, ) + input_size, dtype=input_.dtype, device=input_.device)
    torch.distributed.all_gather_into_tensor(output_tensor, input_, group=get_tp_group())
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(input_size[:dim] + (world_size * input_size[dim], ) + input_size[dim + 1:])
    return output_tensor

def tp_gather(input_: torch.Tensor, dst: int = 0, dim: int = -1) -> torch.Tensor:
    world_size = get_tp_world_size()
    if world_size == 1: return input_
    assert -input_.dim() <= dim < input_.dim(), f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
    if dim < 0: dim += input_.dim()
    if get_tp_rank() == dst: gather_list = [torch.empty_like(input_) for _ in range(world_size)]
    else: gather_list = None
    torch.distributed.gather(input_, gather_list, dst=dst, group=get_tp_group())
    if get_tp_rank() == dst: output_tensor = torch.cat(gather_list, dim=dim)
    else: output_tensor = None
    return output_tensor

def broadcast_common(src: int = 0, group: Optional[ProcessGroup] = None):
    group = group or torch.distributed.group.WORLD
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"
    world_size = torch.distributed.get_world_size(group=group)
    return world_size, group

def broadcast(input_: torch.Tensor, src: int = 0, group: Optional[ProcessGroup] = None):
    world_size, group = broadcast_common(src, group)
    if world_size == 1: return input_
    torch.distributed.broadcast(input_, src=src, group=group)
    return input_

def broadcast_obj_list(obj_list: List[Any], src: int = 0, group: Optional[ProcessGroup] = None):
    world_size, group = broadcast_common(src, group)
    if world_size == 1: return obj_list
    torch.distributed.broadcast_object_list(obj_list, src=src, group=group)
    return obj_list

def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0, group: Optional[ProcessGroup] = None
    ) -> Dict[Any, Union[torch.Tensor, Any]]:
    world_size, group = broadcast_common(src, group)
    if world_size == 1: return tensor_dict
    rank = torch.distributed.get_rank()
    if rank == src:
        assert isinstance(tensor_dict, dict), f"Expecting a dictionary, got {type(tensor_dict)}"
        metadata_list = []
        for k, v in tensor_dict.items():
            if isinstance(v, torch.Tensor):
                assert v.is_cuda, f"Tensor {k}: {v} is not on CUDA. Currently we only support broadcasting tensors on CUDA."
                metadata_list.append((k, TensorMetadata(v.dtype, v.size())))
            else:
                metadata_list.append((k, v))
        torch.distributed.broadcast_object_list([metadata_list], src=src, group=group)
        async_handles = []
        for k, v in metadata_list:
            if isinstance(v, TensorMetadata):
                tensor = tensor_dict[k]
                async_handles.append(torch.distributed.broadcast(tensor, src=src, group=group, async_op=True))
        for async_handle in async_handles: async_handle.wait()
    else:
        recv_metadata_list = [None]
        torch.distributed.broadcast_object_list(recv_metadata_list, src=src, group=group)
        metadata_list = recv_metadata_list[0]
        tensor_dict = {}
        async_handles = []
        for k, v in metadata_list:
            if isinstance(v, TensorMetadata):
                tensor = torch.empty(v.size, dtype=v.dtype, device="cuda")
                async_handle = torch.distributed.broadcast(tensor, src=src, async_op=True, group=group)
                async_handles.append(async_handle)
                tensor_dict[k] = tensor
            else:
                tensor_dict[k] = v
        for async_handle in async_handles: async_handle.wait()
    return tensor_dict
