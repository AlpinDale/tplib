import contextlib, torch; from typing import Optional

_TENSOR_MODEL_PARALLEL_GROUP = _DEVICE_WORLD_GROUP = _CPU_WORLD_GROUP = None
_LOCAL_RANK = -1

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

def init_model_parallel(tensor_model_parallel_size: int = 1, backend: Optional[str] = None):
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend()
    if world_size != tensor_model_parallel_size: raise RuntimeError(f"world_size ({world_size}) not equal to TP size ({tensor_model_parallel_size})")
    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    rank = torch.distributed.get_rank()
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, "tensor model parallel group is already initialized"
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        create_group_and_assign(ranks, backend, "_TENSOR_MODEL_PARALLEL_GROUP")

def ensure_init(tensor_model_parallel_size: int, backend: Optional[str] = None) -> None:
        backend = backend or torch.distributed.get_backend()
        if not model_parallel_is_initialized(): init_model_parallel(tensor_model_parallel_size, backend)
        else: assert get_tensor_model_parallel_world_size() == tensor_model_parallel_size, ("TP group already initialized, but of unexpected size: "
                                                                                             f"{get_tensor_model_parallel_world_size()=} vs. "
                                                                                             f"{tensor_model_parallel_size=}")

def model_parallel_is_initialized():
    return _TENSOR_MODEL_PARALLEL_GROUP is not None

def get_cpu_world_group():
    return _CPU_WORLD_GROUP if _CPU_WORLD_GROUP is not None else (_CPU_WORLD_GROUP, "CPU world group is not initialized")[0]

def get_tensor_model_parallel_group():
    return _TENSOR_MODEL_PARALLEL_GROUP is not None and _TENSOR_MODEL_PARALLEL_GROUP or (_TENSOR_MODEL_PARALLEL_GROUP,
                                                                                         "tensor model parallel group is not initialized")[0]

def get_tensor_model_parallel_world_size():
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())

def get_tensor_model_parallel_rank():
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())

def get_tensor_model_parallel_src_rank():
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size

def destroy_model_parallel():
        global _TENSOR_MODEL_PARALLEL_GROUP
        if _TENSOR_MODEL_PARALLEL_GROUP: torch.distributed.destroy_process_group(_TENSOR_MODEL_PARALLEL_GROUP); _TENSOR_MODEL_PARALLEL_GROUP = None
