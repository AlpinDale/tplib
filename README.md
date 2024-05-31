# Tensor Parallelism

This lightweight library provides a small set of utilities for working with distributed tensors in PyTorch. It includes methods for initializing the distributed environment, creating and destroying tensor parallel groups, and performing various collective operations such as all-reduce, all-gather, and broadcast.

## Installation

```sh
git clone https://github.com/AlpinDale/tplib.git && cd tplib
pip install -e .
```

## Usage

First, you will need to initialize a distributed environment. You can use the `init_dist_env` function; it requires the world size (total number of process, e.g. GPUs), rank of the current processes (e.g. machine number), and the method to initialize the distributed environment. The method can be a string specifying a URL like `tcp://localhost:23456` or `env://` to use environment variables.

```py
from tplib import init_dist_env

init_dist_env(world_size=2, rank=0, distributed_init_method="tcp://localhost:23456")
```

> [!WARNING]
> Running `init_dist_env` in a non-distributed environment (i.e., a single process) will cuase the program to hang indefinitely. This is because `torch.distributed.init_process_group()` expects to be run in multiple processes and waits for other processes to join the group. To avoid this, you should either use the built-in `Worker` class (not recommended) or implemented your own method of spawning and handling processes. You can also use a library such as [Ray](https://github.com/ray-project/ray) to handle this.

### The `Worker` class

Provides a very simple way to spawn multiple processes and run a function in each of them.

```py
from tplib import Worker

def print_rank(rank):
    print(f"I am process {rank}")

worker = Worker(world_size=2)

worker.run(print_rank)
```

### Tensor Parallel Groups
To create a tensor parallel group, you can use the `init_model_parallel` function. This function requires the size of the TP group (number of processes) and optionally the backend.

```py
from tplib import init_model_parallel

init_model_parallel(tp_size=2)
```

To check if a TP group has been initialized, you can use the `model_parallel_is_initialized` function:

```py
from tplib import model_parallel_is_initialized

if model_parallel_is_initialized():
    return True
else:
    return False
```

You can destroy the group by calling the `destroy_model_parallel()` function.

### Collective Operations
- `tp_all_reduce`: Performs an all-reduce operation on a tensor across all processes in the TP group.
- `tp_all_gather`: Gathers tensors from all processes in the TP group and returns a tensor with the gathered data.
- `tp_gather`: Gathers tensors from all processes in the TP group and returns a tensor with the gathered data on the specified process.
- `broadcast`: Broadcasts a tensor to all processes in the specified process group.
- `broadcast_obj_list`: Broadcasts a list of objects to all processes in the specified process group.
- `broadcast_tensor_dict`: Broadcasts a dictionary of tensors to all processes in the specified process group.


Be mindful of the distributed environment when using these. For example, the `tp_gather` function returns a tensor only on the specified process (the other processes get `None`), so you should check the return value before using it.