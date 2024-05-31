import torch.multiprocessing as mp
from tplib import init_dist_env

class Worker:
    def __init__(self, world_size, init_method="tcp://localhost:23456", backend="nccl"):
        self.world_size = world_size
        self.init_method = init_method
        self.backend = backend

    def run(self, func, *args, **kwargs):
        processes = []
        for rank in range(self.world_size):
            p = mp.Process(target=self._target, args=(func, rank, args, kwargs))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    def _target(self, func, rank, args, kwargs):
        init_dist_env(world_size=self.world_size, rank=rank, distributed_init_method=self.init_method, local_rank=rank, backend=self.backend)
        func(rank, *args, **kwargs)