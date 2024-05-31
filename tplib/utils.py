from typing import Sequence
import torch

def ensure_divisibility(numerator, denominator):
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"

def divide(numerator, denominator):
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def split_tensor_along_last_dim(tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool = False) -> Sequence[torch.Tensor]:
    last_dim_size = divide(tensor.size()[-1], num_partitions)
    tensor_list = torch.split(tensor, last_dim_size, dim=-1)
    if contiguous_split_chunks: return tuple(chunk.contiguous() for chunk in tensor_list)
