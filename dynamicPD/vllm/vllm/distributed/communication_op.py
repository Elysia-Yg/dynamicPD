from typing import Optional

import torch

import vllm.distributed.parallel_state as parallel_state

def tensor_model_parallel_all_reduce_offload(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across offload model parallel group."""
    # print("current stream in all reduce offload tp:", torch.npu.current_stream().stream_id)
    return parallel_state.get_offload_tp_group().all_reduce(input_)

def tensor_model_parallel_all_gather_offload(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return parallel_state.get_offload_tp_group().all_gather(input_, dim)

def tensor_model_parallel_gather_offload(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> Optional[torch.Tensor]:
    """Gather the input tensor across offload model parallel group."""
    return parallel_state.get_offload_tp_group().gather(input_, dst, dim)
