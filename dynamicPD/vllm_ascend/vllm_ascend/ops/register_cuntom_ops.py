import torch
import torch.nn.functional as F
from vllm.distributed import (get_dp_group, get_ep_group,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_reduce_scatter)
from vllm.forward_context import get_forward_context

from dynamicPD.vllm.vllm.distributed.communication_op import tensor_model_parallel_all_reduce_offload


def _maybe_pad_and_reduce_impl(x: torch.Tensor,
                               is_ep_comm: bool = False) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return tensor_model_parallel_all_reduce(x)

    if not forward_context.sp_enabled:
        if forward_context.use_offload_tp:
            return tensor_model_parallel_all_reduce_offload(x)
        return tensor_model_parallel_all_reduce(x)

    dp_metadata = forward_context.dp_metadata
    if dp_metadata is None or not is_ep_comm:
        pad_size = forward_context.pad_size
        if pad_size > 0:
            x = F.pad(x, (0, 0, 0, pad_size))
        return tensor_model_parallel_reduce_scatter(x, 0)
    else:
        # padding
        dp_size = get_dp_group().world_size
        num_tokens_across_dp_cpu = \
            get_forward_context().dp_metadata.num_tokens_across_dp_cpu
        padded_x = torch.empty(
            (dp_size, forward_context.padded_length, *x.shape[1:]),
            device=x.device,
            dtype=x.dtype)
        offset = 0
        for idx in range(dp_size):
            num_tokens_dp = num_tokens_across_dp_cpu[idx]
            padded_x[idx, :num_tokens_dp] = x[offset:offset + num_tokens_dp]
            offset += num_tokens_dp

        return get_ep_group().reduce_scatter(padded_x.view(-1, *x.shape[1:]),
                                             0)