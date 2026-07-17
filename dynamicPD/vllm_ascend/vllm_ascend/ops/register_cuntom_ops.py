from typing import Any

import torch
import torch.nn.functional as F
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.forward_context import get_forward_context

import vllm_ascend.ops.register_custom_ops as custom_ops
from vllm_ascend.ascend_forward_context import MoECommType, _EXTRA_CTX
from vllm_ascend.utils import enable_sp_by_pass, is_vl_model


def _maybe_chunk_residual_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    try:
        get_forward_context()
    except AssertionError:
        return residual

    if x.size(0) != residual.size(0):
        pad_size = _EXTRA_CTX.pad_size
        if pad_size > 0:
            residual = F.pad(residual, (0, 0, 0, pad_size))
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        residual = torch.chunk(residual, tp_size, dim=0)[tp_rank]

    return residual


def _maybe_all_gather_and_maybe_unpad_impl(
    x: torch.Tensor,
    label: bool,
    is_ep_comm: bool = False,
) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return x

    flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled or (
        enable_sp_by_pass() and is_ep_comm)
    if flash_comm_v1_enabled and label:
        dp_metadata = forward_context.dp_metadata
        if dp_metadata is None or not is_ep_comm:
            x = tensor_model_parallel_all_gather(x, 0)
            pad_size = _EXTRA_CTX.pad_size
            if pad_size > 0:
                x = x[:-pad_size]
        else:
            x = get_ep_group().all_gather(x, 0)
            if enable_sp_by_pass():
                return x
            num_tokens_across_dp_cpu = dp_metadata.num_tokens_across_dp_cpu
            result = torch.empty(
                (num_tokens_across_dp_cpu.sum(), *x.shape[1:]),
                device=x.device,
                dtype=x.dtype,
            )
            dp_size = get_dp_group().world_size
            x = x.view(dp_size, _EXTRA_CTX.padded_length, *x.shape[1:])
            offset = 0
            for idx in range(dp_size):
                num_tokens_dp = num_tokens_across_dp_cpu[idx]
                result[offset:offset + num_tokens_dp] = x[idx, :num_tokens_dp]
                offset += num_tokens_dp
            x = result

    return x


def _maybe_pad_and_reduce_impl(
    x: torch.Tensor,
    is_ep_comm: bool = False,
) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return tensor_model_parallel_all_reduce(x)

    flash_comm_v1_enabled = getattr(
        forward_context, "flash_comm_v1_enabled", False) or (
            enable_sp_by_pass() and is_ep_comm)

    if (not flash_comm_v1_enabled or
            (forward_context.is_draft_model and is_vl_model()
             and not is_ep_comm)):
        return tensor_model_parallel_all_reduce(x)

    dp_metadata = forward_context.dp_metadata
    if dp_metadata is None or not is_ep_comm:
        pad_size = _EXTRA_CTX.pad_size
        if pad_size > 0:
            x = F.pad(x, (0, 0, 0, pad_size))
        return tensor_model_parallel_reduce_scatter(x, 0)

    if enable_sp_by_pass():
        return get_ep_group().reduce_scatter(x.view(-1, *x.shape[1:]), 0)

    dp_size = get_dp_group().world_size
    num_tokens_across_dp_cpu = dp_metadata.num_tokens_across_dp_cpu
    padded_x = torch.empty(
        (dp_size, _EXTRA_CTX.padded_length, *x.shape[1:]),
        device=x.device,
        dtype=x.dtype,
    )
    offset = 0
    for idx in range(dp_size):
        num_tokens_dp = num_tokens_across_dp_cpu[idx]
        padded_x[idx, :num_tokens_dp] = x[offset:offset + num_tokens_dp]
        offset += num_tokens_dp

    return get_ep_group().reduce_scatter(
        padded_x.view(-1, *x.shape[1:]), 0)


def _maybe_all_reduce_tensor_model_parallel_impl(
    final_hidden_states: torch.Tensor,
) -> torch.Tensor:
    moe_comm_type = _EXTRA_CTX.moe_comm_type
    if (moe_comm_type in {
            MoECommType.ALLTOALL,
            MoECommType.MC2,
            MoECommType.FUSED_MC2,
    } or _EXTRA_CTX.flash_comm_v1_enabled):
        return final_hidden_states
    return tensor_model_parallel_all_reduce(final_hidden_states)


_CUSTOM_OP_PATCHES = {
    "_maybe_chunk_residual_impl": _maybe_chunk_residual_impl,
    "_maybe_all_gather_and_maybe_unpad_impl":
    _maybe_all_gather_and_maybe_unpad_impl,
    "_maybe_pad_and_reduce_impl": _maybe_pad_and_reduce_impl,
    "_maybe_all_reduce_tensor_model_parallel_impl":
    _maybe_all_reduce_tensor_model_parallel_impl,
}


def _replace_function_in_place(target: Any, source: Any) -> Any:
    target.__globals__.update({
        "F": F,
        "MoECommType": MoECommType,
        "_EXTRA_CTX": _EXTRA_CTX,
        "enable_sp_by_pass": enable_sp_by_pass,
        "get_dp_group": get_dp_group,
        "get_ep_group": get_ep_group,
        "get_forward_context": get_forward_context,
        "get_tensor_model_parallel_rank": get_tensor_model_parallel_rank,
        "get_tensor_model_parallel_world_size":
        get_tensor_model_parallel_world_size,
        "is_vl_model": is_vl_model,
        "tensor_model_parallel_all_gather": tensor_model_parallel_all_gather,
        "tensor_model_parallel_all_reduce": tensor_model_parallel_all_reduce,
        "tensor_model_parallel_reduce_scatter":
        tensor_model_parallel_reduce_scatter,
        "torch": torch,
    })
    target.__code__ = source.__code__
    target.__defaults__ = source.__defaults__
    target.__kwdefaults__ = source.__kwdefaults__
    target.__annotations__ = dict(getattr(source, "__annotations__", {}))
    target.__doc__ = source.__doc__
    return target


def install_custom_ops_patch() -> None:
    for name, source in _CUSTOM_OP_PATCHES.items():
        original = getattr(custom_ops, name, None)
        patched = (_replace_function_in_place(original, source)
                   if original is not None else source)
        setattr(custom_ops, name, patched)
