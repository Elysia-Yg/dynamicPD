from typing import Any

import torch

import vllm.distributed.communication_op as communication_op
import vllm.distributed.parallel_state as parallel_state

from dynamicPD.patching import dynamicPDPatch

_COMM_OP_NAMES = (
    "tensor_model_parallel_all_reduce",
    "tensor_model_parallel_all_gather",
    "tensor_model_parallel_reduce_scatter",
    "tensor_model_parallel_gather",
    "broadcast_tensor_dict",
)
_ORIGINAL_COMM_FUNCS = {
    name: getattr(communication_op, name)
    for name in _COMM_OP_NAMES
    if hasattr(communication_op, name)
}


def tensor_model_parallel_all_reduce(
    input_: torch.Tensor,
    offload: bool | None = None,
) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return parallel_state.get_tp_group(offload).all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor,
    dim: int = -1,
    offload: bool | None = None,
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return parallel_state.get_tp_group(offload).all_gather(input_, dim)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor,
    dim: int = -1,
    offload: bool | None = None,
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return parallel_state.get_tp_group(offload).reduce_scatter(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor,
    dst: int = 0,
    dim: int = -1,
    offload: bool | None = None,
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group."""
    return parallel_state.get_tp_group(offload).gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None,
    src: int = 0,
    offload: bool | None = None,
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return parallel_state.get_tp_group(offload).broadcast_tensor_dict(
        tensor_dict, src
    )


class CommunicationOpPatch(dynamicPDPatch[communication_op]):
    tensor_model_parallel_all_reduce = tensor_model_parallel_all_reduce
    tensor_model_parallel_all_gather = tensor_model_parallel_all_gather
    tensor_model_parallel_reduce_scatter = tensor_model_parallel_reduce_scatter
    tensor_model_parallel_gather = tensor_model_parallel_gather
    broadcast_tensor_dict = broadcast_tensor_dict


_COMM_OP_PATCHES = {
    "tensor_model_parallel_all_reduce": tensor_model_parallel_all_reduce,
    "tensor_model_parallel_all_gather": tensor_model_parallel_all_gather,
    "tensor_model_parallel_reduce_scatter": tensor_model_parallel_reduce_scatter,
    "tensor_model_parallel_gather": tensor_model_parallel_gather,
    "broadcast_tensor_dict": broadcast_tensor_dict,
}


def _replace_function_in_place(target: Any, source: Any) -> Any:
    target.__globals__.update(
        {
            "parallel_state": parallel_state,
            "torch": torch,
        }
    )
    target.__code__ = source.__code__
    target.__defaults__ = source.__defaults__
    target.__kwdefaults__ = source.__kwdefaults__
    target.__annotations__ = dict(getattr(source, "__annotations__", {}))
    target.__doc__ = source.__doc__
    return target


def install_communication_op_patch() -> None:
    """Patch old communication function objects imported before dynamicPD.

    vLLM model layers and vLLM-Ascend custom ops commonly bind these helpers at
    module import time. Mutating the original function objects keeps those early
    bindings aligned with the offload-aware TP group selection.
    """

    for name, source in _COMM_OP_PATCHES.items():
        original = _ORIGINAL_COMM_FUNCS.get(name)
        patched = (
            _replace_function_in_place(original, source)
            if original is not None
            else source
        )
        setattr(communication_op, name, patched)
