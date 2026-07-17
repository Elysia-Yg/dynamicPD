from typing import Any

import torch

import vllm.distributed.parallel_state as parallel_state
from dynamicPD.patching import dynamicPDPatch
from vllm.distributed.parallel_state import GroupCoordinator, logger

_orig_initialize_model_parallel = parallel_state.initialize_model_parallel
_orig_destroy_model_parallel = parallel_state.destroy_model_parallel
_orig_prepare_communication_buffer_for_model = (
    parallel_state.prepare_communication_buffer_for_model
)

_OFFLOAD_GROUPS = ("TP", "DCP", "PCP", "PP", "DP", "EP", "EPLB")
_GROUP_GETTER_NAMES = (
    "get_tp_group",
    "get_dcp_group",
    "get_context_model_parallel_group",
    "get_pcp_group",
    "get_pp_group",
    "get_dp_group",
    "get_ep_group",
    "get_eplb_group",
)
_ORIGINAL_GROUP_GETTERS = {
    name: getattr(parallel_state, name)
    for name in _GROUP_GETTER_NAMES
    if hasattr(parallel_state, name)
}


def _offload_attr(group_name: str) -> str:
    return f"_OFFLOAD_{group_name}"


def _secondary_attr(group_name: str) -> str:
    return f"_SECONDARY_{group_name}"


def _set_offload_group(group_name: str, group: Any | None) -> None:
    setattr(parallel_state, _offload_attr(group_name), group)
    setattr(parallel_state, _secondary_attr(group_name), group)


def _get_offload_group(group_name: str, message: str) -> Any:
    group = getattr(parallel_state, _offload_attr(group_name), None)
    assert group is not None, message
    return group


def _destroy_group(group: Any | None) -> None:
    if group is not None:
        group.destroy()


def _destroy_offload_model_parallel_groups() -> None:
    for group_name in _OFFLOAD_GROUPS:
        _destroy_group(getattr(parallel_state, _offload_attr(group_name), None))
        _set_offload_group(group_name, None)


def _to_group_ranks(ranks: torch.Tensor) -> list[list[int]]:
    return [x.tolist() for x in ranks.unbind(0)]


def _get_default_backend(backend: str | None, enable_elastic_ep: bool) -> str:
    if backend is not None:
        return backend
    if enable_elastic_ep:
        return "nccl"
    world = parallel_state.get_world_group()
    return torch.distributed.get_backend(world.device_group)


def _current_forward_context_uses_offload() -> bool:
    try:
        from vllm.forward_context import get_forward_context

        forward_context = get_forward_context()
    except AssertionError:
        return False

    additional_kwargs = getattr(forward_context, "additional_kwargs", {})
    return bool(
        getattr(forward_context, "use_offload_tp", False)
        or additional_kwargs.get("use_offload_tp", False)
        or additional_kwargs.get("use_offload_collectives", False)
    )


def _use_offload_group(offload: bool | None = None) -> bool:
    if offload is not None:
        return offload
    return _current_forward_context_uses_offload()


def _init_group(
    group_ranks: list[list[int]],
    backend: str,
    group_name: str,
    *,
    use_message_queue_broadcaster: bool = False,
    use_device_communicator: bool = True,
) -> GroupCoordinator:
    return parallel_state.init_model_parallel_group(
        group_ranks,
        parallel_state.get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=use_message_queue_broadcaster,
        group_name=group_name,
        use_device_communicator=use_device_communicator,
    )


def _init_stateless_group(
    group_ranks: list[list[int]],
    backend: str,
    group_name: str,
    port_getter_name: str,
) -> Any:
    from vllm.config import get_current_vllm_config

    parallel_config = get_current_vllm_config().parallel_config
    get_next_port = getattr(parallel_config, port_getter_name)
    group_ports = [get_next_port() for _ in group_ranks]
    return parallel_state._init_stateless_group(
        group_ranks,
        group_name,
        group_ports,
        parallel_config.data_parallel_master_ip,
        backend,
    )


def _initialize_offload_model_parallel_groups(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    prefill_context_model_parallel_size: int,
    decode_context_model_parallel_size: int | None,
    backend: str | None,
) -> None:
    if getattr(parallel_state, "_OFFLOAD_TP", None) is not None:
        return

    from vllm.config import get_current_vllm_config

    config = get_current_vllm_config()
    parallel_config = config.parallel_config
    data_parallel_size = parallel_config.data_parallel_size
    enable_elastic_ep = parallel_config.enable_elastic_ep
    backend = _get_default_backend(backend, enable_elastic_ep)

    if enable_elastic_ep:
        world_size = parallel_state.get_world_group().world_size
        tp_pp_pcp_size = (
            tensor_model_parallel_size
            * pipeline_model_parallel_size
            * prefill_context_model_parallel_size
        )
        local_all_ranks = torch.arange(tp_pp_pcp_size).reshape(
            pipeline_model_parallel_size,
            prefill_context_model_parallel_size,
            tensor_model_parallel_size,
        )
    else:
        world_size = torch.distributed.get_world_size()
        local_all_ranks = None

    all_ranks = torch.arange(world_size).reshape(
        -1,
        data_parallel_size,
        pipeline_model_parallel_size,
        prefill_context_model_parallel_size,
        tensor_model_parallel_size,
    )

    tp_group_ranks = _to_group_ranks(
        all_ranks.view(-1, tensor_model_parallel_size)
    )
    if enable_elastic_ep:
        assert local_all_ranks is not None
        tp_group_ranks = _to_group_ranks(
            local_all_ranks.view(-1, tensor_model_parallel_size)
        )
    _set_offload_group(
        "TP",
        _init_group(
            tp_group_ranks,
            backend,
            "offload_tp",
            use_message_queue_broadcaster=True,
        ),
    )

    dcp_group_ranks = _to_group_ranks(
        all_ranks.reshape(-1, decode_context_model_parallel_size)
    )
    if enable_elastic_ep:
        assert local_all_ranks is not None
        dcp_group_ranks = _to_group_ranks(
            local_all_ranks.reshape(-1, decode_context_model_parallel_size)
        )
    _set_offload_group(
        "DCP",
        _init_group(
            dcp_group_ranks,
            backend,
            "offload_dcp",
            use_message_queue_broadcaster=True,
        ),
    )

    pcp_group_ranks = _to_group_ranks(
        all_ranks.transpose(3, 4)
        .reshape(-1, prefill_context_model_parallel_size)
    )
    if enable_elastic_ep:
        assert local_all_ranks is not None
        pcp_group_ranks = _to_group_ranks(
            local_all_ranks.transpose(1, 2)
            .reshape(-1, prefill_context_model_parallel_size)
        )
    _set_offload_group(
        "PCP",
        _init_group(pcp_group_ranks, backend, "offload_pcp"),
    )

    pp_group_ranks = _to_group_ranks(
        all_ranks.transpose(2, 4).reshape(-1, pipeline_model_parallel_size)
    )
    if enable_elastic_ep:
        assert local_all_ranks is not None
        pp_group_ranks = _to_group_ranks(
            local_all_ranks.transpose(0, 2)
            .reshape(-1, pipeline_model_parallel_size)
        )
    _set_offload_group(
        "PP",
        _init_group(pp_group_ranks, backend, "offload_pp"),
    )

    dp_group_ranks = _to_group_ranks(
        all_ranks.transpose(1, 4).reshape(-1, data_parallel_size)
    )
    if enable_elastic_ep:
        offload_dp = _init_stateless_group(
            dp_group_ranks,
            backend,
            "offload_dp",
            "get_next_stateless_dp_group_port",
        )
    else:
        offload_dp = _init_group(dp_group_ranks, backend, "offload_dp")
    _set_offload_group("DP", offload_dp)

    should_create_ep = config.model_config is None or config.model_config.is_moe
    if not should_create_ep:
        return

    ep_group_ranks = _to_group_ranks(
        all_ranks.transpose(1, 2)
        .reshape(
            -1,
            data_parallel_size
            * prefill_context_model_parallel_size
            * tensor_model_parallel_size,
        )
    )
    if enable_elastic_ep:
        offload_ep = _init_stateless_group(
            ep_group_ranks,
            backend,
            "offload_ep",
            "get_next_stateless_ep_group_port",
        )
    else:
        offload_ep = _init_group(ep_group_ranks, backend, "offload_ep")
    _set_offload_group("EP", offload_ep)

    if not parallel_config.enable_eplb:
        return
    if enable_elastic_ep:
        offload_eplb = _init_stateless_group(
            ep_group_ranks,
            backend,
            "offload_eplb",
            "get_next_stateless_eplb_group_port",
        )
    else:
        offload_eplb = _init_group(ep_group_ranks, backend, "offload_eplb")
    _set_offload_group("EPLB", offload_eplb)


def _prepare_offload_communication_buffers(model: torch.nn.Module) -> None:
    for group_name in _OFFLOAD_GROUPS:
        group = getattr(parallel_state, _offload_attr(group_name), None)
        if group is not None:
            group.prepare_communication_buffer_for_model(model)


class ParallelStatePatch(dynamicPDPatch[parallel_state]):
    _OFFLOAD_TP = None
    _OFFLOAD_DCP = None
    _OFFLOAD_PCP = None
    _OFFLOAD_PP = None
    _OFFLOAD_DP = None
    _OFFLOAD_EP = None
    _OFFLOAD_EPLB = None

    _SECONDARY_TP = None
    _SECONDARY_DCP = None
    _SECONDARY_PCP = None
    _SECONDARY_PP = None
    _SECONDARY_DP = None
    _SECONDARY_EP = None
    _SECONDARY_EPLB = None

    def get_tp_group(offload: bool | None = None) -> GroupCoordinator:
        if _use_offload_group(offload):
            logger.debug("get offload tp group, offload=%s", offload)
            return _get_offload_group(
                "TP", "offload tensor model parallel group is not initialized"
            )
        assert parallel_state._TP is not None, (
            "tensor model parallel group is not initialized"
        )
        return parallel_state._TP

    def get_dcp_group(offload: bool | None = None) -> GroupCoordinator:
        if _use_offload_group(offload):
            logger.debug("get offload dcp group, offload=%s", offload)
            return _get_offload_group(
                "DCP",
                "offload decode context model parallel group is not initialized",
            )
        assert parallel_state._DCP is not None, (
            "decode context model parallel group is not initialized"
        )
        return parallel_state._DCP

    get_context_model_parallel_group = get_dcp_group
    
    def get_pcp_group(offload: bool | None = None) -> GroupCoordinator:
        if _use_offload_group(offload):
            logger.debug("get offload pcp group, offload=%s", offload)
            return _get_offload_group(
                "PCP",
                "offload prefill context model parallel group is not initialized",
            )
        assert parallel_state._PCP is not None, (
            "prefill context model parallel group is not initialized"
        )
        return parallel_state._PCP

    def get_pp_group(offload: bool | None = None) -> GroupCoordinator:
        if _use_offload_group(offload):
            logger.debug("get offload pp group, offload=%s", offload)
            return _get_offload_group(
                "PP", "offload pipeline model parallel group is not initialized"
            )
        assert parallel_state._PP is not None, (
            "pipeline model parallel group is not initialized"
        )
        return parallel_state._PP

    def get_dp_group(offload: bool | None = None) -> GroupCoordinator:
        if _use_offload_group(offload):
            logger.debug("get offload dp group, offload=%s", offload)
            return _get_offload_group(
                "DP", "offload data parallel group is not initialized"
            )
        assert parallel_state._DP is not None, "data parallel group is not initialized"
        return parallel_state._DP

    def get_ep_group(offload: bool | None = None) -> GroupCoordinator:
        if _use_offload_group(offload):
            logger.debug("get offload ep group, offload=%s", offload)
            return _get_offload_group(
                "EP",
                "offload expert parallel group is not initialized. "
                "EP group is only created for MoE models with num_experts > 0.",
            )
        assert parallel_state._EP is not None, (
            "expert parallel group is not initialized. "
            "EP group is only created for MoE models with num_experts > 0. "
            "This function should only be called for MoE models."
        )
        return parallel_state._EP

    def get_eplb_group(offload: bool | None = None) -> GroupCoordinator:
        if _use_offload_group(offload):
            return _get_offload_group(
                "EPLB",
                "offload EPLB group is not initialized. "
                "EPLB group is only created for MoE models when EPLB is enabled.",
            )
        assert parallel_state._EPLB is not None, (
            "EPLB group is not initialized. "
            "EPLB group is only created for MoE models when EPLB is enabled. "
            "Ensure parallel_config.enable_eplb is True."
        )
        return parallel_state._EPLB

    def get_offload_tp_group() -> GroupCoordinator:
        return _get_offload_group(
            "TP", "offload tensor model parallel group is not initialized"
        )

    get_secondary_tp_group = get_offload_tp_group

    def get_offload_dcp_group() -> GroupCoordinator:
        return _get_offload_group(
            "DCP",
            "offload decode context model parallel group is not initialized",
        )

    get_secondary_dcp_group = get_offload_dcp_group

    def get_offload_pcp_group() -> GroupCoordinator:
        return _get_offload_group(
            "PCP",
            "offload prefill context model parallel group is not initialized",
        )

    get_secondary_pcp_group = get_offload_pcp_group

    def get_offload_pp_group() -> GroupCoordinator:
        return _get_offload_group(
            "PP", "offload pipeline model parallel group is not initialized"
        )

    get_secondary_pp_group = get_offload_pp_group

    def get_offload_dp_group() -> GroupCoordinator:
        return _get_offload_group(
            "DP", "offload data parallel group is not initialized"
        )

    get_secondary_dp_group = get_offload_dp_group

    def get_offload_ep_group() -> GroupCoordinator:
        return _get_offload_group(
            "EP",
            "offload expert parallel group is not initialized. "
            "EP group is only created for MoE models with num_experts > 0.",
        )

    get_secondary_ep_group = get_offload_ep_group

    def get_offload_eplb_group() -> GroupCoordinator:
        return _get_offload_group(
            "EPLB",
            "offload EPLB group is not initialized. "
            "EPLB group is only created for MoE models when EPLB is enabled.",
        )

    get_secondary_eplb_group = get_offload_eplb_group

    def initialize_model_parallel(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        prefill_context_model_parallel_size: int = 1,
        decode_context_model_parallel_size: int | None = 1,
        backend: str | None = None,
    ) -> None:
        _orig_initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            prefill_context_model_parallel_size=prefill_context_model_parallel_size,
            decode_context_model_parallel_size=decode_context_model_parallel_size,
            backend=backend,
        )
        _initialize_offload_model_parallel_groups(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            prefill_context_model_parallel_size,
            decode_context_model_parallel_size,
            backend,
        )

    def prepare_communication_buffer_for_model(model: torch.nn.Module) -> None:
        _orig_prepare_communication_buffer_for_model(model)
        _prepare_offload_communication_buffers(model)

    def destroy_model_parallel() -> None:
        _destroy_offload_model_parallel_groups()
        _orig_destroy_model_parallel()


_GROUP_GETTER_PATCHES = {
    "get_tp_group": ParallelStatePatch.get_tp_group,
    "get_dcp_group": ParallelStatePatch.get_dcp_group,
    "get_context_model_parallel_group": (
        ParallelStatePatch.get_context_model_parallel_group
    ),
    "get_pcp_group": ParallelStatePatch.get_pcp_group,
    "get_pp_group": ParallelStatePatch.get_pp_group,
    "get_dp_group": ParallelStatePatch.get_dp_group,
    "get_ep_group": ParallelStatePatch.get_ep_group,
    "get_eplb_group": ParallelStatePatch.get_eplb_group,
}

_OFFLOAD_GETTER_NAMES = (
    "get_offload_tp_group",
    "get_secondary_tp_group",
    "get_offload_dcp_group",
    "get_secondary_dcp_group",
    "get_offload_pcp_group",
    "get_secondary_pcp_group",
    "get_offload_pp_group",
    "get_secondary_pp_group",
    "get_offload_dp_group",
    "get_secondary_dp_group",
    "get_offload_ep_group",
    "get_secondary_ep_group",
    "get_offload_eplb_group",
    "get_secondary_eplb_group",
)


def _replace_function_in_place(target: Any, source: Any) -> Any:
    target.__globals__.update(
        {
            "_get_offload_group": _get_offload_group,
            "_use_offload_group": _use_offload_group,
            "parallel_state": parallel_state,
        }
    )
    target.__code__ = source.__code__
    target.__defaults__ = source.__defaults__
    target.__kwdefaults__ = source.__kwdefaults__
    target.__annotations__ = dict(getattr(source, "__annotations__", {}))
    target.__doc__ = source.__doc__
    return target


def install_parallel_state_patch() -> None:
    """Patch old getter objects so early `from ... import get_*` bindings work.

    Several vLLM/vLLM-Ascend modules import communication-group getters before
    dynamicPD patches are installed. Replacing only module attributes leaves
    those old references pointing at the original groups, which can deadlock
    when decode and async prefill communicate concurrently.
    """

    for name, source in _GROUP_GETTER_PATCHES.items():
        original = _ORIGINAL_GROUP_GETTERS.get(name)
        patched = (
            _replace_function_in_place(original, source)
            if original is not None
            else source
        )
        setattr(parallel_state, name, patched)

    for name in _OFFLOAD_GETTER_NAMES:
        setattr(parallel_state, name, getattr(ParallelStatePatch, name))
