import torch
from vllm.config import ParallelConfig, get_current_vllm_config
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_world_group,
    init_model_parallel_group,
)

import vllm_ascend.distributed.parallel_state as parallel_state
import vllm.distributed.parallel_state as vllm_parallel_state
from vllm.forward_context import get_forward_context
from dynamicPD.patching import dynamicPDPatch
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import flashcomm2_enable

_orig_init_ascend_model_parallel = parallel_state.init_ascend_model_parallel
_orig_destroy_ascend_model_parallel = parallel_state.destroy_ascend_model_parallel

_ASCEND_GROUP_GETTER_NAMES = (
    "get_mc2_group",
    "get_mlp_tp_group",
    "get_otp_group",
    "get_lmhead_tp_group",
    "get_embed_tp_group",
    "get_flashcomm2_otp_group",
    "get_flashcomm2_odp_group",
    "get_p_tp_group",
    "get_fc3_quant_x_group",
    "get_dynamic_eplb_group",
)
_ORIGINAL_ASCEND_GROUP_GETTERS = {
    name: getattr(parallel_state, name)
    for name in _ASCEND_GROUP_GETTER_NAMES
    if hasattr(parallel_state, name)
}


def _maybe_destroy_group(group: GroupCoordinator | None) -> None:
    if group is not None:
        group.destroy()


def _use_offload_group(offload: bool | None = None) -> bool:
    if offload is not None:
        return offload
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return False
    additional_kwargs = getattr(forward_context, "additional_kwargs", {})
    return bool(
        getattr(forward_context, "use_offload_tp", False)
        or additional_kwargs.get("use_offload_tp", False)
        or additional_kwargs.get("use_offload_collectives", False)
    )


def _select_group(
    normal_group: GroupCoordinator | None,
    offload_group: GroupCoordinator | None,
    normal_message: str,
    offload_message: str,
    offload: bool | None = None,
) -> GroupCoordinator:
    if _use_offload_group(offload):
        assert offload_group is not None, offload_message
        return offload_group
    assert normal_group is not None, normal_message
    return normal_group


class AscendParallelStatePatch(dynamicPDPatch[parallel_state]):
    _OFFLOAD_MC2 = None
    _OFFLOAD_MLP_TP = None
    _OFFLOAD_OTP = None
    _OFFLOAD_LMTP = None
    _OFFLOAD_EMBED_TP = None
    _OFFLOAD_FLASHCOMM2_OTP = None
    _OFFLOAD_FLASHCOMM2_ODP = None
    _OFFLOAD_FC3_QUANT_X = None
    _OFFLOAD_P_TP = None
    _OFFLOAD_DYNAMIC_EPLB = None

    def init_ascend_model_parallel(parallel_config: ParallelConfig) -> None:
        already_initialized = parallel_state.model_parallel_initialized()
        _orig_init_ascend_model_parallel(parallel_config)
        if already_initialized or parallel_state._OFFLOAD_MC2 is not None:
            return

        world_size = torch.distributed.get_world_size()
        backend = torch.distributed.get_backend(get_world_group().device_group)
        global_tp_size = parallel_config.tensor_parallel_size
        global_dp_size = parallel_config.data_parallel_size
        global_pp_size = parallel_config.pipeline_parallel_size
        global_pcp_size = parallel_config.prefill_context_parallel_size

        all_ranks = torch.arange(world_size).reshape(
            -1,
            global_dp_size,
            global_pp_size,
            global_pcp_size,
            global_tp_size,
        )

        pd_tp_ratio = get_ascend_config().pd_tp_ratio
        pd_head_ratio = get_ascend_config().pd_head_ratio
        if pd_head_ratio > 1 and get_current_vllm_config().kv_transfer_config.is_kv_producer:
            num_head_replica = get_ascend_config().num_head_replica
            remote_tp_size = global_tp_size // pd_tp_ratio
            if num_head_replica <= 1:
                group_ranks = all_ranks.view(-1, pd_tp_ratio).unbind(0)
            else:
                group_ranks = all_ranks.clone().view(
                    global_dp_size * global_pp_size * global_pcp_size,
                    -1,
                    num_head_replica,
                )
                group_ranks = group_ranks.permute(0, 2, 1)
                group_ranks = group_ranks.reshape(-1, group_ranks.size(-1))
                alltoall_group_size = group_ranks.size(-1) // remote_tp_size
                group_ranks = group_ranks.unsqueeze(-1).view(
                    global_dp_size * global_pp_size * global_pcp_size,
                    num_head_replica,
                    -1,
                    alltoall_group_size,
                )
                group_ranks = group_ranks.reshape(-1, alltoall_group_size).unbind(0)
            group_ranks = [x.tolist() for x in group_ranks]
            local_rank = get_world_group().local_rank
            num = next((i for i, ranks in enumerate(group_ranks) if local_rank in ranks), None)
            parallel_state._OFFLOAD_P_TP = init_model_parallel_group(
                group_ranks,
                local_rank,
                backend,
                group_name=f"secondary_p_tp_{num}",
            )

        group_ranks = (
            all_ranks.transpose(1, 2)
            .reshape(
                -1,
                global_dp_size * global_pcp_size * global_tp_size,
            )
            .unbind(0)
        )
        group_ranks = [x.tolist() for x in group_ranks]
        parallel_state._OFFLOAD_MC2 = init_model_parallel_group(
            group_ranks, get_world_group().local_rank, backend, group_name="secondary_mc2"
        )

        if get_ascend_config().eplb_config.dynamic_eplb:
            parallel_state._OFFLOAD_DYNAMIC_EPLB = init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                group_name="secondary_dynamic_eplb",
            )

        if get_ascend_config().multistream_overlap_gate:
            parallel_state._OFFLOAD_FC3_QUANT_X = init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                group_name="secondary_fc3_quant_x",
            )

        group_cache = {}

        def create_or_get_group(group_size: int, group_name: str) -> GroupCoordinator | None:
            if group_size is None:
                return None
            if group_size not in group_cache:
                rank_grid = torch.arange(world_size).reshape(global_pp_size, global_dp_size, global_tp_size)
                num_chunks = global_dp_size // group_size
                group_ranks = []
                for pp_idx in range(global_pp_size):
                    stage_ranks = rank_grid[pp_idx]
                    for chunk in range(num_chunks):
                        for tp_idx in range(global_tp_size):
                            group = stage_ranks[chunk * group_size : (chunk + 1) * group_size, tp_idx].tolist()
                            group_ranks.append(group)
                group_cache[group_size] = init_model_parallel_group(
                    group_ranks,
                    get_world_group().local_rank,
                    backend,
                    group_name=group_name,
                )
            return group_cache[group_size]

        finegrained_tp_config = get_ascend_config().finegrained_tp_config
        otp_size = finegrained_tp_config.oproj_tensor_parallel_size
        lmhead_tp_size = finegrained_tp_config.lmhead_tensor_parallel_size
        embedding_tp_size = finegrained_tp_config.embedding_tensor_parallel_size
        mlp_tp_size = finegrained_tp_config.mlp_tensor_parallel_size

        if otp_size > 0:
            parallel_state._OFFLOAD_OTP = create_or_get_group(otp_size, "secondary_otp")
        if lmhead_tp_size > 0:
            parallel_state._OFFLOAD_LMTP = create_or_get_group(lmhead_tp_size, "secondary_lmheadtp")
        if embedding_tp_size > 0:
            parallel_state._OFFLOAD_EMBED_TP = create_or_get_group(embedding_tp_size, "secondary_emtp")
        if mlp_tp_size > 0:
            parallel_state._OFFLOAD_MLP_TP = create_or_get_group(mlp_tp_size, "secondary_mlptp")

        if flashcomm2_enable():
            flashcomm2_otp_size = get_ascend_config().flashcomm2_oproj_tensor_parallel_size
            parallel_state._OFFLOAD_FLASHCOMM2_OTP = None
            parallel_state._OFFLOAD_FLASHCOMM2_ODP = vllm_parallel_state.get_secondary_tp_group()

            if flashcomm2_otp_size > 1:
                num_fc2_oproj_tensor_parallel_groups = global_tp_size // flashcomm2_otp_size
                flashcomm2_otp_group_ranks = []
                odp_group_ranks: list[list[int]] = [
                    [] for _ in range(flashcomm2_otp_size * global_dp_size * global_pp_size)
                ]
                for dp_group_index in range(global_dp_size):
                    for pp_group_index in range(global_pp_size):
                        dp_pp_serial_index = dp_group_index * global_pp_size + pp_group_index
                        tp_base_rank = dp_pp_serial_index * global_tp_size
                        odp_base_index = dp_pp_serial_index * flashcomm2_otp_size

                        for i in range(num_fc2_oproj_tensor_parallel_groups):
                            ranks = []
                            for j in range(flashcomm2_otp_size):
                                tp_local_rank = i + j * num_fc2_oproj_tensor_parallel_groups
                                global_rank = tp_base_rank + tp_local_rank
                                ranks.append(global_rank)
                                odp_group_index = odp_base_index + j
                                odp_group_ranks[odp_group_index].append(global_rank)
                            flashcomm2_otp_group_ranks.append(ranks)

                parallel_state._OFFLOAD_FLASHCOMM2_OTP = init_model_parallel_group(
                    flashcomm2_otp_group_ranks,
                    get_world_group().local_rank,
                    backend,
                    group_name="secondary_flashcomm2_otp",
                )
                parallel_state._OFFLOAD_FLASHCOMM2_ODP = init_model_parallel_group(
                    odp_group_ranks,
                    get_world_group().local_rank,
                    backend,
                    group_name="secondary_flashcomm2_odp",
                )

    def get_mc2_group(offload: bool | None = None) -> GroupCoordinator:
        return _select_group(
            parallel_state._MC2,
            parallel_state._OFFLOAD_MC2,
            "mc2 group is not initialized",
            "secondary mc2 group is not initialized",
            offload,
        )
        
    def get_mlp_tp_group(offload: bool | None = None) -> GroupCoordinator:
        return _select_group(
            parallel_state._MLP_TP,
            parallel_state._OFFLOAD_MLP_TP,
            "mlp group is not initialized",
            "secondary mlp group is not initialized",
            offload,
        )
    
    def get_otp_group(offload: bool | None = None) -> GroupCoordinator:
        return _select_group(
            parallel_state._OTP,
            parallel_state._OFFLOAD_OTP,
            "otp group is not initialized",
            "secondary otp group is not initialized",
            offload,
        )
    
    def get_lmhead_tp_group(offload: bool | None = None) -> GroupCoordinator:
        return _select_group(
            parallel_state._LMTP,
            parallel_state._OFFLOAD_LMTP,
            "lm head tensor parallel group is not initialized",
            "secondary lm head tensor parallel group is not initialized",
            offload,
        )
        
    def get_embed_tp_group(offload: bool | None = None) -> GroupCoordinator:
        return _select_group(
            parallel_state._EMBED_TP,
            parallel_state._OFFLOAD_EMBED_TP,
            "embedding tensor parallel group is not initialized",
            "secondary embedding tensor parallel group is not initialized",
            offload,
        )
        
    def get_flashcomm2_otp_group(
        offload: bool | None = None,
    ) -> GroupCoordinator | None:
        if _use_offload_group(offload):
            return parallel_state._OFFLOAD_FLASHCOMM2_OTP
        return parallel_state._FLASHCOMM2_OTP
        
    def get_flashcomm2_odp_group(offload: bool | None = None) -> GroupCoordinator:
        return _select_group(
            parallel_state._FLASHCOMM2_ODP,
            parallel_state._OFFLOAD_FLASHCOMM2_ODP,
            "flashcomm2 odp group is not initialized",
            "secondary flashcomm2 odp group is not initialized",
            offload,
        )
        
    def get_p_tp_group(offload: bool | None = None) -> GroupCoordinator:
        return _select_group(
            parallel_state._P_TP,
            parallel_state._OFFLOAD_P_TP,
            "distributed prefill tensor parallel group is not initialized",
            "secondary distributed prefill tensor parallel group is not initialized",
            offload,
        )

    def get_fc3_quant_x_group(offload: bool | None = None) -> GroupCoordinator:
        return _select_group(
            parallel_state._FC3_QUANT_X,
            parallel_state._OFFLOAD_FC3_QUANT_X,
            "fc3 quant x group is not initialized",
            "secondary fc3 quant x group is not initialized",
            offload,
        )
        
    def get_dynamic_eplb_group(offload: bool | None = None) -> GroupCoordinator:
        return _select_group(
            parallel_state._DYNAMIC_EPLB,
            parallel_state._OFFLOAD_DYNAMIC_EPLB,
            "dynamic eplb group is not initialized",
            "secondary dynamic eplb group is not initialized",
            offload,
        )

    def destroy_ascend_model_parallel():
        _maybe_destroy_group(parallel_state._OFFLOAD_MC2)
        parallel_state._OFFLOAD_MC2 = None

        _maybe_destroy_group(parallel_state._OFFLOAD_MLP_TP)
        parallel_state._OFFLOAD_MLP_TP = None

        _maybe_destroy_group(parallel_state._OFFLOAD_LMTP)
        parallel_state._OFFLOAD_LMTP = None

        _maybe_destroy_group(parallel_state._OFFLOAD_EMBED_TP)
        parallel_state._OFFLOAD_EMBED_TP = None

        _maybe_destroy_group(parallel_state._OFFLOAD_OTP)
        parallel_state._OFFLOAD_OTP = None

        _maybe_destroy_group(parallel_state._OFFLOAD_P_TP)
        parallel_state._OFFLOAD_P_TP = None

        if (
            parallel_state._OFFLOAD_FLASHCOMM2_OTP
            and get_ascend_config().flashcomm2_oproj_tensor_parallel_size != 1
        ):
            parallel_state._OFFLOAD_FLASHCOMM2_OTP.destroy()
        parallel_state._OFFLOAD_FLASHCOMM2_OTP = None

        if (
            parallel_state._OFFLOAD_FLASHCOMM2_ODP
            and get_ascend_config().flashcomm2_oproj_tensor_parallel_size != 1
        ):
            parallel_state._OFFLOAD_FLASHCOMM2_ODP.destroy()
        parallel_state._OFFLOAD_FLASHCOMM2_ODP = None

        _maybe_destroy_group(parallel_state._OFFLOAD_FC3_QUANT_X)
        parallel_state._OFFLOAD_FC3_QUANT_X = None

        _maybe_destroy_group(parallel_state._OFFLOAD_DYNAMIC_EPLB)
        parallel_state._OFFLOAD_DYNAMIC_EPLB = None

        _orig_destroy_ascend_model_parallel()


_ASCEND_GROUP_GETTER_PATCHES = {
    "get_mc2_group": AscendParallelStatePatch.get_mc2_group,
    "get_mlp_tp_group": AscendParallelStatePatch.get_mlp_tp_group,
    "get_otp_group": AscendParallelStatePatch.get_otp_group,
    "get_lmhead_tp_group": AscendParallelStatePatch.get_lmhead_tp_group,
    "get_embed_tp_group": AscendParallelStatePatch.get_embed_tp_group,
    "get_flashcomm2_otp_group": AscendParallelStatePatch.get_flashcomm2_otp_group,
    "get_flashcomm2_odp_group": AscendParallelStatePatch.get_flashcomm2_odp_group,
    "get_p_tp_group": AscendParallelStatePatch.get_p_tp_group,
    "get_fc3_quant_x_group": AscendParallelStatePatch.get_fc3_quant_x_group,
    "get_dynamic_eplb_group": AscendParallelStatePatch.get_dynamic_eplb_group,
}


def _replace_function_in_place(target, source):
    target.__globals__.update(
        {
            "_select_group": _select_group,
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


def install_ascend_parallel_state_patch() -> None:
    """Patch early-bound Ascend communication-group getters in place."""

    for name, source in _ASCEND_GROUP_GETTER_PATCHES.items():
        original = _ORIGINAL_ASCEND_GROUP_GETTERS.get(name)
        patched = (
            _replace_function_in_place(original, source)
            if original is not None
            else source
        )
        setattr(parallel_state, name, patched)
