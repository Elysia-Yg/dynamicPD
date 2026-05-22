from typing import Optional

import torch
import torch.distributed
from typing_extensions import deprecated
from vllm.logger import logger
from vllm.distributed.parallel_state import get_world_group, GroupCoordinator, init_model_parallel_group

import vllm.distributed.parallel_state as parallel_state

from dynamicPD.patching import dynamicPDPatch

class ParallelStatePatch(dynamicPDPatch[parallel_state]):
    _offload_TP = None

    def get_offload_tp_group() -> GroupCoordinator:
        assert parallel_state._offload_TP is not None, ("offload tensor model parallel group is not initialized")
        return parallel_state._offload_TP

    @deprecated("`get_offload_tensor_model_parallel_group` has been replaced with "
                "`get_offload_tp_group` and may be removed after v0.12. Please use "
                "`get_offload_tp_group` instead.")
    def get_offload_tensor_model_parallel_group():
        return parallel_state.get_offload_tp_group()

    def initialize_model_parallel(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        decode_context_model_parallel_size: Optional[int] = 1,
        backend: Optional[str] = None,
    ) -> None:
        from vllm.distributed.parallel_state import _DP, _EP, _PP, _TP, _DCP
        # Get world size and rank. Ensure some consistencies.
        assert torch.distributed.is_initialized()
        world_size: int = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        backend = backend or torch.distributed.get_backend(
            get_world_group().device_group)

        data_parallel_size = 1
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config()
        if config is not None:
            data_parallel_size = config.parallel_config.data_parallel_size

        # the layout order is: ExternalDP x DP x PP x TP
        # ExternalDP is the data parallel group that is not part of the model,
        # every dp rank can generate independently (in verl integration).
        # DP is the data parallel group that is part of the model,
        # all the ranks in the same DP group should generate simultaneously,
        # i.e. the `generate` call in the same DP group should be called together,
        # otherwise it will cause deadlock.
        # to get group_ranks for each dimension, transpose that dimension to the
        # last dimension, then reshape to 2D, then unbind the last dimension
        all_ranks = torch.arange(world_size).reshape(
            -1, data_parallel_size, pipeline_model_parallel_size,
            tensor_model_parallel_size)  # noqa
        logger.info("all_ranks shape: %s", all_ranks.shape)
        # Build the tensor model-parallel groups.
        assert _TP is None, ("tensor model parallel group is already initialized")
        group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]

        # message queue broadcaster is only used in tensor model parallel group
        _TP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        use_message_queue_broadcaster=True,
                                        group_name="tp")
        _offload_TP = init_model_parallel_group(
                                        group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        use_message_queue_broadcaster=False,
                                        group_name="offload_tp")

        # Build the DCP model-parallel groups.
        assert _DCP is None, (
            "decode context model parallel group is already initialized")
        # Note(hc): In the current implementation of decode context parallel,
        # dcp_size must not exceed tp_size, because the world size does not
        # change by DCP, it simply reuses the GPUs of TP group, and split one
        # TP group into tp_size//dcp_size DCP groups.
        group_ranks = all_ranks.reshape(
            -1, decode_context_model_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        _DCP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        use_message_queue_broadcaster=True,
                                        group_name="dcp")

        # Build the pipeline model-parallel groups.
        assert _PP is None, (
            "pipeline model parallel group is already initialized")
        group_ranks = all_ranks.transpose(2, 3).reshape(
            -1, pipeline_model_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        _PP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="pp")

        assert _DP is None, ("data parallel group is already initialized")
        group_ranks = all_ranks.transpose(1,
                                        3).reshape(-1,
                                                    data_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        _DP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="dp")

        assert _EP is None, ("expert parallel group is already initialized")
        group_ranks = all_ranks.transpose(1, 2).reshape(
            -1, data_parallel_size * tensor_model_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        _EP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="ep")

        logger.info(
            "rank %s in world size %s is assigned as "
            "DP rank %s, PP rank %s, TP rank %s, EP rank %s", rank, world_size,
            _DP.rank_in_group, _PP.rank_in_group, _TP.rank_in_group,
            _EP.rank_in_group)

        parallel_state._TP = _TP
        parallel_state._PP = _PP
        parallel_state._DP = _DP
        parallel_state._EP = _EP
        parallel_state._DCP = _DCP
        parallel_state._offload_TP = _offload_TP
