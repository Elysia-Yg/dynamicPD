from vllm.logger import logger
from vllm.v1.core.sched import utils as sched_utils
from vllm.v1.engine import utils as engine_utils


from vllm_ascend.platform import NPUPlatform
from vllm_ascend.worker.worker import NPUWorker

from dynamicPD.vllm.vllm.outputs import RequestOutputPatch
from dynamicPD.vllm.vllm.distributed.communication_op import (
    CommunicationOpPatch,
    install_communication_op_patch,
)
from dynamicPD.vllm.vllm.distributed.parallel_state import (
    ParallelStatePatch,
    install_parallel_state_patch,
)
from dynamicPD.vllm.vllm.model_executor.layers.linear import ColumnParallelLinearPatch, RowParallelLinearPatch
from dynamicPD.vllm.vllm.model_executor.layers.logits_processor import LogitsProcessorPatch
from dynamicPD.vllm.vllm.model_executor.layers.vocab_parallel_embeding import VocabParallelEmbeddingPatch
from dynamicPD.vllm.vllm.model_executor.models.qwen2 import Qwen2AttentionPatch, Qwen2DecoderLayerPatch, Qwen2ForCausalLMPatch
from dynamicPD.vllm.vllm.model_executor.models.qwen3 import Qwen3AttentionPatch, Qwen3DecoderLayerPatch, Qwen3ForCausalLMPatch
from dynamicPD.vllm.vllm.v1.core.sched.output import SchedulerOutputPatch
from dynamicPD.vllm.vllm.v1.core.sched.scheduler import SchedulerPatch
from dynamicPD.vllm.vllm.v1.core.sched.utils import check_stop
from dynamicPD.vllm.vllm.v1.engine.core import EngineCorePatch, EngineCoreProcPatch
from dynamicPD.vllm.vllm.v1.engine.output_processor import RequestStatePatch
from dynamicPD.vllm.vllm.v1.engine.utils import get_device_indices
from dynamicPD.vllm.vllm.v1.executor.multiproc_executor import MultiprocExecutorPatch, WorkerProcPatch
from dynamicPD.vllm.vllm.v1.outputs import ModelRunnerOutputPatch
from dynamicPD.vllm.vllm.v1.request import RequestPatch

from dynamicPD.vllm_ascend.vllm_ascend.distributed.parallel_state import (
    AscendParallelStatePatch,
    install_ascend_parallel_state_patch,
)
from dynamicPD.vllm_ascend.vllm_ascend.ops.register_cuntom_ops import (
    install_custom_ops_patch,
)
from dynamicPD.vllm_ascend.vllm_ascend.ops.vocab_parallel_embedding import AscendLogitsProcessorPatch
from dynamicPD.vllm_ascend.vllm_ascend.worker.model_runner_v1 import NPUModelRunnerPatch

_installed = False

def apply_dynamicPD_patches():
    global _installed
    if _installed:
        return
    _installed = True
    orig_pre_register = NPUPlatform.pre_register_and_update
    orig_worker_init = NPUWorker.__init__
    
    @classmethod
    def patched_pre_register(cls, parser=None):
        logger.info("Applying dynamicPD patch to NPUPlatform.pre_register_and_update")
        result = orig_pre_register(parser)
        return result
    
    def patched_worker_init(self, *args, **kwargs):
        logger.info("Applying dynamicPD patch to NPUWorker.__init__")

        # vllm-ascend is loaded here; install its distributed patches before
        # the worker initializes model-parallel groups.
        import vllm_ascend.worker.worker as ascend_worker
        import vllm_ascend.distributed.parallel_state as ascend_parallel_state
        if not getattr(ascend_parallel_state,
                       "_dynamic_pd_parallel_state_patched", False):
            AscendParallelStatePatch.apply_patch()
            install_ascend_parallel_state_patch()
            ascend_parallel_state._dynamic_pd_parallel_state_patched = True
        ascend_worker.init_ascend_model_parallel = ascend_parallel_state.init_ascend_model_parallel

        # install_custom_ops_patch()
        result = orig_worker_init(self, *args, **kwargs)
        # AscendLogitsProcessorPatch.apply_patch()
        NPUModelRunnerPatch.apply_patch()
        return result
    
    NPUPlatform.pre_register_and_update = patched_pre_register
    NPUWorker.__init__ = patched_worker_init

    #vllm patch
    # ColumnParallelLinearPatch.apply_patch()
    EngineCorePatch.apply_patch()
    EngineCoreProcPatch.apply_patch()
    # LogitsProcessorPatch.apply_patch()
    ModelRunnerOutputPatch.apply_patch()
    MultiprocExecutorPatch.apply_patch()
    ParallelStatePatch.apply_patch()
    install_parallel_state_patch()
    CommunicationOpPatch.apply_patch()
    install_communication_op_patch()
    import vllm.distributed as vllm_distributed
    import vllm.distributed.communication_op as communication_op
    import vllm.distributed.parallel_state as parallel_state

    for name in (
        "get_tp_group",
        "get_dcp_group",
        "get_context_model_parallel_group",
        "get_pcp_group",
        "get_pp_group",
        "get_dp_group",
        "get_ep_group",
        "get_eplb_group",
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
    ):
        setattr(vllm_distributed, name, getattr(parallel_state, name))

    for name in (
        "tensor_model_parallel_all_reduce",
        "tensor_model_parallel_all_gather",
        "tensor_model_parallel_reduce_scatter",
        "tensor_model_parallel_gather",
        "broadcast_tensor_dict",
    ):
        setattr(vllm_distributed, name, getattr(communication_op, name))
    # Qwen2AttentionPatch.apply_patch()
    # Qwen2DecoderLayerPatch.apply_patch()
    # Qwen2ForCausalLMPatch.apply_patch()
    # Qwen3AttentionPatch.apply_patch()
    # Qwen3DecoderLayerPatch.apply_patch()
    # Qwen3ForCausalLMPatch.apply_patch()
    RequestOutputPatch.apply_patch()
    RequestPatch.apply_patch()
    RequestStatePatch.apply_patch()
    # RowParallelLinearPatch.apply_patch()
    SchedulerOutputPatch.apply_patch()
    SchedulerPatch.apply_patch()
    # VocabParallelEmbeddingPatch.apply_patch()
    WorkerProcPatch.apply_patch()
