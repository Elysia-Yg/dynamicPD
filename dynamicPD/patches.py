from vllm.logger import logger
from vllm.v1.core.sched import utils as sched_utils
from vllm.v1.engine import utils as engine_utils


from vllm_ascend.platform import NPUPlatform
from vllm_ascend.worker.worker_v1 import NPUWorker
from vllm_ascend.ops import register_custom_ops

from dynamicPD.vllm.vllm.outputs import RequestOutputPatch
from dynamicPD.vllm.vllm.distributed.parallel_state import ParallelStatePatch
from dynamicPD.vllm.vllm.engine.protocol import EngineClientPatch
from dynamicPD.vllm.vllm.forward_context import ForwardContextPatch
from dynamicPD.vllm.vllm.model_executor.layers.linear import ColumnParallelLinearPatch, RowParallelLinearPatch
from dynamicPD.vllm.vllm.model_executor.layers.logits_processor import LogitsProcessorPatch
from dynamicPD.vllm.vllm.model_executor.layers.vocab_parallel_embeding import VocabParallelEmbeddingPatch
from dynamicPD.vllm.vllm.model_executor.models.qwen2 import Qwen2AttentionPatch, Qwen2DecoderLayerPatch, Qwen2ForCausalLMPatch
from dynamicPD.vllm.vllm.model_executor.models.qwen3 import Qwen3AttentionPatch, Qwen3DecoderLayerPatch, Qwen3ForCausalLMPatch
from dynamicPD.vllm.vllm.v1.core.sched.output import SchedulerOutputPatch
from dynamicPD.vllm.vllm.v1.core.sched.scheduler import SchedulerPatch
from dynamicPD.vllm.vllm.v1.core.sched.utils import check_stop
from dynamicPD.vllm.vllm.v1.engine.async_llm import AsyncLLMPatch
from dynamicPD.vllm.vllm.v1.engine.core_client import AsyncMPClientPatch
from dynamicPD.vllm.vllm.v1.engine.core import EngineCorePatch, EngineCoreProcPatch
from dynamicPD.vllm.vllm.v1.engine.output_processor import RequestStatePatch
from dynamicPD.vllm.vllm.v1.engine.utils import get_device_indices
from dynamicPD.vllm.vllm.v1.executor.multiproc_executor import MultiprocExecutorPatch, WorkerProcPatch
from dynamicPD.vllm.vllm.v1.outputs import ModelRunnerOutputPatch
from dynamicPD.vllm.vllm.v1.request import RequestPatch

from dynamicPD.vllm_ascend.vllm_ascend.distributed.llmdatadist_c_mgr_connector import LLMDataDistCMgrConnectorWorkerPatch
from dynamicPD.vllm_ascend.vllm_ascend.ops.register_cuntom_ops import _maybe_pad_and_reduce_impl
from dynamicPD.vllm_ascend.vllm_ascend.ops.vocab_parallel_embedding import AscendLogitsProcessorPatch
from dynamicPD.vllm_ascend.vllm_ascend.worker.block_table import BlockTablePatch
from dynamicPD.vllm_ascend.vllm_ascend.worker.model_runner_v1 import AsyncNPUModelRunnerOutputPatch, NPUModelRunnerPatch

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

        result = orig_worker_init(self, *args, **kwargs)
        #放这里是为了保障该patch是在vllm-ascend被加载后才应用的;vllm-ascend的patch应该都放在这
        #vllm-ascend patch
        AscendLogitsProcessorPatch.apply_patch()
        AsyncNPUModelRunnerOutputPatch.apply_patch()
        BlockTablePatch.apply_patch()
        LLMDataDistCMgrConnectorWorkerPatch.apply_patch()
        NPUModelRunnerPatch.apply_patch()

        register_custom_ops._maybe_pad_and_reduce_impl = _maybe_pad_and_reduce_impl
        return result
    
    NPUPlatform.pre_register_and_update = patched_pre_register
    NPUWorker.__init__ = patched_worker_init

    #vllm patch
    AsyncLLMPatch.apply_patch()
    AsyncMPClientPatch.apply_patch()
    ColumnParallelLinearPatch.apply_patch()
    EngineClientPatch.apply_patch()
    EngineCorePatch.apply_patch()
    EngineCoreProcPatch.apply_patch()
    ForwardContextPatch.apply_patch()
    LogitsProcessorPatch.apply_patch()
    ModelRunnerOutputPatch.apply_patch()
    MultiprocExecutorPatch.apply_patch()
    ParallelStatePatch.apply_patch()
    Qwen2AttentionPatch.apply_patch()
    Qwen2DecoderLayerPatch.apply_patch()
    Qwen2ForCausalLMPatch.apply_patch()
    Qwen3AttentionPatch.apply_patch()
    Qwen3DecoderLayerPatch.apply_patch()
    Qwen3ForCausalLMPatch.apply_patch()
    RequestOutputPatch.apply_patch()
    RequestPatch.apply_patch()
    RequestStatePatch.apply_patch()
    RowParallelLinearPatch.apply_patch()
    SchedulerOutputPatch.apply_patch()
    SchedulerPatch.apply_patch()
    VocabParallelEmbeddingPatch.apply_patch()
    WorkerProcPatch.apply_patch()
