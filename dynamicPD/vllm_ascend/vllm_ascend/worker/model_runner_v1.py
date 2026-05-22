import math
from multiprocessing import Manager
from typing import (TYPE_CHECKING, Any, Dict, List, Optional,
                    Union, cast)

import numpy as np
import numpy.typing as npt
import torch
import torch.distributed as dist
import torch.nn as nn
from vllm.attention import get_attn_backend
from vllm.config import (CUDAGraphMode, VllmConfig)
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import (get_pp_group,
                                             get_tp_group)
from vllm.forward_context import BatchDescriptor
from vllm.logger import logger
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.multimodal.inputs import MultiModalKwargsItem, PlaceholderRange
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LazyLoader, cdiv, is_pin_memory_available)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import reorder_batch_to_split_decodes_and_prefills
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
# yapf conflicts with isort for this block
# yapf: disable
from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec
# yapf: enable
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput,
                             LogprobsTensors, ModelRunnerOutput)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput
from vllm.v1.worker.utils import (AttentionGroup, 
                                  gather_mm_placeholders,
                                  sanity_check_mm_encoder_outputs,
                                  scatter_mm_placeholders)

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.eplb.core.eplb_device_transfer_loader import \
    D2DExpertWeightLoader
from vllm_ascend.eplb.core.eplb_utils import EPLBParamUtils
from vllm_ascend.eplb.core.eplb_worker import EplbProcess
from vllm_ascend.eplb.eplb_updator import EplbUpdator
from vllm_ascend.ops.weight_prefetch import WeightPrefetchMethod
from vllm_ascend.sample.logits_processor import build_logitsprocs
from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler
from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.spec_decode.eagle_proposer import EagleProposer
from vllm_ascend.spec_decode.interface import SpecDcodeType
from vllm_ascend.spec_decode.mtp_proposer import MtpProposer
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ,
                               ProfileExecuteDuration,
                               enable_sp, is_310p, is_moe_model, lmhead_tp_enable)
from vllm_ascend.worker.npu_input_batch import CachedRequestState, InputBatch

if TYPE_CHECKING:
    import xgrammar as xgr  # type: ignore[import-untyped]
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

import torch_npu

from dynamicPD.vllm_ascend.vllm_ascend.ascend_forward_context import set_ascend_forward_context

# if true, allow tensor initialization and casting with internal format (e.g., NZ)
torch.npu.config.allow_internal_format = True

if is_310p():
    torch_npu.npu.set_compile_mode(jit_compile=False)
    ACL_FORMAT = ACL_FORMAT_FRACTAL_NZ
else:
    ACL_FORMAT = ACL_FORMAT_FRACTAL_ND


from vllm_ascend.worker.model_runner_v1 import AsyncNPUModelRunnerOutput, NPUModelRunner

from dynamicPD.patching import dynamicPDPatch

# Ascend profiler 
# experimental_config = torch_npu.profiler._ExperimentalConfig(
# 	export_type=torch_npu.profiler.ExportType.Text,
# 	profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
# 	mstx=True,
# 	aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
# 	l2_cache=False,
# 	op_attr=False,
# 	data_simplification=False,
# 	record_op_args=False
# )

# prof = torch_npu.profiler.profile(
#         activities=[
#                 torch_npu.profiler.ProfilerActivity.CPU,
#                 torch_npu.profiler.ProfilerActivity.NPU
#                 ],
#         schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1000, repeat=1, skip_first=1),
#         on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./latest_two_stream"),
#         record_shapes=False,
#         profile_memory=False,
#         with_stack=False,
#         with_modules=False,
#         with_flops=False,
#         experimental_config=experimental_config)

# mstx = torch_npu.npu.mstx()

class AsyncNPUModelRunnerOutputPatch(dynamicPDPatch[AsyncNPUModelRunnerOutput]):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        prefill_model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        prefill_sampled_token_ids: torch.Tensor,
        invalid_req_indices: list[int],
        prefill_invalid_req_indices: list[int],
        async_output_copy_stream: torch.npu.Stream,
        prefill_async_output_copy_stream: torch.npu.Stream,
        decode_stream: torch.npu.Stream,
        prefill_stream: torch.npu.Stream,
    ):
        self._model_runner_output = model_runner_output
        self._prefill_model_runner_output = prefill_model_runner_output
        self._invalid_req_indices = invalid_req_indices
        self._prefill_invalid_req_indices = prefill_invalid_req_indices

        # Event on the copy stream so we can synchronize the non-blocking copy.
        self._async_copy_ready_event = torch.npu.Event()
        self._prefill_async_copy_ready_event = torch.npu.Event()

        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        self._sampled_token_ids = sampled_token_ids
        self._prefill_sampled_token_ids = prefill_sampled_token_ids

        # Initiate the copy on a separate stream, but do not synchronize it.
        if self._model_runner_output is not None:
            with torch.npu.stream(async_output_copy_stream):
                async_output_copy_stream.wait_stream(decode_stream)
                self._sampled_token_ids_cpu = self._sampled_token_ids.to(
                    'cpu', non_blocking=True)
                self._async_copy_ready_event.record()
        if self._prefill_model_runner_output is not None and not self._prefill_model_runner_output.sampled_token_ids:
            logger.info("use prefill stream to process prefill request in decode")
            with torch.npu.stream(prefill_async_output_copy_stream):
                prefill_async_output_copy_stream.wait_stream(prefill_stream)
                self._prefill_sampled_token_ids_cpu = self._prefill_sampled_token_ids.to(
                    'cpu', non_blocking=True)
                self._prefill_async_copy_ready_event.record()

    def get_prefill_output(self) -> ModelRunnerOutput:
        if not self._prefill_model_runner_output.sampled_token_ids:
            self._prefill_async_copy_ready_event.synchronize()
            del self._prefill_sampled_token_ids
            
            valid_sampled_token_ids = self._prefill_sampled_token_ids_cpu.tolist()
            for i in self._prefill_invalid_req_indices:
                valid_sampled_token_ids[i].clear()

            logger.debug(f"_prefill_model_runner_output : {self._prefill_model_runner_output}")
            output = self._prefill_model_runner_output
            output.sampled_token_ids = valid_sampled_token_ids
            return output
        else:
            return self._prefill_model_runner_output
        
class NPUModelRunnerPatch(dynamicPDPatch[NPUModelRunner]):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.load_config = vllm_config.load_config
        self.lora_config = vllm_config.lora_config
        self.parallel_config = vllm_config.parallel_config
        self.pin_memory = is_pin_memory_available()
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.block_size = vllm_config.cache_config.block_size
        self.max_num_blocks_per_req = cdiv(self.model_config.max_model_len,
                                           self.block_size)
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        decode_max_num_seqs = getattr(self.scheduler_config,
                                      'decode_max_num_seqs', 0)
        self.max_num_reqs = max(self.scheduler_config.max_num_seqs,
                                decode_max_num_seqs)
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.device = device
        if envs_ascend.VLLM_ASCEND_ENABLE_PREFETCH_MLP:
            self.prefetch_stream = torch.npu.Stream(device=device)
        else:
            self.prefetch_stream = None
        self.dtype = self.model_config.dtype
        if envs_ascend.VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION:
            # TODO: drop the env config to use ascend sampler by default
            from vllm_ascend.sample.sampler import AscendSampler

            self.sampler = AscendSampler()
        else:
            from vllm.v1.sample.sampler import Sampler

            self.sampler = Sampler()
        self.reorder_batch_threshold: Optional[int] = None

        # Lazy initialization, these will be set after __init__
        self.kv_caches: List[torch.Tensor] = []
        self.attn_groups: list[list[AttentionGroup]] = []
        self.encoder_cache: Dict[str, torch.Tensor] = {}
        self.attn_mask = None
        self.attn_state = None
        self.prefill_attn_state = None
        self.requests: Dict[str, CachedRequestState] = {}
        self.intermediate_tensors: Optional[IntermediateTensors] = None
        self.runner_only_attn_layers: set[str] = set()

        self.ascend_config = get_ascend_config()
        if self.ascend_config.ascend_scheduler_config.enabled:
            self.chunked_prefill_enabled = self.scheduler_config.chunked_prefill_enabled
        else:
            self.chunked_prefill_enabled = True
        self.weight_prefetch_method = WeightPrefetchMethod(
            self.ascend_config.weight_prefetch_config)

        if self.cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype]
        # use_hybrid_blocks: if hybrid blocks is used.
        self.use_hybrid_blocks: bool = False
        self.need_accepted_tokens: bool = False

        self.is_multimodal_model = self.model_config.is_multimodal_model
        self.is_pooling_model = self.model_config.pooler_config is not None
        if self.is_multimodal_model:
            self.inputs_embeds = torch.zeros(
                (self.max_num_tokens, self.model_config.get_hidden_size()),
                dtype=self.dtype,
                device=self.device)
        # Set up Attention
        self.use_sparse = hasattr(self.vllm_config.model_config.hf_config,
                                  "index_topk")
        self.attn_backend = get_attn_backend(0,
                                             self.dtype,
                                             None,
                                             self.block_size,
                                             use_mla=self.model_config.use_mla,
                                             use_sparse=self.use_sparse)
        self.attn_mask_builder = AttentionMaskBuilder(
            self.scheduler_config.max_num_batched_tokens, self.dtype,
            self.device)

        # Set up speculative decoding.
        self.spec_attn_mask = None
        self.drafter: Optional[Union[NgramProposer, EagleProposer,
                                     MtpProposer]] = None
        self.actual_seq_lengths_q: list[int] = []
        self.decode_token_per_req = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            assert spec_token_num > 0
            self.decode_token_per_req = 1 + spec_token_num
            self.spec_attn_mask = torch.triu(torch.ones(2048,
                                                        2048,
                                                        dtype=torch.bool),
                                             diagonal=1).to(self.device)
            if get_pp_group().is_last_rank:
                self.drafter = get_spec_decode_method(
                    self.speculative_config.method, self.vllm_config,
                    self.device, self)
                self.rejection_sampler = AscendRejectionSampler()
            self.actual_seq_lengths_q = list(
                range(self.decode_token_per_req, self.max_num_tokens + 1,
                      self.decode_token_per_req))

        # kv role
        self.is_kv_producer = False
        self.is_kv_consumer = False
        if vllm_config.kv_transfer_config is not None:
            self.is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
            self.is_kv_consumer = vllm_config.kv_transfer_config.is_kv_consumer

        self._may_pad_kv_consumer_num_seq()

        # Persistent batch.
        self.input_ids = torch.zeros(self.max_num_tokens*8,
                                     dtype=torch.int32,
                                     device=self.device)
        self.prefill_input_ids = torch.zeros(self.max_num_tokens*8,
                                     dtype=torch.int32,
                                     device=self.device)
        self.positions = torch.zeros(self.max_num_tokens*8,
                                     dtype=torch.int64,
                                     device=self.device)
        self.prefill_positions = torch.zeros(self.max_num_tokens*8,
                                     dtype=torch.int64,
                                     device=self.device)
        self.query_start_loc = torch.zeros(self.max_num_reqs + 1,
                                           dtype=torch.int32,
                                           device=self.device)
        self.seq_lens = torch.zeros(self.max_num_reqs,
                                    dtype=torch.int32,
                                    device=self.device)
        self.slot_mapping = torch.zeros(self.max_num_tokens*8,
                                        dtype=torch.int32,
                                        device=self.device)
        self.prefill_slot_mapping = torch.zeros(self.max_num_tokens*8,
                                        dtype=torch.int32,
                                        device=self.device)

        if self.vllm_config.model_config.use_mla and \
            self.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY:
            rope_dim = self.model_config.hf_text_config.qk_rope_head_dim
            self.cos = torch.ones(self.max_num_reqs *
                                  self.decode_token_per_req,
                                  1,
                                  1,
                                  rope_dim,
                                  dtype=self.dtype,
                                  device=self.device)
            self.sin = torch.zeros(self.max_num_reqs *
                                   self.decode_token_per_req,
                                   1,
                                   1,
                                   rope_dim,
                                   dtype=self.dtype,
                                   device=self.device)
        else:
            self.cos = None
            self.sin = None

        self.uses_mrope = self.model_config.uses_mrope
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = torch.zeros((3, self.max_num_tokens + 1),
                                               dtype=torch.int64,
                                               device=self.device)
            self.mrope_positions_cpu = torch.zeros(
                (3, self.max_num_tokens + 1),
                dtype=torch.int64,
                device="cpu",
                pin_memory=True)
            self.mrope_positions_np = self.mrope_positions_cpu.numpy()

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        self.arange_np: npt.NDArray[np.int32] = np.arange(max(
            self.max_num_reqs + 1, self.model_config.max_model_len,
            self.max_num_tokens),
                                                          dtype=np.int32)
        # NOTE(woosuk): These tensors are "stateless", i.e., they are literally
        # a faster version of creating a new tensor every time. Thus, we should
        # not make any assumptions about the values in these tensors.
        self.input_ids_cpu = torch.zeros(self.max_num_tokens*8,
                                         dtype=torch.int32,
                                         device="cpu",
                                         pin_memory=True)
        self.prefill_input_ids_cpu = torch.zeros(self.max_num_tokens*8,
                                         dtype=torch.int32,
                                         device="cpu",
                                         pin_memory=True)
        self.positions_cpu = torch.zeros(self.max_num_tokens*8,
                                         dtype=torch.int64,
                                         device="cpu",
                                         pin_memory=True)
        self.prefill_positions_cpu = torch.zeros(self.max_num_tokens*8,
                                         dtype=torch.int64,
                                         device="cpu",
                                         pin_memory=True)
        self.prefill_positions_np = self.prefill_positions_cpu.numpy()
        self.positions_np = self.positions_cpu.numpy()

        self.slot_mapping_cpu = torch.zeros(self.max_num_tokens*8,
                                            dtype=torch.int32,
                                            device="cpu",
                                            pin_memory=True)
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()
        self.prefill_slot_mapping_cpu = torch.zeros(self.max_num_tokens*8,
                                            dtype=torch.int32,
                                            device="cpu",
                                            pin_memory=True)
        self.prefill_slot_mapping_np = self.prefill_slot_mapping_cpu.numpy()
        self.query_start_loc_cpu = torch.zeros(self.max_num_reqs + 1,
                                               dtype=torch.int32,
                                               device="cpu",
                                               pin_memory=True)
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()
        self.seq_lens_cpu = torch.zeros(self.max_num_reqs,
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=True)
        self.seq_lens_np = self.seq_lens_cpu.numpy()

        self.use_aclgraph = self._use_aclgraph()
        self.aclgraph_batch_sizes = list(
            reversed(self.compilation_config.cudagraph_capture_sizes))

        self.uniform_decode_query_len = 1 if not self.speculative_config else \
            1 + self.speculative_config.num_speculative_tokens
        # aclgraph dispatcher for runtime aclgraph dispatching.
        self.aclgraph_dispatcher = CudagraphDispatcher(self.vllm_config)
        # Cached outputs.
        self._draft_token_ids: Optional[Union[list[list[int]],
                                              torch.Tensor]] = None

        # NOTE: we need to use `in_profile_run` to determine whether `enable_force_load_balance` is True
        self.in_profile_run = False

        self._init_mc2_tokens_capacity()
        if is_moe_model(vllm_config):
            self.reserved_mc2_mask = torch.zeros(
                self.mc2_tokens_capacity,
                dtype=torch.bool,
                device=self.device,
            )
        else:
            self.reserved_mc2_mask = None
        self.dynamic_eplb = self.ascend_config.dynamic_eplb or self.ascend_config.expert_map_record_path
        if self.dynamic_eplb:
            EPLBParamUtils.check_dynamic_eplb(self.ascend_config.dynamic_eplb)
            EPLBParamUtils.check_expert_map_record_path(
                self.ascend_config.expert_map_record_path)
            self.is_eplb_warmuped = False
            self.policy_type = self.ascend_config.eplb_policy_type
            self.eplb_loader = D2DExpertWeightLoader()
            self.manager = Manager()
            self.shared_dict = self.manager.dict({
                "expert_map": None,
                "moe_load": None,
                "expert_maps": None
            })
            self.eplb_process = EplbProcess(shared_dict=self.shared_dict,
                                            policy_type=self.policy_type,
                                            enable_d2d=True)
            self.process = self.eplb_process._launch_process()
            ascend_config = get_ascend_config()
            self.eplb_updator = EplbUpdator(ascend_config, self.eplb_loader,
                                            self.eplb_process, self.process)

        self.use_async_scheduling = self.scheduler_config.async_scheduling
        self.async_output_copy_stream = torch.npu.Stream() if \
            self.use_async_scheduling else None
        self.prefill_async_output_copy_stream = torch.npu.Stream() if \
            self.use_async_scheduling else None
        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.model_config.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config, self.device, self.pin_memory,
                self.is_pooling_model,
                self.vllm_config.model_config.logits_processors),
            is_pooling_model=self.is_pooling_model,
            kernel_block_sizes=[[self.vllm_config.cache_config.block_size]],
        )
        self.num_accepted_tokens = self._make_buffer(self.max_num_reqs,
                                                     dtype=torch.int64)
        self.num_draft_tokens = self._make_buffer(self.max_num_reqs,
                                                  dtype=torch.int32)
        self.prefill_input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.model_config.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config, self.device, self.pin_memory,
                self.is_pooling_model,
                self.vllm_config.model_config.logits_processors),
            is_pooling_model=self.is_pooling_model,
            kernel_block_sizes=[[self.vllm_config.cache_config.block_size]],
        )
        self.finished_prefill_reqs : set[str] = set()
        self.event = torch.npu.Event()
        
        self.async_model_runner_output: List[AsyncNPUModelRunnerOutput] = []
        self.prefill_stream = torch.npu.Stream()
        self.decode_stream = torch.npu.Stream()
        self.default_stream = torch.npu.current_stream()
    
        # prof.start()

    def _prefill_update_states_after_model_execute(
            self, output_token_ids: torch.Tensor) -> None:
        """Update the cached states after model execution.

        This is used for MTP/EAGLE for hybrid models, as in linear attention,
        only the last token's state is kept. In MTP/EAGLE, for draft tokens
        the state are kept util we decide how many tokens are accepted for
        each sequence, and a shifting is done during the next iteration
        based on the number of accepted tokens.
        """
        if not self.model_config.is_hybrid or not self.speculative_config:
            return

        # Find the number of accepted tokens for each sequence.
        num_accepted_tokens = (torch.cat(
            [
                output_token_ids,
                torch.full((output_token_ids.size(0), 1),
                           -1,
                           device=output_token_ids.device),
            ],
            dim=1) == -1).int().argmax(-1).cpu().numpy()
        for i, num_tokens in enumerate(num_accepted_tokens):
            self.prefill_input_batch.num_accepted_tokens_cpu[i] = num_tokens

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            if req_id in scheduler_output.prefill_request_not_put:
                self.prefill_input_batch.remove_request(req_id)
                logger.debug(f"remove request {req_id} in prefill batch from finished reqs")
            else: 
                self.input_batch.remove_request(req_id)
                logger.debug(f"remove request {req_id} in decode batch from finished reqs")
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)
        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        prefill_cached_req_ids = self.prefill_input_batch.req_id_to_index.keys()
        scheduled_req_ids_set = set(scheduled_req_ids)
        cached_req_ids_set = set(cached_req_ids)
        prefill_cached_req_ids_set = set(prefill_cached_req_ids)
        unscheduled_req_ids = (cached_req_ids_set | prefill_cached_req_ids_set) - scheduled_req_ids_set
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.prefill_input_batch.req_id_to_index.get(req_id)
            if req_index is not None:
                self.prefill_input_batch.remove_request(req_id)
                logger.debug(f"remove request {req_id} in prefill batch from unscheduled reqs")
            else:
                self.input_batch.remove_request(req_id)
                logger.debug(f"remove request {req_id} in decode batch from unscheduled reqs")
        req_ids_to_add: list[str] = []
        req_ids_to_add_prefill: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if sampling_params and \
                sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if pooling_params:
                assert (task := pooling_params.task) is not None, (
                    "You did not set `task` in the API")
                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            backward_kwargs = {}
            backward_kwargs["mm_features"] = new_req_data.mm_features

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
                **backward_kwargs,
            )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self._init_mrope_positions(self.requests[req_id])

            logger.debug(f"num_computed_tokens : {new_req_data.num_computed_tokens}, prompt_token_ids length: {len(new_req_data.prompt_token_ids)}")
            logger.debug(f"prefill_request_not_put: {scheduler_output.prefill_request_not_put}")
            if req_id in scheduler_output.prefill_request_not_put:
                req_ids_to_add_prefill.append(req_id)
                logger.info(f"add request {req_id} in prefill batch")
            else: 
                req_ids_to_add.append(req_id)
                logger.info(f"add request {req_id} in decode batch")

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_data.resumed_from_preemption[i]

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens

            num_new_tokens = len(req_data.new_token_ids[i])
            if not is_last_rank:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker.
                new_token_ids = req_data.new_token_ids[i]
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec tokens.
                num_new_tokens = (num_computed_tokens + len(new_token_ids) -
                                  req_state.num_tokens)
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(
                        new_token_ids[-num_new_tokens:])

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids,
                                                  new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.prefill_input_batch.req_id_to_index.get(req_id)
            logger.debug(f"rnum_prompt_tokens : {req_state.num_prompt_tokens}, num_computed_tokens : {num_computed_tokens}, num_new_tokens : {num_new_tokens}, req_id : {req_id}")
            if req_index is not None:
                if req_id in self.finished_prefill_reqs:
                    self.prefill_input_batch.remove_request(req_id)
                    logger.info(f"remove request {req_id} in prefill batch")
                    self.finished_prefill_reqs.remove(req_id)
                    
            prefill_req_index = self.prefill_input_batch.req_id_to_index.get(req_id)
            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None and prefill_req_index is None:
                if req_id in scheduler_output.prefill_request_not_put:
                    req_ids_to_add_prefill.append(req_id)
                    logger.info(f"add request {req_id} in prefill batch before persistent")
                else: 
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                    req_ids_to_add.append(req_id)
                    logger.info(f"add request {req_id} in decode batch before persistent")
                continue
            
            if req_index is not None:
                # Update the persistent batch.
                self.input_batch.num_computed_tokens_cpu[req_index] = (
                    num_computed_tokens)
                if new_block_ids is not None:
                    self.input_batch.block_table.append_row(
                        new_block_ids, req_index)

                # For the last rank, we don't need to update the token_ids_cpu
                # because the sampled tokens are already cached.
                if not is_last_rank:
                    # Add new_token_ids to token_ids_cpu.
                    start_token_index = num_computed_tokens
                    end_token_index = num_computed_tokens + len(new_token_ids)
                    self.input_batch.token_ids_cpu[
                        req_index,
                        start_token_index:end_token_index] = new_token_ids
                    self.input_batch.num_tokens_no_spec[
                        req_index] = end_token_index
                    self.input_batch.num_tokens[req_index] = end_token_index

                # Add spec_token_ids to token_ids_cpu.
                spec_token_ids = (
                    scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))
                if spec_token_ids:
                    num_spec_tokens = len(spec_token_ids)
                    start_index = self.input_batch.num_tokens_no_spec[req_index]
                    end_token_index = start_index + num_spec_tokens
                    self.input_batch.token_ids_cpu[
                        req_index, start_index:end_token_index] = spec_token_ids
                    # NOTE(woosuk): `num_tokens` here may include spec tokens.
                    self.input_batch.num_tokens[req_index] += num_spec_tokens
                    
            if prefill_req_index is not None:
                # Update the persistent batch.
                self.prefill_input_batch.num_computed_tokens_cpu[prefill_req_index] = (
                    num_computed_tokens)
                if new_block_ids is not None:
                    self.prefill_input_batch.block_table.append_row(
                        new_block_ids, prefill_req_index)

                # For the last rank, we don't need to update the token_ids_cpu
                # because the sampled tokens are already cached.
                if not is_last_rank:
                    # Add new_token_ids to token_ids_cpu.
                    start_token_index = num_computed_tokens
                    end_token_index = num_computed_tokens + len(new_token_ids)
                    self.prefill_input_batch.token_ids_cpu[
                        prefill_req_index,
                        start_token_index:end_token_index] = new_token_ids
                    self.prefill_input_batch.num_tokens_no_spec[
                        prefill_req_index] = end_token_index
                    self.prefill_input_batch.num_tokens[prefill_req_index] = end_token_index

                # Add spec_token_ids to token_ids_cpu.
                spec_token_ids = (
                    scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))
                if spec_token_ids:
                    num_spec_tokens = len(spec_token_ids)
                    start_index = self.prefill_input_batch.num_tokens_no_spec[prefill_req_index]
                    end_token_index = start_index + num_spec_tokens
                    self.prefill_input_batch.token_ids_cpu[
                        prefill_req_index, start_index:end_token_index] = spec_token_ids
                    # NOTE(woosuk): `num_tokens` here may include spec tokens.
                    self.prefill_input_batch.num_tokens[prefill_req_index] += num_spec_tokens

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            self.input_batch.add_request(req_state)
        for req_id in req_ids_to_add_prefill:
            req_state = self.requests[req_id]
            self.prefill_input_batch.add_request(req_state)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        self.prefill_input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()
        self.prefill_input_batch.refresh_metadata()

    def _prefill_calc_mrope_positions(self, scheduler_output: "SchedulerOutput"):
        mrope_pos_ptr = 0
        for index, req_id in enumerate(self.prefill_input_batch.req_ids):
            req = self.requests[req_id]
            assert req.mrope_positions is not None

            num_computed_tokens = \
                self.prefill_input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = \
                scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = len(req.prompt_token_ids)

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0,
                                      num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(
                    0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's mrope_positions are pre-computed
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.mrope_positions_cpu[:, dst_start:dst_end] = \
                    req.mrope_positions[:,src_start:src_end]

                mrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's mrope_positions on-the-fly
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + completion_part_len
                MRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.mrope_positions_np,
                    out_offset=dst_start,
                    mrope_position_delta=req.mrope_position_delta,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                mrope_pos_ptr += completion_part_len

    def _prefill_execute_mm_encoder(self, scheduler_output: "SchedulerOutput"):
        scheduled_encoder_inputs = scheduler_output.prefill_scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_kwargs, mm_hashes_pos = self._prefill_batch_mm_kwargs_from_scheduler(
            scheduler_output)
        encoder_outputs = []

        for _, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
                mm_kwargs,
                device=self.device,
                pin_memory=True,
        ):
            # Run the encoder.
            # `curr_group_outputs` is either of the following:
            # 1. A tensor of shape (num_items, feature_size, hidden_size)
            # in case feature_size is fixed across all multimodal items.
            # 2. A list or tuple (length: num_items) of tensors, each of shape
            # (feature_size, hidden_size) in case the feature size is dynamic
            # depending on the input multimodal items.
            curr_group_outputs = self.model.get_multimodal_embeddings(
                **mm_kwargs_group)

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=num_items,
            )

            for output in curr_group_outputs:
                encoder_outputs.append(output)

        for (mm_hash, pos_info), output in zip(mm_hashes_pos, encoder_outputs):
            self.encoder_cache[mm_hash] = scatter_mm_placeholders(
                output,
                is_embed=pos_info.is_embed,
            )

    def _prefill_batch_mm_kwargs_from_scheduler(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[list[MultiModalKwargsItem], list[tuple[str, PlaceholderRange]]]:
        """Batch multimodal kwargs from scheduled encoder inputs.

        Args:
            scheduler_output: The scheduler output containing scheduled encoder
              inputs.

        Returns:
            A tuple of (mm_kwargs, req_ids_pos) where:
            - mm_kwargs: List of multimodal kwargs items to be batched
            - mm_hashes_pos: List of (mm_hash, position_info) tuples
        """
        scheduled_encoder_inputs = scheduler_output.prefill_scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return [], []
        # Batch the multi-modal inputs.
        mm_kwargs = list[MultiModalKwargsItem]()
        # list of tuple (mm_hash, position_info)
        mm_hashes_pos = list[tuple[str, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]
            assert req_state.mm_features is not None
            for mm_input_id in encoder_input_ids:
                mm_feature = req_state.mm_features[mm_input_id]
                mm_hash = mm_feature.identifier
                mm_kwargs.append(mm_feature.data)
                mm_hashes_pos.append((mm_hash, mm_feature.mm_position))

        return mm_kwargs, mm_hashes_pos

    def _prefill_gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> list[torch.Tensor]:

        def _iter_mm_features(req_state: CachedRequestState):
            assert req_state.mm_features is not None
            for mm_feature in req_state.mm_features:
                pos_info = mm_feature.mm_position
                yield mm_feature.identifier, pos_info, getattr(
                    pos_info, "is_embed", None)

        mm_embeds: list[torch.Tensor] = []

        for req_id in self.prefill_input_batch.req_ids:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens

            for mm_hash, pos_info, is_embed in _iter_mm_features(req_state):
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens,
                )
                assert start_idx < end_idx

                encoder_output = self.encoder_cache.get(mm_hash, None)
                assert encoder_output is not None, \
                    f"Encoder cache miss for {mm_hash}."

                if is_embed is not None:
                    is_embed = is_embed[start_idx:end_idx]

                mm_embeds_item = gather_mm_placeholders(
                    encoder_output[start_idx:end_idx],
                    is_embed=is_embed,
                )
                mm_embeds.append(mm_embeds_item)
        return mm_embeds
    
    def _prepare_input_ids(self, total_num_scheduled_tokens: int,
                           cu_num_tokens: np.ndarray, prefill_in_decode: bool) -> None:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        NPU need to be copied into the corresponding slots into input_ids."""
        
        input_batch = self.input_batch if not prefill_in_decode else self.prefill_input_batch 
        input_ids = self.input_ids if not prefill_in_decode else self.prefill_input_ids
        input_ids_cpu = self.input_ids_cpu if not prefill_in_decode else self.prefill_input_ids_cpu

        if input_batch.prev_sampled_token_ids is None:
            # Normal scheduling case
            input_ids[:total_num_scheduled_tokens].copy_(
                input_ids_cpu[:total_num_scheduled_tokens],
                non_blocking=True)
            return

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the NPU from prev_sampled_token_ids.
        prev_req_id_to_index = input_batch.prev_req_id_to_index
        assert prev_req_id_to_index is not None
        flattened_indices = []
        prev_common_req_indices = []
        indices_match = True
        max_flattened_index = -1
        for req_id, cur_index in input_batch.req_id_to_index.items():
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                # We need to compute the flattened input_ids index of the
                # last token in each common request.
                flattened_index = cu_num_tokens[cur_index].item() - 1
                flattened_indices.append(flattened_index)
                indices_match &= (prev_index == flattened_index)
                max_flattened_index = max(max_flattened_index, flattened_index)
        num_commmon_tokens = len(flattened_indices)
        if num_commmon_tokens < total_num_scheduled_tokens:
            # If not all requests are decodes from the last iteration,
            # We need to copy the input_ids_cpu to the NPU first.
            input_ids[:total_num_scheduled_tokens].copy_(
                input_ids_cpu[:total_num_scheduled_tokens],
                non_blocking=True)
        if num_commmon_tokens == 0:
            # No requests in common with the previous iteration
            # So input_ids_cpu will have all the input ids.
            return
        if indices_match and max_flattened_index == (num_commmon_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            input_ids[:num_commmon_tokens].copy_(
                input_batch.prev_sampled_token_ids[:num_commmon_tokens,
                                                        0],
                non_blocking=True)
            return
        # Upload the index tensors asynchronously
        # so the scatter can be non-blocking.
        input_ids_index_tensor = torch.tensor(flattened_indices,
                                              dtype=torch.int64,
                                              pin_memory=self.pin_memory).to(
                                                  self.device,
                                                  non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_common_req_indices,
            dtype=torch.int64,
            pin_memory=self.pin_memory).to(self.device, non_blocking=True)
        input_ids.scatter_(dim=0,
                                index=input_ids_index_tensor,
                                src=input_batch.prev_sampled_token_ids[
                                    prev_common_req_indices_tensor, 0])

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """
        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.
        """
        # Attention free models have zero kv_cache_goups, however models
        # like Mamba are also attention free but use the kv_cache for
        # keeping its internal state. This is why we check the number
        # of kv_cache groups instead of solely checking
        # for self.model_config.is_attention_free.
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return

        if self.reorder_batch_threshold is not None:
            reorder_batch_to_split_decodes_and_prefills(
                self.input_batch,
                scheduler_output,
                decode_threshold=self.reorder_batch_threshold)
            reorder_batch_to_split_decodes_and_prefills(
                self.prefill_input_batch,
                scheduler_output,
                decode_threshold=self.reorder_batch_threshold
            )

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> tuple[dict[str, Any], torch.Tensor, np.ndarray, int, torch.Tensor,
               int, torch.Tensor, SpecDecodeMetadata, Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor], int]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens - scheduler_output.prefill_total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get the number of scheduled tokens for each request.
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_tokens = sum(tokens)
        if num_tokens != total_num_scheduled_tokens:
            logger.info(f"num_tokens {num_tokens} does not match total_num_scheduled_tokens {total_num_scheduled_tokens}, req_ids: {req_ids}, tokens: {tokens}")
            logger.info(f"prefill_req_ids : {scheduler_output.prefill_request_not_put}")
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = num_scheduled_tokens.max()
        num_valid_tokens = np.array([
            num_tokens -
            len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
            for num_tokens, i in zip(tokens, req_ids)
        ],
                                    dtype=np.int32)

        if (self.use_aclgraph and total_num_scheduled_tokens
                <= self.aclgraph_batch_sizes[-1]):
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                total_num_scheduled_tokens)
        elif self.use_aclgraph and enable_sp(self.vllm_config):
            # When using aclgraph, if total_num_scheduled_tokens exceeds the maximum graph size,
            # the model will fall back to running its FX graph in eager mode.
            # In this case, when sequence parallelism is enabled, we need to pad tokens to align
            # with tp_size because pad_size cannot be captured by the FX graph
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            num_input_tokens = math.ceil(
                total_num_scheduled_tokens / tp_size) * tp_size
        else:
            # Eager mode.
            num_input_tokens = total_num_scheduled_tokens

        # Get the attention state.
        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens,
                                            num_valid_tokens)
        self.attn_state = attn_state  # type: ignore

        # Determine if it's a splitfuse batch
        with_prefill = attn_state not in [
            AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding
        ]

        self.query_lens = torch.from_numpy(num_scheduled_tokens)
        enable_dbo = self._check_dbo_is_valid(self.query_lens.tolist(),
                                              attn_state,
                                              total_num_scheduled_tokens)

        # Get info across DP ranks.
        # NOTE: maybe_padded_num_tokens is only used when using TorchAir with DP,
        # Otherwise, it's just max_tokens_across_dp_cpu
        (maybe_padded_num_tokens, num_tokens_across_dp, with_prefill,
         enable_dbo) = self._sync_metadata_across_dp(num_input_tokens,
                                                     with_prefill, enable_dbo)

        # TODO: Now that num_input_tokens is basically identical with maybe_padded_num_tokens
        # We should consider removing maybe_padded_num_tokens later
        num_input_tokens = maybe_padded_num_tokens

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)

        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions_cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])

        # Prepare input_ids.
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])

        # Prepare some information for building Attention-Metadata
        # Compute and commit slot mapping
        self.input_batch.block_table.compute_slot_mapping(
            req_indices, positions_np)
        self.input_batch.block_table.commit_slot_mapping(
            total_num_scheduled_tokens)

        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens
        self.query_start_loc[:num_reqs + 1].copy_(
            self.query_start_loc_cpu[:num_reqs + 1], non_blocking=True)

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                       non_blocking=True)

        # Fill unused with -1. Needed for reshape_and_cache
        self.query_start_loc[num_reqs + 1:].fill_(-1)
        self.seq_lens[num_reqs:].fill_(0)

        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        # Copy the tensors to the NPU.
        self._prepare_input_ids(total_num_scheduled_tokens, cu_num_tokens, False)
        self.positions_cpu[total_num_scheduled_tokens:num_input_tokens].zero_()
        self.positions[:num_input_tokens].copy_(
            self.positions_cpu[:num_input_tokens], non_blocking=True)

        # Make Attention metadata
        positions_cpu = self.positions_cpu[:num_input_tokens]
        positions = self.positions[:num_input_tokens]
        seq_lens_cpu = self.seq_lens_cpu[:num_reqs]
        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens,
                                            num_valid_tokens)
        self.attn_mask = self._make_attention_mask(seq_lens=seq_lens_cpu,
                                                   position=positions_cpu,
                                                   attn_state=attn_state)
        self.attn_state = attn_state  # type: ignore

        self.with_prefill = with_prefill
        self.num_tokens_across_dp = num_tokens_across_dp
        self._update_graph_pad_size(with_prefill, maybe_padded_num_tokens)
        attn_metadata: dict[str, Any] = {}

        # _prepare_inputs may reorder the batch, so we must gather
        # multi-modal outputs after that to ensure the correct order
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)

            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.input_ids[:total_num_scheduled_tokens]
            if mm_embeds:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids, mm_embeds)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:total_num_scheduled_tokens].copy_(
                inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the ACL graph.
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
        positions = self.positions[:num_input_tokens]
        input_ids, positions = self._update_input_ids_and_positions(
            input_ids, positions, num_input_tokens, with_prefill,
            maybe_padded_num_tokens)

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                self.intermediate_tensors[k][:num_input_tokens].copy_(
                    v[:num_input_tokens], non_blocking=True)
            intermediate_tensors = IntermediateTensors({
                k: v[:num_input_tokens]
                for k, v in self.intermediate_tensors.items()
            })

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            spec_decode_metadata = None
            logits_indices = torch.from_numpy(cu_num_tokens - 1).to(
                self.device, non_blocking=True)
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for req_id, draft_token_ids in (
                set(scheduler_output.scheduled_spec_decode_tokens.items())
                - set(scheduler_output.prefill_scheduled_spec_decode_tokens.items())
            ):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens, False)
            logits_indices = spec_decode_metadata.logits_indices
            self.num_draft_tokens.np[:num_reqs] = num_draft_tokens
            self.num_draft_tokens.np[num_reqs:].fill(0)
            self.num_draft_tokens.copy_to_gpu()

        # Used in the below loop.
        # query_start_loc_cpu = self.query_start_loc.cpu[:num_reqs + 1]
        num_computed_tokens_cpu = (
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs])
        spec_decode_common_attn_metadata = None
        if use_spec_decode and self.need_accepted_tokens:
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs])
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            if isinstance(kv_cache_group_spec.kv_cache_spec,
                          EncoderOnlyAttentionSpec):
                # Encoder-only layers do not have KV cache, so we need to
                # create a dummy block table and slot mapping for them.
                blk_table_tensor = torch.zeros(
                    (num_reqs, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (total_num_scheduled_tokens, ),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_table[kv_cache_group_id]
                blk_table_tensor = blk_table.get_device_tensor()
                slot_mapping = blk_table.slot_mapping_cpu[:
                                                          total_num_scheduled_tokens]
                self.slot_mapping[:total_num_scheduled_tokens].copy_(
                    slot_mapping[:total_num_scheduled_tokens],
                    non_blocking=True,
                )
                self.slot_mapping[total_num_scheduled_tokens:].fill_(0)

            # Make AscendCommonAttentionMetadata
            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.query_start_loc[:num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc_cpu[:num_reqs + 1],
                seq_lens_cpu=self.seq_lens_cpu,
                seq_lens=self.seq_lens_cpu[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                num_input_tokens=num_input_tokens,
                actual_seq_lengths_q=self.actual_seq_lengths_q,
                # TODO: change this to the right block table for linear attn
                block_table_tensor=blk_table_tensor[:num_reqs],
                slot_mapping=self.slot_mapping,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                positions=self.positions,
                attn_mask=self.attn_mask,
                spec_attn_mask=self.spec_attn_mask,
                attn_state=self.attn_state,
                enable_dbo_across_dp=enable_dbo,
                is_only_prefill=bool(np.all(num_valid_tokens != 1)),
                max_query_len=max_num_scheduled_tokens,
                graph_pad_size=self.graph_pad_size,
                decode_token_per_req=self.decode_token_per_req,
                cos=self.cos,
                sin=self.sin,
            )

            if self.speculative_config and \
                spec_decode_common_attn_metadata is None:
                spec_decode_common_attn_metadata = common_attn_metadata

            for attn_group in self.attn_groups[kv_cache_group_id]:
                common_prefix_len = 0
                extra_attn_metadata_args = {}
                builder = attn_group.get_metadata_builder()
                if isinstance(builder, GDNAttentionMetadataBuilder
                              ) or self.model_config.runner_type == "pooling":
                    if use_spec_decode:
                        extra_attn_metadata_args = dict(
                            num_accepted_tokens=self.num_accepted_tokens.
                            gpu[:num_reqs],
                            num_draft_tokens=self.num_draft_tokens.
                            gpu[:num_reqs],
                        )
                    attn_metadata_i = builder.build(
                        common_prefix_len=common_prefix_len,
                        common_attn_metadata=common_attn_metadata,
                        **extra_attn_metadata_args)
                else:
                    attn_metadata_i = builder.build(
                        common_prefix_len=common_prefix_len,
                        common_attn_metadata=common_attn_metadata,
                        model=self.get_model(),
                        **extra_attn_metadata_args)

                for layer_name in attn_group.layer_names:
                    attn_metadata[layer_name] = attn_metadata_i

        if lmhead_tp_enable():
            max_num_reqs_across_dp = maybe_padded_num_tokens if not with_prefill else self.max_num_reqs
            logits_indices = nn.functional.pad(
                logits_indices,
                (0, max_num_reqs_across_dp - logits_indices.shape[0]))

        return (attn_metadata, positions, num_scheduled_tokens,
                num_input_tokens, num_tokens_across_dp,
                maybe_padded_num_tokens, logits_indices, spec_decode_metadata,
                input_ids, inputs_embeds, intermediate_tensors,
                max_num_scheduled_tokens)

    def _prepare_prefill_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> tuple[dict[str, Any], torch.Tensor, np.ndarray, int, torch.Tensor,
               int, torch.Tensor, SpecDecodeMetadata, Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor], int]:
        total_num_scheduled_tokens = scheduler_output.prefill_total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.prefill_input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.prefill_input_batch.block_table.commit_block_table(num_reqs)

        # Get the number of scheduled tokens for each request.
        req_ids = self.prefill_input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = num_scheduled_tokens.max()
        num_valid_tokens = np.array([
            num_tokens -
            len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
            for num_tokens, i in zip(tokens, req_ids)
        ],
                                    dtype=np.int32)

        if (self.use_aclgraph and total_num_scheduled_tokens
                <= self.aclgraph_batch_sizes[-1]):
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                total_num_scheduled_tokens)
        elif self.use_aclgraph and enable_sp(self.vllm_config):
            # When using aclgraph, if total_num_scheduled_tokens exceeds the maximum graph size,
            # the model will fall back to running its FX graph in eager mode.
            # In this case, when sequence parallelism is enabled, we need to pad tokens to align
            # with tp_size because pad_size cannot be captured by the FX graph
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            num_input_tokens = math.ceil(
                total_num_scheduled_tokens / tp_size) * tp_size
        else:
            # Eager mode.
            num_input_tokens = total_num_scheduled_tokens

        # Get the attention state.
        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens,
                                            num_valid_tokens)
        self.attn_state = attn_state  # type: ignore

        # Determine if it's a splitfuse batch
        with_prefill = attn_state not in [
            AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding
        ]

        self.query_lens = torch.from_numpy(num_scheduled_tokens)
        enable_dbo = self._check_dbo_is_valid(self.query_lens.tolist(),
                                              attn_state,
                                              total_num_scheduled_tokens)

        # Get info across DP ranks.
        # NOTE: maybe_padded_num_tokens is only used when using TorchAir with DP,
        # Otherwise, it's just max_tokens_across_dp_cpu
        (maybe_padded_num_tokens, num_tokens_across_dp, with_prefill,
         enable_dbo) = self._sync_metadata_across_dp(num_input_tokens,
                                                     with_prefill, enable_dbo)

        # TODO: Now that num_input_tokens is basically identical with maybe_padded_num_tokens
        # We should consider removing maybe_padded_num_tokens later
        num_input_tokens = maybe_padded_num_tokens

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.prefill_input_batch, num_scheduled_tokens)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)

        positions_np = self.prefill_positions_np[:total_num_scheduled_tokens]
        logger.info(f"positions_np shape: {positions_np.shape}, arange shape: {arange.shape}, prefill_total_num_scheduled_tokens: {total_num_scheduled_tokens}")
        np.add(self.prefill_input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        logger.info(f"uses_mrope: {self.uses_mrope}")
        if self.uses_mrope:
            self._prefill_calc_mrope_positions(scheduler_output)

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions_cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.prefill_input_batch.token_ids_cpu.shape[1])

        # Prepare input_ids.
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.prefill_input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.prefill_input_ids_cpu[:total_num_scheduled_tokens])

        # Prepare some information for building Attention-Metadata
        # Compute and commit slot mapping
        self.prefill_input_batch.block_table.compute_slot_mapping(
            req_indices, positions_np)
        self.prefill_input_batch.block_table.commit_slot_mapping(
            total_num_scheduled_tokens)

        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens
        self.query_start_loc[:num_reqs + 1].copy_(
            self.query_start_loc_cpu[:num_reqs + 1], non_blocking=True)

        self.seq_lens_np[:num_reqs] = (
            self.prefill_input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                       non_blocking=True)

        # Fill unused with -1. Needed for reshape_and_cache
        self.query_start_loc[num_reqs + 1:].fill_(-1)
        self.seq_lens[num_reqs:].fill_(0)

        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        # Copy the tensors to the NPU.
        self._prepare_input_ids(total_num_scheduled_tokens, cu_num_tokens, True)
        self.prefill_positions_cpu[total_num_scheduled_tokens:num_input_tokens].zero_()
        self.prefill_positions[:num_input_tokens].copy_(
            self.prefill_positions_cpu[:num_input_tokens], non_blocking=True)

        # Make Attention metadata
        positions_cpu = self.prefill_positions_cpu[:num_input_tokens]
        positions = self.prefill_positions[:num_input_tokens]
        seq_lens_cpu = self.seq_lens_cpu[:num_reqs]
        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens,
                                            num_valid_tokens)
        self.attn_mask = self._make_attention_mask(seq_lens=seq_lens_cpu,
                                                   position=positions_cpu,
                                                   attn_state=attn_state)
        self.attn_state = attn_state  # type: ignore

        self.with_prefill = with_prefill
        self.num_tokens_across_dp = num_tokens_across_dp
        self._update_graph_pad_size(with_prefill, maybe_padded_num_tokens)
        attn_metadata: dict[str, Any] = {}

        # _prepare_inputs may reorder the batch, so we must gather
        # multi-modal outputs after that to ensure the correct order
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._prefill_execute_mm_encoder(scheduler_output)
            mm_embeds = self._prefill_gather_mm_embeddings(scheduler_output)

            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.prefill_input_ids[:total_num_scheduled_tokens]
            if mm_embeds:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids, mm_embeds)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:total_num_scheduled_tokens].copy_(
                inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the ACL graph.
            input_ids = self.prefill_input_ids[:num_input_tokens]
            inputs_embeds = None
        positions = self.prefill_positions[:num_input_tokens]
        input_ids, positions = self._update_input_ids_and_positions(
            input_ids, positions, num_input_tokens, with_prefill,
            maybe_padded_num_tokens)

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                self.intermediate_tensors[k][:num_input_tokens].copy_(
                    v[:num_input_tokens], non_blocking=True)
            intermediate_tensors = IntermediateTensors({
                k: v[:num_input_tokens]
                for k, v in self.intermediate_tensors.items()
            })

        use_spec_decode = len(
            scheduler_output.prefill_scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            spec_decode_metadata = None
            logits_indices = torch.from_numpy(cu_num_tokens - 1).to(
                self.device, non_blocking=True)
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.prefill_scheduled_spec_decode_tokens.items()):
                req_idx = self.prefill_input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens, True)
            logits_indices = spec_decode_metadata.logits_indices
            self.num_draft_tokens.np[:num_reqs] = num_draft_tokens
            self.num_draft_tokens.np[num_reqs:].fill(0)
            self.num_draft_tokens.copy_to_gpu()

        # Used in the below loop.
        # query_start_loc_cpu = self.query_start_loc.cpu[:num_reqs + 1]
        num_computed_tokens_cpu = (
            self.prefill_input_batch.num_computed_tokens_cpu_tensor[:num_reqs])
        spec_decode_common_attn_metadata = None
        if use_spec_decode and self.need_accepted_tokens:
            self.num_accepted_tokens.np[:num_reqs] = (
                self.prefill_input_batch.num_accepted_tokens_cpu[:num_reqs])
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            if isinstance(kv_cache_group_spec.kv_cache_spec,
                          EncoderOnlyAttentionSpec):
                # Encoder-only layers do not have KV cache, so we need to
                # create a dummy block table and slot mapping for them.
                blk_table_tensor = torch.zeros(
                    (num_reqs, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (total_num_scheduled_tokens, ),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                blk_table = self.prefill_input_batch.block_table[kv_cache_group_id]
                blk_table_tensor = blk_table.get_device_tensor()
                slot_mapping = blk_table.slot_mapping_cpu[:
                                                          total_num_scheduled_tokens]
                self.prefill_slot_mapping[:total_num_scheduled_tokens].copy_(
                    slot_mapping[:total_num_scheduled_tokens],
                    non_blocking=True,
                )
                self.prefill_slot_mapping[total_num_scheduled_tokens:].fill_(0)
            # Make AscendCommonAttentionMetadata
            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.query_start_loc[:num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc_cpu[:num_reqs + 1],
                seq_lens_cpu=self.seq_lens_cpu,
                seq_lens=self.seq_lens_cpu[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                num_input_tokens=num_input_tokens,
                actual_seq_lengths_q=self.actual_seq_lengths_q,
                # TODO: change this to the right block table for linear attn
                block_table_tensor=blk_table_tensor[:num_reqs],
                slot_mapping=self.prefill_slot_mapping,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                positions=self.prefill_positions,
                attn_mask=self.attn_mask,
                spec_attn_mask=self.spec_attn_mask,
                attn_state=self.attn_state,
                enable_dbo_across_dp=enable_dbo,
                is_only_prefill=bool(np.all(num_valid_tokens != 1)),
                max_query_len=max_num_scheduled_tokens,
                graph_pad_size=self.graph_pad_size,
                decode_token_per_req=self.decode_token_per_req,
                cos=self.cos,
                sin=self.sin,
            )

            if self.speculative_config and \
                spec_decode_common_attn_metadata is None:
                spec_decode_common_attn_metadata = common_attn_metadata

            for attn_group in self.attn_groups[kv_cache_group_id]:
                common_prefix_len = 0
                extra_attn_metadata_args = {}
                builder = attn_group.get_metadata_builder()
                if isinstance(builder, GDNAttentionMetadataBuilder
                              ) or self.model_config.runner_type == "pooling":
                    if use_spec_decode:
                        extra_attn_metadata_args = dict(
                            num_accepted_tokens=self.num_accepted_tokens.
                            gpu[:num_reqs],
                            num_draft_tokens=self.num_draft_tokens.
                            gpu[:num_reqs],
                        )
                    attn_metadata_i = builder.build(
                        common_prefix_len=common_prefix_len,
                        common_attn_metadata=common_attn_metadata,
                        **extra_attn_metadata_args)
                else:
                    attn_metadata_i = builder.build(
                        common_prefix_len=common_prefix_len,
                        common_attn_metadata=common_attn_metadata,
                        model=self.get_model(),
                        **extra_attn_metadata_args)

                for layer_name in attn_group.layer_names:
                    attn_metadata[layer_name] = attn_metadata_i

        if lmhead_tp_enable():
            max_num_reqs_across_dp = maybe_padded_num_tokens if not with_prefill else self.max_num_reqs
            logits_indices = nn.functional.pad(
                logits_indices,
                (0, max_num_reqs_across_dp - logits_indices.shape[0]))

        return (attn_metadata, positions, num_scheduled_tokens,
                num_input_tokens, num_tokens_across_dp,
                maybe_padded_num_tokens, logits_indices, spec_decode_metadata,
                input_ids, inputs_embeds, intermediate_tensors,
                max_num_scheduled_tokens)

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
        prefill_in_decode: bool,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1
        # Step 1. [4, 5, 8, 9, 11]
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        total_num_sampled_tokens = cu_num_sampled_tokens[-1]
        # Step 2. [0, 0, 0, 0, 4, 5, 5, 5, 8, 9, 9]
        cumsums_offsets = np.repeat(cu_num_sampled_tokens - num_sampled_tokens,
                                    num_sampled_tokens)
        # Step 3. [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        arange = self.arange_np[:total_num_sampled_tokens] - cumsums_offsets
        # Step 4. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 5. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # [3, 3, 5, 5, 6]
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        total_num_draft_tokens = cu_num_draft_tokens[-1]
        # [0, 0, 0, 3, 3, 5]
        cumsums_offsets = np.repeat(cu_num_draft_tokens - num_draft_tokens,
                                    num_draft_tokens)
        # [0, 1, 2, 0, 1, 0]
        arange = self.arange_np[:total_num_draft_tokens] - cumsums_offsets
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> NPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True)
        logits_indices = torch.from_numpy(logits_indices).to(self.device,
                                                             non_blocking=True)
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True)
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True)

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        if prefill_in_decode:
            draft_token_ids = self.prefill_input_ids[logits_indices]
        else:
            draft_token_ids = self.input_ids[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        metadata = SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )
        return metadata
    
    def prefill_apply_grammar_bitmask(
        self,
        scheduler_output: "SchedulerOutput",
        logits: torch.Tensor,
    ) -> torch.Tensor:
        grammar_bitmask = scheduler_output.prefill_grammar_bitmask

        # We receive the structured output bitmask from the scheduler,
        # compacted to contain bitmasks only for structured output requests.
        # The order of the requests in the bitmask is not guaranteed to be the
        # same as the order of the requests in the gpu runner's batch. We need
        # to sort the bitmask to match the order of the requests used here.

        # Get the batch indices of the structured output requests.
        # Keep track of the number of speculative tokens scheduled for every
        # request in the batch, as the logit indices are offset by this amount.
        struct_out_req_batch_indices: dict[str, int] = {}
        cumulative_offset = 0
        seq = sorted(self.prefill_input_batch.req_id_to_index.items(),
                     key=lambda x: x[1])
        for req_id, batch_index in seq:
            logit_index = batch_index + cumulative_offset
            cumulative_offset += len(
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            if req_id in scheduler_output.prefill_structured_output_request_ids:
                struct_out_req_batch_indices[req_id] = logit_index

        out_indices = []

        # Reorder the bitmask to match the order of the requests in the batch.
        sorted_bitmask = np.zeros_like(grammar_bitmask,
                                       shape=(logits.shape[0],
                                              grammar_bitmask.shape[1]))
        cumulative_index = 0
        seq = sorted(scheduler_output.prefill_structured_output_request_ids.items(),
                     key=lambda x: x[1])
        for req_id, _ in seq:
            logit_index = struct_out_req_batch_indices[req_id]
            num_spec_tokens = len(
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            for i in range(1 + num_spec_tokens):
                sorted_bitmask[logit_index + i] = \
                    grammar_bitmask[cumulative_index + i]
                out_indices.append(logit_index + i)
            cumulative_index += 1 + num_spec_tokens
        grammar_bitmask = sorted_bitmask

        # Serialization of np.ndarray is much more efficient than a tensor,
        # so we receive it in that format.
        grammar_bitmask = torch.from_numpy(grammar_bitmask)

        # NOTE:
        # 1. XGrammar bitmask applying only supports CPU and GPU.
        # 2. The logits and bitmask should be on the same device.
        # 3. XGrammar logits on CPU only supports float32 dtype.
        logits_dtype = logits.dtype
        logits = logits.to("cpu").float()
        xgr.apply_token_bitmask_inplace(
            logits,
            grammar_bitmask,
            indices=out_indices,
        )
        return logits.to(self.device).to(logits_dtype)
    
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:
        with ProfileExecuteDuration().capture_async("prepare input"):
            self._update_states(scheduler_output)
            if not scheduler_output.total_num_scheduled_tokens:
                if not has_kv_transfer_group():
                    logger.debug(
                        "skip this step for we receive the data from remote disaggregate prefill node"
                    )
                    # Return empty ModelRunnerOuptut if there's no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output)

            if self.dynamic_eplb:#None
                self.eplb_updator.forward_before()
                
            model_runner_output = None
            sampled_token_ids = None
            invalid_req_indices = None
            prefill_model_runner_output = None
            prefill_sampled_token_ids = None
            prefill_invalid_req_indices = None
                
            if self.input_batch.num_reqs > 0:
                (attn_metadata, positions, num_scheduled_tokens_np,
                num_input_tokens, num_tokens_across_dp, maybe_padded_num_tokens,
                logits_indices, spec_decode_metadata, input_ids, inputs_embeds,
                decode_intermediate_tensors,
                max_query_len) = (self._prepare_inputs(scheduler_output,
                                                        intermediate_tensors))
                
                with self.decode_stream:
                    # prof.step()
                    model_runner_output, sampled_token_ids, invalid_req_indices = self.execute_input_batch(scheduler_output,
                                                                                attn_metadata,
                                                                                positions,
                                                                                num_scheduled_tokens_np,
                                                                                num_input_tokens,
                                                                                num_tokens_across_dp,
                                                                                maybe_padded_num_tokens,
                                                                                logits_indices,
                                                                                spec_decode_metadata,
                                                                                max_query_len,
                                                                                input_ids=input_ids,
                                                                                inputs_embeds=inputs_embeds,
                                                                                decode_intermediate_tensors=decode_intermediate_tensors)
                
                
            if self.prefill_input_batch.num_reqs > 0 :
                (prefill_attn_metadata, prefill_positions, prefill_num_scheduled_tokens_np,
                prefill_num_input_tokens, prefill_num_tokens_across_dp, prefill_maybe_padded_num_tokens,
                prefill_logits_indices, prefill_spec_decode_metadata, prefill_input_ids, prefill_inputs_embeds,
                prefill_intermediate_tensors,
                prefill_max_query_len) = (self._prepare_prefill_inputs(scheduler_output,
                                                        intermediate_tensors))    
                
                with self.prefill_stream:
                    # prof.step()
                    prefill_model_runner_output, prefill_sampled_token_ids, prefill_invalid_req_indices = self.execute_prefill_input_batch(scheduler_output,
                                                                                                        prefill_attn_metadata,
                                                                                                        prefill_positions,
                                                                                                        prefill_num_scheduled_tokens_np,
                                                                                                        prefill_num_input_tokens,
                                                                                                        prefill_num_tokens_across_dp,
                                                                                                        prefill_maybe_padded_num_tokens,
                                                                                                        prefill_logits_indices,
                                                                                                        prefill_spec_decode_metadata,
                                                                                                        prefill_max_query_len,
                                                                                                        prefill_input_ids=prefill_input_ids,
                                                                                                        prefill_inputs_embeds=prefill_inputs_embeds,
                                                                                                        prefill_intermediate_tensors=prefill_intermediate_tensors)
                prefill_async_modelrunner_output = AsyncNPUModelRunnerOutput(
                    model_runner_output=None,
                    prefill_model_runner_output=prefill_model_runner_output,
                    sampled_token_ids=None,
                    prefill_sampled_token_ids=prefill_sampled_token_ids,
                    invalid_req_indices=None,
                    prefill_invalid_req_indices=prefill_invalid_req_indices,
                    async_output_copy_stream=None,
                    prefill_async_output_copy_stream=self.prefill_async_output_copy_stream,
                    decode_stream=None,
                    prefill_stream=self.prefill_stream,
                )
                
                self.async_model_runner_output.append(prefill_async_modelrunner_output)
                logger.debug(f"add {prefill_model_runner_output.req_ids} to async_model_runner_output")
                
            if self.dynamic_eplb:
                self.eplb_updator.take_update_info_from_eplb_process()
            
        is_finished_prefill_model_runner_output = None
        prefill_output = None
    
        if self.async_model_runner_output:
            output = self.async_model_runner_output[0]
            if model_runner_output is not None:
                device = "cpu"
                group = get_tp_group().cpu_group
                local_finished_query = output._prefill_async_copy_ready_event.query()
                local_finished_query_tensor = torch.tensor(local_finished_query, device=device, dtype=torch.bool)
                logger.debug(f"local_finished_query_tensor : {local_finished_query_tensor}")
                dist.all_reduce(local_finished_query_tensor, op=dist.ReduceOp.MIN, group=group)
                is_global_finished = bool(local_finished_query_tensor.item())
                
                logger.info(f"is_global_finished : {is_global_finished}")
                if is_global_finished:
                    prefill_output = output.get_prefill_output()
                    is_finished_prefill_model_runner_output = output
            else:
                prefill_output = output.get_prefill_output()
                is_finished_prefill_model_runner_output = output
            
        if is_finished_prefill_model_runner_output is not None:
            self.async_model_runner_output.remove(is_finished_prefill_model_runner_output)
            for req in is_finished_prefill_model_runner_output._prefill_model_runner_output.req_ids:
                if req not in self.finished_prefill_reqs:
                    self.finished_prefill_reqs.add(req)
        self.default_stream.wait_stream(self.decode_stream)
        final_model_runner_output = AsyncNPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            prefill_model_runner_output=prefill_output,
            sampled_token_ids=sampled_token_ids,
            prefill_sampled_token_ids=None,
            invalid_req_indices=invalid_req_indices,
            prefill_invalid_req_indices=None,
            async_output_copy_stream=self.async_output_copy_stream,
            prefill_async_output_copy_stream=self.prefill_async_output_copy_stream,
            decode_stream=self.decode_stream,
            prefill_stream=self.prefill_stream,
        )
        
        return final_model_runner_output
        
        
    def execute_input_batch(
        self,
        scheduler_output: "SchedulerOutput",
        attn_metadata: dict[str, Any],
        positions: torch.Tensor,
        num_scheduled_tokens_np: np.ndarray,
        num_input_tokens: int,
        num_tokens_across_dp: torch.Tensor,
        maybe_padded_num_tokens: int,
        logits_indices: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata,
        max_query_len: int,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decode_intermediate_tensors: Optional[torch.Tensor] = None, 
    ) -> tuple[ModelRunnerOutput, torch.Tensor, list[int]]:
        if self.input_batch.num_reqs > 0:
            moe_comm_type = self._select_moe_comm_method(num_input_tokens,
                                                        self.with_prefill)

            uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
                (scheduler_output.total_num_scheduled_tokens - scheduler_output.prefill_total_num_scheduled_tokens)
                == self.input_batch.num_reqs * max_query_len)
            batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                            uniform_decode=uniform_decode)
            aclgraph_runtime_mode, batch_descriptor = \
                self.aclgraph_dispatcher.dispatch(batch_descriptor)
        
        #run forward pass
        if self.input_batch.num_reqs > 0:
            with ProfileExecuteDuration().capture_async("forward"):
                with set_ascend_forward_context(
                        attn_metadata,
                        self.vllm_config,
                        num_tokens=num_input_tokens,
                        num_tokens_across_dp=num_tokens_across_dp,
                        with_prefill=self.with_prefill,
                        reserved_mc2_mask=self.reserved_mc2_mask,
                        moe_comm_type=moe_comm_type,
                        aclgraph_runtime_mode=aclgraph_runtime_mode,
                        batch_descriptor=batch_descriptor,
                        num_actual_tokens=scheduler_output.total_num_scheduled_tokens - scheduler_output.prefill_total_num_scheduled_tokens,
                        prefetch_stream=self.prefetch_stream,
                        model_instance=self.model,
                        weight_prefetch_method=self.weight_prefetch_method,
                        use_offload_tp = False):
                    self.maybe_setup_kv_connector(scheduler_output)
                    hidden_states = self._generate_process_reqs_hidden_states(
                        attn_metadata, self.with_prefill, maybe_padded_num_tokens,
                        input_ids, positions, decode_intermediate_tensors, inputs_embeds)

                self.maybe_wait_for_kv_save()
                finished_sending, finished_recving = self.get_finished_kv_transfer(
                    scheduler_output)
                logger.debug(f"finished_sending: {finished_sending}, finished_recving: {finished_recving}")

                aux_hidden_states = None
                if self.drafter and self.drafter.name == SpecDcodeType.EAGLE3:
                    hidden_states, aux_hidden_states = hidden_states
        
        if self.input_batch.num_reqs > 0:
            kv_connector_output = KVConnectorOutput(
                finished_sending=finished_sending,
                finished_recving=finished_recving)
            finished_sending = None
            finished_recving = None
            
        sampled_token_ids = None
        invalid_req_indices = None
        if self.input_batch.num_reqs > 0:
            with ProfileExecuteDuration().capture_async("post process"):
                # Broadcast PP output for external_launcher (torchrun)
                # to make sure we are synced across pp ranks
                # TODO: Support overlapping mirco-batches
                # https://github.com/vllm-project/vllm/issues/18019
                #False
                broadcast_pp_output = \
                    self.parallel_config.distributed_executor_backend \
                    == "external_launcher" and len(get_pp_group().ranks) > 0
                if not get_pp_group().is_last_rank:
                    # For mid-pipeline stages, return the hidden states.
                    if not broadcast_pp_output:
                        hidden_states.kv_connector_output = kv_connector_output
                        return hidden_states
                    assert isinstance(hidden_states, IntermediateTensors)
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors, all_gather_group=get_tp_group())
                    logits = None
                else:
                    if self.input_batch.pooling_params:
                        return self._pool(
                            hidden_states,
                            scheduler_output.total_num_scheduled_tokens - scheduler_output.prefill_total_num_scheduled_tokens,
                            num_scheduled_tokens_np, finished_sending,
                            finished_recving, kv_connector_output)
                    sample_hidden_states = hidden_states[logits_indices] #
                    logits = self.model.compute_logits(sample_hidden_states)
                if broadcast_pp_output:
                    model_output_broadcast_data = {
                        "logits": logits.contiguous(),
                    } if logits is not None else {}
                    model_output_broadcast_data = get_pp_group(
                    ).broadcast_tensor_dict(model_output_broadcast_data,
                                            src=len(get_pp_group().ranks) - 1)
                    assert model_output_broadcast_data is not None
                    logits = model_output_broadcast_data["logits"]

                # Apply structured output bitmasks if present
                if scheduler_output.grammar_bitmask is not None:#is None
                    logits = self.apply_grammar_bitmask(scheduler_output, logits)

                # Sample the next token and get logprobs if needed.
                sampling_metadata = self.input_batch.sampling_metadata
                if spec_decode_metadata is None: #here
                    if lmhead_tp_enable() and logits is not None:#no here
                        logits = logits[:self.input_batch.num_reqs]
                    sampler_output = self.sampler(
                        logits=logits,
                        sampling_metadata=sampling_metadata,
                    )
                else:
                    if lmhead_tp_enable() and logits is not None:
                        logits = logits[:len(spec_decode_metadata.logits_indices)]
                    # When indexing with a tensor (bonus_logits_indices), PyTorch
                    # creates a new tensor with separate storage from the original
                    # logits tensor. This means any in-place operations on bonus_logits
                    # won't affect the original logits tensor.
                    assert logits is not None
                    bonus_logits = logits[
                        spec_decode_metadata.bonus_logits_indices]
                    sampler_output = self.sampler(
                        logits=bonus_logits,
                        sampling_metadata=sampling_metadata,
                    )
                    bonus_token_ids = sampler_output.sampled_token_ids

                    # Just like `bonus_logits`, `target_logits` is a new tensor with
                    # separate storage from the original `logits` tensor. Therefore,
                    # it is safe to update `target_logits` in place.
                    target_logits = logits[
                        spec_decode_metadata.target_logits_indices]
                    output_token_ids = self.rejection_sampler(
                        spec_decode_metadata,
                        None,  # draft_probs
                        target_logits,
                        bonus_token_ids,
                        sampling_metadata,
                    )
                    sampler_output.sampled_token_ids = output_token_ids
                    logger.info(f"need_accepted_tokens: {self.need_accepted_tokens}")
                    if self.need_accepted_tokens:
                        self._update_states_after_model_execute(output_token_ids)

                discard_sampled_tokens_req_indices: list[int] = []
                # TODO(woosuk): The following loop can be slow since it iterates over
                # the requests one by one. Optimize.
                discard_sampled_tokens_req_indices = []
                for i, req_id in enumerate(self.input_batch.req_ids):
                    req_state = self.requests[req_id]
                    seq_len = (req_state.num_computed_tokens +
                            scheduler_output.num_scheduled_tokens[req_id])
                    if seq_len < req_state.num_tokens:
                        # Ignore the sampled token.
                        # Rewind the generator state as if the token was not sampled.
                        generator = self.input_batch.generators.get(i)
                        if generator is not None:
                            generator.set_offset(generator.get_offset() - 4)
                        discard_sampled_tokens_req_indices.append(i)

                # Copy some objects so they don't get modified after returning.
                # This is important when using async scheduling.
                req_ids_output_copy = self.input_batch.req_ids.copy()
                req_id_to_index_output_copy = \
                    self.input_batch.req_id_to_index.copy()

                # NOTE: NPU -> CPU Sync happens here.
                # Move as many CPU operations as possible before this sync point.
                logprobs_tensors = sampler_output.logprobs_tensors
                logprobs_lists = logprobs_tensors.tolists() \
                    if logprobs_tensors is not None else None

                # Compute prompt logprobs if needed.
                prompt_logprobs_dict = self._get_prompt_logprobs_dict(
                    hidden_states[:scheduler_output.total_num_scheduled_tokens - scheduler_output.prefill_total_num_scheduled_tokens],
                    scheduler_output,
                )

                num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
                sampled_token_ids = sampler_output.sampled_token_ids
                if not self.use_async_scheduling:
                    # Get the valid generated tokens.
                    max_gen_len = sampled_token_ids.shape[-1]
                    if max_gen_len == 1:
                        # No spec decode tokens.
                        valid_sampled_token_ids = sampled_token_ids.tolist()
                    else:
                        # Includes spec decode tokens.
                        valid_sampled_token_ids = self.rejection_sampler.parse_output(
                            sampled_token_ids,
                            self.input_batch.vocab_size,
                        )
                    # Mask out the sampled tokens that should not be sampled.
                    for i in discard_sampled_tokens_req_indices:
                        valid_sampled_token_ids[i].clear()
                else:
                    valid_sampled_token_ids = []
                    invalid_req_indices = list(discard_sampled_tokens_req_indices)
                    invalid_req_indices_set = set(invalid_req_indices)
                    assert sampled_token_ids.shape[-1] == 1

                    # Cache the sampled tokens on the NPU and avoid CPU sync.
                    # These will be copied into input_ids in the next step
                    # when preparing inputs.
                    self.input_batch.prev_sampled_token_ids = \
                        sampled_token_ids
                    self.input_batch.prev_sampled_token_ids_invalid_indices = \
                        invalid_req_indices_set
                    self.input_batch.prev_req_id_to_index = {
                        req_id: i
                        for i, req_id in enumerate(self.input_batch.req_ids)
                        if i not in invalid_req_indices_set
                    }
                # Cache the sampled tokens in the model runner, so that the scheduler
                # doesn't need to send them back.
                # NOTE(woosuk): As an exception, when using PP, the scheduler sends
                # the sampled tokens back, because there's no direct communication
                # between the first-stage worker and the last-stage worker.
                for req_idx in range(num_sampled_tokens):
                    if self.use_async_scheduling:
                        sampled_ids = [-1] * 1 if \
                            req_idx not in invalid_req_indices_set else None
                    else:
                        sampled_ids = valid_sampled_token_ids[req_idx]
                    if not sampled_ids:
                        continue

                    start_idx = self.input_batch.num_tokens_no_spec[req_idx]
                    end_idx = start_idx + len(sampled_ids)
                    assert end_idx <= self.model_config.max_model_len, (
                        "Sampled token IDs exceed the max model length. "
                        f"Total number of tokens: {end_idx} > max_model_len: "
                        f"{self.model_config.max_model_len}")

                    self.input_batch.token_ids_cpu[req_idx,
                                                start_idx:end_idx] = sampled_ids
                    self.input_batch.num_tokens_no_spec[req_idx] = end_idx
                    self.input_batch.num_tokens[req_idx] = end_idx
                    req_id = self.input_batch.req_ids[req_idx]
                    req_state = self.requests[req_id]
                    req_state.output_token_ids.extend(sampled_ids)

                # logger.info(f"speculative_config: {self.speculative_config}") #None
                if self.speculative_config:
                    self._draft_token_ids = self.propose_draft_token_ids(
                        valid_sampled_token_ids,
                        sampling_metadata,
                        scheduler_output,
                        spec_decode_metadata,
                        positions,
                        scheduler_output.total_num_scheduled_tokens - scheduler_output.prefill_total_num_scheduled_tokens,
                        hidden_states,
                        attn_metadata,
                        aux_hidden_states,
                    )

                if has_kv_transfer_group():
                    get_kv_transfer_group().clear_connector_metadata()
                extra_args = ({"kv_connector_output": kv_connector_output})
                
        model_runner_output = None
        if self.input_batch.num_reqs > 0:
            model_runner_output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                pooler_output=[],
                **extra_args,
            )
            
        durations = ProfileExecuteDuration().pop_captured_sync()
        if durations:
            dr_str = [
                f"[{tag}]:{duration:.2f}ms"
                for tag, duration in durations.items()
            ]
            captured_name = "Decode" if self.attn_state == AscendAttentionState.DecodeOnly else "Prefill"
            logger.info("Profile execute duration [%s]:%s", captured_name,
                        " ".join(dr_str))
        if self.dynamic_eplb:
            self.eplb_updator.forward_end()
            
        return model_runner_output, sampled_token_ids, invalid_req_indices
        

    def execute_prefill_input_batch(
        self,
        scheduler_output: "SchedulerOutput",
        prefill_attn_metadata: dict[str, Any],
        prefill_positions: torch.Tensor,
        prefill_num_scheduled_tokens_np: np.ndarray,
        prefill_num_input_tokens: int,
        prefill_num_tokens_across_dp: torch.Tensor,
        prefill_maybe_padded_num_tokens: int,
        prefill_logits_indices: torch.Tensor,
        prefill_spec_decode_metadata: SpecDecodeMetadata,
        prefill_max_query_len: int,
        prefill_input_ids: Optional[torch.Tensor] = None,
        prefill_inputs_embeds: Optional[torch.Tensor] = None,
        prefill_intermediate_tensors: Optional[torch.Tensor] = None,
    ) -> tuple[ModelRunnerOutput, torch.Tensor, list[int]]:
        if self.prefill_input_batch.num_reqs > 0 :
            prefill_moe_comm_type = self._select_moe_comm_method(
                prefill_num_input_tokens, self.with_prefill)
            prefill_uniform_decode = (prefill_max_query_len == self.uniform_decode_query_len) and (
                scheduler_output.prefill_total_num_scheduled_tokens
                == self.prefill_input_batch.num_reqs * prefill_max_query_len)
            prefill_batch_descriptor = BatchDescriptor(num_tokens=prefill_num_input_tokens,
                                                    uniform_decode=prefill_uniform_decode)
            prefill_aclgraph_runtime_mode, prefill_batch_descriptor = \
                self.aclgraph_dispatcher.dispatch(prefill_batch_descriptor)
                
        #run forward pass
        if self.prefill_input_batch.num_reqs > 0 :
            with ProfileExecuteDuration().capture_async("forward"):
                with set_ascend_forward_context(
                        prefill_attn_metadata,
                        self.vllm_config,
                        num_tokens=prefill_num_input_tokens,
                        num_tokens_across_dp=prefill_num_tokens_across_dp,
                        with_prefill=self.with_prefill,
                        reserved_mc2_mask=self.reserved_mc2_mask,
                        moe_comm_type=prefill_moe_comm_type,
                        aclgraph_runtime_mode=prefill_aclgraph_runtime_mode,
                        batch_descriptor=prefill_batch_descriptor,
                        num_actual_tokens=scheduler_output.prefill_total_num_scheduled_tokens,
                        prefetch_stream=self.prefetch_stream,
                        model_instance=self.model,
                        weight_prefetch_method=self.weight_prefetch_method,
                        use_offload_tp = True):
                    prefill_hidden_states = self._generate_process_reqs_hidden_states(
                        prefill_attn_metadata, self.with_prefill, prefill_maybe_padded_num_tokens,
                        prefill_input_ids, prefill_positions, prefill_intermediate_tensors, prefill_inputs_embeds)

            self.maybe_wait_for_kv_save()

            prefill_aux_hidden_states = None
            if self.drafter and self.drafter.name == SpecDcodeType.EAGLE3:
                prefill_hidden_states, prefill_aux_hidden_states = prefill_hidden_states
                
        prefill_finished_sending = None
        prefill_finished_recving = None
        prefill_kv_connector_output = None
            
        prefill_sampled_token_ids = None
        prefill_invalid_req_indices = None
        if self.prefill_input_batch.num_reqs > 0 :
            with ProfileExecuteDuration().capture_async("preprocess"):
                broadcast_pp_output = \
                    self.parallel_config.distributed_executor_backend \
                    == "external_launcher" and len(get_pp_group().ranks) > 0
                if not get_pp_group().is_first_rank:
                    if not broadcast_pp_output:
                        prefill_hidden_states.kv_connector_output = prefill_kv_connector_output
                        return prefill_hidden_states
                    assert isinstance(prefill_hidden_states, IntermediateTensors)
                    get_pp_group().send_tensor_dict(
                        prefill_hidden_states.tensors, all_gather_group=get_tp_group())
                    prefill_logits = None
                else:
                    if self.prefill_input_batch.pooling_params:
                        return self._pool(
                            prefill_hidden_states,
                            scheduler_output.prefill_total_num_scheduled_tokens,
                            prefill_num_scheduled_tokens_np, prefill_finished_sending,
                            prefill_finished_recving, prefill_kv_connector_output)
                    prefill_sample_hidden_states = prefill_hidden_states[prefill_logits_indices]
                    prefill_logits = self.model.compute_logits(prefill_sample_hidden_states, True)
                logger.debug(f"broadcast_pp_output: {broadcast_pp_output}")
                if broadcast_pp_output:
                    model_output_broadcast_data = {
                        "logits": prefill_logits.contiguous(),
                    } if prefill_logits is not None else {}
                    model_output_broadcast_data = get_pp_group(
                    ).broadcast_tensor_dict(model_output_broadcast_data,
                                            src=len(get_pp_group().ranks) - 1)
                    assert model_output_broadcast_data is not None
                    prefill_logits = model_output_broadcast_data["logits"]
                
                if scheduler_output.prefill_grammar_bitmask is not None:
                    prefill_logits = self.prefill_apply_grammar_bitmask(scheduler_output, prefill_logits)

                prefill_sampling_metadata = self.prefill_input_batch.sampling_metadata
                if prefill_spec_decode_metadata is None:
                    if lmhead_tp_enable() and prefill_logits is not None:
                        prefill_logits = prefill_logits[:self.prefill_input_batch.num_reqs]
                    prefill_sampler_output = self.sampler(
                        logits=prefill_logits,
                        sampling_metadata=prefill_sampling_metadata,
                    )
                else:
                    if lmhead_tp_enable() and prefill_logits is not None:
                        prefill_logits = prefill_logits[:len(prefill_spec_decode_metadata.prefill_logits_indices)]
                    assert prefill_logits is not None
                    prefill_bonus_logits = prefill_logits[
                        prefill_spec_decode_metadata.bonus_prefill_logits_indices]
                    prefill_sampler_output = self.sampler(
                        logits=prefill_bonus_logits,
                        sampling_metadata=prefill_sampling_metadata,
                    )
                    prefill_bonus_token_ids = prefill_sampler_output.sampled_token_ids

                    prefill_target_logits = prefill_logits[
                        prefill_spec_decode_metadata.target_prefill_logits_indices]
                    prefill_output_token_ids = self.rejection_sampler(
                        prefill_spec_decode_metadata,
                        None,  # draft_probs
                        prefill_target_logits,
                        prefill_bonus_token_ids,
                        prefill_sampling_metadata,
                    )
                    prefill_sampler_output.sampled_token_ids = prefill_output_token_ids
                    if self.need_accepted_tokens:
                        self._prefill_update_states_after_model_execute(prefill_output_token_ids)

                prefill_discard_sampled_tokens_req_indices: list[int] = []
                prefill_discard_sampled_tokens_req_indices = []
                for i, req_id in enumerate(self.prefill_input_batch.req_ids):
                    prefill_req_state = self.requests[req_id]
                    seq_len = (prefill_req_state.num_computed_tokens +
                            scheduler_output.prefill_num_scheduled_tokens[req_id])
                    if seq_len < prefill_req_state.num_tokens:
                        generator = self.prefill_input_batch.generators.get(i)
                        if generator is not None:
                            generator.set_offset(generator.get_offset() - 4)
                        prefill_discard_sampled_tokens_req_indices.append(i)

                prefill_req_ids_output_copy = self.prefill_input_batch.req_ids.copy()
                prefill_req_id_to_index_output_copy = self.prefill_input_batch.req_id_to_index.copy()

                prefill_logprobs_tensors = prefill_sampler_output.logprobs_tensors
                prefill_logprobs_lists = prefill_logprobs_tensors.tolists() if prefill_logprobs_tensors is not None else None

                prefill_prompt_logprobs_dict = self._prefill_get_prompt_logprobs_dict(
                    prefill_hidden_states[:scheduler_output.prefill_total_num_scheduled_tokens],
                    scheduler_output,
                )

                prefill_num_sampled_tokens = prefill_sampler_output.sampled_token_ids.shape[0]
                prefill_sampled_token_ids = prefill_sampler_output.sampled_token_ids
                if not self.use_async_scheduling:
                    prefill_max_gen_len = prefill_sampled_token_ids.shape[-1]
                    if prefill_max_gen_len == 1:
                        prefill_valid_sampled_token_ids = prefill_sampled_token_ids.tolist()
                    else:
                        prefill_valid_sampled_token_ids = self.rejection_sampler.parse_output(
                            prefill_sampled_token_ids,
                            self.prefill_input_batch.vocab_size,
                        )
                    for i in prefill_discard_sampled_tokens_req_indices:
                        prefill_valid_sampled_token_ids[i].clear()
                else:
                    prefill_valid_sampled_token_ids = []
                    prefill_invalid_req_indices = list(prefill_discard_sampled_tokens_req_indices)
                    prefill_invalid_req_indices_set = set(prefill_invalid_req_indices)
                    assert prefill_sampled_token_ids.shape[-1] == 1

                    self.prefill_input_batch.prev_sampled_token_ids = prefill_sampled_token_ids
                    self.prefill_input_batch.prev_sampled_token_ids_invalid_indices = prefill_invalid_req_indices_set
                    self.prefill_input_batch.prev_req_id_to_index = {
                        req_id: i
                        for i, req_id in enumerate(self.prefill_input_batch.req_ids)
                        if i not in prefill_invalid_req_indices_set
                    }

                for req_idx in range(prefill_num_sampled_tokens):
                    if self.use_async_scheduling:
                        sampled_ids = [-1] * 1 if req_idx not in prefill_invalid_req_indices_set else None
                    else:
                        sampled_ids = prefill_valid_sampled_token_ids[req_idx]
                    if not sampled_ids:
                        continue

                    start_idx = self.prefill_input_batch.num_tokens_no_spec[req_idx]
                    end_idx = start_idx + len(sampled_ids)
                    assert end_idx <= self.model_config.max_model_len, (
                        "Sampled token IDs exceed the max model length. "
                        f"Total number of tokens: {end_idx} > max_model_len: "
                        f"{self.model_config.max_model_len}")
                    
                    self.prefill_input_batch.token_ids_cpu[req_idx,
                                                start_idx:end_idx] = sampled_ids
                    self.prefill_input_batch.num_tokens_no_spec[req_idx] = end_idx
                    self.prefill_input_batch.num_tokens[req_idx] = end_idx
                    req_id = self.prefill_input_batch.req_ids[req_idx]
                    prefill_req_state = self.requests[req_id]
                    prefill_req_state.output_token_ids.extend(sampled_ids)

                if self.speculative_config:
                    self._prefill_draft_token_ids = self.propose_draft_token_ids(
                        prefill_valid_sampled_token_ids,
                        prefill_sampling_metadata,
                        scheduler_output,
                        prefill_spec_decode_metadata,
                        prefill_positions,
                        scheduler_output.prefill_total_num_scheduled_tokens,
                        prefill_hidden_states,
                        prefill_attn_metadata,
                        prefill_aux_hidden_states,
                    )
                if has_kv_transfer_group():
                    get_kv_transfer_group().clear_connector_metadata()
                    
                prefill_extra_args = ({"kv_connector_output": prefill_kv_connector_output})
                
        prefill_model_runner_output = None
        if self.prefill_input_batch.num_reqs > 0 :
            prefill_model_runner_output = ModelRunnerOutput(
                req_ids=prefill_req_ids_output_copy,
                req_id_to_index=prefill_req_id_to_index_output_copy,
                sampled_token_ids=prefill_valid_sampled_token_ids,
                logprobs=prefill_logprobs_lists,
                prompt_logprobs_dict=prefill_prompt_logprobs_dict,
                pooler_output=[],
                **prefill_extra_args,
            )
            
        durations = ProfileExecuteDuration().pop_captured_sync()
        if durations:
            dr_str = [
                f"[{tag}]:{duration:.2f}ms"
                for tag, duration in durations.items()
            ]
            captured_name = "Decode" if self.attn_state == AscendAttentionState.DecodeOnly else "Prefill"
            logger.info("Profile execute duration [%s]:%s", captured_name,
                        " ".join(dr_str))
        if self.dynamic_eplb:
            self.eplb_updator.forward_end()
            
        return prefill_model_runner_output, prefill_sampled_token_ids, prefill_invalid_req_indices

    def profile_npu(self, is_start: bool = True) -> None:
        if is_start:
            prof.stop()
    
    @staticmethod
    def get_finished_kv_transfer(
        scheduler_output: "SchedulerOutput",
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if has_kv_transfer_group():
            return get_kv_transfer_group().get_finished(
                scheduler_output.finished_req_ids-scheduler_output.prefill_finished_req_ids)
        return None, None
    
    @staticmethod
    def prefill_get_finished_kv_transfer(
        scheduler_output: "SchedulerOutput",
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if has_kv_transfer_group():
            return get_kv_transfer_group().get_finished(
                scheduler_output.prefill_finished_req_ids)
        return None, None
    
    def _prefill_get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        scheduler_output: "SchedulerOutput",
    ) -> dict[str, Optional[LogprobsTensors]]:
        num_prompt_logprobs_dict = self.prefill_input_batch.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            logger.debug(f"_prefill_get_prompt_logprobs_dict: return None")
            return {}

        in_progress_dict = self.prefill_input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():

            num_tokens = scheduler_output.num_scheduled_tokens[req_id]

            # Get metadata for this request.
            request = self.requests[req_id]
            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True)

            # Set up target LogprobsTensors object.
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1)
                in_progress_dict[req_id] = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.prefill_input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc_np[req_idx].item()
            prompt_hidden_states = hidden_states[offset:offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok:start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids)

            # Transfer NPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True)
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs,
                                                         non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True)

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        # Must synchronize the non-blocking NPU->CPU transfers.
        if prompt_logprobs_dict:
            torch.npu.synchronize()

        return prompt_logprobs_dict