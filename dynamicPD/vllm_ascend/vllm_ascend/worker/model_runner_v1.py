from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Iterable, List

import numpy as np
import torch
import torch.distributed as dist
import time

from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import get_tp_group
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    ECConnectorOutput,
    KVConnectorOutput,
    LogprobsLists,
    ModelRunnerOutput,
)
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

from vllm_ascend.worker.npu_input_batch import NPUInputBatch
import vllm_ascend.worker.model_runner_v1 as ascend_model_runner
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner, logger

from dynamicPD.patching import dynamicPDPatch
from dynamicPD.vllm_ascend.vllm_ascend.ascend_forward_context import (
    set_ascend_forward_context as dynamic_pd_forward_context,
)


_USE_OFFLOAD_TP_CONTEXT: ContextVar[bool] = ContextVar(
    "dynamic_pd_use_offload_tp", default=False)
_DECODE_PENDING_PREFILL_MODE: ContextVar[str | None] = ContextVar(
    "dynamic_pd_decode_pending_prefill_mode", default=None)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _prefill_decode_fence_mode() -> str:
    if _env_flag("DYNAMIC_PD_SYNC_PREFILL_BEFORE_DECODE"):
        return "copy"
    return os.getenv("DYNAMIC_PD_PREFILL_DECODE_FENCE", "none").lower()


def _decode_pending_prefill_mode() -> str:
    return os.getenv("DYNAMIC_PD_DECODE_PENDING_PREFILL_MODE",
                     "acl_none").lower()


def _patch_forward_context_once() -> None:
    if getattr(ascend_model_runner, "_dynamic_pd_forward_context_patched", False):
        return

    def _set_ascend_forward_context(*args: Any, **kwargs: Any):
        decode_pending_mode = _DECODE_PENDING_PREFILL_MODE.get()
        if decode_pending_mode in {"acl_none", "skip_compiled"}:
            # kwargs["aclgraph_runtime_mode"] = CUDAGraphMode.NONE
            # kwargs["batch_descriptor"] = None
            pass
        if decode_pending_mode == "skip_compiled":
            kwargs["skip_compiled"] = True

        if _USE_OFFLOAD_TP_CONTEXT.get(False):
            kwargs.setdefault("use_offload_tp", True)
            # The async prefill lane owns separate buffers and runs on a
            # different stream. Keep it out of decode-lane ACL graph replay;
            # the compiled model path itself is still useful for large
            # prefill throughput, so do not force skip_compiled here.
            kwargs["aclgraph_runtime_mode"] = CUDAGraphMode.NONE
            kwargs["batch_descriptor"] = None
            kwargs["skip_compiled"] = True
        return dynamic_pd_forward_context(*args, **kwargs)

    ascend_model_runner.set_ascend_forward_context = _set_ascend_forward_context
    ascend_model_runner._dynamic_pd_forward_context_patched = True

def _is_empty_output(output: ModelRunnerOutput | None) -> bool:
    return output is None or len(output.req_ids) == 0


def _materialize_output(
    output: ModelRunnerOutput | AsyncModelRunnerOutput | None,
) -> ModelRunnerOutput | None:
    if output is None:
        return None
    if isinstance(output, AsyncModelRunnerOutput):
        return output.get_output()
    return output


def _async_output_ready(output: ModelRunnerOutput | AsyncModelRunnerOutput) -> bool:
    local_ready = True
    if isinstance(output, AsyncGPUModelRunnerOutput):
        event = getattr(output, "async_copy_ready_event", None)
        local_ready = event is None or bool(event.query())
    if isinstance(output, AsyncModelRunnerOutput):
        event = getattr(output, "async_copy_ready_event", None)
        local_ready = event is None or bool(event.query())
    local_ready = torch.tensor(int(local_ready), device="npu")
    dist.all_reduce(local_ready, op=dist.ReduceOp.MIN, group=get_tp_group().device_group)
    global_ready = local_ready.item() == 1
    return global_ready


def _merge_logprobs(items: list[LogprobsLists | None]) -> LogprobsLists | None:
    non_empty = [item for item in items if item is not None]
    if not non_empty:
        return None
    if len(non_empty) == 1:
        return non_empty[0]

    logprob_token_ids = np.concatenate(
        [item.logprob_token_ids for item in non_empty], axis=0)
    logprobs = np.concatenate([item.logprobs for item in non_empty], axis=0)
    sampled_token_ranks = np.concatenate(
        [item.sampled_token_ranks for item in non_empty], axis=0)

    cu_num_generated_tokens: list[int] | None = None
    if any(item.cu_num_generated_tokens is not None for item in non_empty):
        cu_num_generated_tokens = []
        offset = 0
        for item in non_empty:
            if item.cu_num_generated_tokens is None:
                cu_num_generated_tokens.extend(
                    range(offset, offset + len(item.sampled_token_ranks)))
            else:
                cu_num_generated_tokens.extend(
                    offset + value for value in item.cu_num_generated_tokens)
            offset += len(item.sampled_token_ranks)

    return LogprobsLists(
        logprob_token_ids=logprob_token_ids,
        logprobs=logprobs,
        sampled_token_ranks=sampled_token_ranks,
        cu_num_generated_tokens=cu_num_generated_tokens,
    )


def _merge_kv_outputs(
    outputs: Iterable[KVConnectorOutput | None],
) -> KVConnectorOutput | None:
    non_empty = [output for output in outputs if output is not None]
    if not non_empty:
        return None
    if len(non_empty) == 1:
        return non_empty[0]
    try:
        return KVConnectorOutput.merge(*non_empty)
    except AssertionError:
        logger.debug("Falling back to conservative KVConnectorOutput merge")
        merged = KVConnectorOutput()
        merged.finished_sending = set().union(
            *(out.finished_sending or set() for out in non_empty))
        merged.finished_recving = set().union(
            *(out.finished_recving or set() for out in non_empty))
        merged.invalid_block_ids = set().union(
            *(out.invalid_block_ids or set() for out in non_empty))
        stats = [out.kv_connector_stats for out in non_empty
                 if out.kv_connector_stats is not None]
        if stats:
            merged.kv_connector_stats = stats[0]
            for item in stats[1:]:
                merged.kv_connector_stats = merged.kv_connector_stats.aggregate(
                    item)
        return merged


def _merge_ec_outputs(
    outputs: Iterable[ECConnectorOutput | None],
) -> ECConnectorOutput | None:
    non_empty = [output for output in outputs if output is not None]
    if not non_empty:
        return None
    if len(non_empty) == 1:
        return non_empty[0]
    return ECConnectorOutput(
        finished_sending=set().union(
            *(out.finished_sending or set() for out in non_empty)),
        finished_recving=set().union(
            *(out.finished_recving or set() for out in non_empty)),
    )


def _merge_outputs(
    decode_output: ModelRunnerOutput | None,
    prefill_outputs: list[ModelRunnerOutput],
) -> ModelRunnerOutput:
    outputs = [output for output in [decode_output, *prefill_outputs]
               if not _is_empty_output(output)]
    prefill_output_ids = {id(output) for output in prefill_outputs}
    if not outputs:
        return ModelRunnerOutput(req_ids=[], req_id_to_index={})
    if len(outputs) == 1:
        output = outputs[0]
        if id(output) in prefill_output_ids:
            output.finished_prefill_reqs = set(output.req_ids)
        return output

    req_ids: list[str] = []
    sampled_token_ids: list[list[int]] = []
    pooler_output: list[Any] = []
    prompt_logprobs_dict: dict[str, Any] = {}
    num_nans_in_logits: dict[str, int] = {}

    finished_prefill_reqs: set[str] = set()
    for output in outputs:
        req_ids.extend(output.req_ids)
        sampled_token_ids.extend(output.sampled_token_ids or [])
        if output.pooler_output:
            pooler_output.extend(output.pooler_output)
        prompt_logprobs_dict.update(output.prompt_logprobs_dict or {})
        if output.num_nans_in_logits:
            num_nans_in_logits.update(output.num_nans_in_logits)
        if id(output) in prefill_output_ids:
            finished_prefill_reqs.update(output.req_ids)

    merged = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: idx for idx, req_id in enumerate(req_ids)},
        sampled_token_ids=sampled_token_ids,
        logprobs=_merge_logprobs([output.logprobs for output in outputs]),
        prompt_logprobs_dict=prompt_logprobs_dict,
        pooler_output=pooler_output,
        kv_connector_output=_merge_kv_outputs(
            output.kv_connector_output for output in outputs),
        ec_connector_output=_merge_ec_outputs(
            output.ec_connector_output for output in outputs),
        num_nans_in_logits=num_nans_in_logits or None,
        cudagraph_stats=(decode_output.cudagraph_stats
                         if decode_output is not None else None),
    )
    merged.finished_prefill_reqs = finished_prefill_reqs
    merged.is_merged = True
    return merged


class AsyncMergedNPUModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        decode_output: ModelRunnerOutput | AsyncModelRunnerOutput | AsyncGPUModelRunnerOutput | None,
        prefill_outputs: list[ModelRunnerOutput | AsyncModelRunnerOutput | AsyncGPUModelRunnerOutput],
    ) -> None:
        self.decode_output = decode_output
        self.prefill_outputs = prefill_outputs

    def get_output(self) -> ModelRunnerOutput:
        decode_output = _materialize_output(self.decode_output)
        prefill_outputs = [
            output for output in
            (_materialize_output(output) for output in self.prefill_outputs)
            if output is not None
        ]
        for output in prefill_outputs:
            output.finished_prefill_reqs = set(output.req_ids)
        return _merge_outputs(decode_output, prefill_outputs)


@dataclass
class _LaneState:
    name: str
    input_batch: NPUInputBatch
    input_ids: Any
    positions: Any
    query_start_loc: Any
    seq_lens: Any
    async_output_copy_stream: Any = None
    prepare_inputs_event: torch.npu.Event | None = None
    execute_model_state: Any = None
    kv_connector_output: Any = None
    num_discarded_requests: int = 0
    query_lens: Any = None
    attn_state: Any = None
    with_prefill: bool | None = None
    cpu_slot_mapping: Any = None
    slot_mapping: Any = None
    draft_token_ids: Any = None
    work_stream: torch.npu.Stream | None = None


def _split_cached_request_data(
    cached: CachedRequestData,
    moved_req_ids: set[str],
) -> tuple[CachedRequestData, CachedRequestData]:
    if not cached.req_ids:
        empty = CachedRequestData.make_empty()
        return empty, empty

    kept_indices: list[int] = []
    moved_indices: list[int] = []
    for i, req_id in enumerate(cached.req_ids):
        (moved_indices if req_id in moved_req_ids else kept_indices).append(i)

    def build(indices: list[int]) -> CachedRequestData:
        req_ids = [cached.req_ids[i] for i in indices]
        req_id_set = set(req_ids)
        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=cached.resumed_req_ids & req_id_set,
            new_token_ids=[
                cached.new_token_ids[i] for i in indices
                if i < len(cached.new_token_ids)
            ],
            all_token_ids={
                req_id: token_ids
                for req_id, token_ids in cached.all_token_ids.items()
                if req_id in req_id_set
            },
            new_block_ids=[cached.new_block_ids[i] for i in indices],
            num_computed_tokens=[
                cached.num_computed_tokens[i] for i in indices
            ],
            num_output_tokens=[cached.num_output_tokens[i] for i in indices],
        )

    return build(kept_indices), build(moved_indices)


def _filter_mapping(
    mapping: dict[str, Any],
    excluded_req_ids: set[str],
) -> dict[str, Any]:
    return {
        req_id: value
        for req_id, value in mapping.items()
        if req_id not in excluded_req_ids
    }


def _copy_prefill_attrs(dst: SchedulerOutput, src: SchedulerOutput) -> None:
    for name in (
        "prefill_scheduled_new_reqs",
        "prefill_scheduled_cached_reqs",
        "prefill_num_scheduled_tokens",
        "prefill_total_num_scheduled_tokens",
        "prefill_scheduled_spec_decode_tokens",
        "prefill_scheduled_encoder_inputs",
        "prefill_num_common_prefix_blocks",
        "prefill_finished_req_ids",
        "prefill_structured_output_request_ids",
        "prefill_grammar_bitmask",
        "prefill_request_ids",
        "prefill_request_not_put",
    ):
        if hasattr(src, name):
            setattr(dst, name, getattr(src, name))


def _empty_prefill_attrs(output: SchedulerOutput) -> None:
    output.prefill_scheduled_new_reqs = []
    output.prefill_scheduled_cached_reqs = CachedRequestData.make_empty()
    output.prefill_num_scheduled_tokens = {}
    output.prefill_total_num_scheduled_tokens = 0
    output.prefill_scheduled_spec_decode_tokens = {}
    output.prefill_scheduled_encoder_inputs = {}
    output.prefill_num_common_prefix_blocks = []
    output.prefill_finished_req_ids = set()
    output.prefill_structured_output_request_ids = {}
    output.prefill_grammar_bitmask = None
    output.prefill_request_ids = set()
    output.prefill_request_not_put = set()


class NPUModelRunnerPatch(dynamicPDPatch[NPUModelRunner]):
    _original_init = NPUModelRunner.__init__
    _original_execute_model = NPUModelRunner.execute_model
    _original_sample_tokens = NPUModelRunner.sample_tokens
    _original_may_reinitialize_input_batch = (
        NPUModelRunner.may_reinitialize_input_batch)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        NPUModelRunnerPatch._original_init(self, *args, **kwargs)
        self._dynamic_pd_ready = False
        self._ensure_dynamic_pd_runtime()
        torch.npu.synchronize(self.device)
        self.default_stream = torch.npu.default_stream()

    def may_reinitialize_input_batch(self, *args: Any, **kwargs: Any) -> None:
        NPUModelRunnerPatch._original_may_reinitialize_input_batch(
            self, *args, **kwargs)
        if getattr(self, "_dynamic_pd_ready", False):
            self._dynamic_pd_decode_lane = self._make_dynamic_pd_lane(
                with_new_batch=False)
            self._dynamic_pd_prefill_lane = self._make_dynamic_pd_lane(
                with_new_batch=True)

    def _ensure_dynamic_pd_runtime(self) -> None:
        if getattr(self, "_dynamic_pd_ready", False):
            # logger.info("Dynamic prefill runtime already initialized")
            return
        _patch_forward_context_once()
        self._dynamic_pd_prefill_stream = torch.npu.Stream(device=self.device)
        self._dynamic_pd_prefill_copy_streams = [torch.npu.Stream(device=self.device) for _ in range(8)]
        self.copy_stream_counter = 0
        self._dynamic_pd_decode_stream = (torch.npu.Stream(device=self.device) if self.use_async_scheduling else torch.npu.current_stream(device=self.device))
        self._dynamic_pd_prepare_inputs_event = (torch.npu.Event() if self.use_async_scheduling else None)
        self._dynamic_pd_decode_lane = self._make_dynamic_pd_lane(with_new_batch=False, name="decode")
        self._dynamic_pd_prefill_lane = self._make_dynamic_pd_lane(with_new_batch=True, name="prefill")
        self._dynamic_pd_pending_prefill_outputs: list[ModelRunnerOutput | AsyncModelRunnerOutput] = []
        self._dynamic_pd_decode_immediate_output = None
        self._dynamic_pd_prefill_immediate_output = None
        self._dynamic_pd_active = False
        self._dynamic_pd_prefill_decode_fence = _prefill_decode_fence_mode()
        self._dynamic_pd_decode_pending_prefill_mode = (
            _decode_pending_prefill_mode())
        self._dynamic_pd_ready = True

    def _make_dynamic_pd_lane(self, with_new_batch: bool, name: str = "") -> _LaneState:
        input_batch = (self._make_dynamic_pd_input_batch() if with_new_batch else self.input_batch)
        lane = _LaneState(
            name=name,
            input_batch=input_batch,
            input_ids=(self._clone_buffer(self.input_ids) if with_new_batch else self.input_ids),
            positions=(self._clone_buffer(self.positions) if with_new_batch else self.positions),
            query_start_loc=(self._clone_buffer(self.query_start_loc) if with_new_batch else self.query_start_loc),
            seq_lens=(self._clone_buffer(self.seq_lens) if with_new_batch else self.seq_lens),
            async_output_copy_stream=(
                self._dynamic_pd_prefill_copy_streams[self.copy_stream_counter]
                if with_new_batch and hasattr(self, "_dynamic_pd_prefill_copy_streams")
                else self.async_output_copy_stream),
            prepare_inputs_event=(
                self._dynamic_pd_prepare_inputs_event
                if with_new_batch and hasattr(self, "_dynamic_pd_prepare_inputs_event")
                else self.prepare_inputs_event),
        )
        for name in self._dynamic_pd_optional_buffer_names():
            if hasattr(self, name):
                setattr(lane, name,
                        self._clone_buffer(getattr(self, name))
                        if with_new_batch else getattr(self, name))
        return lane

    def _make_dynamic_pd_input_batch(self) -> NPUInputBatch:
        block_tables = self.input_batch.block_table.block_tables
        block_sizes = [table.physical_block_size for table in block_tables]
        kernel_block_sizes = [table.kernel_sizes for table in block_tables]
        max_num_blocks = [
            table.max_num_blocks_per_req for table in block_tables
        ]
        max_model_len = max(self.max_model_len, self.max_encoder_len)
        return NPUInputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=block_sizes,
            kernel_block_sizes=kernel_block_sizes,
            max_num_blocks_per_req=max_num_blocks,
            logitsprocs=self.input_batch.logitsprocs,
            logitsprocs_need_output_token_ids=(
                self.input_batch.logitsprocs_need_output_token_ids),
            is_spec_decode=bool(self.vllm_config.speculative_config),
            is_pooling_model=self.is_pooling_model,
            num_speculative_tokens=(
                self.vllm_config.speculative_config.num_speculative_tokens
                if self.vllm_config.speculative_config else 0),
            cp_kv_cache_interleave_size=(
                self.parallel_config.cp_kv_cache_interleave_size),
        )

    def _dynamic_pd_optional_buffer_names(self) -> tuple[str, ...]:
        return (
            "gdn_query_start_loc",
            "encoder_seq_lens",
            "dcp_local_seq_lens",
            "inputs_embeds",
            "is_token_ids",
            "discard_request_indices",
            "discard_request_mask",
            "num_decode_draft_tokens",
            "num_draft_tokens",
            "num_accepted_tokens",
            "mrope_positions",
            "xdrope_positions",
        )

    def _clone_buffer(self, buffer: Any) -> Any:
        gpu_tensor = getattr(buffer, "gpu", None)
        if gpu_tensor is None:
            return buffer
        shape = tuple(gpu_tensor.shape)
        numpy_enabled = getattr(buffer, "np", None) is not None
        return self._make_buffer(
            *shape, dtype=gpu_tensor.dtype, numpy=numpy_enabled)

    @contextmanager
    def _dynamic_pd_lane_context(
        self,
        lane: _LaneState,
        *,
        use_offload_tp: bool = False,
    ):
        attrs = [
            "input_batch",
            "input_ids",
            "positions",
            "query_start_loc",
            "seq_lens",
            "async_output_copy_stream",
            "prepare_inputs_event",
            "execute_model_state",
            "kv_connector_output",
            "num_discarded_requests",
            "query_lens",
            "attn_state",
            "with_prefill",
            "cpu_slot_mapping",
            "slot_mapping",
            "_draft_token_ids",
        ]
        attrs.extend(name for name in self._dynamic_pd_optional_buffer_names()
                     if hasattr(lane, name))

        saved = {name: getattr(self, name, None) for name in attrs}
        for name in attrs:
            lane_name = "draft_token_ids" if name == "_draft_token_ids" else name
            if hasattr(lane, lane_name):
                setattr(self, name, getattr(lane, lane_name))

        token = _USE_OFFLOAD_TP_CONTEXT.set(use_offload_tp)
        if lane.name == "prefill" and self._dynamic_pd_prefill_copy_stream is not None:
           self.async_output_copy_stream = self._dynamic_pd_prefill_copy_streams[self.copy_stream_counter%len(self._dynamic_pd_prefill_copy_streams)]
           self.copy_stream_counter += 1
        try:
            yield
        finally:
            _USE_OFFLOAD_TP_CONTEXT.reset(token)
            for name in attrs:
                lane_name = "draft_token_ids" if name == "_draft_token_ids" else name
                if hasattr(lane, lane_name):
                    setattr(lane, lane_name, getattr(self, name, None))
            for name, value in saved.items():
                setattr(self, name, value)

    def _has_async_prefill(self, scheduler_output: SchedulerOutput) -> bool:
        return bool(getattr(scheduler_output, "prefill_num_scheduled_tokens",
                            None))

    def _make_decode_scheduler_output(
        self,
        scheduler_output: SchedulerOutput,
    ) -> SchedulerOutput:
        moved_req_ids = set(scheduler_output.prefill_num_scheduled_tokens)
        if not moved_req_ids:
            return scheduler_output

        scheduled_new_reqs = [
            req for req in scheduler_output.scheduled_new_reqs
            if req.req_id not in moved_req_ids
        ]
        scheduled_cached_reqs, _ = _split_cached_request_data(
            scheduler_output.scheduled_cached_reqs, moved_req_ids)
        num_scheduled_tokens = _filter_mapping(
            scheduler_output.num_scheduled_tokens, moved_req_ids)

        decode_output = replace(
            scheduler_output,
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=scheduled_cached_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
            scheduled_spec_decode_tokens=_filter_mapping(
                scheduler_output.scheduled_spec_decode_tokens, moved_req_ids),
            scheduled_encoder_inputs=_filter_mapping(
                scheduler_output.scheduled_encoder_inputs, moved_req_ids),
        )
        _empty_prefill_attrs(decode_output)
        return decode_output

    def _make_prefill_scheduler_output(
        self,
        scheduler_output: SchedulerOutput
    ) -> SchedulerOutput:
        prefill_output = replace(
            scheduler_output,
            scheduled_new_reqs=scheduler_output.prefill_scheduled_new_reqs,
            scheduled_cached_reqs=scheduler_output.prefill_scheduled_cached_reqs,
            num_scheduled_tokens=scheduler_output.prefill_num_scheduled_tokens,
            total_num_scheduled_tokens=(
                scheduler_output.prefill_total_num_scheduled_tokens),
            scheduled_spec_decode_tokens=(
                scheduler_output.prefill_scheduled_spec_decode_tokens),
            scheduled_encoder_inputs=(
                scheduler_output.prefill_scheduled_encoder_inputs),
            num_common_prefix_blocks=(
                scheduler_output.prefill_num_common_prefix_blocks
                or scheduler_output.num_common_prefix_blocks),
            finished_req_ids=scheduler_output.finished_req_ids,
            free_encoder_mm_hashes=[],
            kv_connector_metadata=scheduler_output.kv_connector_metadata.__class__(),
        )
        _copy_prefill_attrs(prefill_output, scheduler_output)
        return prefill_output

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: Any = None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | Any | None:
        self._ensure_dynamic_pd_runtime()

        if not self._has_async_prefill(scheduler_output):
            t1 = time.perf_counter()
            self._fence_pending_prefill_before_decode()
            t2 = time.perf_counter()
            with self._maybe_disable_decode_aclgraph_for_pending_prefill():
                _decode_stream = self.default_stream
                with (
                    torch.npu.stream(_decode_stream),
                    self._dynamic_pd_lane_context(self._dynamic_pd_decode_lane)
                ):
                    t1 = time.perf_counter()
                    result = NPUModelRunnerPatch._original_execute_model(
                        self, scheduler_output, intermediate_tensors)
                    t2 = time.perf_counter()
                    self._dynamic_pd_decode_lane.work_stream = _decode_stream
            logger.info(
                "DynamicPD: execute_model with no async prefill, num_scheduled_tokens: %d, "
                "result type: %s, execution time: %.6f seconds",
                scheduler_output.total_num_scheduled_tokens,
                type(result).__name__,
                t2 - t1,
            )
            if result is not None:
                ready = self._take_ready_prefill_outputs(block=False)
                if ready:
                    logger.info(
                        "DynamicPD: execute_model with no async prefill, "
                        "returning AsyncMergedNPUModelRunnerOutput with %d ready prefill outputs, blocking: false",
                        len(ready)
                    )
                    return AsyncMergedNPUModelRunnerOutput(result, ready)
            return result

        self._dynamic_pd_active = True
        self._dynamic_pd_decode_immediate_output = None
        self._dynamic_pd_prefill_immediate_output = None

        decode_output = self._make_decode_scheduler_output(scheduler_output)
        prefill_output = self._make_prefill_scheduler_output(scheduler_output)
        _decode_stream = self.default_stream
        _prefill_stream = self._dynamic_pd_prefill_stream
        with (
            torch.npu.stream(_decode_stream),
            self._dynamic_pd_lane_context(self._dynamic_pd_decode_lane)
        ):
            t1 = time.perf_counter()
            self._dynamic_pd_decode_immediate_output = (
                NPUModelRunnerPatch._original_execute_model(
                    self, decode_output, intermediate_tensors))
            t2 = time.perf_counter()
            logger.info(
                "DynamicPD: Executed decode lane with %d scheduled tokens, "
                "execution time: %.6f seconds",
                decode_output.total_num_scheduled_tokens,
                t2 - t1,
            )
            self._dynamic_pd_decode_lane.work_stream = _decode_stream

        if prefill_output.total_num_scheduled_tokens:
            with (
                torch.npu.stream(_prefill_stream),
                self._dynamic_pd_lane_context(
                    self._dynamic_pd_prefill_lane, use_offload_tp=True),
            ):
                t1 = time.perf_counter()
                self._dynamic_pd_prefill_immediate_output = (
                    NPUModelRunnerPatch._original_execute_model(
                        self, prefill_output, intermediate_tensors))
                t2 = time.perf_counter()
                
                logger.info(
                    "DynamicPD: Executed prefill lane with %d scheduled tokens, "
                    "execution time: %.6f seconds",
                    prefill_output.total_num_scheduled_tokens,
                    t2 - t1,
                )
                self._dynamic_pd_prefill_lane.work_stream = _prefill_stream

        decode_has_state = (
            self._dynamic_pd_decode_lane.execute_model_state is not None)
        prefill_has_state = (
            self._dynamic_pd_prefill_lane.execute_model_state is not None)
        if decode_has_state or prefill_has_state:
            return None

        self._dynamic_pd_active = False
        return self._merge_dynamic_pd_outputs(
            self._dynamic_pd_decode_immediate_output,
            self._dynamic_pd_prefill_immediate_output,
            block_when_no_decode=True,
        )

    def sample_tokens(self, grammar_output: Any | None):
        self._ensure_dynamic_pd_runtime()

        decode_output = self._dynamic_pd_decode_immediate_output
        prefill_output = self._dynamic_pd_prefill_immediate_output

        if self._dynamic_pd_decode_lane.execute_model_state is not None:
            t1 = time.perf_counter()
            assert self._dynamic_pd_decode_lane.work_stream is not None, (
                "DynamicPD: Decode lane has execute_model_state but no work_stream")
            num_tokens = self._dynamic_pd_decode_lane.execute_model_state.scheduler_output.total_num_scheduled_tokens
            logits_shape = self._dynamic_pd_decode_lane.execute_model_state.logits.shape
            with (
                torch.npu.stream(self._dynamic_pd_decode_lane.work_stream),
                self._dynamic_pd_lane_context(self._dynamic_pd_decode_lane)
            ):
                decode_output = NPUModelRunnerPatch._original_sample_tokens(
                    self, grammar_output)
            t2 = time.perf_counter()
            logger.info(
                "DynamicPD: Sampling tokens for decode lane, "
                "execution time: %.6f seconds, state tokens %s",
                t2 - t1,
                num_tokens
            )

        if self._dynamic_pd_prefill_lane.execute_model_state is not None:
            assert self._dynamic_pd_prefill_lane.work_stream is not None, (
                "DynamicPD: Prefill lane has execute_model_state but no work_stream")
            num_tokens = self._dynamic_pd_prefill_lane.execute_model_state.scheduler_output.total_num_scheduled_tokens
            logit_shape = self._dynamic_pd_prefill_lane.execute_model_state.logits.shape
            t1 = time.perf_counter()
            with (
                torch.npu.stream(self._dynamic_pd_prefill_lane.work_stream),
                self._dynamic_pd_lane_context(
                    self._dynamic_pd_prefill_lane, use_offload_tp=True),
            ):
                prefill_output = NPUModelRunnerPatch._original_sample_tokens(
                    self, None)
            t2 = time.perf_counter()
            logger.info(
                "DynamicPD: Sampling tokens for prefill lane, "
                "execution time: %.6f seconds, state tokens %s",
                t2 - t1,
                num_tokens
            )

        if prefill_output is not None:
            self._dynamic_pd_pending_prefill_outputs.append(prefill_output)

        self._dynamic_pd_active = len(self._dynamic_pd_pending_prefill_outputs) > 0
        
        ready = self._take_ready_prefill_outputs(block=decode_output is None)
        if ready:
            logger.info(
                "DynamicPD: sample_tokens find finished prefill outputs, "
                "returning AsyncMergedNPUModelRunnerOutput with %d ready prefill outputs, blocking: %s",
                len(ready),
                decode_output is None
            )
        return self._merge_dynamic_pd_outputs(
            decode_output, ready if ready else None, block_when_no_decode=decode_output is None)

    def _take_ready_prefill_outputs(
        self,
        *,
        block: bool,
    ) -> list[ModelRunnerOutput | AsyncModelRunnerOutput]:
        pending = self._dynamic_pd_pending_prefill_outputs
        if not pending:
            return []
        if block:
            return [pending.pop(0)]

        ready: list[ModelRunnerOutput | AsyncModelRunnerOutput] = []
        while pending and _async_output_ready(pending[0]):
            ready.append(pending.pop(0))
        return ready

    def _merge_dynamic_pd_outputs(
        self,
        decode_output: ModelRunnerOutput | AsyncModelRunnerOutput | None,
        prefill_output: ModelRunnerOutput | AsyncModelRunnerOutput | List[ModelRunnerOutput | AsyncModelRunnerOutput] | None,
        *,
        block_when_no_decode: bool,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        prefill_outputs = []
        if prefill_output is not None:
            if isinstance(prefill_output, list):
                prefill_outputs.extend(prefill_output)
            else:
                prefill_outputs.append(prefill_output)
        prefill_outputs.extend(
            self._take_ready_prefill_outputs(block=block_when_no_decode
                                             and decode_output is None))

        if prefill_outputs:
            logger.info(
                "Merging decode output (%s) with %d prefill outputs (%s), blocking: %s",
                type(decode_output).__name__,
                len(prefill_outputs),
                ", ".join(type(output).__name__ for output in prefill_outputs),
                block_when_no_decode and decode_output is None
            )
            return AsyncMergedNPUModelRunnerOutput(decode_output, prefill_outputs)
        logger.info(
            "No prefill outputs to merge, returning decode output (%s)",
            type(decode_output).__name__,
        )
        return decode_output

    def _fence_pending_prefill_before_decode(self) -> None:
        if not self._dynamic_pd_pending_prefill_outputs:
            return
        if all(_async_output_ready(output)
               for output in self._dynamic_pd_pending_prefill_outputs):
            return
        fence_mode = self._dynamic_pd_prefill_decode_fence
        if fence_mode in {"0", "false", "none", "off"}:
            return

        if fence_mode != "copy":
            snapshot_events = [
                event for event in (
                    getattr(output, "device_snapshot_ready_event", None)
                    for output in self._dynamic_pd_pending_prefill_outputs
                )
                if event is not None
            ]
            if snapshot_events:
                logger.debug(
                    "Waiting for %d pending async prefill device snapshots "
                    "before decode", len(snapshot_events))
                current_stream = torch.npu.current_stream()
                for event in snapshot_events:
                    current_stream.wait_event(event)
                return

        first_pending = self._dynamic_pd_pending_prefill_outputs[0]
        copy_event = getattr(first_pending, "async_copy_ready_event", None)
        logger.debug("Waiting for pending async prefill output before decode")
        if copy_event is not None:
            copy_event.synchronize()
            return
        torch.npu.current_stream().synchronize()

    @contextmanager
    def _maybe_disable_decode_aclgraph_for_pending_prefill(self):
        pending = self._dynamic_pd_pending_prefill_outputs
        mode = self._dynamic_pd_decode_pending_prefill_mode
        if (not pending or mode in {"0", "false", "none", "off"}
                or all(_async_output_ready(output) for output in pending)):
            yield
            return

        token = _DECODE_PENDING_PREFILL_MODE.set(mode)
        try:
            logger.debug(
                "Decode is running with pending async prefill outputs; "
                "decode_pending_prefill_mode=%s", mode)
            yield
        finally:
            _DECODE_PENDING_PREFILL_MODE.reset(token)
