from __future__ import annotations

import time
from typing import Any, List, Tuple


from vllm.logger import logger
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput

from dynamicPD.patching import dynamicPDPatch


DEFAULT_ASYNC_THRESHOLD = 1024
DEFAULT_ASYNC_CHUNK_SIZE = 2048


def _get_dynamic_pd_config(vllm_config: Any) -> dict[str, Any]:
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return {}

    extra_config = getattr(kv_transfer_config, "get_from_extra_config", None)
    if extra_config is None:
        return {}

    config = extra_config("dynamic_pd_config", {}) or {}
    if not isinstance(config, dict):
        raise TypeError("dynamic_pd_config must be a dict")
    return config

def _empty_prefill_fields(output: SchedulerOutput) -> None:
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


def _split_mapping(mapping: dict[str, Any],
                   offload_req_ids: set[str]) -> tuple[dict[str, Any],
                                                       dict[str, Any]]:
    kept: dict[str, Any] = {}
    moved: dict[str, Any] = {}
    for req_id, value in mapping.items():
        if req_id in offload_req_ids:
            moved[req_id] = value
        else:
            kept[req_id] = value
    return kept, moved


def _split_cached_request_data(
    cached: CachedRequestData,
    offload_req_ids: set[str],
) -> tuple[CachedRequestData, CachedRequestData]:
    if not cached.req_ids:
        empty = CachedRequestData.make_empty()
        return empty, empty

    kept_req_ids: list[str] = []
    moved_req_ids: list[str] = []
    kept_new_token_ids: list[list[int]] = []
    moved_new_token_ids: list[list[int]] = []
    kept_new_block_ids: list[tuple[list[int], ...] | None] = []
    moved_new_block_ids: list[tuple[list[int], ...] | None] = []
    kept_num_computed_tokens: list[int] = []
    moved_num_computed_tokens: list[int] = []
    kept_num_output_tokens: list[int] = []
    moved_num_output_tokens: list[int] = []

    for i, req_id in enumerate(cached.req_ids):
        target_req_ids = moved_req_ids if req_id in offload_req_ids else kept_req_ids
        target_new_token_ids = (
            moved_new_token_ids if req_id in offload_req_ids else kept_new_token_ids)
        target_new_block_ids = (
            moved_new_block_ids if req_id in offload_req_ids else kept_new_block_ids)
        target_num_computed_tokens = (
            moved_num_computed_tokens
            if req_id in offload_req_ids else kept_num_computed_tokens)
        target_num_output_tokens = (
            moved_num_output_tokens
            if req_id in offload_req_ids else kept_num_output_tokens)

        target_req_ids.append(req_id)
        target_new_block_ids.append(cached.new_block_ids[i])
        target_num_computed_tokens.append(cached.num_computed_tokens[i])
        target_num_output_tokens.append(cached.num_output_tokens[i])

        if i < len(cached.new_token_ids):
            target_new_token_ids.append(cached.new_token_ids[i])

    kept_set = set(kept_req_ids)
    moved_set = set(moved_req_ids)
    kept = CachedRequestData(
        req_ids=kept_req_ids,
        resumed_req_ids=cached.resumed_req_ids & kept_set,
        new_token_ids=kept_new_token_ids,
        all_token_ids={
            req_id: token_ids
            for req_id, token_ids in cached.all_token_ids.items()
            if req_id in kept_set
        },
        new_block_ids=kept_new_block_ids,
        num_computed_tokens=kept_num_computed_tokens,
        num_output_tokens=kept_num_output_tokens,
    )
    moved = CachedRequestData(
        req_ids=moved_req_ids,
        resumed_req_ids=cached.resumed_req_ids & moved_set,
        new_token_ids=moved_new_token_ids,
        all_token_ids={
            req_id: token_ids
            for req_id, token_ids in cached.all_token_ids.items()
            if req_id in moved_set
        },
        new_block_ids=moved_new_block_ids,
        num_computed_tokens=moved_num_computed_tokens,
        num_output_tokens=moved_num_output_tokens,
    )
    return kept, moved

def _merge_cached_request_data(
    cached1: CachedRequestData,
    cached2: CachedRequestData,
) -> CachedRequestData:
    merged_req_ids = cached1.req_ids + cached2.req_ids
    merged_resumed_req_ids = cached1.resumed_req_ids | cached2.resumed_req_ids
    merged_new_token_ids = cached1.new_token_ids + cached2.new_token_ids
    merged_all_token_ids = {**cached1.all_token_ids, **cached2.all_token_ids}
    merged_new_block_ids = cached1.new_block_ids + cached2.new_block_ids
    merged_num_computed_tokens = (
        cached1.num_computed_tokens + cached2.num_computed_tokens)
    merged_num_output_tokens = (
        cached1.num_output_tokens + cached2.num_output_tokens)

    return CachedRequestData(
        req_ids=merged_req_ids,
        resumed_req_ids=merged_resumed_req_ids,
        new_token_ids=merged_new_token_ids,
        all_token_ids=merged_all_token_ids,
        new_block_ids=merged_new_block_ids,
        num_computed_tokens=merged_num_computed_tokens,
        num_output_tokens=merged_num_output_tokens,
    )


class SchedulerPatch(dynamicPDPatch[Scheduler]):
    _original_init = Scheduler.__init__
    _original_schedule = Scheduler.schedule
    _original_update_from_output = Scheduler.update_from_output
    _original_has_requests = Scheduler.has_requests

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        SchedulerPatch._original_init(self, *args, **kwargs)

        dynamic_pd_config = _get_dynamic_pd_config(self.vllm_config)
        self.use_async_offload = bool(dynamic_pd_config.get("use_async_offload", False))
        self.async_threshold = int(dynamic_pd_config.get("async_threshold", DEFAULT_ASYNC_THRESHOLD))
        configured_chunk_size = dynamic_pd_config.get("async_chunk_size", DEFAULT_ASYNC_CHUNK_SIZE)
        if configured_chunk_size is None:
            configured_chunk_size = self.async_threshold
        self.async_chunk_size = int(configured_chunk_size)

        if self.async_threshold < 0:
            raise ValueError("async_threshold must be non-negative")
        if self.async_chunk_size < 0:
            raise ValueError("async_chunk_size must be non-negative")

        if self.use_async_offload and not self.scheduler_config.async_scheduling:
            raise ValueError(
                "dynamicPD use_async_offload requires "
                "scheduler_config.async_scheduling=True")

        self.is_prefill = False
        kv_transfer_config = getattr(self.vllm_config, "kv_transfer_config", None)
        if kv_transfer_config is not None:
            self.is_prefill = bool(
                getattr(kv_transfer_config, "is_kv_producer", False))

        self.async_offload_inflight_req_ids: set[str] = set()
        self.async_offload_inflight_tokens: dict[str, int] = {}
        self.async_offload_continuing_req_ids: set[str] = set()
        self.async_offload_req_ids = self.async_offload_inflight_req_ids
        self.prefill_tokens_in_decode = 0
        self.prefill_reqs_in_decode_batch: set[str] = set()

        if self.use_async_offload and self._is_decode_side_scheduler():
            self._configure_async_offload_chunking()

        logger.info(
            "dynamicPD scheduler async offload: enabled=%s threshold=%s chunk_size=%s",
            self.use_async_offload,
            self.async_threshold,
            self.async_chunk_size,
        )
        self.batch_infos = []

    def _configure_async_offload_chunking(self) -> None:
        if self.async_chunk_size <= 0:
            return

        threshold = self.scheduler_config.long_prefill_token_threshold
        if threshold <= 0:
            threshold = self.async_chunk_size
        else:
            threshold = min(threshold, self.async_chunk_size)
        self.scheduler_config.long_prefill_token_threshold = threshold
        self.scheduler_config.enable_chunked_prefill = True
        logger.info(
            "dynamicPD async offload chunking: "
            "long_prefill_token_threshold=%s",
            threshold,
        )

    def has_requests(self) -> bool:
        return (SchedulerPatch._original_has_requests(self)
                or bool(self.async_offload_inflight_req_ids))   

    def schedule(self) -> SchedulerOutput:
        skipped_inflight_reqs = self._hold_inflight_async_offload_requests()
        if skipped_inflight_reqs:
            logger.info(
                "skipped_inflight_reqs: %s",
                (
                    len(skipped_inflight_reqs),
                    [req.request_id for _, req in skipped_inflight_reqs],
                ),
            )
        try:
            output = SchedulerPatch._original_schedule(self)
        finally:
            self._restore_inflight_async_offload_requests(skipped_inflight_reqs)

        _empty_prefill_fields(output)
        if self.use_async_offload and self._is_decode_side_scheduler():
            offload_req_ids = self._select_async_offload_requests(output)
            if offload_req_ids:
                self._move_requests_to_prefill_output(output, offload_req_ids)
                logger.info(
                    "offloaded async prefill requests: count=%s req_ids=%s",
                    len(offload_req_ids),
                    offload_req_ids,
                )

        self._refresh_async_offload_pressure(output)
        logger.info(
            "scheduler output: %s, prefill: %s, running: %d, waiting: %d, finished: %d",
            output.num_scheduled_tokens,
            output.prefill_num_scheduled_tokens,
            len(self.running),
            len(self.waiting),
            len(self.finished_req_ids),
        )
        self.batch_infos.append(output)
        return output

    def update_from_output(        
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput
    ) -> Any:
        self._mark_async_prefill_finished(model_runner_output)
        self.batch_infos.pop(0) if self.batch_infos else None
        self._move_prefill_output_back(scheduler_output)
        return SchedulerPatch._original_update_from_output(self, scheduler_output, model_runner_output)

    def _hold_inflight_async_offload_requests(self) -> list[tuple[int, Any]]:
        if not self.async_offload_inflight_req_ids:
            return []

        skipped: list[tuple[int, Any]] = []
        kept_running = []
        for index, request in enumerate(self.running):
            if request.request_id in self.async_offload_inflight_req_ids:
                skipped.append((index, request))
            else:
                kept_running.append(request)
        self.running = kept_running
        return skipped

    def _restore_inflight_async_offload_requests(
        self,
        skipped_reqs: list[tuple[int, Any]],
    ) -> None:
        if not skipped_reqs:
            return
        logger.info(
            "restoring skipped_inflight_reqs: %s",
            (len(skipped_reqs), [req.request_id for _, req in skipped_reqs]),
        )

        running_req_ids = {request.request_id for request in self.running}
        for index, request in sorted(skipped_reqs, key=lambda item: item[0]):
            if request.request_id not in self.requests:
                continue
            if request.request_id in running_req_ids:
                continue
            self.running.insert(min(index, len(self.running)), request)
            running_req_ids.add(request.request_id)

    def _is_decode_side_scheduler(self) -> bool:
        kv_transfer_config = getattr(self.vllm_config, "kv_transfer_config", None)
        if kv_transfer_config is None:
            return True
        return bool(getattr(kv_transfer_config, "is_kv_consumer", False))

    def _select_async_offload_requests(
        self,
        output: SchedulerOutput,
    ) -> set[str]:
        offload_req_ids: set[str] = set()
        start_num_computed_tokens = {
            req.req_id: req.num_computed_tokens
            for req in output.scheduled_new_reqs
        }
        start_num_computed_tokens.update({
            req_id: num_computed_tokens
            for req_id, num_computed_tokens in zip(
                output.scheduled_cached_reqs.req_ids,
                output.scheduled_cached_reqs.num_computed_tokens)
        })

        for req_id, num_scheduled_tokens in output.num_scheduled_tokens.items():
            request = self.requests.get(req_id)
            if request is None:
                continue

            start_num_computed = start_num_computed_tokens.get(req_id)
            if start_num_computed is None:
                continue

            prompt_tokens_scheduled = max(
                0,
                min(num_scheduled_tokens,
                    request.num_prompt_tokens - start_num_computed),
            )
            if prompt_tokens_scheduled == 0:
                continue

            prompt_tokens_remaining = max(
                0, request.num_prompt_tokens - start_num_computed)
            already_offloaded = req_id in self.async_offload_inflight_req_ids
            continuing = req_id in self.async_offload_continuing_req_ids
            large_enough = prompt_tokens_remaining >= self.async_threshold
            if already_offloaded or continuing or large_enough:
                offload_req_ids.add(req_id)

        return offload_req_ids

    def _move_requests_to_prefill_output(
        self,
        output: SchedulerOutput,
        offload_req_ids: set[str],
    ) -> None:
        # The Ascend runner still uses the main scheduled_new/cached fields to
        # refresh persistent request state, then routes requests to the decode
        # or async-prefill batch through prefill_request_not_put. Keep the main
        # scheduler output intact and publish the async-prefill view as metadata.
        output.prefill_scheduled_new_reqs = [
            req for req in output.scheduled_new_reqs
            if req.req_id in offload_req_ids
        ]
        _, output.prefill_scheduled_cached_reqs = _split_cached_request_data(
            output.scheduled_cached_reqs, offload_req_ids)
        output.num_scheduled_tokens, output.prefill_num_scheduled_tokens = _split_mapping(
            output.num_scheduled_tokens, offload_req_ids)
        output.scheduled_spec_decode_tokens, (
            output.prefill_scheduled_spec_decode_tokens) = _split_mapping(
                output.scheduled_spec_decode_tokens, offload_req_ids)
        output.scheduled_encoder_inputs, output.prefill_scheduled_encoder_inputs = (
            _split_mapping(output.scheduled_encoder_inputs, offload_req_ids))

        output.prefill_total_num_scheduled_tokens = sum(
            output.prefill_num_scheduled_tokens.values())
        output.prefill_num_common_prefix_blocks = output.num_common_prefix_blocks
        output.prefill_request_ids = set(output.prefill_num_scheduled_tokens)
        output.prefill_request_not_put = set(output.prefill_num_scheduled_tokens)

        self.async_offload_inflight_req_ids.update(output.prefill_request_ids)
        self.async_offload_inflight_tokens.update(
            output.prefill_num_scheduled_tokens)
        self.async_offload_continuing_req_ids.update(output.prefill_request_ids)
        self.prefill_reqs_in_decode_batch.update(output.prefill_request_ids)
        
    def _move_prefill_output_back(self, output: SchedulerOutput) -> None:
        output.scheduled_new_reqs.extend(output.prefill_scheduled_new_reqs)
        output.scheduled_cached_reqs = _merge_cached_request_data(
            output.scheduled_cached_reqs, output.prefill_scheduled_cached_reqs)
        output.num_scheduled_tokens.update(output.prefill_num_scheduled_tokens)
        output.scheduled_spec_decode_tokens.update(
            output.prefill_scheduled_spec_decode_tokens)
        output.scheduled_encoder_inputs.update(
            output.prefill_scheduled_encoder_inputs)

        output.num_common_prefix_blocks = output.prefill_num_common_prefix_blocks

    def _refresh_async_offload_pressure(self, output: SchedulerOutput) -> None:
        active_req_ids = set(self.requests)
        finished_req_ids = set(output.finished_req_ids)
        stale_req_ids = finished_req_ids | (
            self.async_offload_inflight_req_ids - active_req_ids)
        self.async_offload_inflight_req_ids.difference_update(stale_req_ids)
        for req_id in stale_req_ids:
            self.async_offload_inflight_tokens.pop(req_id, None)
        self.async_offload_continuing_req_ids.intersection_update(
            active_req_ids - finished_req_ids)

        self.prefill_reqs_in_decode_batch.intersection_update(
            self.async_offload_inflight_req_ids)

        self.prefill_tokens_in_decode = sum(
            self.async_offload_inflight_tokens.values())

    def _mark_async_prefill_finished(self, model_runner_output: ModelRunnerOutput) -> None:
        finished_prefill_reqs = getattr(
            model_runner_output, "finished_prefill_reqs", None)
        if not isinstance(finished_prefill_reqs, (set, list, tuple)):
            return
        if not finished_prefill_reqs:
            return
        
        logger.info("finished_prefill_reqs: %s", finished_prefill_reqs)

        for req_id in finished_prefill_reqs:
            self.async_offload_inflight_req_ids.discard(req_id)
            self.async_offload_inflight_tokens.pop(req_id, None)
            self.prefill_reqs_in_decode_batch.discard(req_id)
            request = self.requests.get(req_id)
            if (request is None
                    or request.num_computed_tokens >= request.num_prompt_tokens):
                self.async_offload_continuing_req_ids.discard(req_id)
        self.prefill_tokens_in_decode = sum(
            self.async_offload_inflight_tokens.values())
