from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Optional

from vllm.config import VllmConfig
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.logger import logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.output import (NewRequestData,
                                       SchedulerOutput)
from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                              create_request_queue)
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm.v1.core.sched.scheduler import Scheduler

from dynamicPD.patching import dynamicPDPatch
from dynamicPD.vllm.vllm.entrypoints.openai.protocol import UpdateRequest

USE_TWO_BATCH = True
SPLIT_THRESHOLD = 1024

class SchedulerPatch(dynamicPDPatch[Scheduler]):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.use_two_batch = USE_TWO_BATCH
        self.split_threshold = SPLIT_THRESHOLD
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder
        
        self.is_prefill = False
        if vllm_config.kv_transfer_config:
            self.is_prefill = vllm_config.kv_transfer_config.is_kv_producer

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: Optional[dict[int, set[str]]] = (
            defaultdict(set) if include_finished_set else None)

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        logger.info(f"self.max_num_scheduled_tokens : {self.max_num_scheduled_tokens} ")
        self.max_model_len = self.scheduler_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events)

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        if self.vllm_config.kv_transfer_config is not None:
            assert len(self.kv_cache_config.kv_cache_groups) == 1, (
                "Multiple KV cache groups are not currently supported "
                "with KV connectors")
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported "
                "with KV connectors")
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config, role=KVConnectorRole.SCHEDULER)

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_rank,
        )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = self.cache_config.block_size

        self.dcp_world_size = \
            vllm_config.parallel_config.decode_context_parallel_size
        # Note(hc): The scheduler’s block_size must be multiplied
        # by dcp_world_size, since block hashes are computed on the
        # original full token sequence at a granularity of
        # original_block_size × dcp_world_size.
        if self.dcp_world_size > 1:
            self.block_size *= self.dcp_world_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        if self.scheduler_config.policy == "priority":
            self.policy = SchedulingPolicy.PRIORITY
        elif self.scheduler_config.policy == "fcfs":
            self.policy = SchedulingPolicy.FCFS
        else:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}")
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()
        self.prefill_finished_req_ids: set[str] = set()

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed) for MM models as well as encoder-decoder
        # transformers.
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.to_migrate_reqs: list[Request] = []

        self.prefill_request_in_decode : dict[str, int] = {} #用来记录放到prefill batch中的pre_req,并在完成时删除
        
        self.prefill_tokens_in_decode : int = 0
        self.scheduled_butnot_finished_pre_req : list[Request] = [] #记录还未完成的prefill_req，当里面还有req时不停止调度，以及在完成prefill阶段后及时添加到running的队头继续调度
        #上述都记录的是需要单独占用prefill_input_batch的prefill_reqs；下面的则是记录所有migrate的prefill_reqs
        self.prefill_request_not_put : dict[str, int] = {} #记录那些已经在decode里但还没有调度完所有prompt的prefill reqs，key是req_id，value是还未放到running队列里的prompt token数量
        self.prefill_reqs_in_decode_batch : list[str] = [] #记录所有migrate的prefill_reqs，用来判断是否扩大chunk_size

    def has_requests(self) -> bool:
        """Returns True if there are unfinished requests, or finished requests
        not yet returned in SchedulerOutputs."""
        return self.has_unfinished_requests() or self.has_finished_requests() or self.to_migrate_reqs or self.scheduled_butnot_finished_pre_req or self.prefill_reqs_in_decode_batch

    def schedule(self, finished_prefill_reqs: set[str]) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.
        if not self.is_prefill:
            if not self.prefill_reqs_in_decode_batch and not self.prefill_request_not_put:
                self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens
                logger.debug(f"change chunk_size to {self.max_num_scheduled_tokens}")
                
        prefill_reqs: dict[str, int] = {} #每轮调度更新，用来储存本轮调度中的prefill reqs和它们的prompt token数量，key是req_id，value是prompt token数量

        scheduled_new_reqs: list[Request] = []
        prefill_scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        prefill_req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        prefill_num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        prefill_scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        prefill_scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        # First, schedule the RUNNING requests.
        logger.debug(f"self.scheduled_butnot_finished_pre_req : {self.scheduled_butnot_finished_pre_req} ; self.prefill_request_in_decode : {self.prefill_request_in_decode} ; finished_prefill_reqs : {finished_prefill_reqs}")
        pre_req_gen_tokens: list[Request] = []
        for req in self.scheduled_butnot_finished_pre_req:
            if not self.is_prefill:
                if req.request_id in self.prefill_request_in_decode:
                    if req.request_id in finished_prefill_reqs:
                        self.prefill_request_in_decode.pop(req.request_id)
                logger.debug(f"check prefill request {req.request_id} ; prefill_request_in_decode : {self.prefill_request_in_decode}")
            if req.request_id not in self.prefill_request_in_decode and len(self.running) < self.max_num_running_reqs:
                pre_req_gen_tokens.append(req)
                if req.status == RequestStatus.RUNNING:
                    self.running.insert(0,req)
                    logger.info(f"put back prefill request {req.request_id} to running queue")
                    # break
                
        for req in pre_req_gen_tokens:
            self.scheduled_butnot_finished_pre_req.remove(req)
            
        self.prefill_tokens_in_decode = sum(self.prefill_request_not_put.values())
        prefill_request_not_put: set[str] = set()
        prefill_request_not_put_copy = self.prefill_reqs_in_decode_batch.copy()
        prefill_request_not_put = set(prefill_request_not_put_copy)
        
        
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            if request.request_id in self.prefill_request_in_decode: 
                logger.info(f"skip request : {request.request_id} ; prefill_request_in_decode : {self.prefill_request_in_decode}")
                self.running.remove(request)
                self.scheduled_butnot_finished_pre_req.append(request)
                logger.info(
                    f"self.scheduled_butnot_finished_pre_req : {self.scheduled_butnot_finished_pre_req}"
                )
                continue
            
            

            num_new_tokens = (request.num_tokens_with_spec +
                              request.num_output_placeholders -
                              request.num_computed_tokens)
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - 1 - request.num_computed_tokens)

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_compute_budget
                 ) = self._try_schedule_encoder_inputs(
                     request, request.num_computed_tokens, num_new_tokens,
                     encoder_compute_budget)

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens)
                if new_blocks is None:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)
                    else:
                        preempted_req = self.running.pop()
                        if preempted_req.request_id in self.prefill_reqs_in_decode_batch:
                            self.prefill_request_not_put[preempted_req.request_id] += preempted_req.num_prompt_tokens
                            self.prefill_request_in_decode[preempted_req.request_id] += preempted_req.num_prompt_tokens

                    self.kv_cache_manager.free(preempted_req)
                    self.encoder_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                    self.waiting.prepend_request(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt.
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_running_reqs.append(request)
            if request.request_id in self.prefill_reqs_in_decode_batch:
                prefill_reqs[request.request_id] = num_new_tokens #如果是migrate，先加到临时队列中，等调度完成计算所有token，然后判断是否要分离batch
            req_to_new_blocks[request.request_id] = new_blocks
            if request.request_id in prefill_request_not_put:
                prefill_req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            logger.debug(f"Scheduled RUNNING request {request.request_id} , num_new_tokens: {num_new_tokens}")
            token_budget -= num_new_tokens
            if not self.is_prefill:
                if request.request_id in self.prefill_request_in_decode:
                    if request.request_id in finished_prefill_reqs:
                        self.prefill_request_in_decode.pop(request.request_id)
                if request.request_id in self.prefill_request_not_put:
                    self.prefill_request_not_put[request.request_id] -= num_new_tokens
                    logger.info(f"prefill RUNNING request {request.request_id} in decode , num_new_tokens: {num_new_tokens} , remain prompt tokens : {self.prefill_request_not_put[request.request_id]}")
                    if self.prefill_request_not_put[request.request_id] <=0:
                        self.prefill_request_not_put.pop(request.request_id)
                        if request.request_id in self.prefill_reqs_in_decode_batch:
                            self.prefill_reqs_in_decode_batch.remove(request.request_id)
                        
            req_index += 1

            # Speculative decode related.
            if request.request_id in prefill_request_not_put: 
                if request.spec_token_ids:
                    num_scheduled_spec_tokens = (num_new_tokens +
                                                 request.num_computed_tokens -
                                                 request.num_tokens)
                    if num_scheduled_spec_tokens > 0:
                        # Trim spec_token_ids list to num_scheduled_spec_tokens.
                        prefill_scheduled_spec_decode_tokens[request.request_id] = (
                            request.spec_token_ids)
                        
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (num_new_tokens +
                                             request.num_computed_tokens -
                                             request.num_tokens)
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids)

            # Encoder-related.
            if encoder_inputs_to_schedule:
                if request.request_id in prefill_request_not_put: 
                    prefill_scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0)
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)

        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()       

                # KVTransfer: skip request if still waiting for remote kvs.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (self.lora_config and request.lora_request and
                    (len(scheduled_loras) == self.lora_config.max_loras and
                     request.lora_request.lora_int_id not in scheduled_loras)):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False

                logger.debug(f"request.num_computed_tokens: {request.num_computed_tokens} ")
                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = \
                        self.kv_cache_manager.get_computed_blocks(
                            request)

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        num_external_computed_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens( #use num_external_computed_tokens to judge prefill request, when = 0 , is prefill request
                                request, num_new_local_computed_tokens))
                        logger.debug(f"request_id : {request.request_id} ; num_external_computed_tokens: {num_external_computed_tokens} ; load_kv_async: {load_kv_async} ; new_local_computed_tokens : {num_new_local_computed_tokens}")
                        if num_external_computed_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                    # Total computed tokens (local + external).
                    num_computed_tokens = (num_new_local_computed_tokens +
                                           num_external_computed_tokens)
                    
                    logger.debug(f"request_id : {request.request_id}, num_computed_tokens: {num_computed_tokens} ")
                            
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed.
                else:
                    new_computed_blocks = (
                        self.kv_cache_manager.create_empty_block_list())
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                new_encoder_compute_budget = encoder_compute_budget

                # KVTransfer: loading remote KV, do not allocate for new work.  
                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                # Number of tokens to be scheduled.
                else:
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    if (0 < self.scheduler_config.long_prefill_token_threshold
                            < num_new_tokens):
                        num_new_tokens = (
                            self.scheduler_config.long_prefill_token_threshold)

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if not self.scheduler_config.chunked_prefill_enabled and \
                        num_new_tokens > token_budget:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (encoder_inputs_to_schedule, num_new_tokens,
                         new_encoder_compute_budget
                         ) = self._try_schedule_encoder_inputs(
                             request, num_computed_tokens, num_new_tokens,
                             encoder_compute_budget)
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (0 if request.num_computed_tokens
                                              == 0 else
                                              self.num_lookahead_tokens)

                # Determine if we need to allocate cross-attention blocks.
                if self.is_encoder_decoder and request.has_encoder_inputs:
                    # TODO(russellb): For Whisper, we know that the input is
                    # always padded to the maximum length. If we support other
                    # encoder-decoder models, this will need to be updated if we
                    # want to only allocate what is needed.
                    num_encoder_tokens =\
                        self.scheduler_config.max_num_encoder_input_tokens
                else:
                    num_encoder_tokens = 0

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_computed_tokens,
                    num_new_local_computed_tokens,
                    new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        new_computed_blocks + new_blocks,
                        num_external_computed_tokens,
                    )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = self.waiting.pop_request()
                logger.info(f"Scheduling WAITING request {request.request_id} with num_new_tokens: {num_new_tokens} and num_external_computed_tokens: {num_external_computed_tokens}")
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                req_index += 1
                self.running.append(request)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    if request.request_id in prefill_request_not_put: 
                        prefill_scheduled_new_reqs.append(request)
                        prefill_reqs[request.request_id] = num_new_tokens
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    logger.info(f"add preempted request {request.request_id} to resumed_reqs")
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id))
                if request.request_id in prefill_request_not_put: 
                    prefill_req_to_new_blocks[request.request_id] = (
                        self.kv_cache_manager.get_blocks(request.request_id))
                num_scheduled_tokens[request.request_id] = num_new_tokens
                    
                token_budget -= num_new_tokens
                if not self.is_prefill:
                    if request.request_id in self.prefill_request_in_decode:
                        if request.request_id in finished_prefill_reqs:
                            self.prefill_request_in_decode.pop(request.request_id)
                    if request.request_id in self.prefill_request_not_put:
                        self.prefill_request_not_put[request.request_id] -= num_new_tokens
                        logger.info(f"prefill WAITING request {request.request_id} in decode , num_new_tokens: {num_new_tokens} , remain prompt tokens : {self.prefill_request_not_put[request.request_id]}")
                        if self.prefill_request_not_put[request.request_id] <=0:
                            self.prefill_request_not_put.pop(request.request_id)
                            if request.request_id in self.prefill_reqs_in_decode_batch:
                                self.prefill_reqs_in_decode_batch.remove(request.request_id)
                    
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    if request.request_id in prefill_request_not_put: 
                        prefill_scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                    
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
        else:
            logger.info(f"preemption requests : {[req.request_id for req in preempted_reqs]}")

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        logger.debug(
            f"test self.prefill_request_not_put : {self.prefill_request_not_put}"
        )
        for req_id in prefill_request_not_put: 
            if req_id in num_scheduled_tokens:
                prefill_num_scheduled_tokens[req_id] = num_scheduled_tokens[req_id]
        total_prefill_num_scheduled_tokens = sum(prefill_num_scheduled_tokens.values())
        
        reqs_prefill_batch:dict[str, int] = {}
        if self.use_two_batch:
            if total_prefill_num_scheduled_tokens >  self.split_threshold :   #判断是否分成两个batch
                self.prefill_request_in_decode.update(prefill_reqs)
                reqs_prefill_batch.update(prefill_reqs)
            else:
                total_prefill_num_scheduled_tokens = 0
        else:
            total_prefill_num_scheduled_tokens = 0
        
        logger.debug(f"prefill_request_in_decode : {self.prefill_request_in_decode}, prefill_reqs : {prefill_reqs}")
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens, f"total_num_scheduled_tokens: {num_scheduled_tokens.values()} exceeds max_num_scheduled_tokens: {self.max_num_scheduled_tokens}"
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) <= len(self.running))

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(
            self.kv_cache_config.kv_cache_groups)
        prefill_num_common_prefix_blocks = [0] * len(
            self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            logger.debug(f"any_request_id : {any_request.request_id} ; prefill_request_not_put : {prefill_request_not_put}") 
            if any_request.request_id in prefill_request_not_put: 
                prefill_num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(
                        any_request, len(self.running)))
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids())
            for req in scheduled_new_reqs
        ]
        prefill_new_reqs_data = []
        prefill_new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids())
            for req in prefill_scheduled_new_reqs
        ]
        
        prefill_scheduled_running_reqs = []
        for req in prefill_request_not_put : 
            if req in scheduled_running_reqs :
                prefill_scheduled_running_reqs.append(req) 
        
        prefill_scheduled_resumed_reqs = [] 
        for req in prefill_request_not_put :  
            if req in scheduled_resumed_reqs :
                prefill_scheduled_resumed_reqs.append(req)  
                
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_blocks,
        )
        prefill_cached_reqs_data = self._make_cached_request_data(
            prefill_scheduled_running_reqs,
            prefill_scheduled_resumed_reqs,
            prefill_num_scheduled_tokens,
            prefill_scheduled_spec_decode_tokens,
            prefill_req_to_new_blocks,
        )
        prefill_finish_req_ids = set()
        for req_id in self.finished_req_ids:
            if req_id in prefill_request_not_put: 
                prefill_finish_req_ids.add(req_id)
        self.finished_req_ids = self.finished_req_ids
        self.prefill_finished_req_ids = prefill_finish_req_ids
        scheduled_requests = (scheduled_new_reqs + scheduled_running_reqs +
                              scheduled_resumed_reqs)
        prefill_scheduled_requests = (prefill_scheduled_new_reqs + prefill_scheduled_running_reqs +
                                     prefill_scheduled_resumed_reqs)
        structured_output_request_ids, grammar_bitmask = (
            self.get_grammar_bitmask(scheduled_requests,
                                     scheduled_spec_decode_tokens))
        prefill_structured_output_request_ids, prefill_grammar_bitmask = (
            self.get_grammar_bitmask(prefill_scheduled_requests,
                                     prefill_scheduled_spec_decode_tokens))
        
        
        #有prefill req时，chunk_size会变为16*256=4096 
        if total_prefill_num_scheduled_tokens == 0:
            prefill_new_reqs_data = []
            prefill_cached_reqs_data = []
            prefill_num_scheduled_tokens = {}
            total_prefill_num_scheduled_tokens = 0
            prefill_scheduled_spec_decode_tokens = {}
            prefill_scheduled_encoder_inputs = {}
            prefill_num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
            prefill_finish_req_ids = set()
            prefill_structured_output_request_ids = set()
            prefill_grammar_bitmask = [0] * len(self.kv_cache_config.kv_cache_groups)
            prefill_request_ids = set()
            prefill_request_not_put = set()
            self.prefill_reqs_in_decode_batch = [
                req for req in self.prefill_reqs_in_decode_batch
                if req not in prefill_reqs
            ] 
        scheduler_output = SchedulerOutput(  #前半部分包含所有req信息，后半部分只包含prefill req信息 
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens, 
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
            
            free_encoder_mm_hashes=self.encoder_cache_manager.
            get_freed_mm_hashes(),
            
            prefill_scheduled_new_reqs=prefill_new_reqs_data,
            prefill_scheduled_cached_reqs=prefill_cached_reqs_data,
            prefill_num_scheduled_tokens=prefill_num_scheduled_tokens,
            prefill_total_num_scheduled_tokens=total_prefill_num_scheduled_tokens,
            prefill_scheduled_spec_decode_tokens=prefill_scheduled_spec_decode_tokens,
            prefill_scheduled_encoder_inputs=prefill_scheduled_encoder_inputs,
            prefill_num_common_prefix_blocks=prefill_num_common_prefix_blocks,
            prefill_finished_req_ids=prefill_finish_req_ids,
            prefill_structured_output_request_ids=prefill_structured_output_request_ids,
            prefill_grammar_bitmask=prefill_grammar_bitmask,

            prefill_request_not_put=set(reqs_prefill_batch.keys()) 
        )
        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        self._update_after_schedule(scheduler_output)
        return scheduler_output

    def add_request(self, request: Request) -> None:
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)
        if not self.is_prefill :
            new_computed_blocks, num_new_local_computed_tokens = \
                self.kv_cache_manager.get_computed_blocks(
                    request)
            num_external_computed_tokens, _ = (
                            self.connector.get_num_new_matched_tokens( #use num_external_computed_tokens to judge prefill request, when = 0 , is prefill request
                                request, num_new_local_computed_tokens))
            if not self.is_prefill and num_external_computed_tokens == 0:
                logger.info(f"Add prefill request {request.request_id} to prefill_reqs_in_decode_batch and prefill_request_not_put with prompt tokens: {request.num_prompt_tokens}")
                self.prefill_request_not_put[request.request_id] = request.num_prompt_tokens
                self.prefill_reqs_in_decode_batch.append(request.request_id)
                if self.prefill_request_not_put or self.prefill_reqs_in_decode_batch:
                    self.max_num_scheduled_tokens = 8*self.scheduler_config.max_num_batched_tokens

    def _free_request(self, request: Request) -> Optional[dict[str, Any]]:
        assert request.is_finished()

        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        if not delay_free_blocks:
            self._free_blocks(request)
            return None

        return kv_xfer_params
    
    def update_params(self, update_request: UpdateRequest) -> None:
        self.use_two_batch = update_request.use_split if update_request.use_split is not None else self.use_two_batch
        self.split_threshold = update_request.split_trd if update_request.split_trd is not None else self.split_threshold
        logger.info(f"Updated scheduler params: USE_TWO_BATCH={self.use_two_batch}, split_threshold={self.split_threshold}")