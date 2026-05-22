import os
import threading
import time
from collections import deque
from concurrent.futures import Future
from contextlib import ExitStack
from typing import Any, Callable, Optional

import zmq
import vllm

from dynamicPD.vllm.vllm.v1.core.sched.utils import check_stop
from vllm.config import VllmConfig

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import engine_receiver_cache_from_config
from vllm.utils import (get_hash_fn_by_name, make_zmq_socket,
                        resolve_obj_by_qualname)
from vllm.v1.core.kv_cache_utils import (BlockHash,
                                         get_request_block_hasher,
                                         init_none_hash)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler as V1Scheduler
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreRequest, EngineCoreOutput,
                            EngineCoreRequestType,
                            UtilityOutput, UtilityResult)
from vllm.v1.engine.utils import get_device_indices
from vllm.v1.engine.core import EngineCore, EngineCoreProc, logger
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.version import __version__ as VLLM_VERSION

from dynamicPD.patching import dynamicPDPatch
from dynamicPD.vllm.vllm.entrypoints.openai.protocol import UpdateRequest


SLO = 1000  # ms
USE_MIGRATE = True
TOKENS_IN_DECODE_THRESHOLD = 4096
DECODE_KV_CACHE_THRESHOLD = 0.7
decode_address = f"tcp://127.0.0.1:6666"
prefill_address = f"tcp://127.0.0.1:8888"

class EngineCorePatch(dynamicPDPatch[EngineCore]):
    def recv_from_decode(self):
        ctx = zmq.Context()
        input_socket = make_zmq_socket(ctx, decode_address, zmq.PULL, bind = True) 
        while True:
            busy_check = input_socket.recv()
            if busy_check:
                self.decode_is_busy = busy_check == b'\x01'
                logger.debug(f"recv is_busy from decode : {busy_check} , {self.decode_is_busy}")
            time.sleep(0.01)
                
    def get_model_params(self):  
        config = self.vllm_config
        model_name = config.model_config.model 
        tp_size = config.parallel_config.tensor_parallel_size 
        c = 0
        model_name = model_name.lower()     
        logger.info(f"model_name : {model_name}, tp_size : {tp_size}")   
        if "qwen3-32b" in model_name:  #T = c*n*n + a*n + b, only for burst
            if tp_size == 2:  
                a, b ,c = 0.147759, 109.788268, 0.000017
            elif tp_size == 4:  
                a, b = 0.110023, 177.269328
        elif "qwen3-8b" in model_name:  
            if tp_size == 1:  
                a, b ,c =  0.024287, 65.212887, 0.000012
            elif tp_size == 2:  
                a, b ,c = 0.018977, 85.453370, 0.000010
        elif "qwen2.5-14b" in model_name:  
            if tp_size == 1:  
                a, b ,c = 0.119724, 103.350059, 0.000002
            if tp_size == 2:  
                a, b ,c = 0.121550, 65.129719, 0.000001
        elif "qwen2.5-32b" in model_name:  
            if tp_size == 2:  
                # a, b = 0.215177, 105.916339
                a, b = 0.23, 110
        elif "qwen72b" in model_name:
            if tp_size == 4:
                a ,b = 0.2487, 89.1819
        else:
            a,b=0,0
                
        
        return a, b ,c
    
    def predict_waiting_time(self, total_waiting_tokens):
        a, b, c = self.get_model_params()
        return a * total_waiting_tokens + b + c * total_waiting_tokens * total_waiting_tokens + 100  # ms

    def __init__(self,
                 vllm_config: VllmConfig,
                 executor_class: type[Executor],
                 log_stats: bool,
                 executor_fail_callback: Optional[Callable] = None):
        self.slo = SLO
        self.use_migrate = USE_MIGRATE
        self.tokens_in_decode_threshold = TOKENS_IN_DECODE_THRESHOLD
        self.decode_kv_cache_threshold = DECODE_KV_CACHE_THRESHOLD

        # plugins need to be loaded at the engine/scheduler level too
        from vllm.plugins import load_general_plugins
        load_general_plugins()

        self.vllm_config = vllm_config
        logger.info("Initializing a V1 LLM engine (v%s) with config: %s",
                    VLLM_VERSION, vllm_config)
        
        if vllm_config.parallel_config.data_parallel_size ==1:
            self._set_visible_devices(vllm_config, 0)

        self.log_stats = log_stats

        # Setup Model.
        self.model_executor = executor_class(vllm_config)
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(
                executor_fail_callback)

        self.available_gpu_memory_for_kv_cache = -1

        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks, kv_cache_config = \
            self._initialize_kv_caches(vllm_config)

        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks
        self.collective_rpc("initialize_cache",
                            args=(num_gpu_blocks, num_cpu_blocks))

        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # Setup scheduler.
        if isinstance(vllm_config.scheduler_config.scheduler_cls, str):
            Scheduler = resolve_obj_by_qualname(
                vllm_config.scheduler_config.scheduler_cls)
        else:
            Scheduler = vllm_config.scheduler_config.scheduler_cls

        # This warning can be removed once the V1 Scheduler interface is
        # finalized and we can maintain support for scheduler classes that
        # implement it
        if Scheduler is not V1Scheduler:
            logger.warning(
                "Using configured V1 scheduler class %s. "
                "This scheduler interface is not public and "
                "compatibility may not be maintained.",
                vllm_config.scheduler_config.scheduler_cls)

        if len(kv_cache_config.kv_cache_groups) == 0:
            # Encoder models without KV cache don't support
            # chunked prefill. But do SSM models?
            logger.info("Disabling chunked prefill for model without KVCache")
            vllm_config.scheduler_config.chunked_prefill_enabled = False

        self.scheduler: V1Scheduler = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=vllm_config.parallel_config.data_parallel_size
            > 1,
            log_stats=self.log_stats,
        )
        self.use_spec_decode = vllm_config.speculative_config is not None
        if self.scheduler.connector is not None:  # type: ignore
            self.model_executor.init_kv_output_aggregator(
                self.scheduler.connector.get_finished_count())  # type: ignore

        self.mm_registry = mm_registry = MULTIMODAL_REGISTRY
        self.mm_receiver_cache = engine_receiver_cache_from_config(
            vllm_config, mm_registry)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: Optional[deque[tuple[Future[ModelRunnerOutput],
                                               SchedulerOutput]]] = None
        self.current_batch: Optional[deque[tuple[float, float, int]]] = None
        self.temp_tokens = 0
        if self.batch_queue_size > 1:
            logger.info("Batch queue is enabled with size %d",
                        self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)
            self.current_batch = deque(maxlen=self.batch_queue_size)

        self.request_block_hasher: Optional[Callable[[Request],
                                                     list[BlockHash]]] = None
        if (self.vllm_config.cache_config.enable_prefix_caching
                or self.scheduler.get_kv_connector() is not None):

            block_size = vllm_config.cache_config.block_size
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo)
            init_none_hash(caching_hash_fn)

            self.request_block_hasher = get_request_block_hasher(
                block_size, caching_hash_fn)

        self.step_fn = (self.step if self.batch_queue is None else
                        self.step_with_batch_queue)

        self.is_prefill = False
        if vllm_config.kv_transfer_config:
            self.is_prefill = vllm_config.kv_transfer_config.is_kv_producer
            
        logger.info(f"Is prefill: {self.is_prefill}")
        if self.is_prefill:
            self.decode_is_busy = False
            t = threading.Thread(target = self.recv_from_decode, args = ())
            t.daemon = True
            t.start()
            logger.info("Starting recv thread for prefill.")
            
        self.busy = False
        if not self.is_prefill:
            ctx = zmq.Context()
            self.output_socket = make_zmq_socket(ctx, decode_address, zmq.PUSH, bind = False)
            
        self.prefill_scheduler_output_list = []
        self.finished_prefill_reqs : set[str] = set()

    def _set_visible_devices(self, vllm_config: VllmConfig,
                             local_dp_rank: int):
        from vllm.platforms import current_platform
        if current_platform.is_xpu():
            logger.info("XPU platform detected, skipping setting visible devices.")
            pass
        else:
            device_control_env_var = current_platform.device_control_env_var
            logger.info("Setting %s for local data parallel rank %d",
                        device_control_env_var, local_dp_rank)
            self._set_cuda_visible_devices(vllm_config, local_dp_rank,
                                           device_control_env_var)

    def _set_cuda_visible_devices(self, vllm_config: VllmConfig,
                                  local_dp_rank: int,
                                  device_control_env_var: str):
        world_size = vllm_config.parallel_config.world_size
        # Set CUDA_VISIBLE_DEVICES or equivalent.
        try:
            value = get_device_indices(device_control_env_var, local_dp_rank,
                                       world_size)
            os.environ[device_control_env_var] = value
            logger.info("Set %s=%s for local data parallel rank %d",
                        device_control_env_var, value, local_dp_rank)
        except IndexError as e:
            raise Exception(
                f"Error setting {device_control_env_var}: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f"base value: \"{os.getenv(device_control_env_var)}\"") from e
        
    def step_with_batch_queue(
            self) -> tuple[Optional[dict[int, EngineCoreOutputs]], bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """
        batch_queue = self.batch_queue
        assert batch_queue is not None

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        start_time = time.perf_counter()
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule(self.finished_prefill_reqs)
            self.finished_prefill_reqs.clear()
            logger.debug(f"scheduler_output  {scheduler_output.num_scheduled_tokens}")
            # if scheduler_output.total_num_scheduled_tokens > 0 and self.is_prefill:
            if self.is_prefill:
                predict_time = self.predict_waiting_time(scheduler_output.total_num_scheduled_tokens)
                self.current_batch.appendleft((start_time, predict_time, scheduler_output.total_num_scheduled_tokens))  # start time, predict time
            future = self.model_executor.execute_model(scheduler_output,
                                                       non_block=True)
            logger.debug(f"scheduler_output before future : {scheduler_output.num_scheduled_tokens}")
            batch_queue.appendleft(
                (future, scheduler_output))  # type: ignore[arg-type]

            model_executed = scheduler_output.total_num_scheduled_tokens > 0
            if model_executed and len(batch_queue) < self.batch_queue_size \
                and not batch_queue[-1][0].done():
                # Don't block on next worker response unless the queue is full
                # or there are no more requests to schedule.
                return None, True

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False

        # Block until the next result is available.
        future, scheduler_output = batch_queue.pop()
        logger.debug(f"scheduler_output in future : {scheduler_output.num_scheduled_tokens}")
        model_output = self.execute_model_with_error_logging(
            lambda _: future.result(), scheduler_output)
        if self.is_prefill:
            start_time, predict_time, total_num_scheduled_tokens = self.current_batch.pop()
            logger.debug(f"num_tokens : {total_num_scheduled_tokens}; execute_time : {(time.perf_counter() - start_time)*1000}; predict_time : {predict_time} ms")

        logger.debug(f"model_output.is_merged : {model_output.is_merged}")

        logger.debug(f"scheduler_output after future : {scheduler_output.num_scheduled_tokens}")
        logger.debug(f"scheduler_output.prefill_finished_req_ids : {scheduler_output.prefill_finished_req_ids} ; prefill_num_scheduled_tokens : {scheduler_output.prefill_num_scheduled_tokens}")
        if scheduler_output.prefill_num_scheduled_tokens:
            logger.debug("store prefill scheduler output")
            prefill_scheduler_output = SchedulerOutput(
                scheduled_new_reqs=scheduler_output.prefill_scheduled_new_reqs,
                scheduled_cached_reqs=scheduler_output.prefill_scheduled_cached_reqs,
                num_scheduled_tokens=scheduler_output.prefill_num_scheduled_tokens,
                total_num_scheduled_tokens=scheduler_output.prefill_total_num_scheduled_tokens, 
                scheduled_spec_decode_tokens=scheduler_output.prefill_scheduled_spec_decode_tokens,
                scheduled_encoder_inputs=scheduler_output.prefill_scheduled_encoder_inputs,
                num_common_prefix_blocks=scheduler_output.prefill_num_common_prefix_blocks,
                finished_req_ids=scheduler_output.prefill_finished_req_ids,
                structured_output_request_ids=scheduler_output.prefill_structured_output_request_ids,
                grammar_bitmask=scheduler_output.prefill_grammar_bitmask,
                free_encoder_mm_hashes=None,
                prefill_scheduled_new_reqs=[],
                prefill_scheduled_cached_reqs=None,
                prefill_num_scheduled_tokens=None,
                prefill_total_num_scheduled_tokens=None,
                prefill_scheduled_spec_decode_tokens=None,
                prefill_scheduled_encoder_inputs=None,
                prefill_num_common_prefix_blocks=None,
                prefill_finished_req_ids=None,
                prefill_structured_output_request_ids=None,
                prefill_grammar_bitmask=None,
                prefill_request_ids=scheduler_output.prefill_request_ids,
            )
            self.prefill_scheduler_output_list.append(prefill_scheduler_output)
            scheduler_output.num_scheduled_tokens = {req_id: num - prefill_scheduler_output.num_scheduled_tokens.get(req_id, 0) for req_id, num in scheduler_output.num_scheduled_tokens.items() if num - prefill_scheduler_output.num_scheduled_tokens.get(req_id, 0) >0}
            scheduler_output.scheduled_spec_decode_tokens = {req_id: num - prefill_scheduler_output.scheduled_spec_decode_tokens.get(req_id, 0) for req_id, num in scheduler_output.scheduled_spec_decode_tokens.items() if num - prefill_scheduler_output.scheduled_spec_decode_tokens.get(req_id, 0) >0}
            logger.debug(f"num_scheduled_tokens : {scheduler_output.num_scheduled_tokens}, scheduled_spec_decode_tokens : {scheduler_output.scheduled_spec_decode_tokens}")
        
        if model_output.is_merged:
            logger.debug("merge prefill scheduler output")
            prefill_scheduler_output = self.prefill_scheduler_output_list.pop(0)
            self.finished_prefill_reqs.update(model_output.finished_prefill_reqs)
            logger.debug(f"model_output finished_prefill_reqs : {self.finished_prefill_reqs}")
            logger.debug(f"prefill_scheduler_output req_ids : {prefill_scheduler_output.scheduled_new_reqs} ; prefill_scheduler_output scheduled_cached_reqs : {prefill_scheduler_output.scheduled_cached_reqs}")
            scheduler_output.num_scheduled_tokens = {**scheduler_output.num_scheduled_tokens, **prefill_scheduler_output.num_scheduled_tokens}
            scheduler_output.scheduled_spec_decode_tokens = {**scheduler_output.scheduled_spec_decode_tokens, **prefill_scheduler_output.scheduled_spec_decode_tokens}
            

        
        logger.debug(f"update_from_output model_runner_output : {model_output}")
        logger.debug(f"update_from_output scheduler_output before update : {scheduler_output.num_scheduled_tokens}")
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output)
        
        if not self.is_prefill:
            logger.debug(f"kv_cache_usage : {self.scheduler.kv_cache_manager.usage}")
            if  self.scheduler.prefill_tokens_in_decode > self.tokens_in_decode_threshold or self.scheduler.kv_cache_manager.usage >= self.decode_kv_cache_threshold:
                if self.busy == False: #judge decode is busy or not
                    self.busy = True
                    self.output_socket.send(b'\x01')
                    logger.info(f"send {self.busy} to prefill")
            else:
                if self.busy == True:
                    self.busy = False
                    self.output_socket.send(b'\x00')
                    logger.info(f"send {self.busy} to prefill")

        return engine_core_outputs, model_executed
    
    def profile_npu(self, is_start: bool = True):
        self.model_executor.profile_npu(is_start)
        
    def update_params(self, update_request: UpdateRequest) -> None:
        self.slo = update_request.slo if update_request.slo is not None else self.slo
        self.use_migrate = update_request.use_migrate if update_request.use_migrate is not None else self.use_migrate
        self.tokens_in_decode_threshold = update_request.busy_threshold if update_request.busy_threshold is not None else self.tokens_in_decode_threshold
        logger.info(f"update_params : SLO : {self.slo} ms ; USE_MIGRATE : {self.use_migrate} ; tokens_in_decode_threshold : {self.tokens_in_decode_threshold}")
        self.scheduler.update_params(update_request)

    def preprocess_add_request(
            self, request: EngineCoreRequest) -> tuple[Request, int]:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        # Note on thread safety: no race condition.
        # `mm_receiver_cache` is reset at the end of LLMEngine init,
        # and will only be accessed in the input processing thread afterwards.
        if self.mm_receiver_cache is not None and request.mm_features:
            request.mm_features = (
                self.mm_receiver_cache.get_and_update_features(
                    request.mm_features))

        req = Request.from_engine_core_request(request,
                                               self.request_block_hasher)
        self.turn_to_migrate(req)
        if req.use_structured_output:
            # Note on thread safety: no race condition.
            # `grammar_init` is only invoked in input processing thread. For
            # `structured_output_manager`, each request is independent and
            # grammar compilation is async. Scheduler always checks grammar
            # compilation status before scheduling request.
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave
    
    def turn_to_migrate(self,request: Request):
        arrive_= time.perf_counter()
        logger.debug(f"request {request.request_id} arrive at engine at {arrive_}, max_token : {request.sampling_params.max_tokens}")
        req = request
        if self.is_prefill:
            total_waiting_tokens = sum(req.num_tokens for req in self.scheduler.waiting) + self.temp_tokens
            waiting_time = self.predict_waiting_time(total_waiting_tokens + req.num_tokens)
            waiting_time_sum = waiting_time  # ms
            predict_time = 0
            tokens_ = 0
            predict_time1, tokens_1 = 0, 0
            if self.current_batch and len(self.current_batch) > 0:
                start_time, predict_time, tokens_ = self.current_batch[-1]
                if len(self.current_batch) > 1:
                    start_time1, predict_time1, tokens_1 = self.current_batch[0] 
                elapsed_time = (time.perf_counter() - start_time) * 1000  # ms
                waiting_time_sum += max(0, predict_time + predict_time1 - elapsed_time)
                
            logger.info(f"req : {req.request_id}; waiting_time : {waiting_time} ms ; waiting_time_sum : {waiting_time_sum} ms ; total_waiting_tokens : {total_waiting_tokens}; temp_tokens : {self.temp_tokens}; predict_time : {predict_time + predict_time1} ms; tokens_ : {tokens_ + tokens_1} ")
            client_index = req.client_index
            logger.info(f"decode is busy : {self.decode_is_busy}")
            if not self.decode_is_busy and waiting_time_sum > self.slo and self.use_migrate:
                logger.info(f"request {req.request_id} is moved to decode")
                request.to_migrate = True
                request.status = RequestStatus.FINISHED_MIGRATED
                check_stop(req, self.vllm_config.scheduler_config.max_model_len)
                outputs: list[EngineCoreOutput] = []
                outputs.append(
                    EngineCoreOutput(
                        request_id=req.request_id,
                        new_token_ids=[],
                        trace_headers=req.trace_headers,
                        finish_reason=req.get_finished_reason(),
                        stop_reason=req.stop_reason,
                    ))
                
                engine_core_outputs = EngineCoreOutputs(
                    outputs=outputs)

                logger.debug(f"Request {req.request_id} being migrated.")
                # Include ids of requests that are being migrated.
                if engine_core_outputs.migrated_requests is None:
                    engine_core_outputs.migrated_requests = set()
                engine_core_outputs.migrated_requests.add(req.request_id)
                self.output_queue.put_nowait(
                    (client_index, engine_core_outputs))
        

class EngineCoreProcPatch(dynamicPDPatch[EngineCoreProc]):

    _orig_run_engine_core = EngineCoreProc.run_engine_core

    @staticmethod
    def run_engine_core(*args, **kwargs):
        # When starting the API server, it will spawn a new process to run the
        # EngineCore. We need to load the plugins in the new process before it
        # initializes the Executor.
        vllm.plugins.load_general_plugins()
        return EngineCoreProcPatch._orig_run_engine_core(*args, **kwargs)
    
    def _handle_client_request(self, request_type: EngineCoreRequestType,
                               request: Any) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            self.add_request(req, request_wave)
            self.temp_tokens -= req.num_tokens
            
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            output = UtilityOutput(call_id)
            try:
                method = getattr(self, method_name)
                result = method(*self._convert_msgspec_args(method, args))
                output.result = UtilityResult(result)
            except BaseException as e:
                logger.exception("Invocation of %s method failed", method_name)
                output.failure_message = (f"Call to {method_name} method"
                                          f" failed: {str(e)}")
            self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=output)))
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error("Unrecognized input request type encountered: %s",
                         request_type)

    def process_input_sockets(self, input_addresses: list[str],
                              coord_input_address: Optional[str],
                              identity: bytes, ready_event: threading.Event):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx,
                                    input_address,
                                    zmq.DEALER,
                                    identity=identity,
                                    bind=False))
                for input_address in input_addresses
            ]
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(ctx,
                                    coord_input_address,
                                    zmq.XSUB,
                                    identity=identity,
                                    bind=False))
                # Send subscription message to coordinator.
                coord_socket.send(b'\x01')

            # Register sockets with poller.
            poller = zmq.Poller()
            for input_socket in input_sockets:
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                input_socket.send(b'')
                poller.register(input_socket, zmq.POLLIN)

            if coord_socket is not None:
                # Wait for ready message from coordinator.
                assert coord_socket.recv() == b"READY"
                poller.register(coord_socket, zmq.POLLIN)

            ready_event.set()
            del ready_event
            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(
                        copy=False)
                    request_type = EngineCoreRequestType(
                        bytes(type_frame.buffer))

                    # Deserialize the request data.
                    if request_type == EngineCoreRequestType.ADD:
                        request = add_request_decoder.decode(data_frames)
                        request = self.preprocess_add_request(request)
                    else:
                        request = generic_decoder.decode(data_frames)

                    # Push to input queue for core busy loop.
                    if request_type != EngineCoreRequestType.ADD or not request[0].to_migrate:
                        self.temp_tokens += request[0].num_tokens if request_type == EngineCoreRequestType.ADD else 0
                        self.input_queue.put_nowait((request_type, request))