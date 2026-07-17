import os
import time
from collections.abc import Callable
from typing import Any

from vllm.config import VllmConfig
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.engine.core import EngineCore, EngineCoreProc, logger
from vllm.v1.executor import Executor
from vllm.v1.request import Request, RequestStatus


from dynamicPD.patching import dynamicPDPatch
from dynamicPD.vllm.vllm.v1.core.sched.utils import check_stop
from dynamicPD.vllm.vllm.v1.engine.pd_coordinator import (
    PDCoordinator,
    PDLoadClient,
    PDLoadReporter,
)


def _get_dynamic_pd_config(vllm_config: VllmConfig) -> dict[str, Any]:
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None:
        return {}
    return kv_transfer_config.get_from_extra_config("dynamic_pd_config", {}) or {}


def _get_pd_role(vllm_config: VllmConfig, config: dict[str, Any]) -> str:
    if role := config.get("role"):
        return role
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None:
        return "decode"
    is_producer = kv_transfer_config.is_kv_producer
    is_consumer = kv_transfer_config.is_kv_consumer
    if is_producer and is_consumer:
        return "both"
    return "prefill" if is_producer else "decode"

def _scheduler_load(scheduler: SchedulerInterface) -> dict[str, int | float]:
    stats = scheduler.make_stats()
    running_reqs, waiting_reqs, kv_usage = stats.num_running_reqs, stats.num_waiting_reqs, stats.kv_cache_usage
    def _get_sum_request_tokens(requests: RequestQueue) -> int:
        return sum(req.num_tokens - req.num_computed_tokens for req in requests)
    waiting_tokens = _get_sum_request_tokens(scheduler.waiting)
    waiting_tokens += _get_sum_request_tokens(scheduler.skipped_waiting)
    return {
        "waiting_reqs": waiting_reqs,
        "running_reqs": running_reqs,
        "waiting_tokens": waiting_tokens,
        "prefill_tokens_in_decode": int(
            getattr(scheduler, "prefill_tokens_in_decode", 0) or 0
        ),
        "kv_cache_usage": kv_usage,
    }


class EngineCorePatch(dynamicPDPatch[EngineCore]):
    _orig_init = EngineCore.__init__
    _orig_preprocess_add_request = EngineCore.preprocess_add_request
    _orig_post_step = EngineCore.post_step
    _orig_shutdown = EngineCore.shutdown

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
        include_finished_set: bool = False,
    ):
        EngineCorePatch._orig_init(
            self,
            vllm_config,
            executor_class,
            log_stats,
            executor_fail_callback,
            include_finished_set,
        )
        self.dynamic_pd_config = _get_dynamic_pd_config(vllm_config)
        if not self.dynamic_pd_config:
            self.pd_coordinator = None
            logger.info("DynamicPD is disabled: no dynamic_pd_config found")
            return

        parallel_config = vllm_config.parallel_config
        self.pd_group = self.dynamic_pd_config.get("group", "default")
        self.pd_role = _get_pd_role(vllm_config, self.dynamic_pd_config)
        self.prefill_overload_trd = int(self.dynamic_pd_config.get("overload_trd", 1000))
        self.pd_decode_kv_watermark = float(
            self.dynamic_pd_config.get("decode_kv_watermark", 0.7)
        )
        self.pd_max_decode_offload_tokens = int(
            self.dynamic_pd_config.get("max_decode_offload_tokens", 2048)
        )
        self.pd_stale_ms = int(self.dynamic_pd_config.get("stale_ms", 0xffffffff))
        input_address = self.dynamic_pd_config.get(
            "coordinator_input_address", "tcp://127.0.0.1:16666"
        )
        publish_address = self.dynamic_pd_config.get(
            "coordinator_publish_address", "tcp://127.0.0.1:16667"
        )
        
        if self.pd_role == "prefill":
            self.pd_coordinator = PDCoordinator(
                input_address,
                publish_address,
                stale_ms=self.pd_stale_ms,
                min_publish_interval_ms=int(
                    self.dynamic_pd_config.get("publish_interval_ms", 100)
                ),
            )
        engine_index = int(getattr(self, "engine_index", 0))
        instance_id = self.dynamic_pd_config.get(
            "instance_id",
            (
                f"{os.uname().nodename}:"
                f"{os.getpid()}:"
                f"{parallel_config.data_parallel_rank}:"
                f"{engine_index}:"
                f"{self.pd_role}"
            ),
        )
        self.pd_reporter = PDLoadReporter(
            input_address=input_address,
            instance_id=instance_id,
            role=self.pd_role,
            group=self.pd_group,
            dp_rank=parallel_config.data_parallel_rank,
            engine_index=engine_index,
        )
        self.pd_load_client = PDLoadClient(
            publish_address=publish_address,
            stale_ms=self.pd_stale_ms,
        )
        self._pd_publish_load()
        logger.info(f"DynamicPD initialized: role={self.pd_role}, group={self.pd_group}, instance_id={instance_id}")

    @property
    def is_prefill(self) -> bool:
        kv_transfer_config = self.vllm_config.kv_transfer_config
        return kv_transfer_config is not None and kv_transfer_config.is_kv_producer
    
    def _cauculate_residual_running_time(self) -> float:
        time_now = time.perf_counter()
        predict_finish_time = 0.0
        for scheduler_output in self.scheduler.batch_infos:
            start_time = scheduler_output.time_stamp
            if start_time is not None:
                num_reqs = len(scheduler_output.num_scheduled_tokens.keys())
                _predict_finish_time = self._pd_predict_wait_ms(scheduler_output.total_num_scheduled_tokens, num_reqs) / 1000.0
                predict_finish_time = max(predict_finish_time, start_time + _predict_finish_time)
                logger.info("scheduler_output.time_stamp: %.6f, predict_finish_time: %.6f, tokens: %d", start_time, predict_finish_time, scheduler_output.total_num_scheduled_tokens)
            else:
                logger.warning("scheduler_output.time_stamp is None, cannot predict finish time")

        residual_running_time = max(0.0, predict_finish_time - time_now)
        return residual_running_time

    def _pd_publish_load(self) -> None:
        if not getattr(self, "dynamic_pd_config", False):
            logger.info("DynamicPD is disabled: no dynamic_pd_config found")
            return
        self.pd_reporter.publish(**_scheduler_load(self.scheduler))

    def _pd_should_migrate(self, request: Request) -> bool:
        if not getattr(self, "dynamic_pd_config", False):
            return False
        if self.pd_role not in ("prefill", "both"):
            return False

        load = _scheduler_load(self.scheduler)
        waiting_tokens = load["waiting_tokens"]
        waiting_reqs = load["waiting_reqs"]
        waiting_tokens = int(waiting_tokens) + request.num_tokens
        residual_running_time = self._cauculate_residual_running_time() * 1000.0
        predicted_wait_ms = self._pd_predict_wait_ms(waiting_tokens, waiting_reqs) + residual_running_time
        if predicted_wait_ms <= self.prefill_overload_trd:
            logger.info(
                "DynamicPD migrate skipped for req %s, predicted_wait_ms=%d <= overload_trd=%d, residual_running_time=%d", 
                request.request_id, predicted_wait_ms, self.prefill_overload_trd, residual_running_time
            )
            return False

        decode = self.pd_load_client.poll().best_decode(
            self.pd_group,
            self.pd_decode_kv_watermark,
            self.pd_max_decode_offload_tokens,
        )
        if decode is None:
            logger.info("DynamicPD migrate skipped: no decode capacity available")
            return False
        logger.info(
            "DynamicPD migrating request %s to decode candidate %s "
            "(predicted_wait_ms=%d, residual_running_time=%d, prefill_waiting_tokens=%d, decode_waiting_tokens=%d, "
            "decode_prefill_tokens=%d, decode_kv=%.3f)",
            request.request_id,
            decode.instance_id,
            predicted_wait_ms,
            residual_running_time,
            waiting_tokens,
            decode.waiting_tokens,
            decode.prefill_tokens_in_decode,
            decode.kv_cache_usage,
        )
        return True

    def _pd_predict_wait_ms(self, waiting_tokens: int, waiting_reqs: int = 0) -> int:
        # Conservative default until model-specific calibration is supplied.
        token_cost_ms = float(self.dynamic_pd_config.get("prefill_token_cost_ms", 0.2))
        fixed_cost_ms = float(self.dynamic_pd_config.get("prefill_fixed_cost_ms", 80))
        return int(fixed_cost_ms + token_cost_ms * waiting_tokens + waiting_reqs * fixed_cost_ms)

    def _pd_emit_migrate_output(self, request: Request) -> None:
        request.to_migrate = True
        request.status = RequestStatus.FINISHED_MIGRATED
        check_stop(request, self.vllm_config.model_config.max_model_len)
        output = EngineCoreOutput(
            request_id=request.request_id,
            new_token_ids=[],
            trace_headers=request.trace_headers,
            finish_reason=FinishReason.MIGRATE,
            stop_reason=request.stop_reason,
        )
        outputs = EngineCoreOutputs(
            outputs=[output],
            migrated_requests={request.request_id},
        )
        self.output_queue.put_nowait((request.client_index, outputs))

    def preprocess_add_request(self, request):
        req, request_wave = EngineCorePatch._orig_preprocess_add_request(self, request)
        req.to_migrate = False
        if self._pd_should_migrate(req):
            self._pd_emit_migrate_output(req)
        logger.info("add_request: request_id=%s, to_migrate=%s", req.request_id, req.to_migrate)
        return req, request_wave

    def post_step(self, model_executed: bool) -> None:
        EngineCorePatch._orig_post_step(self, model_executed)
        self._pd_publish_load()

    def shutdown(self):
        if getattr(self, "dynamic_pd_config", False):
            self.pd_reporter.close()
            self.pd_load_client.close()
            if self.pd_coordinator is not None:
                self.pd_coordinator.shutdown()
        return EngineCorePatch._orig_shutdown(self)


class EngineCoreProcPatch(dynamicPDPatch[EngineCoreProc]):
    _orig_run_engine_core = EngineCoreProc.run_engine_core
    _orig_handle_client_request = EngineCoreProc._handle_client_request

    @staticmethod
    def run_engine_core(*args, **kwargs):
        import vllm

        vllm.plugins.load_general_plugins()
        return EngineCoreProcPatch._orig_run_engine_core(*args, **kwargs)

    def _handle_client_request(self, request_type, request: Any) -> None:
        if request_type.name == "ADD":
            req, _ = request
            if getattr(req, "to_migrate", False):
                return
        return EngineCoreProcPatch._orig_handle_client_request(
            self, request_type, request
        )
