import multiprocessing
import time
import weakref
from dataclasses import dataclass, field
from typing import Any

import msgspec
import zmq

from vllm.v1.engine.core import logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import get_mp_context, set_process_title
from vllm.v1.utils import shutdown

class PDEngineLoad(msgspec.Struct, omit_defaults=True):
    instance_id: str
    role: str
    group: str = "default"
    dp_rank: int = 0
    engine_index: int = 0
    waiting_reqs: int = 0
    running_reqs: int = 0
    waiting_tokens: int = 0
    running_tokens: int = 0
    prefill_tokens_in_decode: int = 0
    kv_cache_usage: float = 0.0
    ts_ms: int = 0


@dataclass
class PDLoadView:
    engines: dict[str, PDEngineLoad] = field(default_factory=dict)

    def update(self, loads: list[PDEngineLoad], stale_ms: int) -> None:
        now = int(time.time() * 1000)
        self.engines = {
            load.instance_id: load
            for load in loads
            if now - load.ts_ms <= stale_ms
        }

    def best_decode(
        self,
        group: str,
        kv_watermark: float,
        max_decode_offload_tokens: int,
    ) -> PDEngineLoad | None:
        candidates = []
        for load in self.engines.values():
            if load.group != group or load.role not in ("decode", "both"):
                continue
            if load.kv_cache_usage >= kv_watermark:
                continue
            if load.prefill_tokens_in_decode >= max_decode_offload_tokens:
                continue
            candidates.append(load)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda load: (
                load.prefill_tokens_in_decode,
                load.waiting_tokens + load.running_tokens,
                load.waiting_reqs * 4 + load.running_reqs,
                load.kv_cache_usage,
            ),
        )


class PDLoadReporter:
    def __init__(
        self,
        input_address: str,
        instance_id: str,
        role: str,
        group: str,
        dp_rank: int,
        engine_index: int,
    ):
        self.ctx = zmq.Context()
        self.socket = make_zmq_socket(
            self.ctx, input_address, zmq.PUSH, bind=False, linger=0
        )
        self.instance_id = instance_id
        self.role = role
        self.group = group
        self.dp_rank = dp_rank
        self.engine_index = engine_index

    def publish(self, **kwargs: Any) -> None:
        load = PDEngineLoad(
            instance_id=self.instance_id,
            role=self.role,
            group=self.group,
            dp_rank=self.dp_rank,
            engine_index=self.engine_index,
            ts_ms=int(time.time() * 1000),
            **kwargs,
        )
        try:
            self.socket.send(msgspec.msgpack.encode(load), flags=zmq.NOBLOCK)
        except zmq.Again:
            logger.debug("PD load reporter dropped update for %s", self.instance_id)

    def close(self) -> None:
        self.socket.close(linger=0)
        self.ctx.term()


class PDLoadClient:
    def __init__(self, publish_address: str, stale_ms: int = 2000):
        self.ctx = zmq.Context()
        self.socket = make_zmq_socket(
            self.ctx, publish_address, zmq.SUB, bind=False, linger=0
        )
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.stale_ms = stale_ms
        self.view = PDLoadView()

    def poll(self) -> PDLoadView:
        latest = None
        while True:
            try:
                latest = self.socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        if latest is not None:
            loads = msgspec.msgpack.decode(latest, type=list[PDEngineLoad])
            self.view.update(loads, self.stale_ms)
        return self.view

    def close(self) -> None:
        self.socket.close(linger=0)
        self.ctx.term()


class PDCoordinator:
    def __init__(
        self,
        input_address: str,
        publish_address: str,
        stale_ms: int = 2000,
        min_publish_interval_ms: int = 100,
    ):
        context = get_mp_context()
        self.proc: multiprocessing.Process = context.Process(
            target=PDCoordinatorProc.run_coordinator,
            name="DynamicPD_Coordinator",
            kwargs={
                "input_address": input_address,
                "publish_address": publish_address,
                "stale_ms": stale_ms,
                "min_publish_interval_ms": min_publish_interval_ms,
            },
            daemon=True,
        )
        self.proc.start()
        self._finalizer = weakref.finalize(self, shutdown, [self.proc])

    def shutdown(self, timeout: float | None = None) -> None:
        if self._finalizer.detach() is not None:
            shutdown([self.proc], timeout=timeout)


class PDCoordinatorProc:
    @staticmethod
    def run_coordinator(
        input_address: str,
        publish_address: str,
        stale_ms: int = 2000,
        min_publish_interval_ms: int = 100,
    ):
        set_process_title("DynamicPDCoordinator")
        ctx = zmq.Context()
        loads: dict[str, PDEngineLoad] = {}
        decoder = msgspec.msgpack.Decoder(PDEngineLoad)
        with (
            make_zmq_socket(ctx, input_address, zmq.PULL, bind=True) as input_socket,
            make_zmq_socket(ctx, publish_address, zmq.PUB, bind=True) as pub_socket,
        ):
            logger.info(
                "DynamicPD Coordinator started, input_address=%s, publish_address=%s",
                input_address,
                publish_address,
            )
            poller = zmq.Poller()
            poller.register(input_socket, zmq.POLLIN)
            last_publish_ms = 0
            while True:
                now = int(time.time() * 1000)
                wait_ms = max(0, min_publish_interval_ms - (now - last_publish_ms))
                events = dict(poller.poll(timeout=wait_ms))
                if input_socket in events:
                    load = decoder.decode(input_socket.recv())
                    loads[load.instance_id] = load
                    logger.info("DynamicPD Coordinator received load update: %s", load.instance_id)

                now = int(time.time() * 1000)
                if now - last_publish_ms >= min_publish_interval_ms:
                    active_loads = [
                        load for load in loads.values()
                        if now - load.ts_ms <= stale_ms
                    ]
                    # logger.info("DynamicPD Coordinator publishing load updates: %d", len(active_loads))
                    pub_socket.send(msgspec.msgpack.encode(active_loads))
                    last_publish_ms = now
