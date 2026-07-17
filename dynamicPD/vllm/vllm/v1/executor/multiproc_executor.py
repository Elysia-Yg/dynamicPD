from __future__ import annotations

from typing import Any
import time

from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc, logger
from vllm.v1.outputs import AsyncModelRunnerOutput

from dynamicPD.patching import dynamicPDPatch


class MultiprocExecutorPatch(dynamicPDPatch[MultiprocExecutor]):
    def profile_npu(self, is_start: bool = True) -> None:
        self.collective_rpc(
            "profile_npu",
            args=(is_start,),
            unique_reply_rank=self.output_rank,
        )


class WorkerProcPatch(dynamicPDPatch[WorkerProc]):
    def enqueue_output(self, output: Any) -> None:
        if isinstance(output, AsyncModelRunnerOutput):
            output_type = type(output).__name__
            t1 = time.perf_counter()
            output = output.get_output()
            t2 = time.perf_counter()
            logger.info(
                "%s get_output over:%s, Time taken: %.6f seconds",
                output_type,
                output.req_ids,
                t2 - t1,
            )

        if isinstance(output, Exception):
            result = (WorkerProc.ResponseStatus.FAILURE, str(output))
        else:
            result = (WorkerProc.ResponseStatus.SUCCESS, output)

        if (response_mq := self.worker_response_mq) is not None:
            response_mq.enqueue(result)

    def handle_output(self, output: Any):
        """Handles output from the worker. If async scheduling is enabled,
        it is passed to the async_output_busy_loop thread. Otherwise, it is
        enqueued directly to the worker_response_mq.
        """
        if self.use_async_scheduling:
            self.async_output_queue.put(output)
            if isinstance(output, AsyncModelRunnerOutput):
                logger.debug(
                    "%s enqueued to async_output_queue",
                    type(output).__name__,
                )
        else:
            self.enqueue_output(output)

    def async_output_busy_loop(self):
        """Entrypoint for the thread which handles outputs asynchronously."""
        while True:
            output = self.async_output_queue.get()
            self.enqueue_output(output)
