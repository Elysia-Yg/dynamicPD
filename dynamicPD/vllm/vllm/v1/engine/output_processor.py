from typing import Any, cast

import torch
import numpy as np

from vllm.outputs import (CompletionOutput, PoolingOutput,
                          PoolingRequestOutput, RequestOutput)
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import FinishReason


from vllm.v1.engine.output_processor import RequestState

from dynamicPD.patching import dynamicPDPatch

class RequestStatePatch(dynamicPDPatch[RequestState]):
    _orig_make_request_output = RequestState.make_request_output
    def make_request_output(
        self,
        new_token_ids: list[int],
        pooling_output: torch.Tensor | None,
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        kv_transfer_params: dict[str, Any] | None = None,
        routed_experts: np.ndarray | None = None,
    ) -> RequestOutput | PoolingRequestOutput | None:
        is_migrate = stop_reason == "migrate"
        if is_migrate:
            kv_transfer_params = kv_transfer_params or {}
            kv_transfer_params["migrate"] = True
        return self._orig_make_request_output(
            new_token_ids=new_token_ids,
            pooling_output=pooling_output,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
            kv_transfer_params=kv_transfer_params,
            routed_experts=routed_experts,
        )

    def _new_request_output(
        self,
        external_req_id: str,
        outputs: list[CompletionOutput] | list[PoolingOutput],
        finished: bool,
        kv_transfer_params: dict[str, Any] | None = None,
        is_migrate: bool | None = False,
    ) -> RequestOutput | PoolingRequestOutput:
        # If prompt embeds were used, put placeholder prompt token ids
        prompt_token_ids = self.prompt_token_ids
        if prompt_token_ids is None and self.prompt_embeds is not None:
            prompt_token_ids = [0] * len(self.prompt_embeds)
        assert prompt_token_ids is not None

        first_output = outputs[0]
        if isinstance(first_output, PoolingOutput):
            assert len(outputs) == 1
            return PoolingRequestOutput(
                request_id=external_req_id,
                outputs=first_output,
                num_cached_tokens=self.num_cached_tokens,
                prompt_token_ids=prompt_token_ids,
                finished=finished,
            )
        assert self.logprobs_processor is not None
        if self.output_kind == RequestOutputKind.DELTA:
            # Side effect: logprobs processor forgets prompt logprobs
            prompt_logprobs = self.logprobs_processor.pop_prompt_logprobs()
        else:
            prompt_logprobs = self.logprobs_processor.prompt_logprobs

        return RequestOutput(
            request_id=external_req_id,  # request_id is what was provided externally
            lora_request=self.lora_request,
            prompt=self.prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=prompt_logprobs,
            outputs=cast(list[CompletionOutput], outputs),
            finished=finished,
            kv_transfer_params=kv_transfer_params,
            num_cached_tokens=self.num_cached_tokens,
            metrics=self.stats,
            migrate=kv_transfer_params.get("migrate", False) if kv_transfer_params else False
        )