from typing import Any, Optional, Union, cast

import torch

from vllm.outputs import (CompletionOutput, PoolingOutput,
                          PoolingRequestOutput, RequestOutput)
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import FinishReason


from vllm.v1.engine.output_processor import RequestState

from dynamicPD.patching import dynamicPDPatch

class RequestStatePatch(dynamicPDPatch[RequestState]):
    def make_request_output(
        self,
        new_token_ids: list[int],
        pooling_output: Optional[torch.Tensor],
        finish_reason: Optional[FinishReason],
        stop_reason: Union[int, str, None],
        kv_transfer_params: Optional[dict[str, Any]] = None,
    ) -> Optional[Union[RequestOutput, PoolingRequestOutput]]:

        finished = finish_reason is not None
        final_only = self.output_kind == RequestOutputKind.FINAL_ONLY

        if not finished and final_only:
            # Only the final output is required in FINAL_ONLY mode.
            return None

        request_id = self.request_id
        if pooling_output is not None:
            return self._new_request_output(
                request_id, [self._new_pooling_output(pooling_output)],
                finished)

        output = self._new_completion_output(new_token_ids, finish_reason,
                                             stop_reason)

        if self.parent_req is None:
            outputs = [output]
        else:
            request_id, outputs, finished = self.parent_req.get_outputs(
                request_id, output)
            if not outputs:
                return None

        is_migrate = stop_reason == 'migrate'
        # print(f"Request {request_id} is_migrate: {is_migrate}")
        return self._new_request_output(request_id, outputs, finished,
                                        is_migrate, kv_transfer_params)

    def _new_request_output(
        self,
        request_id: str,
        outputs: Union[list[CompletionOutput], list[PoolingOutput]],
        finished: bool,
        is_migrate: Optional[bool] = False,
        kv_transfer_params: Optional[dict[str, Any]] = None,
    ) -> Union[RequestOutput, PoolingRequestOutput]:

        first_output = outputs[0]
        if isinstance(first_output, PoolingOutput):
            assert len(outputs) == 1
            # Prompt embeddings are currently not supported by pooling requests.
            assert self.prompt_token_ids is not None
            return PoolingRequestOutput(
                request_id=request_id,
                outputs=first_output,
                prompt_token_ids=self.prompt_token_ids,
                finished=finished,
            )
        assert self.logprobs_processor is not None
        if self.output_kind == RequestOutputKind.DELTA:
            # Side effect: logprobs processor forgets prompt logprobs
            prompt_logprobs = self.logprobs_processor.pop_prompt_logprobs()
        else:
            prompt_logprobs = self.logprobs_processor.prompt_logprobs

        # If prompt embeds were used, put placeholder prompt token ids
        prompt_token_ids = self.prompt_token_ids
        if prompt_token_ids is None and self.prompt_embeds is not None:
            prompt_token_ids = [0] * len(self.prompt_embeds)

        return RequestOutput(
            request_id=request_id,
            prompt=self.prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=prompt_logprobs,
            outputs=cast(list[CompletionOutput], outputs),
            finished=finished,
            kv_transfer_params=kv_transfer_params,
            num_cached_tokens=self.num_cached_tokens,
            migrate=is_migrate
        )