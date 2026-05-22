from typing import Any, Optional

from vllm.logger import logger
from vllm.logprobs import PromptLogprobs
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalPlaceholderDict
from vllm.sequence import RequestMetrics


from vllm.outputs import CompletionOutput, RequestOutput

from dynamicPD.patching import dynamicPDPatch

class RequestOutputPatch(dynamicPDPatch[RequestOutput]):
    def __init__(
        self,
        request_id: str,
        prompt: Optional[str],
        prompt_token_ids: Optional[list[int]],
        prompt_logprobs: Optional[PromptLogprobs],
        outputs: list[CompletionOutput],
        finished: bool,
        metrics: Optional[RequestMetrics] = None,
        lora_request: Optional[LoRARequest] = None,
        encoder_prompt: Optional[str] = None,
        encoder_prompt_token_ids: Optional[list[int]] = None,
        num_cached_tokens: Optional[int] = None,
        migrate: Optional[bool] = False,
        *,
        multi_modal_placeholders: Optional[MultiModalPlaceholderDict] = None,
        kv_transfer_params: Optional[dict[str, Any]] = None,
        # Forward compatibility, code that uses args added in new release can
        # still run with older versions of vLLM without breaking.
        **kwargs: Any,
    ) -> None:
        if kwargs:
            logger.warning_once("RequestOutput: Ignoring extra arguments: %s",
                                str(kwargs))
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.multi_modal_placeholders = multi_modal_placeholders or {}
        self.prompt_logprobs = prompt_logprobs
        self.outputs = outputs
        self.finished = finished
        self.metrics = metrics
        self.lora_request = lora_request
        self.encoder_prompt = encoder_prompt
        self.encoder_prompt_token_ids = encoder_prompt_token_ids
        self.num_cached_tokens = num_cached_tokens
        self.kv_transfer_params = kv_transfer_params
        self.migrate = migrate