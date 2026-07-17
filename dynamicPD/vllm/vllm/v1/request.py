import time
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import EngineCoreEvent
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash

from vllm.v1.request import Request, RequestStatus

from dynamicPD.patching import dynamicPDPatch

class RequestPatch(dynamicPDPatch[Request]):
    _orig_init = Request.__init__
    def __init__(
        self, *args, **kwargs
    ) -> None:
        self.to_migrate = kwargs.get("to_migrate", False)
        self._orig_init(*args, **kwargs)

