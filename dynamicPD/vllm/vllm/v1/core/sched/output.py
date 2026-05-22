from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorMetadata)
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import logger

from dynamicPD.patching import dynamicPDPatch

class SchedulerOutputPatch(dynamicPDPatch[SchedulerOutput]):
    prefill_scheduled_new_reqs: list[NewRequestData]
    prefill_scheduled_cached_reqs: CachedRequestData
    prefill_num_scheduled_tokens: dict[str, int]
    prefill_total_num_scheduled_tokens: int
    prefill_scheduled_spec_decode_tokens: dict[str, list[int]]
    prefill_scheduled_encoder_inputs: dict[str, list[int]]
    prefill_num_common_prefix_blocks: list[int]
    prefill_finished_req_ids: set[str]
    prefill_structured_output_request_ids: dict[str, int]
    prefill_grammar_bitmask: Optional[npt.NDArray[np.int32]]
    # KV Cache Connector metadata.
    kv_connector_metadata: Optional[KVConnectorMetadata] = None

    prefill_request_ids: set[str] = None #用来记录还未完成的prefill_req，判断是否能够加入input_batch
    prefill_request_not_put: set[str] = None #用来记录还没加入过prefill_input_batch的req，判断是否该加入prefill_input_batch
    _original_init = SchedulerOutput.__init__
    def __init__(self, *args, **kwargs):
        self.prefill_scheduled_new_reqs = kwargs.pop("prefill_scheduled_new_reqs", [])
        self.prefill_scheduled_cached_reqs = kwargs.pop("prefill_scheduled_cached_reqs", CachedRequestData.make_empty())
        self.prefill_num_scheduled_tokens = kwargs.pop("prefill_num_scheduled_tokens", {})
        self.prefill_total_num_scheduled_tokens = kwargs.pop("prefill_total_num_scheduled_tokens", 0)
        self.prefill_scheduled_spec_decode_tokens = kwargs.pop("prefill_scheduled_spec_decode_tokens", {})
        self.prefill_scheduled_encoder_inputs = kwargs.pop("prefill_scheduled_encoder_inputs", {})
        self.prefill_num_common_prefix_blocks = kwargs.pop("prefill_num_common_prefix_blocks", [])
        self.prefill_finished_req_ids = kwargs.pop("prefill_finished_req_ids", set())
        self.prefill_structured_output_request_ids = kwargs.pop("prefill_structured_output_request_ids", {})
        self.prefill_grammar_bitmask = kwargs.pop("prefill_grammar_bitmask", None)
        self.prefill_request_ids = kwargs.pop("prefill_request_ids", set())
        self.prefill_request_not_put = kwargs.pop("prefill_request_not_put", set())
        self._original_init(*args, **kwargs)