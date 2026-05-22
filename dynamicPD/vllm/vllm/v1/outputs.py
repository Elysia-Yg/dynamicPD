from dataclasses import field
from typing import Optional

from vllm.v1.outputs import ModelRunnerOutput

from dynamicPD.patching import dynamicPDPatch

class ModelRunnerOutputPatch(dynamicPDPatch[ModelRunnerOutput]):
    is_merged: Optional[bool] = False
    
    finished_prefill_reqs : set[str] = field(default_factory=set)