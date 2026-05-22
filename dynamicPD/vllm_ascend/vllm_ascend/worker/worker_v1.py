


from vllm_ascend.worker.worker_v1 import NPUWorker

from dynamicPD.patching import dynamicPDPatch
from dynamicPD.vllm.vllm.entrypoints.openai.protocol import UpdateRequest

class NPUWorkerPatch(dynamicPDPatch[NPUWorker]):

    def profile_npu(self, is_start: bool = True) -> None:
        self.model_runner.profile_npu(is_start)
        
    def update_params(self, update_request: UpdateRequest) -> None:
        self.model_runner.update_params(update_request)