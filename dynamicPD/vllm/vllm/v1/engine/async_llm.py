from vllm.v1.engine.async_llm import AsyncLLM

from dynamicPD.patching import dynamicPDPatch
from dynamicPD.vllm.vllm.entrypoints.openai.protocol import UpdateRequest

class AsyncLLMPatch(dynamicPDPatch[AsyncLLM]):

    async def stop_profile_npu(self) -> None:
        await self.engine_core.profile_npu_async()
        
    async def update_params(self, update_request: UpdateRequest) -> None:
        await self.engine_core.update_params_async(update_request)