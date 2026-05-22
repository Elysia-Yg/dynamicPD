from vllm.v1.engine.core_client import AsyncMPClient

from dynamicPD.patching import dynamicPDPatch
from dynamicPD.vllm.vllm.entrypoints.openai.protocol import UpdateRequest

class AsyncMPClientPatch(dynamicPDPatch[AsyncMPClient]):

    async def profile_npu_async(self, is_start: bool = True) -> None:
        await self.call_utility_async("profile_npu", is_start)
        
    async def update_params_async(self, update_request: UpdateRequest) -> None:
        await self.call_utility_async("update_params", update_request)