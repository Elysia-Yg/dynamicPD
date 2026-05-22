from abc import abstractmethod

from vllm.engine.protocol import EngineClient

from dynamicPD.patching import dynamicPDPatch

class EngineClientPatch(dynamicPDPatch[EngineClient]):
    @abstractmethod
    async def stop_profile(self) -> None:
        """Stop profiling the engine"""
        ...
