from typing import Optional

import torch

from vllm.distributed import (tensor_model_parallel_all_gather,tensor_model_parallel_gather)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)

from dynamicPD.patching import dynamicPDPatch
from dynamicPD.vllm.vllm.distributed.communication_op import tensor_model_parallel_all_gather_offload, tensor_model_parallel_gather_offload

class LogitsProcessorPatch(dynamicPDPatch[LogitsProcessor]):
    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: Optional[torch.Tensor] = None,
        use_offload: bool = False,
    ) -> Optional[torch.Tensor]:
        if self.logits_as_input:
            logits = hidden_states
        else:
            # Get the logits for the next tokens.
            # print(f"get_logits input hidden_states shape: {hidden_states.shape}")
            logits = self._get_logits(hidden_states, lm_head, embedding_bias, use_offload=use_offload)
        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = torch.tanh(logits)
                logits = logits * self.soft_cap

            if self.scale != 1.0:
                logits *= self.scale
        return logits

    def _gather_logits(self, logits: torch.Tensor, use_offload: bool = False) -> torch.Tensor:
        """gather/all-gather the logits tensor across model parallel group."""
        if self.use_all_gather:
            # Gather is not supported for some devices such as TPUs.
            # Use all-gather instead.
            # NOTE(woosuk): Here, the outputs of every device should not be None
            # because XLA requires strict SPMD among all devices. Every device
            # should execute the same operations after gathering the logits.
            if use_offload:
                logits = tensor_model_parallel_all_gather_offload(logits)
            else:
                logits = tensor_model_parallel_all_gather(logits)
        else:
            # None may be returned for rank > 0
            if use_offload:
                logits = tensor_model_parallel_gather_offload(logits)
            else:
                logits = tensor_model_parallel_gather(logits)
        return logits

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor],
        use_offload: bool = False
    ) -> Optional[torch.Tensor]:
        # Get the logits for the next tokens.
        logits = lm_head.quant_method.apply(lm_head,
                                            hidden_states,
                                            bias=embedding_bias)

        # Gather logits for TP
        logits = self._gather_logits(logits, use_offload=use_offload)

        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[..., :self.org_vocab_size]
        return logits

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s