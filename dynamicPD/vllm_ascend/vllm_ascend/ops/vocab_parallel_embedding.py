from typing import Optional

import torch

from vllm_ascend.distributed.parallel_state import get_lmhead_tp_group
from vllm_ascend.utils import lmhead_tp_enable


from vllm_ascend.ops.vocab_parallel_embedding import AscendLogitsProcessor, AscendParallelLMHead

from dynamicPD.patching import dynamicPDPatch

class AscendLogitsProcessorPatch(dynamicPDPatch[AscendLogitsProcessor]):
    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: AscendParallelLMHead,
        embedding_bias: Optional[torch.Tensor] = None,
        use_offload: bool = False,
    ) -> Optional[torch.Tensor]:
        if lmhead_tp_enable():
            return self._get_logits_lmheadtp(hidden_states, lm_head,
                                             embedding_bias)
        else:
            return self._get_logits_normal(hidden_states, lm_head,
                                           embedding_bias, use_offload=use_offload)

    def _get_logits_lmheadtp(
        self,
        hidden_states: torch.Tensor,
        lm_head: AscendParallelLMHead,
        embedding_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        # Gather hidden states from all devices in tensor parallel group
        gathered_hidden_states = get_lmhead_tp_group().all_gather(
            hidden_states, dim=0)
        local_logits = lm_head.quant_method.apply(lm_head,
                                                  gathered_hidden_states,
                                                  bias=embedding_bias)
        # Gather logits for tensor parallel
        logits = get_lmhead_tp_group().all_to_all(local_logits)
        # Remove paddings in vocab (if any)
        if logits is not None:
            logits = logits[..., :self.org_vocab_size]
        return logits

    def _get_logits_normal(
        self,
        hidden_states: torch.Tensor,
        lm_head: AscendParallelLMHead,
        embedding_bias: Optional[torch.Tensor],
        use_offload: bool = False,
    ) -> Optional[torch.Tensor]:
        local_logits = lm_head.quant_method.apply(lm_head,
                                                  hidden_states,
                                                  bias=embedding_bias)
        # Gather logits for tensor parallel
        logits = self._gather_logits(local_logits, use_offload=use_offload)

        # Remove paddings in vocab (if any)
        if logits is not None:
            logits = logits[..., :self.org_vocab_size]

        return logits