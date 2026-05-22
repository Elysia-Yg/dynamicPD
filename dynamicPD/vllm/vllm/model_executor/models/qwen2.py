from collections.abc import Iterable
from typing import Optional

import torch
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.models.qwen2 import Qwen2Attention, Qwen2DecoderLayer, Qwen2ForCausalLM

from dynamicPD.patching import dynamicPDPatch

class Qwen2AttentionPatch(dynamicPDPatch[Qwen2Attention]):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output
    
class Qwen2DecoderLayerPatch(dynamicPDPatch[Qwen2DecoderLayer]):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
    
class Qwen2ForCausalLMPatch(dynamicPDPatch[Qwen2ForCausalLM]):
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        use_offload: bool = False,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, use_offload=use_offload)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)