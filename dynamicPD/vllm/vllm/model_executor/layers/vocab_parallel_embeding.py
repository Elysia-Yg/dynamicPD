from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.vocab_parallel_embedding import get_masked_input_and_mask, VocabParallelEmbedding

from dynamicPD.vllm.vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from dynamicPD.patching import dynamicPDPatch

class VocabParallelEmbeddingPatch(dynamicPDPatch[VocabParallelEmbedding]):
    def forward_native(self, input_):
        if self.tp_size > 1:
            # Build the mask.
            masked_input, input_mask = get_masked_input_and_mask(
                input_, self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index)
        else:
            masked_input = input_
        # Get the embeddings.
        output_parallel = self.quant_method.embedding(self,
                                                      masked_input.long())
        # Mask the output embedding.
        if self.tp_size > 1:
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
        # Reduce across all the model parallel GPUs.
        forward_context: ForwardContext = get_forward_context()
        output = tensor_model_parallel_all_reduce(
            output_parallel,
            offload=forward_context.additional_kwargs.get("use_offload_tp", False),
        )
        return output
