from typing import Optional, Union

import torch
from torch.nn.parameter import Parameter

from vllm.distributed import (split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce)
from vllm.forward_context import get_forward_context


from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear

from dynamicPD.patching import dynamicPDPatch
from dynamicPD.vllm.vllm.distributed.communication_op import tensor_model_parallel_all_gather_offload, tensor_model_parallel_all_reduce_offload


class ColumnParallelLinearPatch(dynamicPDPatch[ColumnParallelLinear]):
    def forward(
        self,
        input_,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self, input_, bias)

        use_offload_tp = get_forward_context().use_offload_tp
        if self.gather_output and self.tp_size > 1:
            if use_offload_tp:
            # All-gather across the partitions.
                output = tensor_model_parallel_all_gather_offload(output_parallel)
            else:
                output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias

class RowParallelLinearPatch(dynamicPDPatch[RowParallelLinear]):
    def forward(
        self,
        input_,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self, input_parallel, bias_)

        use_offload_tp = get_forward_context().use_offload_tp
        if self.reduce_results and self.tp_size > 1:
            if use_offload_tp:
                # logger.info("Using offload tensor parallel all-reduce in linear")
                output = tensor_model_parallel_all_reduce_offload(
                    output_parallel)
            else:
                output = tensor_model_parallel_all_reduce(output_parallel)
            # output = output_parallel
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias