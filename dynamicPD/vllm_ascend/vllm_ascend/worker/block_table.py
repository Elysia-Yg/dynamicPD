from typing import Union

import numpy as np
import torch
from vllm.distributed import get_dcp_group


from vllm_ascend.worker.block_table import BlockTable

from dynamicPD.patching import dynamicPDPatch

class BlockTablePatch(dynamicPDPatch[BlockTable]):
    def __init__(self,
                 block_size: int,
                 max_num_reqs: int,
                 max_num_blocks_per_req: int,
                 max_num_batched_tokens: int,
                 pin_memory: bool,
                 device: torch.device,
                 kernel_sizes: Union[list[int], None] = None):
        self.max_num_reqs = max_num_reqs
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device
        self.physical_block_size = block_size
        # If kernel_sizes is None or [0], use physical block size (no splitting)
        if kernel_sizes is None or kernel_sizes == [0]:
            self.block_size = block_size
            self.logical_block_size = block_size
            self.blocks_per_phys_block = 1
            self.use_hybrid_blocks = False
        else:
            # Find the first kernel size that divides physical_block_size evenly
            selected_kernel_size = None
            for kernel_size in kernel_sizes:
                if kernel_size > 0 \
                    and self.physical_block_size % kernel_size == 0:
                    selected_kernel_size = kernel_size
                    break

            if selected_kernel_size is None:
                raise ValueError(
                    f"None of the kernel sizes {kernel_sizes} can divide "
                    f"physical block size {self.physical_block_size} evenly")

            self.block_size = selected_kernel_size
            self.logical_block_size = selected_kernel_size
            self.blocks_per_phys_block = (self.physical_block_size //
                                          self.logical_block_size)
            if self.blocks_per_phys_block > 1:
                self.use_hybrid_blocks = True
            else:
                self.use_hybrid_blocks = False

        if self.use_hybrid_blocks:
            logical_table_size = (max_num_blocks_per_req *
                                  self.blocks_per_phys_block)
        else:
            logical_table_size = max_num_blocks_per_req

        self.block_table = torch.zeros(
            (max_num_reqs, logical_table_size),
            device=self.device,
            dtype=torch.int32,
        )
        self.block_table_cpu = torch.zeros(
            (max_num_reqs, logical_table_size),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.block_table_np = self.block_table_cpu.numpy()
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

        self.slot_mapping_cpu = torch.zeros(self.max_num_batched_tokens*8,
                                            dtype=torch.int64,
                                            device="cpu",
                                            pin_memory=self.pin_memory)
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()
        self.slot_mapping = torch.zeros(self.max_num_batched_tokens*8,
                                        dtype=torch.int64,
                                        device=self.device)
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        self.kernel_sizes = kernel_sizes