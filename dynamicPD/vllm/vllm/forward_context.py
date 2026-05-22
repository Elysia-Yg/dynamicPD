import time
import torch

from typing import Any, Optional

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, batchsize_forward_time, DPMetadata, ForwardContext, override_forward_context , track_batchsize
from vllm.logger import logger
from vllm.v1.worker.ubatch_utils import UBatchSlices


from dynamicPD.patching import dynamicPDPatch

class ForwardContextPatch(dynamicPDPatch[ForwardContext]):
    use_offload_tp: Optional[bool] = False
    _orig_init = ForwardContext.__init__
    def __init__(self, *args, **kwargs):
        self.use_offload_tp = kwargs.pop("use_offload_tp", False)
        self._orig_init(*args, **kwargs)

def create_forward_context(
        attn_metadata: Any,
        vllm_config: VllmConfig,
        virtual_engine: int = 0,
        dp_metadata: Optional[DPMetadata] = None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor: Optional[BatchDescriptor] = None,
        ubatch_slices: Optional[UBatchSlices] = None,
        use_offload_tp: Optional[bool] = False) -> ForwardContext:
    return ForwardContext(no_compile_layers=vllm_config.compilation_config.
                          static_forward_context,
                          virtual_engine=virtual_engine,
                          attn_metadata=attn_metadata,
                          dp_metadata=dp_metadata,
                          cudagraph_runtime_mode=cudagraph_runtime_mode,
                          batch_descriptor=batch_descriptor,
                          ubatch_slices=ubatch_slices,
                          use_offload_tp=use_offload_tp)

def set_forward_context(
        attn_metadata: Any,
        vllm_config: VllmConfig,
        virtual_engine: int = 0,
        num_tokens: Optional[int] = None,
        num_tokens_across_dp: Optional[torch.Tensor] = None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor: Optional[BatchDescriptor] = None,
        ubatch_slices: Optional[UBatchSlices] = None,
        use_offload_tp: Optional[bool] = False):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    global forward_start_time
    need_to_track_batchsize = track_batchsize and attn_metadata is not None
    if need_to_track_batchsize:
        forward_start_time = time.perf_counter()

    dp_metadata: Optional[DPMetadata] = None
    if vllm_config.parallel_config.data_parallel_size > 1 and (
            attn_metadata is not None or num_tokens is not None):
        dp_metadata = DPMetadata.make(vllm_config.parallel_config,
                                      attn_metadata, num_tokens or 0,
                                      num_tokens_across_dp)

    forward_context = create_forward_context(attn_metadata, vllm_config,
                                             virtual_engine, dp_metadata,
                                             cudagraph_runtime_mode,
                                             batch_descriptor, ubatch_slices,
                                             use_offload_tp)

    try:
        with override_forward_context(forward_context):
            yield
    finally:
        global last_logging_time, batchsize_logging_interval
        if need_to_track_batchsize:
            if hasattr(attn_metadata, "num_prefill_tokens"):
                # for v0 attention backends
                batchsize = attn_metadata.num_prefill_tokens + \
                    attn_metadata.num_decode_tokens
            else:
                # for v1 attention backends
                batchsize = num_tokens
            # we use synchronous scheduling right now,
            # adding a sync point here should not affect
            # scheduling of the next batch
            # from vllm.platforms import current_platform
            # synchronize = current_platform.synchronize
            # if synchronize is not None:
            #     synchronize()
            now = time.perf_counter()
            # time measurement is in milliseconds
            batchsize_forward_time[batchsize].append(
                (now - forward_start_time) * 1000)
            if now - last_logging_time > batchsize_logging_interval:
                last_logging_time = now
                forward_stats = []
                for bs, times in batchsize_forward_time.items():
                    if len(times) <= 1:
                        # can be cudagraph / profiling run
                        continue
                    medium = torch.quantile(torch.tensor(times), q=0.5).item()
                    medium = round(medium, 2)
                    forward_stats.append((bs, len(times), medium))
                forward_stats.sort(key=lambda x: x[1], reverse=True)
                if forward_stats:
                    logger.info(("Batchsize forward time stats "
                                 "(batchsize, count, median_time(ms)): %s"),
                                forward_stats)