#!/bin/sh

export ASCEND_RT_VISIBLE_DEVICES=2,3
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True
# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export VLLM_ASCEND_ENABLE_FLASHCOMM1s=1   
export DYNAMICPD_ENABLED=1

#--quantization ascend \

vllm serve /model/Qwen3.6-35B-A3B-w8a8 \
--host 0.0.0.0 \
--port 13701 \
--data-parallel-size 1 \
--tensor-parallel-size 2 \
--enable-expert-parallel \
--seed 1024 \
--no-enable-prefix-caching \
--quantization ascend \
--served-model-name qwen3.6_35b \
--max-num-seqs 256 \
--max-model-len 262144 \
--max-num-batched-tokens 8192 \
--trust-remote-code \
--gpu-memory-utilization 0.93 \
--no-disable-hybrid-kv-cache-manager \
--speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 1, "enforce_eager": true}' \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
--additional-config '{"enable_cpu_binding":true, "multistream_overlap_shared_expert": true}' \
--async-scheduling \
--kv-transfer-config \
'{"kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_consumer",
  "kv_port": "30100",
  "engine_id": "1",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 1,
                    "tp_size": 2
             },
             "decode": {
                    "dp_size": 1,
                    "tp_size": 2
             },
              "dynamic_pd_config": {
                     "overload_trd": 2000,
                     "max_decode_offload_tokens":4096,
                     "split_trd":1024,
                     "decode_kv_watermark":0.7,
                     "use_async_offload":true,
                     "async_threshold": 1024,
                     "prefill_token_cost_ms": 0.15,
                     "prefill_fixed_cost_ms": 800
              }
      }
  }'  > decode1.txt