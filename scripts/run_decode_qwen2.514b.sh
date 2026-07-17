export ASCEND_RT_VISIBLE_DEVICES=2,3
export HCCL_IF_IP=10.176.30.106  # node ip
export GLOO_SOCKET_IFNAME="enp189s0f0"  # network card name
export TP_SOCKET_IFNAME="enp189s0f0"
export HCCL_SOCKET_IFNAME="enp189s0f0"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export DYNAMICPD_ENABLED=1
unset DYNAMIC_PD_SYNC_PREFILL_BEFORE_DECODE
unset DYNAMIC_PD_PREFILL_DECODE_FENCE
export DYNAMIC_PD_DECODE_PENDING_PREFILL_MODE="acl_none"

vllm serve /model/Qwen2.5-14B-Instruct  \
  --host 0.0.0.0 \
  --port 13701 \
  --no-enable-prefix-caching \
  --tensor-parallel-size 2 \
  --seed 1024 \
  --served-model-name qwen2.5_14b \
  --max-model-len 32768  \
  --max-num-batched-tokens 32768  \
  --trust-remote-code \
  --async-scheduling \
  --gpu-memory-utilization 0.9  \
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
                     "overload_trd": 1000,
                     "max_decode_offload_tokens":4096,
                     "split_trd":1024,
                     "decode_kv_watermark":0.7,
                     "use_async_offload":true,
                     "async_threshold": 1024
              }
      }
  }' > decode1.txt
