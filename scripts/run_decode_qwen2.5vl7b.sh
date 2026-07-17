export ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_IF_IP=10.176.30.106  # node ip
export GLOO_SOCKET_IFNAME="enp189s0f0"  # network card name
export TP_SOCKET_IFNAME="enp189s0f0"
export HCCL_SOCKET_IFNAME="enp189s0f0"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10

vllm serve /model/Qwen2.5-VL-7B-Instruct  \
  --host 0.0.0.0 \
  --port 13701 \
  --no-enable-prefix-caching \
  --tensor-parallel-size 1 \
  --seed 1024 \
  --served-model-name qwen25vl \
  --max-model-len 40000  \
  --max-num-batched-tokens 40000  \
  --trust-remote-code \
  --gpu-memory-utilization 0.9  \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_consumer",
  "kv_port": "30100",
  "engine_id": "1",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 1,
                    "tp_size": 1
             },
             "decode": {
                    "dp_size": 1,
                    "tp_size": 1
             }
      }
  }'
