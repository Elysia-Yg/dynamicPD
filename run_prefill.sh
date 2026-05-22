export HCCL_IF_IP=172.17.0.2 # node ip
export GLOO_SOCKET_IFNAME="eth0"  # network card name
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH=/yangguang/workspace/ranktable.json
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export VLLM_ASCEND_LLMDD_RPC_PORT=5559
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

export DYNAMICPD_ENABLE=1 # enable dynamicPD
unset http_proxy

vllm serve /yangguang/workspace/Qwen/Qwen2.5-14b \
  --host 0.0.0.0 \
  --port 20002 \
  --data-parallel-size 1 \
  --data-parallel-size-local 1 \
  --api-server-count 1 \
  --data-parallel-address 172.17.0.2 \
  --data-parallel-rpc-port 13356 \
  --tensor-parallel-size 2 \
  --seed 1024 \
  --served-model-name qwen2.5_14b \
  --max-model-len 32768  \
  --max-num-batched-tokens 32768  \
  --max-num-seqs 256 \
  --async-scheduling \
  --trust-remote-code \
  --enforce-eager \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.94  \
  --kv-transfer-config  \
  '{"kv_connector": "LLMDataDistCMgrConnector",
  "kv_buffer_device": "npu",
  "kv_role": "kv_producer",
  "kv_parallel_size": 1,
  "kv_port": "20001",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_c_mgr_connector"
  }' > prefill1.txt