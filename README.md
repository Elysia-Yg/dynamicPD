# dynamicPD

dynamicPD 是基于 vLLM / vLLM-Ascend 的 PD 分离实验 (WindServe) 补丁，当前分支已对齐到 vLLM 0.18.0。它主要用于 Ascend 上的分离式 prefill/decode，并支持 decode 侧异步卸载较大的 prefill 请求。

## 版本依赖

- vLLM 0.18.0
- vLLM-Ascend 0.18.0
- torch 2.7.1
- torch-npu 2.7.1
- Python >= 3.10

## 安装

在 dynamicPD 仓库中安装：

```bash
cd dynamicPD
pip install -e .
```

dynamicPD 通过 vLLM plugin 入口加载。启动 vLLM 前需要设置：

```bash
export DYNAMICPD_ENABLED=1
```

如果只想跑原生 vLLM / vLLM-Ascend，去掉这个环境变量或设置为 `0`。

## Docker

仓库提供了 `dynamicPD/Dockerfile`，会把当前 workspace 中的 `vllm/`、`vllm-ascend/` 和 `dynamicPD/` 一起打进镜像，并在镜像内执行 `scripts/apply_patch.sh`。请从 workspace 根目录构建：

```bash
cd /home/wujie/jingqi/vllm_workspace
docker build \
  -f dynamicPD/Dockerfile \
  --build-arg BASE_IMAGE=vllm-ascend:0.18.0 \
  -t dynamicpd:0.18.0 \
  .
```

`BASE_IMAGE` 需要换成你实际可用的 Ascend / CANN / torch-npu 运行镜像。镜像默认设置 `DYNAMICPD_ENABLED=1`，入口命令是 `vllm`，因此可以直接追加 `serve` 参数：

Docker build 阶段不会挂载宿主机 Ascend driver，因此 Dockerfile 在 `pip install -e` 时会临时设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，避免 PyTorch 生成 metadata 时自动加载 `torch_npu` 并查找 `libascend_hal.so`。这个变量只在 build 的安装命令中生效，不会作为运行时环境变量写入镜像。

```bash
docker run --rm -it \
  --network host \
  --privileged \
  dynamicpd:0.18.0 \
  serve /model/Qwen2.5-14B-Instruct --help
```

如果要在容器里使用脚本，可以覆盖入口：

```bash
docker run --rm -it \
  --network host \
  --privileged \
  --entrypoint bash \
  dynamicpd:0.18.0
```

## 当前补丁内容

- vLLM v1 scheduler：拆分 decode batch 与卸载到 decode 侧执行的 prefill batch。
- vLLM v1 engine：新增 PD Coordinator，用 prefill/decode 实例的排队 token、running token、decode 侧 offload token、KV cache 使用率来判断是否迁移请求。
- vLLM distributed parallel state：新增 secondary/offload 通信组，避免异步 prefill 和 decode 共用同一个集合通信组。
- vLLM-Ascend model runner：decode 侧维护两个 persistent batch lane，用不同 NPU stream 执行 decode 与 offload prefill。
- vLLM-Ascend forward context：新增 `use_offload_tp`，并对齐 MoE 通信方法选择，支持 MC2 / LMTP / all-gather / all-to-all 等路径。

## 启动示例

脚本在 `scripts/` 下，常用示例：

```bash
cd dynamicPD/scripts

# 终端 1：prefill 实例
bash run_prefill_qwen2.514b.sh

# 终端 2：decode 实例
bash run_decode_qwen2.514b.sh
```

MoE 示例：

```bash
cd dynamicPD/scripts
bash run_prefill_qwen3.6_35b.sh
bash run_decode_qwen3.6_35b.sh
```

## 关键 vLLM 参数

decode 侧异步卸载依赖 vLLM async scheduling，因此使用：

```bash
--async-scheduling
```

同时在 `--kv-transfer-config` 里配置 `dynamic_pd_config`：

```json
{
  "kv_connector": "MooncakeConnectorV1",
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
      "max_decode_offload_tokens": 4096,
      "decode_kv_watermark": 0.7,
      "use_async_offload": true,
      "async_threshold": 1024,
      "async_chunk_size": 2048,
      "prefill_token_cost_ms": 0.2,
      "prefill_fixed_cost_ms": 80
    }
  }
}
```

`kv_role`：

- `kv_producer` 表示 prefill 实例。
- `kv_consumer` 表示 decode 实例。

`dynamic_pd_config`：

- `use_async_offload`：是否启用 decode 侧异步卸载 prefill。启用时必须同时打开 `--async-scheduling`。
- `async_threshold`：prefill 剩余 token 数达到该阈值后，才拆到异步 offload lane；过小的 prefill 仍和 decode 放在普通 batch 中。
- `async_chunk_size`：异步 offload 的 chunk 大小；未配置时默认使用 2048。
- `overload_trd`：prefill 侧预测等待时间超过该阈值后，才考虑迁移到 decode。
- `decode_kv_watermark`：decode 实例 KV cache 使用率超过该值后，不再接收迁移请求。
- `max_decode_offload_tokens`：decode 侧正在处理的 offload prefill token 上限。
- `prefill_token_cost_ms` / `prefill_fixed_cost_ms`：prefill 等待时间估算参数，需要按模型和硬件校准。

## PD Coordinator

dynamicPD 会在 prefill 侧启动一个轻量 coordinator。默认地址：

```json
{
  "coordinator_input_address": "tcp://127.0.0.1:16666",
  "coordinator_publish_address": "tcp://127.0.0.1:16667",
  "publish_interval_ms": 100,
  "stale_ms": 2000,
  "group": "default"
}
```

多机或多组 PD 时，需要把地址改成各实例都能访问的地址，并用 `group` 隔离不同 PD 集群。

## 运行请求

可以直接访问 decode 实例，或使用已有的代理脚本做 prefill/decode 入口转发。简单 curl 示例：

```bash
curl http://127.0.0.1:13701/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5_14b",
    "prompt": "Who are you?",
    "max_tokens": 200,
    "temperature": 0
  }'
```

benchmark 示例：

```bash
python3 benchmark_serving.py \
  --backend vllm \
  --dataset-name burstgpt \
  --dataset-path ./benchmarks/BurstGPT_without_fails_2.csv \
  --num-prompts 400 \
  --ignore-eos \
  --model qwen2.5_14b \
  --tokenizer /model/Qwen2.5-14B-Instruct \
  --host 127.0.0.1 \
  --port 13701 \
  --endpoint /v1/completions \
  --request-rate 8
```

## 调试开关

这些开关主要用于定位异步 prefill/decode 的 stream 或事件同步问题：

```bash
unset DYNAMIC_PD_SYNC_PREFILL_BEFORE_DECODE
unset DYNAMIC_PD_PREFILL_DECODE_FENCE
export DYNAMIC_PD_DECODE_PENDING_PREFILL_MODE="acl_none"
```

常见含义：

- `DYNAMIC_PD_SYNC_PREFILL_BEFORE_DECODE=1`：强制 decode 前等待 pending prefill 完成，适合确认问题是否来自异步并发。
- `DYNAMIC_PD_PREFILL_DECODE_FENCE=snapshot`：decode 等待 prefill 的 device snapshot ready 事件，通常会更保守。
- `DYNAMIC_PD_PREFILL_DECODE_FENCE=none`：不主动 fence pending prefill。
- `DYNAMIC_PD_DECODE_PENDING_PREFILL_MODE=acl_none`：pending prefill 存在时，decode 侧禁用 ACL graph，避免 graph replay 与另一路异步 prefill 互相影响。
