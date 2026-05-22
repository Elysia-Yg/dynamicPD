# dynamicPD

版本依赖：

- vllm 0.11.0
- vllm-ascend 0.11.0
- torch 2.7.1
- torch_npu 2.7.1

进入仓库文件夹 pip install -e . 安装

决定是否启用的开关在 run_prefill.sh 和 run_decode.sh 中

准备工作

1. 下载BurstGPT到仓库文件夹
2. 修改 run_prefill.sh 和 run_decode.sh 中的各项参数，重点是模型路径
3. 修改 apply_patch.sh 中的 TARGET_DIR 为 vllm 的安装目录
4. 根据 0.11.0版本的 vllm-ascend/examples/disaggreagated_prefill_v1 中的方式生成     ranktable.json ，并放在仓库的根目录下

启动方式

- 在两个终端中分别运行 

  ```
  bash run_prefill.sh
  bash run_decode.sh
  ```

- 新启一个终端，运行

  ```
  python load_balance_proxy_server_example.py --host (ip) --port 1025 --prefiller-hosts (prefill_ip) --prefiller-port 20002 --decoder-hosts (decode_ip) --decoder-ports 20002
  ```

- 然后就可以使用 curl 进行访问或者使用 benchmark 进行基准测试

  ```
  curl http://(ip):1025/v1/completions    -H "Content-Type: application/json"     -d '{
          "model": "qwen2.5_14b",
          "prompt": "Who are you?",
          "max_tokens": 200,
          "temperature": 0
      }'
  ```

  ```
  python3 benchmark_serving.py \
      --backend vllm \
      --dataset-name burstgpt \
      --dataset-path ./benchmarks/BurstGPT_without_fails_2.csv \
      --num-prompts 400 \
      --ignore-eos \
      --model qwen2.5_14b\
      --tokenizer /yangguang/workspace/Qwen/Qwen2.5-14b \
      --host (ip) \
      --port 1025 \
      --endpoint /v1/completions \
      --request-rate 8
  ```

  