import io
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional, Union

import aiohttp
import huggingface_hub.constants
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

# NOTE(simon): do not import vLLM here so the benchmark script
# can run without vLLM installed.

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

END_POINT = "/v1/completions"
# MODEL = "qwen2.5_14b"
# MODEL_NAME = "/model/Qwen2.5-14B-Instruct"

MODEL = "qwen3.6_35b"
MODEL_NAME = "/model/Qwen3.6-35B-A3B-w8a8"
API_URL = "http://10.176.30.106:8181"+END_POINT


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: Optional[str] = None
    logprobs: Optional[int] = None
    extra_body: Optional[dict] = None
    multi_modal_content: Optional[dict] = None
    ignore_eos: bool = False
    language: Optional[str] = None


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(default_factory=list)  # list of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    stream_json_fields: set[str] = field(default_factory=set)
    error: str = ""


def record_json_fields(value, fields: set[str], prefix: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            field_name = f"{prefix}.{key}" if prefix else key
            fields.add(field_name)
            record_json_fields(item, fields, field_name)
    elif isinstance(value, list):
        for item in value:
            record_json_fields(item, fields, f"{prefix}[]")
    
async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(("completions", "profile")), (
        "OpenAI Completions API URL must end with 'completions' or 'profile'."
    )

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        payload = {
            "model": request_func_input.model_name
            if request_func_input.model_name
            else request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "max_tokens": request_func_input.output_len,
            "logprobs": request_func_input.logprobs,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            # print(f"Sending request to {api_url}, payload: {payload}, headers: {headers}")
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                        if chunk != "[DONE]":
                            data = json.loads(chunk)
                            record_json_fields(data, output.stream_json_fields)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if choices := data.get("choices"):
                                # Note that text could be empty here
                                # e.g. for special tokens
                                text = choices[0].get("text")
                                timestamp = time.perf_counter()
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += text or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")
                    if first_chunk_received:
                        output.success = True
                    else:
                        output.success = False
                        output.error = (
                            "Never received a valid chunk to calculate TTFT."
                            "This response will be marked as failed!"
                        )
                    output.generated_text = generated_text
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            print("Exception during request:", output.error)

    if pbar:
        pbar.update(1)
    return output

if __name__ == "__main__":
    import asyncio

    async def main():
        length = 8192
        # prompt = "Tell me the question "*(length//4) + "\nQ: What does OSDI stand for?\nA:"
        # prompt = "Explain quantum computing in simple terms. " * 1024 + "\nQ: What is quantum entanglement?\nA:"
        prompt = "OSDI" * (length // 2)
        # prompt = "who are you?"
        request_input = RequestFuncInput(
            prompt=prompt,
            api_url=API_URL,
            prompt_len=length,
            output_len=1,
            model=MODEL,
        )
        try:
            output = await async_request_openai_completions(request_input)
            print("Generated Text:", output.generated_text)
            print("Success:", output.success)
            print("Latency:", output.latency)
            print("TTFT:", output.ttft)
            print("Output Tokens:", output.output_tokens)
            print("ITL:", output.itl)
            print("Stream JSON Fields:")
            for field_name in sorted(output.stream_json_fields):
                print("  ", field_name)
        except Exception as e:
            print("Error during request:", str(e))

    asyncio.run(main())
