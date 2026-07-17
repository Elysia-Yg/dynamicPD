#!/usr/bin/env python3
"""Profile the relationship between prompt length and TTFT.

This script targets the PD proxy endpoint used by curl_api.py.  It sends
streaming completion requests with different prompt token counts, records TTFT
from the first streamed chunk, writes raw samples to CSV, and writes a compact
model summary to JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    from curl_api import (
        API_URL,
        MODEL,
        MODEL_NAME,
        RequestFuncInput,
        async_request_openai_completions,
    )
    CURL_API_IMPORT_ERROR: Optional[BaseException] = None
except ModuleNotFoundError as exc:
    # Keep CLI help and model fitting code usable in minimal environments.  The
    # real request path still requires curl_api.py and its runtime dependencies.
    API_URL = "http://10.176.30.106:8181/v1/completions"
    MODEL = "qwen3.6_35b"
    MODEL_NAME = "/model/Qwen3.6-35B-A3B-w8a8"
    RequestFuncInput = None
    async_request_openai_completions = None
    CURL_API_IMPORT_ERROR = exc


DEFAULT_LENGTHS = "512,1024,2048,4096,8192,12288,16384"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_lengths(value: str) -> list[int]:
    lengths: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        length = int(item)
        if length <= 0:
            raise argparse.ArgumentTypeError("lengths must be positive")
        lengths.append(length)
    if not lengths:
        raise argparse.ArgumentTypeError("at least one length is required")
    return lengths


def parse_positive_int_list(value: str) -> list[int]:
    items: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("values must be positive")
        items.append(parsed)
    if not items:
        raise argparse.ArgumentTypeError("at least one value is required")
    return items


def parse_json_arg(value: Optional[str]) -> Optional[dict[str, Any]]:
    if not value:
        return None
    if value.startswith("@"):
        with open(value[1:], encoding="utf-8") as f:
            return json.load(f)
    return json.loads(value)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize_by_length(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["prompt_len_for_model"], []).append(row)

    summary: list[dict[str, Any]] = []
    for prompt_len in sorted(grouped):
        samples = grouped[prompt_len]
        successes = [row for row in samples if row["success"]]
        ttfts = [row["ttft_ms"] for row in successes]
        latencies = [row["latency_ms"] for row in successes]
        summary.append(
            {
                "prompt_len": prompt_len,
                "samples": len(samples),
                "successes": len(successes),
                "failures": len(samples) - len(successes),
                "ttft_ms_mean": statistics.fmean(ttfts) if ttfts else 0.0,
                "ttft_ms_median": statistics.median(ttfts) if ttfts else 0.0,
                "ttft_ms_std": statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0,
                "ttft_ms_min": min(ttfts) if ttfts else 0.0,
                "ttft_ms_p90": percentile(ttfts, 90),
                "ttft_ms_p95": percentile(ttfts, 95),
                "ttft_ms_max": max(ttfts) if ttfts else 0.0,
                "latency_ms_mean": statistics.fmean(latencies) if latencies else 0.0,
            }
        )
    return summary


def summarize_by_concurrency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["concurrency_level"], []).append(row)

    summary: list[dict[str, Any]] = []
    for concurrency_level in sorted(grouped):
        samples = grouped[concurrency_level]
        successes = [row for row in samples if row["success"]]
        ttfts = [row["ttft_ms"] for row in successes]
        latencies = [row["latency_ms"] for row in successes]
        item = {
            "concurrency": concurrency_level,
            "samples": len(samples),
            "successes": len(successes),
            "failures": len(samples) - len(successes),
            "ttft_ms_mean": statistics.fmean(ttfts) if ttfts else 0.0,
            "ttft_ms_median": statistics.median(ttfts) if ttfts else 0.0,
            "ttft_ms_std": statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0,
            "ttft_ms_min": min(ttfts) if ttfts else 0.0,
            "ttft_ms_p90": percentile(ttfts, 90),
            "ttft_ms_p95": percentile(ttfts, 95),
            "ttft_ms_max": max(ttfts) if ttfts else 0.0,
            "latency_ms_mean": statistics.fmean(latencies) if latencies else 0.0,
            "by_length": summarize_by_length(samples),
            "linear_model": fit_linear_model(samples),
            "piecewise_linear_model": fit_piecewise_linear_model(samples),
        }
        summary.append(item)
    return summary


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> Optional[list[float]]:
    """Solve Ax=b with Gaussian elimination for small dense systems."""

    n = len(vector)
    aug = [row[:] + [vector[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        pivot_value = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_value

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def fit_least_squares(features: list[list[float]], y: list[float]) -> Optional[dict[str, Any]]:
    if not features or len(features) != len(y):
        return None

    cols = len(features[0])
    xtx = [[0.0 for _ in range(cols)] for _ in range(cols)]
    xty = [0.0 for _ in range(cols)]
    for row, target in zip(features, y):
        for i in range(cols):
            xty[i] += row[i] * target
            for j in range(cols):
                xtx[i][j] += row[i] * row[j]

    coef = solve_linear_system(xtx, xty)
    if coef is None:
        return None

    pred = [sum(c * x for c, x in zip(coef, row)) for row in features]
    residuals = [target - estimate for target, estimate in zip(y, pred)]
    sse = sum(value * value for value in residuals)
    mae = statistics.fmean(abs(value) for value in residuals)
    rmse = math.sqrt(sse / len(y))
    y_mean = statistics.fmean(y)
    sst = sum((target - y_mean) ** 2 for target in y)
    r2 = 1.0 - sse / sst if sst > 0 else 1.0

    return {
        "coefficients": coef,
        "sse": sse,
        "mae_ms": mae,
        "rmse_ms": rmse,
        "r2": r2,
        "num_samples": len(y),
    }


def fit_linear_model(rows: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    points = [
        (float(row["prompt_len_for_model"]), float(row["ttft_ms"]))
        for row in rows
        if row["success"]
    ]
    if len(points) < 2:
        return None

    x = [point[0] for point in points]
    y = [point[1] for point in points]
    model = fit_least_squares([[1.0, value] for value in x], y)
    if model is None:
        return None

    intercept, slope = model.pop("coefficients")
    return {
        "formula": "ttft_ms = fixed_ms + token_cost_ms * prompt_tokens",
        "fixed_ms": intercept,
        "token_cost_ms": slope,
        **model,
    }


def fit_piecewise_linear_model(rows: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    points = [
        (float(row["prompt_len_for_model"]), float(row["ttft_ms"]))
        for row in rows
        if row["success"]
    ]
    unique_lengths = sorted({point[0] for point in points})
    if len(unique_lengths) < 4:
        return None

    best: Optional[dict[str, Any]] = None
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    for breakpoint in unique_lengths[1:-1]:
        features = [[1.0, value, max(0.0, value - breakpoint)] for value in x]
        model = fit_least_squares(features, y)
        if model is None:
            continue
        if best is None or model["sse"] < best["sse"]:
            intercept, slope_before, slope_delta = model.pop("coefficients")
            best = {
                "formula": (
                    "ttft_ms = fixed_ms + slope_before_ms_per_token * tokens "
                    "+ slope_delta_ms_per_token * max(0, tokens - breakpoint)"
                ),
                "breakpoint_tokens": int(breakpoint),
                "fixed_ms": intercept,
                "slope_before_ms_per_token": slope_before,
                "slope_after_ms_per_token": slope_before + slope_delta,
                "slope_delta_ms_per_token": slope_delta,
                **model,
            }

    return best


def fit_concurrency_model(rows: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    points = [
        (
            float(row["prompt_len_for_model"]),
            float(row["concurrency_level"]),
            float(row["ttft_ms"]),
        )
        for row in rows
        if row["success"]
    ]
    if len(points) < 4 or len({point[1] for point in points}) < 2:
        return None

    features = [
        [1.0, prompt_len, concurrency, prompt_len * concurrency]
        for prompt_len, concurrency, _ in points
    ]
    y = [ttft_ms for _, _, ttft_ms in points]
    model = fit_least_squares(features, y)
    if model is None:
        return None

    fixed, token_cost, concurrency_cost, token_concurrency_cost = model.pop(
        "coefficients"
    )
    return {
        "formula": (
            "ttft_ms = fixed_ms + token_cost_ms * prompt_tokens "
            "+ concurrency_cost_ms * concurrency "
            "+ token_concurrency_cost_ms * prompt_tokens * concurrency"
        ),
        "fixed_ms": fixed,
        "token_cost_ms": token_cost,
        "concurrency_cost_ms": concurrency_cost,
        "token_concurrency_cost_ms": token_concurrency_cost,
        **model,
    }


def load_tokenizer(tokenizer_path: Optional[str]) -> Any:
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def make_prompt_with_tokenizer(
    tokenizer: Any,
    prompt_len: int,
    token_unit_text: str,
) -> tuple[str, int]:
    unit_ids = tokenizer.encode(token_unit_text, add_special_tokens=False)
    if not unit_ids:
        raise ValueError(f"token_unit_text produced no tokens: {token_unit_text!r}")

    repeats = math.ceil(prompt_len / len(unit_ids))
    prompt_ids = (unit_ids * repeats)[:prompt_len]
    prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    actual_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    if actual_len != prompt_len:
        raise ValueError(
            "failed to generate an exact prompt length: "
            f"requested={prompt_len}, actual={actual_len}; "
            "try a different --token-unit-text"
        )
    return prompt, actual_len


def make_prompt_without_tokenizer(
    prompt_len: int,
    token_unit_text: str,
    tokens_per_unit: float,
) -> tuple[str, int]:
    repeats = max(1, math.ceil(prompt_len / tokens_per_unit))
    return token_unit_text * repeats, prompt_len


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "concurrency_level",
        "requested_prompt_len",
        "actual_prompt_len",
        "prompt_len_for_model",
        "output_len",
        "success",
        "ttft_ms",
        "latency_ms",
        "output_tokens",
        "generated_chars",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


async def run_request(
    *,
    sample_id: int,
    concurrency_level: int,
    requested_prompt_len: int,
    actual_prompt_len: int,
    prompt: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if RequestFuncInput is None or async_request_openai_completions is None:
        raise RuntimeError(
            "curl_api.py or its dependencies are unavailable. Install the "
            "profiling runtime dependencies, including aiohttp, then rerun."
        ) from CURL_API_IMPORT_ERROR

    request_input = RequestFuncInput(
        prompt=prompt,
        api_url=args.api_url,
        prompt_len=actual_prompt_len,
        output_len=args.output_len,
        model=args.model,
        model_name=args.model_name,
        ignore_eos=args.ignore_eos,
        extra_body=args.extra_body,
    )
    output = await async_request_openai_completions(request_input)
    row = {
        "sample_id": sample_id,
        "concurrency_level": concurrency_level,
        "requested_prompt_len": requested_prompt_len,
        "actual_prompt_len": actual_prompt_len,
        "prompt_len_for_model": actual_prompt_len,
        "output_len": args.output_len,
        "success": output.success,
        "ttft_ms": output.ttft * 1000.0 if output.success else 0.0,
        "latency_ms": output.latency * 1000.0 if output.success else 0.0,
        "output_tokens": output.output_tokens or 0,
        "generated_chars": len(output.generated_text),
        "error": output.error,
    }
    status = "ok" if output.success else "failed"
    print(
        f"[{sample_id:04d}] concurrency={concurrency_level} "
        f"len={actual_prompt_len} {status} "
        f"ttft={row['ttft_ms']:.2f}ms latency={row['latency_ms']:.2f}ms"
    )
    return row


async def run_profile(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = load_tokenizer(args.tokenizer)
    prompts: dict[int, tuple[str, int]] = {}
    for length in args.lengths:
        if tokenizer is None:
            prompts[length] = make_prompt_without_tokenizer(
                length, args.token_unit_text, args.tokens_per_unit
            )
        else:
            prompts[length] = make_prompt_with_tokenizer(
                tokenizer, length, args.token_unit_text
            )

    if args.warmup_repeats > 0:
        warmup_length = args.warmup_length or args.lengths[0]
        if warmup_length not in prompts:
            if tokenizer is None:
                prompts[warmup_length] = make_prompt_without_tokenizer(
                    warmup_length, args.token_unit_text, args.tokens_per_unit
                )
            else:
                prompts[warmup_length] = make_prompt_with_tokenizer(
                    tokenizer, warmup_length, args.token_unit_text
                )
        prompt, actual_len = prompts[warmup_length]
        print(f"Running {args.warmup_repeats} warmup request(s) at len={actual_len}")
        for idx in range(args.warmup_repeats):
            await run_request(
                sample_id=-(idx + 1),
                concurrency_level=1,
                requested_prompt_len=warmup_length,
                actual_prompt_len=actual_len,
                prompt=prompt,
                args=args,
            )

    concurrency_levels = args.concurrency_levels or [args.concurrency]
    jobs_by_concurrency: list[tuple[int, list[tuple[int, int, int, str]]]] = []
    sample_id = 1
    for concurrency_level in concurrency_levels:
        jobs: list[tuple[int, int, int, str]] = []
        for length in args.lengths:
            prompt, actual_len = prompts[length]
            for _ in range(args.repeats):
                jobs.append((sample_id, length, actual_len, prompt))
                sample_id += 1
        if args.shuffle:
            random.Random(args.seed + concurrency_level).shuffle(jobs)
        jobs_by_concurrency.append((concurrency_level, jobs))

    async def run_concurrency_group(
        concurrency_level: int,
        jobs: list[tuple[int, int, int, str]],
    ) -> list[dict[str, Any]]:
        print(f"Running concurrency={concurrency_level} with {len(jobs)} samples")
        semaphore = asyncio.Semaphore(concurrency_level)

        async def guarded(job: tuple[int, int, int, str]) -> dict[str, Any]:
            sid, requested_len, actual_len, prompt = job
            async with semaphore:
                return await run_request(
                    sample_id=sid,
                    concurrency_level=concurrency_level,
                    requested_prompt_len=requested_len,
                    actual_prompt_len=actual_len,
                    prompt=prompt,
                    args=args,
                )

        return await asyncio.gather(*(guarded(job) for job in jobs))

    rows: list[dict[str, Any]] = []
    for concurrency_level, jobs in jobs_by_concurrency:
        group_rows = await run_concurrency_group(concurrency_level, jobs)
        rows.extend(group_rows)
    rows = sorted(rows, key=lambda row: row["sample_id"])

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            key: to_jsonable(value)
            for key, value in vars(args).items()
            if key not in {"extra_body"}
        },
        "extra_body": to_jsonable(args.extra_body),
        "samples": len(rows),
        "successes": sum(1 for row in rows if row["success"]),
        "failures": sum(1 for row in rows if not row["success"]),
        "by_length": summarize_by_length(rows),
        "by_concurrency": summarize_by_concurrency(rows),
        "linear_model": fit_linear_model(rows),
        "piecewise_linear_model": fit_piecewise_linear_model(rows),
        "concurrency_model": fit_concurrency_model(rows),
    }
    return rows, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile prompt-token-count versus TTFT through the PD proxy "
            "OpenAI completions endpoint."
        )
    )
    parser.add_argument("--api-url", default=API_URL, help="PD proxy completions URL")
    parser.add_argument("--model", default=MODEL, help="served model name")
    parser.add_argument(
        "--model-name",
        default=None,
        help=f"optional payload model override, for example {MODEL_NAME}",
    )
    parser.add_argument(
        "--lengths",
        type=parse_lengths,
        default=parse_lengths(DEFAULT_LENGTHS),
        help=f"comma-separated prompt token lengths, default: {DEFAULT_LENGTHS}",
    )
    parser.add_argument("--repeats", type=int, default=3, help="samples per length")
    parser.add_argument(
        "--output-len",
        type=int,
        default=1,
        help="max_tokens for each request; 1 isolates TTFT best",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--concurrency-levels",
        type=parse_positive_int_list,
        default=None,
        help=(
            "comma-separated concurrency levels to sweep, for example 1,2,4,8; "
            "overrides --concurrency when set"
        ),
    )
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--warmup-length", type=int, default=None)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="optional tokenizer path/name for exact prompt token counts",
    )
    parser.add_argument(
        "--token-unit-text",
        default="OSDI",
        help="text fragment used to synthesize prompts",
    )
    parser.add_argument(
        "--tokens-per-unit",
        type=float,
        default=2.0,
        help=(
            "assumed tokens per --token-unit-text when --tokenizer is not set; "
            "keeps compatibility with curl_api.py's OSDI * (length // 2)"
        ),
    )
    parser.add_argument(
        "--extra-body",
        default=None,
        type=parse_json_arg,
        help="JSON string, or @path, merged into the request body",
    )
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--shuffle", action="store_true", help="shuffle sample order")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="raw sample CSV output path",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="summary/model JSON output path",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.output_len <= 0:
        parser.error("--output-len must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.concurrency_levels and any(value <= 0 for value in args.concurrency_levels):
        parser.error("--concurrency-levels values must be positive")
    if args.tokens_per_unit <= 0:
        parser.error("--tokens-per-unit must be positive")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.csv or RESULTS_DIR / f"ttft_profile_{timestamp}.csv"
    json_path = args.json or RESULTS_DIR / f"ttft_profile_{timestamp}.json"

    rows, summary = asyncio.run(run_profile(args))
    write_csv(csv_path, rows)
    write_json(json_path, summary)

    print(f"Wrote raw samples: {csv_path}")
    print(f"Wrote model summary: {json_path}")

    if summary["successes"] == 0:
        raise SystemExit("No successful samples; check endpoint/model arguments.")

    linear = summary["linear_model"]
    if linear:
        print(
            "Linear model: "
            f"ttft_ms = {linear['fixed_ms']:.3f} "
            f"+ {linear['token_cost_ms']:.6f} * prompt_tokens "
            f"(R2={linear['r2']:.4f}, RMSE={linear['rmse_ms']:.2f}ms)"
        )

    piecewise = summary["piecewise_linear_model"]
    if piecewise:
        print(
            "Piecewise model: "
            f"breakpoint={piecewise['breakpoint_tokens']} tokens, "
            f"slope_before={piecewise['slope_before_ms_per_token']:.6f} ms/token, "
            f"slope_after={piecewise['slope_after_ms_per_token']:.6f} ms/token, "
            f"R2={piecewise['r2']:.4f}"
        )

    concurrency_model = summary["concurrency_model"]
    if concurrency_model:
        print(
            "Concurrency model: "
            f"ttft_ms = {concurrency_model['fixed_ms']:.3f} "
            f"+ {concurrency_model['token_cost_ms']:.6f} * prompt_tokens "
            f"+ {concurrency_model['concurrency_cost_ms']:.3f} * concurrency "
            "+ "
            f"{concurrency_model['token_concurrency_cost_ms']:.9f} "
            "* prompt_tokens * concurrency "
            f"(R2={concurrency_model['r2']:.4f}, "
            f"RMSE={concurrency_model['rmse_ms']:.2f}ms)"
        )

    for item in summary["by_concurrency"]:
        model = item["linear_model"]
        if not model:
            continue
        print(
            f"Concurrency={item['concurrency']} linear: "
            f"ttft_ms = {model['fixed_ms']:.3f} "
            f"+ {model['token_cost_ms']:.6f} * prompt_tokens "
            f"(R2={model['r2']:.4f})"
        )


if __name__ == "__main__":
    main()
