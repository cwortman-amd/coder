#!/usr/bin/env python3
"""Build the argument list for `vllm bench serve`.

The binary name stays with the caller (`vllm` or `/opt/vllm/bin/vllm`).
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional


def bench_serve_args(
    *,
    model: str,
    input_len: int,
    output_len: int,
    num_prompts: int,
    max_concurrency: Optional[int] = None,
    host: Optional[str] = "127.0.0.1",
    port: Optional[int] = 8000,
    backend: Optional[str] = "openai-chat",
    endpoint: Optional[str] = "/v1/chat/completions",
    tokenizer: Optional[str] = None,
    request_rate: Optional[str] = "inf",
    ignore_eos: bool = False,
    temperature: Optional[float] = None,
    result_dir: Optional[str] = None,
    result_filename: Optional[str] = None,
    percentile_metrics: Optional[str] = "tpot,ttft,itl,e2el",
    metric_percentiles: Optional[str] = "50,90,95,99",
    save_detailed: bool = True,
    random_prefix_len: Optional[int] = None,
    burstiness: Optional[float] = None,
    seed: Optional[int] = None,
    served_model_name: Optional[str] = None,
    trust_remote_code: bool = False,
    num_warmups: Optional[int] = None,
) -> list[str]:
    args = ["bench", "serve", "--model", model]
    if backend:
        args[2:2] = ["--backend", backend]
    if served_model_name:
        args += ["--served-model-name", served_model_name]
    if tokenizer:
        args += ["--tokenizer", tokenizer]
    if endpoint:
        args += ["--endpoint", endpoint]
    if host:
        args += ["--host", host]
    if port is not None:
        args += ["--port", str(port)]
    if percentile_metrics:
        args += ["--percentile-metrics", percentile_metrics]
    if metric_percentiles:
        args += ["--metric-percentiles", metric_percentiles]
    args += [
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--num-prompts", str(num_prompts),
    ]
    if max_concurrency is not None:
        args += ["--max-concurrency", str(max_concurrency)]
    if request_rate:
        args += ["--request-rate", request_rate]
    if random_prefix_len:
        args += ["--random-prefix-len", str(random_prefix_len)]
    if burstiness is not None:
        args += ["--burstiness", str(burstiness)]
    if seed is not None:
        args += ["--seed", str(seed)]
    if ignore_eos:
        args.append("--ignore-eos")
    if temperature is not None:
        args += ["--temperature", str(temperature)]
    if result_dir and result_filename:
        args += [
            "--save-result",
            "--result-dir", result_dir,
            "--result-filename", result_filename,
        ]
        if save_detailed:
            args.append("--save-detailed")
    if trust_remote_code:
        args.append("--trust-remote-code")
    if num_warmups is not None:
        args += ["--num-warmups", str(num_warmups)]
    return args


def main() -> int:
    parser = argparse.ArgumentParser(description="Print vllm bench serve arguments, one per line")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--no-max-concurrency", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-host", action="store_true")
    parser.add_argument("--no-port", action="store_true")
    parser.add_argument("--backend", default="openai-chat")
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--result-dir", default="")
    parser.add_argument("--result-filename", default="")
    parser.add_argument("--percentile-metrics", default="tpot,ttft,itl,e2el")
    parser.add_argument("--no-percentile-metrics", action="store_true")
    parser.add_argument("--metric-percentiles", default="50,90,95,99")
    parser.add_argument("--no-metric-percentiles", action="store_true")
    parser.add_argument("--served-model-name", default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--num-warmups", type=int, default=None)
    parser.add_argument("--no-save-detailed", action="store_true")
    parser.add_argument("--no-backend", action="store_true")
    parser.add_argument("--no-endpoint", action="store_true")
    parser.add_argument("--no-request-rate", action="store_true")
    parser.add_argument("--random-prefix-len", type=int, default=0)
    parser.add_argument("--burstiness", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.no_max_concurrency and args.max_concurrency is not None:
        parser.error("pass only one of --max-concurrency and --no-max-concurrency")
    if args.max_concurrency is None and not args.no_max_concurrency:
        parser.error("--max-concurrency is required unless --no-max-concurrency is set")
    built = bench_serve_args(
        model=args.model,
        input_len=args.input_len,
        output_len=args.output_len,
        num_prompts=args.num_prompts,
        max_concurrency=None if args.no_max_concurrency else args.max_concurrency,
        host=None if args.no_host else args.host,
        port=None if args.no_port else args.port,
        backend=None if args.no_backend else (args.backend or None),
        endpoint=None if args.no_endpoint else (args.endpoint or None),
        tokenizer=args.tokenizer or None,
        request_rate=None if args.no_request_rate else (args.request_rate or None),
        ignore_eos=args.ignore_eos,
        temperature=args.temperature,
        result_dir=args.result_dir or None,
        result_filename=args.result_filename or None,
        percentile_metrics=None if args.no_percentile_metrics else (args.percentile_metrics or None),
        metric_percentiles=None if args.no_metric_percentiles else (args.metric_percentiles or None),
        save_detailed=not args.no_save_detailed,
        random_prefix_len=args.random_prefix_len or None,
        burstiness=args.burstiness,
        seed=args.seed,
        served_model_name=args.served_model_name or None,
        trust_remote_code=args.trust_remote_code,
        num_warmups=args.num_warmups,
    )
    sys.stdout.write("\n".join(built) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
