#!/usr/bin/env python3
"""Collect report-grade Qwen3.8-27B MXFP4 tail-latency distributions.

The default 1,000 requests per cell gives 0.1% empirical tail resolution.
Output length is intentionally short: this is a TTFT/tail test, not a token
throughput sweep. Raw vLLM detail is written to _results and immediately
normalized into durable docs/results plus docs/profiling.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import bench_serve
from publish_latency_results import publish_latency_result

ROOT = Path(__file__).resolve().parents[1]


def short(value: int) -> str:
    return f"{value // 1024}k" if value % 1024 == 0 else str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="rocm-mxfp4-server")
    parser.add_argument("--model", default="Qwen3.8-27B-Quark-AWQ-MXFP4")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B-FP8")
    parser.add_argument("--gpu-profile", default="mi350p")
    parser.add_argument("--input-lens", nargs="+", type=int, default=[1024, 8192])
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--concurrency-list", nargs="+", type=int, default=[1, 16, 64])
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--result-subdir", default="qwen_tail_latency")
    args = parser.parse_args()
    if args.num_prompts < 100:
        parser.error("--num-prompts must be at least 100 to estimate p99")

    host_dir = ROOT / "_results" / args.result_subdir
    container_dir = Path("/results") / args.result_subdir
    host_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "gpu_profile": args.gpu_profile,
        "input_lens": args.input_lens,
        "output_len": args.output_len,
        "concurrency_list": args.concurrency_list,
        "num_prompts": args.num_prompts,
        "request_rate": args.request_rate,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "runs": [],
    }

    for input_len in args.input_lens:
        for concurrency in args.concurrency_list:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = (
                f"qwen_isl{input_len}_osl{args.output_len}_"
                f"c{concurrency}_n{args.num_prompts}_{stamp}.json"
            )
            vllm_bin = "/usr/local/bin/vllm"
            probe = subprocess.run(
                ["docker", "exec", args.container, "test", "-x", "/opt/vllm/bin/vllm"],
                check=False,
            )
            if probe.returncode == 0:
                vllm_bin = "/opt/vllm/bin/vllm"
            cmd = [
                "docker",
                "exec",
                args.container,
                vllm_bin,
                *bench_serve.bench_serve_args(
                    model=args.model,
                    tokenizer=args.tokenizer,
                    input_len=input_len,
                    output_len=args.output_len,
                    num_prompts=args.num_prompts,
                    max_concurrency=concurrency,
                    request_rate=args.request_rate,
                    ignore_eos=True,
                    temperature=0,
                    result_dir=str(container_dir),
                    result_filename=filename,
                ),
            ]
            print(
                f"Qwen tail: {input_len}:{args.output_len} C{concurrency}, "
                f"{args.num_prompts} requests",
                flush=True,
            )
            completed = subprocess.run(cmd, text=True)
            if completed.returncode:
                raise RuntimeError(f"vLLM tail benchmark failed for {filename}")
            source = host_dir / filename
            if not source.is_file():
                raise FileNotFoundError(
                    f"{source} was not written; verify /results maps to {ROOT / '_results'}"
                )
            raw_path, profile_path = publish_latency_result(
                source,
                gpu_profile=args.gpu_profile,
                workload=f"{short(input_len)}{short(args.output_len)}",
                concurrency=concurrency,
            )
            manifest["runs"].append(
                {
                    "input_len": input_len,
                    "output_len": args.output_len,
                    "concurrency": concurrency,
                    "scratch": str(source.relative_to(ROOT)),
                    "published_raw": str(raw_path.relative_to(ROOT)),
                    "published_profile": str(profile_path.relative_to(ROOT)),
                }
            )

    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path = host_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"scratch manifest: {manifest_path}")
    print("durable profile: docs/profiling/qwen3.8-27b-mxfp4-latency.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
