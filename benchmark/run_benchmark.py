#!/usr/bin/env python3
"""
SWE-bench Model Evaluation & Benchmarking Script for AMD ROCm
=============================================================
This script benchmarks coding models hosted on an AMD ROCm inference server
(vLLM or SGLang) using the SWE-bench evaluation benchmark.

Key Features:
- Evaluates models on SWE-bench Lite, SWE-bench Verified, or local sample sets
- Measures token throughput (tokens/sec), generation latency, and patch validity
- Generates standard SWE-bench predictions.jsonl for evaluation harness
- Optionally runs official swebench.harness.run_evaluation via Docker
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Error: 'openai' package not found. Install with: pip install openai")
    sys.exit(1)


SYSTEM_PROMPT = """You are an expert software engineer resolving a real-world issue in a Python repository.
Given the problem description and repository details below, output a unified git diff that resolves the issue.

Guidelines:
1. Provide ONLY a valid git unified diff inside a ```diff code block.
2. The diff must be applicable via `git apply` against the base commit.
3. Include clear file headers (--- a/path/to/file and +++ b/path/to/file).
4. Do not include commentary, explanations, or text outside the diff block.
"""


def extract_patch(response_text: str) -> str:
    """Extract clean unified git diff patch from model response, stripping thinking blocks."""
    # Strip <think>...</think> blocks if present
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", response_text).strip()

    # 1. Match fenced ```diff block
    fenced_match = re.search(r"```(?:diff)?\s*\n([\s\S]*?)\n```", cleaned)
    if fenced_match:
        inner = fenced_match.group(1).strip()
        if "--- a/" in inner or "diff --git" in inner:
            return inner + "\n"

    # 2. Match diff --git block
    diff_git_match = re.search(r"(diff --git[\s\S]*?)(?:\n\n[A-Z]|\Z)", cleaned)
    if diff_git_match:
        return diff_git_match.group(1).strip() + "\n"

    # 3. Match from --- a/ ... +++ b/
    patch_header_match = re.search(r"(--- a/[\s\S]*?\n\+\+\+ b/[\s\S]*?)(?:\n\n[A-Z]|\Z)", cleaned)
    if patch_header_match:
        return patch_header_match.group(1).strip() + "\n"

    return ""


def get_active_model(client: OpenAI, requested_model: str | None = None) -> str:
    """Detect or verify active model on the inference server."""
    try:
        models = client.models.list()
        available_ids = [m.id for m in models.data]
        if not available_ids:
            raise RuntimeError("No models returned by server.")
        
        if requested_model:
            if requested_model in available_ids:
                return requested_model
            print(f"Warning: Requested model '{requested_model}' not in available list: {available_ids}")
            print(f"Defaulting to available model: '{available_ids[0]}'")
            return available_ids[0]
        
        return available_ids[0]
    except Exception as e:
        if requested_model:
            return requested_model
        raise RuntimeError(f"Could not connect to model server: {e}")


def load_benchmark_dataset(dataset_name: str, split: str = "test", num_samples: int | None = None):
    """Load SWE-bench instances from Hugging Face or local sample file."""
    if dataset_name == "sample":
        sample_path = Path(__file__).parent / "sample_instances.json"
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample instances file not found at {sample_path}")
        with open(sample_path, "r", encoding="utf-8") as f:
            instances = json.load(f)
        if num_samples:
            instances = instances[:num_samples]
        return instances

    if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
        path = Path(dataset_name)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found at {path}")
        instances = []
        if dataset_name.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        instances.append(json.loads(line))
        else:
            with open(path, "r", encoding="utf-8") as f:
                instances = json.load(f)
        if num_samples:
            instances = instances[:num_samples]
        return instances

    # Load from Hugging Face datasets
    try:
        from datasets import load_dataset
        print(f"Downloading/loading SWE-bench dataset: {dataset_name} ({split} split)...")
        ds = load_dataset(dataset_name, split=split)
        instances = list(ds)
        if num_samples:
            instances = instances[:num_samples]
        return instances
    except ImportError:
        print("Error: 'datasets' package required for Hugging Face datasets. Install with: pip install datasets")
        sys.exit(1)


def format_user_prompt(instance: dict) -> str:
    """Format single instance problem statement into evaluation prompt."""
    repo = instance.get("repo", "Unknown repository")
    problem = instance.get("problem_statement", "")
    hints = instance.get("hints_text", "")

    prompt = f"Repository: {repo}\n\nProblem Description:\n{problem}\n"
    if hints:
        prompt += f"\nHints / Context:\n{hints}\n"
    prompt += "\nGenerate the unified git diff to resolve this issue:"
    return prompt


def run_benchmark(args):
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    
    print("=" * 75)
    print(" SWE-bench Model Evaluation & Performance Benchmark")
    print("=" * 75)
    print(f"Server Base URL: {args.base_url}")
    
    model_name = get_active_model(client, args.model)
    print(f"Target Model   : {model_name}")
    print(f"Dataset        : {args.dataset}")
    
    instances = load_benchmark_dataset(args.dataset, split=args.split, num_samples=args.num_samples)
    print(f"Total Instances: {len(instances)}")
    print("-" * 75)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_model = model_name.replace("/", "_").replace(":", "_")
    run_dir = Path(args.output_dir) / f"{sanitized_model}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_file = run_dir / "predictions.jsonl"
    metrics_file = run_dir / "benchmark_metrics.json"

    results = []
    total_tokens_generated = 0
    total_latency_seconds = 0.0
    valid_patch_count = 0

    with open(predictions_file, "w", encoding="utf-8") as pred_out:
        for idx, instance in enumerate(instances, 1):
            instance_id = instance.get("instance_id", f"instance_{idx}")
            print(f"[{idx}/{len(instances)}] Evaluating instance: {instance_id} ...", end="", flush=True)

            user_prompt = format_user_prompt(instance)

            start_time = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                elapsed = time.perf_counter() - start_time
                content = response.choices[0].message.content or ""
                
                # Extract usage statistics
                usage = response.usage
                completion_tokens = usage.completion_tokens if usage else len(content.split())
                prompt_tokens = usage.prompt_tokens if usage else len(user_prompt.split())
                throughput = completion_tokens / elapsed if elapsed > 0 else 0

                patch = extract_patch(content)
                has_patch = bool(patch and "--- a/" in patch)
                if has_patch:
                    valid_patch_count += 1

                record = {
                    "instance_id": instance_id,
                    "model_patch": patch,
                    "model_name_or_path": model_name,
                }
                pred_out.write(json.dumps(record) + "\n")
                pred_out.flush()

                results.append({
                    "instance_id": instance_id,
                    "latency_sec": round(elapsed, 2),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "tokens_per_sec": round(throughput, 2),
                    "valid_patch_format": has_patch,
                })

                total_tokens_generated += completion_tokens
                total_latency_seconds += elapsed

                status = f" OK ({elapsed:.2f}s, {throughput:.1f} tok/s, {'Patch valid' if has_patch else 'No patch'})"
                print(status)

            except Exception as err:
                elapsed = time.perf_counter() - start_time
                print(f" FAILED ({elapsed:.2f}s) - Error: {err}")
                results.append({
                    "instance_id": instance_id,
                    "latency_sec": round(elapsed, 2),
                    "error": str(err),
                    "valid_patch_format": False,
                })

    avg_throughput = (total_tokens_generated / total_latency_seconds) if total_latency_seconds > 0 else 0
    avg_latency = (total_latency_seconds / len(instances)) if instances else 0

    metrics_summary = {
        "model_name": model_name,
        "dataset": args.dataset,
        "total_instances": len(instances),
        "valid_patches_generated": valid_patch_count,
        "patch_formatting_rate_pct": round((valid_patch_count / len(instances)) * 100, 2) if instances else 0,
        "total_tokens_generated": total_tokens_generated,
        "total_latency_seconds": round(total_latency_seconds, 2),
        "avg_tokens_per_sec": round(avg_throughput, 2),
        "avg_latency_per_sample_sec": round(avg_latency, 2),
        "predictions_file": str(predictions_file.resolve()),
        "timestamp": timestamp,
        "results": results,
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\n" + "=" * 75)
    print(" BENCHMARK PERFORMANCE SUMMARY")
    print("=" * 75)
    print(f" Model Tested              : {model_name}")
    print(f" Total Instances Evaluated : {len(instances)}")
    print(f" Valid Git Patches Created : {valid_patch_count} ({metrics_summary['patch_formatting_rate_pct']}%)")
    print(f" Average Throughput        : {avg_throughput:.2f} tokens/second")
    print(f" Average Latency per Sample: {avg_latency:.2f} seconds")
    print(f" Output Predictions File   : {predictions_file}")
    print(f" Metrics Report File       : {metrics_file}")
    print("=" * 75)

    if args.run_evaluation:
        print("\n" + "=" * 75)
        print(" EXECUTING SWE-BENCH DOCKER EVALUATION HARNESS")
        print("=" * 75)
        
        if args.dataset == "sample":
            print("[Notice] The 'sample' dataset is an offline smoke-test subset.")
            print("For full unit test execution against repo Docker images, run with:")
            print(f"  --dataset princeton-nlp/SWE-bench_Lite --run-evaluation\n")
            print("Patch Validity Score on Sample Set: "
                  f"{metrics_summary['patch_formatting_rate_pct']}% "
                  f"({valid_patch_count}/{len(instances)} patches formatted correctly)")
        else:
            eval_run_id = f"{sanitized_model}_eval"
            cmd = [
                sys.executable, "-m", "swebench.harness.run_evaluation",
                "-d", args.dataset,
                "-s", args.split,
                "-p", str(predictions_file.resolve()),
                "-id", eval_run_id,
                "--report_dir", str(run_dir.resolve()),
                "--max_workers", str(args.max_workers),
            ]
            print(f"Running command: {' '.join(cmd)}")
            try:
                import subprocess
                subprocess.run(cmd, check=True)
                
                # Check for generated report
                report_files = list(run_dir.glob(f"*{eval_run_id}*.json")) + list(Path(".").glob(f"*{eval_run_id}*.json"))
                if report_files:
                    with open(report_files[0], "r", encoding="utf-8") as rf:
                        eval_data = json.load(rf)
                    
                    total_inst = eval_data.get("total_instances", len(instances))
                    resolved = eval_data.get("resolved_instances", 0)
                    if isinstance(resolved, list):
                        resolved = len(resolved)
                    res_rate = (resolved / total_inst * 100) if total_inst > 0 else 0.0

                    print("\n" + "=" * 75)
                    print(" SWE-BENCH FUNCTIONAL RESOLUTION SCORE")
                    print("=" * 75)
                    print(f" Model Evaluated       : {model_name}")
                    print(f" Total Problems Tested : {total_inst}")
                    print(f" Issues Resolved (Pass): {resolved}")
                    print(f" FINAL RESOLUTION SCORE: {res_rate:.2f}% RESOLVED")
                    print("=" * 75)
            except Exception as e:
                print(f"Evaluation harness execution encountered an issue: {e}")
                print("Verify Docker socket is mounted (-v /var/run/docker.sock:/var/run/docker.sock)")


def main():
    parser = argparse.ArgumentParser(description="SWE-bench Benchmark Runner for AMD ROCm Local Models")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="dummy", help="API key")
    parser.add_argument("--model", default=None, help="Model name (auto-detected from server if omitted)")
    parser.add_argument("--dataset", default="sample", help="Dataset: 'sample', 'princeton-nlp/SWE-bench_Lite', 'princeton-nlp/SWE-bench_Verified', or local JSON/JSONL path")
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of instances to evaluate (default: all)")
    parser.add_argument("--output-dir", default="benchmark_results", help="Directory to store outputs")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens to generate per solution")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--run-evaluation", action="store_true", help="Print or execute swebench.harness evaluation")
    parser.add_argument("--max-workers", type=int, default=4, help="Max parallel worker containers for evaluation")
    
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
