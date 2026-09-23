#!/usr/bin/env python3
"""
GPQA (Graduate-Level Google-Proof Q&A) Benchmark Runner for AMD ROCm
====================================================================
This script evaluates complex scientific and technical reasoning capabilities
of local models hosted on an AMD ROCm inference server (vLLM or SGLang)
using the GPQA benchmark (Diamond, Main, Extended, or local samples).

Key Features:
- Deterministic option shuffling (A, B, C, D) with seeded randomization
- Chain-of-thought (CoT) prompting with robust final answer extraction
- Domain-specific accuracy breakdowns (Physics, Chemistry, Biology)
- Inference latency, reasoning token lengths, and throughput (tok/s) tracking
"""

import argparse
import json
import os
import random
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


SYSTEM_PROMPT = """You are a world-class scientist and domain expert in Physics, Chemistry, and Biology.
Carefully read the question, analyze the underlying equations and mechanisms, think step by step, and select the best answer.
At the very end of your response, state your final choice in the exact format:
Final Answer: (X)
where X is A, B, C, or D.
"""


def extract_choice(response_text: str) -> str | None:
    """Extract final multiple choice letter (A, B, C, or D) from model response."""
    # 1. Match 'Final Answer: (X)' or 'Final Answer: X'
    match = re.search(r"Final\s+Answer:\s*\(?([A-D])\)?", response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 2. Match 'Answer is (X)' or 'Answer: (X)'
    match = re.search(r"(?:The\s+correct\s+)?answer\s*(?:is|:)\s*\(?([A-D])\)?", response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 3. Match 'option (X)' or 'choice (X)'
    matches = re.findall(r"(?:option|choice)\s+\(?([A-D])\)?", response_text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    # 4. Match trailing (X) near the end of the text
    tail = response_text.strip()[-200:]
    match = re.search(r"\b([A-D])\b(?=[^\w]*\Z)", tail)
    if match:
        return match.group(1).upper()

    return None


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


def load_gpqa_dataset(subset: str, split: str = "train", num_samples: int | None = None, seed: int = 42):
    """Load GPQA questions from Hugging Face or local sample file."""
    if subset == "sample":
        sample_path = Path(__file__).parent / "sample_gpqa.json"
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample GPQA file not found at {sample_path}")
        with open(sample_path, "r", encoding="utf-8") as f:
            instances = json.load(f)
        if num_samples:
            instances = instances[:num_samples]
        return instances

    if subset.endswith(".json") or subset.endswith(".csv"):
        path = Path(subset)
        if not path.exists():
            raise FileNotFoundError(f"File not found at {path}")
        if subset.endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(path)
            instances = df.to_dict(orient="records")
        else:
            with open(path, "r", encoding="utf-8") as f:
                instances = json.load(f)
        if num_samples:
            instances = instances[:num_samples]
        return instances

    # Load from Hugging Face datasets (Idavidrein/gpqa)
    try:
        from datasets import load_dataset
        hf_subset = subset if subset.startswith("gpqa_") else f"gpqa_{subset}"
        print(f"Downloading/loading GPQA dataset: Idavidrein/gpqa ({hf_subset}, split={split})...")
        token = os.environ.get("HF_TOKEN")
        ds = load_dataset("Idavidrein/gpqa", hf_subset, split=split, token=token)
        instances = list(ds)
        if num_samples:
            instances = instances[:num_samples]
        return instances
    except Exception as e:
        print(f"Error loading Hugging Face dataset: {e}")
        print("Tip: Use '--subset sample' for zero-network testing.")
        sys.exit(1)


def build_multiple_choice(instance: dict, rng: random.Random) -> tuple[str, str, dict]:
    """Format question and randomize options A, B, C, D."""
    question = instance.get("Question", "")
    correct = str(instance.get("Correct Answer", "")).strip()
    distractors = [
        str(instance.get("Incorrect Answer 1", "")).strip(),
        str(instance.get("Incorrect Answer 2", "")).strip(),
        str(instance.get("Incorrect Answer 3", "")).strip(),
    ]
    
    # Shuffle options
    options = [correct] + distractors
    rng.shuffle(options)
    
    labels = ["A", "B", "C", "D"]
    choices_dict = {labels[i]: options[i] for i in range(4)}
    
    # Find which label corresponds to the correct answer
    ground_truth_label = next(lbl for lbl, opt in choices_dict.items() if opt == correct)
    
    prompt = f"{question}\n\n"
    for lbl in labels:
        prompt += f"({lbl}) {choices_dict[lbl]}\n"
    prompt += "\nProvide your step-by-step reasoning and conclude with: Final Answer: (X)"
    
    return prompt, ground_truth_label, choices_dict


def run_gpqa_benchmark(args):
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    rng = random.Random(args.seed)

    print("=" * 75)
    print(" GPQA (Graduate-Level Scientific Reasoning) Benchmark")
    print("=" * 75)
    print(f"Server Base URL: {args.base_url}")
    
    model_name = get_active_model(client, args.model)
    print(f"Target Model   : {model_name}")
    print(f"Subset / Split : {args.subset} ({args.split})")
    print(f"Random Seed    : {args.seed}")
    
    instances = load_gpqa_dataset(args.subset, split=args.split, num_samples=args.num_samples, seed=args.seed)
    print(f"Total Questions: {len(instances)}")
    print("-" * 75)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_model = model_name.replace("/", "_").replace(":", "_")
    run_dir = Path(args.output_dir) / f"{sanitized_model}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results_file = run_dir / "gpqa_detailed_results.jsonl"
    summary_file = run_dir / "gpqa_summary.json"

    correct_count = 0
    parsed_count = 0
    total_tokens_generated = 0
    total_latency_seconds = 0.0

    domain_stats = {}

    with open(results_file, "w", encoding="utf-8") as out_f:
        for idx, instance in enumerate(instances, 1):
            domain = instance.get("High-level domain", instance.get("domain", "General"))
            subdomain = instance.get("Subdomain", "")
            
            if domain not in domain_stats:
                domain_stats[domain] = {"total": 0, "correct": 0}
            domain_stats[domain]["total"] += 1

            prompt_text, true_label, choices = build_multiple_choice(instance, rng)
            print(f"[{idx}/{len(instances)}] [{domain}] Question {idx} ...", end="", flush=True)

            start_time = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_text},
                    ],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                elapsed = time.perf_counter() - start_time
                content = response.choices[0].message.content or ""

                usage = response.usage
                completion_tokens = usage.completion_tokens if usage else len(content.split())
                prompt_tokens = usage.prompt_tokens if usage else len(prompt_text.split())
                throughput = completion_tokens / elapsed if elapsed > 0 else 0

                pred_label = extract_choice(content)
                is_correct = (pred_label == true_label)

                if pred_label is not None:
                    parsed_count += 1
                if is_correct:
                    correct_count += 1
                    domain_stats[domain]["correct"] += 1

                total_tokens_generated += completion_tokens
                total_latency_seconds += elapsed

                status_str = "CORRECT" if is_correct else f"INCORRECT (Pred: {pred_label}, True: {true_label})"
                print(f" {status_str} ({elapsed:.2f}s, {throughput:.1f} tok/s)")

                record = {
                    "question_index": idx,
                    "domain": domain,
                    "subdomain": subdomain,
                    "question": instance.get("Question", ""),
                    "choices": choices,
                    "ground_truth": true_label,
                    "predicted_label": pred_label,
                    "is_correct": is_correct,
                    "latency_sec": round(elapsed, 2),
                    "throughput_tok_per_sec": round(throughput, 2),
                    "completion_tokens": completion_tokens,
                    "raw_response": content,
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

            except Exception as e:
                elapsed = time.perf_counter() - start_time
                print(f" FAILED ({elapsed:.2f}s) - Error: {e}")

    total_q = len(instances)
    accuracy_pct = round((correct_count / total_q) * 100, 2) if total_q > 0 else 0.0
    parse_rate_pct = round((parsed_count / total_q) * 100, 2) if total_q > 0 else 0.0
    avg_throughput = round(total_tokens_generated / total_latency_seconds, 2) if total_latency_seconds > 0 else 0.0
    avg_latency = round(total_latency_seconds / total_q, 2) if total_q > 0 else 0.0

    domain_summary = {}
    for d, st in domain_stats.items():
        domain_summary[d] = {
            "total": st["total"],
            "correct": st["correct"],
            "accuracy_pct": round((st["correct"] / st["total"]) * 100, 2) if st["total"] > 0 else 0.0,
        }

    summary = {
        "model_name": model_name,
        "benchmark": "GPQA",
        "subset": args.subset,
        "total_questions": total_q,
        "correct_answers": correct_count,
        "accuracy_pct": accuracy_pct,
        "parsed_answers_pct": parse_rate_pct,
        "average_throughput_tok_per_sec": avg_throughput,
        "average_latency_sec": avg_latency,
        "total_tokens_generated": total_tokens_generated,
        "domain_breakdown": domain_summary,
        "timestamp": timestamp,
        "results_file": str(results_file.resolve()),
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print(" GPQA BENCHMARK PERFORMANCE SUMMARY")
    print("=" * 75)
    print(f" Model Evaluated           : {model_name}")
    print(f" Subset Tested             : {args.subset}")
    print(f" Overall Accuracy          : {accuracy_pct}% ({correct_count}/{total_q})")
    print(f" Answer Parsing Rate       : {parse_rate_pct}%")
    print(f" Average Throughput        : {avg_throughput:.2f} tokens/second")
    print(f" Average Latency / Question: {avg_latency:.2f} seconds")
    print("-" * 75)
    print(" Domain Accuracy Breakdown:")
    for d, st in domain_summary.items():
        print(f"  • {d:<18}: {st['accuracy_pct']:>6.2f}% ({st['correct']}/{st['total']})")
    print("-" * 75)
    print(f" Detailed Output Log       : {results_file}")
    print(f" Summary Metric Report     : {summary_file}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="GPQA Scientific Reasoning Benchmark for AMD ROCm Local Models")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="dummy", help="API key")
    parser.add_argument("--model", default=None, help="Model name (auto-detected from server if omitted)")
    parser.add_argument("--subset", "--dataset", dest="subset", default="sample", help="GPQA subset: 'sample', 'gpqa_diamond', 'gpqa_main', 'gpqa_extended', or local file")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of questions to evaluate")
    parser.add_argument("--output-dir", default="_results/gpqa", help="Output directory")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max reasoning and answer tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Seed for choice randomization")

    args = parser.parse_args()
    run_gpqa_benchmark(args)


if __name__ == "__main__":
    main()
