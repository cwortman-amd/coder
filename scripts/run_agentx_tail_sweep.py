#!/usr/bin/env python3
"""Run a bounded, faithful AgentX concurrency sweep against a live endpoint."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AIPERF = ROOT.parent / ".venv-aiperf" / "bin" / "aiperf"
DEFAULT_TOKENIZER = ROOT / "models" / "Qwen3.8-27B-Quark-AWQ-MXFP4"
DEFAULT_HF_HOME = ROOT / "_results" / "hf-aiperf"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(path)


def endpoint_model(base_url: str, timeout: float = 10.0) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/models",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"vLLM endpoint is unavailable at {base_url}: {exc}") from exc


def terminate_process(process: subprocess.Popen[str], grace_seconds: float = 20.0) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen3.8-27B-Quark-AWQ-MXFP4")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--aiperf", type=Path, default=DEFAULT_AIPERF)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 8, 16])
    parser.add_argument("--duration-seconds", type=int, default=900)
    parser.add_argument("--campaign-seconds", type=int, default=3600)
    parser.add_argument("--random-seed", type=int, default=20260707)
    parser.add_argument("--max-context-length", type=int, default=65536)
    parser.add_argument(
        "--dataset",
        default="semianalysis_cc_traces_weka_062126",
    )
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "_results" / "agentx_tail_sweep",
    )
    args = parser.parse_args()

    if args.duration_seconds < 900:
        parser.error("AgentX requires --duration-seconds >= 900")
    if args.campaign_seconds < args.duration_seconds:
        parser.error("campaign budget must fit at least one profiling window")
    if not args.concurrency or any(value < 1 for value in args.concurrency):
        parser.error("concurrency values must be positive")
    if not args.aiperf.is_file():
        parser.error(f"AIPerf executable does not exist: {args.aiperf}")
    if not args.tokenizer.is_dir():
        parser.error(f"tokenizer directory does not exist: {args.tokenizer}")

    model_document = endpoint_model(args.url)
    served_models = [
        row.get("id")
        for row in model_document.get("data", [])
        if isinstance(row, dict)
    ]
    if args.model not in served_models:
        parser.error(
            f"model {args.model!r} is not served by {args.url}; found {served_models}"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    campaign_dir = args.output_root / stamp
    campaign_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = campaign_dir / "manifest.json"
    deadline = time.monotonic() + args.campaign_seconds
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "agentx_concurrency_tail_pilot",
        "started_utc": utc_now(),
        "campaign_budget_seconds": args.campaign_seconds,
        "profiling_duration_seconds": args.duration_seconds,
        "concurrency_points": args.concurrency,
        "url": args.url,
        "model": args.model,
        "served_models": served_models,
        "tokenizer": str(args.tokenizer),
        "dataset": args.dataset,
        "max_context_length": args.max_context_length,
        "random_seed": args.random_seed,
        "aiperf": str(args.aiperf),
        "runs": [],
    }
    write_json(manifest_path, manifest)

    env = os.environ.copy()
    env["HF_HOME"] = str(args.hf_home)
    env["HF_DATASETS_CACHE"] = str(args.hf_home / "datasets")
    args.hf_home.mkdir(parents=True, exist_ok=True)

    for concurrency in args.concurrency:
        remaining = deadline - time.monotonic()
        # Keep a small allowance for startup and graceful export. If this is not
        # available, leave a machine-readable skipped cell instead of starting a
        # profiling window that cannot meet the campaign deadline.
        if remaining < args.duration_seconds + 30:
            manifest["runs"].append(
                {
                    "concurrency": concurrency,
                    "status": "skipped_budget",
                    "remaining_budget_seconds": round(max(remaining, 0), 3),
                }
            )
            write_json(manifest_path, manifest)
            continue

        run_dir = campaign_dir / f"c{concurrency}"
        run_dir.mkdir()
        command = [
            str(args.aiperf),
            "profile",
            "--scenario",
            "inferencex-agentx-mvp",
            "--url",
            args.url,
            "--model",
            args.model,
            "--tokenizer",
            str(args.tokenizer),
            "--max-context-length",
            str(args.max_context_length),
            "--endpoint-type",
            "chat",
            "--public-dataset",
            args.dataset,
            "--concurrency",
            str(concurrency),
            "--use-server-token-count",
            "--benchmark-duration",
            str(args.duration_seconds),
            "--benchmark-grace-period",
            "30",
            "--random-seed",
            str(args.random_seed),
            "--artifact-dir",
            str(run_dir),
            "--ui",
            "simple",
        ]
        run_record: dict[str, Any] = {
            "concurrency": concurrency,
            "status": "running",
            "started_utc": utc_now(),
            "artifact_dir": str(run_dir),
            "command": command,
        }
        manifest["runs"].append(run_record)
        write_json(manifest_path, manifest)
        started = time.monotonic()
        log_path = run_dir / "campaign_console.log"
        print(
            f"[{run_record['started_utc']}] AgentX C={concurrency}; "
            f"{remaining:.0f}s campaign budget remains",
            flush=True,
        )
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            timed_out = False
            try:
                return_code = process.wait(timeout=max(deadline - time.monotonic(), 1))
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate_process(process)
                return_code = process.returncode

        run_record.update(
            {
                "completed_utc": utc_now(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "return_code": return_code,
                "status": (
                    "campaign_timeout"
                    if timed_out
                    else "completed"
                    if return_code == 0
                    else "failed"
                ),
                "console_log": str(log_path),
            }
        )
        write_json(manifest_path, manifest)
        print(
            f"AgentX C={concurrency}: {run_record['status']} in "
            f"{run_record['elapsed_seconds']:.1f}s",
            flush=True,
        )
        if run_record["status"] != "completed":
            break

    manifest["completed_utc"] = utc_now()
    manifest["elapsed_seconds"] = round(
        args.campaign_seconds - max(deadline - time.monotonic(), 0), 3
    )
    statuses = [run["status"] for run in manifest["runs"]]
    manifest["status"] = (
        "completed"
        if statuses and all(status == "completed" for status in statuses)
        else "partial"
    )
    write_json(manifest_path, manifest)
    print(f"Campaign manifest: {manifest_path}")
    return 0 if manifest["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
