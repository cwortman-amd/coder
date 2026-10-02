#!/usr/bin/env python3
"""Run one named campaign from config/campaigns.yaml.

test.sh --experiments calls this with tail-matrix or tail-quick. The campaign
file owns concurrency lists, idle intervals, prompt counts, and SLO gates.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS_PATH = ROOT / "config" / "campaigns.yaml"


def load_campaigns() -> dict[str, Any]:
    with CAMPAIGNS_PATH.open() as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict) or "campaigns" not in loaded:
        raise SystemExit(f"{CAMPAIGNS_PATH} has no campaigns")
    return loaded


def campaign(name: str) -> dict[str, Any]:
    data = load_campaigns()
    block = data["campaigns"].get(name)
    if not isinstance(block, dict):
        known = ", ".join(sorted(data["campaigns"]))
        raise SystemExit(f"Unknown campaign {name}. Known campaigns: {known}")
    block = dict(block)
    block["name"] = name
    block["slo"] = data.get("slo") or {}
    return block


def _ints(values: list[Any]) -> list[str]:
    return [str(int(value)) for value in values]


def _run(cmd: list[str], env: dict[str, str] | None = None) -> int:
    print("+", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=ROOT, env=env)
    return completed.returncode


def _running_container() -> str:
    sys.path.insert(0, str(ROOT / "scripts"))
    import catalog

    wanted = [catalog.container(key)["name"] for key in ("inference", "mxfp4")]
    listed = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    running = set((listed.stdout or "").split())
    for name in wanted:
        if name in running:
            return name
    return ""


def run_campaign(args: argparse.Namespace) -> int:
    spec = campaign(args.campaign)
    slo = (spec.get("slo") or {}).get("tail") or {}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    api = args.api or args.url.rstrip("/") + "/v1"
    py = sys.executable
    steps = list(spec.get("steps") or [])

    if "closed_loop" in steps:
        row = spec["closed_loop"]
        print("Closed-loop concurrency and synthetic agent chains", flush=True)
        env = os.environ.copy()
        env.update(
            {
                "CONCURRENCY": " ".join(_ints(row["concurrency"])),
                "REQUESTS_PER_CELL": str(row["requests_per_cell"]),
                "NUM_CHAINS": str(row["num_chains"]),
                "CHAIN_LENGTH": str(row["chain_length"]),
                "TOKENS_PER_CALL": str(row["tokens_per_call"]),
                "BASE_URL": api,
                "MODEL": args.model,
                "OUTPUT_ROOT": str(out / "tail"),
                "TTFT_DEADLINE_MS": str(slo.get("ttft_ms", 1000)),
                "TASK_DEADLINE_S": str(slo.get("task_s", 30)),
            }
        )
        rc = _run([str(ROOT / "scripts" / "run_tail_latency_study.sh")], env)
        if rc:
            return rc

    if "open_loop" in steps:
        row = spec["open_loop"]
        print("Open-loop offered load and SLO goodput", flush=True)
        rc = _run(
            [
                py,
                str(ROOT / "scripts" / "bench_open_loop_sweep.py"),
                "--url",
                api,
                "--model",
                args.model,
                "--qps-list",
                *[str(value) for value in row["qps"]],
                "--duration",
                str(row["duration_s"]),
                "--tokens",
                str(row["tokens"]),
                "--ttft-slo-ms",
                str(slo.get("ttft_ms", 1000)),
                "--worst-itl-slo-ms",
                str(slo.get("worst_itl_ms", 100)),
                "--out-dir",
                str(out / "open_loop"),
            ]
        )
        if rc:
            return rc

    if "cold_start" in steps:
        row = spec["cold_start"]
        print("Cold-start idle intervals", flush=True)
        cmd = [
            py,
            str(ROOT / "scripts" / "bench_cold_start_probe.py"),
            "--url",
            api,
            "--model",
            args.model,
            "--idle-intervals",
            *_ints(row["idle_intervals"]),
            "--out",
            str(out / "cold_start.json"),
        ]
        if row.get("warm_reps") is not None:
            cmd += ["--warm-reps", str(row["warm_reps"])]
        rc = _run(cmd)
        if rc:
            return rc

    if "interference" in steps:
        row = spec["interference"]
        print("Collocated prefill/decode interference", flush=True)
        rc = _run(
            [
                py,
                str(ROOT / "scripts" / "bench_prefill_decode_interference.py"),
                "--url",
                api,
                "--model",
                args.model,
                "--decode-streams",
                str(row["decode_streams"]),
                "--decode-tokens",
                str(row["decode_tokens"]),
                "--prefill-tokens",
                str(row["prefill_tokens"]),
                "--burst-size",
                str(row["burst_size"]),
                "--out-dir",
                str(out / "interference"),
            ]
        )
        if rc:
            return rc

    if "ttft_grid" in steps:
        row = spec["ttft_grid"]
        print("Input-length TTFT grid", flush=True)
        container = args.container or _running_container()
        if not container:
            print(
                "Input-length TTFT grid needs a running inference or MXFP4 container.",
                file=sys.stderr,
            )
            return 1
        rc = _run(
            [
                py,
                str(ROOT / "scripts" / "bench_qwen_tail_latency.py"),
                "--container",
                container,
                "--model",
                args.model,
                "--gpu-profile",
                args.gpu_profile,
                "--input-lens",
                *_ints(row["input_lens"]),
                "--output-len",
                str(row["output_len"]),
                "--concurrency-list",
                *_ints(row["concurrency"]),
                "--num-prompts",
                str(row["num_prompts"]),
                "--result-subdir",
                f"experiments/{out.name}/qwen_tail",
            ]
        )
        if rc:
            return rc

    if "agentx" in steps:
        row = spec["agentx"]
        print("AgentX conversation replay", flush=True)
        aiperf = Path(args.aiperf)
        tokenizer = Path(args.tokenizer)
        if not aiperf.is_file() or not tokenizer.is_dir():
            print(f"AgentX replay needs {aiperf} and {tokenizer}.", file=sys.stderr)
            return 1
        cmd = [
            py,
            str(ROOT / "scripts" / "run_agentx_tail_sweep.py"),
            "--url",
            args.url,
            "--model",
            args.model,
            "--aiperf",
            str(aiperf),
            "--tokenizer",
            str(tokenizer),
            "--concurrency",
            *_ints(row["concurrency"]),
            "--duration-seconds",
            str(row["duration_seconds"]),
            "--campaign-seconds",
            str(row["campaign_seconds"]),
            "--output-root",
            str(out / "agentx"),
        ]
        if row.get("allow_short_duration"):
            cmd.append("--allow-short-duration")
        rc = _run(cmd)
        if rc:
            return rc

    if "power_sweep" in steps:
        row = spec["power_sweep"]
        print("Power-of-two concurrency sweep", flush=True)
        container = args.container or _running_container()
        cmd = [
            py,
            str(ROOT / "scripts" / "run_concurrency_sweep.py"),
            "--model",
            args.model,
            "--gpu-profile",
            args.gpu_profile,
            "--concurrency-list",
            *_ints(row["concurrency"]),
        ]
        if container:
            cmd += ["--container", container]
        rc = _run(cmd)
        if rc:
            return rc

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--api", default="")
    parser.add_argument("--model", default="Qwen3.8-27B-Quark-AWQ-MXFP4")
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu-profile", default="r9700")
    parser.add_argument("--container", default="")
    parser.add_argument("--aiperf", default=str(ROOT.parent / ".venv-aiperf" / "bin" / "aiperf"))
    parser.add_argument("--tokenizer", default=str(ROOT / "models" / "Qwen3.8-27B-Quark-AWQ-MXFP4"))
    args = parser.parse_args()
    return run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
