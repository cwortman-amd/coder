#!/usr/bin/env python3
"""Profile mini-decode through vLLM's compiled CUDA/HIP graph path.

This intentionally uses vLLM's LLM engine, CUDAGraphWrapper, static input
buffers, and global graph pool.  It does not create a torch.cuda.CUDAGraph.
Run the host launcher rather than invoking this beside a live 27B server.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import socket
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

GIB = 1 << 30
STOCK_M_LEQ_8 = {
    "BLOCK_SIZE_M": 8,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 2,
    "waves_per_eu": 2,
    "matrix_instr_nonkdim": 32,
    "cache_modifier": ".cg",
    "NUM_KSPLIT": 4,
}
DEFAULT_AITER_CONFIG = (
    "/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/configs/gfx950/"
    "triton/gemm/gemm_afp4wfp4/DEFAULT.json"
)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def server_healthy(port: int = 8000) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=1
        ) as response:
            return response.status == 200
    except Exception:
        return False


def gpu_memory() -> dict[str, int | float | None]:
    try:
        import torch

        free, total = torch.cuda.mem_get_info()
        return {
            "free_bytes": free,
            "total_bytes": total,
            "free_gib": round(free / GIB, 2),
            "total_gib": round(total / GIB, 2),
        }
    except Exception as exc:
        return {"free_bytes": None, "total_bytes": None, "error": str(exc)}


def frozen_control() -> dict[str, Any]:
    path = Path(DEFAULT_AITER_CONFIG)
    result: dict[str, Any] = {
        "VLLM_ROCM_USE_AITER": os.environ.get("VLLM_ROCM_USE_AITER"),
        "AITER_AFP4_DEFAULT_JSON": os.environ.get("AITER_AFP4_DEFAULT_JSON"),
        "config_path": str(path),
        "stock_expected": STOCK_M_LEQ_8,
    }
    if path.exists():
        raw = path.read_bytes()
        config = json.loads(raw)
        result.update(
            sha256=hashlib.sha256(raw).hexdigest(),
            observed_m_leq_8=config.get("M_LEQ_8"),
            stock_m_leq_8=config.get("M_LEQ_8") == STOCK_M_LEQ_8,
        )
    else:
        result["error"] = "AITER DEFAULT.json not found"
    return result


def write_results(out_dir: Path, data: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
    lines = [
        "# vLLM captured mini-decode",
        "",
        f"- UTC: `{data['utc']}`",
        f"- Status: **{data['status']}**",
        f"- Live `:8000` detected: `{data['preflight']['server_8000_healthy']}`",
        f"- GPU free/total GiB: `{data['preflight']['gpu_memory'].get('free_gib')}` / "
        f"`{data['preflight']['gpu_memory'].get('total_gib')}`",
        "- Compile path: vLLM `VLLM_COMPILE` (mode 3/O2), "
        "`FULL_AND_PIECEWISE`, capture sizes 1 and 8",
        "- Control: `VLLM_ROCM_USE_AITER` unset; stock `M_LEQ_8`; no overlay",
        "",
    ]
    if data.get("message"):
        lines += ["## Result", "", data["message"], ""]
    if data.get("decode"):
        lines += [
            "## Decode windows",
            "",
            "| M | generated | steady window ms | tok/s |",
            "|---:|---:|---:|---:|",
        ]
        for row in data["decode"]:
            lines.append(
                f"| {row['m']} | {row['generated_tokens']} | "
                f"{row.get('steady_decode_window_ms', 0):.3f} | "
                f"{row.get('steady_decode_tokens_per_s', 0):.2f} |"
            )
        lines.append("")
    if data.get("projection_ranking"):
        lines += [
            "## Projection ranking",
            "",
            "| Rank | Projection | attributed GPU ms | evidence |",
            "|---:|---|---:|---|",
        ]
        for i, row in enumerate(data["projection_ranking"], 1):
            gpu_ms = (
                f"{row['gpu_ms']:.4f}"
                if row.get("gpu_ms") is not None
                else "unavailable"
            )
            lines.append(
                f"| {i} | {row['projection']} | {gpu_ms} | "
                f"{row['evidence']} |"
            )
        lines += [
            "",
            "Gate and up are represented separately in the report but share the "
            "same fused `gate_up_proj` measurement when vLLM merges them.",
            "",
        ]
    if data.get("limitations"):
        lines += ["## Limitations", ""]
        lines.extend(f"- {item}" for item in data["limitations"])
        lines.append("")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines))


def _load_trace(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload.get("traceEvents", []) if isinstance(payload, dict) else []
    except Exception:
        return []


def _projection(blob: str) -> str | None:
    text = blob.lower().replace(" ", "")
    if "lm_head" in text or ("248320" in text and "5120" in text):
        return "lm_head"
    if "down_proj" in text or ("[5120,8704]" in text and "mxfp4" in text):
        return "mlp.down_proj"
    if "gate_up_proj" in text or "[34816,2560]" in text:
        return "mlp.gate_up_proj"
    if "gate_proj" in text:
        return "mlp.gate_proj"
    if "up_proj" in text:
        return "mlp.up_proj"
    if any(
        token in text
        for token in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "linear_attn.in_proj",
            "linear_attn.out_proj",
        )
    ):
        return "attention.projections"
    # Packed MXFP4 attention weights have these distinctive [N, K/2] shapes.
    attention_shapes = (
        "[12288,2560]",
        "[10240,2560]",
        "[6144,2560]",
        "[5120,3072]",
        "[1024,2560]",
        "[48,2560]",
    )
    if "mxfp4" in text and any(shape in text for shape in attention_shapes):
        return "attention.projections"
    if "mxfp4" in text and "[17408,2560]" in text:
        return "mlp.gate_up_proj"
    return None


def rank_capture_traces(profile_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Best-effort shape attribution from vLLM's capture-time torch traces."""
    paths = sorted(profile_dir.rglob("*.json")) + sorted(profile_dir.rglob("*.json.gz"))
    events: list[dict[str, Any]] = []
    for path in paths:
        events.extend(_load_trace(path))

    id_to_projection: dict[str, str] = {}
    cpu_us: defaultdict[str, float] = defaultdict(float)
    gpu_us: defaultdict[str, float] = defaultdict(float)
    for event in events:
        if event.get("ph") != "X":
            continue
        blob = json.dumps(
            {"name": event.get("name", ""), "args": event.get("args", {})},
            separators=(",", ":"),
        )
        projection = _projection(blob)
        args = event.get("args") or {}
        if projection:
            cpu_us[projection] += float(event.get("dur") or 0)
            for key in ("External id", "External Id", "Sequence number", "correlation"):
                if key in args:
                    id_to_projection[str(args[key])] = projection

    for event in events:
        if event.get("ph") != "X":
            continue
        args = event.get("args") or {}
        cat = str(event.get("cat", "")).lower()
        is_gpu = "kernel" in cat or args.get("Device Type") in (1, "1", "GPU")
        if not is_gpu:
            continue
        projection = _projection(json.dumps(event, separators=(",", ":")))
        if projection is None:
            for key in ("External id", "External Id", "Sequence number", "correlation"):
                if str(args.get(key)) in id_to_projection:
                    projection = id_to_projection[str(args[key])]
                    break
        if projection:
            gpu_us[projection] += float(event.get("dur") or 0)

    # vLLM commonly fuses gate/up into one ParallelLMHead-style linear.
    if gpu_us.get("mlp.gate_up_proj"):
        shared = gpu_us["mlp.gate_up_proj"]
        gpu_us["mlp.gate_proj"] += shared
        gpu_us["mlp.up_proj"] += shared
        del gpu_us["mlp.gate_up_proj"]

    warnings: list[str] = []
    if not gpu_us and cpu_us:
        warnings.append(
            "Trace contained shape-attributed CPU launch time but no linkable GPU "
            "events; ranking is unavailable rather than substituting CPU time."
        )
    if not gpu_us:
        warnings.append(
            "ROCm torch-profiler graph replay did not expose projection-to-kernel "
            "correlation. Whole-decode replay timing remains valid."
        )
    rows = [
        {
            "projection": name,
            "gpu_ms": duration / 1000,
            "evidence": (
                "shared fused gate_up_proj; values are not additive"
                if name in ("mlp.gate_proj", "mlp.up_proj")
                else "capture shape + profiler external-id kernel correlation"
            ),
        }
        for name, duration in gpu_us.items()
    ]
    rows.sort(key=lambda row: row["gpu_ms"], reverse=True)
    required = (
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "attention.projections",
        "lm_head",
    )
    present = {row["projection"] for row in rows}
    rows.extend(
        {
            "projection": name,
            "gpu_ms": None,
            "evidence": "unavailable: no projection-to-replay-kernel correlation",
        }
        for name in required
        if name not in present
    )
    return rows, warnings


def metric_value(metrics: Any, name: str) -> float | None:
    value = getattr(metrics, name, None)
    return float(value) if value is not None else None


def run_decode(llm: Any, sampling_params: Any, m: int, steps: int) -> dict[str, Any]:
    prompts = [f"Count upward briefly. Request {i}:" for i in range(m)]
    begin = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    wall = time.perf_counter() - begin
    generated = sum(len(output.outputs[0].token_ids) for output in outputs)
    first = [
        metric_value(output.metrics, "first_token_time")
        for output in outputs
        if output.metrics is not None
    ]
    finished = [
        metric_value(output.metrics, "finished_time")
        for output in outputs
        if output.metrics is not None
    ]
    first = [value for value in first if value is not None]
    finished = [value for value in finished if value is not None]
    steady_s = max(finished) - min(first) if first and finished else wall
    steady_tokens = max(generated - m, 0)
    return {
        "m": m,
        "requested_decode_steps": steps,
        "generated_tokens": generated,
        "wall_ms": wall * 1000,
        "steady_decode_window_ms": steady_s * 1000,
        "steady_decode_tokens": steady_tokens,
        "steady_decode_tokens_per_s": steady_tokens / steady_s if steady_s > 0 else 0,
        "note": "Steady window excludes first-token prefill using RequestMetrics.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="/models/Qwen3.8-27B-Quark-AWQ-MXFP4-sharded"
    )
    parser.add_argument(
        "--out-dir", default="/results/profiling/vllm_captured"
    )
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--exclusive-confirmed", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    control = frozen_control()
    preflight = {
        "hostname": socket.gethostname(),
        "server_8000_healthy": server_healthy(),
        "gpu_memory": gpu_memory(),
    }
    result: dict[str, Any] = {
        "utc": utc(),
        "status": "preflight",
        "preflight": preflight,
        "frozen_control": control,
        "vllm_config": {
            "compilation_mode": 3,
            "compile_level_name": "O2 / VLLM_COMPILE",
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": [1, 8],
        },
    }

    if os.environ.get("VLLM_ROCM_USE_AITER"):
        result.update(
            status="refused",
            message="VLLM_ROCM_USE_AITER must be unset for the frozen control.",
        )
        write_results(out_dir, result)
        return 2
    if not control.get("stock_m_leq_8"):
        result.update(
            status="refused",
            message="Installed AITER M_LEQ_8 does not match the frozen stock config.",
        )
        write_results(out_dir, result)
        return 2
    if preflight["server_8000_healthy"] and not args.preflight_only:
        result.update(
            status="exclusive_gpu_required",
            message=(
                "Refusing to load a second 27B engine while :8000 is healthy. "
                "Use bench_vllm_captured_decode.sh --exclusive; it stops and restores "
                "the frozen server."
            ),
        )
        write_results(out_dir, result)
        return 3
    if args.preflight_only:
        free_gib = preflight["gpu_memory"].get("free_gib")
        result.update(
            status="exclusive_gpu_required",
            message=(
                f"Live-server preflight only. Free GPU memory is {free_gib} GiB; "
                "serving logs show 17.91 GiB weights plus 6.84 GiB graphs, so a "
                "second engine is unsafe. No model was loaded."
            ),
        )
        write_results(out_dir, result)
        return 0
    if not args.exclusive_confirmed:
        result.update(
            status="refused",
            message="Pass --exclusive-confirmed via the host launcher.",
        )
        write_results(out_dir, result)
        return 2

    from vllm import LLM, SamplingParams

    profile_dir = out_dir / "torch_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        kv_cache_memory_bytes=1 * GIB,
        enforce_eager=False,
        compilation_config={
            "mode": 3,
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": [1, 8],
            "max_cudagraph_capture_size": 8,
        },
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": str(profile_dir),
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": True,
            "torch_profiler_use_gzip": True,
            "capture_torch_profiler": True,
            "ignore_frontend": True,
        },
    )
    sampling = SamplingParams(
        temperature=0, max_tokens=args.steps, min_tokens=args.steps, ignore_eos=True
    )

    # Warm both requested scheduler shapes after initialization/capture.
    for m in (1, 8):
        warm = SamplingParams(temperature=0, max_tokens=3, min_tokens=3, ignore_eos=True)
        llm.generate(["Warmup:"] * m, warm, use_tqdm=False)

    decode = []
    for m in (1, 8):
        llm.start_profile(profile_prefix=f"decode_m{m}")
        decode.append(run_decode(llm, sampling, m, args.steps))
        llm.stop_profile()

    ranking, limitations = rank_capture_traces(profile_dir)
    result.update(
        utc=utc(),
        status="completed",
        decode=decode,
        projection_ranking=ranking,
        profile_files=[
            str(path.relative_to(out_dir)) for path in sorted(profile_dir.rglob("*"))
            if path.is_file()
        ],
        limitations=limitations
        + [
            "This is an intermediate captured-decode screen, not the unprofiled "
            "serving promotion gate.",
            "PyTorch module hooks do not execute during full-graph replay. Projection "
            "ranking therefore requires capture-shape/external-id correlation; the "
            "script reports unavailable if ROCm omits it.",
        ],
        message="vLLM captured M=1 and M=8 decode windows completed.",
    )
    write_results(out_dir, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
