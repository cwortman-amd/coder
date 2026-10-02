#!/usr/bin/env python3
"""Single catalog for GPU envelope numbers and model identity.

Shell profiles and Python benches read TDP, peak bandwidth, VRAM, render GID,
model id, and weight from here. Serving recipes stay in lib/gpu_profile.sh
and lib/model_profile.sh.
"""

from __future__ import annotations

import argparse
import grp
import json
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "config" / "catalog.yaml"


def _load() -> dict[str, Any]:
    with CATALOG_PATH.open() as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"{CATALOG_PATH} did not contain a mapping")
    return loaded


_DATA = _load()
GPUS: dict[str, dict[str, Any]] = _DATA["gpus"]
GPU_ALIASES: dict[str, str] = _DATA["gpu_aliases"]
MODELS: dict[str, dict[str, Any]] = _DATA["models"]
MODEL_ALIASES: dict[str, str] = _DATA["model_aliases"]
CONTAINERS: dict[str, dict[str, Any]] = _DATA["containers"]

GIB_TO_GB = (1024 ** 3) / (1000 ** 3)


def gpu_key(name: Optional[str]) -> str:
    raw = (name or "r9700").strip().lower()
    key = GPU_ALIASES.get(raw, raw)
    if key not in GPUS:
        if "mi350" in key or "gfx950" in key or "instinct" in key:
            return "mi350p"
        if "r9600" in key:
            return "r9600"
        return "r9700"
    return key


def model_key(name: Optional[str]) -> str:
    raw = (name or "").strip().lower()
    if not raw:
        return ""
    key = MODEL_ALIASES.get(raw, raw)
    if key in MODELS:
        return key
    if "gpt-oss-120" in raw:
        return "gpt-oss-120b"
    if "gpt-oss" in raw:
        return "gpt-oss-20b"
    if "qwen3.8" in raw or "qwen3_5" in raw or "quark" in raw:
        return "qwen3.8"
    return ""


def gpu(name: Optional[str]) -> dict[str, Any]:
    key = gpu_key(name)
    spec = dict(GPUS[key])
    spec["profile"] = key
    return spec


def model(name: Optional[str]) -> dict[str, Any]:
    key = model_key(name)
    if not key:
        return {"profile": "", "model_id": "", "weight_gib": None}
    spec = dict(MODELS[key])
    spec["profile"] = key
    return spec


def group_gid(name: str, fallback: int) -> int:
    try:
        return int(grp.getgrnam(name).gr_gid)
    except KeyError:
        return fallback


def gids(profile: Optional[str] = None) -> dict[str, int]:
    spec = gpu(profile or "r9700")
    return {
        "video": group_gid("video", int(spec["video_gid_fallback"])),
        "render": group_gid("render", int(spec["render_gid_fallback"])),
    }


def weight_replay_gbs(output_tok_s: float, concurrency: float, weight_gib: Optional[float]) -> Optional[float]:
    """Per-sequence weight replay in decimal GB/s. None when the model has no weight."""
    if weight_gib is None or concurrency <= 0:
        return None
    return (output_tok_s / concurrency) * float(weight_gib) * GIB_TO_GB


def _exports_gpu(name: str) -> str:
    spec = gpu(name)
    lines = [
        f"GPU_TDP_W={spec['tdp_w']:.0f}",
        f"GPU_PEAK_BW_GBS={spec['peak_bw_gbs']:.0f}",
        f"GPU_VRAM_GB={int(spec['vram_gb'])}",
    ]
    return "\n".join(lines)


def container(name: str) -> dict[str, Any]:
    key = (name or "").strip().lower()
    spec = CONTAINERS.get(key)
    if spec is None:
        raise KeyError(key)
    row = dict(spec)
    row["key"] = key
    return row


def _exports_model(name: str) -> str:
    spec = model(name)
    lines = []
    if spec.get("weight_gib") is not None:
        lines.append(f"MODEL_WEIGHT_GIB={spec['weight_gib']}")
    if spec.get("model_id"):
        lines.append(f"MODEL_CATALOG_ID={spec['model_id']}")
    if spec.get("mxfp4_id"):
        lines.append(f"MODEL_MXFP4_ID={spec['mxfp4_id']}")
    if spec.get("container"):
        lines.append(f"MODEL_CONTAINER={spec['container']}")
    if spec.get("mxfp4_container"):
        lines.append(f"MODEL_MXFP4_CONTAINER={spec['mxfp4_container']}")
    return "\n".join(lines)


def _exports_container(name: str) -> str:
    spec = container(name)
    lines = [f"CONTAINER_NAME={spec['name']}"]
    if spec.get("port") is not None:
        lines.append(f"CONTAINER_PORT={int(spec['port'])}")
    if spec.get("gpu") is not None:
        lines.append(f"CONTAINER_GPU={int(spec['gpu'])}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU and model catalog")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gpu_cmd = sub.add_parser("exports-gpu")
    gpu_cmd.add_argument("profile")

    model_cmd = sub.add_parser("exports-model")
    model_cmd.add_argument("profile")

    sub.add_parser("gids")

    show = sub.add_parser("show-gpu")
    show.add_argument("profile")

    replay = sub.add_parser("replay-gbs")
    replay.add_argument("--tok-s", type=float, required=True)
    replay.add_argument("--concurrency", type=float, required=True)
    replay.add_argument("--model", default="")
    replay.add_argument("--weight-gib", type=float, default=None)

    field = sub.add_parser("gpu-field")
    field.add_argument("profile")
    field.add_argument("field")

    box = sub.add_parser("exports-container")
    box.add_argument("name")

    names = sub.add_parser("container-names")
    names.add_argument("keys", nargs="+")

    args = parser.parse_args()
    if args.cmd == "exports-gpu":
        print(_exports_gpu(args.profile))
    elif args.cmd == "exports-model":
        text = _exports_model(args.profile)
        if text:
            print(text)
    elif args.cmd == "gids":
        ids = gids()
        print(f"VIDEO_GID={ids['video']}")
        print(f"RENDER_GID={ids['render']}")
    elif args.cmd == "show-gpu":
        print(json.dumps(gpu(args.profile)))
    elif args.cmd == "gpu-field":
        spec = gpu(args.profile)
        if args.field not in spec:
            return 1
        print(spec[args.field])
    elif args.cmd == "exports-container":
        print(_exports_container(args.name))
    elif args.cmd == "container-names":
        for key in args.keys:
            print(container(key)["name"])
    elif args.cmd == "replay-gbs":
        weight = args.weight_gib
        if weight is None:
            weight = model(args.model).get("weight_gib")
        value = weight_replay_gbs(args.tok_s, args.concurrency, weight)
        print("N/A" if value is None else f"{value:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
