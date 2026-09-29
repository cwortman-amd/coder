#!/usr/bin/env python3
"""Host GPU profile helpers for Radeon AI PRO R9700 (gfx1201) and Instinct MI350P (gfx950)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import catalog  # noqa: E402

PROFILES: Dict[str, Dict[str, Any]] = {
    "r9700": {
        "family": "radeon",
        "isa": "gfx1201",
        "marketing": "AMD Radeon AI PRO R9700",
        "vram_gb": 32,
        "tdp_w": 300,
        "label": "AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6)",
        "pci_ids": ("7551",),
        "name_tokens": ("R9700", "gfx1201"),
    },
    "mi350p": {
        "family": "instinct",
        "isa": "gfx950",
        "marketing": "AMD Instinct MI350P",
        "vram_gb": 144,
        "tdp_w": 600,
        "label": "AMD Instinct™ MI350P (gfx950, 144 GB HBM3E)",
        "pci_ids": tuple(f"75a{x}" for x in "0123456789abcdef"),
        "name_tokens": ("MI350P", "MI350", "MI355", "gfx950"),
    },
}

ALIASES = {
    "gfx1201": "r9700",
    "radeon": "r9700",
    "gfx950": "mi350p",
    "mi350": "mi350p",
    "instinct": "mi350p",
}


def normalize_profile(name: Optional[str]) -> str:
    raw = (name or os.environ.get("GPU_PROFILE") or "auto").strip().lower()
    if raw in ("auto", ""):
        return detect_profile()
    return ALIASES.get(raw, raw if raw in PROFILES else "r9700")


def _cmd_text(cmd: List[str]) -> str:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return (res.stdout or "") + "\n" + (res.stderr or "")
    except FileNotFoundError:
        return ""


def detect_profile() -> str:
    env = os.environ.get("GPU_PROFILE", "").strip().lower()
    if env and env not in ("auto",):
        return ALIASES.get(env, env if env in PROFILES else "r9700")
    text = " ".join(
        (
            _cmd_text(["rocminfo"]),
            _cmd_text(["amd-smi", "static", "--asic"]),
            _cmd_text(["lspci", "-nn"]),
        )
    ).lower()
    if "gfx950" in text or "mi350" in text or "mi355" in text or "1002:75a" in text:
        return "mi350p"
    if "gfx1201" in text or "r9700" in text:
        return "r9700"
    return "r9700"


def profile_info(name: Optional[str] = None) -> Dict[str, Any]:
    key = normalize_profile(name)
    info = dict(PROFILES.get(key, PROFILES["r9700"]))
    spec = catalog.gpu(key)
    info["tdp_w"] = spec["tdp_w"]
    info["vram_gb"] = spec["vram_gb"]
    info["peak_bw_gbs"] = spec["peak_bw_gbs"]
    info["profile"] = key
    info["label"] = os.environ.get("GPU_LABEL", info["label"])
    return info


def parse_rocminfo_gpus() -> List[Dict[str, str]]:
    devices: List[Dict[str, str]] = []
    try:
        res = subprocess.run(["rocminfo"], capture_output=True, text=True, check=True)
    except Exception:
        return devices
    current: Dict[str, str] = {}
    for line in res.stdout.splitlines():
        line_str = line.strip()
        if line_str.startswith("Agent "):
            if current.get("device_type") == "GPU":
                devices.append(current)
            current = {"agent_id": line_str}
        elif "Device Type:" in line_str:
            current["device_type"] = line_str.split(":")[-1].strip()
        elif "Marketing Name:" in line_str:
            current["marketing_name"] = line_str.split(":")[-1].strip()
        elif "Name:" in line_str and "Marketing Name" not in line_str:
            current["gfx"] = line_str.split(":")[-1].strip()
        elif "Compute Unit:" in line_str:
            current["cu"] = line_str.split(":")[-1].strip()
    if current.get("device_type") == "GPU":
        devices.append(current)
    return devices


def parse_amd_smi_gpus() -> List[Dict[str, str]]:
    devices: List[Dict[str, str]] = []
    try:
        res = subprocess.run(
            ["amd-smi", "static", "--asic", "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        payload = json.loads(res.stdout or "{}")
    except (OSError, json.JSONDecodeError):
        return devices
    for gpu in payload.get("gpu_data") or []:
        asic = gpu.get("asic") or {}
        idx = gpu.get("gpu")
        if idx is None:
            continue
        devices.append(
            {
                "device_type": "GPU",
                "agent_id": f"GPU {idx}",
                "marketing_name": str(asic.get("market_name") or ""),
                "gfx": str(asic.get("target_graphics_version") or ""),
                "pci_id": str(asic.get("device_id") or "").replace("0x", "").lower(),
                "source": "amd-smi",
            }
        )
    return devices


def parse_lspci_gpus() -> List[Dict[str, str]]:
    devices: List[Dict[str, str]] = []
    text = _cmd_text(["lspci", "-nn"])
    for line in text.splitlines():
        m = re.search(r"\[1002:([0-9a-f]{4})\]", line, re.IGNORECASE)
        if not m:
            continue
        pci_id = m.group(1).lower()
        if pci_id.startswith("75a") or pci_id in ("7551",):
            devices.append(
                {
                    "device_type": "GPU",
                    "marketing_name": line.strip(),
                    "pci_id": pci_id,
                    "source": "lspci",
                }
            )
    return devices


def parse_host_gpus() -> List[Dict[str, str]]:
    devices = parse_rocminfo_gpus()
    if devices:
        return devices
    devices = parse_amd_smi_gpus()
    if devices:
        return devices
    return parse_lspci_gpus()


def classify_gpu(gpu: Dict[str, str]) -> Optional[str]:
    blob = f"{gpu.get('gfx', '')} {gpu.get('marketing_name', '')} {gpu.get('pci_id', '')}".upper()
    pci = (gpu.get("pci_id") or "").lower().replace("0x", "")
    if pci.startswith("75a") or "GFX950" in blob or "MI350" in blob or "MI355" in blob:
        return "mi350p"
    if pci == "7551" or "GFX1201" in blob or "R9700" in blob:
        return "r9700"
    return None


def homogeneous_target_gpus(profile: Optional[str] = None) -> Tuple[bool, int, List[Dict[str, str]]]:
    key = normalize_profile(profile)
    all_gpus = parse_host_gpus()
    matched = [g for g in all_gpus if classify_gpu(g) == key]
    return len(matched) >= 2, len(matched), all_gpus


def target_gpu_indices(profile: Optional[str] = None) -> List[int]:
    """amd-smi ordinals whose device matches the profile.

    HIP_VISIBLE_DEVICES uses these ordinals. An integrated APU on the same
    host keeps its own ordinal and is left out.
    """
    key = normalize_profile(profile)
    indices: List[int] = []
    for gpu in parse_amd_smi_gpus():
        if classify_gpu(gpu) != key:
            continue
        match = re.search(r"(\d+)", gpu.get("agent_id", ""))
        if match:
            indices.append(int(match.group(1)))
    return indices


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Host GPU profile helpers")
    parser.add_argument("--gpu-profile", default=None)
    parser.add_argument(
        "--indices",
        action="store_true",
        help="Print HIP ordinals for GPUs that match the profile, one per line",
    )
    args = parser.parse_args()
    if args.indices:
        for index in target_gpu_indices(args.gpu_profile):
            print(index)
        return
    print(profile_info(args.gpu_profile)["profile"])


if __name__ == "__main__":
    main()
