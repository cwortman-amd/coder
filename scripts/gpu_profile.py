#!/usr/bin/env python3
"""Host GPU profile helpers for Radeon AI PRO R9700 (gfx1201) and Instinct MI350P (gfx950)."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

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
            _cmd_text(["rocm-smi", "--showproductname", "--showid"]),
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


def parse_rocm_smi_gpus() -> List[Dict[str, str]]:
    devices: List[Dict[str, str]] = []
    text = _cmd_text(["rocm-smi", "--showproductname", "--showid"])
    if not text.strip():
        return devices
    by_idx: Dict[str, Dict[str, str]] = {}
    for line in text.splitlines():
        m = re.search(r"GPU\[(\d+)\]\s*:\s*(.+)", line)
        if not m:
            continue
        idx, rest = m.group(1), m.group(2)
        rec = by_idx.setdefault(
            idx,
            {"device_type": "GPU", "agent_id": f"GPU {idx}", "source": "rocm-smi"},
        )
        if "Device Name:" in rest or "Card Series:" in rest:
            rec["marketing_name"] = rest.split(":", 1)[-1].strip()
        elif "GFX Version:" in rest:
            rec["gfx"] = rest.split(":", 1)[-1].strip()
        elif "Device ID:" in rest or "Card Model:" in rest:
            rec["pci_id"] = rest.split(":", 1)[-1].strip().replace("0x", "").lower()
    devices = [by_idx[k] for k in sorted(by_idx, key=int)]
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
    devices = parse_rocm_smi_gpus()
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
