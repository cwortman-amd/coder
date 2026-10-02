#!/usr/bin/env python3
"""Compatibility entry for the multi-layer Triton tile sweep."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_mxfp4_triton_sweep import main_tile  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main_tile())
