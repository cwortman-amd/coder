#!/usr/bin/env python3
"""Compatibility entry for the chat throughput client."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_openai_client import main_chat  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main_chat())
