"""Configure rocprofiler for Python-spawned EngineCore processes.

Python ``spawn`` starts a fresh interpreter for EngineCore. Sitecustomize is
an early point to alter the child environment, but dynamic attach is suppressed
when the child inherits ``ROCPROFILER_REGISTER_LIBRARY``: rocprofiler-register
treats a tool as active and skips loading the attach library.

Dynamic attach still requires an attach-enabled rocprofiler-register build.
For the reliable path, VLLM_ENGINECORE_ROCP_EXEC=1 changes Python's
multiprocessing executable in the API parent. The wrapper sends only spawned
workers through rocprofv3, leaving vllm serve itself unwrapped.
"""

from __future__ import annotations

import os
import sys
import multiprocessing
from pathlib import Path


if os.environ.get("VLLM_ENGINECORE_ROCP_ATTACH") == "1":
    os.environ["ROCP_TOOL_ATTACH"] = "1"

if os.environ.get("VLLM_ENGINECORE_ROCP_EXEC") == "1" and os.getpid() == 1:
    multiprocessing.set_executable(
        os.environ.get(
            "VLLM_ENGINECORE_ROCP_EXECUTABLE",
            "/opt/vllm-enginecore-attach/enginecore_rocprof_exec.sh",
        )
    )

if (
    os.environ.get("VLLM_ENGINECORE_ROCP_ATTACH") == "1"
    or os.environ.get("VLLM_ENGINECORE_ROCP_EXEC") == "1"
):
    # Best-effort diagnostics. Never make serving depend on this log.
    try:
        log = Path("/results/profiling/enginecore_attach/sitecustomize.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(
                f"pid={os.getpid()} ppid={os.getppid()} "
                f"argv={sys.argv!r} "
                f"attach={os.environ.get('ROCP_TOOL_ATTACH')} "
                f"exec={os.environ.get('VLLM_ENGINECORE_ROCP_EXEC')}\n"
            )
    except Exception:
        pass
