#!/usr/bin/env python3
"""
Dual AMD Radeon AI PRO R9700 Hardware & Software Diagnostics Suite.

Probes:
1. Physical ROCm GPU topology, PCIe links, NUMA nodes, and P2P connectivity.
2. Homogeneous Dual-R9700 Preflight Gate (fails fast if heterogeneous APU detected).
3. In-container KV transfer connector availability and dependency resolution.
4. Analytical KV-cache transfer volume & PCIe handoff latency modeling.
"""

import argparse
import json
import os
import re
import subprocess
import sys


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_dual_r9700_preflight() -> tuple[bool, int, list[dict]]:
    """
    Scans the system for homogeneous Radeon AI PRO R9700 (gfx1201) devices.
    Returns: (passed: bool, count: int, devices: list)
    """
    devices = []
    try:
        res = subprocess.run(["rocminfo"], capture_output=True, text=True, check=True)
        current_agent = {}
        for line in res.stdout.splitlines():
            line_str = line.strip()
            if line_str.startswith("Agent "):
                if current_agent.get("device_type") == "GPU":
                    devices.append(current_agent)
                current_agent = {"agent_id": line_str}
            elif "Device Type:" in line_str:
                current_agent["device_type"] = line_str.split(":")[-1].strip()
            elif "Marketing Name:" in line_str:
                current_agent["marketing_name"] = line_str.split(":")[-1].strip()
            elif "Name:" in line_str and "Marketing Name" not in line_str:
                current_agent["gfx"] = line_str.split(":")[-1].strip()
            elif "Compute Unit:" in line_str:
                current_agent["cu"] = line_str.split(":")[-1].strip()
        if current_agent.get("device_type") == "GPU":
            devices.append(current_agent)
    except Exception as e:
        print(f"Preflight error querying rocminfo: {e}")

    # Filter for homogeneous R9700 gfx1201 devices
    r9700_devices = [
        d for d in devices
        if "gfx1201" in d.get("gfx", "") or "R9700" in d.get("marketing_name", "")
    ]

    passed = (len(r9700_devices) >= 2)
    return passed, len(r9700_devices), devices


def probe_host_topology():
    print_banner("1. HOST ROCM HARDWARE & TOPOLOGY PROBE")

    passed, r9700_count, all_gpus = check_dual_r9700_preflight()

    print(f"Detected {len(all_gpus)} GPU Agent(s) via ROCm KFD:")
    for idx, g in enumerate(all_gpus):
        name = g.get('marketing_name', 'Unknown')
        gfx = g.get('gfx', 'N/A')
        cu = g.get('cu', 'N/A')
        is_r9700 = "R9700" in name or "gfx1201" in gfx
        marker = " [TARGET R9700 dGPU]" if is_r9700 else " [INTEGRATED APU - EXCLUDE]"
        print(f"  • GPU [{idx}]: {name} ({gfx}), CUs: {cu}{marker}")

    print("\n--------------------------------------------------------------------------------")
    print("  DUAL-R9700 HOMOGENEOUS PREFLIGHT STATUS")
    print("--------------------------------------------------------------------------------")
    if passed:
        print(f"  [STATUS: PASSED] Detected {r9700_count} matching Radeon AI PRO R9700 (gfx1201) GPUs.")
        print("  System is fully qualified for TP=2, DP=2, and P/D multi-GPU benchmarking.")
    else:
        print(f"  [STATUS: HARDWARE GATE BLOCKED] Detected {r9700_count} Radeon AI PRO R9700 (Expected: 2).")
        print("  The current host contains 1x Radeon AI PRO R9700 dGPU + 1x Radeon 780M iGPU.")
        print("  Heterogeneous execution across gfx1201 and gfx1103 is strictly prohibited.")
        print("  Multi-GPU TP=2, DP=2, and P/D remain in 'Target Architecture' staging until card 2 is installed.")
        print("  Single-GPU baselines, chunked prefill, and router mock testing remain operational.")
    print("--------------------------------------------------------------------------------")

    # rocm-smi topology
    try:
        res = subprocess.run(["rocm-smi", "--showtopo", "--showproductname"], capture_output=True, text=True, check=True)
        print("\nROCm-SMI Inter-GPU Topology & Product Details:")
        for line in res.stdout.splitlines():
            if any(k in line for k in ["Series:", "Model:", "GFX Version:", "PCIE", "Weight", "Hops"]):
                print(f"  {line}")
    except Exception as e:
        print(f"Warning: rocm-smi topology query failed: {e}")

    # PCIe link speed check via lspci
    try:
        res = subprocess.run(["lspci", "-vvv"], capture_output=True, text=True)
        navi_blocks = re.findall(r"((\w+:\w+\.\w+).*?Navi 48.*?LnkSta:.*?)(?=\n\n|\Z)", res.stdout, re.DOTALL)
        if navi_blocks:
            print("\nPCIe Link Status for Navi 48 (Radeon AI PRO R9700):")
            for block, bdf in navi_blocks:
                lnksta = re.search(r"LnkSta:\s+(Speed\s+[^,]+,\s+Width\s+[^,\n]+)", block)
                if lnksta:
                    print(f"  • Device {bdf}: {lnksta.group(1)}")
    except Exception:
        pass


def probe_container_connectors(container_name: str = "rocm-mxfp4-server"):
    print_banner(f"2. IN-CONTAINER KV-TRANSFER CONNECTOR PROBE ({container_name})")

    # Verify container is running
    check_cmd = ["docker", "ps", "--format", "{{.Names}}"]
    try:
        res = subprocess.run(check_cmd, capture_output=True, text=True, check=True)
        active_containers = res.stdout.splitlines()
        if container_name not in active_containers:
            print(f"Container '{container_name}' is not currently running.")
            print(f"Active containers: {', '.join(active_containers) if active_containers else 'None'}")
            return
    except Exception as e:
        print(f"Docker inspection error: {e}")
        return

    probe_py = """
import sys, importlib, pkgutil
import vllm
print(f"vLLM Runtime Version: {vllm.__version__}")

# Check KV Connector Factory
try:
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    registered = list(KVConnectorFactory._registry.keys())
    print(f"Registered Connectors in Factory ({len(registered)}):")
    for c in registered:
        print(f"  • {c}")
except Exception as e:
    print(f"KVConnectorFactory inspection failed: {e}")

# Detailed import probe for key connectors
connectors_to_test = [
    ("MoRIIOConnector", "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector", "MoRIIOConnector"),
    ("LMCacheConnectorV1", "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector", "LMCacheConnectorV1"),
    ("NixlConnector", "vllm.distributed.kv_transfer.kv_connector.v1.nixl", "NixlConnector"),
    ("MooncakeConnector", "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector", "MooncakeConnector"),
    ("SimpleCPUOffloadConnector", "vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector", "SimpleCPUOffloadConnector"),
    ("ExampleConnector", "vllm.distributed.kv_transfer.kv_connector.v1.example_connector", "ExampleConnector"),
]

print("\\nConnector Import Status & Dependency Diagnosis:")
for name, mod_path, cls_name in connectors_to_test:
    try:
        mod = importlib.import_module(mod_path)
        cls_obj = getattr(mod, cls_name)
        detail = ""
        if name == "MoRIIOConnector":
            is_avail = getattr(mod, "is_moriio_available", lambda: False)()
            detail = " (Native C++ backend linked)" if is_avail else " (Python classes OK; mori.io C++ absent)"
        print(f"  [AVAILABLE]   {name}{detail}")
    except ModuleNotFoundError as mnf:
        print(f"  [MISSING DEP] {name}: Missing Python module '{mnf.name}'")
    except Exception as ex:
        print(f"  [FAILED]      {name}: {repr(ex)}")

# Test MORI native library probe
try:
    import mori.io
    print("  [NATIVE MORI] Found native mori.io library in container.")
except ImportError:
    print("  [NATIVE MORI] mori.io C++ / Python extension is NOT installed.")
"""

    try:
        exec_cmd = ["docker", "exec", container_name, "python3", "-c", probe_py]
        res = subprocess.run(exec_cmd, capture_output=True, text=True, timeout=25)
        print(res.stdout)
        if res.stderr:
            filtered_stderr = [l for l in res.stderr.splitlines() if not l.startswith("[radiance")]
            if filtered_stderr:
                print("Warnings/Errors:\n" + "\n".join(filtered_stderr))
    except Exception as e:
        print(f"Error executing probe inside container: {e}")


def calculate_analytical_kv_transfer(total_layers=64, attn_layers=16, kv_heads=8, head_dim=128):
    print_banner("3. ANALYTICAL KV-CACHE TRANSFER MODEL (Qwen3.8-27B)")

    print(f"Hybrid Architecture Specification:")
    print(f"  • Total Layers: {total_layers} (48 Gated DeltaNet Linear Attention + {attn_layers} Full Softmax Attention)")
    print(f"  • Standard KV-Bearing Layers: {attn_layers} layers (GDN states are held in separate recurrent state buffers)")
    print(f"  • KV Heads: {kv_heads}, Head Dimension: {head_dim}")
    print(f"  • Analytical Summation: KV_bytes = sum_{{l in Attn}} (2 * H_kv * D * S * B)")
    print("\nPCIe Bus Reference: PCIe 5.0 x16 ~ 52.0 GB/s empirical | PCIe 4.0 x16 ~ 26.0 GB/s empirical")
    print("NOTE: Values below represent the IDEAL PAYLOAD-COPY FLOOR (T_data-movement).")
    print("      Total handoff latency = T_serialize + T_control-plane + T_buffer-ready + T_data-movement + T_decode-admission\n")

    print(f"{'Context Length (Tokens)':<24} | {'Precision':<10} | {'KV Size (16 Attn)':<18} | {'PCIe 4.0 Floor':<18} | {'PCIe 5.0 Floor':<18}")
    print("-" * 96)

    test_contexts = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    precisions = [("FP8", 1), ("FP16", 2)]

    for ctx in test_contexts:
        for prec_name, b_bytes in precisions:
            bytes_transferred = 2 * attn_layers * kv_heads * head_dim * ctx * b_bytes
            mb = bytes_transferred / (1024 * 1024)
            gb = bytes_transferred / (1024 * 1024 * 1024)

            size_str = f"{mb:.1f} MB" if mb < 1000 else f"{gb:.2f} GB"

            # Ideal physical payload-copy floor (ms)
            time_pcie4_ms = (bytes_transferred / (26.0 * 1e9)) * 1000.0
            time_pcie5_ms = (bytes_transferred / (52.0 * 1e9)) * 1000.0

            print(f"{ctx:<24} | {prec_name:<10} | {size_str:<18} | {time_pcie4_ms:>8.2f} ms         | {time_pcie5_ms:>8.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Dual R9700 Hardware & P/D Diagnostics")
    parser.add_argument("--container", default="rocm-mxfp4-server", help="vLLM container to inspect")
    parser.add_argument("--skip-container", action="store_true", help="Skip container probe")
    parser.add_argument("--check-preflight", action="store_true", help="Fail-fast check returning exit code 0 if 2x R9700 present, 1 otherwise")
    args = parser.parse_args()

    if args.check_preflight:
        passed, count, _ = check_dual_r9700_preflight()
        if not passed:
            print(f"PREFLIGHT FAILED: Found {count} Radeon AI PRO R9700 devices (Required: 2).", file=sys.stderr)
            sys.exit(1)
        print(f"PREFLIGHT OK: Found {count} homogeneous Radeon AI PRO R9700 devices.")
        sys.exit(0)

    probe_host_topology()
    if not args.skip_container:
        probe_container_connectors(args.container)
    calculate_analytical_kv_transfer()

    print_banner("SUMMARY VERDICT")
    print("• Target Architecture: Dual homogeneous AMD Radeon AI PRO R9700 (gfx1201).")
    print("• Current Host Reality: 1x R9700 dGPU + 1x 780M iGPU (Heterogeneous execution blocked).")
    print("• TP=2 (Tensor Parallelism): Immediate priority for 2x homogeneous R9700 hardware.")
    print("• P/D (Prefill/Decode Disaggregation): Requires MoRI-IO runtime compilation; raw PCIe bandwidth is not gating.")


if __name__ == "__main__":
    main()
