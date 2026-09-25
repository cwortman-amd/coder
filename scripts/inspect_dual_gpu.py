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

from gpu_profile import classify_gpu, homogeneous_target_gpus, normalize_profile, profile_info


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_dual_target_preflight(profile: str) -> tuple[bool, int, list[dict]]:
    """Require two homogeneous target dGPUs for the selected profile (r9700 or mi350p)."""
    passed, count, devices = homogeneous_target_gpus(profile)
    return passed, count, devices


def probe_host_topology(profile: str):
    print_banner("1. HOST ROCM HARDWARE & TOPOLOGY PROBE")
    info = profile_info(profile)
    passed, target_count, all_gpus = check_dual_target_preflight(profile)

    print(f"Active GPU profile : {info['profile']} ({info['label']})")
    print(f"Detected {len(all_gpus)} GPU Agent(s) via ROCm KFD:")
    for idx, g in enumerate(all_gpus):
        name = g.get("marketing_name", "Unknown")
        gfx = g.get("gfx", "N/A")
        cu = g.get("cu", "N/A")
        kind = classify_gpu(g)
        if kind == info["profile"]:
            marker = f" [TARGET {info['marketing']}]"
        elif kind:
            marker = f" [OTHER TARGET PROFILE: {kind}]"
        else:
            marker = " [NON-TARGET / iGPU - EXCLUDE]"
        print(f"  • GPU [{idx}]: {name} ({gfx}), CUs: {cu}{marker}")

    print("\n--------------------------------------------------------------------------------")
    print(f"  DUAL-{info['profile'].upper()} HOMOGENEOUS PREFLIGHT STATUS")
    print("--------------------------------------------------------------------------------")
    if passed:
        print(f"  [STATUS: PASSED] Detected {target_count} matching {info['marketing']} ({info['isa']}) GPUs.")
        print("  System is qualified for TP=2, DP=2, and P/D multi-GPU benchmarking.")
    else:
        print(f"  [STATUS: HARDWARE GATE BLOCKED] Detected {target_count} matching cards (Expected: 2).")
        print("  Dual-GPU modes require two homogeneous dGPUs of the same ISA (do not mix R9700 with MI350P or iGPU).")
        print("  Single-GPU baselines and router mock testing remain operational.")
    print("--------------------------------------------------------------------------------")

    try:
        res = subprocess.run(["rocm-smi", "--showtopo", "--showproductname"], capture_output=True, text=True, check=True)
        print("\nROCm-SMI Inter-GPU Topology & Product Details:")
        for line in res.stdout.splitlines():
            if any(k in line for k in ["Series:", "Model:", "GFX Version:", "PCIE", "Weight", "Hops"]):
                print(f"  {line}")
    except Exception as e:
        print(f"Warning: rocm-smi topology query failed: {e}")

    try:
        res = subprocess.run(["lspci", "-vvv"], capture_output=True, text=True)
        patterns = (
            r"((\w+:\w+\.\w+).*?Navi 48.*?LnkSta:.*?)",
            r"((\w+:\w+\.\w+).*?Instinct.*?LnkSta:.*?)",
            r"((\w+:\w+\.\w+).*?MI350.*?LnkSta:.*?)",
        )
        print("\nPCIe Link Status (compute GPUs):")
        found = False
        for pat in patterns:
            blocks = re.findall(pat + r"(?=\n\n|\Z)", res.stdout, re.DOTALL | re.IGNORECASE)
            for block, bdf in blocks:
                lnksta = re.search(r"LnkSta:\s+(Speed\s+[^,]+,\s+Width\s+[^,\n]+)", block)
                if lnksta:
                    found = True
                    print(f"  • Device {bdf}: {lnksta.group(1)}")
        if not found:
            print("  (no matching lspci blocks; check rocminfo / amd-smi)")
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
registered = []
try:
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    registered = list(KVConnectorFactory._registry.keys())
    print(f"Registered Connectors in Factory ({len(registered)}):")
    for c in registered:
        print(f"  • {c}")
except Exception as e:
    print(f"KVConnectorFactory inspection failed: {e}")

connectors_to_test = [
    ("MoRIIOConnector", "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector", "MoRIIOConnector", True),
    ("SimpleCPUOffloadConnector", "vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector", "SimpleCPUOffloadConnector", False),
    ("ExampleConnector", "vllm.distributed.kv_transfer.kv_connector.v1.example_connector", "ExampleConnector", False),
    ("LMCacheConnectorV1", "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector", "LMCacheConnectorV1", False),
    ("NixlConnector", "vllm.distributed.kv_transfer.kv_connector.v1.nixl", "NixlConnector", True),
    ("MooncakeConnector", "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector", "MooncakeConnector", True),
]

print("\\nConnector Lifecycle Classification (Registered -> Python Importable -> Native Ready -> Validated):")
for name, mod_path, cls_name, requires_native in connectors_to_test:
    try:
        mod = importlib.import_module(mod_path)
        cls_obj = getattr(mod, cls_name)
        if name == "MoRIIOConnector":
            is_avail = getattr(mod, "is_moriio_available", lambda: False)()
            if is_avail:
                print(f"  [RUNTIME_READY]          {name}: Python module and native mori.io backend verified.")
            else:
                print(f"  [NATIVE_RUNTIME_MISSING] {name}: Python module importable, but native C++ mori.io library is NOT installed.")
                print(f"                           => [NOT_READY] Ineligible for live P/D benchmark until MoRI-IO is compiled.")
        elif name in ("SimpleCPUOffloadConnector", "ExampleConnector"):
            print(f"  [RUNTIME_READY]          {name}: Python module verified; zero external native C++ dependency.")
        else:
            print(f"  [PYTHON_IMPORTABLE]      {name}: Python module importable (external backend qualification required).")
    except ModuleNotFoundError as mnf:
        print(f"  [MISSING_PYTHON_DEP]     {name}: Missing Python module '{mnf.name}'")
    except Exception as ex:
        print(f"  [FAILED]                 {name}: {repr(ex)}")

# Test MORI native library probe
try:
    import mori.io
    print("\\nNative MoRI Status: [INSTALLED] Found native mori.io library in container.")
except ImportError:
    print("\\nNative MoRI Status: [MISSING] mori.io C++ / Python extension is NOT installed in container.")
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
    print("      Total Direct P2P Handoff Latency  = T_serialize + T_control-plane + T_buffer-ready + T_data-movement + T_decode-admission")
    print("      Host-Staged Shared-Memory Latency = T_D2H + T_metadata + T_shm-sync + T_H2D + T_decode-admission (2x bulk transfers)")
    print("      Host-staged /dev/shm is a correctness & lifecycle harness, NOT a zero-copy transport.\n")

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
    parser = argparse.ArgumentParser(description="Dual homogeneous dGPU hardware & P/D diagnostics (R9700 or MI350P)")
    parser.add_argument("--container", default="rocm-mxfp4-server", help="vLLM container to inspect")
    parser.add_argument("--skip-container", action="store_true", help="Skip container probe")
    parser.add_argument("--gpu-profile", default=os.environ.get("GPU_PROFILE", "auto"), help="r9700 | mi350p | auto")
    parser.add_argument("--check-preflight", action="store_true", help="Exit 0 if 2 matching target dGPUs are present")
    args = parser.parse_args()
    profile = normalize_profile(args.gpu_profile)
    info = profile_info(profile)

    if args.check_preflight:
        passed, count, _ = check_dual_target_preflight(profile)
        if not passed:
            print(f"PREFLIGHT FAILED: Found {count} {info['marketing']} devices (Required: 2).", file=sys.stderr)
            sys.exit(1)
        print(f"PREFLIGHT OK: Found {count} homogeneous {info['marketing']} devices.")
        sys.exit(0)

    probe_host_topology(profile)
    if not args.skip_container:
        probe_container_connectors(args.container)
    calculate_analytical_kv_transfer()

    print_banner("SUMMARY VERDICT")
    print(f"• Target Architecture: Dual homogeneous {info['label']}.")
    print("• Dual-GPU TP=2 / DP=2 / P/D require two cards of the same ISA (gfx1201 or gfx950), never mixed with iGPU.")
    print("• Cross-card A/B: run the same traces twice with GPU_PROFILE=r9700 then GPU_PROFILE=mi350p.")


if __name__ == "__main__":
    main()
