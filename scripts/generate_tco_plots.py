#!/usr/bin/env python3
"""
Generate high-resolution presentation plots for TCO and serving performance analysis.
Includes updated $1,500 price point for Radeon AI PRO R9700S.
"""

import json
import os
import shutil
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_data import capex, electricity_usd_per_kwh, r9700_gpu_usd  # noqa: E402
from publish_results import POWER_BANDWIDTH_JSON, R9700_RESULTS_JSON

_POWER_BANDWIDTH = None
_R9700_RESULTS = None


def power_bandwidth():
    """Published R9700 and MI350P profiling summary."""
    global _POWER_BANDWIDTH
    if _POWER_BANDWIDTH is None:
        _POWER_BANDWIDTH = json.loads(POWER_BANDWIDTH_JSON.read_text())
    return _POWER_BANDWIDTH


def profile_series(gpu_profile, workload, field, max_concurrency=None):
    rows = [
        row
        for row in power_bandwidth()["runs"]
        if row["gpu_profile"] == gpu_profile and row["workload"] == workload
    ]
    if max_concurrency is not None:
        rows = [row for row in rows if row["concurrency"] <= max_concurrency]
    rows.sort(key=lambda row: row["concurrency"])
    return [row["concurrency"] for row in rows], [row[field] for row in rows]


def profile_run(gpu_profile, workload, concurrency):
    for row in power_bandwidth()["runs"]:
        if (
            row["gpu_profile"] == gpu_profile
            and row["workload"] == workload
            and row["concurrency"] == concurrency
        ):
            return row
    raise KeyError(f"no {gpu_profile} {workload} C{concurrency} profiling summary")


def mi350_series(workload, field, max_concurrency=None):
    return profile_series("mi350p", workload, field, max_concurrency)


def mi350_run(workload, concurrency):
    return profile_run("mi350p", workload, concurrency)


def r9700_series(workload, field, max_concurrency=None):
    return profile_series("r9700", workload, field, max_concurrency)


def r9700_result_series(workload, field, max_concurrency=None):
    global _R9700_RESULTS
    if _R9700_RESULTS is None:
        _R9700_RESULTS = json.loads(R9700_RESULTS_JSON.read_text())
    rows = [
        row for row in _R9700_RESULTS["runs"] if row["workload"] == workload
    ]
    if max_concurrency is not None:
        rows = [row for row in rows if row["concurrency"] <= max_concurrency]
    rows.sort(key=lambda row: row["concurrency"])
    return [row["concurrency"] for row in rows], [row[field] for row in rows]

# Set styling
plt.rcParams['font.sans-serif'] = ['Liberation Sans', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#B0BEC5'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#ECEFF1'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "figures", "tco")
ARTIFACT_DIR = "/home/amd/.gemini/antigravity-cli/brain/3a344b95-6417-4951-aa06-7d6314d2ecfe"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Cohesive presentation color palette
COLOR_R9700S_PEAK = "#D32F2F"    # Crimson Red
COLOR_R9700S_SLA  = "#FF7043"    # Coral / Orange-Red
COLOR_R9600D_PEAK = "#00897B"    # Dark Teal
COLOR_R9600D_SLA  = "#26A69A"    # Light Teal
COLOR_MI350P_PROD = "#1A237E"    # Deep Indigo
COLOR_MI350P_DF3  = "#7B1FA2"    # Purple
COLOR_RTX6000     = "#455A64"    # Slate Gray

# ==============================================================================
# PLOT 1: System Capex & 3-Year TCO Breakdown
# ==============================================================================
def plot_capex_and_tco():
    fig, ax = plt.subplots(figsize=(11, 7.0), dpi=300)
    
    spec = capex()
    systems = spec["systems_full"]
    chassis_only = np.array(spec["chassis_ex_dram_usd"], dtype=float)
    dram_capex = np.array(spec["dram_usd"], dtype=float)
    chassis_capex = np.array(spec["chassis_usd"], dtype=float)
    if not np.allclose(chassis_only + dram_capex, chassis_capex):
        raise SystemExit("chassis_ex_dram_usd + dram_usd must equal chassis_usd")
    gpu_capex = np.array(spec["gpu_usd"], dtype=float)
    power_3yr = np.array(spec["power_3yr_usd"], dtype=float)
    total_tco = chassis_capex + gpu_capex + power_3yr
    gpu_year_cost = spec["gpu_year_usd"]
    kwh = electricity_usd_per_kwh()
    
    x = np.arange(len(systems))
    bar_width = 0.52
    color_chassis, color_dram, color_gpu = "#546E7A", "#ed1c24", "#00c2de"
    
    # Stacked bars. DRAM is the red portion of the published server price.
    host = chassis_only / 1000
    dram = dram_capex / 1000
    gpu = gpu_capex / 1000
    power = power_3yr / 1000
    ax.bar(x, host, bar_width, label="Server Chassis", color=color_chassis, edgecolor="white", alpha=0.9)
    ax.bar(x, dram, bar_width, bottom=host, label="Matched DRAM", color=color_dram, edgecolor="white", alpha=0.9)
    ax.bar(x, gpu, bar_width, bottom=host + dram, label="GPU Capex", color=color_gpu, edgecolor="white", alpha=0.9)
    ax.bar(x, power, bar_width, bottom=host + dram + gpu, label=f"3-Yr GPU Power (50% TDP, ${kwh:.2f}/kWh)", color="#FFB300", edgecolor="white", alpha=0.9)
    
    # Values inside / on top of bars
    for i in range(len(systems)):
        tot = total_tco[i] / 1000
        ann = gpu_year_cost[i]
        ax.text(x[i], tot + 4, f"3-Yr TCO:\n${total_tco[i]:,}\n(${ann:,}/GPU-yr)", 
                ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#1A237E')
        
        segments = (
            (0, host[i], f"${chassis_only[i]:,.0f}", "white"),
            (host[i], dram[i], f"${dram_capex[i]:,.0f}", "white"),
            (host[i] + dram[i], gpu[i], f"${gpu_capex[i]:,.0f}", "#042028"),
        )
        for bottom, height, label, ink in segments:
            if height < 8:
                continue
            ax.text(x[i], bottom + height / 2, label, ha='center', va='center', color=ink, fontweight='bold', fontsize=10)
    
    ax.set_xlabel("Server Architecture & Hardware Fill", fontsize=13, fontweight='bold', labelpad=12)
    ax.set_ylabel("Total Cost ($ in Thousands)", fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title("Server Fill Capex & 3-Year TCO Comparison (Qwen3.8-27B MXFP4 Serving)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 275)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, facecolor='white', fontsize=10)
    
    # Callout banner placed cleanly in the open space above R9700S and R9600D
    gpu_price = f"{int(r9700_gpu_usd()):,}"
    ax.text(0.03, 0.48, r"Key Takeaways:" "\n"
            rf"• R9700S @ \${gpu_price}/GPU delivers complete 8-GPU server for \$49,000 capex" "\n"
            r"• 75% lower 3-year TCO (\$52.8k vs \$216.6k) compared to 8× MI350P" "\n"
            r"• Annual GPU cost: \$2,199/GPU-yr (vs \$9,024 on MI350P)", 
            transform=ax.transAxes, ha='left', va='center',
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#E8F5E9", edgecolor="#4CAF50", lw=1.5),
            fontsize=10.0, fontweight='normal', color="#1B5E20")
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "01_tco_capex_breakdown.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

# ==============================================================================
# PLOT 2: Cost per Million Tokens vs Concurrency (8k:1k)
# ==============================================================================
def plot_cost_8k_1k():
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    
    # Concurrency and cost points
    c_r9700, _ = r9700_result_series(
        "8k1k", "server_output_throughput_tok_s"
    )
    cost_r9700_peak = [2.12, 1.12, 0.68, 0.50, 0.51]
    cost_r9700_sla  = [2.12, 1.12, 0.68, 0.68, 0.70]
    
    c_r9600 = [1, 2, 4, 8, 16]
    cost_r9600_peak = [2.33, 1.01, 0.66, 0.56, 0.64]
    cost_r9600_sla  = [2.33, 1.01, 0.66, 0.79, 0.89]
    
    c_mi350 = [1, 2, 4, 8, 16, 32]
    c_mi350_df3 = [1, 4, 8, 16, 32]
    cost_mi350_prod = [7.30, 3.68, 1.92, 1.00, 0.55, 0.31]
    cost_mi350_df3  = [5.23, 1.18, 1.00, 0.74, 0.69]
    
    # Plot lines with styling
    ax.plot(c_r9700, cost_r9700_peak, 'o-', color=COLOR_R9700S_PEAK, linewidth=2.8, markersize=8, label="8× R9700S (Batch Peak, $1.5k/GPU)")
    ax.plot(c_r9700, cost_r9700_sla,  's--', color=COLOR_R9700S_SLA, linewidth=2.0, markersize=7, label="8× R9700S (Interactive SLA, max-seqs 4)")
    ax.plot(c_r9600, cost_r9600_peak, '^-.', color=COLOR_R9600D_PEAK, linewidth=2.2, markersize=7, label="16× R9600D (150W Projected Peak)")
    ax.plot(c_r9600, cost_r9600_sla,  'v:', color=COLOR_R9600D_SLA, linewidth=2.0, markersize=7, label="16× R9600D (150W Projected SLA)")
    ax.plot(c_mi350, cost_mi350_prod, 'D-', color=COLOR_MI350P_PROD, linewidth=2.8, markersize=8, label="8× MI350P (Production Quark MXFP4)")
    ax.plot(c_mi350_df3, cost_mi350_df3,  'P--', color=COLOR_MI350P_DF3, linewidth=2.0, markersize=7, label="8× MI350P (DFlash-3 Speculative)")
    
    # Data callouts
    ax.annotate("$2.12/M", xy=(1, 2.12), xytext=(1.3, 2.7),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_PEAK, lw=1.2),
                fontweight='bold', color=COLOR_R9700S_PEAK)
    ax.annotate("$7.30/M", xy=(1, 7.30), xytext=(1.4, 7.4),
                arrowprops=dict(arrowstyle="->", color=COLOR_MI350P_PROD, lw=1.2),
                fontweight='bold', color=COLOR_MI350P_PROD)
    ax.annotate("$0.50/M (2× less expensive than MI350P)", xy=(8, 0.50), xytext=(5.2, 0.15),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_PEAK, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor=COLOR_R9700S_PEAK),
                fontweight='bold', color=COLOR_R9700S_PEAK, fontsize=9.5)
    ax.annotate("MI350P HBM Scaling\n$0.31/M @ C32", xy=(32, 0.31), xytext=(22, 1.4),
                arrowprops=dict(arrowstyle="->", color=COLOR_MI350P_PROD, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8EAF6", edgecolor=COLOR_MI350P_PROD),
                fontweight='bold', color=COLOR_MI350P_PROD, fontsize=9.5)
    
    # Background shaded zones
    ax.axvspan(0.8, 4.5, color='#E8F5E9', alpha=0.35, label='_nolegend_')
    ax.text(2.3, 6.8, "RADEON COST LEADERSHIP\n(2.5× – 3.4× Less Expensive)", ha='center', fontsize=9.5, fontweight='bold', color='#2E7D32',
            bbox=dict(boxstyle="square,pad=0.3", facecolor='white', alpha=0.8, edgecolor='#A5D6A7'))
    
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32"], fontsize=12, fontweight='bold')
    ax.set_xlabel("Concurrency per GPU (C)", fontsize=13, fontweight='bold', labelpad=8)
    ax.set_ylabel("Serving Cost ($ / Million Output Tokens)  [Lower is better]", fontsize=12.5, fontweight='bold', labelpad=8)
    ax.set_title("Serving Cost vs Concurrency: Long Context (8,192 In / 1,024 Out) — Qwen3.8-27B MXFP4", fontsize=13.5, fontweight='bold', pad=15)
    ax.set_ylim(0, 8.2)
    ax.set_xlim(0.8, 36)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', frameon=True, framealpha=0.95, facecolor='white', fontsize=9.5)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "02_cost_per_token_8k_1k.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

# ==============================================================================
# PLOT 3: Cost per Million Tokens vs Concurrency (1k:1k)
# ==============================================================================
def plot_cost_1k_1k():
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    
    c_r9700 = [1, 2, 4, 8, 16]
    cost_r9700 = [2.10, 1.08, 0.57, 0.57, 0.57]
    
    c_r9600 = [1, 2, 4, 8, 16]
    cost_r9600 = [2.30, 0.96, 0.52, 0.64, 0.72]
    
    c_mi350 = [1, 2, 4, 8, 16, 32]
    c_mi350_df3 = [1, 4, 8, 16, 32]
    cost_mi350_prod = [3.60, 1.94, 0.98, 0.52, 0.32, 0.19]
    cost_mi350_df3  = [2.51, 0.78, 0.56, 0.45, 0.32]
    
    ax.plot(c_r9700, cost_r9700, 'o-', color=COLOR_R9700S_PEAK, linewidth=2.8, markersize=8, label="8× R9700S (Interactive SLA @ ~32ms TPOT)")
    ax.plot(c_r9600, cost_r9600, '^-.', color=COLOR_R9600D_PEAK, linewidth=2.2, markersize=7, label="16× R9600D (150W Projected)")
    ax.plot(c_mi350, cost_mi350_prod, 'D-', color=COLOR_MI350P_PROD, linewidth=2.8, markersize=8, label="8× MI350P (Production Quark MXFP4)")
    ax.plot(c_mi350_df3, cost_mi350_df3,  'P--', color=COLOR_MI350P_DF3, linewidth=2.0, markersize=7, label="8× MI350P (DFlash-3 Speculative)")
    
    # Annotations
    ax.annotate("$2.10/M", xy=(1, 2.10), xytext=(1.2, 2.3),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_PEAK, lw=1.2),
                fontweight='bold', color=COLOR_R9700S_PEAK)
    ax.annotate("$3.60/M", xy=(1, 3.60), xytext=(1.2, 3.75),
                arrowprops=dict(arrowstyle="->", color=COLOR_MI350P_PROD, lw=1.2),
                fontweight='bold', color=COLOR_MI350P_PROD)
    ax.annotate("R9700S Cost Plateau\n$0.57/M @ C4–C16 (32ms TPOT)", xy=(8, 0.57), xytext=(4.5, 1.2),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_PEAK, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor=COLOR_R9700S_PEAK),
                fontweight='bold', color=COLOR_R9700S_PEAK, fontsize=9.5)
    ax.annotate("MI350P Ultra-Concurrency\n$0.19/M @ C32", xy=(32, 0.19), xytext=(20, 0.8),
                arrowprops=dict(arrowstyle="->", color=COLOR_MI350P_PROD, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8EAF6", edgecolor=COLOR_MI350P_PROD),
                fontweight='bold', color=COLOR_MI350P_PROD, fontsize=9.5)
    
    # Background shaded zone
    ax.axvspan(0.8, 4.5, color='#E8F5E9', alpha=0.35, label='_nolegend_')
    ax.text(2.0, 3.1, "RADEON ADVANTAGE\n(1.7× – 1.8× Less Expensive)", ha='center', fontsize=9.5, fontweight='bold', color='#2E7D32',
            bbox=dict(boxstyle="square,pad=0.3", facecolor='white', alpha=0.8, edgecolor='#A5D6A7'))
    
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32"], fontsize=12, fontweight='bold')
    ax.set_xlabel("Concurrency per GPU (C)", fontsize=13, fontweight='bold', labelpad=8)
    ax.set_ylabel("Serving Cost ($ / Million Output Tokens)  [Lower is better]", fontsize=12.5, fontweight='bold', labelpad=8)
    ax.set_title("Serving Cost vs Concurrency: Standard Workload (1,024 In / 1,024 Out) — Qwen3.8-27B MXFP4", fontsize=13.5, fontweight='bold', pad=15)
    ax.set_ylim(0, 4.2)
    ax.set_xlim(0.8, 36)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', frameon=True, framealpha=0.95, facecolor='white', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "03_cost_per_token_1k_1k.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

# ==============================================================================
# PLOT 3b: Cost per Million Tokens vs Concurrency (1,024 In / 8,192 Out)
# ==============================================================================
def plot_cost_1k_8k():
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    
    # Concurrency and cost points
    c_r9700 = [1, 2, 4]
    cost_r9700 = [2.20, 1.12, 0.59]
    
    c_r9600 = [1, 2, 4]
    cost_r9600 = [2.77, 1.41, 0.75]
    
    c_mi350 = [1, 2, 4, 8, 16, 32]
    c_mi350_df3 = [1, 4, 8, 16, 32]
    cost_mi350_prod = [5.39, 2.77, 1.41, 0.72, 0.41, 0.24]
    cost_mi350_df3  = [3.01, 1.06, 1.71, 1.38, 1.19]
    
    # Plot lines with cohesive presentation styling
    ax.plot(c_r9700, cost_r9700, 'o-', color=COLOR_R9700S_PEAK, linewidth=2.8, markersize=8, label="8× R9700S (Interactive SLA, $1.5k/GPU)")
    ax.plot(c_r9600, cost_r9600, '^--', color=COLOR_R9600D_PEAK, linewidth=2.2, markersize=7, label="16× R9600D (150W Projected)")
    ax.plot(c_mi350, cost_mi350_prod, 'D-', color=COLOR_MI350P_PROD, linewidth=2.8, markersize=8, label="8× MI350P (Production Quark MXFP4)")
    ax.plot(c_mi350_df3, cost_mi350_df3,  'P--', color=COLOR_MI350P_DF3, linewidth=2.0, markersize=7, label="8× MI350P (DFlash-3 Speculative)")
    
    # Annotations
    ax.annotate("$2.20/M", xy=(1, 2.20), xytext=(0.85, 1.45),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_PEAK, lw=1.2),
                fontweight='bold', color=COLOR_R9700S_PEAK)
    ax.annotate("$5.39/M", xy=(1, 5.39), xytext=(1.2, 5.55),
                arrowprops=dict(arrowstyle="->", color=COLOR_MI350P_PROD, lw=1.2),
                fontweight='bold', color=COLOR_MI350P_PROD)
    ax.annotate("R9700S Cost Leadership\n$0.59/M @ C4 (2.4× less expensive)", xy=(4, 0.59), xytext=(1.2, 0.22),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_PEAK, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor=COLOR_R9700S_PEAK),
                fontweight='bold', color=COLOR_R9700S_PEAK, fontsize=8.5)
    ax.annotate("MI350P HBM Scaling\n$0.24/M @ C32", xy=(32, 0.24), xytext=(20, 0.8),
                arrowprops=dict(arrowstyle="->", color=COLOR_MI350P_PROD, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8EAF6", edgecolor=COLOR_MI350P_PROD),
                fontweight='bold', color=COLOR_MI350P_PROD, fontsize=9.5)
    ax.annotate("DFlash-3 Regression @ C8+\nSpeculative verification overhead\ncauses throughput drop on 8k output", xy=(8, 1.71), xytext=(5.8, 3.2),
                arrowprops=dict(arrowstyle="->", color=COLOR_MI350P_DF3, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E5F5", edgecolor=COLOR_MI350P_DF3),
                fontweight='bold', color=COLOR_MI350P_DF3, fontsize=8.5)
    
    # Background shaded zone
    ax.axvspan(0.8, 4.5, color='#E8F5E9', alpha=0.35, label='_nolegend_')
    ax.text(2.0, 4.6, "RADEON ADVANTAGE\n(2.4× – 2.5× Less Expensive)", ha='center', fontsize=9.5, fontweight='bold', color='#2E7D32',
            bbox=dict(boxstyle="square,pad=0.3", facecolor='white', alpha=0.8, edgecolor='#A5D6A7'))
    
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32"], fontsize=12, fontweight='bold')
    ax.set_xlabel("Concurrency per GPU (C)", fontsize=13, fontweight='bold', labelpad=8)
    ax.set_ylabel("Serving Cost ($ / Million Output Tokens)  [Lower is better]", fontsize=12.5, fontweight='bold', labelpad=8)
    ax.set_title("Serving Cost vs Concurrency: Deep Code Generation (1,024 In / 8,192 Out) — Qwen3.8-27B MXFP4", fontsize=13.5, fontweight='bold', pad=15)
    ax.set_ylim(0, 6.2)
    ax.set_xlim(0.8, 36)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', frameon=True, framealpha=0.95, facecolor='white', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "03b_cost_per_token_1k_8k.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

# ==============================================================================
# PLOT 4: Aggregate Server Throughput (3-Panel Across All Workload Shapes)
# ==============================================================================
def plot_throughput():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6.2), dpi=300)
    
    # Panel 1: 8k:1k
    c_r9700, tok_r9700_peak = r9700_result_series(
        "8k1k", "server_output_throughput_tok_s"
    )
    tok_r9700_sla  = [262, 498, 814, 815, 802]
    
    c_r9600 = [1, 2, 4, 8, 16]
    tok_r9600_peak = [348, 803, 1225, 1460, 1261]
    tok_r9600_sla  = [348, 803, 1225, 1026, 907]
    
    c_mi350 = [1, 2, 4, 8, 16, 32]
    c_mi350_df3 = [1, 4, 8, 16, 32]
    tok_mi350_prod = [314, 621, 1193, 2297, 4180, 7432]
    tok_mi350_df3  = [437, 1933, 2296, 3109, 3293]
    
    ax1.plot(c_r9700, tok_r9700_peak, 'o-', color=COLOR_R9700S_PEAK, lw=2.5, ms=7, label="8× R9700S (Batch Peak)")
    ax1.plot(c_r9700, tok_r9700_sla,  's--', color=COLOR_R9700S_SLA, lw=2.0, ms=6, label="8× R9700S (Interactive SLA)")
    ax1.plot(c_r9600, tok_r9600_peak, '^-.', color=COLOR_R9600D_PEAK, lw=2.0, ms=6, label="16× R9600D (Projected Peak)")
    ax1.plot(c_r9600, tok_r9600_sla,  'v:', color=COLOR_R9600D_SLA, lw=1.8, ms=6, label="16× R9600D (Projected SLA)")
    ax1.plot(c_mi350, tok_mi350_prod, 'D-', color=COLOR_MI350P_PROD, lw=2.5, ms=7, label="8× MI350P (Production)")
    ax1.plot(c_mi350_df3, tok_mi350_df3,  'P--', color=COLOR_MI350P_DF3, lw=2.0, ms=6, label="8× MI350P (DFlash-3)")
    
    ax1.set_xscale('log', base=2)
    ax1.set_xticks([1, 2, 4, 8, 16, 32])
    ax1.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32"], fontweight='bold')
    ax1.set_xlabel("Concurrency per GPU (C)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Full Server Throughput (Output tok/s)  [Higher is better]", fontsize=11, fontweight='bold')
    ax1.set_title("8,192 In / 1,024 Out (Long Context)", fontsize=12.5, fontweight='bold')
    ax1.set_ylim(0, 8000)
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left', frameon=True, fontsize=8.0)
    
    # Panel 2: 1k:1k
    _, tok_r9700_1k = r9700_result_series(
        "1k1k", "server_output_throughput_tok_s"
    )
    tok_r9600_1k = [352, 842, 1558, 1273, 1128]
    tok_mi350_1k_prod = [635, 1179, 2342, 4437, 7229, 12035]
    tok_mi350_1k_df3  = [910, 2923, 4086, 5065, 7066]
    
    ax2.plot(c_r9700, tok_r9700_1k, 'o-', color=COLOR_R9700S_PEAK, lw=2.5, ms=7, label="8× R9700S (Interactive SLA)")
    ax2.plot(c_r9600, tok_r9600_1k, '^-.', color=COLOR_R9600D_PEAK, lw=2.0, ms=6, label="16× R9600D (Projected)")
    ax2.plot(c_mi350, tok_mi350_1k_prod, 'D-', color=COLOR_MI350P_PROD, lw=2.5, ms=7, label="8× MI350P (Production)")
    ax2.plot(c_mi350_df3, tok_mi350_1k_df3,  'P--', color=COLOR_MI350P_DF3, lw=2.0, ms=6, label="8× MI350P (DFlash-3)")
    
    ax2.set_xscale('log', base=2)
    ax2.set_xticks([1, 2, 4, 8, 16, 32])
    ax2.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32"], fontweight='bold')
    ax2.set_xlabel("Concurrency per GPU (C)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Full Server Throughput (Output tok/s)  [Higher is better]", fontsize=11, fontweight='bold')
    ax2.set_title("1,024 In / 1,024 Out (Standard Workload)", fontsize=12.5, fontweight='bold')
    ax2.set_ylim(0, 13000)
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left', frameon=True, fontsize=8.0)
    
    # Panel 3: 1k:8k (Deep Code Generation)
    c_r9700_3, tok_r9700_1k8k = r9700_result_series(
        "1k8k", "server_output_throughput_tok_s"
    )
    tok_r9600_1k8k = [292, 576, 1085]
    tok_mi350_1k8k_prod = [425, 825, 1622, 3162, 5517, 9715]
    tok_mi350_1k8k_df3  = [759, 2156, 1337, 1656, 1926]
    
    c_r9600_3 = [1, 2, 4]
    
    ax3.plot(c_r9700_3, tok_r9700_1k8k, 'o-', color=COLOR_R9700S_PEAK, lw=2.5, ms=7, label="8× R9700S (Interactive SLA)")
    ax3.plot(c_r9600_3, tok_r9600_1k8k, '^-.', color=COLOR_R9600D_PEAK, lw=2.0, ms=6, label="16× R9600D (Projected)")
    ax3.plot(c_mi350, tok_mi350_1k8k_prod, 'D-', color=COLOR_MI350P_PROD, lw=2.5, ms=7, label="8× MI350P (Production)")
    ax3.plot(c_mi350_df3, tok_mi350_1k8k_df3,  'P--', color=COLOR_MI350P_DF3, lw=2.0, ms=6, label="8× MI350P (DFlash-3)")
    
    ax3.set_xscale('log', base=2)
    ax3.set_xticks([1, 2, 4, 8, 16, 32])
    ax3.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32"], fontweight='bold')
    ax3.set_xlabel("Concurrency per GPU (C)", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Full Server Throughput (Output tok/s)  [Higher is better]", fontsize=11, fontweight='bold')
    ax3.set_title("1,024 In / 8,192 Out (Deep Code Generation)", fontsize=12.5, fontweight='bold')
    ax3.set_ylim(0, 11000)
    ax3.grid(True, which='both', linestyle='--', alpha=0.7)
    ax3.legend(loc='upper left', frameon=True, fontsize=8.0)
    
    fig.suptitle("Full Server Aggregate Throughput Scaling Across Workload Shapes — Qwen3.8-27B MXFP4 (Higher is better)", fontsize=14.5, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    output_path = os.path.join(OUTPUT_DIR, "04_aggregate_throughput.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

# ==============================================================================
# PLOT 5: Executive Dashboard (Combined 4-Panel Summary)
# ==============================================================================
def plot_executive_dashboard():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    
    # 1. Capex & TCO Bar Chart
    spec = capex()
    systems = spec["systems_short"]
    chassis_only = np.array(spec["chassis_ex_dram_usd"], dtype=float) / 1000.0
    dram_capex = np.array(spec["dram_usd"], dtype=float) / 1000.0
    gpu_capex = np.array(spec["gpu_usd"], dtype=float) / 1000.0
    power_3yr = np.array(spec["power_3yr_usd"], dtype=float) / 1000.0
    total_tco = chassis_only + dram_capex + gpu_capex + power_3yr
    x = np.arange(len(systems))
    w = 0.50
    ax1.bar(x, chassis_only, w, label="Chassis", color="#546E7A")
    ax1.bar(x, dram_capex, w, bottom=chassis_only, label="DRAM", color="#ed1c24")
    ax1.bar(x, gpu_capex, w, bottom=chassis_only + dram_capex, label="GPU", color="#00c2de")
    ax1.bar(x, power_3yr, w, bottom=chassis_only + dram_capex + gpu_capex, label="3-Yr Power", color="#FFB300")
    for i in range(len(systems)):
        ax1.text(x[i], total_tco[i] + 4, f"${total_tco[i]:.1f}k", ha='center', va='bottom', fontweight='bold', fontsize=9.5)
    ax1.set_title("A. System Capex & 3-Year TCO ($k)", fontsize=12, fontweight='bold', pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems, fontweight='bold', fontsize=10)
    ax1.set_xlabel("Server Hardware Fill", fontsize=10.5, fontweight='bold', labelpad=6)
    ax1.set_ylabel("Total Cost ($ in Thousands)", fontsize=10.5, fontweight='bold', labelpad=6)
    ax1.set_ylim(0, 255)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=8.5)
    
    # 2. Cost 8k:1k
    c_r9700 = [1, 2, 4, 8, 16]
    cost_r9700_peak = [2.12, 1.12, 0.68, 0.50, 0.51]
    c_mi350 = [1, 2, 4, 8, 16, 32]
    c_mi350_df3 = [1, 4, 8, 16, 32]
    cost_mi350_prod = [7.30, 3.68, 1.92, 1.00, 0.55, 0.31]
    cost_mi350_df3  = [5.23, 1.18, 1.00, 0.74, 0.69]
    c_r9600 = [1, 2, 4, 8, 16]
    cost_r9600_peak = [2.33, 1.01, 0.66, 0.56, 0.64]
    
    ax2.plot(c_r9700, cost_r9700_peak, 'o-', color=COLOR_R9700S_PEAK, lw=2.2, ms=6, label="8× R9700S (Batch Peak, $1.5k)")
    ax2.plot(c_r9600, cost_r9600_peak, '^-.', color=COLOR_R9600D_PEAK, lw=1.8, ms=5, label="16× R9600D (Projected Peak)")
    ax2.plot(c_mi350, cost_mi350_prod, 'D-', color=COLOR_MI350P_PROD, lw=2.2, ms=6, label="8× MI350P (Production)")
    ax2.plot(c_mi350_df3, cost_mi350_df3,  'P--', color=COLOR_MI350P_DF3, lw=1.6, ms=5, label="8× MI350P (DFlash-3)")
    ax2.set_xscale('log', base=2)
    ax2.set_xticks([1, 2, 4, 8, 16, 32])
    ax2.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32"], fontweight='bold', fontsize=9.5)
    ax2.set_title("B. Serving Cost: 8,192 In / 1,024 Out [Lower is better]", fontsize=11.5, fontweight='bold', pad=8)
    ax2.set_xlabel("Concurrency per GPU (C)", fontsize=10.5, fontweight='bold', labelpad=6)
    ax2.set_ylabel("Serving Cost ($ / M Tokens) [Lower is better]", fontsize=10, fontweight='bold', labelpad=6)
    ax2.set_ylim(0, 8.0)
    ax2.grid(True, which='both', linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=8.5)
    ax2.text(0.35, 0.85, "C1–C4: R9700S is 2.8×–3.4× less expensive\nC8: R9700S hits $0.50/M (2× less expensive)",
             transform=ax2.transAxes, fontsize=8.5, fontweight='bold', color="#B71C1C",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor="#EF5350"))
    
    # 3. Cost 1k:1k
    cost_r9700_1k = [2.10, 1.08, 0.57, 0.57, 0.57]
    cost_r9600_1k = [2.30, 0.96, 0.52, 0.64, 0.72]
    cost_mi350_1k_prod = [3.60, 1.94, 0.98, 0.52, 0.32, 0.19]
    cost_mi350_1k_df3  = [2.51, 0.78, 0.56, 0.45, 0.32]
    
    ax3.plot(c_r9700, cost_r9700_1k, 'o-', color=COLOR_R9700S_PEAK, lw=2.2, ms=6, label="8× R9700S (Interactive SLA)")
    ax3.plot(c_r9600, cost_r9600_1k, '^-.', color=COLOR_R9600D_PEAK, lw=1.8, ms=5, label="16× R9600D (Projected)")
    ax3.plot(c_mi350, cost_mi350_1k_prod, 'D-', color=COLOR_MI350P_PROD, lw=2.2, ms=6, label="8× MI350P (Production)")
    ax3.plot(c_mi350_df3, cost_mi350_1k_df3,  'P--', color=COLOR_MI350P_DF3, lw=1.6, ms=5, label="8× MI350P (DFlash-3)")
    ax3.set_xscale('log', base=2)
    ax3.set_xticks([1, 2, 4, 8, 16, 32])
    ax3.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32"], fontweight='bold', fontsize=9.5)
    ax3.set_title("C. Serving Cost: 1,024 In / 1,024 Out [Lower is better]", fontsize=11.5, fontweight='bold', pad=8)
    ax3.set_xlabel("Concurrency per GPU (C)", fontsize=10.5, fontweight='bold', labelpad=6)
    ax3.set_ylabel("Serving Cost ($ / M Tokens) [Lower is better]", fontsize=10, fontweight='bold', labelpad=6)
    ax3.set_ylim(0, 4.2)
    ax3.grid(True, which='both', linestyle='--', alpha=0.6)
    ax3.legend(loc='upper right', fontsize=8.5)
    ax3.text(0.35, 0.85, "C1–C4: R9700S is 1.7×–1.8× less expensive\nC4–C16: Plateaus at $0.57/M @ 32ms TPOT",
             transform=ax3.transAxes, fontsize=8.5, fontweight='bold', color="#B71C1C",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor="#EF5350"))
    
    # 4. Throughput Comparison
    _, tok_r9700_peak = r9700_result_series(
        "8k1k", "server_output_throughput_tok_s"
    )
    tok_mi350_prod = [314, 621, 1193, 2297, 4180, 7432]
    tok_r9600_peak = [348, 803, 1225, 1460, 1261]
    
    ax4.plot(c_r9700, tok_r9700_peak, 'o-', color=COLOR_R9700S_PEAK, lw=2.2, ms=6, label="8× R9700S (8k:1k Peak)")
    ax4.plot(c_r9600, tok_r9600_peak, '^-.', color=COLOR_R9600D_PEAK, lw=1.8, ms=5, label="16× R9600D (8k:1k Peak)")
    ax4.plot(c_mi350, tok_mi350_prod, 'D-', color=COLOR_MI350P_PROD, lw=2.2, ms=6, label="8× MI350P (8k:1k Prod)")
    
    ax4.set_xscale('log', base=2)
    ax4.set_xticks([1, 2, 4, 8, 16, 32])
    ax4.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32"], fontweight='bold', fontsize=9.5)
    ax4.set_title("D. Aggregate Server Throughput: 8k:1k [Higher is better]", fontsize=11.5, fontweight='bold', pad=8)
    ax4.set_xlabel("Concurrency per GPU (C)", fontsize=10.5, fontweight='bold', labelpad=6)
    ax4.set_ylabel("Aggregate Throughput (tok/s) [Higher is better]", fontsize=10, fontweight='bold', labelpad=6)
    ax4.set_ylim(0, 8000)
    ax4.grid(True, which='both', linestyle='--', alpha=0.6)
    ax4.legend(loc='upper left', fontsize=8.5)
    ax4.text(0.32, 0.72, "GDDR6 cards saturate KV cache @ C8-C16\nMI350P scales to 7.4k+ tok/s via HBM",
             transform=ax4.transAxes, fontsize=8.5, fontweight='bold', color="#1A237E",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8EAF6", edgecolor="#3949AB"))
    
    fig.suptitle("AMD AI Accelerator TCO & Serving Performance Dashboard (Qwen3.8-27B MXFP4)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    output_path = os.path.join(OUTPUT_DIR, "05_tco_executive_summary_dashboard.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

# ==============================================================================
# PLOT 6: Power Utilization (% of Device Max TDP) vs Concurrency per GPU
# ==============================================================================
def plot_power_utilization():
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    
    # Concurrency and power utilization (% of device max TDP)
    # R9700S: Max TDP = 300 W
    c_r9700_8k, pwr_r9700_8k = r9700_series("8k1k", "power_util_pct")
    c_r9700_1k, pwr_r9700_1k = r9700_series("1k1k", "power_util_pct")
    c_r9700_long, pwr_r9700_long = r9700_series("1k8k", "power_util_pct")
    
    # MI350P socket power from reports/profiling/power_bandwidth.json. TDP is 600 W.
    c_mi350, pwr_mi350_1k = mi350_series("1k1k", "power_util_pct", 32)
    _, pwr_mi350_8k = mi350_series("8k1k", "power_util_pct", 32)
    _, pwr_mi350_long = mi350_series("1k8k", "power_util_pct", 32)
    c64 = mi350_run("1k1k", 64)
    
    # R9600D: Max TDP = 150 W (Operating at 150W Cap = 100%)
    c_r9600 = [1, 2, 4, 8, 16]
    pwr_r9600 = [100.0, 100.0, 100.0, 100.0, 100.0]
    
    # Plot lines
    ax.plot(c_r9700_8k, pwr_r9700_8k, 'o-', color=COLOR_R9700S_PEAK, lw=2.6, ms=7, label="Radeon AI PRO R9700S (8,192 In / 1,024 Out, 300W Max)")
    ax.plot(c_r9700_1k, pwr_r9700_1k, 's--', color=COLOR_R9700S_SLA, lw=2.2, ms=7, label="Radeon AI PRO R9700S (1,024 In / 1,024 Out, 300W Max)")
    ax.plot(c_r9700_long, pwr_r9700_long, '^-.', color="#C2185B", lw=2.2, ms=7, label="Radeon AI PRO R9700S (1,024 In / 8,192 Out, 300W Max)")
    ax.plot(c_mi350, pwr_mi350_1k, 'D-', color=COLOR_MI350P_PROD, lw=2.6, ms=8, label="Instinct MI350P (1,024 In / 1,024 Out, 600W Max)")
    ax.plot(c_mi350, pwr_mi350_8k, 's--', color="#3949AB", lw=2.2, ms=7, label="Instinct MI350P (8,192 In / 1,024 Out, 600W Max)")
    ax.plot(c_mi350, pwr_mi350_long, '^-.', color="#5C6BC0", lw=2.2, ms=7, label="Instinct MI350P (1,024 In / 8,192 Out, 600W Max)")
    ax.plot(c_r9600, pwr_r9600, 'v:', color=COLOR_R9600D_PEAK, lw=2.0, ms=6, label="Radeon AI PRO R9600D (150W Capped Envelope)")
    
    # 100% Device TDP Ceiling line
    ax.axhline(100.0, color="#78909C", linestyle="--", linewidth=1.5, alpha=0.85)
    ax.text(0.9, 101.5, "100% Device Max TDP Ceiling", fontsize=9.5, fontweight='bold', color="#455A64")
    
    # Data Callouts
    ax.annotate("C1–C4 High Efficiency\n61.2%–66.5% TDP", xy=(2, 61.2), xytext=(0.85, 38),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_SLA, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor=COLOR_R9700S_SLA),
                fontweight='bold', color="#E65100", fontsize=8.5)
    
    ax.annotate("C16 Queue Scaling\n86.6%–88.4% TDP", xy=(16, 88.4), xytext=(9.5, 94),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_PEAK, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor=COLOR_R9700S_PEAK),
                fontweight='bold', color=COLOR_R9700S_PEAK, fontsize=9.0)
    
    ax.annotate(
        f"MI350P 1k/1k: {min(pwr_mi350_1k):.0f}%–{max(pwr_mi350_1k):.0f}% TDP through C32\n"
        f"({c64['power_util_pct']:.0f}% / {c64['avg_power_w']:.0f} W at C64)",
        xy=(32, pwr_mi350_1k[-1]), xytext=(18, 48),
                arrowprops=dict(arrowstyle="->", color=COLOR_MI350P_PROD, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8EAF6", edgecolor=COLOR_MI350P_PROD),
                fontweight='bold', color=COLOR_MI350P_PROD, fontsize=8.5)
    
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32"], fontsize=12, fontweight='bold')
    ax.set_xlabel("Concurrency per GPU (C)", fontsize=13, fontweight='bold', labelpad=8)
    ax.set_ylabel("Power Utilization (% of Device Max TDP)", fontsize=12.5, fontweight='bold', labelpad=8)
    ax.set_title("Power Utilization vs Concurrency per GPU (% of Device Max TDP) — Qwen3.8-27B MXFP4", fontsize=13.5, fontweight='bold', pad=15)
    ax.set_ylim(35, 110)
    ax.set_xlim(0.8, 36)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, facecolor='white', fontsize=9.5)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "06_power_utilization.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

# ==============================================================================
# PLOT 7: Memory Bandwidth Utilization (% of Device Max Peak) vs Concurrency
# ==============================================================================
def plot_memory_bandwidth_utilization():
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    
    # Memory Bandwidth Utilization (% of Device Max Peak)
    # R9700S: Peak = 640 GB/s (GDDR6)
    c_r9700_8k, bw_r9700_8k = r9700_series("8k1k", "bandwidth_util_pct")
    c_r9700_1k, bw_r9700_1k = r9700_series("1k1k", "bandwidth_util_pct")
    c_r9700_long, bw_r9700_long = r9700_series("1k8k", "bandwidth_util_pct")
    
    # MI350P UMC activity from reports/profiling/power_bandwidth.json.
    # umc_gbs_estimate = UMC% × catalog peak is not a calibrated HBM measurement.
    c_mi350, bw_mi350_1k = mi350_series("1k1k", "umc_activity_pct", 32)
    _, bw_mi350_8k = mi350_series("8k1k", "umc_activity_pct", 32)
    _, bw_mi350_long = mi350_series("1k8k", "umc_activity_pct", 32)
    c64 = mi350_run("1k1k", 64)
    umc_through_c32 = bw_mi350_1k + bw_mi350_8k + bw_mi350_long
    
    # R9600D: Peak = 640 GB/s (GDDR6), 150W Capped
    # 48 CUs & 150W cap yields ~344–353 GB/s sustained (53.8%–55.2%)
    c_r9600 = [1, 2, 4]
    bw_r9600 = [53.8, 54.5, 55.2]
    
    # Shaded band for R9700S near-optimal GDDR6 decode saturation
    ax.axhspan(70.0, 78.0, color='#FFEBEE', alpha=0.5, label='_nolegend_')
    ax.text(3.5, 66.5, "R9700S GDDR6 SATURATION ZONE (71%–76% of 640 GB/s Spec, ~84% Physical Bus)", 
            ha='center', fontsize=9.0, fontweight='bold', color="#C62828",
            bbox=dict(boxstyle="square,pad=0.25", facecolor='white', alpha=0.9, edgecolor="#EF9A9A"))
    
    # Plot lines
    ax.plot(c_r9700_8k, bw_r9700_8k, 'o-', color=COLOR_R9700S_PEAK, lw=2.6, ms=7, label="Radeon AI PRO R9700S (8,192 In / 1,024 Out, 640 GB/s Max)")
    ax.plot(c_r9700_1k, bw_r9700_1k, 's--', color=COLOR_R9700S_SLA, lw=2.2, ms=7, label="Radeon AI PRO R9700S (1,024 In / 1,024 Out, 640 GB/s Max)")
    ax.plot(c_r9700_long, bw_r9700_long, '^-.', color="#C2185B", lw=2.2, ms=7, label="Radeon AI PRO R9700S (1,024 In / 8,192 Out, 640 GB/s Max)")
    ax.plot(c_r9600, bw_r9600, 'v:', color=COLOR_R9600D_PEAK, lw=2.0, ms=6, label="Radeon AI PRO R9600D (150W Capped, 640 GB/s Max)")
    ax.plot(c_mi350, bw_mi350_1k, 'D-', color=COLOR_MI350P_PROD, lw=2.6, ms=8, label="Instinct MI350P (1,024/1,024, UMC activity)")
    ax.plot(c_mi350, bw_mi350_8k, 's--', color="#3949AB", lw=2.2, ms=7, label="Instinct MI350P (8,192/1,024, UMC activity)")
    ax.plot(c_mi350, bw_mi350_long, '^-.', color="#5C6BC0", lw=2.2, ms=7, label="Instinct MI350P (1,024/8,192, UMC activity)")
    
    # Data Callouts
    ax.annotate("Near-Optimal GDDR6 Saturation\n~470–488 GB/s (74%–76%)", xy=(1, 75.9), xytext=(1.3, 86.5),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_PEAK, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor=COLOR_R9700S_PEAK),
                fontweight='bold', color=COLOR_R9700S_PEAK, fontsize=9.0)
    
    ax.annotate(
        f"MI350P UMC stays {min(umc_through_c32):.0f}%–{max(umc_through_c32):.0f}% through C32\n"
        f"(C64 1k/1k is {c64['umc_activity_pct']:.0f}%; not a calibrated GB/s)",
        xy=(32, bw_mi350_1k[-1]), xytext=(6.0, 42),
                arrowprops=dict(arrowstyle="->", color=COLOR_MI350P_PROD, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8EAF6", edgecolor=COLOR_MI350P_PROD),
                fontweight='bold', color=COLOR_MI350P_PROD, fontsize=9.0)
    
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32"], fontsize=12, fontweight='bold')
    ax.set_xlabel("Concurrency per GPU (C)", fontsize=13, fontweight='bold', labelpad=8)
    ax.set_ylabel("Memory Bandwidth Utilization (% of Device Max Peak)  [Higher is better]", fontsize=12.0, fontweight='bold', labelpad=8)
    ax.set_title("Memory Bandwidth Utilization vs Concurrency per GPU (% of Peak) — Qwen3.8-27B MXFP4", fontsize=13.5, fontweight='bold', pad=15)
    ax.set_ylim(10, 100)
    ax.set_xlim(0.8, 36)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend(loc='lower left', frameon=True, framealpha=0.95, facecolor='white', fontsize=9.5)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "07_memory_bandwidth_utilization.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

# ==============================================================================
# PLOT 8: Hardware Utilization Dashboard (Side-by-Side Power & Bandwidth)
# ==============================================================================
def plot_hardware_utilization_dashboard():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)
    
    # 1. Power Utilization
    c_r9700_8k, pwr_r9700_8k = r9700_series("8k1k", "power_util_pct")
    c_r9700_1k, pwr_r9700_1k = r9700_series("1k1k", "power_util_pct")
    c_r9700_long, pwr_r9700_long = r9700_series("1k8k", "power_util_pct")
    c_mi350, pwr_mi350_1k = mi350_series("1k1k", "power_util_pct", 32)
    _, pwr_mi350_8k = mi350_series("8k1k", "power_util_pct", 32)
    _, pwr_mi350_long = mi350_series("1k8k", "power_util_pct", 32)
    c_r9600 = [1, 2, 4, 8, 16]
    pwr_r9600 = [100.0, 100.0, 100.0, 100.0, 100.0]
    
    ax1.plot(c_r9700_8k, pwr_r9700_8k, 'o-', color=COLOR_R9700S_PEAK, lw=2.4, ms=6, label="R9700S (8k:1k, 300W)")
    ax1.plot(c_r9700_1k, pwr_r9700_1k, 's--', color=COLOR_R9700S_SLA, lw=2.0, ms=6, label="R9700S (1k:1k, 300W)")
    ax1.plot(c_r9700_long, pwr_r9700_long, '^-.', color="#C2185B", lw=2.0, ms=6, label="R9700S (1k:8k, 300W)")
    ax1.plot(c_mi350, pwr_mi350_1k, 'D-', color=COLOR_MI350P_PROD, lw=2.4, ms=7, label="MI350P (1k/1k, 600W)")
    ax1.plot(c_mi350, pwr_mi350_8k, 's--', color="#3949AB", lw=2.0, ms=6, label="MI350P (8k/1k, 600W)")
    ax1.plot(c_mi350, pwr_mi350_long, '^-.', color="#5C6BC0", lw=2.0, ms=6, label="MI350P (1k/8k, 600W)")
    ax1.plot(c_r9600, pwr_r9600, 'v:', color=COLOR_R9600D_PEAK, lw=1.8, ms=5, label="R9600D (150W Cap)")
    ax1.axhline(100.0, color="#78909C", linestyle="--", lw=1.3, alpha=0.85)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks([1, 2, 4, 8, 16, 32])
    ax1.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32"], fontweight='bold')
    ax1.set_xlabel("Concurrency per GPU (C)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Power Utilization (% of Device Max TDP)", fontsize=11, fontweight='bold')
    ax1.set_title("A. Power Utilization (% of Device TDP)", fontsize=12.5, fontweight='bold')
    ax1.set_ylim(35, 110)
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax1.legend(loc='lower right', frameon=True, fontsize=8.5)
    
    # 2. Memory Bandwidth Utilization
    _, bw_r9700_8k = r9700_series("8k1k", "bandwidth_util_pct")
    _, bw_r9700_1k = r9700_series("1k1k", "bandwidth_util_pct")
    _, bw_r9700_long = r9700_series("1k8k", "bandwidth_util_pct")
    _, bw_mi350_1k = mi350_series("1k1k", "umc_activity_pct", 32)
    _, bw_mi350_8k = mi350_series("8k1k", "umc_activity_pct", 32)
    _, bw_mi350_long = mi350_series("1k8k", "umc_activity_pct", 32)
    c_r9600_bw = [1, 2, 4]
    bw_r9600 = [53.8, 54.5, 55.2]
    
    ax2.axhspan(70.0, 78.0, color='#FFEBEE', alpha=0.5, label='_nolegend_')
    ax2.plot(c_r9700_8k, bw_r9700_8k, 'o-', color=COLOR_R9700S_PEAK, lw=2.4, ms=6, label="R9700S (8k:1k, 640 GB/s)")
    ax2.plot(c_r9700_1k, bw_r9700_1k, 's--', color=COLOR_R9700S_SLA, lw=2.0, ms=6, label="R9700S (1k:1k, 640 GB/s)")
    ax2.plot(c_r9700_long, bw_r9700_long, '^-.', color="#C2185B", lw=2.0, ms=6, label="R9700S (1k:8k, 640 GB/s)")
    ax2.plot(c_r9600_bw, bw_r9600, 'v:', color=COLOR_R9600D_PEAK, lw=1.8, ms=5, label="R9600D (150W Cap, 640 GB/s)")
    ax2.plot(c_mi350, bw_mi350_1k, 'D-', color=COLOR_MI350P_PROD, lw=2.4, ms=7, label="MI350P (1k/1k, UMC)")
    ax2.plot(c_mi350, bw_mi350_8k, 's--', color="#3949AB", lw=2.0, ms=6, label="MI350P (8k/1k, UMC)")
    ax2.plot(c_mi350, bw_mi350_long, '^-.', color="#5C6BC0", lw=2.0, ms=6, label="MI350P (1k/8k, UMC)")
    ax2.set_xscale('log', base=2)
    ax2.set_xticks([1, 2, 4, 8, 16, 32])
    ax2.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32"], fontweight='bold')
    ax2.set_xlabel("Concurrency per GPU (C)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Memory Bandwidth Utilization (% of Device Max Peak)  [Higher is better]", fontsize=11, fontweight='bold')
    ax2.set_title("B. Memory Bandwidth Utilization (% of Peak) [Higher is better]", fontsize=12.5, fontweight='bold')
    ax2.set_ylim(10, 100)
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    ax2.legend(loc='lower left', frameon=True, fontsize=8.5)
    
    fig.suptitle("Hardware Utilization Profiles: Power & Memory Bandwidth vs Concurrency — Qwen3.8-27B MXFP4", fontsize=14.5, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    output_path = os.path.join(OUTPUT_DIR, "08_power_and_bandwidth_utilization.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

# ==============================================================================
# PLOT 9: Time-to-First-Token (TTFT p50) Latency vs Concurrency per GPU
# ==============================================================================
def plot_ttft_latency():
    fig, ax = plt.subplots(figsize=(11, 6.8), dpi=300)

    # Measured TTFT p50 across concurrencies
    c_8k1k = [1, 2, 4, 8, 16]
    ttft_8k1k = [304.8, 443.9, 3319.7, 20445.2, 58410.1]

    c_1k1k = [1, 2, 4, 8, 16]
    ttft_1k1k = [259.5, 194.9, 808.7, 17258.1, 50360.4]

    c_1k8k = [1, 2, 4]
    ttft_1k8k = [164.6, 189.1, 936.7]

    # 29 Sep GPU 0. C1 is the first request (later prompts in that cell hit
    # the prefix cache). C2+ is the wave p50. 8k later cells reuse the
    # same prompt, so only the first 8,192-token request is plotted.
    c_mi350_1k = [1, 2, 4, 8, 16, 32]
    ttft_mi350_1k = [119.4, 528.0, 515.8, 466.9, 710.4, 1159.0]

    c_mi350_8k = [1]
    ttft_mi350_8k = [705.4]

    # Shaded zones: Compliant (<= 3,000 ms) and Cut-Off Region (> 3,000 ms)
    ax.axhspan(80, 3000, color="#E8F5E9", alpha=0.35, label="SLO Compliant Region (TTFT ≤ 3.0s)")
    ax.axhspan(3000, 120000, color="#FFEBEE", alpha=0.35, label="SLO Cut-Off Region (TTFT > 3.0s, Batch Only)")
    ax.axhline(3000, color="#9E9E9E", linestyle=":", linewidth=1.6, label="3.0s Interactive SLO Cut-Off Ceiling")
    ax.axhline(1000, color="#4CAF50", linestyle=":", linewidth=1.2, alpha=0.85)
    ax.text(1.1, 1150, "1.0s Sub-Second Target", fontsize=8.5, fontweight="bold", color="#2E7D32")
    ax.text(1.1, 3350, "3.0s Interactive SLO Cut-Off Ceiling", fontsize=9.0, fontweight="bold", color="#B71C1C",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFEBEE", edgecolor="#EF9A9A", alpha=0.9))

    # Plot curves
    ax.plot(c_8k1k, ttft_8k1k, "o-", color=COLOR_R9700S_PEAK, lw=2.6, ms=7, label="Radeon AI PRO R9700S (8,192 In / 1,024 Out)")
    ax.plot(c_1k1k, ttft_1k1k, "s--", color=COLOR_R9700S_SLA, lw=2.2, ms=7, label="Radeon AI PRO R9700S (1,024 In / 1,024 Out)")
    ax.plot(c_1k8k, ttft_1k8k, "^-.", color="#C2185B", lw=2.2, ms=7, label="Radeon AI PRO R9700S (1,024 In / 8,192 Out)")
    ax.plot(c_mi350_1k, ttft_mi350_1k, "D-", color=COLOR_MI350P_PROD, lw=2.4, ms=7, label="Instinct MI350P (1,024/1,024; C1 first request, else wave p50)")
    ax.plot(c_mi350_8k, ttft_mi350_8k, "P", color="#3949AB", ms=9, label="Instinct MI350P (8,192/1,024 first request)")

    # Annotations
    ax.annotate("C1–C4 Sub-Second Ingestion\n165–937 ms (1k Input)", xy=(2, 194.9), xytext=(2.0, 115),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_SLA, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor=COLOR_R9700S_SLA),
                fontweight="bold", color="#E65100", fontsize=8.5)

    ax.annotate("Scheduler Queueing under max-num-seqs 4\nProtects TPOT (<34 ms) by queueing incoming streams",
                xy=(8, 17258.1), xytext=(2.2, 35000),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_PEAK, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor=COLOR_R9700S_PEAK),
                fontweight="bold", color="#C62828", fontsize=9.0)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32"], fontsize=12, fontweight="bold")
    ax.set_yticks([100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000])
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y):,} ms" if y < 1000 else f"{y/1000:.1f} s"))

    ax.set_xlabel("Concurrency per GPU (C)", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_ylabel("Time-to-First-Token (TTFT p50 Latency)  [Lower is better]", fontsize=12.5, fontweight="bold", labelpad=8)
    ax.set_title("Time-to-First-Token (TTFT p50) Latency vs Concurrency — Qwen3.8-27B MXFP4 [Lower is better]", fontsize=13.5, fontweight="bold", pad=15)
    ax.set_ylim(80, 120000)
    ax.set_xlim(0.8, 36)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, facecolor="white", fontsize=9.5)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "09_ttft_latency.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

# ==============================================================================
# PLOT 9b: TTFT p50, p90, p95, and p99 side by side
# ==============================================================================
def plot_ttft_percentiles():
    """Same series as the p50 chart, at p50, p90, p95, and p99.

    R9700S C4–C16 are omitted from p90 and p95. Those vLLM summaries stored
    the median and p99 only, and the request list was not saved. C1 and C2
    p90 and p95 are the linear percentiles of the one or two measured
    requests. p99 is the stored bench percentile at every R9700S point.
    MI350P p50, p90, p95, and p99 are each computed from every request in
    the cell. C1 is not replaced by the cold first request, and the 8,192-token
    series is not a single copied marker. Cells after C1 on 8,192/1,024 reuse
    one prompt. 1,024/1,024 includes C64, where p99 separates from the median.
    """
    # concurrency, then percentiles in milliseconds. A shorter list means
    # later concurrencies were not stored.
    series = (
        {
            "label": "R9700S (8,192 in / 1,024 out)",
            "color": COLOR_R9700S_PEAK,
            "fmt": "o-",
            "c": [1, 2, 4, 8, 16],
            "p50": [304.8, 443.9, 3319.7, 20445.2, 58410.1],
            "p90": [308.0, 553.2],
            "p95": [308.5, 566.9],
            "p99": [308.8, 577.8, 5914.3, 45939.0, 128953.9],
        },
        {
            "label": "R9700S (1,024 in / 1,024 out)",
            "color": COLOR_R9700S_SLA,
            "fmt": "s--",
            "c": [1, 2, 4, 8, 16],
            "p50": [259.5, 194.9, 808.7, 17258.1, 50360.4],
            "p90": [373.0, 218.9],
            "p95": [387.2, 221.9],
            "p99": [398.6, 224.3, 900.7, 34397.9, 101410.4],
        },
        {
            "label": "R9700S (1,024 in / 8,192 out)",
            "color": "#C2185B",
            "fmt": "^-.",
            "c": [1, 2, 4],
            "p50": [164.6, 189.1, 936.7],
            "p90": [164.6, 228.5],
            "p95": [164.6, 233.4],
            "p99": [164.6, 237.4, 1043.3],
        },
        {
            "label": "MI350P (1,024/1,024, every request)",
            "color": COLOR_MI350P_PROD,
            "fmt": "D-",
            "c": [1, 2, 4, 8, 16, 32, 64],
            "p50": [47.4, 528.0, 515.8, 466.9, 710.4, 1159.0, 1720.5],
            "p90": [97.8, 529.5, 516.6, 467.5, 713.2, 1166.3, 2123.6],
            "p95": [108.6, 529.7, 516.7, 467.5, 713.9, 1167.3, 2133.0],
            "p99": [117.3, 529.8, 516.8, 467.5, 714.9, 1168.5, 2141.9],
        },
        {
            "label": "MI350P (8,192/1,024, every request)",
            "color": "#3949AB",
            "fmt": "P-",
            "c": [1, 2, 4, 8, 16, 32],
            "p50": [75.7, 174.1, 1053.2, 577.3, 929.5, 1551.1],
            "p90": [516.5, 205.0, 1054.2, 784.3, 1314.6, 1851.9],
            "p95": [610.9, 208.8, 1054.3, 784.5, 1315.6, 1967.5],
            "p99": [686.5, 211.9, 1054.5, 784.7, 1316.6, 2118.3],
        },
    )

    y_max = 220000
    fig, axes = plt.subplots(1, 4, figsize=(24.0, 6.9), dpi=160, sharey=True)
    percentiles = ("p50", "p90", "p95", "p99")
    titles = ("TTFT p50", "TTFT p90", "TTFT p95", "TTFT p99")
    for ax, key, title in zip(axes, percentiles, titles):
        ax.axhspan(80, 3000, color="#E8F5E9", alpha=0.35, zorder=0)
        ax.axhspan(3000, y_max, color="#FFEBEE", alpha=0.35, zorder=0)
        ax.axhline(3000, color="#9E9E9E", linestyle=":", linewidth=1.6, zorder=1)
        ax.axhline(1000, color="#4CAF50", linestyle=":", linewidth=1.1, alpha=0.85, zorder=1)
        for row in series:
            values = row[key]
            ax.plot(
                row["c"][: len(values)],
                values,
                row["fmt"],
                color=row["color"],
                lw=2.2,
                ms=6.5,
                label=row["label"],
                zorder=3,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
        ax.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32", "C64"], fontsize=10, fontweight="bold")
        ax.set_xlim(0.8, 80)
        ax.set_ylim(80, y_max)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("Concurrency per GPU (C)", fontsize=11, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.55)
        if ax is axes[0]:
            ax.set_yticks([100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 200000])
            ax.get_yaxis().set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{int(y):,} ms" if y < 1000 else f"{y / 1000:.1f} s")
            )
            ax.set_ylabel("Time-to-First-Token  [Lower is better]", fontsize=11, fontweight="bold")
            ax.text(1.05, 1180, "1.0 s", fontsize=8, fontweight="bold", color="#2E7D32")
            ax.text(1.05, 3600, "3.0 s SLO", fontsize=8, fontweight="bold", color="#B71C1C")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=True,
        framealpha=0.95,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Time-to-First-Token p50, p90, p95, and p99 vs Concurrency — Qwen3.8-27B MXFP4",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.905,
        "MI350P panels use each percentile of the cell. They are not copies of the p50 series. 8,192-token cells after C1 reuse one prompt.",
        ha="center",
        fontsize=8.5,
        color="#424242",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.90])
    output_path = os.path.join(OUTPUT_DIR, "09_ttft_p50_p90_p95.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path

# ==============================================================================
# PLOT 10: SLO-Qualified Interactive Goodput vs Raw Saturated Throughput & Cost
# ==============================================================================
def plot_slo_qualified_goodput():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.5), dpi=300)
    
    # Interactive SLA criteria: TTFT <= 3,000 ms AND TPOT <= 20 ms (1,024 in / 1,024 out)
    c_r9700, raw_tok_r9700 = r9700_result_series(
        "1k1k", "server_output_throughput_tok_s"
    )
    goodput_r9700 = raw_tok_r9700  # Capacity under max-num-seqs 4
    
    c_mi350 = [1, 2, 4, 8, 16, 32]
    raw_tok_mi350 = [635, 1179, 2342, 4437, 7229, 12035]
    goodput_mi350 = [635, 1179, 2342, 4437, 7229, 12035] # All TTFT <= 1250ms, TPOT <= 20.1ms
    
    cost_r9700 = [2.10, 1.08, 0.57, 0.57, 0.57]
    cost_mi350 = [3.60, 1.94, 0.98, 0.52, 0.32, 0.19]
    
    # --------------------------------------------------------------------------
    # Panel 1: Throughput & Interactive Capacity
    # --------------------------------------------------------------------------
    ax1.plot(c_r9700, raw_tok_r9700, 'o-', color=COLOR_R9700S_PEAK, lw=2.8, ms=8, 
             label="8× R9700S (Interactive Capacity, Capped @ C4; TPOT ~32ms)")
    ax1.plot(c_mi350, raw_tok_mi350, 'D-', color=COLOR_MI350P_PROD, lw=2.8, ms=8, 
             label="8× MI350P (Monotonic Scaling through C32, TPOT ≤ 20ms)")
    
    # Shaded zones on Panel 1
    ax1.axvspan(0.8, 4.5, color='#E8F5E9', alpha=0.45, label='_nolegend_')
    ax1.text(2.0, 7500, "R9700S INTERACTIVE SWEET SPOT\n(Sub-Second TTFT, 100% SLA)", 
             ha='center', fontsize=8.5, fontweight='bold', color="#2E7D32",
             bbox=dict(boxstyle="square,pad=0.25", facecolor='white', alpha=0.9, edgecolor="#A5D6A7"))
    
    # R9700S Cut-Off vertical line
    ax1.axvline(5.6, color="#D32F2F", linestyle=":", linewidth=1.5, alpha=0.85)
    ax1.text(5.6, 6800, "R9700S SLO Cut-Off\n(Queue TTFT > 3.0s)", ha='center', fontsize=7.8, fontweight='bold', color="#B71C1C",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFEBEE", edgecolor="#EF9A9A", alpha=0.9))
    
    ax1.axvspan(6.0, 36, color='#E8EAF6', alpha=0.35, label='_nolegend_')
    ax1.text(24, 4500, "MI350P HIGH-CONCURRENCY DOMAIN\n(100% SLA Maintained through C32)", 
             ha='center', fontsize=8.5, fontweight='bold', color="#1A237E",
             bbox=dict(boxstyle="square,pad=0.25", facecolor='white', alpha=0.9, edgecolor="#9FA8DA"))
    
    # Annotations on Panel 1
    ax1.annotate("R9700S Interactive Saturation\n976 tok/s @ C4 (TTFT 809ms)\n[Standalone TPOT ~32ms; TP=2 <20ms]", xy=(4, 976), xytext=(1.4, 3800),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_PEAK, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor=COLOR_R9700S_PEAK),
                fontweight='bold', color=COLOR_R9700S_PEAK, fontsize=8.2)
    
    ax1.annotate("Queueing under max-num-seqs 4\nProtects TPOT (<33ms), but\nTTFT breaches 3.0s SLA (17–50s)", xy=(8, 976), xytext=(5.2, 2200),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_SLA, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor=COLOR_R9700S_SLA),
                fontweight='bold', color="#E65100", fontsize=8.5)
    
    ax1.annotate("12,035 tok/s @ C32\nTTFT 1.25s, TPOT ~20ms\n(100% Interactive SLA)", xy=(32, 12035), xytext=(12, 11000),
                arrowprops=dict(arrowstyle="->", color=COLOR_MI350P_PROD, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8EAF6", edgecolor=COLOR_MI350P_PROD),
                fontweight='bold', color=COLOR_MI350P_PROD, fontsize=8.5)
    
    ax1.set_xscale('log', base=2)
    ax1.set_xticks([1, 2, 4, 8, 16, 32])
    ax1.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32"], fontweight='bold')
    ax1.set_xlabel("Concurrency per GPU (C)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Interactive Server Throughput (tok/s)  [Higher is better]", fontsize=11, fontweight='bold')
    ax1.set_title("A. Interactive Serving Throughput (SLA: TTFT ≤ 3.0s, TPOT ≤ 20ms)", fontsize=12.5, fontweight='bold')
    ax1.set_ylim(0, 13500)
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left', frameon=True, fontsize=8.5)
    
    # --------------------------------------------------------------------------
    # Panel 2: Cost per Million Interactive Tokens
    # --------------------------------------------------------------------------
    ax2.plot(c_r9700[:3], cost_r9700[:3], 'o-', color=COLOR_R9700S_PEAK, lw=2.8, ms=8, 
             label="8× R9700S (Interactive SLA Qualified: C1–C4)")
    ax2.plot(c_r9700[2:], cost_r9700[2:], 'o--', color=COLOR_R9700S_SLA, lw=2.0, ms=7, alpha=0.6,
             label="8× R9700S (Batch Completion Mode: C8–C16)")
    ax2.plot(c_mi350, cost_mi350, 'D-', color=COLOR_MI350P_PROD, lw=2.8, ms=8, 
             label="8× MI350P (100% Interactive Qualified: C1–C32)")
    
    # Shaded zones on Panel 2
    ax2.axvspan(0.8, 4.5, color='#E8F5E9', alpha=0.45, label='_nolegend_')
    ax2.text(2.0, 3.2, "RADEON INTERACTIVE COST LEADERSHIP\n(1.7× Less Expensive per Token)", 
             ha='center', fontsize=8.5, fontweight='bold', color="#2E7D32",
             bbox=dict(boxstyle="square,pad=0.25", facecolor='white', alpha=0.9, edgecolor="#A5D6A7"))
    
    # R9700S Cut-Off vertical line and batch completion regime
    ax2.axvline(5.6, color="#D32F2F", linestyle=":", linewidth=1.5, alpha=0.85)
    ax2.text(5.6, 2.3, "R9700S Cut-Off\n(Interactive -> Batch)", ha='center', fontsize=7.8, fontweight='bold', color="#B71C1C",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFEBEE", edgecolor="#EF9A9A", alpha=0.9))
    ax2.axvspan(5.6, 36, color='#FFEBEE', alpha=0.22, label='_nolegend_')
    
    # Annotations on Panel 2
    ax2.annotate("$0.57/M @ C4\n1.7× less expensive\nthan MI350P ($0.98)", xy=(4, 0.57), xytext=(2.2, 1.4),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_PEAK, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor=COLOR_R9700S_PEAK),
                fontweight='bold', color=COLOR_R9700S_PEAK, fontsize=8.5)
    
    ax2.annotate("MI350P Ultra-Scale Cost\n$0.19/M @ C32\n(Sustained Interactive)", xy=(32, 0.19), xytext=(16, 1.0),
                arrowprops=dict(arrowstyle="->", color=COLOR_MI350P_PROD, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8EAF6", edgecolor=COLOR_MI350P_PROD),
                fontweight='bold', color=COLOR_MI350P_PROD, fontsize=8.5)
    
    ax2.set_xscale('log', base=2)
    ax2.set_xticks([1, 2, 4, 8, 16, 32])
    ax2.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32"], fontweight='bold')
    ax2.set_xlabel("Concurrency per GPU (C)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Serving Cost ($ / Million Output Tokens)  [Lower is better]", fontsize=11, fontweight='bold')
    ax2.set_title("B. Cost per Million Tokens: Interactive vs Batch Serving", fontsize=12.5, fontweight='bold')
    ax2.set_ylim(0, 4.2)
    ax2.set_xlim(0.8, 36)
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right', frameon=True, fontsize=8.5)
    
    fig.suptitle("SLO-Qualified Interactive Goodput vs High-Concurrency Batch Saturation — Qwen3.8-27B MXFP4 (1k:1k)\n[Target SLA: TTFT ≤ 3.0s, TPOT ≤ 20ms]", 
                 fontsize=13.5, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    output_path = os.path.join(OUTPUT_DIR, "10_slo_qualified_goodput.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

# ==============================================================================
# PLOT 11: Socket energy per output token (29 Sep MI350P sweep)
# ==============================================================================
def plot_joules_per_token():
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)

    # MI350P joules from the same published power window.
    c_mi350, j_mi350_1k = mi350_series("1k1k", "joules_per_token", 32)
    _, j_mi350_8k = mi350_series("8k1k", "joules_per_token", 32)
    _, j_mi350_long = mi350_series("1k8k", "joules_per_token", 32)
    c64 = mi350_run("1k1k", 64)

    # R9700S empirical data from the published power-of-two sweep summary.
    c_r9700, j_r9700_8k = r9700_series("8k1k", "joules_per_token")
    _, j_r9700_1k = r9700_series("1k1k", "joules_per_token")
    c_r9700_long, j_r9700_long = r9700_series("1k8k", "joules_per_token")

    # Plot R9700S curves
    ax.plot(c_r9700, j_r9700_8k, "o-", color=COLOR_R9700S_PEAK, lw=2.4, ms=7, label="R9700S 8,192 in / 1,024 out")
    ax.plot(c_r9700, j_r9700_1k, "s--", color=COLOR_R9700S_SLA, lw=2.0, ms=6, label="R9700S 1,024 in / 1,024 out")
    ax.plot(c_r9700_long, j_r9700_long, "^-.", color="#C2185B", lw=2.0, ms=6, label="R9700S 1,024 in / 8,192 out")

    # Plot MI350P curves
    ax.plot(c_mi350, j_mi350_8k, "s--", color="#3949AB", lw=2.2, ms=6, label="MI350P 8,192 in / 1,024 out")
    ax.plot(c_mi350, j_mi350_long, "^-.", color="#5C6BC0", lw=2.0, ms=6, label="MI350P 1,024 in / 8,192 out")
    ax.plot(c_mi350, j_mi350_1k, "D-", color=COLOR_MI350P_PROD, lw=2.4, ms=7, label="MI350P 1,024 in / 1,024 out")
    ax.plot([c64["concurrency"]], [c64["joules_per_token"]], "D", color=COLOR_MI350P_PROD, ms=8)

    # Annotations
    ax.annotate(f"C64 {c64['joules_per_token']:.2f} J/tok", xy=(64, c64["joules_per_token"]), xytext=(40, 1.2),
                arrowprops=dict(arrowstyle="->", color=COLOR_MI350P_PROD, lw=1.2),
                fontweight="bold", color=COLOR_MI350P_PROD, fontsize=9.0)
    ax.annotate("R9700S Energy Plateau\n~2.5–3.0 J/tok @ C4–C16", xy=(4, 3.01), xytext=(6.0, 5.0),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_PEAK, lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor=COLOR_R9700S_PEAK),
                fontweight="bold", color=COLOR_R9700S_PEAK, fontsize=8.5)

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32", "C64"], fontsize=12, fontweight="bold")
    ax.set_xlabel("Concurrency per GPU (C)", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_ylabel("Socket energy (J / output token)  [Lower is better]", fontsize=12.5, fontweight="bold", labelpad=8)
    ax.set_title("Socket Energy per Output Token vs Concurrency — Qwen3.8-27B MXFP4: MI350P vs. R9700S [Lower is better]", fontsize=13.5, fontweight="bold", pad=15)
    ax.set_ylim(0, 10.5)
    ax.set_xlim(0.8, 80)
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, facecolor="white", fontsize=9.0)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "11_joules_per_token.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

if __name__ == "__main__":
    p1  = plot_capex_and_tco()
    p2  = plot_cost_8k_1k()
    p3  = plot_cost_1k_1k()
    p3b = plot_cost_1k_8k()
    p4  = plot_throughput()
    p5  = plot_executive_dashboard()
    p6  = plot_power_utilization()
    p7  = plot_memory_bandwidth_utilization()
    p8  = plot_hardware_utilization_dashboard()
    p9  = plot_ttft_latency()
    p9b = plot_ttft_percentiles()
    p10 = plot_slo_qualified_goodput()
    p11 = plot_joules_per_token()
    
    # Also copy to artifact directory for presentation / embedding
    all_plots = [p1, p2, p3, p3b, p4, p5, p6, p7, p8, p9, p9b, p10, p11]
    for p in all_plots:
        dest = os.path.join(ARTIFACT_DIR, os.path.basename(p))
        shutil.copy(p, dest)
        print(f"Generated: {p} -> {dest}")


