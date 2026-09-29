#!/usr/bin/env python3
"""
Generate high-resolution presentation plots for TCO and serving performance analysis.
Includes updated $1,500 price point for Radeon AI PRO R9700S.
"""

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

# Set styling
plt.rcParams['font.sans-serif'] = ['Liberation Sans', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#B0BEC5'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#ECEFF1'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures", "tco")
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
    
    systems = [
        "8× R9700S\n(256GB DRAM)",
        "16× R9600D\n(512GB DRAM)",
        "8× RTX PRO 6000\n(768GB DRAM)",
        "8× MI350P\n(1,536GB DRAM)"
    ]
    
    chassis_capex = np.array([37000, 49000, 62000, 89000])
    gpu_capex     = np.array([12000, 24000, 128000, 120000])
    power_3yr     = np.array([3787,  3787,  7574,   7574])
    total_tco     = chassis_capex + gpu_capex + power_3yr
    gpu_year_cost = [2199, 1600, 8232, 9024]
    
    x = np.arange(len(systems))
    bar_width = 0.52
    
    # Stacked bars
    p1 = ax.bar(x, chassis_capex / 1000, bar_width, label="Server Chassis (Matched DRAM)", color="#546E7A", edgecolor="white", alpha=0.9)
    p2 = ax.bar(x, gpu_capex / 1000, bar_width, bottom=chassis_capex / 1000, label="GPU Capex", color="#E53935", edgecolor="white", alpha=0.9)
    p3 = ax.bar(x, power_3yr / 1000, bar_width, bottom=(chassis_capex + gpu_capex) / 1000, label="3-Yr GPU Power (50% TDP, $0.12/kWh)", color="#FFB300", edgecolor="white", alpha=0.9)
    
    # Values inside / on top of bars
    for i in range(len(systems)):
        tot = total_tco[i] / 1000
        ann = gpu_year_cost[i]
        ax.text(x[i], tot + 4, f"3-Yr TCO:\n${total_tco[i]:,}\n(${ann:,}/GPU-yr)", 
                ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#1A237E')
        
        # Internal labels
        ch_val = chassis_capex[i] / 1000
        gpu_val = gpu_capex[i] / 1000
        ax.text(x[i], ch_val / 2, f"${chassis_capex[i]:,}", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        ax.text(x[i], ch_val + gpu_val / 2, f"${gpu_capex[i]:,}", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    
    ax.set_xlabel("Server Architecture & Hardware Fill", fontsize=13, fontweight='bold', labelpad=12)
    ax.set_ylabel("Total Cost ($ in Thousands)", fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title("Server Fill Capex & 3-Year TCO Comparison (Qwen3.8-27B Serving)", fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 275)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, facecolor='white', fontsize=10)
    
    # Callout banner placed cleanly in the open space above R9700S and R9600D
    ax.text(0.03, 0.48, r"Key Takeaways:" "\n"
            r"• R9700S @ \$1,500/GPU delivers complete 8-GPU server for \$49,000 capex" "\n"
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
    c_r9700 = [1, 2, 4, 8, 16]
    cost_r9700_peak = [2.12, 1.12, 0.68, 0.50, 0.51]
    cost_r9700_sla  = [2.12, 1.12, 0.68, 0.68, 0.70]
    
    c_r9600 = [1, 2, 4, 8, 16]
    cost_r9600_peak = [2.33, 1.01, 0.66, 0.56, 0.64]
    cost_r9600_sla  = [2.33, 1.01, 0.66, 0.79, 0.89]
    
    c_mi350 = [1, 4, 8, 16, 32]
    cost_mi350_prod = [7.30, 1.92, 1.00, 0.55, 0.31]
    cost_mi350_df3  = [5.23, 1.18, 1.00, 0.74, 0.69]
    
    # Plot lines with styling
    ax.plot(c_r9700, cost_r9700_peak, 'o-', color=COLOR_R9700S_PEAK, linewidth=2.8, markersize=8, label="8× R9700S (Batch Peak, $1.5k/GPU)")
    ax.plot(c_r9700, cost_r9700_sla,  's--', color=COLOR_R9700S_SLA, linewidth=2.0, markersize=7, label="8× R9700S (Interactive SLA, max-seqs 4)")
    ax.plot(c_r9600, cost_r9600_peak, '^-.', color=COLOR_R9600D_PEAK, linewidth=2.2, markersize=7, label="16× R9600D (150W Projected Peak)")
    ax.plot(c_r9600, cost_r9600_sla,  'v:', color=COLOR_R9600D_SLA, linewidth=2.0, markersize=7, label="16× R9600D (150W Projected SLA)")
    ax.plot(c_mi350, cost_mi350_prod, 'D-', color=COLOR_MI350P_PROD, linewidth=2.8, markersize=8, label="8× MI350P (Production Quark MXFP4)")
    ax.plot(c_mi350, cost_mi350_df3,  'P--', color=COLOR_MI350P_DF3, linewidth=2.0, markersize=7, label="8× MI350P (DFlash-3 Speculative)")
    
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
    ax.set_title("Serving Cost vs Concurrency: Long Context (8,192 In / 1,024 Out)", fontsize=15, fontweight='bold', pad=15)
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
    
    c_mi350 = [1, 4, 8, 16, 32]
    cost_mi350_prod = [3.60, 0.98, 0.52, 0.32, 0.19]
    cost_mi350_df3  = [2.51, 0.78, 0.56, 0.45, 0.32]
    
    ax.plot(c_r9700, cost_r9700, 'o-', color=COLOR_R9700S_PEAK, linewidth=2.8, markersize=8, label="8× R9700S (Interactive SLA @ ~32ms TPOT)")
    ax.plot(c_r9600, cost_r9600, '^-.', color=COLOR_R9600D_PEAK, linewidth=2.2, markersize=7, label="16× R9600D (150W Projected)")
    ax.plot(c_mi350, cost_mi350_prod, 'D-', color=COLOR_MI350P_PROD, linewidth=2.8, markersize=8, label="8× MI350P (Production Quark MXFP4)")
    ax.plot(c_mi350, cost_mi350_df3,  'P--', color=COLOR_MI350P_DF3, linewidth=2.0, markersize=7, label="8× MI350P (DFlash-3 Speculative)")
    
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
    ax.set_title("Serving Cost vs Concurrency: Standard Workload (1,024 In / 1,024 Out)", fontsize=15, fontweight='bold', pad=15)
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
    
    c_mi350 = [1, 4, 8, 16, 32]
    cost_mi350_prod = [5.39, 1.41, 0.72, 0.41, 0.24]
    cost_mi350_df3  = [3.01, 1.06, 1.71, 1.38, 1.19]
    
    # Plot lines with cohesive presentation styling
    ax.plot(c_r9700, cost_r9700, 'o-', color=COLOR_R9700S_PEAK, linewidth=2.8, markersize=8, label="8× R9700S (Interactive SLA, $1.5k/GPU)")
    ax.plot(c_r9600, cost_r9600, '^--', color=COLOR_R9600D_PEAK, linewidth=2.2, markersize=7, label="16× R9600D (150W Projected)")
    ax.plot(c_mi350, cost_mi350_prod, 'D-', color=COLOR_MI350P_PROD, linewidth=2.8, markersize=8, label="8× MI350P (Production Quark MXFP4)")
    ax.plot(c_mi350, cost_mi350_df3,  'P--', color=COLOR_MI350P_DF3, linewidth=2.0, markersize=7, label="8× MI350P (DFlash-3 Speculative)")
    
    # Annotations
    ax.annotate("$2.20/M", xy=(1, 2.20), xytext=(0.85, 1.45),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_PEAK, lw=1.2),
                fontweight='bold', color=COLOR_R9700S_PEAK)
    ax.annotate("$5.39/M", xy=(1, 5.39), xytext=(1.2, 5.55),
                arrowprops=dict(arrowstyle="->", color=COLOR_MI350P_PROD, lw=1.2),
                fontweight='bold', color=COLOR_MI350P_PROD)
    ax.annotate("R9700S Cost Leadership\n$0.59/M @ C4 (2.4× less expensive)", xy=(4, 0.59), xytext=(2.2, 0.25),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_PEAK, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor=COLOR_R9700S_PEAK),
                fontweight='bold', color=COLOR_R9700S_PEAK, fontsize=9.5)
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
    ax.set_title("Serving Cost vs Concurrency: Deep Code Generation (1,024 In / 8,192 Out)", fontsize=15, fontweight='bold', pad=15)
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
    c_r9700 = [1, 2, 4, 8, 16]
    tok_r9700_peak = [262, 498, 814, 1109, 1087]
    tok_r9700_sla  = [262, 498, 814, 815, 802]
    
    c_r9600 = [1, 2, 4, 8, 16]
    tok_r9600_peak = [348, 803, 1225, 1460, 1261]
    tok_r9600_sla  = [348, 803, 1225, 1026, 907]
    
    c_mi350 = [1, 4, 8, 16, 32]
    tok_mi350_prod = [314, 1193, 2297, 4180, 7432]
    tok_mi350_df3  = [437, 1933, 2296, 3109, 3293]
    
    ax1.plot(c_r9700, tok_r9700_peak, 'o-', color=COLOR_R9700S_PEAK, lw=2.5, ms=7, label="8× R9700S (Batch Peak)")
    ax1.plot(c_r9700, tok_r9700_sla,  's--', color=COLOR_R9700S_SLA, lw=2.0, ms=6, label="8× R9700S (Interactive SLA)")
    ax1.plot(c_r9600, tok_r9600_peak, '^-.', color=COLOR_R9600D_PEAK, lw=2.0, ms=6, label="16× R9600D (Projected Peak)")
    ax1.plot(c_r9600, tok_r9600_sla,  'v:', color=COLOR_R9600D_SLA, lw=1.8, ms=6, label="16× R9600D (Projected SLA)")
    ax1.plot(c_mi350, tok_mi350_prod, 'D-', color=COLOR_MI350P_PROD, lw=2.5, ms=7, label="8× MI350P (Production)")
    ax1.plot(c_mi350, tok_mi350_df3,  'P--', color=COLOR_MI350P_DF3, lw=2.0, ms=6, label="8× MI350P (DFlash-3)")
    
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
    tok_r9700_1k = [266, 516, 976, 976, 977]
    tok_r9600_1k = [352, 842, 1558, 1273, 1128]
    tok_mi350_1k_prod = [635, 2342, 4437, 7229, 12035]
    tok_mi350_1k_df3  = [910, 2923, 4086, 5065, 7066]
    
    ax2.plot(c_r9700, tok_r9700_1k, 'o-', color=COLOR_R9700S_PEAK, lw=2.5, ms=7, label="8× R9700S (Interactive SLA)")
    ax2.plot(c_r9600, tok_r9600_1k, '^-.', color=COLOR_R9600D_PEAK, lw=2.0, ms=6, label="16× R9600D (Projected)")
    ax2.plot(c_mi350, tok_mi350_1k_prod, 'D-', color=COLOR_MI350P_PROD, lw=2.5, ms=7, label="8× MI350P (Production)")
    ax2.plot(c_mi350, tok_mi350_1k_df3,  'P--', color=COLOR_MI350P_DF3, lw=2.0, ms=6, label="8× MI350P (DFlash-3)")
    
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
    tok_r9700_1k8k = [254, 500, 942]
    tok_r9600_1k8k = [292, 576, 1085]
    tok_mi350_1k8k_prod = [425, 1622, 3162, 5517, 9715]
    tok_mi350_1k8k_df3  = [759, 2156, 1337, 1656, 1926]
    
    c_r9700_3 = [1, 2, 4]
    c_r9600_3 = [1, 2, 4]
    
    ax3.plot(c_r9700_3, tok_r9700_1k8k, 'o-', color=COLOR_R9700S_PEAK, lw=2.5, ms=7, label="8× R9700S (Interactive SLA)")
    ax3.plot(c_r9600_3, tok_r9600_1k8k, '^-.', color=COLOR_R9600D_PEAK, lw=2.0, ms=6, label="16× R9600D (Projected)")
    ax3.plot(c_mi350, tok_mi350_1k8k_prod, 'D-', color=COLOR_MI350P_PROD, lw=2.5, ms=7, label="8× MI350P (Production)")
    ax3.plot(c_mi350, tok_mi350_1k8k_df3,  'P--', color=COLOR_MI350P_DF3, lw=2.0, ms=6, label="8× MI350P (DFlash-3)")
    
    ax3.set_xscale('log', base=2)
    ax3.set_xticks([1, 2, 4, 8, 16, 32])
    ax3.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32"], fontweight='bold')
    ax3.set_xlabel("Concurrency per GPU (C)", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Full Server Throughput (Output tok/s)  [Higher is better]", fontsize=11, fontweight='bold')
    ax3.set_title("1,024 In / 8,192 Out (Deep Code Generation)", fontsize=12.5, fontweight='bold')
    ax3.set_ylim(0, 11000)
    ax3.grid(True, which='both', linestyle='--', alpha=0.7)
    ax3.legend(loc='upper left', frameon=True, fontsize=8.0)
    
    fig.suptitle("Full Server Aggregate Throughput Scaling Across Workload Shapes (Higher is better)", fontsize=15, fontweight='bold', y=0.98)
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
    systems = ["8× R9700S", "16× R9600D", "8× RTX 6000", "8× MI350P"]
    chassis_capex = np.array([37, 49, 62, 89])
    gpu_capex     = np.array([12, 24, 128, 120])
    power_3yr     = np.array([3.8, 3.8, 7.6, 7.6])
    total_tco     = chassis_capex + gpu_capex + power_3yr
    x = np.arange(len(systems))
    w = 0.50
    ax1.bar(x, chassis_capex, w, label="Chassis Capex", color="#546E7A")
    ax1.bar(x, gpu_capex, w, bottom=chassis_capex, label="GPU Capex", color="#E53935")
    ax1.bar(x, power_3yr, w, bottom=chassis_capex + gpu_capex, label="3-Yr Power", color="#FFB300")
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
    c_mi350 = [1, 4, 8, 16, 32]
    cost_mi350_prod = [7.30, 1.92, 1.00, 0.55, 0.31]
    cost_mi350_df3  = [5.23, 1.18, 1.00, 0.74, 0.69]
    c_r9600 = [1, 2, 4, 8, 16]
    cost_r9600_peak = [2.33, 1.01, 0.66, 0.56, 0.64]
    
    ax2.plot(c_r9700, cost_r9700_peak, 'o-', color=COLOR_R9700S_PEAK, lw=2.2, ms=6, label="8× R9700S (Batch Peak, $1.5k)")
    ax2.plot(c_r9600, cost_r9600_peak, '^-.', color=COLOR_R9600D_PEAK, lw=1.8, ms=5, label="16× R9600D (Projected Peak)")
    ax2.plot(c_mi350, cost_mi350_prod, 'D-', color=COLOR_MI350P_PROD, lw=2.2, ms=6, label="8× MI350P (Production)")
    ax2.plot(c_mi350, cost_mi350_df3,  'P--', color=COLOR_MI350P_DF3, lw=1.6, ms=5, label="8× MI350P (DFlash-3)")
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
    cost_mi350_1k_prod = [3.60, 0.98, 0.52, 0.32, 0.19]
    cost_mi350_1k_df3  = [2.51, 0.78, 0.56, 0.45, 0.32]
    
    ax3.plot(c_r9700, cost_r9700_1k, 'o-', color=COLOR_R9700S_PEAK, lw=2.2, ms=6, label="8× R9700S (Interactive SLA)")
    ax3.plot(c_r9600, cost_r9600_1k, '^-.', color=COLOR_R9600D_PEAK, lw=1.8, ms=5, label="16× R9600D (Projected)")
    ax3.plot(c_mi350, cost_mi350_1k_prod, 'D-', color=COLOR_MI350P_PROD, lw=2.2, ms=6, label="8× MI350P (Production)")
    ax3.plot(c_mi350, cost_mi350_1k_df3,  'P--', color=COLOR_MI350P_DF3, lw=1.6, ms=5, label="8× MI350P (DFlash-3)")
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
    tok_r9700_peak = [262, 498, 814, 1109, 1087]
    tok_mi350_prod = [314, 1193, 2297, 4180, 7432]
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
    ax4.text(0.48, 0.15, "GDDR6 cards saturate KV cache @ C8-C16\nMI350P scales to 7.4k+ tok/s via HBM",
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
    c_r9700_8k = [1, 2, 4, 8, 16]
    pwr_r9700_8k = [226.5 / 300 * 100, 186.1 / 300 * 100, 199.4 / 300 * 100, 238.5 / 300 * 100, 265.1 / 300 * 100]
    
    c_r9700_1k = [1, 2, 4, 8, 16]
    pwr_r9700_1k = [226.8 / 300 * 100, 183.7 / 300 * 100, 187.9 / 300 * 100, 230.0 / 300 * 100, 259.9 / 300 * 100]
    
    c_r9700_long = [1, 2, 4]
    pwr_r9700_long = [195.3 / 300 * 100, 195.2 / 300 * 100, 195.4 / 300 * 100]
    
    # MI350P: Max TDP = 600 W (Control 1k/1k: 400W @ C1, 410W @ C8)
    c_mi350 = [1, 8]
    pwr_mi350 = [400.0 / 600 * 100, 410.0 / 600 * 100]
    
    # R9600D: Max TDP = 150 W (Operating at 150W Cap = 100%)
    c_r9600 = [1, 2, 4, 8, 16]
    pwr_r9600 = [100.0, 100.0, 100.0, 100.0, 100.0]
    
    # Plot lines
    ax.plot(c_r9700_8k, pwr_r9700_8k, 'o-', color=COLOR_R9700S_PEAK, lw=2.6, ms=7, label="Radeon AI PRO R9700S (8,192 In / 1,024 Out, 300W Max)")
    ax.plot(c_r9700_1k, pwr_r9700_1k, 's--', color=COLOR_R9700S_SLA, lw=2.2, ms=7, label="Radeon AI PRO R9700S (1,024 In / 1,024 Out, 300W Max)")
    ax.plot(c_r9700_long, pwr_r9700_long, '^-.', color="#C2185B", lw=2.2, ms=7, label="Radeon AI PRO R9700S (1,024 In / 8,192 Out, 300W Max)")
    ax.plot(c_mi350, pwr_mi350, 'D-', color=COLOR_MI350P_PROD, lw=2.6, ms=8, label="Instinct MI350P (CDNA 4, 600W Max TDP)")
    ax.plot(c_r9600, pwr_r9600, 'v:', color=COLOR_R9600D_PEAK, lw=2.0, ms=6, label="Radeon AI PRO R9600D (150W Capped Envelope)")
    
    # 100% Device TDP Ceiling line
    ax.axhline(100.0, color="#78909C", linestyle="--", linewidth=1.5, alpha=0.85)
    ax.text(0.9, 101.5, "100% Device Max TDP Ceiling", fontsize=9.5, fontweight='bold', color="#455A64")
    
    # Data Callouts
    ax.annotate("C1–C4 High Efficiency\n61.2%–66.5% TDP", xy=(2, 61.2), xytext=(1.4, 45),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_SLA, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor=COLOR_R9700S_SLA),
                fontweight='bold', color="#E65100", fontsize=9.0)
    
    ax.annotate("C16 Queue Scaling\n86.6%–88.4% TDP", xy=(16, 88.4), xytext=(9.5, 94),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_PEAK, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor=COLOR_R9700S_PEAK),
                fontweight='bold', color=COLOR_R9700S_PEAK, fontsize=9.0)
    
    ax.annotate("MI350P: 66.7%–68.3% TDP\n(400W–410W of 600W Cap)", xy=(8, 68.3), xytext=(5.5, 75),
                arrowprops=dict(arrowstyle="->", color=COLOR_MI350P_PROD, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8EAF6", edgecolor=COLOR_MI350P_PROD),
                fontweight='bold', color=COLOR_MI350P_PROD, fontsize=9.0)
    
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32"], fontsize=12, fontweight='bold')
    ax.set_xlabel("Concurrency per GPU (C)", fontsize=13, fontweight='bold', labelpad=8)
    ax.set_ylabel("Power Utilization (% of Device Max TDP)", fontsize=12.5, fontweight='bold', labelpad=8)
    ax.set_title("Power Utilization vs Concurrency per GPU (% of Device Max TDP)", fontsize=15, fontweight='bold', pad=15)
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
    c_r9700_8k = [1, 2, 4, 8, 16]
    bw_r9700_8k = [76.3, 75.4, 71.8, 75.6, 71.1]
    
    c_r9700_1k = [1, 2, 4, 8, 16]
    bw_r9700_1k = [75.9, 74.8, 74.3, 74.3, 74.2]
    
    c_r9700_long = [1, 2, 4]
    bw_r9700_long = [73.6, 74.8, 74.9]
    
    # MI350P: Peak = 4,096 GB/s (HBM3E)
    # Measured weight traffic: 1,530 GB/s @ C1 (37.4%), 1,340 GB/s @ C8 (32.7%)
    c_mi350 = [1, 8]
    bw_mi350 = [37.4, 32.7]
    
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
    ax.plot(c_mi350, bw_mi350, 'D-', color=COLOR_MI350P_PROD, lw=2.6, ms=8, label="Instinct MI350P (CDNA 4, 4,096 GB/s Max HBM3E)")
    
    # Data Callouts
    ax.annotate("Near-Optimal GDDR6 Saturation\n~470–488 GB/s (74%–76%)", xy=(1, 75.9), xytext=(1.3, 86.5),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_PEAK, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor=COLOR_R9700S_PEAK),
                fontweight='bold', color=COLOR_R9700S_PEAK, fontsize=9.0)
    
    ax.annotate("MI350P HBM3E Headroom\n33%–37% Utilized @ C1–C8\n(Massive Headroom for C32+)", xy=(8, 32.7), xytext=(4.0, 18),
                arrowprops=dict(arrowstyle="->", color=COLOR_MI350P_PROD, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8EAF6", edgecolor=COLOR_MI350P_PROD),
                fontweight='bold', color=COLOR_MI350P_PROD, fontsize=9.0)
    
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32"], fontsize=12, fontweight='bold')
    ax.set_xlabel("Concurrency per GPU (C)", fontsize=13, fontweight='bold', labelpad=8)
    ax.set_ylabel("Memory Bandwidth Utilization (% of Device Max Peak)  [Higher is better]", fontsize=12.0, fontweight='bold', labelpad=8)
    ax.set_title("Memory Bandwidth Utilization vs Concurrency per GPU (% of Device Max)", fontsize=15, fontweight='bold', pad=15)
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
    c_r9700_8k = [1, 2, 4, 8, 16]
    pwr_r9700_8k = [75.5, 62.0, 66.5, 79.5, 88.4]
    c_r9700_1k = [1, 2, 4, 8, 16]
    pwr_r9700_1k = [75.6, 61.2, 62.6, 76.7, 86.6]
    c_r9700_long = [1, 2, 4]
    pwr_r9700_long = [65.1, 65.1, 65.1]
    c_mi350 = [1, 8]
    pwr_mi350 = [66.7, 68.3]
    c_r9600 = [1, 2, 4, 8, 16]
    pwr_r9600 = [100.0, 100.0, 100.0, 100.0, 100.0]
    
    ax1.plot(c_r9700_8k, pwr_r9700_8k, 'o-', color=COLOR_R9700S_PEAK, lw=2.4, ms=6, label="R9700S (8k:1k, 300W)")
    ax1.plot(c_r9700_1k, pwr_r9700_1k, 's--', color=COLOR_R9700S_SLA, lw=2.0, ms=6, label="R9700S (1k:1k, 300W)")
    ax1.plot(c_r9700_long, pwr_r9700_long, '^-.', color="#C2185B", lw=2.0, ms=6, label="R9700S (1k:8k, 300W)")
    ax1.plot(c_mi350, pwr_mi350, 'D-', color=COLOR_MI350P_PROD, lw=2.4, ms=7, label="MI350P (CDNA 4, 600W)")
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
    bw_r9700_8k = [76.3, 75.4, 71.8, 75.6, 71.1]
    bw_r9700_1k = [75.9, 74.8, 74.3, 74.3, 74.2]
    bw_r9700_long = [73.6, 74.8, 74.9]
    bw_mi350 = [37.4, 32.7]
    c_r9600_bw = [1, 2, 4]
    bw_r9600 = [53.8, 54.5, 55.2]
    
    ax2.axhspan(70.0, 78.0, color='#FFEBEE', alpha=0.5, label='_nolegend_')
    ax2.plot(c_r9700_8k, bw_r9700_8k, 'o-', color=COLOR_R9700S_PEAK, lw=2.4, ms=6, label="R9700S (8k:1k, 640 GB/s)")
    ax2.plot(c_r9700_1k, bw_r9700_1k, 's--', color=COLOR_R9700S_SLA, lw=2.0, ms=6, label="R9700S (1k:1k, 640 GB/s)")
    ax2.plot(c_r9700_long, bw_r9700_long, '^-.', color="#C2185B", lw=2.0, ms=6, label="R9700S (1k:8k, 640 GB/s)")
    ax2.plot(c_r9600_bw, bw_r9600, 'v:', color=COLOR_R9600D_PEAK, lw=1.8, ms=5, label="R9600D (150W Cap, 640 GB/s)")
    ax2.plot(c_mi350, bw_mi350, 'D-', color=COLOR_MI350P_PROD, lw=2.4, ms=7, label="MI350P (CDNA 4, 4,096 GB/s)")
    ax2.set_xscale('log', base=2)
    ax2.set_xticks([1, 2, 4, 8, 16, 32])
    ax2.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32"], fontweight='bold')
    ax2.set_xlabel("Concurrency per GPU (C)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Memory Bandwidth Utilization (% of Device Max Peak)  [Higher is better]", fontsize=11, fontweight='bold')
    ax2.set_title("B. Memory Bandwidth Utilization (% of Peak) [Higher is better]", fontsize=12.5, fontweight='bold')
    ax2.set_ylim(10, 100)
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    ax2.legend(loc='lower left', frameon=True, fontsize=8.5)
    
    fig.suptitle("Hardware Utilization Profiles: Power & Memory Bandwidth vs Concurrency per GPU", fontsize=15, fontweight='bold', y=0.98)
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

    c_mi350_1k = [1, 8]
    ttft_mi350_1k = [100.7, 245.0]

    c_mi350_8k = [1]
    ttft_mi350_8k = [925.0]

    # Shaded interactive zone (< 1,000 ms)
    ax.axhspan(80, 1000, color="#E8F5E9", alpha=0.55, label="_nolegend_")
    ax.axhline(1000, color="#4CAF50", linestyle="--", linewidth=1.5, alpha=0.85)
    ax.text(2.2, 1120, "1.0s Sub-Second Interactive Threshold", fontsize=9.5, fontweight="bold", color="#2E7D32")

    # Plot curves
    ax.plot(c_8k1k, ttft_8k1k, "o-", color=COLOR_R9700S_PEAK, lw=2.6, ms=7, label="Radeon AI PRO R9700S (8,192 In / 1,024 Out)")
    ax.plot(c_1k1k, ttft_1k1k, "s--", color=COLOR_R9700S_SLA, lw=2.2, ms=7, label="Radeon AI PRO R9700S (1,024 In / 1,024 Out)")
    ax.plot(c_1k8k, ttft_1k8k, "^-.", color="#C2185B", lw=2.2, ms=7, label="Radeon AI PRO R9700S (1,024 In / 8,192 Out)")
    ax.plot(c_mi350_1k, ttft_mi350_1k, "D-", color=COLOR_MI350P_PROD, lw=2.4, ms=7, label="Instinct MI350P (1,024 In Reference)")
    ax.plot(c_mi350_8k, ttft_mi350_8k, "P", color=COLOR_MI350P_DF3, ms=9, label="Instinct MI350P (8,192 In Reference)")

    # Annotations
    ax.annotate("C1–C4 Sub-Second Ingestion\n165–937 ms (1k Input)", xy=(2, 194.9), xytext=(1.5, 420),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_SLA, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor=COLOR_R9700S_SLA),
                fontweight="bold", color="#E65100", fontsize=9.0)

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
    ax.set_title("Time-to-First-Token (TTFT p50) Latency vs Concurrency per GPU [Lower is better]", fontsize=15, fontweight="bold", pad=15)
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
# PLOT 10: SLO-Qualified Interactive Goodput vs Raw Saturated Throughput & Cost
# ==============================================================================
def plot_slo_qualified_goodput():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.5), dpi=300)
    
    # Interactive SLA criteria: TTFT <= 3,500 ms AND TPOT <= 50 ms (1,024 in / 1,024 out)
    c_r9700 = [1, 2, 4, 8, 16]
    raw_tok_r9700 = [266, 516, 976, 976, 977]
    goodput_r9700 = [266, 516, 976, 976, 977] # Capacity under max-num-seqs 4
    
    c_mi350 = [1, 4, 8, 16, 32]
    raw_tok_mi350 = [635, 2342, 4437, 7229, 12035]
    goodput_mi350 = [635, 2342, 4437, 7229, 12035] # All TTFT <= 1250ms, TPOT <= 22ms
    
    cost_r9700 = [2.10, 1.08, 0.57, 0.57, 0.57]
    cost_mi350 = [3.60, 0.98, 0.52, 0.32, 0.19]
    
    # --------------------------------------------------------------------------
    # Panel 1: Throughput & Interactive Capacity
    # --------------------------------------------------------------------------
    ax1.plot(c_r9700, raw_tok_r9700, 'o-', color=COLOR_R9700S_PEAK, lw=2.8, ms=8, 
             label="8× R9700S (Interactive Capacity, Capped @ C4)")
    ax1.plot(c_mi350, raw_tok_mi350, 'D-', color=COLOR_MI350P_PROD, lw=2.8, ms=8, 
             label="8× MI350P (Monotonic Scaling through C32)")
    
    # Shaded zones on Panel 1
    ax1.axvspan(0.8, 4.5, color='#E8F5E9', alpha=0.45, label='_nolegend_')
    ax1.text(2.0, 7500, "R9700S INTERACTIVE SWEET SPOT\n(Sub-Second TTFT, 100% SLA)", 
             ha='center', fontsize=8.5, fontweight='bold', color="#2E7D32",
             bbox=dict(boxstyle="square,pad=0.25", facecolor='white', alpha=0.9, edgecolor="#A5D6A7"))
    
    ax1.axvspan(6.0, 36, color='#E8EAF6', alpha=0.35, label='_nolegend_')
    ax1.text(18, 2500, "MI350P HIGH-CONCURRENCY DOMAIN\n(100% SLA Maintained through C32)", 
             ha='center', fontsize=8.5, fontweight='bold', color="#1A237E",
             bbox=dict(boxstyle="square,pad=0.25", facecolor='white', alpha=0.9, edgecolor="#9FA8DA"))
    
    # Annotations on Panel 1
    ax1.annotate("R9700S Interactive Saturation\n976 tok/s @ C4 (TTFT 809ms)", xy=(4, 976), xytext=(1.8, 2200),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_PEAK, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor=COLOR_R9700S_PEAK),
                fontweight='bold', color=COLOR_R9700S_PEAK, fontsize=8.5)
    
    ax1.annotate("Queueing under max-num-seqs 4\nProtects TPOT (<33ms), but\nTTFT breaches 3.5s SLA (17–50s)", xy=(8, 976), xytext=(5.5, 4500),
                arrowprops=dict(arrowstyle="->", color=COLOR_R9700S_SLA, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor=COLOR_R9700S_SLA),
                fontweight='bold', color="#E65100", fontsize=8.5)
    
    ax1.annotate("12,035 tok/s @ C32\nTTFT 1.25s, TPOT 21ms\n(100% Interactive SLA)", xy=(32, 12035), xytext=(12, 11000),
                arrowprops=dict(arrowstyle="->", color=COLOR_MI350P_PROD, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8EAF6", edgecolor=COLOR_MI350P_PROD),
                fontweight='bold', color=COLOR_MI350P_PROD, fontsize=8.5)
    
    ax1.set_xscale('log', base=2)
    ax1.set_xticks([1, 2, 4, 8, 16, 32])
    ax1.set_xticklabels(["C1", "C2", "C4", "C8", "C16", "C32"], fontweight='bold')
    ax1.set_xlabel("Concurrency per GPU (C)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Interactive Server Throughput (tok/s)  [Higher is better]", fontsize=11, fontweight='bold')
    ax1.set_title("A. Interactive Serving Throughput (SLA: TTFT ≤ 3.5s, TPOT ≤ 50ms)", fontsize=12.5, fontweight='bold')
    ax1.set_ylim(0, 13500)
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left', frameon=True, fontsize=9.0)
    
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
    
    fig.suptitle("SLO-Qualified Interactive Goodput vs High-Concurrency Batch Saturation (1,024 In / 1,024 Out)", 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    output_path = os.path.join(OUTPUT_DIR, "10_slo_qualified_goodput.png")
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
    p10 = plot_slo_qualified_goodput()
    
    # Also copy to artifact directory for presentation / embedding
    all_plots = [p1, p2, p3, p3b, p4, p5, p6, p7, p8, p9, p10]
    for p in all_plots:
        dest = os.path.join(ARTIFACT_DIR, os.path.basename(p))
        shutil.copy(p, dest)
        print(f"Generated: {p} -> {dest}")


