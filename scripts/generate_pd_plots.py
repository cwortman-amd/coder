#!/usr/bin/env python3
"""
Generate Publication-Grade 4-Panel Master Dashboard for Prefill/Decode Disaggregation (P/D) vs Data Parallelism (DP=2).

Methodology & Audit Compliance:
- Subtitle: Measured single-R9700 collocation effects and emulated two-R9700 P/D outcomes; dual-R9700 validation pending.
- Visual convention: Solid marks/bars for Hardware Measurements; dashed/hatched marks for Emulator Output.
- All-or-nothing SLO definition: TTFT <= 3,500 ms, p95 ITL <= 100 ms, Peak ITL <= 500 ms.
- Defensible break-even condition: D_PD(C, lambda) > D_DP0(C0, lambda0) + D_DP1(C1, lambda1).
- Injected handoff delays: 5.16 ms (idealized PCIe floor), 25-250 ms (assumed delays, not measured transport).
"""

import os
import shutil
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "/home/amd/workspace/coder/docs/figures/pd"
ARTIFACT_DIR = "/home/amd/.gemini/antigravity-cli/brain/3a344b95-6417-4951-aa06-7d6314d2ecfe"
SUMMARY_JSON = "/home/amd/workspace/coder/_results/pd_emulator/pd_emulator_summary_20260929_042526.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Styling palette
AMD_RED = "#D32F2F"
AMD_DARK_RED = "#B71C1C"
AMD_ORANGE = "#E65100"
AMD_CORAL = "#FF7043"
EMU_BLUE = "#1565C0"
EMU_LIGHT_BLUE = "#42A5F5"
EMU_CYAN = "#00838F"
PASS_GREEN = "#2E7D32"
FAIL_RED = "#C62828"
GRAY_TEXT = "#424242"
BG_GRID = "#E0E0E0"

def generate_master_dashboard():
    # 2x2 layout, 17x12 inches, 300 DPI
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(17.0, 12.0), dpi=300)
    plt.subplots_adjust(top=0.90, bottom=0.11, left=0.07, right=0.94, hspace=0.36, wspace=0.25)
    
    # Suptitle & Subtitle
    fig.suptitle("Prefill/Decode Disaggregation (P/D) vs. Collocated Data Parallelism (DP=2) Analysis — Qwen3.8-27B MXFP4\n"
                 "Measured Single-R9700 Collocation Effects & Emulated Two-R9700 P/D Outcomes (Validation Pending)",
                 fontsize=14.5, fontweight='bold', y=0.965)

    # ----------------------------------------------------
    # PANEL A: Token-Level Jitter & Forward Stalls
    # ----------------------------------------------------
    conditions = [
        "4K Chunk\n(Collocated)",
        "2K Chunk\n(Collocated)",
        "1K Chunk\n(Collocated)",
        "Sustained 8K\n(Collocated)",
        "P/D 1P1D\n(Disaggregated)"
    ]
    p95_itl = [57.09, 62.55, 261.63, 1363.38, 48.16]
    max_itl = [59.52, 613.31, 364.48, 1363.38, 48.16]
    is_emulated = [False, False, False, False, True]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, p95_itl, width, 
                    color=[AMD_CORAL if not emu else "none" for emu in is_emulated],
                    edgecolor=[AMD_DARK_RED if not emu else EMU_BLUE for emu in is_emulated],
                    hatch=['' if not emu else '///' for emu in is_emulated],
                    linewidth=1.8, label="ITL p95 (ms) [Solid=Measured, Dashed=Emulated]")
    
    bars2 = ax1.bar(x + width/2, max_itl, width,
                    color=[AMD_RED if not emu else "none" for emu in is_emulated],
                    edgecolor=[AMD_DARK_RED if not emu else EMU_BLUE for emu in is_emulated],
                    hatch=['' if not emu else '\\\\\\' for emu in is_emulated],
                    linewidth=1.8, label="Peak Max ITL Stall (ms)")
    
    # Value annotations avoiding collision
    for i, (b1, b2, p95, peak, emu) in enumerate(zip(bars1, bars2, p95_itl, max_itl, is_emulated)):
        if abs(p95 - peak) < 1.0:
            ax1.text(x[i], peak * 1.15, f"{peak:.1f} ms", ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                     color=AMD_DARK_RED if not emu else EMU_BLUE)
        else:
            ax1.text(b1.get_x() + b1.get_width()/2, p95 * 1.15, f"{p95:.1f}", ha='center', va='bottom', fontsize=8.0, fontweight='bold', color=AMD_DARK_RED if not emu else EMU_BLUE)
            ax1.text(b2.get_x() + b2.get_width()/2, peak * 1.15, f"{peak:.1f}", ha='center', va='bottom', fontsize=8.0, fontweight='bold', color=AMD_DARK_RED if not emu else EMU_BLUE)
    
    # SLO Thresholds
    ax1.axhline(100, color="#D84315", linestyle="--", linewidth=1.5, label="Generative Streaming SLO (p95 ≤ 100 ms)")
    ax1.axhline(500, color="#B71C1C", linestyle=":", linewidth=1.5, label="Interactive Stall Ceiling (Max ITL ≤ 500 ms)")
    
    ax1.set_yscale('log')
    ax1.set_ylim(20, 3500)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=9.0, fontweight='bold')
    ax1.set_ylabel("Inter-Token Latency (ITL in ms, Log Scale)", fontsize=10.5, fontweight='bold')
    ax1.set_title("A. Measured Single-Card Forward Stalls vs. Emulated P/D Cadence", fontsize=11.5, fontweight='bold', pad=10)
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(loc='upper left', fontsize=8.0, framealpha=0.9)
    
    # Diagnostic Callout
    ax1.text(0.03, 0.44, 
             "• Chunk 4K: Stalls bounded to 59.5 ms (0% >100ms violations)\n"
             "• Chunk 2K: Periodic prefill causes 613 ms stalls (3.5% >100ms)\n"
             "• Sustained Ingestion: Decode freezes for 1,363 ms (100% fail)\n"
             "• P/D 1P1D: Bounded at 48.2 ms [Emulated — 2-GPU validation pending]",
             transform=ax1.transAxes, fontsize=7.8,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4", edgecolor="#FBC02D", alpha=0.95))

    # ----------------------------------------------------
    # PANEL B: Usable Capacity: Raw Throughput vs All-or-Nothing SLO Goodput
    # ----------------------------------------------------
    workloads = ["Light (J1)", "Moderate (J2)", "Saturated", "J3 Sustained"]
    
    dp_raw = [24.11, 49.53, 22.85, 22.98]
    dp_slo = [0.19, 0.26, 0.04, 0.28]
    
    pd_raw = [24.11, 41.35, 41.53, 29.55]
    pd_slo = [24.11, 41.29, 41.15, 29.52]
    
    xw = np.arange(len(workloads))
    w_b = 0.20
    
    b_dp_raw = ax2.bar(xw - 1.5*w_b, dp_raw, w_b, color="#9E9E9E", edgecolor="#424242", linewidth=1.5, label="DP=2 Raw tok/s [Modeled baseline]")
    b_dp_slo = ax2.bar(xw - 0.5*w_b, dp_slo, w_b, color=FAIL_RED, edgecolor="#8E0000", linewidth=1.5, label="DP=2 Qualified Goodput (All-or-Nothing)")
    b_pd_raw = ax2.bar(xw + 0.5*w_b, pd_raw, w_b, color="none", edgecolor=EMU_BLUE, hatch="///", linewidth=1.5, label="P/D 1P1D Raw tok/s [Emulated]")
    b_pd_slo = ax2.bar(xw + 1.5*w_b, pd_slo, w_b, color="none", edgecolor=PASS_GREEN, hatch="\\\\\\", linewidth=1.8, label="P/D 1P1D Qualified Goodput [Emulated]")
    
    for i in range(len(workloads)):
        ax2.text(xw[i] - 1.5*w_b, dp_raw[i] + 1.0, f"{dp_raw[i]:.1f}", ha='center', va='bottom', fontsize=7.5, fontweight='bold', color="#424242")
        ax2.text(xw[i] - 0.5*w_b, dp_slo[i] + 1.0, f"{dp_slo[i]:.2f}", ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=FAIL_RED)
        ax2.text(xw[i] + 0.5*w_b, pd_raw[i] + 1.0, f"{pd_raw[i]:.1f}", ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=EMU_BLUE)
        ax2.text(xw[i] + 1.5*w_b, pd_slo[i] + 1.0, f"{pd_slo[i]:.1f}", ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=PASS_GREEN)
        
    ax2.set_xticks(xw)
    ax2.set_xticklabels(workloads, fontsize=9.5, fontweight='bold')
    ax2.set_ylabel("Output Throughput (Output tok/s)", fontsize=10.5, fontweight='bold')
    ax2.set_title("B. Raw Output vs. SLO-Qualified Goodput (All-or-Nothing Criteria)", fontsize=11.5, fontweight='bold', pad=10)
    ax2.set_ylim(0, 68)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc='upper right', fontsize=8.0, framealpha=0.9)
    
    # Note on all-or-nothing placed cleanly
    ax2.text(0.03, 0.72,
             "All-or-Nothing Goodput: G_output = (sum q_i * O_i) / T\n"
             "• Any request failing TTFT > 3.5s or Peak ITL > 500ms counts 0 qualified tokens.\n"
             "• DP=2 produces raw output, but streaming sessions fail max-ITL ceiling.\n"
             "• P/D goodput advantage reflects stall isolation, not zero DP raw output.",
             transform=ax2.transAxes, fontsize=7.5,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#E8F5E9", edgecolor="#81C784", alpha=0.95))

    # ----------------------------------------------------
    # PANEL C: Sustainable Capacity & Break-Even Condition
    # ----------------------------------------------------
    lam = np.array([0.1, 0.2, 0.33, 0.5, 0.75, 1.0, 1.2])
    dp_completed = np.array([32.0, 48.0, 49.5, 36.5, 26.2, 22.8, 21.5])
    pd_completed = np.array([24.1, 36.0, 41.3, 41.4, 41.5, 41.5, 38.5])
    
    ax3.plot(lam, dp_completed, marker='s', markersize=6, color=AMD_RED, linewidth=2.2,
             label="Collocated DP=2 [Modeled from Single-Card Phase Data]")
    ax3.plot(lam, pd_completed, marker='D', markersize=6, color=EMU_BLUE, linestyle='--', linewidth=2.2,
             label="P/D 1P1D Disaggregated [Emulated Two-Card Projection]")
    
    # Break-even crossover shading
    ax3.axvspan(0.42, 0.58, color="#FFF9C4", alpha=0.5, label="Break-Even Zone (D_PD > D_DP0 + D_DP1)")
    ax3.axvline(0.50, color="#F57F17", linestyle=":", linewidth=1.5)
    ax3.text(0.51, 46.5, "Break-Even Condition:\nη_DP < 0.50 (D_collocated < 17.0 tok/s)", 
             fontsize=7.8, color="#E65100", fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF8E1", edgecolor="#FFE082"))
    
    # Queue growth onset line
    ax3.axvline(0.72, color="#7B1FA2", linestyle="-.", linewidth=1.5, label="Prefill Saturation Onset (Queue Growth)")
    ax3.text(0.74, 53.5, "Prefill Queue Growth Onset\n(λ > sustainable prefill)", 
             fontsize=7.8, color="#7B1FA2", fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#F3E5F5", edgecolor="#CE93D8"))
    
    ax3.set_xlabel("Offered Prompt Arrival Rate λ (req/s)", fontsize=10.5, fontweight='bold', labelpad=6)
    ax3.set_ylabel("Completed Output Throughput (tok/s)", fontsize=10.5, fontweight='bold')
    ax3.set_title("C. Sustainable Capacity & Break-Even Condition vs. Offered Load", fontsize=11.5, fontweight='bold', pad=10)
    ax3.set_xlim(0.05, 1.25)
    ax3.set_ylim(10, 60)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc='lower left', fontsize=8.0, framealpha=0.9)

    # ----------------------------------------------------
    # PANEL D: Injected KV Handoff Delay Sensitivity
    # ----------------------------------------------------
    handoff_labels = ["5.16 ms\n(Ideal PCIe)", "25.0 ms\n(IPC Delay)", "50.0 ms\n(Net / UCX)", "100.0 ms\n(Host Bounce)", "250.0 ms\n(Slow Queue)"]
    handoff_vals = [5.16, 25.0, 50.0, 100.0, 250.0]
    
    goodput_sat = [41.15, 41.13, 41.11, 41.07, 40.95]
    goodput_j3 = [29.52, 29.51, 29.50, 29.43, 14.72]
    ttft_p95_j3 = [3452.7, 3472.5, 3497.5, 3547.5, 3697.5]
    
    xd = np.arange(len(handoff_vals))
    width_d = 0.28
    
    ax4_twin = ax4.twinx()
    
    b1_d = ax4.bar(xd - width_d/2, goodput_sat, width_d, color="none", edgecolor=EMU_CYAN, hatch="//", linewidth=1.5, label="Goodput: Saturated Trace [Emulated]")
    b2_d = ax4.bar(xd + width_d/2, goodput_j3, width_d, color="none", edgecolor=PASS_GREEN, hatch="\\\\", linewidth=1.5, label="Goodput: J3 Sustained Trace [Emulated]")
    
    l_ttft = ax4_twin.plot(xd, ttft_p95_j3, color="#D32F2F", marker='o', linewidth=2.0, linestyle="--", label="TTFT p95 (J3 Sustained) [Right Axis]")
    ax4_twin.axhline(3500, color="#B71C1C", linestyle=":", linewidth=1.5, label="TTFT 3.5s SLA Boundary")
    
    ax4.set_xticks(xd)
    ax4.set_xticklabels(handoff_labels, fontsize=8.8, fontweight='bold')
    ax4.set_xlabel("Assumed Injected Handoff Delay H (Model Parameter, Not Measured Transport)", fontsize=10.0, fontweight='bold', labelpad=6)
    ax4.set_ylabel("SLO-Qualified Goodput (tok/s)", fontsize=10.5, fontweight='bold')
    ax4_twin.set_ylabel("TTFT p95 Latency (ms)", fontsize=10.5, fontweight='bold', color="#D32F2F")
    ax4_twin.tick_params(axis='y', labelcolor="#D32F2F")
    ax4_twin.set_ylim(3200, 3950)
    ax4.set_ylim(0, 55)
    
    ax4.set_title("D. Assumed Injected Handoff Delay Sensitivity & TTFT Inflation", fontsize=11.5, fontweight='bold', pad=10)
    ax4.grid(True, linestyle="--", alpha=0.5)
    
    lines_4, labels_4 = ax4.get_legend_handles_labels()
    lines_4t, labels_4t = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines_4 + lines_4t, labels_4 + labels_4t, loc='lower left', fontsize=7.8, framealpha=0.9)
    
    ax4.annotate("TTFT breaches 3.5s SLA\n(Goodput drops to 14.7 tok/s)",
                 xy=(4, 15.5), xytext=(2.3, 26),
                 arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.5),
                 fontsize=7.8, fontweight='bold', color="#B71C1C",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor="#EF9A9A"))

    # ----------------------------------------------------
    # FOOTER METADATA
    # ----------------------------------------------------
    footer_text = (
        "Workload Shape: Qwen3.8-27B MXFP4 (8,192 In / 1,024 Out) | Hardware: AMD Radeon™ AI PRO R9700 (32GB GDDR6, gfx1201) | Stack: ROCm / vLLM MXFP4 (Chunk 2048, FP8 KV)\n"
        "SLO Criteria: TTFT ≤ 3,500 ms ∧ p95 ITL ≤ 100 ms ∧ Peak ITL ≤ 500 ms | Solid Marks = Single-Card Physical Measurements | Dashed/Hatched Marks = Emulated Two-Card P/D Projections"
    )
    fig.text(0.5, 0.025, footer_text, ha='center', fontsize=8.2, style='italic', color=GRAY_TEXT,
             bbox=dict(boxstyle="square,pad=0.4", facecolor="#F5F5F5", edgecolor="#CCCCCC", alpha=0.9))

    # Save
    out_file = os.path.join(OUTPUT_DIR, "01_pdd_benefits_master_dashboard.png")
    plt.savefig(out_file, dpi=300)
    print(f"Generated: {out_file}")
    
    artifact_file = os.path.join(ARTIFACT_DIR, "01_pdd_benefits_master_dashboard.png")
    shutil.copy(out_file, artifact_file)
    print(f"Copied to: {artifact_file}")

if __name__ == "__main__":
    generate_master_dashboard()
