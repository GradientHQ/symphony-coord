#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot Exp2: Simulation vs Real execution (same figure)

Input:
- sim trajectory CSV (linucb)
- real trajectory JSON
"""

import argparse
import os
import json
import csv
from typing import List
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def rolling_success(xs: List[int], window: int = 50) -> List[float]:
    out = []
    buf = []
    for x in xs:
        buf.append(x)
        if len(buf) > window:
            buf.pop(0)
        out.append(sum(buf) / len(buf))
    return out


# -----------------------------
# Loaders
# -----------------------------
def load_sim_trajectory(path: str):
    ts, rs = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts.append(int(row["t"]))
            rs.append(float(row["rolling_success"]))
    return ts, rs


def load_real_trajectory(path: str):
    ts, succ = [], []
    with open(path) as f:
        data = json.load(f)
        for row in data:
            ts.append(int(row["t"]))
            succ.append(1 if row["success"] else 0)
    rs = rolling_success(succ)
    return ts, rs


# -----------------------------
# Plot
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Plot Exp2: sim vs real")
    ap.add_argument("--sim", required=True, help="trajectory_linucb.csv")
    ap.add_argument("--real", required=True, help="trajectory_real.json")
    ap.add_argument("--shock-point-sim", type=int, help="Shock point for sim (if different from real)")
    ap.add_argument("--shock-point-real", type=int, help="Shock point for real (if different from sim)")
    ap.add_argument("--shock-point", type=int, help="Shock point (if same for both)")
    ap.add_argument("--shock-type", type=str, help="Shock type (A_unavailable or A_degraded) for title")
    ap.add_argument("--out", default="sim_vs_real.png")
    args = ap.parse_args()

    ts_sim, rs_sim = load_sim_trajectory(args.sim)
    ts_real, rs_real = load_real_trajectory(args.real)

    # Determine shock points
    shock_sim = args.shock_point_sim or args.shock_point
    shock_real = args.shock_point_real or args.shock_point
    
    if shock_sim is None or shock_real is None:
        raise ValueError("Must specify --shock-point or both --shock-point-sim and --shock-point-real")

    # Figure structure: Two subplots (shared y-axis) to compare response patterns
    # This emphasizes "behavioral consistency" rather than "numerical comparison"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=180, sharex=True, sharey=True)
    
    # ========== Upper subplot: Simulation ==========
    shock_window = 50  # Transition window width (aligns with rolling window size)
    
    ax1.plot(ts_sim, rs_sim, label="LinUCB", linewidth=2.5, 
             color='#3498db', alpha=0.9)
    
    # Shock transition window: gray band to indicate transition period (paper-grade)
    # This addresses potential reviewer question: "Is the pre-shock dip caused by smoothing?"
    if shock_sim <= max(ts_sim):
        shock_end = min(shock_sim + shock_window, max(ts_sim))
        ax1.axvspan(shock_sim, shock_end, alpha=0.15, color='gray', 
                   label='Transition window', zorder=0)
        ax1.axvline(
            x=shock_sim,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Shock (t={shock_sim})",
            zorder=5,
        )
    
    # Optional: Post-shock stable regime (horizontal band to indicate stable performance region)
    # Calculate post-shock average from trajectory (after transition window)
    if shock_sim + shock_window < len(rs_sim):
        post_shock_values = rs_sim[shock_sim + shock_window:]
        if post_shock_values:
            post_shock_mean = sum(post_shock_values) / len(post_shock_values)
            post_shock_std = (sum((x - post_shock_mean) ** 2 for x in post_shock_values) / len(post_shock_values)) ** 0.5
            # Show stable regime as horizontal band (mean ± 1 std)
            ax1.axhspan(max(0, post_shock_mean - post_shock_std), 
                       min(1.05, post_shock_mean + post_shock_std),
                       alpha=0.08, color='green', zorder=0)
            # Optional: show mean line (very subtle)
            # ax1.axhline(y=post_shock_mean, color='green', linestyle=':', linewidth=1, alpha=0.3, zorder=1)
    
    ax1.set_ylabel("Rolling Success Rate", fontsize=12, fontweight='bold')
    ax1.set_title("(a) Simulation (LinUCB)", fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlim(0, max(max(ts_sim), max(ts_real)) * 1.02)
    
    # Add annotation for simulation
    ax1.text(0.02, 0.98, 
             f"n={len(ts_sim)}, shock@t={shock_sim}",
             transform=ax1.transAxes,
             fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # ========== Lower subplot: Real Benchmark Replay ==========
    ax2.plot(ts_real, rs_real, label="LinUCB", linewidth=2.5, 
             color='#e74c3c', alpha=0.9)
    
    # Shock transition window: gray band to indicate transition period (paper-grade)
    if shock_real <= max(ts_real):
        shock_end_real = min(shock_real + shock_window, max(ts_real))
        ax2.axvspan(shock_real, shock_end_real, alpha=0.15, color='gray', 
                   label='Transition window', zorder=0)
        ax2.axvline(
            x=shock_real,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Shock (t={shock_real})",
            zorder=5,
        )
    
    # Optional: Post-shock stable regime (horizontal band to indicate stable performance region)
    if shock_real + shock_window < len(rs_real):
        post_shock_values_real = rs_real[shock_real + shock_window:]
        if post_shock_values_real:
            post_shock_mean_real = sum(post_shock_values_real) / len(post_shock_values_real)
            post_shock_std_real = (sum((x - post_shock_mean_real) ** 2 for x in post_shock_values_real) / len(post_shock_values_real)) ** 0.5
            # Show stable regime as horizontal band (mean ± 1 std)
            ax2.axhspan(max(0, post_shock_mean_real - post_shock_std_real), 
                       min(1.05, post_shock_mean_real + post_shock_std_real),
                       alpha=0.08, color='green', zorder=0)
            # Optional: show mean line (very subtle)
            # ax2.axhline(y=post_shock_mean_real, color='green', linestyle=':', linewidth=1, alpha=0.3, zorder=1)
    
    ax2.set_xlabel("Task Index", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Rolling Success Rate", fontsize=12, fontweight='bold')
    ax2.set_title("(b) Real Benchmark Replay (LinUCB)", fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, loc='best', framealpha=0.9)
    ax2.set_ylim(0.0, 1.05)
    
    # Add annotation for real
    ax2.text(0.02, 0.98, 
             f"n={len(ts_real)}, shock@t={shock_real}",
             transform=ax2.transAxes,
             fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ========== Main title (emphasizes consistent patterns, not numerical comparison) ==========
    shock_type_str = args.shock_type if args.shock_type else "Role Shock"
    if shock_type_str == "A_unavailable":
        title = "Consistent Shock Response Patterns under Role Unavailability in Simulation and Real Benchmark Replay"
    elif shock_type_str == "A_degraded":
        title = "Consistent Shock Response Patterns under Role Degradation in Simulation and Real Benchmark Replay"
    else:
        title = f"Consistent Shock Response Patterns under {shock_type_str} in Simulation and Real Benchmark Replay"
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # ========== Caption note (critical: emphasize pattern comparison, not numerical comparison) ==========
    # This addresses potential reviewer confusion about comparing absolute success rates
    caption_note = ("Note: Absolute success rates are not directly comparable across settings due to differences "
                   "in task distributions; the comparison focuses on shock response patterns.")
    fig.text(0.5, 0.01, caption_note, transform=fig.transFigure,
            ha='center', va='bottom', fontsize=10, style='italic', color='gray',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])  # Leave space for suptitle and caption
    plt.savefig(args.out, dpi=180, bbox_inches='tight')
    plt.close()

    print(f"✅ Figure saved to: {args.out}")
    print(f"   Sim: {len(ts_sim)} tasks, shock at t={shock_sim}")
    print(f"   Real: {len(ts_real)} tasks, shock at t={shock_real}")
    print(f"   Title: {title}")


if __name__ == "__main__":
    main()
