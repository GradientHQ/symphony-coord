#!/usr/bin/env python3
"""
Run select_only_stats -> generate phase_stats.json / plan_weight_sum_overall.json,
then run plot_from_json -> generate Cold Start vs Overall radar charts,
and copy to specified pretrain directory.

Usage:
  python run_plan_subtask_radar.py

  Will run for both gsm8k (llama) and bbh (deepseek-v3), outputs to:
    - pretrain_results/2026-01-24_17-14-56_gsm8k_llama-3.1-70b-instruct_topL3_plan3_n600/
    - pretrain_results/2026-01-24_01-18-05_bbh_deepseek-v3_topL3_plan1_n600/

  Radar chart: radar_combined_cold_start_vs_overall.png

Dependencies: select_only_stats.py, plot_from_json.py
Task pool: experiments/configs/balanced_task_pool_1_2001.jsonl (400 gsm8k, 400 bbh)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
_SCRIPTS_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))

os.chdir(_PROJECT_ROOT)

# Two pretrain directories (to write radar charts to)
GSM8K_PRETRAIN_DIR = os.path.join(
    _PROJECT_ROOT,
    "pretrain_results/2026-01-24_17-14-56_gsm8k_llama-3.1-70b-instruct_topL3_plan3_n600",
)
BBH_PRETRAIN_DIR = os.path.join(
    _PROJECT_ROOT,
    "pretrain_results/2026-01-24_01-18-05_bbh_deepseek-v3_topL3_plan1_n600",
)

TASK_POOL = os.path.join(_PROJECT_ROOT, "experiments/configs/balanced_task_pool_1_2001.jsonl")
RUNTIME_DIR = os.path.join(_PROJECT_ROOT, "experiments/configs/openrouter")

# n=400 (pool only 400 gsm8k / 400 bbh), split 100 / 200 / 100
COLD_N, PRETRAIN_N, TEST_N = 100, 200, 100
N = COLD_N + PRETRAIN_N + TEST_N
SEED = 42
TOP_L = 3
PLAN_K = 3
COT_COUNT = 1

CONFIGS = [
    {
        "name": "gsm8k_llama",
        "benchmark": "gsm8k",
        "agents": "66,67,68,69,70",
        "outdir_base": os.path.join(_PROJECT_ROOT, "plan_subtask_radar_runs/gsm8k_llama"),
        "pretrain_dir": GSM8K_PRETRAIN_DIR,
    },
    {
        "name": "bbh_deepseek",
        "benchmark": "bbh",
        "agents": "16,17,18,19,20",
        "outdir_base": os.path.join(_PROJECT_ROOT, "plan_subtask_radar_runs/bbh_deepseek"),
        "pretrain_dir": BBH_PRETRAIN_DIR,
    },
]


def run_select_only(cfg: dict) -> str | None:
    """Run select_only_stats, return final outdir (with timestamp subdirectory)."""
    os.makedirs(cfg["outdir_base"], exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(_THIS_DIR, "select_only_stats.py"),
        "--task-pool", TASK_POOL,
        "--benchmark", cfg["benchmark"],
        "--n", str(N),
        "--cold-n", str(COLD_N),
        "--pretrain-n", str(PRETRAIN_N),
        "--test-n", str(TEST_N),
        "--seed", str(SEED),
        "--runtime-dir", RUNTIME_DIR,
        "--agents", cfg["agents"],
        "--topL", str(TOP_L),
        "--plan-k", str(PLAN_K),
        "--cot-count", str(COT_COUNT),
        "--test-only",
        "--outdir", cfg["outdir_base"],
    ]
    print(f"[run_plan_subtask_radar] Running select_only_stats for {cfg['name']} ...")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    if r.returncode != 0:
        print(r.stderr)
        return None
    # Parse "Output directory: ..." or "outdir=..."
    for line in (r.stdout + "\n" + r.stderr).splitlines():
        if "[INFO] Output directory:" in line or "Output directory:" in line:
            m = re.search(r"Output directory:\s*(.+)", line)
            if m:
                return m.group(1).strip()
        if "outdir=" in line and "=" in line:
            return line.strip().split("outdir=", 1)[1].strip().split()[0].rstrip(",")
    # Fallback: get newest timestamp subdirectory in outdir_base
    subdirs = []
    for e in os.listdir(cfg["outdir_base"]):
        p = os.path.join(cfg["outdir_base"], e)
        if os.path.isdir(p) and re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", e):
            subdirs.append((os.path.getmtime(p), p))
    if subdirs:
        subdirs.sort(reverse=True)
        return subdirs[0][1]
    return None


def run_plot_from_json(result_dir: str) -> bool:
    """Run plot_from_json on result_dir."""
    plot_script = os.path.join(_SCRIPTS_DIR, "plotting/routing/plot_from_json.py")
    cmd = [sys.executable, plot_script, result_dir]
    print(f"[run_plan_subtask_radar] Running plot_from_json on {result_dir} ...")
    r = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    return r.returncode == 0


def copy_radar_to_pretrain(result_dir: str, pretrain_dir: str, name: str) -> None:
    """Copy radar_combined_cold_start_vs_overall.png to pretrain directory."""
    src = os.path.join(result_dir, "radar_combined_cold_start_vs_overall.png")
    if not os.path.isfile(src):
        print(f"[run_plan_subtask_radar] Not found {src}, skipping copy")
        return
    os.makedirs(pretrain_dir, exist_ok=True)
    dst = os.path.join(pretrain_dir, "radar_combined_cold_start_vs_overall.png")
    shutil.copy2(src, dst)
    print(f"[run_plan_subtask_radar] Copied -> {dst} ({name})")


def main() -> None:
    for cfg in CONFIGS:
        outdir = run_select_only(cfg)
        if not outdir:
            print(f"[run_plan_subtask_radar] select_only failed: {cfg['name']}")
            continue
        if not run_plot_from_json(outdir):
            print(f"[run_plan_subtask_radar] plot_from_json failed: {outdir}")
            continue
        copy_radar_to_pretrain(outdir, cfg["pretrain_dir"], cfg["name"])
    print("[run_plan_subtask_radar] done.")


if __name__ == "__main__":
    main()
