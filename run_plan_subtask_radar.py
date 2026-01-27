#!/usr/bin/env python3
"""
跑 select_only_stats → 生成 phase_stats.json / plan_weight_sum_overall.json，
再跑 plot_from_json → 生成 Cold Start vs Overall 雷达图，
并复制到指定 pretrain 目录。

用法:
  python run_plan_subtask_radar.py

  会为 gsm8k (llama) 和 bbh (deepseek-v3) 各跑一遍，输出到:
    - pretrain_results/2026-01-24_17-14-56_gsm8k_llama-3.1-70b-instruct_topL3_plan3_n600/
    - pretrain_results/2026-01-24_01-18-05_bbh_deepseek-v3_topL3_plan1_n600/

  雷达图: radar_combined_cold_start_vs_overall.png

依赖: select_only_stats.py, plot_from_json.py
Task pool: runtime/balanced_task_pool_1_2001.jsonl (400 gsm8k, 400 bbh)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_ROOT)

# 两个 pretrain 目录（要写入雷达图）
GSM8K_PRETRAIN_DIR = os.path.join(
    _ROOT,
    "pretrain_results/2026-01-24_17-14-56_gsm8k_llama-3.1-70b-instruct_topL3_plan3_n600",
)
BBH_PRETRAIN_DIR = os.path.join(
    _ROOT,
    "pretrain_results/2026-01-24_01-18-05_bbh_deepseek-v3_topL3_plan1_n600",
)

TASK_POOL = os.path.join(_ROOT, "runtime/balanced_task_pool_1_2001.jsonl")
RUNTIME_DIR = os.path.join(_ROOT, "runtime/configs/openrouter")

# n=400 (pool 仅 400 gsm8k / 400 bbh), split 100 / 200 / 100
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
        "outdir_base": os.path.join(_ROOT, "plan_subtask_radar_runs/gsm8k_llama"),
        "pretrain_dir": GSM8K_PRETRAIN_DIR,
    },
    {
        "name": "bbh_deepseek",
        "benchmark": "bbh",
        "agents": "16,17,18,19,20",
        "outdir_base": os.path.join(_ROOT, "plan_subtask_radar_runs/bbh_deepseek"),
        "pretrain_dir": BBH_PRETRAIN_DIR,
    },
]


def run_select_only(cfg: dict) -> str | None:
    """跑 select_only_stats，返回最终 outdir（带时间戳的子目录）。"""
    os.makedirs(cfg["outdir_base"], exist_ok=True)
    cmd = [
        sys.executable,
        "select_only_stats.py",
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
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=_ROOT)
    if r.returncode != 0:
        print(r.stderr)
        return None
    # 解析 "Output directory: ..." 或 "outdir=..."
    for line in (r.stdout + "\n" + r.stderr).splitlines():
        if "[INFO] Output directory:" in line or "Output directory:" in line:
            m = re.search(r"Output directory:\s*(.+)", line)
            if m:
                return m.group(1).strip()
        if "outdir=" in line and "=" in line:
            return line.strip().split("outdir=", 1)[1].strip().split()[0].rstrip(",")
    # 回退：取 outdir_base 下最新时间戳子目录
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
    """对 result_dir 跑 plot_from_json。"""
    cmd = [sys.executable, "plot_from_json.py", result_dir]
    print(f"[run_plan_subtask_radar] Running plot_from_json on {result_dir} ...")
    r = subprocess.run(cmd, cwd=_ROOT)
    return r.returncode == 0


def copy_radar_to_pretrain(result_dir: str, pretrain_dir: str, name: str) -> None:
    """将 radar_combined_cold_start_vs_overall.png 复制到 pretrain 目录。"""
    src = os.path.join(result_dir, "radar_combined_cold_start_vs_overall.png")
    if not os.path.isfile(src):
        print(f"[run_plan_subtask_radar] 未找到 {src}，跳过复制")
        return
    os.makedirs(pretrain_dir, exist_ok=True)
    dst = os.path.join(pretrain_dir, "radar_combined_cold_start_vs_overall.png")
    shutil.copy2(src, dst)
    print(f"[run_plan_subtask_radar] 已复制 -> {dst} ({name})")


def main() -> None:
    for cfg in CONFIGS:
        outdir = run_select_only(cfg)
        if not outdir:
            print(f"[run_plan_subtask_radar] select_only 失败: {cfg['name']}")
            continue
        if not run_plot_from_json(outdir):
            print(f"[run_plan_subtask_radar] plot_from_json 失败: {outdir}")
            continue
        copy_radar_to_pretrain(outdir, cfg["pretrain_dir"], cfg["name"])
    print("[run_plan_subtask_radar] done.")


if __name__ == "__main__":
    main()
