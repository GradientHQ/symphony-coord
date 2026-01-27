#!/usr/bin/env python3
"""
从 pretrain 目录已有的 accuracy_summary.csv 聚合出 phase_stats.json 与
plan_weight_sum_overall.json，无需 task pool、不调 API。

依据：每行 (phase, node_id) 表示该任务选中的 agent；按 phase 汇总 plan_pick_count
与 plan_weight_sum（权重用 1.0）。无 trace 故无真实 per-step 的 subtask 数据；
采用 plan = subtask 代理：subtask 分布与 plan 相同，便于雷达图两条线都可见。
subtask_total_steps / subtask_selected_steps 用 CSV 的 subtask_count 之和。

用法:
  python stats_from_pretrain_dir.py <pretrain_dir_1> [<pretrain_dir_2> ...]

示例:
  python stats_from_pretrain_dir.py \\
    pretrain_results/2026-01-24_17-14-56_gsm8k_llama-3.1-70b-instruct_topL3_plan3_n600 \\
    pretrain_results/2026-01-24_01-18-05_bbh_deepseek-v3_topL3_plan1_n600

然后对同一目录跑 plot_from_json 即可出雷达图。
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple


def _load_summary(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _agent_from_node_id(node_id: str) -> str:
    s = (node_id or "").strip()
    if not s:
        return ""
    return s.split(",")[0].strip()


def _collect_agents(rows: List[Dict[str, Any]]) -> List[str]:
    seen: Set[str] = set()
    for r in rows:
        a = _agent_from_node_id(r.get("node_id") or "")
        if a:
            seen.add(a)
    return sorted(seen)


def _int_subtask_count(r: Dict[str, Any]) -> int:
    v = r.get("subtask_count", 0)
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _build_phase_stats(rows: List[Dict[str, Any]], agent_keys: List[str]) -> Dict[str, Dict[str, Any]]:
    phase_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        phase = (r.get("phase") or "").strip().lower()
        if not phase:
            continue
        phase_rows[phase].append(r)

    all_phase_stats: Dict[str, Dict[str, Any]] = {}
    for phase_name, prs in phase_rows.items():
        plan_pick_count: Dict[str, int] = {k: 0 for k in agent_keys}
        plan_weight_sum: Dict[str, float] = {k: 0.0 for k in agent_keys}
        subtask_steps_total = 0
        for r in prs:
            agent = _agent_from_node_id(r.get("node_id") or "")
            if agent not in agent_keys:
                continue
            plan_pick_count[agent] += 1
            plan_weight_sum[agent] += 1.0
            n = _int_subtask_count(r)
            subtask_steps_total += max(n, 1)  # 至少 1 step（无 subtask 时整题算 1 step）

        plan_count = sum(plan_pick_count.values())
        # plan = subtask 代理：同分布，subtask 曲线可见
        subtask_pick_count = dict(plan_pick_count)
        subtask_weight_sum = {k: float(v) for k, v in plan_weight_sum.items()}
        st_total = subtask_steps_total
        st_selected = subtask_steps_total

        all_phase_stats[phase_name] = {
            "phase": phase_name,
            "plan_weight_sum": plan_weight_sum,
            "plan_pick_count": plan_pick_count,
            "plan_count": plan_count,
            "plan_parse_fail": 0,
            "plan_unknown_agents": {},
            "plan_unknown_count": {},
            "subtask_pick_count": subtask_pick_count,
            "subtask_weight_sum": subtask_weight_sum,
            "subtask_total_steps": st_total,
            "subtask_selected_steps": st_selected,
            "subtask_unknown_count": {},
            "subtask_unknown_weight": {},
        }

    return all_phase_stats


def _build_overall(phase_stats: Dict[str, Dict[str, Any]], agent_keys: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    plan_weight_sum: Dict[str, float] = {k: 0.0 for k in agent_keys}
    plan_pick_count: Dict[str, int] = {k: 0 for k in agent_keys}
    subtask_weight_sum: Dict[str, float] = {k: 0.0 for k in agent_keys}
    subtask_pick_count: Dict[str, int] = {k: 0 for k in agent_keys}
    plan_count = 0
    st_total = 0
    st_selected = 0
    for st in phase_stats.values():
        for k in agent_keys:
            plan_weight_sum[k] += st["plan_weight_sum"].get(k, 0.0)
            plan_pick_count[k] += st["plan_pick_count"].get(k, 0)
            subtask_weight_sum[k] += st["subtask_weight_sum"].get(k, 0.0)
            subtask_pick_count[k] += st["subtask_pick_count"].get(k, 0)
        plan_count += st["plan_count"]
        st_total += st["subtask_total_steps"]
        st_selected += st["subtask_selected_steps"]

    overall_plan = {
        "plan_count": plan_count,
        "plan_parse_fail": 0,
        "weights": plan_weight_sum,
        "pick_counts": plan_pick_count,
        "unknown_agents": {"weights": {}, "counts": {}},
    }

    overall_subtask = {
        "subtask_total_steps": st_total,
        "subtask_selected_steps": st_selected,
        "weights": subtask_weight_sum,
        "pick_counts": subtask_pick_count,
        "unknown_agents": {"counts": {}},
    }
    return overall_plan, overall_subtask


def run_one(pretrain_dir: str) -> None:
    pretrain_dir = os.path.abspath(pretrain_dir)
    csv_path = os.path.join(pretrain_dir, "accuracy_summary.csv")
    if not os.path.isfile(csv_path):
        print(f"[stats_from_pretrain_dir] 跳过（无 accuracy_summary.csv）: {pretrain_dir}")
        return

    rows = _load_summary(csv_path)
    if not rows:
        print(f"[stats_from_pretrain_dir] 空 CSV: {csv_path}")
        return

    agent_keys = _collect_agents(rows)
    if not agent_keys:
        print(f"[stats_from_pretrain_dir] 未解析出 agent: {csv_path}")
        return

    phase_stats = _build_phase_stats(rows, agent_keys)
    overall_plan, overall_subtask = _build_overall(phase_stats, agent_keys)

    os.makedirs(pretrain_dir, exist_ok=True)
    with open(os.path.join(pretrain_dir, "phase_stats.json"), "w", encoding="utf-8") as f:
        json.dump(phase_stats, f, ensure_ascii=False, indent=2)
    with open(os.path.join(pretrain_dir, "plan_weight_sum_overall.json"), "w", encoding="utf-8") as f:
        json.dump(overall_plan, f, ensure_ascii=False, indent=2)
    with open(os.path.join(pretrain_dir, "subtask_weight_sum_overall.json"), "w", encoding="utf-8") as f:
        json.dump(overall_subtask, f, ensure_ascii=False, indent=2)

    print(f"[stats_from_pretrain_dir] 已写入 phase_stats.json / plan_weight_sum_overall.json / subtask_weight_sum_overall.json -> {pretrain_dir}")


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python stats_from_pretrain_dir.py <pretrain_dir_1> [<pretrain_dir_2> ...]")
        sys.exit(1)
    for d in sys.argv[1:]:
        run_one(d)
    print("[stats_from_pretrain_dir] done.")


if __name__ == "__main__":
    main()
