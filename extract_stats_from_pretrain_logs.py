#!/usr/bin/env python3
"""
从 pretrain_*.jsonl 中的 trace_raw 提取真实的 plan/subtask 数据，生成 phase_stats.json 和 plan_weight_sum_overall.json
用法: python3 extract_stats_from_pretrain_logs.py <pretrain_dir>
"""

from __future__ import annotations

import os
import sys
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# 复用 select_only_stats 的解析函数
def _normalize_agent_id(agent_id: str, alias_map: Optional[Dict[str, str]] = None) -> str:
    """归一化 agent ID"""
    if not agent_id:
        return ""
    s = str(agent_id).strip()
    if alias_map:
        s = alias_map.get(s, s)
    return s

def _extract_plan_winner(trace: Dict[str, Any], alias_map: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], float]:
    """提取 plan winner (planner agent_id, weight)"""
    if alias_map is None:
        alias_map = {}
    plans = trace.get("plans")
    if isinstance(plans, list) and plans:
        best = None
        best_w = -1e18
        for p in plans:
            if not isinstance(p, dict):
                continue
            w = p.get("w", None)
            try:
                wv = float(w)
            except Exception:
                continue
            if wv > best_w:
                best_w = wv
                best = p
        if best is not None:
            planner = str(best.get("planner") or "").strip()
            if planner:
                planner = _normalize_agent_id(planner, alias_map)
            return (planner or None), float(best_w)

    keys = trace.get("keys") or []
    weights = trace.get("weights") or []
    if isinstance(keys, list) and isinstance(weights, list) and keys and weights:
        best_i = max(range(min(len(keys), len(weights))), key=lambda i: float(weights[i] or -1e18))
        planner_key = str(keys[best_i]).strip() or None
        if planner_key:
            planner_key = _normalize_agent_id(planner_key, alias_map)
        return planner_key, float(weights[best_i])

    return None, 0.0

def _extract_selected_agents_in_winner_plan(trace: Dict[str, Any], alias_map: Optional[Dict[str, str]] = None) -> List[Tuple[str, float]]:
    """提取 winner plan 的 subtask selections: List[(agent_id, match_score)]"""
    selected: List[Tuple[str, float]] = []
    if alias_map is None:
        alias_map = {}

    plans = trace.get("plans")
    if isinstance(plans, list) and plans:
        best_idx = None
        best_w = -1e18
        for i, p in enumerate(plans):
            if not isinstance(p, dict):
                continue
            try:
                wv = float(p.get("w"))
            except Exception:
                continue
            if wv > best_w:
                best_w = wv
                best_idx = i
        if best_idx is None:
            best_idx = 0

        p = plans[best_idx] if 0 <= best_idx < len(plans) else None
        if isinstance(p, dict):
            p_trace = p.get("trace") or {}
            steps = p_trace.get("steps") if isinstance(p_trace, dict) else None
            if isinstance(steps, list):
                for st in steps:
                    if not isinstance(st, dict):
                        continue
                    meta = st.get("meta") if isinstance(st.get("meta"), dict) else {}
                    sel = str(meta.get("selected") or "").strip()
                    match_score = float(meta.get("match_score", 0.0))
                    if sel:
                        normalized = _normalize_agent_id(sel, alias_map)
                        if normalized:
                            selected.append((normalized, match_score))
            return selected

    traces = trace.get("traces") or {}
    if isinstance(traces, dict):
        for _sid, st in traces.items():
            if not isinstance(st, dict):
                continue
            meta = st.get("meta") if isinstance(st.get("meta"), dict) else {}
            sel = str(meta.get("selected") or "").strip()
            match_score = float(meta.get("match_score", 0.0))
            if sel:
                normalized = _normalize_agent_id(sel, alias_map)
                if normalized:
                    selected.append((normalized, match_score))
                continue
            runs = st.get("runs") or []
            if isinstance(runs, list) and runs:
                r0 = runs[0] if isinstance(runs[0], dict) else {}
                aid = str(r0.get("agent_id") or "").strip()
                match_score = float(r0.get("match_score", 0.0))
                if aid:
                    normalized = _normalize_agent_id(aid, alias_map)
                    if normalized:
                        selected.append((normalized, match_score))

    return selected

def _collect_agents_from_logs(pretrain_dir: str) -> List[str]:
    """从 pretrain_*.jsonl 收集所有 agent IDs"""
    seen = set()
    for phase in ["cold_start", "pretrain", "test"]:
        path = os.path.join(pretrain_dir, f"pretrain_{phase}.jsonl")
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    for aid in rec.get("agent_ids", []):
                        if aid:
                            seen.add(aid)
                    trace_raw = rec.get("trace_raw")
                    if isinstance(trace_raw, dict):
                        winner, _ = _extract_plan_winner(trace_raw)
                        if winner:
                            seen.add(winner)
                        selected = _extract_selected_agents_in_winner_plan(trace_raw)
                        for aid, _ in selected:
                            if aid:
                                seen.add(aid)
                except Exception:
                    continue
    return sorted(seen)

def extract_from_dir(pretrain_dir: str) -> None:
    pretrain_dir = os.path.abspath(pretrain_dir)
    if not os.path.isdir(pretrain_dir):
        print(f"[extract_stats] 错误: 目录不存在: {pretrain_dir}")
        return

    agent_keys = _collect_agents_from_logs(pretrain_dir)
    if not agent_keys:
        print(f"[extract_stats] 警告: 未找到 agent IDs，尝试从 accuracy_summary.csv 读取")
        csv_path = os.path.join(pretrain_dir, "accuracy_summary.csv")
        if os.path.isfile(csv_path):
            import csv as csv_module
            with open(csv_path, "r", encoding="utf-8") as f:
                for r in csv_module.DictReader(f):
                    aid = (r.get("node_id") or "").strip().split(",")[0].strip()
                    if aid:
                        agent_keys.append(aid)
        agent_keys = sorted(set(agent_keys))

    if not agent_keys:
        print(f"[extract_stats] 错误: 无法确定 agent 列表")
        return

    print(f"[extract_stats] 找到 {len(agent_keys)} 个 agents: {agent_keys[:3]}...")

    phase_stats: Dict[str, Dict[str, Any]] = {}
    for phase_name in ["cold_start", "pretrain", "test"]:
        log_path = os.path.join(pretrain_dir, f"pretrain_{phase_name}.jsonl")
        if not os.path.isfile(log_path):
            continue

        plan_weight_sum: Dict[str, float] = {k: 0.0 for k in agent_keys}
        plan_pick_count: Dict[str, int] = {k: 0 for k in agent_keys}
        subtask_weight_sum: Dict[str, float] = {k: 0.0 for k in agent_keys}
        subtask_pick_count: Dict[str, int] = {k: 0 for k in agent_keys}
        subtask_total_steps = 0
        subtask_selected_steps = 0
        plan_count = 0
        plan_parse_fail = 0

        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    trace_raw = rec.get("trace_raw")
                    if not isinstance(trace_raw, dict):
                        plan_parse_fail += 1
                        continue

                    plan_count += 1
                    winner, winner_w = _extract_plan_winner(trace_raw)
                    if winner and winner in agent_keys:
                        plan_weight_sum[winner] += winner_w
                        plan_pick_count[winner] += 1
                    else:
                        plan_parse_fail += 1

                    selected_agents = _extract_selected_agents_in_winner_plan(trace_raw)
                    subtask_total_steps += len(selected_agents)
                    for aid, match_score in selected_agents:
                        if aid in agent_keys:
                            subtask_weight_sum[aid] += match_score
                            subtask_pick_count[aid] += 1
                            subtask_selected_steps += 1

                except Exception as e:
                    plan_parse_fail += 1
                    continue

        phase_stats[phase_name] = {
            "phase": phase_name,
            "plan_weight_sum": plan_weight_sum,
            "plan_pick_count": plan_pick_count,
            "plan_count": plan_count,
            "plan_parse_fail": plan_parse_fail,
            "plan_unknown_agents": {},
            "plan_unknown_count": {},
            "subtask_pick_count": subtask_pick_count,
            "subtask_weight_sum": subtask_weight_sum,
            "subtask_total_steps": subtask_total_steps,
            "subtask_selected_steps": subtask_selected_steps,
            "subtask_unknown_count": {},
            "subtask_unknown_weight": {},
        }

    # 生成 overall
    overall_plan_weight: Dict[str, float] = {k: 0.0 for k in agent_keys}
    overall_plan_count: Dict[str, int] = {k: 0 for k in agent_keys}
    overall_subtask_weight: Dict[str, float] = {k: 0.0 for k in agent_keys}
    overall_subtask_count: Dict[str, int] = {k: 0 for k in agent_keys}
    total_plan_count = 0
    total_subtask_steps = 0
    total_subtask_selected = 0

    for st in phase_stats.values():
        for k in agent_keys:
            overall_plan_weight[k] += st["plan_weight_sum"].get(k, 0.0)
            overall_plan_count[k] += st["plan_pick_count"].get(k, 0)
            overall_subtask_weight[k] += st["subtask_weight_sum"].get(k, 0.0)
            overall_subtask_count[k] += st["subtask_pick_count"].get(k, 0)
        total_plan_count += st["plan_count"]
        total_subtask_steps += st["subtask_total_steps"]
        total_subtask_selected += st["subtask_selected_steps"]

    overall_plan = {
        "plan_count": total_plan_count,
        "plan_parse_fail": sum(st["plan_parse_fail"] for st in phase_stats.values()),
        "weights": overall_plan_weight,
        "pick_counts": overall_plan_count,
        "unknown_agents": {"weights": {}, "counts": {}},
    }

    overall_subtask = {
        "subtask_total_steps": total_subtask_steps,
        "subtask_selected_steps": total_subtask_selected,
        "weights": overall_subtask_weight,
        "pick_counts": overall_subtask_count,
        "unknown_agents": {"counts": {}},
    }

    # 写入文件
    with open(os.path.join(pretrain_dir, "phase_stats.json"), "w", encoding="utf-8") as f:
        json.dump(phase_stats, f, ensure_ascii=False, indent=2)
    with open(os.path.join(pretrain_dir, "plan_weight_sum_overall.json"), "w", encoding="utf-8") as f:
        json.dump(overall_plan, f, ensure_ascii=False, indent=2)
    with open(os.path.join(pretrain_dir, "subtask_weight_sum_overall.json"), "w", encoding="utf-8") as f:
        json.dump(overall_subtask, f, ensure_ascii=False, indent=2)

    print(f"[extract_stats] ✅ 已写入 phase_stats.json / plan_weight_sum_overall.json / subtask_weight_sum_overall.json -> {pretrain_dir}")

def main():
    if len(sys.argv) < 2:
        print("用法: python3 extract_stats_from_pretrain_logs.py <pretrain_dir>")
        sys.exit(1)
    for d in sys.argv[1:]:
        extract_from_dir(d)
    print("[extract_stats] done.")

if __name__ == "__main__":
    main()
