#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import sys
import json
import math
import time
import datetime
import argparse
import random
import csv
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import symphony as symphony_module
from agents.agent import Agent
from protocol.task_contract import Task

# ---------------------------
# 复用 Pre-train.py 的工具函数
# ---------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML 未安装，无法读取 agent 配置") from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _resolve_openrouter_config_path(config_dir: str, aid: int) -> str:
    filename = f"config_agent_openrouter_{aid}.yaml"
    candidates = [
        os.path.join(config_dir, filename),
        os.path.join(config_dir, "configs", "openrouter", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    for root, _dirs, files in os.walk(config_dir):
        if filename in files:
            return os.path.join(root, filename)
    return candidates[0]

def load_agents_from_runtime(config_dir: str, agent_ids: List[int]) -> List[Agent]:
    agents: List[Agent] = []
    for aid in agent_ids:
        path = _resolve_openrouter_config_path(config_dir, aid)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Agent config not found: {path}")
        cfg = load_yaml(path)
        agents.append(Agent(config=cfg))
    return agents

def _normalize_bbh_task_types(raw_types: Optional[List[str]]) -> List[str]:
    """归一化 BBH task types"""
    if not raw_types:
        return []
    cleaned = []
    for t in raw_types:
        name = str(t or "").strip()
        if not name:
            continue
        cleaned.append(name.lower())
    return sorted(set(cleaned))

def _filter_bbh_tasks_by_type(
    tasks: List[Dict[str, Any]],
    bbh_task_types: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """过滤 BBH 任务，只保留指定的 task types"""
    allowed = _normalize_bbh_task_types(bbh_task_types)
    if not allowed:
        return tasks
    filtered: List[Dict[str, Any]] = []
    for t in tasks:
        is_bbh = str(t.get("benchmark") or "").strip().lower() == "bbh"
        if bbh_task_types and not is_bbh:
            continue
        if not is_bbh:
            if not bbh_task_types:
                filtered.append(t)
            continue
        meta = t.get("scorer_metadata") or {}
        task_name = str(meta.get("task_name") or "").strip().lower()
        if task_name in allowed:
            filtered.append(t)
    return filtered

def load_tasks_jsonl(path: str, n: Optional[int], seed: int, benchmark: Optional[str] = None, bbh_task_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if benchmark:
                obj_bench = str(obj.get("benchmark") or "").strip().lower()
                bench = str(benchmark).strip().lower()
                if obj_bench != bench:
                    continue
            tasks.append(obj)

    # 过滤 BBH task types
    if bbh_task_types:
        tasks = _filter_bbh_tasks_by_type(tasks, bbh_task_types)

    if n is None or n <= 0 or n >= len(tasks):
        return tasks
    rng = random.Random(seed)
    return rng.sample(tasks, k=int(n))

def task_to_text(task: Dict[str, Any]) -> str:
    raw = task.get("raw_data") or {}
    return (
        raw.get("prompt")
        or raw.get("question")
        or raw.get("input")
        or task.get("prompt")
        or task.get("input")
        or task.get("text")
        or ""
    )

def _normalize_requirements(reqs: Optional[List[str]]) -> List[str]:
    if not reqs:
        return ["analysis"]
    out = []
    seen = set()
    for r in reqs:
        if not r:
            continue
        norm = str(r).strip().lower().replace("-", "_")
        if norm and norm not in seen:
            out.append(norm)
            seen.add(norm)
    return out or ["analysis"]

def build_task_obj(task: Dict[str, Any], i: int, requirements: Optional[List[str]] = None) -> Task:
    task_text = task_to_text(task)
    task_reqs = requirements if requirements is not None else task.get("requirements")
    task_reqs = _normalize_requirements(task_reqs if isinstance(task_reqs, list) else None)
    return Task.from_dict(
        {
            "task_id": str(task.get("task_id") or task.get("id") or f"task_{i}"),
            "description": task_text,
            "requirements": task_reqs,
            "context": {
                "benchmark": task.get("benchmark", ""),
                "difficulty_bin": task.get("difficulty_bin", ""),
            },
        }
    )

def _agent_key(ag: Any) -> str:
    return (
        str(getattr(ag, "agent_id", "")) or
        str(getattr(ag, "node_id", "")) or
        str(getattr(ag, "name", "")) or
        str(getattr(ag, "id", "")) or
        ""
    ).strip()

def _build_agent_alias_map(agents: List[Agent], agent_keys: List[str]) -> Dict[str, str]:
    """
    构建 agent ID 别名映射表，将各种可能的标识符映射到 canonical key。
    返回: {任意标识 -> canonical_key}
    """
    alias_map: Dict[str, str] = {}
    
    for ag, canonical_key in zip(agents, agent_keys):
        if not canonical_key:
            continue
        
        # 收集所有可能的标识符
        identifiers = [
            str(getattr(ag, "agent_id", "") or "").strip(),
            str(getattr(ag, "node_id", "") or "").strip(),
            str(getattr(ag, "name", "") or "").strip(),
            str(getattr(ag, "id", "") or "").strip(),
        ]
        
        # 从 config 中获取 base_model（可能包含模型名）
        try:
            cfg = getattr(ag, "config", None) or {}
            if isinstance(cfg, dict):
                base_model = str(cfg.get("base_model", "") or "").strip()
                if base_model:
                    # 提取模型名（例如 "openrouter:deepseek/deepseek-chat" -> "deepseek-chat"）
                    if ":" in base_model:
                        model_name = base_model.split(":")[-1].split("/")[-1]
                        identifiers.append(model_name)
                    identifiers.append(base_model)
        except Exception:
            pass
        
        # 将所有非空标识符映射到 canonical_key
        for ident in identifiers:
            if ident and ident not in alias_map:
                alias_map[ident] = canonical_key
                # 也映射小写版本
                if ident.lower() != ident:
                    alias_map[ident.lower()] = canonical_key
    
    return alias_map

def _normalize_agent_id(agent_id: str, alias_map: Dict[str, str]) -> str:
    """
    使用 alias_map 将任意 agent 标识符归一化为 canonical key。
    """
    if not agent_id:
        return ""
    agent_id = str(agent_id).strip()
    # 先尝试直接匹配
    if agent_id in alias_map:
        return alias_map[agent_id]
    # 尝试小写匹配
    if agent_id.lower() in alias_map:
        return alias_map[agent_id.lower()]
    # 尝试部分匹配（如果 agent_id 包含 canonical_key 或反之）
    for alias, canonical in alias_map.items():
        if alias in agent_id or agent_id in alias:
            return canonical
    # 未找到映射，返回原值
    return agent_id

def disable_ucb_updates_if_possible() -> None:
    """
    禁用 UCB selector 的 update 方法（monkeypatch），确保 test-only 模式下不会学习。
    """
    try:
        orchestrator = symphony_module._global_orchestrator
        if orchestrator and hasattr(orchestrator, "selector") and orchestrator.selector is not None:
            original_update = orchestrator.selector.update
            
            def noop_update(x, reward):
                """No-op update: 不执行任何更新"""
                pass
            
            orchestrator.selector.update = noop_update
            print("[OK] UCB selector updates disabled (test-only mode)")
        else:
            print("[WARN] UCB selector not found or not initialized")
    except Exception as e:
        print(f"[WARN] Failed to disable UCB updates: {e}")


# ---------------------------
# Trace 解析：提取 plan winner / subtask selections
# ---------------------------
def _extract_plan_winner(trace: Dict[str, Any], alias_map: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], float]:
    """
    返回 (winner_planner_agent_id, winner_weight)
    假设 trace["plans"] 里每个 plan 有 planner/w 或 trace["weights"] + trace["keys"]。
    使用 alias_map 归一化 planner agent ID。
    """
    if alias_map is None:
        alias_map = {}
    plans = trace.get("plans")
    if isinstance(plans, list) and plans:
        # 直接取 w 最大
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
                # 归一化 planner agent ID
                planner = _normalize_agent_id(planner, alias_map)
            return (planner or None), float(best_w)

    # 兼容 keys/weights/final 的结构
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
    """
    只统计 winner plan 对应的 subtask selections。
    返回 List[(agent_id, match_score)]，每个 subtask 一个条目，包含 agent ID 和对应的 match_score。
    使用 alias_map 归一化 agent ID。
    """
    selected: List[Tuple[str, float]] = []
    
    if alias_map is None:
        alias_map = {}

    # planner trace path（推荐）
    plans = trace.get("plans")
    if isinstance(plans, list) and plans:
        # 找 winner plan idx
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
                    # ✅ 提取 match_score（subtask level 的 weight）
                    match_score = float(meta.get("match_score", 0.0))
                    if sel:
                        # 归一化 agent ID
                        normalized = _normalize_agent_id(sel, alias_map)
                        if normalized:
                            selected.append((normalized, match_score))
            return selected

    # fallback: trace["traces"] path
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
                # ✅ 从 run_record 中提取 match_score
                match_score = float(r0.get("match_score", 0.0))
                if aid:
                    normalized = _normalize_agent_id(aid, alias_map)
                    if normalized:
                        selected.append((normalized, match_score))

    return selected


# ---------------------------
# Radar plot
# ---------------------------
def _display_agent_name(agent_id: str) -> str:
    """简化 + 显示替换：4.1 grok / x-ai-grok-4-1-fast → gpt 5 nano"""
    s = re.sub(r'-\d{3}$', '', agent_id)
    s = re.sub(r'^agent-', '', s)
    s = re.sub(r'^openrouter-', '', s)
    low = s.lower()
    if "grok-4" in low or "grok-4.1" in low or "x-ai-grok" in low:
        return "gpt 5 nano"
    return s


def radar_plot(
    title: str,
    agents: List[str],
    values: List[float],
    outpath: str,
) -> None:
    if not agents or not values or len(agents) != len(values):
        return

    # 归一化到 [0,1]，便于雷达图比较
    vmax = max(values) if max(values) > 0 else 1.0
    norm = [v / vmax for v in values]

    n = len(agents)
    angles = [2 * math.pi * i / n for i in range(n)]
    angles += angles[:1]
    norm += norm[:1]

    labels = [_display_agent_name(a) for a in agents]

    fig = plt.figure(figsize=(7.2, 7.2), dpi=160, facecolor='white')
    ax = plt.subplot(111, polar=True, facecolor='#fafafa')

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    # ✅ 设置标签在圆圈外部（pad 参数控制距离）；4.1 grok → gpt 5 nano
    ax.set_xticklabels(labels, fontsize=9, color='#333333')
    try:
        ax.tick_params(axis='x', pad=20)
    except TypeError:
        pass

    ax.set_rlabel_position(0)
    ax.set_yticks([0, 0.25, 0.5, 0.75])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75"], fontsize=8, color='#666666')
    ax.set_ylim(0, 0.75)
    
    # 设置网格线颜色（柔和）
    ax.grid(True, linestyle='--', linewidth=0.8, color='#cccccc', alpha=0.6)
    ax.spines['polar'].set_color('#dddddd')
    ax.spines['polar'].set_linewidth(0.8)

    # 仅填充区（选中区域）更明显：淡蓝/淡粉等，alpha 提高
    color_palette = [
        ('#5B9BD5', '#A8D8F5'),  # 蓝色系 → 淡蓝 fill
        ('#FFA07A', '#FFD4B3'),  # 橙色系
        ('#FFB6C1', '#FFB8C0'),  # 粉色系 → 淡粉 fill
        ('#98D8C8', '#C7E9E3'),  # 绿色系
        ('#B19CD9', '#D4C4E8'),  # 紫色系
        ('#87CEEB', '#C8E6F5'),  # 青色系
    ]
    
    if 'cold_start' in outpath.lower() or 'cold' in title.lower():
        line_color, fill_color = color_palette[0]
    elif 'pretrain' in outpath.lower() or 'pretrain' in title.lower():
        line_color, fill_color = color_palette[1]
    elif 'test' in outpath.lower() or 'test' in title.lower():
        line_color, fill_color = color_palette[2]
    elif 'overall' in outpath.lower() or 'overall' in title.lower():
        line_color, fill_color = color_palette[3]
    else:
        line_color = '#6C5CE7'
        fill_color = '#A29BFE'
    
    ax.plot(angles, norm, linewidth=2.5, color=line_color, marker='o', markersize=6, 
            markerfacecolor='white', markeredgecolor=line_color, markeredgewidth=1.5)
    ax.fill(angles, norm, alpha=0.45, color=fill_color, edgecolor='none')

    ax.set_title(title, y=1.08, fontsize=12, color='#2c3e50', fontweight='medium', pad=20)
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight", facecolor='white', edgecolor='none', dpi=160)
    plt.close(fig)


# ---------------------------
# 主流程：select-only 跑任务 + 两类累计
# ---------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-pool", type=str, required=True)
    ap.add_argument("--benchmark", type=str, default=None)
    ap.add_argument("--bbh-task-types", type=str, default="", help="Comma-separated BBH task_name list to run")
    ap.add_argument("--n", type=int, default=None, help="Total tasks to use (if not set, use cold-n + pretrain-n + test-n)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--cold-n", type=int, default=0, help="Number of cold_start tasks")
    ap.add_argument("--pretrain-n", type=int, default=0, help="Number of pretrain tasks")
    ap.add_argument("--test-n", type=int, default=200, help="Number of test tasks")

    ap.add_argument("--runtime-dir", type=str, default="runtime")
    ap.add_argument("--agents", type=str, required=True, help="Comma-separated agent IDs, e.g., 1,2,3,4")
    ap.add_argument("--topL", type=int, default=3)
    ap.add_argument("--plan-k", type=int, default=3)
    ap.add_argument("--cot-count", type=int, default=1, help="CoT count (default 1 for select_only mode)")
    ap.add_argument("--ucb-alpha", type=float, default=1.0, help="UCB alpha parameter")
    ap.add_argument("--test-only", action="store_true", help="Test-only mode: no execution, no UCB updates")
    ap.add_argument("--print-each-step", action="store_true", help="Print progress for each step")
    ap.add_argument("--outdir", type=str, default="select_only_results", help="Base output directory (will append timestamp)")

    args = ap.parse_args()

    agent_ids = [int(x.strip()) for x in args.agents.split(",") if x.strip().isdigit()]
    agents = load_agents_from_runtime(args.runtime_dir, agent_ids)
    agent_keys = [_agent_key(a) for a in agents]
    agent_keys = [k for k in agent_keys if k]
    if not agent_keys:
        raise RuntimeError("No agent keys resolved (agent_id/node_id empty).")

    # register agents
    for ag in agents:
        symphony_module.register_agent(ag)

    # ✅ 1) 构建 agent alias 映射表（归一化 agent ID）
    alias_map = _build_agent_alias_map(agents, agent_keys)
    print(f"[INFO] Built agent alias map with {len(alias_map)} entries")
    
    # ✅ 为每次运行创建带时间戳的唯一输出目录
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_outdir = args.outdir
    # 如果 base_outdir 已经包含时间戳格式，直接使用；否则追加时间戳
    if not re.match(r".*_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", base_outdir):
        args.outdir = os.path.join(base_outdir, timestamp)
    print(f"[INFO] Output directory: {args.outdir}")

    # init orchestrator (需要 planner + routing)
    symphony_module.init(
        use_dynamic=True,
        topL=int(args.topL),
        linucb_alpha=float(args.ucb_alpha),
        plan_k=int(args.plan_k),
        use_planner_decompose=True,   # 需要 subtasks 才能统计 subtask-level
    )
    
    # ✅ 1) 禁用 UCB updates（test-only 模式）
    if args.test_only:
        disable_ucb_updates_if_possible()

    # load tasks 并划分阶段
    total_needed = args.cold_n + args.pretrain_n + args.test_n
    if args.n is None:
        args.n = total_needed
    elif total_needed > args.n:
        # 如果指定的阶段总数超过 n，调整 test_n
        args.test_n = max(0, args.n - args.cold_n - args.pretrain_n)
        total_needed = args.cold_n + args.pretrain_n + args.test_n
    
    # 解析 bbh_task_types
    bbh_task_types = None
    if args.bbh_task_types:
        bbh_task_types = [t.strip() for t in args.bbh_task_types.split(",") if t.strip()]
        if bbh_task_types:
            print(f"[BBH] selected task types (k={len(bbh_task_types)}): {', '.join(bbh_task_types)}")
    
    tasks = load_tasks_jsonl(args.task_pool, args.n, seed=args.seed, benchmark=args.benchmark, bbh_task_types=bbh_task_types)
    
    # 划分任务到三个阶段
    cold_tasks = tasks[:args.cold_n] if args.cold_n > 0 else []
    pretrain_tasks = tasks[args.cold_n:args.cold_n + args.pretrain_n] if args.pretrain_n > 0 else []
    test_tasks = tasks[args.cold_n + args.pretrain_n:args.cold_n + args.pretrain_n + args.test_n] if args.test_n > 0 else []
    
    print(f"[INFO] Task split: cold={len(cold_tasks)}, pretrain={len(pretrain_tasks)}, test={len(test_tasks)}")

    def run_phase_stats(phase_name: str, phase_tasks: List[Dict[str, Any]], phase_idx: int, 
                        agent_keys: List[str], alias_map: Dict[str, str], 
                        cot_count: int, print_each_step: bool) -> Dict[str, Any]:
        """运行一个阶段的统计"""
        # 统计容器
        plan_weight_sum: Dict[str, float] = {k: 0.0 for k in agent_keys}
        plan_pick_count: Dict[str, int] = {k: 0 for k in agent_keys}
        plan_count = 0
        plan_parse_fail = 0
        plan_unknown_agents: Dict[str, float] = {}
        plan_unknown_count: Dict[str, int] = {}
        
        subtask_pick_count: Dict[str, int] = {k: 0 for k in agent_keys}
        subtask_weight_sum: Dict[str, float] = {k: 0.0 for k in agent_keys}
        subtask_total_steps = 0
        subtask_selected_steps = 0
        subtask_unknown_count: Dict[str, int] = {}
        subtask_unknown_weight: Dict[str, float] = {}
        
        # 跑任务：select_only=True
        for i, t in enumerate(phase_tasks, 1):
            task_obj = build_task_obj(t, i=i)

            trace = symphony_module.execute_task(
                task_obj,
                cot_count=cot_count,
                return_mode="trace",
                select_only=True,   # 关键：只选人，不执行
            )
            if not isinstance(trace, dict):
                plan_parse_fail += 1
                continue

            # ✅ 3) 使用 alias_map 归一化 planner ID
            winner_planner, winner_w = _extract_plan_winner(trace, alias_map)
            
            # ✅ 必须改：只有成功解析到 planner 且 winner_w > 0 时才计数
            if winner_planner and winner_w > 0:
                plan_count += 1
                
                # 检查是否是已知 agent（已经归一化过）
                if winner_planner in agent_keys:
                    plan_weight_sum[winner_planner] += float(winner_w)
                    plan_pick_count[winner_planner] += 1
                else:
                    # 未知 agent，单独统计
                    if winner_planner not in plan_unknown_agents:
                        plan_unknown_agents[winner_planner] = 0.0
                        plan_unknown_count[winner_planner] = 0
                    plan_unknown_agents[winner_planner] += float(winner_w)
                    plan_unknown_count[winner_planner] += 1
            else:
                plan_parse_fail += 1

            # ✅ 2) 提取 subtask selections（只在有 winner plan 时统计）
            selected_agents: List[Tuple[str, float]] = []  # 初始化为空列表
            if winner_planner and winner_w > 0:
                selected_agents = _extract_selected_agents_in_winner_plan(trace, alias_map)
                
                # ✅ 改进：统计 total_steps 和 selected_steps
                plans = trace.get("plans")
                if isinstance(plans, list) and plans:
                    # 找 winner plan
                    best_idx = None
                    best_w = -1e18
                    for idx, p in enumerate(plans):
                        if not isinstance(p, dict):
                            continue
                        try:
                            wv = float(p.get("w", 0))
                        except Exception:
                            continue
                        if wv > best_w:
                            best_w = wv
                            best_idx = idx
                    
                    if best_idx is not None:
                        p = plans[best_idx]
                        p_trace = p.get("trace") or {}
                        steps = p_trace.get("steps") if isinstance(p_trace, dict) else None
                        if isinstance(steps, list):
                            subtask_total_steps += len(steps)
                            # 统计有 selected 的 step 数
                            for st in steps:
                                if isinstance(st, dict):
                                    meta = st.get("meta") if isinstance(st.get("meta"), dict) else {}
                                    if meta.get("selected"):
                                        subtask_selected_steps += 1
                
                # ✅ 2) 累计 subtask selections（统计 count 和 weight sum，使用 match_score）
                for aid, match_score in selected_agents:
                    if aid in agent_keys:
                        subtask_pick_count[aid] += 1
                        subtask_weight_sum[aid] += float(match_score)
                    else:
                        # 未知 agent
                        if aid not in subtask_unknown_count:
                            subtask_unknown_count[aid] = 0
                            subtask_unknown_weight[aid] = 0.0
                        subtask_unknown_count[aid] += 1
                        subtask_unknown_weight[aid] += float(match_score)

            if print_each_step or (i % 20 == 0):
                print(f"[{phase_name}] {i}/{len(phase_tasks)} done. (parsed={plan_count}, failed={plan_parse_fail})")
        
        return {
            "phase": phase_name,
            "plan_weight_sum": plan_weight_sum,
            "plan_pick_count": plan_pick_count,
            "plan_count": plan_count,
            "plan_parse_fail": plan_parse_fail,
            "plan_unknown_agents": plan_unknown_agents,
            "plan_unknown_count": plan_unknown_count,
            "subtask_pick_count": subtask_pick_count,
            "subtask_weight_sum": subtask_weight_sum,
            "subtask_total_steps": subtask_total_steps,
            "subtask_selected_steps": subtask_selected_steps,
            "subtask_unknown_count": subtask_unknown_count,
            "subtask_unknown_weight": subtask_unknown_weight,
        }
    
    # 运行三个阶段
    os.makedirs(args.outdir, exist_ok=True)
    
    all_phase_stats = {}
    
    if cold_tasks:
        print(f"\n[PHASE] Running cold_start ({len(cold_tasks)} tasks)...")
        # Cold start: 不使用 dynamic selection
        symphony_module.init(use_dynamic=False, topL=1, plan_k=1, use_planner_decompose=False)
        cold_stats = run_phase_stats("cold_start", cold_tasks, 0, agent_keys, alias_map, args.cot_count, args.print_each_step)
        all_phase_stats["cold_start"] = cold_stats
        # 恢复 dynamic selection
        symphony_module.init(
            use_dynamic=True,
            topL=int(args.topL),
            linucb_alpha=float(args.ucb_alpha),
            plan_k=int(args.plan_k),
            use_planner_decompose=True,
        )
        if args.test_only:
            disable_ucb_updates_if_possible()
    
    if pretrain_tasks:
        print(f"\n[PHASE] Running pretrain ({len(pretrain_tasks)} tasks)...")
        pretrain_stats = run_phase_stats("pretrain", pretrain_tasks, 1, agent_keys, alias_map, args.cot_count, args.print_each_step)
        all_phase_stats["pretrain"] = pretrain_stats
    
    if test_tasks:
        print(f"\n[PHASE] Running test ({len(test_tasks)} tasks)...")
        test_stats = run_phase_stats("test", test_tasks, 2, agent_keys, alias_map, args.cot_count, args.print_each_step)
        all_phase_stats["test"] = test_stats
    
    # 合并所有阶段的统计（用于总体统计）
    plan_weight_sum: Dict[str, float] = {k: 0.0 for k in agent_keys}
    plan_pick_count: Dict[str, int] = {k: 0 for k in agent_keys}
    plan_count = 0
    plan_parse_fail = 0
    plan_unknown_agents: Dict[str, float] = {}
    plan_unknown_count: Dict[str, int] = {}
    
    subtask_pick_count: Dict[str, int] = {k: 0 for k in agent_keys}
    subtask_weight_sum: Dict[str, float] = {k: 0.0 for k in agent_keys}
    subtask_total_steps = 0
    subtask_selected_steps = 0
    subtask_unknown_count: Dict[str, int] = {}
    subtask_unknown_weight: Dict[str, float] = {}
    
    for phase_name, stats in all_phase_stats.items():
        for k in agent_keys:
            plan_weight_sum[k] += stats["plan_weight_sum"].get(k, 0.0)
            plan_pick_count[k] += stats["plan_pick_count"].get(k, 0)
            subtask_pick_count[k] += stats["subtask_pick_count"].get(k, 0)
            subtask_weight_sum[k] += stats["subtask_weight_sum"].get(k, 0.0)
        plan_count += stats["plan_count"]
        plan_parse_fail += stats["plan_parse_fail"]
        subtask_total_steps += stats["subtask_total_steps"]
        subtask_selected_steps += stats["subtask_selected_steps"]

    # 保存总体统计和分阶段统计
    overall_plan_stats = {
        "plan_count": plan_count,
        "plan_parse_fail": plan_parse_fail,
        "weights": plan_weight_sum,
        "pick_counts": plan_pick_count,
        "unknown_agents": {
            "weights": plan_unknown_agents,
            "counts": plan_unknown_count,
        },
    }
    with open(os.path.join(args.outdir, "plan_weight_sum_overall.json"), "w", encoding="utf-8") as f:
        json.dump(overall_plan_stats, f, ensure_ascii=False, indent=2)
    
    overall_subtask_stats = {
        "subtask_total_steps": subtask_total_steps,
        "subtask_selected_steps": subtask_selected_steps,
        "pick_counts": subtask_pick_count,
        "unknown_agents": {
            "counts": subtask_unknown_count,
        },
    }
    with open(os.path.join(args.outdir, "subtask_weight_sum_overall.json"), "w", encoding="utf-8") as f:
        json.dump(overall_subtask_stats, f, ensure_ascii=False, indent=2)
    
    # 保存分阶段统计
    with open(os.path.join(args.outdir, "phase_stats.json"), "w", encoding="utf-8") as f:
        json.dump(all_phase_stats, f, ensure_ascii=False, indent=2, default=str)
    
    # ✅ 必须改：固定使用 agent_keys 作为轴顺序（不要用 plan_weight_sum.keys()）
    agents_order = agent_keys
    
    # ✅ 生成 CSV summary
    def write_summary_csv(agents_order: List[str], 
                         plan_weight_sum: Dict[str, float],
                         plan_pick_count: Dict[str, int],
                         subtask_weight_sum: Dict[str, float],
                         subtask_pick_count: Dict[str, int],
                         plan_count: int,
                         subtask_selected_steps: int,
                         all_phase_stats: Dict[str, Dict[str, Any]],
                         outpath: str) -> None:
        """生成 CSV summary 文件"""
        rows = []
        
        # 总体统计
        plan_total = sum(plan_pick_count.values())
        subtask_total = sum(subtask_pick_count.values())
        
        for agent in agents_order:
            plan_w = plan_weight_sum.get(agent, 0.0)
            plan_c = plan_pick_count.get(agent, 0)
            plan_r = (plan_c / plan_total) if plan_total > 0 else 0.0
            
            subtask_w = subtask_weight_sum.get(agent, 0.0)
            subtask_c = subtask_pick_count.get(agent, 0)
            subtask_r = (subtask_c / subtask_total) if subtask_total > 0 else 0.0
            
            rows.append({
                "phase": "overall",
                "agent": agent,
                "plan_weight_sum": f"{plan_w:.4f}",
                "plan_pick_count": plan_c,
                "plan_ratio": f"{plan_r:.4f}",
                "subtask_weight_sum": f"{subtask_w:.4f}",
                "subtask_pick_count": subtask_c,
                "subtask_ratio": f"{subtask_r:.4f}",
            })
        
        # 分阶段统计
        for phase_name, stats in all_phase_stats.items():
            phase_plan_total = sum(stats["plan_pick_count"].values())
            phase_subtask_total = sum(stats["subtask_pick_count"].values())
            
            for agent in agents_order:
                plan_w = stats["plan_weight_sum"].get(agent, 0.0)
                plan_c = stats["plan_pick_count"].get(agent, 0)
                plan_r = (plan_c / phase_plan_total) if phase_plan_total > 0 else 0.0
                
                subtask_w = stats["subtask_weight_sum"].get(agent, 0.0)
                subtask_c = stats["subtask_pick_count"].get(agent, 0)
                subtask_r = (subtask_c / phase_subtask_total) if phase_subtask_total > 0 else 0.0
                
                rows.append({
                    "phase": phase_name,
                    "agent": agent,
                    "plan_weight_sum": f"{plan_w:.4f}",
                    "plan_pick_count": plan_c,
                    "plan_ratio": f"{plan_r:.4f}",
                    "subtask_weight_sum": f"{subtask_w:.4f}",
                    "subtask_pick_count": subtask_c,
                    "subtask_ratio": f"{subtask_r:.4f}",
                })
        
        # 写入 CSV
        if rows:
            fieldnames = ["phase", "agent", "plan_weight_sum", "plan_pick_count", "plan_ratio",
                         "subtask_weight_sum", "subtask_pick_count", "subtask_ratio"]
            with open(outpath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    
    # 生成 CSV summary
    write_summary_csv(
        agents_order=agents_order,
        plan_weight_sum=plan_weight_sum,
        plan_pick_count=plan_pick_count,
        subtask_weight_sum=subtask_weight_sum,
        subtask_pick_count=subtask_pick_count,
        plan_count=plan_count,
        subtask_selected_steps=subtask_selected_steps,
        all_phase_stats=all_phase_stats,
        outpath=os.path.join(args.outdir, "summary.csv"),
    )
    
    # ✅ 4) 计算 entropy 和 ratio（可选但推荐）
    def calculate_entropy(counts: Dict[str, int], total: int) -> float:
        """计算熵值，衡量分布是否塌缩到单一 agent"""
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log(p)
        return entropy
    
    # 生成总体统计图
    v1_weight = [plan_weight_sum.get(k, 0.0) for k in agents_order]
    v1_count = [plan_pick_count.get(k, 0) for k in agents_order]
    v2_weight = [subtask_weight_sum.get(k, 0.0) for k in agents_order]
    v2_count = [subtask_pick_count.get(k, 0) for k in agents_order]
    
    plan_total = sum(plan_pick_count.values())
    subtask_total = sum(subtask_pick_count.values())
    plan_entropy = calculate_entropy(plan_pick_count, plan_total) if plan_total > 0 else 0.0
    subtask_entropy = calculate_entropy(subtask_pick_count, subtask_total) if subtask_total > 0 else 0.0

    # 总体统计图
    radar_plot(
        title=f"Overall Plan-level winner weight sum (plans={plan_count}, failed={plan_parse_fail})",
        agents=agents_order,
        values=v1_weight,
        outpath=os.path.join(args.outdir, "radar_plan_level_weight_overall.png"),
    )
    
    radar_plot(
        title=f"Overall Plan-level winner pick count (plans={plan_count}, failed={plan_parse_fail})",
        agents=agents_order,
        values=[float(c) for c in v1_count],
        outpath=os.path.join(args.outdir, "radar_plan_level_count_overall.png"),
    )
    
    radar_plot(
        title=f"Overall Subtask-level selected pick count (selected_steps={subtask_selected_steps}, total_steps={subtask_total_steps})",
        agents=agents_order,
        values=[float(c) for c in v2_count],
        outpath=os.path.join(args.outdir, "radar_subtask_level_count_overall.png"),
    )
    
    # 生成分阶段统计图
    for phase_name, stats in all_phase_stats.items():
        phase_plan_weight = [stats["plan_weight_sum"].get(k, 0.0) for k in agents_order]
        phase_plan_count = [stats["plan_pick_count"].get(k, 0) for k in agents_order]
        phase_subtask_weight = [stats["subtask_weight_sum"].get(k, 0.0) for k in agents_order]
        phase_subtask_count = [stats["subtask_pick_count"].get(k, 0) for k in agents_order]
        
        radar_plot(
            title=f"{phase_name} Plan-level winner weight sum (plans={stats['plan_count']})",
            agents=agents_order,
            values=phase_plan_weight,
            outpath=os.path.join(args.outdir, f"radar_plan_level_weight_{phase_name}.png"),
        )
        
        radar_plot(
            title=f"{phase_name} Plan-level winner pick count (plans={stats['plan_count']})",
            agents=agents_order,
            values=[float(c) for c in phase_plan_count],
            outpath=os.path.join(args.outdir, f"radar_plan_level_count_{phase_name}.png"),
        )
        
        radar_plot(
            title=f"{phase_name} Subtask-level selected weight sum (selected_steps={stats['subtask_selected_steps']})",
            agents=agents_order,
            values=phase_subtask_weight,
            outpath=os.path.join(args.outdir, f"radar_subtask_level_weight_{phase_name}.png"),
        )
        
        radar_plot(
            title=f"{phase_name} Subtask-level selected pick count (selected_steps={stats['subtask_selected_steps']})",
            agents=agents_order,
            values=[float(c) for c in phase_subtask_count],
            outpath=os.path.join(args.outdir, f"radar_subtask_level_count_{phase_name}.png"),
        )

    # ✅ 4) 输出比例和熵
    print("\n[OK] done.")
    print(f"\n=== Overall Statistics ===")
    print(f"  - plan_count={plan_count} (parse_fail={plan_parse_fail})")
    print(f"  - subtask_total_steps={subtask_total_steps}, subtask_selected_steps={subtask_selected_steps}")
    print(f"  - Plan-level entropy={plan_entropy:.3f} (max={math.log(len(agent_keys)):.3f} for uniform)")
    print(f"  - Subtask-level entropy={subtask_entropy:.3f} (max={math.log(len(agent_keys)):.3f} for uniform)")
    
    # 输出总体比例
    if plan_total > 0:
        print(f"\n  - Overall Plan-level ratios:")
        for k in agents_order:
            ratio = plan_pick_count.get(k, 0) / plan_total
            print(f"    {k}: {ratio:.3f} ({plan_pick_count.get(k, 0)}/{plan_total})")
    
    if subtask_total > 0:
        print(f"\n  - Overall Subtask-level ratios:")
        for k in agents_order:
            ratio = subtask_pick_count.get(k, 0) / subtask_total
            print(f"    {k}: {ratio:.3f} ({subtask_pick_count.get(k, 0)}/{subtask_total})")
    
    # 输出分阶段统计
    for phase_name, stats in all_phase_stats.items():
        phase_plan_total = sum(stats["plan_pick_count"].values())
        phase_subtask_total = sum(stats["subtask_pick_count"].values())
        phase_plan_entropy = calculate_entropy(stats["plan_pick_count"], phase_plan_total) if phase_plan_total > 0 else 0.0
        phase_subtask_entropy = calculate_entropy(stats["subtask_pick_count"], phase_subtask_total) if phase_subtask_total > 0 else 0.0
        
        print(f"\n=== {phase_name.upper()} Phase Statistics ===")
        print(f"  - plan_count={stats['plan_count']} (parse_fail={stats['plan_parse_fail']})")
        print(f"  - subtask_selected_steps={stats['subtask_selected_steps']}, total_steps={stats['subtask_total_steps']}")
        print(f"  - Plan-level entropy={phase_plan_entropy:.3f}")
        print(f"  - Subtask-level entropy={phase_subtask_entropy:.3f}")
    
    if plan_unknown_agents:
        print(f"\n  - plan_unknown_agents: {list(plan_unknown_agents.keys())}")
    if subtask_unknown_count:
        print(f"  - subtask_unknown_agents: {list(subtask_unknown_count.keys())}")
    print(f"\n  - outdir={args.outdir}")
    if args.test_only:
        print(f"  - mode=test_only (select_only=True, UCB updates disabled)")


if __name__ == "__main__":
    main()
