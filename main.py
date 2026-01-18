#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import json
import time
import yaml
import argparse
import random
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR) if os.path.basename(_THIS_DIR) == "main" else os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agents.agent import Agent
from core.linucb_selector import GlobalLinUCB
from core.voting import plan_vote_and_update

# Symphony uses Task objects
from protocol.task_contract import Task


# -------------------------
# IO helpers
# -------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def dump_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------
# Agent loading
# -------------------------

def load_openrouter_agents_from_dir(
    config_dir: str,
    pattern_prefix: str = "config_agent_openrouter_",
) -> Dict[str, Agent]:
    """
    Load agents from runtime/config_agent_openrouter_*.yaml
    Returns: {node_id: Agent}
    """
    agents: Dict[str, Agent] = {}

    files: List[str] = []
    for name in os.listdir(config_dir):
        if name.startswith(pattern_prefix) and name.endswith(".yaml"):
            files.append(os.path.join(config_dir, name))
    files.sort()

    if not files:
        raise FileNotFoundError(f"No agent configs found in {config_dir} with prefix {pattern_prefix}")

    for p in files:
        cfg = load_yaml(p)
        a = Agent(config=cfg)
        agents[a.agent_id] = a

    return agents


# -------------------------
# Cold start / Top-L placeholders (interfaces only)
# -------------------------

def maybe_cold_start(*args, **kwargs):
    """
    TODO (reserved): warm-up selector by evaluating each agent.
    """
    return None


def maybe_topL_filter(*args, **kwargs):
    """
    TODO (reserved): beacon-stage / embedding stage filter candidates.
    """
    return None


# -------------------------
# Symphony run loop
# -------------------------

def _task_to_text(task: Dict[str, Any]) -> str:
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


def run_tasks_with_symphony(
    *,
    tasks: List[Dict[str, Any]],
    symphony_module: Any,
    # external selector is NOT injected into symphony (symphony owns its selector);
    # keep here only for future extension, and for optional external logging
    selector: Optional[GlobalLinUCB],
    policy_name: str,
    win_bonus: float,
    outdir: str,
    cot_count: int,
    plan_k: int,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    # Use Symphony's own extractor for consistent key parsing
    extract_fn = None
    if hasattr(symphony_module, "SymphonyOrchestrator") and hasattr(
        symphony_module.SymphonyOrchestrator, "_extract_final_from_text"
    ):
        extract_fn = symphony_module.SymphonyOrchestrator._extract_final_from_text

    logs: List[Dict[str, Any]] = []

    for i, task in enumerate(tasks):
        t0 = time.time()

        task_text = _task_to_text(task)

        # Build a Task for Symphony
        # - requirements: keep simple now; later you can map benchmark/modality -> requirements
        # - context: keep benchmark/difficulty for trace/logging
        task_obj = Task.from_dict(
            {
                "task_id": str(task.get("task_id") or task.get("id") or f"task_{i}"),
                "description": task_text,
                "requirements": ["general-reasoning"],
                "context": {
                    "benchmark": task.get("benchmark", ""),
                    "difficulty_bin": task.get("difficulty_bin", ""),
                },
            }
        )

        # IMPORTANT: symphony return_mode supports: "aggregate" | "final" | "trace"
        trace = symphony_module.execute_task(task_obj, cot_count=int(cot_count), return_mode="trace")

        # Optional: plan-level voting summary for logging (DO NOT update selector here to avoid double-update)
        vote_res = None
        if int(plan_k) > 1 and isinstance(trace, dict) and ("keys" in trace) and ("weights" in trace):
            keys = trace.get("keys") or []
            weights = trace.get("weights") or []

            # Adapt to core.voting's expected plan_runs shape
            plan_runs = [{"final_text": k, "plan_weight": w, "step_records": []} for k, w in zip(keys, weights)]

            vote_res = plan_vote_and_update(
                plan_runs=plan_runs,
                selector=None,  # ✅ avoid double update; symphony already updates internally
                policy_name=policy_name,
                win_bonus=win_bonus,
                extract_fn=extract_fn,
            )

        dt = time.time() - t0

        # Keep logs reasonably small: store a compact trace summary
        trace_summary: Dict[str, Any] = {}
        if isinstance(trace, dict):
            for k in ("final", "final_text", "keys", "weights", "win_key"):
                if k in trace:
                    trace_summary[k] = trace.get(k)

        logs.append(
            {
                "i": i,
                "task_id": task.get("task_id") or task.get("id"),
                "benchmark": task.get("benchmark"),
                "difficulty_bin": task.get("difficulty_bin"),
                "latency_s": dt,
                "trace_summary": trace_summary,
                "vote": vote_res,
            }
        )

        if (i + 1) % 10 == 0:
            print(f"[main] {i+1}/{len(tasks)} done, last latency={dt:.2f}s")

    dump_jsonl(logs, os.path.join(outdir, "main_run.jsonl"))
    print(f"[OK] logs saved to {outdir}")


# -------------------------
# Entry
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-config-dir", type=str, default=os.path.join(_ROOT, "runtime"))
    ap.add_argument("--task-pool", type=str, required=True, help="jsonl task pool path")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)

    # Symphony knobs (leave interfaces for cold-start/topL for later)
    ap.add_argument("--use-dynamic", action="store_true")
    ap.add_argument("--topL", type=int, default=3)
    ap.add_argument("--plan-k", type=int, default=3)
    ap.add_argument("--cot-count", type=int, default=3)

    # LinUCB params (passed into symphony.init; symphony owns the selector)
    ap.add_argument("--ucb-d", type=int, default=6)
    ap.add_argument("--ucb-l2", type=float, default=1.0)
    ap.add_argument("--ucb-alpha", type=float, default=1.0)
    ap.add_argument("--ucb-delta", type=float, default=0.05)
    ap.add_argument("--ucb-S", type=float, default=1.0)

    # voting (for logging summary only; actual update handled by symphony internally)
    ap.add_argument("--policy-name", type=str, default="linucb")
    ap.add_argument("--win-bonus", type=float, default=0.5)

    ap.add_argument("--outdir", type=str, default=os.path.join(_ROOT, "results", "main"))
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # 1) load agents
    agents = load_openrouter_agents_from_dir(args.agent_config_dir)
    print(f"[main] loaded agents: {list(agents.keys())}")

    # 2) import symphony + init global orchestrator
    import symphony as sym

    # symphony owns selector; init config here
    sym.init(
        use_dynamic=bool(args.use_dynamic),
        topL=int(args.topL),
        plan_k=int(args.plan_k),
        linucb_alpha=float(args.ucb_alpha),
        linucb_l2=float(args.ucb_l2),
        delta=float(args.ucb_delta),
        S=float(args.ucb_S),
    )

    # 3) register agents into symphony
    for a in agents.values():
        sym.register_agent(a)

    # 4) (reserved) cold start placeholder
    # maybe_cold_start(...)

    # 5) load tasks + sample n
    all_tasks = load_jsonl(args.task_pool)
    if args.n < len(all_tasks):
        tasks = rng.sample(all_tasks, k=int(args.n))
    else:
        tasks = all_tasks

    # 6) (optional) external selector placeholder
    # NOTE: not used for orchestration now; symphony uses internal selector.
    selector: Optional[GlobalLinUCB] = None

    # 7) run
    run_tasks_with_symphony(
        tasks=tasks,
        symphony_module=sym,
        selector=selector,
        policy_name=str(args.policy_name),
        win_bonus=float(args.win_bonus),
        outdir=args.outdir,
        cot_count=int(args.cot_count),
        plan_k=int(args.plan_k),
    )


if __name__ == "__main__":
    main()

