#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_routing.py

Purpose
- From Symphony pretrain jsonl logs (pretrain_cold_start.jsonl / pretrain_pretrain.jsonl / pretrain_test.jsonl),
  analyze whether routing "matched" the most suitable agent for each task type.

Key outputs (written to --outdir)
1) per_type_per_agent.csv
   - For each (phase, benchmark, task_type, difficulty_bin, agent): n, acc
2) best_agent_by_type.csv
   - For each (phase, benchmark, task_type, difficulty_bin): oracle_best_agent, oracle_acc, n_oracle
3) routing_match_summary.csv
   - For each (phase, benchmark, task_type, difficulty_bin):
     router_acc, oracle_acc, regret, match_rate, n
4) routing_match_by_agent.csv
   - For each (phase, benchmark): chosen_count per agent + overall acc
5) summary.md
   - Human-readable Markdown summary with tables and insights
6) (optional) prints top-k worst-regret types to stdout.

How to run
- Single file:
    python3 analyze_routing.py --jsonl path/to/pretrain_test.jsonl --outdir analysis_out
- Directory (auto-detect pretrain_*.jsonl files):
    python3 analyze_routing.py --dir path/to/outdir --outdir path/to/outdir/analysis

Notes
- We treat the "chosen agent" as agent_ids[0] when agent_ids is a list.
- "Oracle best agent" per type is the agent with highest empirical accuracy within that type
  (subject to --min-n-per-agent).
"""

from __future__ import annotations

import os
import json
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional
import csv


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                # skip malformed line
                continue
    return rows


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _pick_chosen_agent(row: Dict[str, Any]) -> str:
    """
    Your logger stores agent_ids as list (often).
    Use agent_ids[0] as chosen executor. Fallback to node_id if present.
    """
    aids = row.get("agent_ids")
    if isinstance(aids, list) and aids:
        a0 = _safe_str(aids[0]).strip()
        return a0 or "NA"
    if isinstance(aids, str) and aids.strip():
        return aids.strip()
    # sometimes you also output node_id in csv summary but not in jsonl
    node_id = row.get("node_id")
    if isinstance(node_id, str) and node_id.strip():
        return node_id.strip()
    return "NA"


def _phase(row: Dict[str, Any]) -> str:
    return _safe_str(row.get("phase") or "unknown").strip() or "unknown"


def _bench(row: Dict[str, Any]) -> str:
    return _safe_str(row.get("benchmark") or "unknown").strip().lower() or "unknown"


def _task_type(row: Dict[str, Any]) -> str:
    # your code uses task_type field (BBH task_name), else unknown
    return _safe_str(row.get("task_type") or "unknown").strip() or "unknown"


def _difficulty(row: Dict[str, Any], use: bool) -> str:
    if not use:
        return "__all__"
    return _safe_str(row.get("difficulty_bin") or "unknown").strip() or "unknown"


def _acc(row: Dict[str, Any]) -> int:
    try:
        return int(row.get("acc") or 0)
    except Exception:
        return 0


def _key(row: Dict[str, Any], use_difficulty: bool) -> Tuple[str, str, str, str]:
    return (_phase(row), _bench(row), _task_type(row), _difficulty(row, use_difficulty))


def _mean(xs: List[int]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / float(len(xs))


def _write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_markdown_summary(
    path: str,
    match_rows: List[Dict[str, Any]],
    by_agent_rows: List[Dict[str, Any]],
    best_agent_rows: List[Dict[str, Any]],
    worst_regrets: List[Tuple[float, Dict[str, Any]]],
) -> None:
    """Write a human-readable Markdown summary."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Routing Match Analysis Summary\n\n")
        f.write("This report analyzes whether Symphony's routing matched the most suitable agent for each task type.\n\n")
        f.write("## Methodology\n\n")
        f.write("- **Oracle Best Agent**: The agent with highest empirical accuracy within each task type (minimum samples required: configurable)\n")
        f.write("- **Router Accuracy**: Actual accuracy achieved by the routing system\n")
        f.write("- **Regret**: `oracle_acc - router_acc` (how much better the oracle would have been)\n")
        f.write("- **Match Rate**: Fraction of tasks where the router selected the oracle-best agent\n\n")
        f.write("---\n\n")

        # Overall statistics by phase and benchmark
        f.write("## Overall Statistics by Phase & Benchmark\n\n")
        phase_bench_stats: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(lambda: {"n": 0, "router_acc_sum": 0.0, "oracle_acc_sum": 0.0, "regret_sum": 0.0, "match_count": 0})
        
        for row in match_rows:
            phase = row["phase"]
            bench = row["benchmark"]
            k = (phase, bench)
            try:
                n = int(row.get("n", 0))
                router_acc = float(row.get("router_acc", 0.0))
                oracle_acc = float(row.get("oracle_acc", 0.0))
                regret = float(row.get("regret", 0.0))
                match_rate = float(row.get("match_rate", 0.0))
            except Exception:
                continue
            
            phase_bench_stats[k]["n"] += n
            phase_bench_stats[k]["router_acc_sum"] += router_acc * n
            phase_bench_stats[k]["oracle_acc_sum"] += oracle_acc * n
            phase_bench_stats[k]["regret_sum"] += regret * n
            phase_bench_stats[k]["match_count"] += match_rate * n

        f.write("| Phase | Benchmark | Total Tasks | Router Acc | Oracle Acc | Avg Regret | Match Rate |\n")
        f.write("|-------|-----------|-------------|------------|------------|------------|------------|\n")
        for (phase, bench), stats in sorted(phase_bench_stats.items()):
            n = stats["n"]
            if n == 0:
                continue
            router_acc = stats["router_acc_sum"] / n
            oracle_acc = stats["oracle_acc_sum"] / n
            avg_regret = stats["regret_sum"] / n
            match_rate = stats["match_count"] / n
            f.write(f"| {phase} | {bench} | {n} | {router_acc:.3f} | {oracle_acc:.3f} | {avg_regret:.3f} | {match_rate:.1%} |\n")
        f.write("\n")

        # Agent selection statistics
        f.write("## Agent Selection Statistics\n\n")
        f.write("| Phase | Benchmark | Agent | Chosen Count | Acc When Chosen |\n")
        f.write("|-------|-----------|-------|---------------|------------------|\n")
        for row in by_agent_rows:
            if row.get("agent") == "__overall__":
                continue
            phase = row["phase"]
            bench = row["benchmark"]
            agent = row["agent"]
            chosen_n = row["chosen_n"]
            acc = row["acc_when_chosen"]
            f.write(f"| {phase} | {bench} | {agent} | {chosen_n} | {acc} |\n")
        f.write("\n")

        # Top worst-regret task types
        f.write("## Top 20 Worst-Regret Task Types\n\n")
        f.write("These are task types where the router's choice deviated most from the oracle-best agent.\n\n")
        f.write("| Phase | Benchmark | Task Type | Difficulty | N | Router Acc | Oracle Agent | Oracle Acc | Regret | Match Rate |\n")
        f.write("|-------|-----------|-----------|------------|---|------------|--------------|------------|--------|------------|\n")
        shown = 0
        for regret, row in worst_regrets:
            if shown >= 20:
                break
            try:
                n = int(row.get("n", 0))
            except Exception:
                n = 0
            if n <= 0:
                continue
            phase = row["phase"]
            bench = row["benchmark"]
            ttype = row["task_type"]
            diff = row["difficulty_bin"]
            router_acc = row["router_acc"]
            oracle_agent = row["oracle_best_agent"]
            oracle_acc = row["oracle_acc"]
            regret_str = row["regret"]
            match_rate = row["match_rate"]
            f.write(f"| {phase} | {bench} | {ttype} | {diff} | {n} | {router_acc} | {oracle_agent} | {oracle_acc} | {regret_str} | {match_rate} |\n")
            shown += 1
        f.write("\n")

        # Best-performing task types (low regret or high match rate)
        f.write("## Best-Matched Task Types (Top 15 by Match Rate)\n\n")
        f.write("Task types where the router frequently selected the oracle-best agent.\n\n")
        best_matched = sorted(match_rows, key=lambda r: float(r.get("match_rate", 0.0)), reverse=True)
        f.write("| Phase | Benchmark | Task Type | Difficulty | N | Router Acc | Oracle Agent | Oracle Acc | Regret | Match Rate |\n")
        f.write("|-------|-----------|-----------|------------|---|------------|--------------|------------|--------|------------|\n")
        shown = 0
        for row in best_matched:
            if shown >= 15:
                break
            try:
                n = int(row.get("n", 0))
            except Exception:
                n = 0
            if n <= 0:
                continue
            phase = row["phase"]
            bench = row["benchmark"]
            ttype = row["task_type"]
            diff = row["difficulty_bin"]
            router_acc = row["router_acc"]
            oracle_agent = row["oracle_best_agent"]
            oracle_acc = row["oracle_acc"]
            regret = row["regret"]
            match_rate = row["match_rate"]
            f.write(f"| {phase} | {bench} | {ttype} | {diff} | {n} | {router_acc} | {oracle_agent} | {oracle_acc} | {regret} | {match_rate} |\n")
            shown += 1
        f.write("\n")

        # Oracle best agents by type
        f.write("## Oracle Best Agents by Task Type\n\n")
        f.write("| Phase | Benchmark | Task Type | Difficulty | Oracle Best Agent | Oracle Acc | N |\n")
        f.write("|-------|-----------|-----------|------------|-------------------|------------|---|\n")
        for row in sorted(best_agent_rows, key=lambda r: (r["phase"], r["benchmark"], r["task_type"], r["difficulty_bin"])):
            phase = row["phase"]
            bench = row["benchmark"]
            ttype = row["task_type"]
            diff = row["difficulty_bin"]
            oracle_agent = row["oracle_best_agent"]
            oracle_acc = row["oracle_acc"]
            n_oracle = row["n_oracle"]
            f.write(f"| {phase} | {bench} | {ttype} | {diff} | {oracle_agent} | {oracle_acc} | {n_oracle} |\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("*Generated by analyze_routing.py*\n")


def _collect_jsonl_files_from_dir(d: str) -> List[str]:
    cand = []
    if not os.path.isdir(d):
        return cand
    for name in os.listdir(d):
        if name.startswith("pretrain_") and name.endswith(".jsonl"):
            cand.append(os.path.join(d, name))
    # also support nested timestamped names if you have them
    cand.sort()
    return cand


def analyze(rows: List[Dict[str, Any]], min_n_per_agent: int, use_difficulty: bool) -> Dict[str, Any]:
    """
    Returns a dict of computed tables (as list-of-dicts).
    """
    # per_type_agent_acc[(phase,bench,type,diff)][agent] -> [accs]
    per_type_agent_acc: Dict[Tuple[str, str, str, str], Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    # router_acc[(phase,bench,type,diff)] -> [accs]
    router_acc: Dict[Tuple[str, str, str, str], List[int]] = defaultdict(list)
    # router_chosen[(phase,bench,type,diff)] -> [chosen_agent]
    router_chosen: Dict[Tuple[str, str, str, str], List[str]] = defaultdict(list)
    # overall chosen stats per (phase, bench) by agent
    chosen_agent_acc: Dict[Tuple[str, str], Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

    for r in rows:
        k = _key(r, use_difficulty)
        a = _pick_chosen_agent(r)
        acc = _acc(r)

        per_type_agent_acc[k][a].append(acc)
        router_acc[k].append(acc)
        router_chosen[k].append(a)
        chosen_agent_acc[(_phase(r), _bench(r))][a].append(acc)

    # 1) per_type_per_agent.csv rows
    per_type_per_agent_rows: List[Dict[str, Any]] = []
    for k, agent_map in per_type_agent_acc.items():
        phase, bench, ttype, diff = k
        for agent, accs in agent_map.items():
            n = len(accs)
            per_type_per_agent_rows.append(
                {
                    "phase": phase,
                    "benchmark": bench,
                    "task_type": ttype,
                    "difficulty_bin": diff,
                    "agent": agent,
                    "n": n,
                    "acc": f"{_mean(accs):.6f}",
                }
            )

    # 2) best_agent_by_type (oracle best) per type
    best_agent_by_type: Dict[Tuple[str, str, str, str], Tuple[str, float, int]] = {}
    best_agent_rows: List[Dict[str, Any]] = []

    for k, agent_map in per_type_agent_acc.items():
        phase, bench, ttype, diff = k
        best_agent = ""
        best_acc = -1.0
        best_n = 0

        for agent, accs in agent_map.items():
            if agent == "NA":
                continue
            n = len(accs)
            if n < min_n_per_agent:
                continue
            a = _mean(accs)
            # tie-break: higher n wins
            if a > best_acc or (abs(a - best_acc) < 1e-12 and n > best_n):
                best_agent, best_acc, best_n = agent, a, n

        if best_agent:
            best_agent_by_type[k] = (best_agent, best_acc, best_n)
            best_agent_rows.append(
                {
                    "phase": phase,
                    "benchmark": bench,
                    "task_type": ttype,
                    "difficulty_bin": diff,
                    "oracle_best_agent": best_agent,
                    "oracle_acc": f"{best_acc:.6f}",
                    "n_oracle": best_n,
                }
            )

    # 3) routing_match_summary: match_rate, router_acc, oracle_acc, regret
    match_rows: List[Dict[str, Any]] = []
    worst_regrets: List[Tuple[float, Dict[str, Any]]] = []

    for k, accs in router_acc.items():
        if k not in best_agent_by_type:
            continue
        phase, bench, ttype, diff = k
        oracle_agent, oracle_acc, oracle_n = best_agent_by_type[k]
        chosen_list = router_chosen.get(k, [])
        n = len(accs)
        router_a = _mean(accs)

        match = 0
        for a in chosen_list:
            if a == oracle_agent:
                match += 1
        match_rate = (match / n) if n else 0.0
        regret = oracle_acc - router_a

        row = {
            "phase": phase,
            "benchmark": bench,
            "task_type": ttype,
            "difficulty_bin": diff,
            "n": n,
            "router_acc": f"{router_a:.6f}",
            "oracle_best_agent": oracle_agent,
            "oracle_acc": f"{oracle_acc:.6f}",
            "regret": f"{regret:.6f}",
            "match_rate": f"{match_rate:.6f}",
        }
        match_rows.append(row)
        worst_regrets.append((regret, row))

    # sort match_rows by regret desc (largest regret first)
    match_rows.sort(key=lambda r: float(r["regret"]), reverse=True)
    worst_regrets.sort(key=lambda x: x[0], reverse=True)

    # 4) routing_match_by_agent: overall agent selection counts and acc
    by_agent_rows: List[Dict[str, Any]] = []
    for (phase, bench), amap in chosen_agent_acc.items():
        # overall
        total_n = 0
        total_ok = 0
        # per agent
        for agent, accs in sorted(amap.items(), key=lambda x: (-len(x[1]), x[0])):
            n = len(accs)
            a = _mean(accs)
            by_agent_rows.append(
                {
                    "phase": phase,
                    "benchmark": bench,
                    "agent": agent,
                    "chosen_n": n,
                    "acc_when_chosen": f"{a:.6f}",
                }
            )
            total_n += n
            total_ok += sum(accs)
        by_agent_rows.append(
            {
                "phase": phase,
                "benchmark": bench,
                "agent": "__overall__",
                "chosen_n": total_n,
                "acc_when_chosen": f"{(total_ok / total_n) if total_n else 0.0:.6f}",
            }
        )

    return {
        "per_type_per_agent_rows": per_type_per_agent_rows,
        "best_agent_rows": best_agent_rows,
        "match_rows": match_rows,
        "by_agent_rows": by_agent_rows,
        "worst_regrets": worst_regrets,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, default="", help="Path to a single pretrain_*.jsonl file")
    ap.add_argument("--dir", type=str, default="", help="Directory containing pretrain_*.jsonl files")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory for analysis CSVs")
    ap.add_argument("--min-n-per-agent", type=int, default=10, help="Min samples per agent within a type to consider oracle-best")
    ap.add_argument("--use-difficulty", action="store_true", help="Stratify by difficulty_bin as well")
    ap.add_argument("--topk", type=int, default=25, help="Print top-k worst regret types")
    args = ap.parse_args()

    files: List[str] = []
    if args.jsonl:
        files = [args.jsonl]
    elif args.dir:
        files = _collect_jsonl_files_from_dir(args.dir)
        if not files:
            raise SystemExit(f"No pretrain_*.jsonl found in dir: {args.dir}")
    else:
        raise SystemExit("Provide either --jsonl or --dir")

    os.makedirs(args.outdir, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    for fp in files:
        if not os.path.exists(fp):
            continue
        all_rows.extend(_read_jsonl(fp))

    if not all_rows:
        raise SystemExit("No rows loaded from jsonl files.")

    res = analyze(
        all_rows,
        min_n_per_agent=int(args.min_n_per_agent),
        use_difficulty=bool(args.use_difficulty),
    )

    # Write CSVs
    _write_csv(
        os.path.join(args.outdir, "per_type_per_agent.csv"),
        ["phase", "benchmark", "task_type", "difficulty_bin", "agent", "n", "acc"],
        res["per_type_per_agent_rows"],
    )
    _write_csv(
        os.path.join(args.outdir, "best_agent_by_type.csv"),
        ["phase", "benchmark", "task_type", "difficulty_bin", "oracle_best_agent", "oracle_acc", "n_oracle"],
        res["best_agent_rows"],
    )
    _write_csv(
        os.path.join(args.outdir, "routing_match_summary.csv"),
        ["phase", "benchmark", "task_type", "difficulty_bin", "n", "router_acc", "oracle_best_agent", "oracle_acc", "regret", "match_rate"],
        res["match_rows"],
    )
    _write_csv(
        os.path.join(args.outdir, "routing_match_by_agent.csv"),
        ["phase", "benchmark", "agent", "chosen_n", "acc_when_chosen"],
        res["by_agent_rows"],
    )

    # Write Markdown summary
    _write_markdown_summary(
        os.path.join(args.outdir, "summary.md"),
        res["match_rows"],
        res["by_agent_rows"],
        res["best_agent_rows"],
        res["worst_regrets"],
    )

    # Print top-k worst regrets
    print("\n=== Top worst-regret (oracle_acc - router_acc) types ===")
    k = max(0, int(args.topk))
    shown = 0
    for regret, row in res["worst_regrets"]:
        if shown >= k:
            break
        # Skip tiny n
        try:
            n = int(row.get("n") or 0)
        except Exception:
            n = 0
        if n <= 0:
            continue
        print(
            f"[{row['phase']}][{row['benchmark']}][{row['task_type']}][{row['difficulty_bin']}] "
            f"n={row['n']} router_acc={row['router_acc']} oracle={row['oracle_best_agent']}({row['oracle_acc']}) "
            f"match_rate={row['match_rate']} regret={row['regret']}"
        )
        shown += 1

    print(f"\n[OK] Wrote analysis CSVs and summary.md to: {args.outdir}")


if __name__ == "__main__":
    main()
