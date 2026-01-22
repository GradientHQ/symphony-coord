#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def _read_tasks(
    path: str,
    line_start: Optional[int],
    line_end: Optional[int],
    benchmark_filter: Optional[str],
) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line_start is not None and line_no < line_start:
                continue
            if line_end is not None and line_no > line_end:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if benchmark_filter and obj.get("benchmark") != benchmark_filter:
                continue
            tasks.append(obj)
    return tasks


def _stratified_sample_bbh(
    tasks: List[Dict[str, object]],
    n: int,
    seed: int,
) -> List[Dict[str, object]]:
    if n <= 0 or not tasks:
        return tasks
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for t in tasks:
        meta = t.get("scorer_metadata") or {}
        name = str(meta.get("task_name") or "").strip() or "unknown"
        groups[name].append(t)
    if not groups:
        return tasks

    rng = random.Random(seed)
    keys = list(groups.keys())
    rng.shuffle(keys)

    base = n // len(keys)
    remainder = n % len(keys)
    alloc: Dict[str, int] = {k: base for k in keys}
    for k in keys[:remainder]:
        alloc[k] += 1

    selected: List[Dict[str, object]] = []
    deficit = 0
    capacity: Dict[str, int] = {}
    for k in keys:
        pool = list(groups[k])
        rng.shuffle(pool)
        take = min(len(pool), alloc[k])
        selected.extend(pool[:take])
        if take < alloc[k]:
            deficit += (alloc[k] - take)
        capacity[k] = max(0, len(pool) - take)

    if deficit > 0:
        fill_keys = [k for k in keys if capacity.get(k, 0) > 0]
        rng.shuffle(fill_keys)
        idx = 0
        while deficit > 0 and fill_keys:
            k = fill_keys[idx % len(fill_keys)]
            if capacity[k] <= 0:
                idx += 1
                continue
            pool = groups[k]
            start = len(pool) - capacity[k]
            selected.append(pool[start])
            capacity[k] -= 1
            deficit -= 1
            idx += 1

    return selected


def _balance_across_benchmarks(
    tasks: List[Dict[str, object]],
    n: Optional[int],
    seed: int,
) -> List[Dict[str, object]]:
    if not tasks:
        return tasks

    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for t in tasks:
        bench = str(t.get("benchmark") or "unknown").strip() or "unknown"
        groups[bench].append(t)

    bench_keys = list(groups.keys())
    if len(bench_keys) <= 1:
        return tasks

    rng = random.Random(seed)
    rng.shuffle(bench_keys)

    if n is None or n <= 0:
        min_count = min(len(groups[k]) for k in bench_keys)
        n_target = min_count * len(bench_keys)
    else:
        n_target = min(int(n), len(tasks))

    base = n_target // len(bench_keys)
    remainder = n_target % len(bench_keys)

    alloc: Dict[str, int] = {}
    deficit = 0
    capacity: Dict[str, int] = {}
    for i, k in enumerate(bench_keys):
        want = base + (1 if i < remainder else 0)
        cap = len(groups[k])
        if want > cap:
            deficit += want - cap
            want = cap
        alloc[k] = want
        capacity[k] = max(0, cap - want)

    if deficit > 0:
        fill_keys = [k for k in bench_keys if capacity.get(k, 0) > 0]
        rng.shuffle(fill_keys)
        idx = 0
        while deficit > 0 and fill_keys:
            k = fill_keys[idx % len(fill_keys)]
            if capacity[k] <= 0:
                idx += 1
                continue
            alloc[k] += 1
            capacity[k] -= 1
            deficit -= 1
            idx += 1

    selected: List[Dict[str, object]] = []
    for i, k in enumerate(bench_keys):
        pool = list(groups[k])
        take = alloc.get(k, 0)
        if take <= 0:
            continue
        if k == "bbh":
            chosen = _stratified_sample_bbh(pool, take, seed + 17 + i)
        else:
            rng_k = random.Random(seed + 31 + i)
            chosen = pool if take >= len(pool) else rng_k.sample(pool, k=take)
        selected.extend(chosen)

    rng.shuffle(selected)
    return selected


def _count_benchmarks(tasks: List[Dict[str, object]]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for t in tasks:
        bench = str(t.get("benchmark") or "unknown").strip() or "unknown"
        counts[bench] += 1
    return dict(counts)


def _write_jsonl(tasks: List[Dict[str, object]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a balanced task_pool.jsonl with per-benchmark averaging.",
    )
    parser.add_argument("--task-pool", required=True, help="Path to input task_pool.jsonl")
    parser.add_argument("--out", required=True, help="Path to output JSONL")
    parser.add_argument("--line-start", type=int, default=None, help="1-based line start (inclusive)")
    parser.add_argument("--line-end", type=int, default=None, help="1-based line end (inclusive)")
    parser.add_argument("--n", type=int, default=None, help="Total tasks to output (optional)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--benchmark", type=str, default=None, help="Filter to a single benchmark")
    args = parser.parse_args()

    tasks = _read_tasks(
        args.task_pool,
        line_start=args.line_start,
        line_end=args.line_end,
        benchmark_filter=args.benchmark,
    )

    if args.benchmark:
        rng = random.Random(args.seed)
        if args.n is None or args.n <= 0 or args.n >= len(tasks):
            selected = tasks
        elif args.benchmark.strip().lower() == "bbh":
            selected = _stratified_sample_bbh(tasks, args.n, args.seed)
        else:
            selected = rng.sample(tasks, k=args.n)
    else:
        selected = _balance_across_benchmarks(tasks, n=args.n, seed=args.seed)

    _write_jsonl(selected, args.out)

    counts = _count_benchmarks(selected)
    total = sum(counts.values())
    print(f"[balanced_task_pool] saved={args.out} total={total}")
    print(f"[balanced_task_pool] benchmarks={counts}")


if __name__ == "__main__":
    main()
