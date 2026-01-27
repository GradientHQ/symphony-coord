#!/usr/bin/env python3
"""
Sanity Check Radar: 4-axis 0–1 normalized health scores + optional companion plots.

Each axis (1 = healthy, 0 = abnormal):
  1. Weight normalization health   — routing weights sum ≈ 1 per decision (fallback: 1.0 if no weights)
  2. Task-type coverage balance    — test set balance vs target or entropy / log(K)
  3. Appropriate match             — match-rate: tasks assigned to the most appropriate agent for that task_type (not necessarily "strongest")
  4. Trajectory smoothness         — cold→train→test selection stability (JSD of adjacent windows)

Usage:
  python sanity_check_radar.py --dir <result_dir> [--dir <dir2> ...] [--labels "A,B,C"]
  python sanity_check_radar.py --dir <result_dir> --companion
  python sanity_check_radar.py -d run1 -d run2 -d run3 -l "Random,UCB-cold,UCB-warm2" -o ./figs

Data:
  Requires: accuracy_summary.csv per result dir (e.g. from pretrain.py).
  Optional: phase_stats.json, plan_weight_sum_overall.json (e.g. from select_only_stats) for axis 1.

Companion plots (--companion): task-type distribution bar chart + task_type × agent accuracy heatmap.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from scipy.spatial.distance import jensenshannon as _scipy_js
except ImportError:
    _scipy_js = None


def _jensenshannon(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon distance (sqrt of JSD), [0, 1]. Use scipy if available else numpy fallback."""
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    n = max(len(p), len(q))
    if n == 0:
        return 0.0
    pp = np.zeros(n)
    qq = np.zeros(n)
    pp[: len(p)] = p
    qq[: len(q)] = q
    if pp.sum() <= 0:
        pp = np.ones(n) / n
    else:
        pp = pp / pp.sum()
    if qq.sum() <= 0:
        qq = np.ones(n) / n
    else:
        qq = qq / qq.sum()
    if _scipy_js is not None:
        try:
            return float(_scipy_js(pp, qq))
        except Exception:
            pass
    # numpy fallback: JS = (KL(p||m) + KL(q||m))/2, m = (p+q)/2; distance = sqrt(JS)
    eps = 1e-12
    m = (pp + qq) / 2
    kl_pm = np.sum(pp * (np.log(pp + eps) - np.log(m + eps)))
    kl_qm = np.sum(qq * (np.log(qq + eps) - np.log(m + eps)))
    js = 0.5 * (kl_pm + kl_qm)
    return float(np.sqrt(min(1.0, max(0.0, js))))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _infer_task_type(task_id: str) -> str:
    """Infer task_type from task_id (e.g. gsm8k_1309 -> math, bbh_193 -> reasoning)."""
    s = (task_id or "").strip().lower()
    if not s or "_" not in s:
        return "other"
    prefix = s.split("_")[0]
    if prefix in ("gsm8k", "gsm", "math"):
        return "math"
    if prefix in ("medical_qa", "medical", "medqa"):
        return "medical"
    if prefix == "bbh":
        return "reasoning"
    return prefix if prefix else "other"


def _load_accuracy_summary(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            row["_index"] = i
            rows.append(row)
    return rows


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _safe_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon distance (sqrt of JSD), in [0, 1]. Uses _jensenshannon (scipy or numpy)."""
    try:
        return _jensenshannon(p, q)
    except Exception:
        return 0.0


# -----------------------------------------------------------------------------
# Axis 1: Weight normalization health
# -----------------------------------------------------------------------------


def score_weight_normalization(
    result_dir: str,
    phase_stats: Optional[Dict[str, Any]],
    plan_weight: Optional[Dict[str, Any]],
    *,
    tau: float = 0.05,
) -> float:
    """
    Score in [0, 1]. 1 = healthy (weights sum ≈ 1 per decision).
    Without per-step weights we use 选择频率分布 fallback: each pick is one-hot (sum=1) -> 1.0.
    With phase_stats/plan_weight, optionally use plan_parse_fail as proxy for "recording health".
    """
    # If you add per-step weights (e.g. weights/topL_probs in logs), compute:
    #   S = sum(weights), e = |S - 1|, E_phase = mean(e), score = max(0, 1 - E_phase / tau)
    if not phase_stats and not plan_weight:
        return 1.0  # Selection-frequency fallback: one pick per task -> sum=1
    plan_weight = plan_weight or {}
    fail = plan_weight.get("plan_parse_fail", 0)
    total = plan_weight.get("plan_count", 0) + fail
    if total <= 0:
        return 1.0
    fail_ratio = fail / total
    return max(0.0, 1.0 - fail_ratio / tau) if tau > 0 else 1.0


# -----------------------------------------------------------------------------
# Axis 2: Task-type coverage balance
# -----------------------------------------------------------------------------


def score_task_type_coverage(
    rows: List[Dict[str, Any]],
    *,
    use_test_only: bool = True,
    target_dist: Optional[Dict[str, float]] = None,
) -> float:
    """
    Score in [0, 1]. With target: score = 1 - JSD(empirical || target).
    Without target: score = H / log(K) (entropy / max entropy).
    """
    if not rows:
        return 0.0
    subset = [r for r in rows if str(r.get("phase", "")).strip().lower() == "test"] if use_test_only else rows
    if not subset:
        subset = rows
    type_counts: Dict[str, int] = defaultdict(int)
    for r in subset:
        t = _infer_task_type(str(r.get("task_id", "")))
        type_counts[t] += 1
    total = sum(type_counts.values())
    if total <= 0:
        return 0.0
    types = sorted(type_counts.keys())
    K = len(types)
    if K == 0:
        return 0.0
    if K == 1 and target_dist is None:
        return 1.0  # No imbalance possible when no target.

    if target_dist is not None:
        types = sorted(set(types) | set(target_dist.keys()))
        K = len(types)
        p_emp = np.array([type_counts.get(t, 0) / total for t in types])
        q = np.array([target_dist.get(t, 0.0) for t in types])
        if q.sum() <= 0:
            q = np.ones(K) / K
        else:
            q = q / q.sum()
        js = _safe_jsd(p_emp, q)
        return max(0.0, 1.0 - js)

    p_emp = np.array([type_counts[t] / total for t in types])
    # Entropy: H = -sum p log p, max = log(K)
    eps = 1e-12
    H = -float(np.sum(p_emp * np.log(p_emp + eps)))
    H_max = math.log(K)
    if H_max <= 0:
        return 1.0
    return min(1.0, H / H_max)


# -----------------------------------------------------------------------------
# Axis 3: Appropriate match (task–agent fit)
# -----------------------------------------------------------------------------


def score_top1_alignment(
    rows: List[Dict[str, Any]],
    *,
    use_test_only: bool = True,
) -> float:
    """
    Match rate: fraction of test tasks where the selected agent is the most *appropriate*
    for that task_type. Currently operationalized as argmax_a acc[a, task_type] (accuracy
    as proxy for suitability); can be extended to other notions of appropriateness.
    Score in [0, 1].
    """
    subset = [r for r in rows if str(r.get("phase", "")).strip().lower() == "test"] if use_test_only else rows
    if not subset:
        subset = rows
    if not subset:
        return 0.0

    # Build acc[agent, task_type] from same subset (could use train for acc in future)
    ag_tp_correct: Dict[Tuple[str, str], int] = defaultdict(int)
    ag_tp_total: Dict[Tuple[str, str], int] = defaultdict(int)
    for r in subset:
        node_id = (r.get("node_id") or "").strip()
        agents = [a.strip() for a in node_id.split(",") if a.strip()]
        agent = agents[0] if agents else ""
        task_type = _infer_task_type(str(r.get("task_id", "")))
        acc = int(r.get("acc") or 0)
        key = (agent, task_type)
        ag_tp_correct[key] += acc
        ag_tp_total[key] += 1

    # Per (agent, task_type) accuracy
    acc_at: Dict[Tuple[str, str], float] = {}
    for k, n in ag_tp_total.items():
        acc_at[k] = ag_tp_correct[k] / max(1, n)

    # Most appropriate agent per task_type (current proxy: highest acc)
    by_type: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for (a, t), v in acc_at.items():
        by_type[t].append((a, v))
    appropriate_per_type: Dict[str, str] = {}
    for t, lst in by_type.items():
        if lst:
            appropriate_per_type[t] = max(lst, key=lambda x: x[1])[0]

    aligned = 0
    for r in subset:
        node_id = (r.get("node_id") or "").strip()
        agents = [a.strip() for a in node_id.split(",") if a.strip()]
        agent = agents[0] if agents else ""
        task_type = _infer_task_type(str(r.get("task_id", "")))
        apro = appropriate_per_type.get(task_type)
        if apro is not None and agent == apro:
            aligned += 1
    return aligned / max(1, len(subset))


# -----------------------------------------------------------------------------
# Axis 4: Trajectory smoothness
# -----------------------------------------------------------------------------


def score_trajectory_smoothness(
    rows: List[Dict[str, Any]],
    *,
    window_size: int = 50,
    tau: float = 0.2,
) -> float:
    """
    Stability of agent selection over time. Bin by task index, p_t(agent) per bin,
    Delta_t = JSD(p_t || p_{t-1}), score = max(0, 1 - mean(Delta) / tau).
    """
    if not rows or tau <= 0:
        return 0.0
    # Sort by (phase order, then _index)
    phase_order = {"cold_start": 0, "pretrain": 1, "train": 1, "test": 2}
    def key(r):
        p = str(r.get("phase", "")).strip().lower()
        return (phase_order.get(p, 1), r.get("_index", 0))
    sorted_rows = sorted(rows, key=key)

    agent_set: set = set()
    for r in sorted_rows:
        node_id = (r.get("node_id") or "").strip()
        for a in node_id.split(","):
            a = a.strip()
            if a:
                agent_set.add(a)
    agents = sorted(agent_set)
    if not agents:
        return 1.0
    K = len(agents)
    aidx = {a: i for i, a in enumerate(agents)}

    distributions: List[np.ndarray] = []
    for i in range(0, len(sorted_rows), window_size):
        chunk = sorted_rows[i : i + window_size]
        cnt = np.zeros(K)
        for r in chunk:
            node_id = (r.get("node_id") or "").strip()
            for a in node_id.split(","):
                a = a.strip()
                if a in aidx:
                    cnt[aidx[a]] += 1
                    break
        s = cnt.sum()
        if s <= 0:
            cnt = np.ones(K) / K
        else:
            cnt = cnt / s
        distributions.append(cnt)

    if len(distributions) < 2:
        return 1.0
    deltas = []
    for i in range(1, len(distributions)):
        d = _safe_jsd(distributions[i - 1], distributions[i])
        deltas.append(d)
    mean_d = np.mean(deltas) if deltas else 0.0
    return max(0.0, 1.0 - mean_d / tau)


# -----------------------------------------------------------------------------
# Aggregate scores per result dir
# -----------------------------------------------------------------------------


def compute_scores(
    result_dir: str,
    *,
    tau_weight: float = 0.05,
    tau_smooth: float = 0.2,
    window_size: int = 50,
    target_dist: Optional[Dict[str, float]] = None,
) -> Tuple[float, float, float, float]:
    csv_path = os.path.join(result_dir, "accuracy_summary.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"accuracy_summary.csv not found: {csv_path}")
    rows = _load_accuracy_summary(csv_path)
    phase_stats = _load_json(os.path.join(result_dir, "phase_stats.json"))
    plan_weight = _load_json(os.path.join(result_dir, "plan_weight_sum_overall.json"))

    s1 = score_weight_normalization(result_dir, phase_stats, plan_weight, tau=tau_weight)
    s2 = score_task_type_coverage(rows, use_test_only=True, target_dist=target_dist)
    s3 = score_top1_alignment(rows, use_test_only=True)
    s4 = score_trajectory_smoothness(rows, window_size=window_size, tau=tau_smooth)
    return (s1, s2, s3, s4)


# -----------------------------------------------------------------------------
# Radar plot (4 axes, 0–1, multiple policies)
# -----------------------------------------------------------------------------

AXES = [
    "Weight norm.",
    "Task-type balance",
    "Appropriate match",
    "Trajectory smoothness",
]


def plot_radar(
    all_scores: List[Tuple[str, Tuple[float, float, float, float]]],
    outpath: str,
    *,
    title: str = "Sanity Check Radar (higher is better)",
) -> None:
    """Plot 4-axis radar, 0–1, one polygon per (label, scores)."""
    if plt is None:
        print("matplotlib not available, skip radar plot")
        return
    n_axes = len(AXES)
    angles = [2 * math.pi * i / n_axes for i in range(n_axes)]
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8), dpi=150, facecolor="white")
    ax = fig.add_subplot(111, polar=True, facecolor="#fafafa")
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AXES, fontsize=11, color="#333")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=9, color="#666")
    ax.grid(True, linestyle="--", linewidth=0.8, color="#ccc", alpha=0.6)

    colors = ["#e67e22", "#3498db", "#2ecc71", "#9b59b6", "#1abc9c"]
    for idx, (label, (s1, s2, s3, s4)) in enumerate(all_scores):
        vals = [s1, s2, s3, s4]
        vals += vals[:1]
        color = colors[idx % len(colors)]
        ax.plot(angles, vals, linewidth=2, color=color, label=label, marker="o", markersize=6)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0), fontsize=10, framealpha=0.95)
    ax.set_title(title, y=1.12, fontsize=13, fontweight="bold", color="#1f2937")
    fig.text(0.5, 0.02, "1 = healthy, 0 = abnormal", ha="center", fontsize=9, color="#666")
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight", facecolor="white", edgecolor="none", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Companion plots
# -----------------------------------------------------------------------------


def plot_task_type_distribution(
    rows: List[Dict[str, Any]],
    outpath: str,
    *,
    use_test_only: bool = True,
    title: str = "Task-type distribution (test set)",
) -> None:
    subset = [r for r in rows if str(r.get("phase", "")).strip().lower() == "test"] if use_test_only else rows
    if not subset:
        subset = rows
    type_counts: Dict[str, int] = defaultdict(int)
    for r in subset:
        t = _infer_task_type(str(r.get("task_id", "")))
        type_counts[t] += 1
    types = sorted(type_counts.keys())
    counts = [type_counts[t] for t in types]
    if not types:
        return
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150, facecolor="white")
    ax.bar(types, counts, color="#3498db", alpha=0.8, edgecolor="#2c3e50")
    ax.set_xlabel("Task type")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)


def plot_accuracy_heatmap(
    rows: List[Dict[str, Any]],
    outpath: str,
    *,
    use_test_only: bool = True,
    title: str = "Accuracy by task-type × agent (test set)",
) -> None:
    subset = [r for r in rows if str(r.get("phase", "")).strip().lower() == "test"] if use_test_only else rows
    if not subset:
        subset = rows
    ag_tp_correct: Dict[Tuple[str, str], int] = defaultdict(int)
    ag_tp_total: Dict[Tuple[str, str], int] = defaultdict(int)
    for r in subset:
        node_id = (r.get("node_id") or "").strip()
        agents = [a.strip() for a in node_id.split(",") if a.strip()]
        agent = agents[0] if agents else ""
        task_type = _infer_task_type(str(r.get("task_id", "")))
        acc = int(r.get("acc") or 0)
        k = (agent, task_type)
        ag_tp_correct[k] += acc
        ag_tp_total[k] += 1
    agents = sorted(set(a for a, _ in ag_tp_total.keys()))
    types = sorted(set(t for _, t in ag_tp_total.keys()))
    if not agents or not types:
        return
    M = np.zeros((len(types), len(agents)))
    for i, t in enumerate(types):
        for j, a in enumerate(agents):
            n = ag_tp_total.get((a, t), 0)
            M[i, j] = ag_tp_correct.get((a, t), 0) / max(1, n)
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(agents) * 0.8), max(4, len(types) * 0.5)), dpi=150, facecolor="white")
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels([a.split("-")[-1][:20] for a in agents], rotation=45, ha="right")
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types)
    ax.set_xlabel("Agent")
    ax.set_ylabel("Task type")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)


def _short_agent_label(agent_id: str, max_len: int = 16) -> str:
    """Short label for radar axes, e.g. deepseek-v3-0324-001 -> 001 or ds-001."""
    s = (agent_id or "").strip()
    if not s:
        return ""
    # use suffix like -001, -002
    if "-" in s:
        tail = s.split("-")[-1]
        if tail.isdigit() or (len(tail) >= 3 and tail[:3].isdigit()):
            return tail
    return s[:max_len] if len(s) > max_len else s


def plot_accuracy_radar(
    rows: List[Dict[str, Any]],
    outpath: str,
    *,
    use_test_only: bool = True,
    title: str = "Accuracy by task-type × agent (test set)",
) -> None:
    """
    Same data as accuracy heatmap, but as a radar chart.
    Axes = agents; one polygon per task type (values = acc per agent for that type).
    """
    subset = [r for r in rows if str(r.get("phase", "")).strip().lower() == "test"] if use_test_only else rows
    if not subset:
        subset = rows
    ag_tp_correct: Dict[Tuple[str, str], int] = defaultdict(int)
    ag_tp_total: Dict[Tuple[str, str], int] = defaultdict(int)
    for r in subset:
        node_id = (r.get("node_id") or "").strip()
        agents = [a.strip() for a in node_id.split(",") if a.strip()]
        agent = agents[0] if agents else ""
        task_type = _infer_task_type(str(r.get("task_id", "")))
        acc = int(r.get("acc") or 0)
        k = (agent, task_type)
        ag_tp_correct[k] += acc
        ag_tp_total[k] += 1
    agents = sorted(set(a for a, _ in ag_tp_total.keys()))
    types = sorted(set(t for _, t in ag_tp_total.keys()))
    if not agents or not types:
        return
    # M[type_idx, agent_idx] = acc
    M = np.zeros((len(types), len(agents)))
    for i, t in enumerate(types):
        for j, a in enumerate(agents):
            n = ag_tp_total.get((a, t), 0)
            M[i, j] = ag_tp_correct.get((a, t), 0) / max(1, n)
    if plt is None:
        return

    n = len(agents)
    angles = [2 * math.pi * i / n for i in range(n)]
    angles += angles[:1]
    labels = [_short_agent_label(a) for a in agents]

    fig = plt.figure(figsize=(8, 8), dpi=150, facecolor="white")
    ax = fig.add_subplot(111, polar=True, facecolor="#fafafa")
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color="#333")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=9, color="#666")
    ax.grid(True, linestyle="--", linewidth=0.8, color="#ccc", alpha=0.6)

    colors = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#1abc9c", "#e74c3c"]
    for idx, t in enumerate(types):
        vals = [M[idx, j] for j in range(n)]
        vals += vals[:1]
        color = colors[idx % len(colors)]
        ax.plot(angles, vals, linewidth=2, color=color, label=t, marker="o", markersize=6)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0), fontsize=10, framealpha=0.95)
    ax.set_title(title, y=1.12, fontsize=12, fontweight="bold", color="#1f2937")
    fig.text(0.5, 0.02, "Accuracy [0, 1] per agent", ha="center", fontsize=9, color="#666")
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight", facecolor="white", edgecolor="none", dpi=150)
    plt.close(fig)


def run_companion_plots(result_dir: str, outdir: str, rows: List[Dict[str, Any]]) -> None:
    base = os.path.join(outdir, "sanity")
    plot_task_type_distribution(
        rows,
        f"{base}_task_type_dist.png",
        use_test_only=True,
        title="Task-type distribution (test set)",
    )
    plot_accuracy_heatmap(
        rows,
        f"{base}_accuracy_heatmap.png",
        use_test_only=True,
        title="Accuracy by task-type × agent (test set)",
    )
    plot_accuracy_radar(
        rows,
        f"{base}_accuracy_radar.png",
        use_test_only=True,
        title="Accuracy by task-type × agent (test set)",
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Sanity Check Radar: 4-axis 0–1 health scores + optional companion plots")
    ap.add_argument("--dir", "-d", action="append", dest="dirs", default=[], help="Result dir(s); each must have accuracy_summary.csv")
    ap.add_argument("--labels", "-l", type=str, default=None, help="Comma-separated labels for each --dir (default: dir basename)")
    ap.add_argument("--out", "-o", type=str, default=None, help="Output directory for radar + companion (default: first --dir)")
    ap.add_argument("--tau-weight", type=float, default=0.05, help="Tolerance for weight norm. axis (default: 0.05)")
    ap.add_argument("--tau-smooth", type=float, default=0.2, help="Tolerance for trajectory smoothness (default: 0.2)")
    ap.add_argument("--window", type=int, default=50, help="Bin size for trajectory smoothness (default: 50)")
    ap.add_argument("--target-dist", type=str, default=None, help="Target task-type dist e.g. 'math:0.33,reasoning:0.33,medical:0.34' (optional; types from task_id prefix)")
    ap.add_argument("--companion", action="store_true", help="Also plot task-type bar chart and accuracy heatmap")
    ap.add_argument("--radar-only", action="store_true", help="Only plot radar, skip companion even if --companion")
    args = ap.parse_args()

    if not args.dirs:
        print("Provide at least one --dir (result directory with accuracy_summary.csv)")
        sys.exit(1)
    dirs = [os.path.abspath(d) for d in args.dirs]
    labels_raw = (args.labels or "").strip()
    if labels_raw:
        labels_list = [s.strip() for s in labels_raw.split(",") if s.strip()]
    else:
        labels_list = []
    while len(labels_list) < len(dirs):
        labels_list.append(os.path.basename(os.path.normpath(dirs[len(labels_list)])))
    labels_list = labels_list[: len(dirs)]

    target_dist: Optional[Dict[str, float]] = None
    if args.target_dist:
        target_dist = {}
        for part in args.target_dist.split(","):
            part = part.strip()
            if ":" in part:
                k, v = part.split(":", 1)
                target_dist[k.strip()] = float(v.strip())

    all_scores: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for d, lab in zip(dirs, labels_list):
        try:
            s1, s2, s3, s4 = compute_scores(
                d,
                tau_weight=args.tau_weight,
                tau_smooth=args.tau_smooth,
                window_size=args.window,
                target_dist=target_dist,
            )
            all_scores.append((lab, (s1, s2, s3, s4)))
            print(f"[{lab}] weight_norm={s1:.3f} task_balance={s2:.3f} appropriate_match={s3:.3f} smooth={s4:.3f}")
        except FileNotFoundError as e:
            print(f"Skip {d}: {e}")
            continue

    if not all_scores:
        print("No valid result dirs.")
        sys.exit(1)

    outdir = args.out or dirs[0]
    outdir = os.path.abspath(outdir)
    radar_path = os.path.join(outdir, "sanity_check_radar.png")
    plot_radar(all_scores, radar_path, title="Sanity Check Radar (higher is better)")
    print(f"Saved radar -> {radar_path}")

    if args.companion and not args.radar_only:
        csv_path = os.path.join(dirs[0], "accuracy_summary.csv")
        if os.path.isfile(csv_path):
            rows = _load_accuracy_summary(csv_path)
            run_companion_plots(dirs[0], outdir, rows)
            d1 = os.path.join(outdir, "sanity_task_type_dist.png")
            d2 = os.path.join(outdir, "sanity_accuracy_heatmap.png")
            d3 = os.path.join(outdir, "sanity_accuracy_radar.png")
            print(f"Saved companion plots -> {d1}, {d2}, {d3}")
        else:
            print("Companion plots require accuracy_summary.csv in first --dir.")


if __name__ == "__main__":
    main()
