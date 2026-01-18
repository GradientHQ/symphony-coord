#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo/sim_efficiency_cost.py

Symphony 2.0 模拟实验：任务路由效率对比（Efficiency & Cost）
- 不跑大模型，不需要 GPU
- 复用 core/linucb_selector.py 的 GlobalLinUCB + build_x(d=6)
- 两阶段选择：Top-L(静态 match_score) + LinUCB(动态 state)

输出：
- outdir/summary.csv         各策略汇总（cost / success / latency / fallback 等）
- outdir/trajectory_*.csv    逐步轨迹（LinUCB 学习曲线用）
- outdir/plot_*.png          图（柱状图+学习曲线）

运行示例：
  python3 sim_efficiency_cost.py --n 1000 --seed 123 --topL 2 --p-hard 0.2 --fallback

  # 让动态性更明显：加入漂移（t=500 后 A 变忙/变慢 或 B 变强）
  python3 demo/sim_efficiency_cost.py --n 1000 --seed 123 --topL 2 --p-hard 0.2 --fallback --drift
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

# ---- make project root importable ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
import sys
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---- reuse your LinUCB & build_x ----
try:
    from core.linucb_selector import GlobalLinUCB, build_x  # local mode
except Exception:
    # package mode fallback (if you run as installed package)
    from symphony.core.linucb_selector import GlobalLinUCB, build_x  # type: ignore


# -----------------------------
# 1) Data model
# -----------------------------
@dataclass
class SimTask:
    tid: int
    difficulty: str  # "simple" | "hard"
    requirement: str  # used for capability match (Top-L)
    # You can extend context dims here if you later want richer features


@dataclass
class SimAgentProfile:
    agent_id: str
    # static costs (token-like)
    call_cost: float
    # base latency in ms
    base_latency_ms: float
    # base success probs by difficulty (before load penalty / drift)
    p_simple: float
    p_hard: float
    # static capability match scores by requirement
    match_simple: float
    match_hard: float


@dataclass
class SimAgentState:
    # dynamic state used in build_x
    load: float = 0.0          # [0,1]
    latency_ms: float = 500.0  # moving estimate
    reputation: float = 0.5    # [0,1]
    available: bool = True


class SimAgent:
    """
    Simulation-only agent:
    - Provides get_dynamic_state() for build_x
    - Provides static match_score(req) for Top-L
    - Executes by Bernoulli(success_prob) and generates latency
    """

    def __init__(self, profile: SimAgentProfile, rng: random.Random):
        self.p = profile
        self.rng = rng
        self.s = SimAgentState(load=0.0, latency_ms=profile.base_latency_ms, reputation=0.5, available=True)

    def reset(self):
        self.s = SimAgentState(load=0.0, latency_ms=self.p.base_latency_ms, reputation=0.5, available=True)

    def match_score(self, requirement: str) -> float:
        if requirement == "simple":
            return float(self.p.match_simple)
        if requirement == "hard":
            return float(self.p.match_hard)
        # fallback
        return 0.5

    def get_dynamic_state(self) -> Dict[str, float]:
        return {
            "available": bool(self.s.available),
            "load": float(max(0.0, min(1.0, self.s.load))),
            "latency_ms": float(max(1.0, self.s.latency_ms)),
            "reputation": float(max(0.0, min(1.0, self.s.reputation))),
        }

    def _sample_latency(self) -> float:
        """
        Latency grows with load; add mild noise
        """
        load = max(0.0, min(1.0, self.s.load))
        mean = self.p.base_latency_ms * (1.0 + 1.2 * load)
        noise = self.rng.gauss(0.0, 0.08 * mean)
        return max(10.0, mean + noise)

    def _success_prob(self, difficulty: str, drift: bool = False, t: int = 0) -> float:
        """
        success prob affected by difficulty, load, and optional drift.
        """
        base = self.p.p_simple if difficulty == "simple" else self.p.p_hard

        # load penalty: heavy load reduces quality
        load = max(0.0, min(1.0, self.s.load))
        base = base * (1.0 - 0.05 * load)

        # optional drift after half time: make environment non-stationary
        # You can customize: e.g., A becomes worse after t>=500; B becomes better after t>=500
        if drift:
            if t >= 500:
                if self.p.agent_id == "A":
                    base *= 0.92  # A slightly degrades
                if self.p.agent_id == "B":
                    # B improves on hard a bit
                    if difficulty == "hard":
                        base = min(0.85, base + 0.20)

        return max(0.0, min(1.0, base))

    def step_dynamics_after_call(self, ok: bool, latency_ms: float):
        """
        Update dynamic state to create temporal correlation:
        - chosen agent load goes up; decays each step externally too (handled by env)
        - reputation EMA
        - latency EMA
        """
        # latency EMA
        beta = 0.20
        self.s.latency_ms = (1.0 - beta) * self.s.latency_ms + beta * float(latency_ms)

        # reputation EMA
        self.s.reputation = max(0.0, min(1.0, 0.95 * self.s.reputation + 0.05 * (1.0 if ok else 0.0)))

        # load spike for being chosen
        self.s.load = max(0.0, min(1.0, self.s.load + 0.30))

        # availability rule: too loaded => temporarily unavailable
        self.s.available = bool(self.s.load < 0.95)

    def decay_load(self):
        """
        Global decay each round (simulate queue drain)
        """
        self.s.load = max(0.0, min(1.0, 0.88 * self.s.load))
        # availability may recover
        self.s.available = bool(self.s.load < 0.95)

    def execute(self, task: SimTask, drift: bool = False, t: int = 0) -> Tuple[bool, float]:
        """
        Execute once: return (success, latency_ms)
        """
        lat = self._sample_latency()
        p = self._success_prob(task.difficulty, drift=drift, t=t)
        ok = (self.rng.random() < p)
        self.step_dynamics_after_call(ok=ok, latency_ms=lat)
        return ok, lat


# -----------------------------
# 2) Policies / Evaluation
# -----------------------------
@dataclass
class StepLog:
    t: int
    policy: str
    task_difficulty: str
    chosen_agent: str
    match_score: float
    load: float
    latency_ms: float
    call_cost: float
    success: int
    fallback_used: int
    total_cost_this_task: float
    total_latency_this_task: float
    reward_used_for_update: float


@dataclass
class SummaryRow:
    policy: str
    n: int
    p_hard: float
    topL: int
    success_rate: float
    total_cost: float
    avg_cost: float
    avg_latency_ms: float
    fallback_rate: float
    choose_A: int
    choose_B: int
    choose_C: int
    choose_D: int
    choose_E: int



def generate_tasks(n: int, p_hard: float, rng: random.Random) -> List[SimTask]:
    tasks: List[SimTask] = []
    for i in range(n):
        hard = (rng.random() < p_hard)
        diff = "hard" if hard else "simple"
        # requirement string used for Top-L matching
        req = "hard" if hard else "simple"
        tasks.append(SimTask(tid=i, difficulty=diff, requirement=req))
    return tasks


def pick_topL_candidates(
    agents: List[SimAgent],
    requirement: str,
    topL: int,
) -> List[Tuple[SimAgent, float]]:
    scored: List[Tuple[SimAgent, float]] = []
    for ag in agents:
        ms = ag.match_score(requirement)
        scored.append((ag, float(ms)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(1, int(topL))]


def normalize_cost(cost: float, max_cost: float) -> float:
    return float(cost / max(1e-9, max_cost))


def reward_shaping(
    success: bool,
    latency_ms: float,
    call_cost: float,
    latency_scale_ms: float,
    latency_penalty: float,
    cost_lambda: float,
    max_cost: float,
) -> float:
    """
    Reward in [0,1] (clipped), similar spirit to your agent-side update:
      reward = 1(success) - latency_penalty*sqrt(lat_norm) - cost_lambda*cost_norm
    """
    base = 1.0 if success else 0.0
    lat_norm = min(1.0, float(latency_ms) / max(1.0, float(latency_scale_ms)))
    cost_norm = normalize_cost(call_cost, max_cost)
    r = base - float(latency_penalty) * math.sqrt(lat_norm) - float(cost_lambda) * cost_norm
    return max(0.0, min(1.0, r))


def run_policy(
    policy_name: str,
    tasks: List[SimTask],
    agents: List[SimAgent],
    topL: int,
    *,
    linucb_alpha: float,
    linucb_l2: float,
    delta: float,
    S: float,
    latency_scale_ms: float,
    latency_penalty: float,
    cost_lambda: float,
    fallback: bool,
    drift: bool,
    seed_for_policy: int,
) -> Tuple[SummaryRow, List[StepLog]]:
    """
    Run one policy from scratch (reset agent states).
    """
    rng = random.Random(seed_for_policy)

    # reset agents
    for ag in agents:
        ag.reset()

    # selector only for LinUCB policy
    selector: Optional[GlobalLinUCB] = None
    if policy_name == "linucb":
        selector = GlobalLinUCB(d=6, l2=float(linucb_l2), alpha=float(linucb_alpha), delta=float(delta), S=float(S))

    max_cost = max(a.p.call_cost for a in agents)
    # for summaries
    choose_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    total_cost = 0.0
    total_latency = 0.0
    success_cnt = 0
    fallback_cnt = 0
    step_logs: List[StepLog] = []

    # helper: get agent by id
    agent_by_id = {a.p.agent_id: a for a in agents}
    strong_agent_id = "A"  # treat A as strongest fallback

    for t, task in enumerate(tasks):
        # decay load for all agents each round (global dynamics)
        for ag in agents:
            ag.decay_load()

        # ---- candidate pool by Top-L ----
        top = pick_topL_candidates(agents, task.requirement, topL=topL)

        # filter available candidates (like your orchestrator)
        avail = [(ag, ms) for (ag, ms) in top if ag.get_dynamic_state().get("available", True)]
        if not avail:
            # if no available in topL, allow choosing from top anyway (mark available=False in x)
            avail = top

        # ---- choose action (agent) ----
        chosen_ag: SimAgent
        chosen_ms: float
        chosen_x: Optional[List[float]] = None

        if policy_name == "always_A":
            chosen_ag = agent_by_id[strong_agent_id]
            chosen_ms = chosen_ag.match_score(task.requirement)

        elif policy_name == "static_rule":
            # Simple->B, Hard->A
            chosen_id = "B" if task.difficulty == "simple" else "A"
            chosen_ag = agent_by_id.get(chosen_id, agent_by_id[strong_agent_id])
            chosen_ms = chosen_ag.match_score(task.requirement)

        elif policy_name == "random":
            chosen_ag, chosen_ms = rng.choice(avail)

        elif policy_name == "linucb":
            assert selector is not None
            candidates: List[Tuple[str, List[float], float]] = []
            # build x for each candidate (aid, x)
            for (ag, ms) in avail:
                st = ag.get_dynamic_state()
                x = build_x(
                    match_score=float(ms),
                    dynamic_state={
                        "load": float(st.get("load", 0.0)),
                        "latency_ms": float(st.get("latency_ms", 500.0)),
                        "reputation": float(st.get("reputation", 0.5)),
                    },
                    available=bool(st.get("available", True)),
                    latency_scale_ms=float(latency_scale_ms),
                )
                candidates.append((ag.p.agent_id, x, float(ms)))

            chosen_id = selector.select([(aid, x) for (aid, x, _) in candidates])
            # recover chosen
            chosen_ag = agent_by_id[chosen_id]
            chosen_ms = next(ms for (aid, _x, ms) in candidates if aid == chosen_id)
            chosen_x = next(_x for (aid, _x, _ms) in candidates if aid == chosen_id)

        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        choose_counts[chosen_ag.p.agent_id] = choose_counts.get(chosen_ag.p.agent_id, 0) + 1

        # ---- execute chosen agent ----
        ok, lat_ms = chosen_ag.execute(task, drift=drift, t=t)
        call_cost = chosen_ag.p.call_cost

        # ---- optional fallback to A if failed (more realistic cost) ----
        total_cost_this = call_cost
        total_lat_this = lat_ms
        fallback_used = 0
        final_ok = ok

        if fallback and (not ok) and (chosen_ag.p.agent_id != strong_agent_id):
            fallback_used = 1
            fallback_cnt += 1
            fallback_ag = agent_by_id[strong_agent_id]
            ok2, lat2 = fallback_ag.execute(task, drift=drift, t=t)
            final_ok = ok2  # final success after fallback
            total_cost_this += fallback_ag.p.call_cost
            total_lat_this += lat2

        # ---- update metrics ----
        total_cost += total_cost_this
        total_latency += total_lat_this
        success_cnt += 1 if final_ok else 0

        # ---- LinUCB online update (only update chosen action; fallback doesn't change chosen's reward) ----
        used_reward = 0.0
        if policy_name == "linucb":
            assert selector is not None
            # if chosen_x wasn't built (shouldn't happen), build it now
            if chosen_x is None:
                st = chosen_ag.get_dynamic_state()
                chosen_x = build_x(
                    match_score=float(chosen_ms),
                    dynamic_state={
                        "load": float(st.get("load", 0.0)),
                        "latency_ms": float(st.get("latency_ms", 500.0)),
                        "reputation": float(st.get("reputation", 0.5)),
                    },
                    available=bool(st.get("available", True)),
                    latency_scale_ms=float(latency_scale_ms),
                )

            used_reward = reward_shaping(
                success=ok,  # reward is for the chosen agent outcome (pre-fallback)
                latency_ms=lat_ms,
                call_cost=call_cost,
                latency_scale_ms=latency_scale_ms,
                latency_penalty=latency_penalty,
                cost_lambda=cost_lambda,
                max_cost=max_cost,
            )
            selector.update(chosen_x, used_reward)

        # ---- logging ----
        st_now = chosen_ag.get_dynamic_state()
        step_logs.append(
            StepLog(
                t=t,
                policy=policy_name,
                task_difficulty=task.difficulty,
                chosen_agent=chosen_ag.p.agent_id,
                match_score=float(chosen_ms),
                load=float(st_now.get("load", 0.0)),
                latency_ms=float(lat_ms),
                call_cost=float(call_cost),
                success=1 if final_ok else 0,
                fallback_used=fallback_used,
                total_cost_this_task=float(total_cost_this),
                total_latency_this_task=float(total_lat_this),
                reward_used_for_update=float(used_reward),
            )
        )

    # ---- summary ----
    n = len(tasks)
    avg_latency = (total_latency / max(1, n))
    avg_cost = (total_cost / max(1, n))
    fallback_rate = (fallback_cnt / max(1, n))

    row = SummaryRow(
        policy=policy_name,
        n=n,
        p_hard=float(sum(1 for x in tasks if x.difficulty == "hard") / max(1, n)),
        topL=int(topL),
        success_rate=float(success_cnt / max(1, n)),
        total_cost=float(total_cost),
        avg_cost=float(avg_cost),
        avg_latency_ms=float(avg_latency),
        fallback_rate=float(fallback_rate),
        choose_A=int(choose_counts.get("A", 0)),
        choose_B=int(choose_counts.get("B", 0)),
        choose_C=int(choose_counts.get("C", 0)),
        choose_D=int(choose_counts.get("D", 0)),
        choose_E=int(choose_counts.get("E", 0)),
    )

    return row, step_logs



# -----------------------------
# 3) Plotting (matplotlib)
def try_plot(outdir: str, summary: List[SummaryRow], traj: Dict[str, List[StepLog]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib not available: {e}")
        return

    import os  # ✅ 必须有

    # ---------- nicer labels ----------
    name_map = {
        "always_A": "Always-A",
        "static_rule": "Static rule",
        "random": "Random",
        "linucb": "LinUCB (Ours)",
    }

    labels = [name_map.get(r.policy, r.policy) for r in summary]
    costs = [r.total_cost for r in summary]
    succ = [r.success_rate for r in summary]

    # ---------- Figure 1: two-panel (no twin axis) ----------
    fig = plt.figure(figsize=(9.5, 3.6), dpi=180, constrained_layout=True)

    # Cost bar
    ax1 = fig.add_subplot(1, 2, 1)
    bars = ax1.bar(range(len(labels)), costs)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Total cost")
    ax1.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ymax = max(costs) if costs else 1.0
    for b, v in zip(bars, costs):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            v + 0.02 * ymax,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Success
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(range(len(labels)), succ, marker="o", linewidth=1.8)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Success rate")
    ax2.set_ylim(min(succ) - 0.01, min(1.0, max(succ) + 0.01))
    ax2.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for i, v in enumerate(succ):
        ax2.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.savefig(os.path.join(outdir, "plot_cost_success_pretty.png"))
    plt.close(fig)

    # ---------- helper: rolling mean ----------
    def _rolling_mean(arr, window: int):
        if window <= 1:
            return arr
        out, s, q = [], 0.0, []
        for v in arr:
            q.append(v)
            s += v
            if len(q) > window:
                s -= q.pop(0)
            out.append(s / len(q))
        return out

    # ---------- plot 4 policies on one figure ----------
    def plot_all_policies_curves_clean(rolling: int = 50) -> None:
        """
        两张图（不同策略集合）：
          1) 成本：不画 Always-A，y轴固定 0~0.6
          2) 成功率：画 Always-A + 其它三种
        rolling: 滑动平均窗口（仅用于视觉平滑，不改变最终值）
        """
        order_cost = ["static_rule", "random", "linucb"]  # ✅ cost 不画 always_A
        order_succ = ["always_A", "static_rule", "random", "linucb"]  # ✅ success 画 always_A

        def _rolling_mean(arr, window: int):
            if window <= 1:
                return arr
            out, s, q = [], 0.0, []
            for v in arr:
                q.append(v)
                s += v
                if len(q) > window:
                    s -= q.pop(0)
                out.append(s / len(q))
            return out

        # ------------------ (1) cumulative avg cost ------------------
        fig1 = plt.figure(figsize=(7.6, 3.8), dpi=240, constrained_layout=True)
        ax = fig1.add_subplot(111)

        for pol in order_cost:
            logs = traj.get(pol, [])
            if not logs:
                continue

            costs_step = [float(st.total_cost_this_task) for st in logs]  # 含 fallback 的额外成本
            cum, s = [], 0.0
            for i, c in enumerate(costs_step, 1):
                s += c
                cum.append(s / i)

            cum = _rolling_mean(cum, rolling)
            xs = list(range(1, len(cum) + 1))

            ax.plot(xs, cum, linewidth=1.05, alpha=0.90, label=name_map.get(pol, pol))

        ax.set_xlabel("t")
        ax.set_ylabel("Cumulative avg cost")
        ax.set_ylim(0.0, 1)  # ✅ 按你要求固定范围
        ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, ncol=2, fontsize=9,
                  loc="upper center", bbox_to_anchor=(0.5, 1.16))

        fig1.savefig(os.path.join(outdir, "plot_all_cum_cost_clean.png"))
        plt.close(fig1)

        # ------------------ (2) cumulative avg success ------------------
        fig2 = plt.figure(figsize=(7.6, 3.8), dpi=240, constrained_layout=True)
        ax = fig2.add_subplot(111)

        for pol in order_succ:
            logs = traj.get(pol, [])
            if not logs:
                continue

            succ_step = [float(st.success) for st in logs]  # 最终成功（含 fallback 后结果）
            cum, s = [], 0.0
            for i, v in enumerate(succ_step, 1):
                s += v
                cum.append(s / i)

            cum = _rolling_mean(cum, rolling)
            xs = list(range(1, len(cum) + 1))

            if pol == "always_A":
                # ✅ 让 Always-A 更“弱存在感”：虚线 + 半透明 + 略细
                ax.plot(xs, cum, linewidth=0.95, linestyle="--", alpha=0.65,
                        label=name_map.get(pol, pol))
            else:
                ax.plot(xs, cum, linewidth=1.05, alpha=0.90,
                        label=name_map.get(pol, pol))

        ax.set_xlabel("t")
        ax.set_ylabel("Cumulative avg success (final)")
        ax.set_ylim(0.94, 1.0)  # 想更紧凑可改 0.96~1.0
        ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, ncol=2, fontsize=9,
                  loc="lower center", bbox_to_anchor=(0.5, -0.30))

        fig2.savefig(os.path.join(outdir, "plot_all_cum_success_clean.png"))
        plt.close(fig2)
    plot_all_policies_curves_clean(rolling=50)


# -----------------------------
# 4) IO helpers
# -----------------------------
def write_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="number of tasks")
    ap.add_argument("--p-hard", type=float, default=0.2, help="probability of hard task")
    ap.add_argument("--seed", type=int, default=123, help="random seed")
    ap.add_argument("--topL", type=int, default=2, help="Top-L candidates by static match_score")
    ap.add_argument("--outdir", type=str, default="runtime/sim_efficiency_cost", help="output directory")
    ap.add_argument("--no-plots", action="store_true", help="do not generate png plots")

    # LinUCB params
    ap.add_argument("--alpha", type=float, default=1.0, help="LinUCB exploration scale")
    ap.add_argument("--l2", type=float, default=1.0, help="LinUCB l2 regularization lambda")
    ap.add_argument("--delta", type=float, default=0.05, help="LinUCB confidence")
    ap.add_argument("--S", type=float, default=1.0, help="bound on ||theta*|| (for beta())")

    # reward shaping params
    ap.add_argument("--latency-scale-ms", type=float, default=2000.0, help="latency normalization scale")
    ap.add_argument("--latency-penalty", type=float, default=0.2, help="penalty multiplier for latency")
    ap.add_argument("--cost-lambda", type=float, default=0.15, help="penalty multiplier for cost")

    # realism toggles
    ap.add_argument("--fallback", action="store_true", help="if chosen agent fails, fallback to strong agent A")
    ap.add_argument("--drift", action="store_true", help="enable non-stationary drift after t>=500")

    # agent set: 2 agents (A,B) by default; optionally add C
    ap.add_argument("--with-C", action="store_true", help="add an extra medium agent C")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    base_rng = random.Random(args.seed)
    tasks = generate_tasks(args.n, args.p_hard, base_rng)

    # ---- Define agent pool (you can edit these numbers) ----
    profiles: List[SimAgentProfile] = [
        # A: strongest, most expensive, stable
        SimAgentProfile(
            agent_id="A",
            call_cost=1.00,
            base_latency_ms=900.0,
            p_simple=0.99,
            p_hard=0.99,
            match_simple=0.80,
            match_hard=0.95,
        ),

        # B: cheapest, good for simple, bad for hard
        SimAgentProfile(
            agent_id="B",
            call_cost=0.10,
            base_latency_ms=350.0,
            p_simple=0.88,
            p_hard=0.18,
            match_simple=0.95,
            match_hard=0.20,
        ),

        # C: medium (balanced)
        SimAgentProfile(
            agent_id="C",
            call_cost=0.35,
            base_latency_ms=550.0,
            p_simple=0.92,
            p_hard=0.65,
            match_simple=0.75,
            match_hard=0.75,
        ),

        # D: fast-ish, ok at simple, mediocre at hard (can represent "tool-ish" / "fast model")
        SimAgentProfile(
            agent_id="D",
            call_cost=0.22,
            base_latency_ms=420.0,
            p_simple=0.90,
            p_hard=0.45,
            match_simple=0.85,
            match_hard=0.55,
        ),

        # E: hard-specialist but slower / mid cost (represents "reasoning specialist")
        SimAgentProfile(
            agent_id="E",
            call_cost=0.60,
            base_latency_ms=780.0,
            p_simple=0.95,
            p_hard=0.85,
            match_simple=0.60,
            match_hard=0.90,
        ),
    ]

    # We will rebuild agents per policy to keep RNG independent (fair comparison)
    def make_agents(policy_seed: int) -> List[SimAgent]:
        rng = random.Random(policy_seed)
        return [SimAgent(p, rng) for p in profiles]

    policies = ["always_A", "static_rule", "random", "linucb"]
    summary_rows: List[SummaryRow] = []
    traj_logs: Dict[str, List[StepLog]] = {}

    for i, pol in enumerate(policies):
        pol_seed = args.seed + 1000 * (i + 1)
        agents = make_agents(pol_seed)

        row, logs = run_policy(
            policy_name=pol,
            tasks=tasks,
            agents=agents,
            topL=args.topL,
            linucb_alpha=args.alpha,
            linucb_l2=args.l2,
            delta=args.delta,
            S=args.S,
            latency_scale_ms=args.latency_scale_ms,
            latency_penalty=args.latency_penalty,
            cost_lambda=args.cost_lambda,
            fallback=bool(args.fallback),
            drift=bool(args.drift),
            seed_for_policy=pol_seed,
        )
        summary_rows.append(row)
        traj_logs[pol] = logs

        print(
            f"[{pol}] success={row.success_rate:.3f} "
            f"total_cost={row.total_cost:.1f} avg_cost={row.avg_cost:.3f} "
            f"avg_lat={row.avg_latency_ms:.1f}ms fallback={row.fallback_rate:.3f} "
            f"choose(A,B,C)=({row.choose_A},{row.choose_B},{row.choose_C})"
        )

    # ---- write outputs ----
    write_csv(
        os.path.join(args.outdir, "summary.csv"),
        [asdict(r) for r in summary_rows],
    )
    for pol, logs in traj_logs.items():
        write_csv(
            os.path.join(args.outdir, f"trajectory_{pol}.csv"),
            [asdict(x) for x in logs],
        )

    if not args.no_plots:
        try_plot(args.outdir, summary_rows, traj_logs)

    print(f"\n✅ Done. Outputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()
