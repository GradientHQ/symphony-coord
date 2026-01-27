#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/exp3/sim_system_optimization.py

Symphony 2.0 Experiment 3: System Performance Optimization
Test LinUCB's ability to learn system-level optimization (latency, load balancing)
under heterogeneous and dynamic conditions.

Key scenarios:
- Latency heterogeneous: agents have different latencies
- Load burst: periodic load spikes on specific agents
- Combined: both latency and load challenges

Comparison policies (aligned with Exp1 - 4 policies):
- always_A: always select strongest agent A (upper bound)
- static_rule: Simple->B, Hard->A (hard-coded rule, same as Exp1)
- random: random selection from TopL
- linucb: LinUCB intelligent routing (Symphony 2.0)

Usage:
  python3 sim_system_optimization.py --config configs/scenarios.yaml
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

# ---- make project root importable ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
import sys
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---- reuse LinUCB & build_x ----
from core.linucb_selector import GlobalLinUCB, build_x


# -----------------------------
# 1) Data model
# -----------------------------
@dataclass
class SimTask:
    tid: int
    difficulty: str  # "simple" | "hard"
    requirement: str  # used for capability match


@dataclass
class SimAgentProfile:
    agent_id: str
    call_cost: float
    base_latency_ms: float
    p_simple: float
    p_hard: float
    match_simple: float
    match_hard: float


@dataclass
class SimAgentState:
    load: float = 0.0
    latency_ms: float = 500.0
    reputation: float = 0.5
    available: bool = True


class SimAgent:
    """Base simulation agent (from Exp1)"""
    def __init__(self, profile: SimAgentProfile, rng: random.Random):
        self.p = profile
        self.rng = rng
        self.s = SimAgentState(
            load=0.0,
            latency_ms=profile.base_latency_ms,
            reputation=0.5,
            available=True
        )

    def reset(self):
        self.s = SimAgentState(
            load=0.0,
            latency_ms=self.p.base_latency_ms,
            reputation=0.5,
            available=True
        )

    def match_score(self, requirement: str) -> float:
        if requirement == "simple":
            return float(self.p.match_simple)
        if requirement == "hard":
            return float(self.p.match_hard)
        return 0.5

    def get_dynamic_state(self) -> Dict[str, float]:
        return {
            "available": bool(self.s.available),
            "load": float(max(0.0, min(1.0, self.s.load))),
            "latency_ms": float(max(1.0, self.s.latency_ms)),
            "reputation": float(max(0.0, min(1.0, self.s.reputation))),
        }

    def _sample_latency(self) -> float:
        """Latency grows with load"""
        load = max(0.0, min(1.0, self.s.load))
        mean = self.p.base_latency_ms * (1.0 + 1.2 * load)
        noise = self.rng.gauss(0.0, 0.08 * mean)
        return max(10.0, mean + noise)

    def _success_prob(self, difficulty: str) -> float:
        """Success probability affected by difficulty and load"""
        base = self.p.p_simple if difficulty == "simple" else self.p.p_hard
        load = max(0.0, min(1.0, self.s.load))
        base = base * (1.0 - 0.05 * load)
        return max(0.0, min(1.0, base))

    def step_dynamics_after_call(self, ok: bool, latency_ms: float):
        """Update dynamic state after execution"""
        # Latency EMA
        beta = 0.20
        self.s.latency_ms = (1.0 - beta) * self.s.latency_ms + beta * float(latency_ms)
        
        # Reputation EMA
        self.s.reputation = max(0.0, min(1.0, 
            0.95 * self.s.reputation + 0.05 * (1.0 if ok else 0.0)))
        
        # Load spike
        self.s.load = max(0.0, min(1.0, self.s.load + 0.30))
        
        # Availability
        self.s.available = bool(self.s.load < 0.95)

    def decay_load(self):
        """Global load decay each round"""
        self.s.load = max(0.0, min(1.0, 0.88 * self.s.load))
        self.s.available = bool(self.s.load < 0.95)

    def execute(self, task: SimTask) -> Tuple[bool, float]:
        """Execute task and return (success, latency_ms)"""
        lat = self._sample_latency()
        p = self._success_prob(task.difficulty)
        ok = (self.rng.random() < p)
        self.step_dynamics_after_call(ok=ok, latency_ms=lat)
        return ok, lat


class SimAgentWithLatencyMultiplier(SimAgent):
    """Extended agent with configurable latency multiplier"""
    def __init__(self, profile: SimAgentProfile, rng: random.Random, 
                 latency_multiplier: float = 1.0):
        super().__init__(profile, rng)
        self.latency_multiplier = latency_multiplier
        # Update initial latency
        self.s.latency_ms = profile.base_latency_ms * latency_multiplier

    def _sample_latency(self) -> float:
        """Apply multiplier to base latency"""
        base_lat = super()._sample_latency()
        return base_lat * self.latency_multiplier


# -----------------------------
# 2) Scenario configuration
# -----------------------------
@dataclass
class LoadBurst:
    agent: str
    start_t: int
    end_t: int
    load_increase: float


@dataclass
class ScenarioConfig:
    name: str
    description: str
    enabled: bool
    agent_latency_multipliers: Dict[str, float]
    load_bursts: List[LoadBurst]


def load_config(config_path: str) -> Dict:
    """Load experiment configuration from YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def apply_load_bursts(
    agents_dict: Dict[str, SimAgent],
    t: int,
    load_bursts: List[LoadBurst]
):
    """Apply load bursts to agents at time t"""
    for burst in load_bursts:
        if burst.start_t <= t < burst.end_t:
            agent = agents_dict.get(burst.agent)
            if agent:
                # Force load increase during burst period
                agent.s.load = min(1.0, agent.s.load + burst.load_increase)


# -----------------------------
# 3) Policies
# -----------------------------
def pick_topL_candidates(
    agents: List[SimAgent],
    requirement: str,
    topL: int,
) -> List[Tuple[SimAgent, float]]:
    """Select TopL candidates by match_score"""
    scored: List[Tuple[SimAgent, float]] = []
    for ag in agents:
        ms = ag.match_score(requirement)
        scored.append((ag, float(ms)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max(1, int(topL))]


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
    Reward shaping for LinUCB (consistent with Exp1):
    reward = 1(success) - latency_penalty*sqrt(lat_norm) - cost_lambda*cost_norm
    
    Key for Exp3: latency_penalty encourages LinUCB to learn low-latency routes
    
    ✅ Aligned with Exp1: normalize latency to ensure reward ∈ [0,1]
    """
    base = 1.0 if success else 0.0
    
    # Normalize latency to [0,1]
    lat_norm = min(1.0, float(latency_ms) / max(1.0, float(latency_scale_ms)))
    
    # Cost penalty (already normalized)
    cost_norm = normalize_cost(call_cost, max_cost)
    
    # Reward with sqrt for latency penalty (same as Exp1)
    r = base - float(latency_penalty) * math.sqrt(lat_norm) - float(cost_lambda) * cost_norm
    return max(0.0, min(1.0, r))


# -----------------------------
# 4) Metrics
# -----------------------------
@dataclass
class StepLog:
    t: int
    policy: str
    scenario: str
    task_difficulty: str
    chosen_agent: str
    match_score: float
    load: float
    latency_ms: float
    call_cost: float
    success: int
    reward_used_for_update: float


@dataclass
class SummaryRow:
    policy: str
    scenario: str
    n_tasks: int
    success_rate: float
    avg_latency_ms: float
    latency_p95_ms: float
    total_cost: float
    avg_cost: float
    agent_utilization_gini: float
    latency_efficiency: float  # success_rate / avg_latency_ms
    choose_A: int
    choose_B: int
    choose_C: int
    choose_D: int
    choose_E: int


def compute_gini_coefficient(counts: List[int]) -> float:
    """
    Compute Gini coefficient for agent utilization distribution.
    0 = perfect equality, 1 = perfect inequality
    """
    if not counts or sum(counts) == 0:
        return 0.0
    
    sorted_counts = sorted(counts)
    n = len(sorted_counts)
    total = sum(sorted_counts)
    
    cumsum = 0.0
    gini_sum = 0.0
    
    for i, c in enumerate(sorted_counts, 1):
        cumsum += c
        gini_sum += (2 * i - n - 1) * c
    
    return gini_sum / (n * total)


def compute_summary(
    policy: str,
    scenario: str,
    step_logs: List[StepLog]
) -> SummaryRow:
    """Compute summary metrics from step logs"""
    n = len(step_logs)
    if n == 0:
        return SummaryRow(
            policy=policy, scenario=scenario, n_tasks=0,
            success_rate=0.0, avg_latency_ms=0.0, latency_p95_ms=0.0,
            total_cost=0.0, avg_cost=0.0, agent_utilization_gini=0.0,
            latency_efficiency=0.0,
            choose_A=0, choose_B=0, choose_C=0, choose_D=0, choose_E=0
        )
    
    # Basic metrics
    success_cnt = sum(log.success for log in step_logs)
    success_rate = success_cnt / n
    
    latencies = [log.latency_ms for log in step_logs]
    avg_latency = statistics.mean(latencies)
    latency_p95 = sorted(latencies)[int(0.95 * len(latencies))]
    
    total_cost = sum(log.call_cost for log in step_logs)
    avg_cost = total_cost / n
    
    # Agent selection counts
    agent_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    for log in step_logs:
        agent_counts[log.chosen_agent] = agent_counts.get(log.chosen_agent, 0) + 1
    
    # Gini coefficient for load balancing
    counts_list = [agent_counts[aid] for aid in ["A", "B", "C", "D", "E"]]
    gini = compute_gini_coefficient(counts_list)
    
    # Latency efficiency
    latency_efficiency = success_rate / max(1.0, avg_latency) * 1000  # per second
    
    return SummaryRow(
        policy=policy,
        scenario=scenario,
        n_tasks=n,
        success_rate=success_rate,
        avg_latency_ms=avg_latency,
        latency_p95_ms=latency_p95,
        total_cost=total_cost,
        avg_cost=avg_cost,
        agent_utilization_gini=gini,
        latency_efficiency=latency_efficiency,
        choose_A=agent_counts["A"],
        choose_B=agent_counts["B"],
        choose_C=agent_counts["C"],
        choose_D=agent_counts["D"],
        choose_E=agent_counts["E"],
    )


# -----------------------------
# 5) Main simulation
# -----------------------------
def run_policy(
    policy_name: str,
    scenario_name: str,
    tasks: List[SimTask],
    agents: List[SimAgent],
    agents_dict: Dict[str, SimAgent],
    topL: int,
    load_bursts: List[LoadBurst],
    *,
    linucb_alpha: float,
    linucb_l2: float,
    delta: float,
    S: float,
    latency_scale_ms: float,
    latency_penalty: float,
    cost_lambda: float,
    seed_for_policy: int,
) -> Tuple[SummaryRow, List[StepLog]]:
    """Run one policy under a specific scenario"""
    rng = random.Random(seed_for_policy)
    
    # Reset agents
    for ag in agents:
        ag.reset()
    
    # Selector only for LinUCB
    selector: Optional[GlobalLinUCB] = None
    if policy_name == "linucb":
        selector = GlobalLinUCB(
            d=6,
            l2=float(linucb_l2),
            alpha=float(linucb_alpha),
            delta=float(delta),
            S=float(S)
        )
    
    max_cost = max(a.p.call_cost for a in agents)
    step_logs: List[StepLog] = []
    
    for t, task in enumerate(tasks):
        # Global load decay
        for ag in agents:
            ag.decay_load()
        
        # Apply load bursts
        apply_load_bursts(agents_dict, t, load_bursts)
        
        # Get TopL candidates
        top = pick_topL_candidates(agents, task.requirement, topL=topL)
        
        # Filter available
        avail = [(ag, ms) for (ag, ms) in top 
                 if ag.get_dynamic_state().get("available", True)]
        if not avail:
            avail = top
        
        # Select agent based on policy
        chosen_ag: SimAgent
        chosen_ms: float
        chosen_x: Optional[List[float]] = None
        
        if policy_name == "always_A":
            # Always select strongest agent A (upper bound baseline)
            chosen_ag = agents_dict["A"]
            chosen_ms = chosen_ag.match_score(task.requirement)
        
        elif policy_name == "random":
            chosen_ag, chosen_ms = rng.choice(avail)
        
        elif policy_name == "static_rule":
            # Static rule (aligned with Exp1): Simple->B, Hard->A
            chosen_id = "B" if task.difficulty == "simple" else "A"
            chosen_ag = agents_dict.get(chosen_id, agents_dict["A"])
            chosen_ms = chosen_ag.match_score(task.requirement)
        
        elif policy_name == "linucb":
            assert selector is not None
            candidates: List[Tuple[str, List[float], float]] = []
            
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
            chosen_ag = agents_dict[chosen_id]
            chosen_ms = next(ms for (aid, _, ms) in candidates if aid == chosen_id)
            chosen_x = next(x for (aid, x, _) in candidates if aid == chosen_id)
        
        else:
            raise ValueError(f"Unknown policy: {policy_name}")
        
        # Execute
        ok, lat_ms = chosen_ag.execute(task)
        call_cost = chosen_ag.p.call_cost
        
        # Update LinUCB
        used_reward = 0.0
        if policy_name == "linucb":
            assert selector is not None
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
                success=ok,
                latency_ms=lat_ms,
                call_cost=call_cost,
                latency_scale_ms=latency_scale_ms,
                latency_penalty=latency_penalty,
                cost_lambda=cost_lambda,
                max_cost=max_cost,
            )
            selector.update(chosen_x, used_reward)
        
        # Log
        st_now = chosen_ag.get_dynamic_state()
        step_logs.append(StepLog(
            t=t,
            policy=policy_name,
            scenario=scenario_name,
            task_difficulty=task.difficulty,
            chosen_agent=chosen_ag.p.agent_id,
            match_score=float(chosen_ms),
            load=float(st_now.get("load", 0.0)),
            latency_ms=float(lat_ms),
            call_cost=float(call_cost),
            success=1 if ok else 0,
            reward_used_for_update=float(used_reward),
        ))
    
    # Compute summary
    summary = compute_summary(policy_name, scenario_name, step_logs)
    return summary, step_logs


def generate_tasks(n: int, p_hard: float, rng: random.Random) -> List[SimTask]:
    """Generate task stream"""
    tasks: List[SimTask] = []
    for i in range(n):
        hard = (rng.random() < p_hard)
        diff = "hard" if hard else "simple"
        req = "hard" if hard else "simple"
        tasks.append(SimTask(tid=i, difficulty=diff, requirement=req))
    return tasks


def write_csv(path: str, rows: List[dict]) -> None:
    """Write rows to CSV"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------------
# 6) Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default="experiments/exp3/configs/scenarios.yaml",
        help="Path to scenario configuration YAML"
    )
    args = ap.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    exp_config = config["experiment"]
    scenarios_config = config["scenarios"]
    reward_config = config.get("reward", {})
    
    # Experiment parameters
    n_tasks = exp_config["n_tasks"]
    topL = exp_config["topL"]
    seed = exp_config["seed"]
    p_hard = exp_config["p_hard"]
    policies = exp_config["policies"]
    
    # Output directory
    outdir = config["output"]["result_dir"]
    os.makedirs(outdir, exist_ok=True)
    
    # Generate tasks
    base_rng = random.Random(seed)
    tasks = generate_tasks(n_tasks, p_hard, base_rng)
    
    # Agent profiles with differentiated match scores
    # Following exp1's design: different agents have different strengths
    agent_configs = exp_config["agents"]
    profiles: List[SimAgentProfile] = []
    
    # Match score mapping: reflects agent specialization
    # Higher match_score → higher priority in TopL candidate selection
    match_scores = {
        "A": (0.80, 0.95),  # Code specialist: good for hard tasks
        "B": (0.95, 0.60),  # Math specialist: excellent for simple, weak for hard
        "C": (0.85, 0.85),  # Generalist: balanced for both
        "D": (0.90, 0.70),  # Fast agent: good for simple, moderate for hard
        "E": (0.70, 0.90),  # Reasoning specialist: weak for simple, strong for hard
    }
    
    for aid, cfg in agent_configs.items():
        match_simple, match_hard = match_scores.get(aid, (0.8, 0.8))
        profiles.append(SimAgentProfile(
            agent_id=aid,
            call_cost=cfg["call_cost"],
            base_latency_ms=cfg["base_latency_ms"],
            p_simple=cfg["easy_capability"],
            p_hard=cfg["hard_capability"],
            match_simple=match_simple,
            match_hard=match_hard,
        ))
    
    # Run experiments
    all_summaries: List[SummaryRow] = []
    all_step_logs: List[StepLog] = []
    
    for scenario_name, scenario_cfg in scenarios_config.items():
        if not scenario_cfg.get("enabled", True):
            print(f"[INFO] Skipping disabled scenario: {scenario_name}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Running scenario: {scenario_name}")
        print(f"Description: {scenario_cfg['description']}")
        print(f"{'='*60}")
        
        # Parse scenario config
        latency_multipliers = scenario_cfg.get("agent_latency_multipliers", {})
        load_bursts_raw = scenario_cfg.get("load_bursts", [])
        load_bursts = [
            LoadBurst(
                agent=b["agent"],
                start_t=b["start_t"],
                end_t=b["end_t"],
                load_increase=b["load_increase"]
            )
            for b in load_bursts_raw
        ]
        
        # Run each policy
        for policy_name in policies:
            policy_seed = seed + hash(f"{scenario_name}_{policy_name}") % 10000
            
            # Create agents with latency multipliers
            rng = random.Random(policy_seed)
            agents: List[SimAgent] = []
            for profile in profiles:
                multiplier = latency_multipliers.get(profile.agent_id, 1.0)
                agent = SimAgentWithLatencyMultiplier(profile, rng, multiplier)
                agents.append(agent)
            
            agents_dict = {ag.p.agent_id: ag for ag in agents}
            
            # Run policy
            summary, step_logs = run_policy(
                policy_name=policy_name,
                scenario_name=scenario_name,
                tasks=tasks,
                agents=agents,
                agents_dict=agents_dict,
                topL=topL,
                load_bursts=load_bursts,
                linucb_alpha=1.0,
                linucb_l2=1.0,
                delta=0.05,
                S=1.0,
                latency_scale_ms=2000.0,
                latency_penalty=reward_config.get("latency_penalty", 0.2),
                cost_lambda=reward_config.get("cost_lambda", 0.15),
                seed_for_policy=policy_seed,
            )
            
            all_summaries.append(summary)
            all_step_logs.extend(step_logs)
            
            print(f"  [{policy_name:20s}] "
                  f"success={summary.success_rate:.3f} "
                  f"avg_lat={summary.avg_latency_ms:.1f}ms "
                  f"p95_lat={summary.latency_p95_ms:.1f}ms "
                  f"gini={summary.agent_utilization_gini:.3f} "
                  f"eff={summary.latency_efficiency:.3f}")
    
    # Save results
    write_csv(
        os.path.join(outdir, "summary_all_scenarios.csv"),
        [asdict(s) for s in all_summaries]
    )
    
    write_csv(
        os.path.join(outdir, "step_logs_all.csv"),
        [asdict(log) for log in all_step_logs]
    )
    
    print(f"\n✅ Experiment completed!")
    print(f"📊 Results saved to: {outdir}")
    print(f"   - summary_all_scenarios.csv")
    print(f"   - step_logs_all.csv")


if __name__ == "__main__":
    main()

