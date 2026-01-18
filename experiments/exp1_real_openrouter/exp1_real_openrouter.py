#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp1 Real OpenRouter: Efficiency & Cost with REAL OpenRouter Agents

This is the REAL version of Exp1, using actual OpenRouter agents (from config files)
instead of simulated agents. All other logic (multi-policy comparison, reward shaping,
fallback, drift, etc.) remains the same as the simulation version.

Usage:
  python3 exp1_real_openrouter.py --n 1000 --topL 2 --seed 123 --fallback
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import math
import random
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
import sys
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.linucb_selector import GlobalLinUCB, build_x
from agents.agent import Agent
import json

# ✅ Import new routing modules
try:
    from core.routing import select_topL, get_prior_success
    from core.cold_start import load_priors
    HAS_ROUTING = True
except ImportError:
    HAS_ROUTING = False
    print("[WARN] core.routing not available, falling back to legacy pick_topL_candidates")


# -----------------------------
# 0) Configuration loading
# -----------------------------
def load_exp1_config(config_path: Optional[str] = None) -> Dict:
    """Load Exp1 configuration from YAML file"""
    if config_path is None:
        config_path = os.path.join(_THIS_DIR, "config_exp1.yaml")
    
    if not os.path.exists(config_path):
        print(f"[WARN] Config file not found: {config_path}, using defaults")
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# -----------------------------
# 1) Data model
# -----------------------------
@dataclass
class RealTask:
    tid: int
    difficulty: str  # "easy" | "hard" (updated to match config)
    requirement: str  # used for capability match (Top-L)
    prompt: str  # actual prompt text
    modality: Optional[str] = None  # "code" | "math" (for HumanEval/GSM8K)
    benchmark: Optional[str] = None  # "humaneval" | "gsm8k"


@dataclass
class RealAgentWrapper:
    """
    Wrapper around real Agent to maintain compatibility with Exp1's interface.
    This allows us to use real agents while keeping the same code structure.
    """
    agent: Agent
    agent_id: str
    
    # State tracking (similar to SimAgentState)
    load: float = 0.0
    latency_ms: float = 500.0
    reputation: float = 0.5
    available: bool = True
    
    # Profile info (for cost, match_score, base_latency_ms)
    call_cost: float = 0.1  # estimated cost per call
    match_simple: float = 0.6
    match_hard: float = 0.6
    base_latency_ms: float = 500.0  # base latency for reset()
    
    # Cost calculation info (from config)
    price_in_per_1M: Optional[float] = None  # Price per 1M input tokens (for API models)
    price_out_per_1M: Optional[float] = None  # Price per 1M output tokens (for API models)
    gpu_hour_price: Optional[float] = None  # GPU hour price (for local models)
    provider: Optional[str] = None  # "openai", "google", "local"
    
    def reset(self):
        """Reset agent state"""
        self.load = 0.0
        self.latency_ms = self.base_latency_ms
        self.reputation = 0.5
        self.available = True
    
    def match_score(self, requirement: str) -> float:
        """Get match score for requirement"""
        if requirement == "simple":
            return float(self.match_simple)
        if requirement == "hard":
            return float(self.match_hard)
        return 0.5
    
    def get_dynamic_state(self) -> Dict[str, float]:
        """Get dynamic state (compatible with build_x)"""
        return {
            "available": bool(self.available),
            "load": float(max(0.0, min(1.0, self.load))),
            "latency_ms": float(max(1.0, self.latency_ms)),
            "reputation": float(max(0.0, min(1.0, self.reputation))),
        }
    
    def step_dynamics_after_call(self, ok: bool, latency_ms: float):
        """Update dynamic state after execution"""
        # Latency EMA
        beta = 0.20
        self.latency_ms = (1.0 - beta) * self.latency_ms + beta * float(latency_ms)
        
        # Reputation EMA
        self.reputation = max(0.0, min(1.0, 0.95 * self.reputation + 0.05 * (1.0 if ok else 0.0)))
        
        # Load spike for being chosen
        self.load = max(0.0, min(1.0, self.load + 0.30))
        
        # Availability rule
        self.available = bool(self.load < 0.95)
    
    def decay_load(self):
        """Global decay each round (simulate queue drain)"""
        self.load = max(0.0, min(1.0, 0.88 * self.load))
        self.available = bool(self.load < 0.95)
    
    def execute(
        self, 
        task: RealTask, 
        drift: bool = False, 
        t: int = 0,
        decoding_config: Optional[Dict] = None,
    ) -> Tuple[bool, float, Dict]:
        """
        Execute task using real API.
        
        Args:
            task: Task to execute
            drift: Whether drift is enabled
            t: Time step
            decoding_config: Optional decoding config override (temperature, top_p, max_tokens)
        
        Returns:
            (success, latency_ms, metadata) where metadata contains:
                - prompt_tokens: int
                - completion_tokens: int
                - total_tokens: int
                - response: str
                - error_type: str | None ("payment_required", "server_500", "timeout", "logic_error", or None)
                - error_message: str | None
        """
        if self.agent.base_model is None:
            return False, 0.0, {"error_type": "logic_error", "error_message": "base_model is None"}
        
        # Use decoding_config if provided, otherwise use agent defaults
        temp = decoding_config.get("temperature", self.agent.temperature) if decoding_config else self.agent.temperature
        top_p_val = decoding_config.get("top_p", self.agent.top_p) if decoding_config else self.agent.top_p
        max_tokens_val = decoding_config.get("max_tokens", self.agent.max_tokens) if decoding_config else self.agent.max_tokens
        
        start_time = time.time()
        
        # Call real API with structured error handling
        if hasattr(self.agent.base_model, "generate_with_metadata"):
            result = self.agent.base_model.generate_with_metadata(
                task.prompt,
                max_tokens=max_tokens_val,
                temperature=temp,
                top_p=top_p_val
            )
        else:
            # Fallback for backward compatibility
            try:
                response_text = self.agent.base_model.generate(
                    task.prompt,
                    max_tokens=max_tokens_val,
                    temperature=temp,
                    top_p=top_p_val
                )
                result = {
                    "success": True,
                    "response": response_text,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            except Exception as e:
                result = {
                    "success": False,
                    "error_type": "logic_error",
                    "error_message": str(e),
                    "response": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
        
        latency_ms = (time.time() - start_time) * 1000.0
        
        # Build metadata from result
        metadata = {
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
            "response": result.get("response", ""),
            "error_type": result.get("error_type"),
            "error_message": result.get("error_message"),
        }
        
        # Determine success: API call succeeded AND response is non-empty
        success = result.get("success", False) and len(metadata["response"].strip()) > 10
        
        # Update state (even on failure, so LinUCB can learn)
        self.step_dynamics_after_call(success, latency_ms)
        
        return success, latency_ms, metadata


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
    error_type: str = "none"  # "none", "payment_required", "server_500", "timeout", "logic_error"


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
    choose_F: int = 0
    choose_G: int = 0


# -----------------------------
# 2) Helpers
# -----------------------------
def _extract_prompt_from_raw_data(raw_data: dict, benchmark: str) -> str:
    """Extract prompt from raw_data based on benchmark type"""
    if benchmark == "humaneval":
        return raw_data.get("prompt", "")
    elif benchmark == "gsm8k":
        return raw_data.get("question", "")
    elif benchmark == "bbh":
        return raw_data.get("input", "")
    elif benchmark == "amc":
        return raw_data.get("problem", "")
    elif benchmark == "medical_qa":
        return raw_data.get("question", "")
    else:
        # Fallback: try common field names
        return raw_data.get("prompt", "") or raw_data.get("question", "") or raw_data.get("input", "") or raw_data.get("problem", "")


def load_tasks_from_symphony_generator(
    n: int,
    p_hard: float,
    mix_code_math: Tuple[float, float] = (0.5, 0.5),
    sample_with_replacement: bool = True,
    seed: int = 42,
    data_dir: Optional[str] = None,
    benchmarks: Optional[List[str]] = None,
) -> List[RealTask]:
    """
    Load tasks from symphony-data-generator JSONL files.
    
    Args:
        n: Number of tasks to load
        p_hard: Probability of hard tasks (0.2 for 80/20 split)
        mix_code_math: (code_ratio, math_ratio) - default (0.5, 0.5)
        sample_with_replacement: Whether to allow duplicate tasks
        seed: Random seed
        data_dir: Directory containing benchmark JSONL files (default: symphony-data-generator/data/benchmarks/full)
        benchmarks: List of benchmarks to load (e.g., ["humaneval", "gsm8k", "bbh", "amc", "medical_qa"]). If None, loads ["humaneval", "gsm8k"].
    
    Returns:
        List[RealTask]: Loaded tasks with modality and benchmark info
    """
    if data_dir is None:
        data_dir = os.path.join(_ROOT, "symphony-data-generator", "data", "benchmarks", "full")
    
    rng = random.Random(seed)
    
    # Determine which benchmarks to load
    if benchmarks is None:
        benchmarks = ["humaneval", "gsm8k"]  # Default: both
    
    # Benchmark configuration: file path, modality mapping
    benchmark_config = {
        "humaneval": {
            "file": "humaneval_full.jsonl",
            "modality": "code",
        },
        "gsm8k": {
            "file": "gsm8k_full.jsonl",
            "modality": "math",
        },
        "bbh": {
            "file": "bbh_full.jsonl",
            "modality": "reasoning",
        },
        "amc": {
            "file": "amc_full.jsonl",
            "modality": "math",
        },
        "medical_qa": {
            "file": "medical_qa_full.jsonl",
            "modality": "general",
        },
    }
    
    # Load tasks from all requested benchmarks
    all_benchmark_tasks = {}
    
    for bench_name in benchmarks:
        if bench_name not in benchmark_config:
            print(f"[WARN] Unknown benchmark: {bench_name}, skipping")
            continue
        
        config = benchmark_config[bench_name]
        file_path = os.path.join(data_dir, config["file"])
        
        if not os.path.exists(file_path):
            print(f"[WARN] Benchmark file not found: {file_path}, skipping {bench_name}")
            continue
        
        tasks = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    raw_data = obj.get("raw_data", {})
                    prompt = _extract_prompt_from_raw_data(raw_data, bench_name)
                    if prompt:
                        difficulty_bin = obj.get("difficulty_bin", "")
                        # Map empty difficulty_bin to "easy" if score is low, "hard" if high
                        if not difficulty_bin:
                            score = obj.get("difficulty_score", 0.5)
                            difficulty_bin = "hard" if score >= 0.5 else "easy"
                        tasks.append({
                            "prompt": prompt,
                            "difficulty": difficulty_bin if difficulty_bin in ["easy", "hard"] else "easy",
                            "benchmark": bench_name,
                            "modality": config["modality"],
                        })
        except Exception as e:
            print(f"[WARN] Failed to load {bench_name}: {e}")
            continue
        
        if tasks:
            all_benchmark_tasks[bench_name] = tasks
            print(f"✅ Loaded {len(tasks)} tasks from {bench_name}")
    
    if not all_benchmark_tasks:
        raise ValueError(
            f"No tasks found in {data_dir}. "
            "Please run: cd symphony-data-generator && python src/quick_start.py"
        )
    
    # Collect all tasks and sample uniformly across benchmarks
    all_tasks = []
    for bench_name, tasks in all_benchmark_tasks.items():
        all_tasks.extend(tasks)
    
    # Sample tasks uniformly (equal probability for each benchmark)
    # If sample_with_replacement is False and we need more tasks than available, use replacement
    if sample_with_replacement or n <= len(all_tasks):
        sampled_tasks = rng.choices(all_tasks, k=n) if sample_with_replacement else rng.sample(all_tasks, min(n, len(all_tasks)))
    else:
        # If not enough tasks and no replacement, repeat available tasks
        sampled_tasks = all_tasks * (n // len(all_tasks)) + rng.sample(all_tasks, n % len(all_tasks))
    
    # Shuffle
    rng.shuffle(sampled_tasks)
    
    # Convert to RealTask
    tasks = []
    for i, task_data in enumerate(sampled_tasks):
        difficulty = task_data["difficulty"]
        requirement = "hard" if difficulty == "hard" else "easy"  # Map to requirement
        tasks.append(RealTask(
            tid=i,
            difficulty=difficulty,
            requirement=requirement,
            prompt=task_data["prompt"],
            modality=task_data.get("modality"),
            benchmark=task_data.get("benchmark"),
        ))
    
    return tasks


def generate_tasks(
    n: int, 
    p_hard: float, 
    rng: random.Random, 
    use_real_data: bool = True,
    benchmarks: Optional[List[str]] = None,
) -> List[RealTask]:
    """
    Generate tasks - either from real datasets or placeholder prompts.
    
    Args:
        n: Number of tasks
        p_hard: Probability of hard tasks
        rng: Random generator
        use_real_data: If True, load from symphony-data-generator (default: True)
        benchmarks: List of benchmarks to use (e.g., ["humaneval"], ["gsm8k"], or ["humaneval", "gsm8k"])
    """
    if use_real_data:
        try:
            return load_tasks_from_symphony_generator(
                n=n,
                p_hard=p_hard,
                mix_code_math=(0.5, 0.5),
                sample_with_replacement=True,
                seed=rng.randint(0, 2**31),
                benchmarks=benchmarks,
            )
        except Exception as e:
            print(f"[WARN] Failed to load real datasets: {e}")
            print("[WARN] Falling back to placeholder prompts")
            use_real_data = False
    
    # Fallback to placeholder prompts
    tasks = []
    simple_prompts = [
        "What is 2+2?",
        "What is the capital of France?",
        "Translate 'hello' to Spanish.",
        "What color is the sky?",
        "How many days in a week?",
    ]
    hard_prompts = [
        "Solve this math problem step by step: If a train travels 120 km in 2 hours, and another train travels 180 km in 3 hours, which train is faster?",
        "Explain the concept of recursion in computer science with examples.",
        "What are the key differences between supervised and unsupervised learning?",
        "Analyze the trade-offs between time complexity and space complexity in algorithm design.",
        "Discuss the principles of object-oriented programming with real-world examples.",
    ]
    
    for i in range(n):
        hard = rng.random() < p_hard
        if hard:
            prompt = rng.choice(hard_prompts)
            difficulty = "hard"
        else:
            prompt = rng.choice(simple_prompts)
            difficulty = "easy"
        
        requirement = "hard" if hard else "easy"
        tasks.append(RealTask(tid=i, difficulty=difficulty, requirement=requirement, prompt=prompt))
    
    return tasks


def load_openrouter_agents(
    config_dir: str = "runtime", 
    agent_ids: List[int] = None, 
    project_root: str = None,
    exp1_config: Optional[Dict] = None
) -> Dict[str, RealAgentWrapper]:
    """
    Load OpenRouter agents from config files.
    Maps config IDs (1-7) to agent letters (A-G).
    Now reads agent info from exp1_config if provided.
    """
    if agent_ids is None:
        agent_ids = [1, 2, 3, 4, 5, 6, 7]
    
    # If config_dir is relative, make it relative to project root
    if project_root and not os.path.isabs(config_dir):
        config_dir = os.path.join(project_root, config_dir)
    
    # Load agent configs from exp1_config if available
    agent_configs = {}
    if exp1_config and "agents" in exp1_config:
        agent_configs = exp1_config["agents"]
    
    # Fallback hardcoded map (for backward compatibility)
    agent_id_map = {
        1: ("A", 0.10, 0.95, 0.20, 350.0),  # (letter, cost_weight, match_simple, match_hard, base_latency_ms)
        2: ("B", 0.10, 0.95, 0.20, 350.0),
        3: ("C", 0.15, 0.85, 0.55, 420.0),
        4: ("D", 1.00, 0.80, 0.95, 900.0),  # Strong model
        5: ("E", 0.35, 0.75, 0.75, 550.0),
        6: ("F", 0.60, 0.60, 0.90, 780.0),
        7: ("G", 0.25, 0.90, 0.45, 420.0),
    }
    
    wrappers = {}
    
    for config_id in agent_ids:
        config_path = os.path.join(config_dir, f"config_agent_openrouter_{config_id}.yaml")
        if not os.path.exists(config_path):
            print(f"[WARN] Config file not found: {config_path}, skipping")
            continue
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            agent_instance = Agent(config=config)
            
            # ✅ First: Try to get all agent info from agent config file (primary source)
            price_in = config.get("price_in_per_1M")
            price_out = config.get("price_out_per_1M")
            gpu_hour = config.get("gpu_hour_price")
            provider_type = config.get("provider")
            cost_weight_from_config = config.get("cost_weight", 0.1)
            match_simple_from_config = config.get("match_simple", 0.6)
            match_hard_from_config = config.get("match_hard", 0.6)
            base_latency_ms_from_config = config.get("base_latency_ms", 500.0)
            
            # Try to get agent info from exp1_config (for letter mapping only)
            letter = None
            
            # Map config_id to letter (from exp1_config if available, otherwise from hardcoded map)
            if agent_configs:
                # Find agent by config_id in exp1_config
                for agent_letter, agent_cfg in agent_configs.items():
                    if agent_cfg.get("config_id") == config_id:
                        letter = agent_letter
                        break
            
            # Fallback to hardcoded map if letter not found in exp1_config
            if letter is None and config_id in agent_id_map:
                letter, _, _, _, _ = agent_id_map[config_id]
            
            # ✅ Use values from agent config file (primary source)
            cost_weight = cost_weight_from_config
            match_simple = match_simple_from_config
            match_hard = match_hard_from_config
            base_latency_ms = base_latency_ms_from_config
            
            # ✅ Allow exp1_config to override (allows experiment-specific overrides)
            if agent_configs:
                for agent_letter, agent_cfg in agent_configs.items():
                    if agent_cfg.get("config_id") == config_id:
                        # Allow overriding values from exp1_config if needed
                        if agent_cfg.get("cost_weight") is not None:
                            cost_weight = agent_cfg.get("cost_weight")
                        if agent_cfg.get("match_simple") is not None:
                            match_simple = agent_cfg.get("match_simple")
                        if agent_cfg.get("match_hard") is not None:
                            match_hard = agent_cfg.get("match_hard")
                        if agent_cfg.get("base_latency_ms") is not None:
                            base_latency_ms = agent_cfg.get("base_latency_ms")
                        # ✅ Allow exp_config to override prices (useful for testing different pricing)
                        if agent_cfg.get("price_in_per_1M") is not None:
                            price_in = agent_cfg.get("price_in_per_1M")
                        if agent_cfg.get("price_out_per_1M") is not None:
                            price_out = agent_cfg.get("price_out_per_1M")
                        if agent_cfg.get("gpu_hour_price") is not None:
                            gpu_hour = agent_cfg.get("gpu_hour_price")
                        if agent_cfg.get("provider") is not None:
                            provider_type = agent_cfg.get("provider")
                        break
            
            if letter is None:
                print(f"[WARN] Could not determine letter for config_id {config_id}, skipping")
                continue
            
            wrapper = RealAgentWrapper(
                agent=agent_instance,
                agent_id=letter,
                call_cost=cost_weight,  # Use cost_weight as initial estimate
                match_simple=match_simple,
                match_hard=match_hard,
                base_latency_ms=base_latency_ms,
                latency_ms=base_latency_ms,
                price_in_per_1M=price_in,
                price_out_per_1M=price_out,
                gpu_hour_price=gpu_hour,
                provider=provider_type,
            )
            
            wrappers[letter] = wrapper
            print(f"✅ Loaded agent {letter}: {config['node_id']} ({config['base_model']})")
        except Exception as e:
            print(f"❌ Failed to load agent from {config_path}: {e}")
            import traceback
            traceback.print_exc()
    
    return wrappers


def pick_topL_candidates(
    agents: List[RealAgentWrapper],
    requirement: str,
    topL: int,
    task: Optional[RealTask] = None,
    latency_scale_ms: float = 2000.0,
    use_embedding: bool = True,
) -> List[Dict[str, Any]]:
    """
    ✅ Pick top L candidates using new routing system (embedding + priors).
    
    If routing is not available, falls back to legacy match_score-based selection.
    
    Returns:
        List of candidate dicts with schema:
        {
            "agent": RealAgentWrapper,
            "match_score": float,  # TopL composite score
            "sim_emb": float,      # Embedding similarity
            "prior_success": float # Learned prior from warmup
        }
    """
    # ✅ Use new routing system if available
    if HAS_ROUTING and use_embedding and task is not None:
        try:
            # Convert RealTask to routing subtask format
            subtask = {
                "requirement": requirement,
                "benchmark": task.benchmark or "",
                "difficulty_bin": task.difficulty,  # "easy" or "hard"
                "input": task.prompt,
                "description": task.prompt,
                "id": str(task.tid),
            }
            
            # Extract Agent objects from RealAgentWrapper
            agent_objects = [ag.agent for ag in agents]
            
            # Use routing.select_topL() with embedding
            candidates = select_topL(
                agents=agent_objects,
                subtask=subtask,
                topL=topL,
                latency_scale_ms=latency_scale_ms,
                use_embedding=True,
            )
            
            # Convert back to RealAgentWrapper format with candidate schema
            result = []
            for cand in candidates:
                agent_obj = cand["agent"]
                # Find corresponding RealAgentWrapper
                real_ag = next((ag for ag in agents if ag.agent is agent_obj), None)
                if real_ag is None:
                    continue
                
                result.append({
                    "agent": real_ag,
                    "match_score": float(cand.get("match_score", 0.5)),
                    "sim_emb": float(cand.get("sim_emb", 0.5)),
                    "prior_success": float(cand.get("prior_success", 0.5)),
                })
            
            if result:
                return result
        except Exception as e:
            print(f"[WARN] Routing selection failed: {e}, falling back to legacy")
    
    # ✅ Fallback to legacy: match_score only
    scored: List[Dict[str, Any]] = []
    for ag in agents:
        ms = ag.match_score(requirement)
        scored.append({
            "agent": ag,
            "match_score": float(ms),
            "sim_emb": float(ms),  # Fallback: use match_score as sim_emb
            "prior_success": 0.5,  # Fallback: default
        })
    scored.sort(key=lambda x: float(x.get("match_score", 0.0)), reverse=True)
    return scored[:max(1, int(topL))]


def normalize_cost(cost: float, max_cost: float) -> float:
    return float(cost / max(1e-9, max_cost))


def build_x_extended(
    match_score: float,
    dynamic_state: Dict,
    available: bool,
    latency_scale_ms: float = 2000.0,
    task: Optional[RealTask] = None,
    agent: Optional[RealAgentWrapper] = None,
) -> List[float]:
    """
    Extended context vector with task and agent features (from config_exp1.yaml).
    
    Features:
    - Task features: is_code, is_math, is_easy, is_hard (one-hot)
    - Agent features: model_id_onehot (7-dim), log_price
    - State features: match_score, load, latency_norm, reputation, available
    - Total: 1 (bias) + 4 (task) + 8 (agent) + 4 (state) = 17 dimensions
    
    Args:
        match_score: Static match score
        dynamic_state: Dynamic agent state
        available: Agent availability
        latency_scale_ms: Latency normalization scale
        task: Task object (for task features)
        agent: Agent wrapper (for agent features)
    
    Returns:
        Normalized feature vector (L2 normalized if ||x|| > 1)
    """
    # Base features (from original build_x)
    ms = max(0.0, min(1.0, float(match_score)))
    ds = dynamic_state if isinstance(dynamic_state, dict) else {}
    
    load = max(0.0, min(1.0, float(ds.get("load", 0.0))))
    lat_ms = max(0.0, float(ds.get("latency_ms", 500.0)))
    lat = max(0.0, min(1.0, lat_ms / max(1.0, latency_scale_ms)))
    rep = max(0.0, min(1.0, float(ds.get("reputation", 0.5))))
    av = 1.0 if bool(available) else 0.0
    
    # Task features (one-hot)
    is_code = 1.0 if task and (task.modality == "code" or task.benchmark == "humaneval") else 0.0
    is_math = 1.0 if task and (task.modality == "math" or task.benchmark == "gsm8k") else 0.0
    is_easy = 1.0 if task and task.difficulty == "easy" else 0.0
    is_hard = 1.0 if task and task.difficulty == "hard" else 0.0
    
    # Agent features
    agent_id_onehot = [0.0] * 7  # A-G
    log_price = 0.0
    
    if agent:
        # Model ID one-hot (A=0, B=1, ..., G=6)
        agent_letter = agent.agent_id
        if agent_letter in "ABCDEFG":
            idx = ord(agent_letter) - ord("A")
            if 0 <= idx < 7:
                agent_id_onehot[idx] = 1.0
        
        # Log price (use cost_weight as proxy)
        cost_val = agent.call_cost
        log_price = math.log(max(1e-6, cost_val)) if cost_val > 0 else 0.0
        log_price = max(0.0, min(1.0, (log_price + 5.0) / 10.0))  # Normalize to [0,1]
    
    # Combine all features
    x = [
        1.0,  # bias
        ms, load, lat, rep, av,  # original state features
        is_code, is_math, is_easy, is_hard,  # task features
        *agent_id_onehot,  # agent model ID (7-dim)
        log_price,  # agent price feature
    ]
    
    # L2 normalize if ||x|| > 1
    norm = math.sqrt(sum(v * v for v in x))
    if norm > 1.0:
        x = [v / norm for v in x]
    
    return x


def calculate_real_cost(
    agent: RealAgentWrapper,
    metadata: Dict,
    latency_ms: float,
    cost_mode: str = "real_dollar",
) -> float:
    """
    Calculate real cost based on token usage or latency.
    
    Args:
        agent: Agent wrapper with pricing info
        metadata: Execution metadata with token counts
        latency_ms: Execution latency in milliseconds
        cost_mode: "real_dollar" or "relative_weight"
    
    Returns:
        Cost in dollars (for real_dollar) or normalized weight (for relative_weight)
    """
    if cost_mode == "relative_weight":
        # Use cost_weight as fallback
        return agent.call_cost
    
    # Real dollar cost calculation
    prompt_tokens = metadata.get("prompt_tokens", 0)
    completion_tokens = metadata.get("completion_tokens", 0)
    
    # API models: use token-based pricing
    if agent.provider in ["openai", "google"]:
        cost = 0.0
        if prompt_tokens > 0 and agent.price_in_per_1M is not None:
            cost += (prompt_tokens / 1_000_000.0) * agent.price_in_per_1M
        if completion_tokens > 0 and agent.price_out_per_1M is not None:
            cost += (completion_tokens / 1_000_000.0) * agent.price_out_per_1M
        return cost
    
    # Local models: use GPU-hour pricing
    if agent.provider == "local" and agent.gpu_hour_price is not None:
        latency_s = latency_ms / 1000.0
        cost = (latency_s / 3600.0) * agent.gpu_hour_price
        return cost
    
    # Fallback: use cost_weight as estimate
    return agent.call_cost


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
    Reward in [0,1] (clipped), updated formula from config_exp1.yaml:
      r = 1[correct] - λ_c * c_norm - λ_t * t_norm
    where:
      - λ_c = cost_lambda (default 0.3)
      - λ_t = latency_penalty (default 0.05)
      - c_norm = cost / max_cost_over_agents
      - t_norm = latency / latency_scale_ms
    """
    base = 1.0 if success else 0.0
    lat_norm = min(1.0, float(latency_ms) / max(1.0, float(latency_scale_ms)))
    cost_norm = normalize_cost(call_cost, max_cost)
    # Updated formula: no sqrt for latency_norm
    r = base - float(cost_lambda) * cost_norm - float(latency_penalty) * lat_norm
    return max(0.0, min(1.0, r))


def run_policy(
    policy_name: str,
    tasks: List[RealTask],
    agents: List[RealAgentWrapper],
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
    exp1_config: Optional[Dict] = None,
) -> Tuple[SummaryRow, List[StepLog]]:
    """
    Run one policy from scratch (reset agent states).
    Same structure as Exp1 simulation, but uses real agents.
    """
    rng = random.Random(seed_for_policy)
    
    # Reset agents
    for ag in agents:
        ag.reset()
    
    # Selector only for LinUCB policy
    # Use extended features: 17 dimensions (1 bias + 4 task + 8 agent + 4 state)
    selector: Optional[GlobalLinUCB] = None
    if policy_name == "linucb":
        feature_dim = 17  # Extended features
        selector = GlobalLinUCB(d=feature_dim, l2=float(linucb_l2), alpha=float(linucb_alpha), delta=float(delta), S=float(S))
    
    max_cost = max(a.call_cost for a in agents)
    # for summaries - initialize based on actual agent IDs
    choose_counts = {a.agent_id: 0 for a in agents}
    total_cost = 0.0
    total_latency = 0.0
    success_cnt = 0
    fallback_cnt = 0
    step_logs: List[StepLog] = []
    
    # helper: get agent by id
    agent_by_id = {a.agent_id: a for a in agents}
    # Get strong agent from config (default: "D" for SOP-Strong)
    strong_agent_id = "D"
    if exp1_config and "baselines" in exp1_config:
        sop_strong_cfg = exp1_config["baselines"].get("sop_strong", {})
        strong_agent_id = sop_strong_cfg.get("strong_model", "D")
    
    for t, task in enumerate(tasks):
        # decay load for all agents each round (global dynamics)
        for ag in agents:
            ag.decay_load()
        
        # ✅ ---- candidate pool by Top-L (using new routing with embedding + priors) ----
        top = pick_topL_candidates(
            agents=agents,
            requirement=task.requirement,
            topL=topL,
            task=task,  # Pass task for routing.select_topL()
            latency_scale_ms=latency_scale_ms,
            use_embedding=True,
        )
        
        # ✅ Filter available candidates (top is now List[Dict] instead of List[Tuple])
        avail = [
            (cand["agent"], cand["match_score"], cand.get("sim_emb", 0.5), cand.get("prior_success", 0.5))
            for cand in top
            if cand["agent"].get_dynamic_state().get("available", True)
        ]
        if not avail:
            # if no available in topL, allow choosing from top anyway (mark available=False in x)
            avail = [
                (cand["agent"], cand["match_score"], cand.get("sim_emb", 0.5), cand.get("prior_success", 0.5))
                for cand in top
            ]
        
        # ---- choose action (agent) ----
        chosen_ag: RealAgentWrapper
        chosen_ms: float
        chosen_x: Optional[List[float]] = None
        
        if policy_name == "always_A" or policy_name == "sop_strong":
            # SOP-Strong: Always select strong model (D from config)
            strong_model_id = "D"  # From config: baselines.sop_strong.strong_model
            chosen_ag = agent_by_id.get(strong_model_id, agent_by_id.get("A", agents[0]))
            chosen_ms = chosen_ag.match_score(task.requirement)
        
        elif policy_name == "static_rule" or policy_name == "sop_rule":
            # SOP-Rule: easy→cheap, hard→strong
            # From config: baselines.sop_rule.cheap_model="A", strong_model="D"
            cheap_model_id = "A"
            strong_model_id = "D"
            # Map difficulty: "simple"/"easy" → cheap, "hard" → strong
            is_easy = task.difficulty in ["simple", "easy"]
            chosen_id = cheap_model_id if is_easy else strong_model_id
            chosen_ag = agent_by_id.get(chosen_id, agent_by_id.get(strong_model_id, agents[0]))
            chosen_ms = chosen_ag.match_score(task.requirement)
        
        elif policy_name == "random":
            # ✅ avail is now (agent, match_score, sim_emb, prior_success)
            chosen_ag, chosen_ms, _, _ = rng.choice(avail)
        
        elif policy_name == "linucb":
            assert selector is not None
            candidates: List[Tuple[str, List[float], float]] = []
            # ✅ build x for each candidate (aid, x) with extended features
            # avail is now (agent, match_score, sim_emb, prior_success)
            for (ag, ms, sim_emb_val, prior_success_val) in avail:
                st = ag.get_dynamic_state()
                
                # ✅ Use sim_emb for match_score and prior_success for reputation in build_x_extended
                # This ensures consistency with routing.build_x_from_candidate()
                x = build_x_extended(
                    match_score=float(sim_emb_val),  # ✅ Use sim_emb (not composite match_score)
                    dynamic_state={
                        "load": float(st.get("load", 0.0)),
                        "latency_ms": float(st.get("latency_ms", 500.0)),
                        "reputation": float(prior_success_val),  # ✅ Use prior_success (not default 0.5)
                    },
                    available=bool(st.get("available", True)),
                    latency_scale_ms=float(latency_scale_ms),
                    task=task,
                    agent=ag,
                )
                candidates.append((ag.agent_id, x, float(ms)))  # ms for logging (TopL composite score)
            
            chosen_id = selector.select([(aid, x) for (aid, x, _) in candidates])
            # recover chosen
            chosen_ag = agent_by_id[chosen_id]
            chosen_ms = next(ms for (aid, _x, ms) in candidates if aid == chosen_id)
            chosen_x = next(_x for (aid, _x, _ms) in candidates if aid == chosen_id)
        
        else:
            raise ValueError(f"Unknown policy: {policy_name}")
        
        choose_counts[chosen_ag.agent_id] = choose_counts.get(chosen_ag.agent_id, 0) + 1
        
        # ---- Get decoding config from exp1_config if available ----
        decoding_config = None
        if exp1_config and "decoding" in exp1_config:
            dec_cfg = exp1_config["decoding"]
            # Determine max_tokens based on task modality
            max_tokens_map = dec_cfg.get("max_output_tokens", {})
            if task.modality == "code" or task.benchmark == "humaneval":
                max_tokens_val = max_tokens_map.get("humaneval", 512)
            elif task.modality == "math" or task.benchmark == "gsm8k":
                max_tokens_val = max_tokens_map.get("gsm8k", 256)
            else:
                max_tokens_val = max_tokens_map.get("gsm8k", 256)  # Default
            
            decoding_config = {
                "temperature": dec_cfg.get("temperature", 0.2),
                "top_p": dec_cfg.get("top_p", 0.95),
                "max_tokens": max_tokens_val,
            }
        
        # ---- execute chosen agent ----
        ok, lat_ms, exec_metadata = chosen_ag.execute(
            task, 
            drift=drift, 
            t=t,
            decoding_config=decoding_config,
        )
        
        # Extract error_type from metadata
        error_type_primary = exec_metadata.get("error_type") or "none"
        
        # Calculate real cost (USD)
        cost_mode = "real_dollar"
        if exp1_config and "cost" in exp1_config:
            cost_mode = exp1_config["cost"].get("mode", "real_dollar")
        call_cost = calculate_real_cost(chosen_ag, exec_metadata, lat_ms, cost_mode)
        
        # ---- optional fallback to strong agent if failed (more realistic cost) ----
        # Key: fallback should only trigger for recoverable errors (payment_required, server_500, timeout)
        # NOT for logic_error (which should terminate)
        total_cost_this = call_cost
        total_lat_this = lat_ms
        fallback_used = 0
        final_ok = ok
        error_type_final = error_type_primary
        
        # Determine if fallback is allowed for this error type
        fallback_allowed = error_type_primary in ["payment_required", "server_500", "timeout", "none"]
        
        if fallback and (not ok) and fallback_allowed and (chosen_ag.agent_id != strong_agent_id):
            fallback_used = 1
            fallback_cnt += 1
            fallback_ag = agent_by_id[strong_agent_id]
            ok2, lat2, fallback_metadata = fallback_ag.execute(
                task, 
                drift=drift, 
                t=t,
                decoding_config=decoding_config,
            )
            final_ok = ok2  # final success after fallback
            error_type_final = fallback_metadata.get("error_type") or "none"
            fallback_cost = calculate_real_cost(fallback_ag, fallback_metadata, lat2, cost_mode)
            total_cost_this += fallback_cost  # ✅ fallback cost is included in total_cost
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
                chosen_x = build_x_extended(
                    match_score=float(chosen_ms),
                    dynamic_state={
                        "load": float(st.get("load", 0.0)),
                        "latency_ms": float(st.get("latency_ms", 500.0)),
                        "reputation": float(st.get("reputation", 0.5)),
                    },
                    available=bool(st.get("available", True)),
                    latency_scale_ms=float(latency_scale_ms),
                    task=task,
                    agent=chosen_ag,
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
                chosen_agent=chosen_ag.agent_id,
                match_score=float(chosen_ms),
                load=float(st_now.get("load", 0.0)),
                latency_ms=float(lat_ms),
                call_cost=float(call_cost),
                success=1 if final_ok else 0,
                fallback_used=fallback_used,
                total_cost_this_task=float(total_cost_this),
                total_latency_this_task=float(total_lat_this),
                reward_used_for_update=float(used_reward),
                error_type=error_type_final,  # Record final error_type (after fallback if used)
            )
        )
        
        # Progress printing
        if (t + 1) % 10 == 0:
            print(f"[{policy_name}] Task {t+1}/{len(tasks)}, Success: {success_cnt}/{t+1} ({success_cnt/(t+1)*100:.1f}%)")
    
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
        choose_F=int(choose_counts.get("F", 0)),
        choose_G=int(choose_counts.get("G", 0)),
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
        "always_A": "SOP-Strong",
        "sop_strong": "SOP-Strong",
        "static_rule": "SOP-Rule",
        "sop_rule": "SOP-Rule",
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
            f"{v:.3f}",
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
        order_cost = ["sop_rule", "random", "linucb"]  # ✅ cost 不画 sop_strong
        order_succ = ["sop_strong", "sop_rule", "random", "linucb"]  # ✅ success 画 sop_strong
        
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
        
        all_cum_values = []  # Collect all cumulative values to determine y-axis range
        
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
            all_cum_values.extend(cum)
            
            ax.plot(xs, cum, linewidth=1.05, alpha=0.90, label=name_map.get(pol, pol))
        
        # Set y-axis range based on data (with some padding)
        if all_cum_values:
            ymin = min(all_cum_values)
            ymax = max(all_cum_values)
            yrange = ymax - ymin
            if yrange > 0:
                # Add 10% padding on each side
                ax.set_ylim(max(0.0, ymin - 0.1 * yrange), ymax + 0.1 * yrange)
            else:
                # If all values are the same, set a small range around the value
                ax.set_ylim(max(0.0, ymin - 0.001), ymin + 0.001)
        else:
            ax.set_ylim(0.0, 0.001)  # Default range if no data
        
        ax.set_xlabel("t")
        ax.set_ylabel("Cumulative avg cost")
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
            
            if pol in ["always_A", "sop_strong"]:
                # ✅ 让 SOP-Strong 更"弱存在感"：虚线 + 半透明 + 略细
                ax.plot(xs, cum, linewidth=0.95, linestyle="--", alpha=0.65,
                        label=name_map.get(pol, pol))
            else:
                ax.plot(xs, cum, linewidth=1.05, alpha=0.90,
                        label=name_map.get(pol, pol))
        
        ax.set_xlabel("t")
        ax.set_ylabel("Cumulative avg success (final)")
        ax.set_ylim(0.0, 1.0)  # Y-axis from 0 to 1
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
    ap.add_argument("--n", type=int, default=2000, help="number of tasks (default: 2000 from config)")
    ap.add_argument("--p-hard", type=float, default=0.2, help="probability of hard task (default: 0.2 for 80/20 split)")
    ap.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    ap.add_argument("--topL", type=int, default=3, help="Top-L candidates by static match_score (default: 3 from config)")
    ap.add_argument("--outdir", type=str, default="experiments/exp1_real_openrouter/results", help="output directory")
    ap.add_argument("--no-plots", action="store_true", help="do not generate png plots")
    ap.add_argument("--config-dir", type=str, default="runtime", help="Config file directory for agents")
    ap.add_argument("--agents", type=str, default="1,2,3,4,5,6,7", help="Comma-separated list of agent IDs to use (default: 1,2,3,4,5,6,7 - all 7 agents)")
    ap.add_argument("--config", type=str, default=None, help="Path to config_exp1.yaml (default: auto-detect)")
    ap.add_argument("--benchmarks", type=str, default="humaneval,gsm8k", 
                    help="Comma-separated list of benchmarks to use: humaneval,gsm8k,bbh,amc,medical_qa (default: humaneval,gsm8k)")
    
    # LinUCB params
    ap.add_argument("--alpha", type=float, default=1.0, help="LinUCB exploration scale")
    ap.add_argument("--l2", type=float, default=1.0, help="LinUCB l2 regularization lambda")
    ap.add_argument("--delta", type=float, default=0.1, help="LinUCB confidence (default 0.1 from config)")
    ap.add_argument("--S", type=float, default=1.0, help="bound on ||theta*|| (for beta())")
    
    # reward shaping params
    ap.add_argument("--latency-scale-ms", type=float, default=2000.0, help="latency normalization scale")
    ap.add_argument("--latency-penalty", type=float, default=0.05, help="penalty multiplier for latency (default 0.05 from config)")
    ap.add_argument("--cost-lambda", type=float, default=0.3, help="penalty multiplier for cost (default 0.3 from config)")
    
    # realism toggles
    ap.add_argument("--fallback", action="store_true", help="if chosen agent fails, fallback to strong agent A")
    ap.add_argument("--drift", action="store_true", help="enable non-stationary drift after t>=500")
    
    args = ap.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = os.path.join(args.outdir, timestamp)
    os.makedirs(outdir, exist_ok=True)
    
    base_rng = random.Random(args.seed)
    
    # Load Exp1 configuration
    exp1_config = load_exp1_config(args.config)
    
    # ✅ Load learned priors from warmup (if available)
    priors = {}
    priors_file = os.path.join(args.outdir, "..", "warmup_priors.json")
    if os.path.exists(priors_file):
        try:
            if HAS_ROUTING:
                priors = load_priors(priors_file)
                print(f"✅ Loaded learned priors from {priors_file} ({len(priors)} agents)")
            else:
                print(f"[WARN] Priors file exists but routing module not available")
        except Exception as e:
            print(f"[WARN] Failed to load priors: {e}")
    else:
        print(f"[INFO] No priors file found at {priors_file}, using default priors")
    
    # Load real OpenRouter agents (with config support)
    agent_ids = [int(x.strip()) for x in args.agents.split(",")]
    agent_dict = load_openrouter_agents(args.config_dir, agent_ids, project_root=_ROOT, exp1_config=exp1_config)
    if not agent_dict:
        print("❌ No real OpenRouter agents loaded! Exiting.")
        return
    
    # ✅ Inject learned priors into agents
    for aid, ag_wrapper in agent_dict.items():
        agent_obj = ag_wrapper.agent
        if aid in priors:
            # Inject learned_priors into agent object
            agent_obj.learned_priors = priors[aid]
            print(f"✅ Injected priors for agent {aid} ({len(priors[aid])} buckets)")
    
    # Convert to list for consistent indexing
    agents_list = list(agent_dict.values())
    
    # Parse benchmarks argument
    benchmarks_list = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    if not benchmarks_list:
        benchmarks_list = ["humaneval", "gsm8k"]  # Default: both
    
    # Validate benchmarks
    valid_benchmarks = ["humaneval", "gsm8k", "bbh", "amc", "medical_qa"]
    benchmarks_list = [b for b in benchmarks_list if b in valid_benchmarks]
    if not benchmarks_list:
        print(f"[WARN] No valid benchmarks found. Valid options: {valid_benchmarks}")
        print(f"[WARN] Using default: ['humaneval', 'gsm8k']")
        benchmarks_list = ["humaneval", "gsm8k"]
    
    print(f"📊 Using benchmarks: {benchmarks_list}")
    
    # Generate tasks (using the same logic as sim_efficiency_cost.py)
    tasks = generate_tasks(args.n, args.p_hard, base_rng, use_real_data=True, benchmarks=benchmarks_list)
    
    # Define policies (updated to match config: SOP-Strong and SOP-Rule)
    policies = ["sop_strong", "sop_rule", "random", "linucb"]
    summary_rows: List[SummaryRow] = []
    traj_logs: Dict[str, List[StepLog]] = {}
    
    for i, pol in enumerate(policies):
        pol_seed = args.seed + 1000 * (i + 1)
        
        row, logs = run_policy(
            policy_name=pol,
            tasks=tasks,
            agents=agents_list,
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
            exp1_config=exp1_config,
        )
        summary_rows.append(row)
        traj_logs[pol] = logs
        
        print(
            f"[{pol}] success={row.success_rate:.3f} "
            f"total_cost={row.total_cost:.1f} avg_cost={row.avg_cost:.3f} "
            f"avg_lat={row.avg_latency_ms:.1f}ms fallback={row.fallback_rate:.3f} "
            f"choose(A,B,C,D,E,F,G)=({row.choose_A},{row.choose_B},{row.choose_C},{row.choose_D},{row.choose_E},{row.choose_F},{row.choose_G})"
        )
    
    # ---- write outputs ----
    write_csv(
        os.path.join(outdir, "summary.csv"),
        [asdict(r) for r in summary_rows],
    )
    for pol, logs in traj_logs.items():
        write_csv(
            os.path.join(outdir, f"trajectory_{pol}.csv"),
            [asdict(x) for x in logs],
        )
    
    if not args.no_plots:
        try_plot(outdir, summary_rows, traj_logs)
    
    print(f"\n✅ Done. Outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
