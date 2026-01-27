#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/common/task_loader.py

Unified task loader for all Symphony 2.0 experiments (Exp1-5).
Wraps DatasetBuilder for real benchmark tasks, with fallback to simulation-style tasks.

Usage:
    from experiments.common.task_loader import load_task_stream, TaskWrapper
    
    # Load real benchmark tasks
    tasks = load_task_stream("mixed_80_20", seed=123)
    
    # Or use simulation fallback
    tasks = load_task_stream("mixed_80_20", seed=123, use_simulation=True)
"""

from __future__ import annotations

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to path
_THIS_DIR = Path(__file__).parent
_EXP_DIR = _THIS_DIR.parent
_ROOT = _EXP_DIR.parent
sys.path.insert(0, str(_ROOT))

# Try to import DatasetBuilder from symphony-data-generator
try:
    sys.path.insert(0, str(_ROOT / "symphony-data-generator" / "src"))
    from data_generator import DatasetBuilder, Task
    HAS_DATA_GENERATOR = True
except ImportError as e:
    HAS_DATA_GENERATOR = False
    Task = None  # Will use TaskWrapper instead

# Import Exp1's generate_tasks for simulation fallback
try:
    from experiments.exp1_sim_efficiency_cost.sim_efficiency_cost import (
        generate_tasks as exp1_generate_tasks,
        SimTask,
    )
    HAS_EXP1 = True
except ImportError:
    HAS_EXP1 = False
    SimTask = None


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TaskWrapper:
    """
    Unified task wrapper that works with both real benchmark tasks and simulation tasks.
    Provides a consistent interface for all experiments.
    """
    task_id: str
    benchmark: str  # "humaneval", "gsm8k", or "simulation"
    difficulty_score: float
    difficulty_bin: str  # "easy" or "hard"
    raw_data: Dict[str, Any]
    
    # Additional fields for compatibility with Exp1 SimTask
    requirement: str = "simple"  # "simple" or "hard" for routing
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_data_generator_task(cls, task) -> "TaskWrapper":
        """Convert DatasetBuilder Task to TaskWrapper"""
        return cls(
            task_id=task.task_id,
            benchmark=task.benchmark,
            difficulty_score=task.difficulty_score,
            difficulty_bin=task.difficulty_bin,
            raw_data=task.raw_data,
            requirement=task.difficulty_bin,  # Map difficulty_bin to requirement
        )
    
    @classmethod
    def from_sim_task(cls, sim_task, idx: int) -> "TaskWrapper":
        """Convert Exp1 SimTask to TaskWrapper"""
        return cls(
            task_id=f"sim_{idx:05d}",
            benchmark="simulation",
            difficulty_score=1.0 if sim_task.difficulty == "hard" else 0.0,
            difficulty_bin=sim_task.difficulty if sim_task.difficulty in ("easy", "hard") else (
                "hard" if sim_task.difficulty == "hard" else "easy"
            ),
            raw_data={"tid": sim_task.tid, "difficulty": sim_task.difficulty},
            requirement=sim_task.requirement,
        )


# ============================================================================
# TASK STREAM CONFIGURATIONS
# ============================================================================

TASK_STREAMS = {
    "mixed_80_20": {
        "benchmarks": ["humaneval", "gsm8k"],
        "difficulty_split": "80:20",
        "n_tasks": 1000,
        "description": "Mixed-80:20 for Exp1, 3, 4 (80% easy, 20% hard)",
    },
    "balanced_50_50": {
        "benchmarks": ["humaneval", "gsm8k"],
        "difficulty_split": "50:50",
        "n_tasks": 1000,
        "description": "Balanced-50:50 for Exp2 (50% easy, 50% hard)",
    },
    "exp3_system_opt": {
        "benchmarks": ["humaneval", "gsm8k"],
        "difficulty_split": "80:20",
        "n_tasks": 1000,
        "description": "Task stream for Exp3 system optimization testing",
    },
}


# ============================================================================
# MAIN LOADER FUNCTIONS
# ============================================================================

def load_task_stream(
    stream_name: str,
    seed: int = 123,
    n_tasks: Optional[int] = None,
    cache_dir: Optional[str] = None,
    use_simulation: bool = False,
    force_regenerate: bool = False,
) -> List[TaskWrapper]:
    """
    Load a task stream for experiments.
    
    Args:
        stream_name: Task stream name (e.g., "mixed_80_20", "balanced_50_50")
        seed: Random seed for reproducibility
        n_tasks: Override number of tasks (uses config default if None)
        cache_dir: Directory for caching (default: data/task_streams)
        use_simulation: If True, use Exp1-style simulation tasks instead of real benchmarks
        force_regenerate: If True, regenerate even if cached
    
    Returns:
        List[TaskWrapper]: List of unified task wrappers
    """
    if stream_name not in TASK_STREAMS:
        raise ValueError(
            f"Unknown task stream: {stream_name}. "
            f"Available: {list(TASK_STREAMS.keys())}"
        )
    
    config = TASK_STREAMS[stream_name]
    actual_n_tasks = n_tasks or config["n_tasks"]
    
    # Set up cache directory
    if cache_dir is None:
        cache_dir = str(_ROOT / "data" / "task_streams")
    cache_path = Path(cache_dir) / f"{stream_name}_seed{seed}_n{actual_n_tasks}.jsonl"
    
    # Try to load from cache
    if cache_path.exists() and not force_regenerate:
        print(f"[task_loader] Loading cached task stream: {cache_path}")
        return _load_from_cache(cache_path)
    
    # Generate task stream
    if use_simulation or not HAS_DATA_GENERATOR:
        if not use_simulation:
            print("[task_loader] DatasetBuilder not available, falling back to simulation")
        tasks = _generate_simulation_tasks(config, actual_n_tasks, seed)
    else:
        tasks = _generate_real_tasks(config, actual_n_tasks, seed)
    
    # Save to cache
    _save_to_cache(tasks, cache_path)
    
    return tasks


def _generate_real_tasks(
    config: Dict[str, Any],
    n_tasks: int,
    seed: int,
) -> List[TaskWrapper]:
    """Generate tasks using DatasetBuilder with real benchmarks"""
    print(f"[task_loader] Generating real benchmark tasks using DatasetBuilder")
    
    # Initialize DatasetBuilder
    config_path = _ROOT / "symphony-data-generator" / "config" / "data_config.yaml"
    if not config_path.exists():
        # Use default config
        config_path = None
    
    builder = DatasetBuilder(str(config_path) if config_path else None)
    
    # Build task stream
    raw_tasks = builder.build_task_stream(
        benchmarks_to_include=config["benchmarks"],
        difficulty_split=config["difficulty_split"],
        n_total_tasks=n_tasks,
        random_seed=seed,
    )
    
    # Convert to TaskWrapper
    tasks = [TaskWrapper.from_data_generator_task(t) for t in raw_tasks]
    
    print(f"[task_loader] Generated {len(tasks)} tasks from real benchmarks")
    _print_task_stats(tasks)
    
    return tasks


def _generate_simulation_tasks(
    config: Dict[str, Any],
    n_tasks: int,
    seed: int,
) -> List[TaskWrapper]:
    """Generate simulation-style tasks (fallback when DatasetBuilder unavailable)"""
    print(f"[task_loader] Generating simulation tasks (Exp1-style)")
    
    # Parse difficulty split
    parts = config["difficulty_split"].split(":")
    easy_pct = int(parts[0])
    hard_pct = int(parts[1])
    p_hard = hard_pct / (easy_pct + hard_pct)
    
    rng = random.Random(seed)
    tasks: List[TaskWrapper] = []
    
    for i in range(n_tasks):
        is_hard = rng.random() < p_hard
        difficulty_bin = "hard" if is_hard else "easy"
        
        # Simulate benchmark assignment (alternating for balance)
        benchmark = config["benchmarks"][i % len(config["benchmarks"])]
        
        task = TaskWrapper(
            task_id=f"sim_{i:05d}",
            benchmark=benchmark,
            difficulty_score=1.0 if is_hard else 0.0,
            difficulty_bin=difficulty_bin,
            raw_data={"simulated": True, "index": i},
            requirement=difficulty_bin,
        )
        tasks.append(task)
    
    print(f"[task_loader] Generated {len(tasks)} simulation tasks")
    _print_task_stats(tasks)
    
    return tasks


def _load_from_cache(path: Path) -> List[TaskWrapper]:
    """Load tasks from JSONL cache"""
    tasks = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            tasks.append(TaskWrapper(**data))
    return tasks


def _save_to_cache(tasks: List[TaskWrapper], path: Path) -> None:
    """Save tasks to JSONL cache"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for task in tasks:
            f.write(json.dumps(task.to_dict(), ensure_ascii=False) + '\n')
    print(f"[task_loader] Saved {len(tasks)} tasks to {path}")


def _print_task_stats(tasks: List[TaskWrapper]) -> None:
    """Print task stream statistics"""
    easy_count = sum(1 for t in tasks if t.difficulty_bin == "easy")
    hard_count = sum(1 for t in tasks if t.difficulty_bin == "hard")
    
    benchmarks = {}
    for t in tasks:
        benchmarks[t.benchmark] = benchmarks.get(t.benchmark, 0) + 1
    
    print(f"  Total: {len(tasks)} tasks")
    print(f"  Difficulty: {easy_count} easy ({100*easy_count/len(tasks):.1f}%), "
          f"{hard_count} hard ({100*hard_count/len(tasks):.1f}%)")
    print(f"  Benchmarks: {benchmarks}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_available_streams() -> Dict[str, Dict[str, Any]]:
    """Return available task stream configurations"""
    return TASK_STREAMS.copy()


def task_to_sim_task(task: TaskWrapper):
    """
    Convert TaskWrapper to Exp1-compatible format for routing simulation.
    Returns a simple object with tid, difficulty, and requirement attributes.
    """
    class SimTaskCompat:
        def __init__(self, task: TaskWrapper):
            self.tid = task.task_id
            self.difficulty = task.difficulty_bin
            self.requirement = task.requirement
    
    return SimTaskCompat(task)


# ============================================================================
# CLI / TEST
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Task Loader CLI")
    parser.add_argument("--stream", type=str, default="mixed_80_20",
                        help="Task stream name")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of tasks (override)")
    parser.add_argument("--simulation", action="store_true",
                        help="Use simulation tasks")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Task Loader Test")
    print(f"{'='*60}")
    print(f"Stream: {args.stream}")
    print(f"Seed: {args.seed}")
    print(f"Simulation mode: {args.simulation}")
    print(f"DatasetBuilder available: {HAS_DATA_GENERATOR}")
    print(f"{'='*60}\n")
    
    tasks = load_task_stream(
        stream_name=args.stream,
        seed=args.seed,
        n_tasks=args.n,
        use_simulation=args.simulation,
        force_regenerate=args.force,
    )
    
    print(f"\nFirst 5 tasks:")
    for t in tasks[:5]:
        print(f"  {t.task_id}: {t.benchmark} / {t.difficulty_bin}")

