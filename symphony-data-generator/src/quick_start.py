#!/usr/bin/env python3
"""
Quick Start Script for Symphony 2.0 Data Generation

Run this with:
    python quick_start.py
"""

import os
import sys
from pathlib import Path
from data_generator import DatasetBuilder
import numpy as np
import json


def print_section(title):
    print("\n" + "="*70)
    print(f"{title:^70}")
    print("="*70 + "\n")


def main():
    print_section("SYMPHONY 2.0 - DATA GENERATION QUICK START")
    
    # Check config
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    if not (config_dir / 'data_config.yaml').exists():
        print("⚠️  Config file not found.")
        print("   Please place data_config.yaml in config/ directory")
        return 1
    
    # Initialize builder
    print("Step 1: Initializing DatasetBuilder...")
    builder = DatasetBuilder('config/data_config.yaml')
    print("✓ DatasetBuilder initialized\n")

    # Preprocess
    print_section("Step 2: Preprocessing Full Benchmarks")
    try:
        full_preprocessed = builder.preprocess_all_benchmarks(
            output_dir='data/benchmarks/full',
            force_reprocess=False,
        )
        print(f"  ✓ Preprocessed {len(full_preprocessed)} benchmark(s)")
        for bn, tasks in full_preprocessed.items():
            print(f"    - {bn}: {len(tasks)} tasks")
    except Exception as e:
        print(f"✗ Error during preprocessing: {e}\n")
        return 1
    
    # Build task streams
    experiments = {
        'exp1': {
            'name': 'Exp 1: Routing Efficiency (80% easy, 20% hard)',
            'benchmarks': ['humaneval', 'gsm8k'],
            'benchmark_ratios': {'humaneval': 0.5, 'gsm8k': 0.5},
            'difficulty_split': '80:20',
            'n_total_tasks': 1000,
            'random_seed': 2025,
        },
        'exp2': {
            'name': 'Exp 2: Learning Curve (50% easy, 50% hard)',
            'benchmarks': ['humaneval', 'gsm8k'],
            'benchmark_ratios': {'humaneval': 0.5, 'gsm8k': 0.5},
            'difficulty_split': '50:50',
            'n_total_tasks': 500,
            'random_seed': 2025,
        },
        'exp5': {
            'name': 'Exp 5: Real Benchmarks (All benchmarks, 50/50)',
            'benchmarks': ['humaneval', 'gsm8k', 'bbh', 'amc', 'medical_qa'],
            'benchmark_ratios': {
                'humaneval': 0.2, 'gsm8k': 0.2, 'bbh': 0.2,
                'amc': 0.2, 'medical_qa': 0.2,
            },
            'difficulty_split': '50:50',
            'n_total_tasks': 2000,
            'random_seed': 2025,
        },
    }
    
    all_tasks = {}
    
    print_section("Step 3: Building task streams")
    
    for exp_id, config in experiments.items():
        print(f"Building {config['name']}...")
        
        try:
            tasks = builder.build_task_stream(
                benchmarks_to_include=config['benchmarks'],
                difficulty_split=config['difficulty_split'],
                n_total_tasks=config['n_total_tasks'],
                random_seed=config['random_seed'],
                benchmark_ratios=config.get('benchmark_ratios'),
            )
            
            all_tasks[exp_id] = tasks
            
            output_dir = Path(f'data/{exp_id}')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            builder.save_task_pool(tasks, f'data/{exp_id}/task_pool.jsonl')
            builder.save_statistics(tasks, f'data/{exp_id}/statistics.json')
            
            print(f"✓ Saved to data/{exp_id}/\n")
            
        except Exception as e:
            print(f"✗ Error building {exp_id}: {e}\n")
            continue
    
    # Summary
    print_section("Step 4: Summary")
    
    for exp_id, tasks in all_tasks.items():
        config = experiments[exp_id]
        easy_count = sum(1 for t in tasks if t.difficulty_bin == 'easy')
        hard_count = sum(1 for t in tasks if t.difficulty_bin == 'hard')
        
        print(f"{exp_id.upper()}: {config['name']}")
        print(f"  Total tasks: {len(tasks)}")
        print(f"  Easy: {easy_count}, Hard: {hard_count}")
        print(f"  Benchmarks: {', '.join(set(t.benchmark for t in tasks))}")
        print()
    
    print_section("Done!")
    print("✓ Ready to start experiments!\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
