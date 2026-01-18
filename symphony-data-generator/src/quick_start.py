#!/usr/bin/env python3
"""
Quick Start Script for Symphony 2.0 Data Generation (Updated)

This script demonstrates the typical workflow:
1. Initialize data builder
2. Build task streams for all 5 experiments
3. OPTIONAL: Validate difficulty definitions for all benchmarks
4. Save results for team use

Key improvements:
- Validation is now COMPLETELY OPTIONAL (interactive prompt)
- Can validate individual benchmarks OR all benchmarks at once
- Smart budget management (stops when budget exceeded)
- Detailed validation report across all benchmarks

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
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"{title:^70}")
    print("="*70 + "\n")

def ask_yes_no(prompt, default='n'):
    """Interactive yes/no question"""
    default_str = "y/N" if default == 'n' else "Y/n"
    response = input(f"{prompt} ({default_str}): ").strip().lower()
    if response == '':
        return default == 'y'
    return response.startswith('y')

def main():
    print_section("SYMPHONY 2.0 - DATA GENERATION QUICK START")
    
    # Step 0: Create config if it doesn't exist
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    if not (config_dir / 'data_config.yaml').exists():
        print("⚠️  Config file not found.")
        print("   Please place data_config.yaml in config/ directory")
        return 1
    
    # Step 1: Initialize builder
    print("Step 1: Initializing DatasetBuilder...")
    builder = DatasetBuilder('config/data_config.yaml')
    print("✓ DatasetBuilder initialized\n")

    # Step 1.5: Preprocess all enabled benchmarks (one-time setup)
    print_section("Step 1.5: Preprocessing Full Benchmarks")
    print("This step downloads and preprocesses FULL benchmarks (all tasks).")
    print("It only needs to run once. Subsequent runs will reuse cached files.\n")

    try:
        print("  Preprocessing full benchmarks...")
        full_preprocessed = builder.preprocess_all_benchmarks(
            output_dir='data/benchmarks/full',
            force_reprocess=False,
        )
        print(f"  ✓ Preprocessed {len(full_preprocessed)} full benchmark(s)")
        
        # Show preprocessing summary
        for bn, tasks in full_preprocessed.items():
            print(f"    - {bn}: {len(tasks)} tasks")
        print()
    except Exception as e:
        print(f"✗ Error during preprocessing: {e}\n")
        return 1
    
    # Step 2: Build task streams for experiments
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
        'exp3': {
            'name': 'Exp 3: Fault Tolerance (Mixed progression)',
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
                'humaneval': 0.2,
                'gsm8k': 0.2,
                'bbh': 0.2,
                'amc': 0.2,
                'medical_qa': 0.2,
            },
            'difficulty_split': '50:50',
            'n_total_tasks': 2000,
            'random_seed': 2025,
        },
    }
    
    all_tasks = {}
    
    print_section("Step 2: Building task streams")
    
    # error log
    error_log_path = Path('data') / 'quick_start_errors.log'
    error_log_path.parent.mkdir(parents=True, exist_ok=True)

    for exp_id, config in experiments.items():
        print(f"Building {config['name']}...")
        
        try:
            tasks = builder.build_task_stream(
                benchmarks_to_include=config['benchmarks'],
                difficulty_split=config['difficulty_split'],
                n_total_tasks=config['n_total_tasks'],
                random_seed=config['random_seed'],
                benchmark_ratios=config.get('benchmark_ratios'),  # Per-experiment ratios
            )
            
            all_tasks[exp_id] = tasks
            
            # Save task pool
            output_dir = Path(f'data/{exp_id}')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            builder.save_task_pool(
                tasks,
                f'data/{exp_id}/task_pool.jsonl'
            )
            
            builder.save_statistics(
                tasks,
                f'data/{exp_id}/statistics.json'
            )
            
            print(f"✓ Saved to data/{exp_id}/\n")
            
        except Exception as e:
            print(f"✗ Error building {exp_id}: {e}\n")
            try:
                with open(error_log_path, 'a', encoding='utf-8') as ef:
                    ef.write(f"[{exp_id}] {config.get('name')}\n")
                    ef.write(f"Error: {repr(e)}\n\n")
            except Exception:
                pass
            continue
    
    # Step 3: Summary of generated task pools
    print_section("Step 3: Task pools summary")
    
    summary_stats = {}
    
    for exp_id, tasks in all_tasks.items():
        config = experiments[exp_id]
        stats = {
            'n_total': len(tasks),
            'benchmarks': list(set(t.benchmark for t in tasks)),
            'mean_difficulty': float(np.mean([t.difficulty_score for t in tasks])),
            'std_difficulty': float(np.std([t.difficulty_score for t in tasks])),
            'easy_count': sum(1 for t in tasks if t.difficulty_bin == 'easy'),
            'hard_count': sum(1 for t in tasks if t.difficulty_bin == 'hard'),
        }
        summary_stats[exp_id] = stats
        
        print(f"{exp_id.upper()}: {config['name']}")
        print(f"  Total tasks: {stats['n_total']}")
        print(f"  Easy: {stats['easy_count']}, Hard: {stats['hard_count']}")
        print(f"  Mean difficulty: {stats['mean_difficulty']:.3f} ± {stats['std_difficulty']:.3f}")
        print(f"  Benchmarks: {', '.join(stats['benchmarks'])}")
        print()
    
    # Step 4: OPTIONAL - Validate difficulty scoring
    print_section("Step 4 (Optional): Validate Difficulty Definitions")
    
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("OpenAI API key not detected (OPENAI_API_KEY env var not set).")
        print("Validation is optional. Skipping.\n")
        validation_done = False
    else:
        print("OpenAI API key detected.")
        print("\nValidation will:")
        print("  ✓ Check all enabled benchmarks")
        print("  ✓ Verify difficulty definitions (should correlate with error rate)")
        print("  ✓ Report accuracy gaps (easy vs hard tasks)")
        print("  ✓ Estimate cost (~$0.50-2.00 depending on sample size)")
        print("\nOptions:")
        print("  1. Skip validation (recommended if time-constrained)")
        print("  2. Validate all benchmarks (comprehensive, costs ~$1-2)")
        print("  3. Validate individual benchmark (fast, minimal cost)")
        
        choice = input("\nYour choice (skip/all/individual, default='skip'): ").strip().lower()
        validation_done = False
        
        if choice == 'all':
            print("\n" + "-"*70)
            print("VALIDATING ALL BENCHMARKS")
            print("-"*70 + "\n")
            
            # Combine all tasks for validation
            combined_tasks = []
            for tasks in all_tasks.values():
                combined_tasks.extend(tasks)
            
            builder.enable_validation(api_key=api_key, budget_limit=2.0)
            
            try:
                validation_results = builder.validate_all_benchmarks(
                    combined_tasks,
                    n_samples=20,
                    budget_limit=2.0
                )
                
                print("\n" + "="*70)
                print("VALIDATION RESULTS FOR ALL BENCHMARKS")
                print("="*70 + "\n")
                
                for benchmark, stats in validation_results.items():
                    print(f"{benchmark.upper()}")
                    
                    if 'error' in stats:
                        print(f"  ✗ Error: {stats['error']}\n")
                        continue
                    
                    correlation = stats.get('correlation', 0)
                    status = "✓" if correlation > 0.6 else "⚠️"
                    
                    print(f"  {status} Correlation: {correlation:.3f} (target > 0.6)")
                    print(f"     Easy accuracy: {stats['easy_accuracy']:.1%}")
                    print(f"     Hard accuracy: {stats['hard_accuracy']:.1%}")
                    print(f"     Accuracy gap: {stats['accuracy_gap']:.1%}")
                    print(f"     Cost: ${stats['total_cost']:.2f}\n")
                
                # Save validation results
                with open('data/validation_results.json', 'w') as f:
                    # Convert to serializable format
                    results_to_save = {}
                    for bn, stats in validation_results.items():
                        results_to_save[bn] = {
                            k: float(v) if isinstance(v, np.number) else v
                            for k, v in stats.items()
                            if k != 'results_df'
                        }
                    json.dump(results_to_save, f, indent=2)
                
                print("✓ Validation results saved to data/validation_results.json")
                validation_done = True
                
            except Exception as e:
                print(f"✗ Validation failed: {e}\n")
        
        elif choice == 'individual':
            print("\n" + "-"*70)
            print("VALIDATING INDIVIDUAL BENCHMARK")
            print("-"*70 + "\n")
            
            # Get all benchmarks in the task pools
            all_benchmarks = set()
            for tasks in all_tasks.values():
                all_benchmarks.update(t.benchmark for t in tasks)
            
            print(f"Available benchmarks: {', '.join(sorted(all_benchmarks))}")
            benchmark_to_validate = input("\nWhich benchmark to validate? ").strip().lower()
            
            if benchmark_to_validate not in all_benchmarks:
                print(f"✗ Benchmark '{benchmark_to_validate}' not found")
            else:
                # Find tasks for this benchmark
                tasks_for_bn = []
                for tasks in all_tasks.values():
                    tasks_for_bn.extend([t for t in tasks if t.benchmark == benchmark_to_validate])
                
                if len(tasks_for_bn) == 0:
                    print(f"✗ No tasks found for {benchmark_to_validate}")
                else:
                    n_samples = 20
                    try:
                        n = int(input(f"\nHow many samples to validate? (default={n_samples}): ").strip())
                        if n > 0:
                            n_samples = n
                    except ValueError:
                        pass
                    
                    print(f"\nValidating {benchmark_to_validate} with {n_samples} samples...")
                    
                    builder.enable_validation(api_key=api_key, budget_limit=1.0)
                    
                    try:
                        stats = builder.validate_difficulties(
                            tasks_for_bn,
                            benchmark=benchmark_to_validate,
                            n_samples=n_samples
                        )
                        
                        print("\n" + "="*70)
                        print(f"VALIDATION RESULTS - {benchmark_to_validate.upper()}")
                        print("="*70 + "\n")
                        
                        if 'error' in stats:
                            print(f"✗ Error: {stats['error']}")
                        else:
                            correlation = stats.get('correlation', 0)
                            status = "✓" if correlation > 0.6 else "⚠️"
                            
                            print(f"{status} Correlation: {correlation:.3f} (target > 0.6)")
                            print(f"   Easy accuracy: {stats['easy_accuracy']:.1%}")
                            print(f"   Hard accuracy: {stats['hard_accuracy']:.1%}")
                            print(f"   Accuracy gap: {stats['accuracy_gap']:.1%}")
                            print(f"   Cost: ${stats['total_cost']:.2f}\n")
                            
                            validation_done = True
                    
                    except Exception as e:
                        print(f"✗ Validation failed: {e}")
    
    # Step 5: Next steps
    print_section("Next Steps")
    
    if validation_done:
        print("✓ Data generation + validation complete!")
    else:
        print("✓ Data generation complete (validation skipped)")
    
    print("\n1. Task pools generated in data/exp{1,2,3,5}/ directories")
    print("   - task_pool.jsonl: Task data for experiments")
    print("   - statistics.json: Difficulty distribution statistics")
    
    if validation_done:
        print("   - validation_results.json: Validation report\n")
    else:
        print()
    
    print("2. Share task_pool.jsonl files with your team:")
    print("   - @Backend: Use for Exp 1, 3, 4 simulations")
    print("   - @ML: Use for Exp 2, 5 evaluations")
    
    print("\n3. Next: Implement your evaluation scripts")
    print("   - Parse JSONL: json.loads(line) for each line")
    print("   - Route by 'benchmark' field to appropriate handler")
    print("   - Log results alongside 'task_id' for analysis")
    
    if not validation_done:
        print("\n4. (Optional) Later: Run validation to verify difficulty definitions")
        print("   - Use: python -c \"from data_generator import DatasetBuilder; ...\"")
        print("   - Or: Re-run this script and choose 'all' or 'individual'")
    
    print("\n5. For detailed documentation: See DATA_GENERATOR_GUIDE.md")
    
    print_section("Done!")
    print("✓ Ready to start experiments!\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
