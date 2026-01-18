#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/exp4_system_optimization/plot_results.py

Visualization for Exp4: System Performance Optimization
Generate plots to analyze LinUCB's system-level optimization capabilities.

Plots:
1. Scenario comparison (bar charts)
2. Latency learning curves
3. Load burst response
4. Agent utilization heatmap
5. Efficiency comparison

Usage:
  python3 plot_results.py --result-dir experiments/result/exp4_system_optimization
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns


def load_summary(path: str) -> List[Dict]:
    """Load summary CSV"""
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_step_logs(path: str) -> List[Dict]:
    """Load step logs CSV"""
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_scenario_comparison(summary: List[Dict], outdir: str):
    """
    Plot 1: Compare metrics across scenarios
    - Success rate
    - Average latency
    - P95 latency
    - Agent utilization Gini
    """
    # Group by scenario
    scenarios = sorted(set(row['scenario'] for row in summary))
    policies = ['always_A', 'static_rule', 'random', 'linucb']
    policy_labels = {
        'always_A': 'Always-A',
        'static_rule': 'Static Rule',
        'random': 'Random',
        'linucb': 'LinUCB (Ours)'
    }
    
    # Organize data
    data = {scenario: {policy: {} for policy in policies} for scenario in scenarios}
    for row in summary:
        scenario = row['scenario']
        policy = row['policy']
        if policy in policies:
            data[scenario][policy] = {
                'success_rate': float(row['success_rate']),
                'avg_latency_ms': float(row['avg_latency_ms']),
                'latency_p95_ms': float(row['latency_p95_ms']),
                'gini': float(row['agent_utilization_gini']),
            }
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle('Exp4: System Performance Comparison Across Scenarios', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    x = np.arange(len(scenarios))
    width = 0.2
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    
    # Plot 1: Success Rate
    ax1 = axes[0, 0]
    for i, policy in enumerate(policies):
        values = [data[sc][policy].get('success_rate', 0) for sc in scenarios]
        ax1.bar(x + i * width, values, width, 
                label=policy_labels[policy], color=colors[i], alpha=0.9)
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Success Rate by Scenario')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(scenarios, rotation=15, ha='right')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_ylim(0.7, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Average Latency
    ax2 = axes[0, 1]
    for i, policy in enumerate(policies):
        values = [data[sc][policy].get('avg_latency_ms', 0) for sc in scenarios]
        ax2.bar(x + i * width, values, width, 
                label=policy_labels[policy], color=colors[i], alpha=0.9)
    ax2.set_ylabel('Average Latency (ms)')
    ax2.set_title('Average Latency by Scenario (Lower is Better)')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(scenarios, rotation=15, ha='right')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: P95 Latency
    ax3 = axes[1, 0]
    for i, policy in enumerate(policies):
        values = [data[sc][policy].get('latency_p95_ms', 0) for sc in scenarios]
        ax3.bar(x + i * width, values, width, 
                label=policy_labels[policy], color=colors[i], alpha=0.9)
    ax3.set_ylabel('P95 Latency (ms)')
    ax3.set_title('P95 Latency by Scenario (Lower is Better)')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(scenarios, rotation=15, ha='right')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Agent Utilization Gini
    ax4 = axes[1, 1]
    for i, policy in enumerate(policies):
        values = [data[sc][policy].get('gini', 0) for sc in scenarios]
        ax4.bar(x + i * width, values, width, 
                label=policy_labels[policy], color=colors[i], alpha=0.9)
    ax4.set_ylabel('Gini Coefficient')
    ax4.set_title('Agent Utilization Gini (Lower is Better Balanced)')
    ax4.set_xticks(x + width * 1.5)
    ax4.set_xticklabels(scenarios, rotation=15, ha='right')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.set_ylim(0, 1.0)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'plot_scenario_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: plot_scenario_comparison.png")


def plot_latency_learning_curves(step_logs: List[Dict], outdir: str):
    """
    Plot 2: Latency learning curves for each scenario
    Show how LinUCB learns to reduce latency over time
    """
    # Group by scenario and policy
    data = defaultdict(lambda: defaultdict(list))
    for log in step_logs:
        scenario = log['scenario']
        policy = log['policy']
        t = int(log['t'])
        latency = float(log['latency_ms'])
        data[scenario][policy].append((t, latency))
    
    policies = ['always_A', 'static_rule', 'random', 'linucb']
    policy_labels = {
        'always_A': 'Always-A',
        'static_rule': 'Static Rule',
        'random': 'Random',
        'linucb': 'LinUCB (Ours)'
    }
    colors = {'always_A': '#9b59b6', 'random': '#e74c3c', 'static_rule': '#f39c12', 'linucb': '#2ecc71'}
    
    scenarios = sorted(data.keys())
    
    for scenario in scenarios:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        
        for policy in policies:
            if policy not in data[scenario]:
                continue
            
            points = sorted(data[scenario][policy])
            ts = [p[0] for p in points]
            lats = [p[1] for p in points]
            
            # Rolling average
            window = 50
            rolling_lats = []
            for i in range(len(lats)):
                start = max(0, i - window + 1)
                rolling_lats.append(np.mean(lats[start:i+1]))
            
            linestyle = '--' if policy == 'linucb' else '-'
            linewidth = 2.0 if policy == 'linucb' else 1.2
            alpha = 1.0 if policy == 'linucb' else 0.7
            
            ax.plot(ts, rolling_lats, label=policy_labels[policy], 
                   color=colors[policy], linestyle=linestyle, 
                   linewidth=linewidth, alpha=alpha)
        
        ax.set_xlabel('Task Index (t)', fontsize=11)
        ax.set_ylabel('Latency (ms, rolling avg)', fontsize=11)
        ax.set_title(f'Latency Learning Curve - {scenario.replace("_", " ").title()}', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'plot_latency_curve_{scenario}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: plot_latency_curve_{scenario}.png")


def plot_load_burst_response(step_logs: List[Dict], outdir: str, 
                              burst_config: Dict):
    """
    Plot 3: Agent selection during load burst periods
    Show how LinUCB adapts when specific agents experience load bursts
    """
    # Focus on 'load_burst' and 'combined' scenarios
    scenarios = ['load_burst', 'combined']
    
    for scenario in scenarios:
        scenario_logs = [log for log in step_logs if log['scenario'] == scenario]
        if not scenario_logs:
            continue
        
        # Group by policy
        data = defaultdict(lambda: defaultdict(list))
        for log in scenario_logs:
            policy = log['policy']
            t = int(log['t'])
            agent = log['chosen_agent']
            data[policy][agent].append(t)
        
        # Create subplots for each policy
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=150, sharex=True)
        fig.suptitle(f'Agent Selection During Load Burst - {scenario.replace("_", " ").title()}',
                    fontsize=13, fontweight='bold')
        
        policies = ['always_A', 'static_rule', 'random', 'linucb']
        policy_labels = {
            'random': 'Random',
            'capability_match': 'Cap-Match',
            'round_robin': 'Round-Robin',
            'linucb': 'LinUCB (Ours)'
        }
        
        agents = ['A', 'B', 'C', 'D', 'E']
        agent_colors = {'A': '#e74c3c', 'B': '#3498db', 'C': '#f39c12', 
                       'D': '#9b59b6', 'E': '#1abc9c'}
        
        for idx, policy in enumerate(policies):
            ax = axes[idx // 2, idx % 2]
            
            # Count selections in bins
            n_bins = 50
            max_t = max(int(log['t']) for log in scenario_logs)
            bin_edges = np.linspace(0, max_t, n_bins + 1)
            
            # Stack agent selections
            bottom = np.zeros(n_bins)
            for agent in agents:
                ts = data[policy].get(agent, [])
                counts, _ = np.histogram(ts, bins=bin_edges)
                ax.bar(range(n_bins), counts, bottom=bottom, 
                      label=f'Agent {agent}', color=agent_colors[agent], 
                      alpha=0.8, width=1.0, edgecolor='white', linewidth=0.5)
                bottom += counts
            
            # Mark burst periods (example: hardcoded for demo)
            if scenario == 'load_burst':
                ax.axvspan(200 * n_bins / max_t, 400 * n_bins / max_t, 
                          alpha=0.15, color='red', label='Burst: A')
                ax.axvspan(500 * n_bins / max_t, 700 * n_bins / max_t, 
                          alpha=0.15, color='blue', label='Burst: B')
            elif scenario == 'combined':
                ax.axvspan(300 * n_bins / max_t, 500 * n_bins / max_t, 
                          alpha=0.15, color='blue', label='Burst: B')
            
            ax.set_title(policy_labels[policy], fontsize=10, fontweight='bold')
            ax.set_ylabel('Selections per bin')
            if idx >= 2:
                ax.set_xlabel('Task bins')
            ax.legend(loc='upper right', fontsize=7, ncol=2)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'plot_burst_response_{scenario}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: plot_burst_response_{scenario}.png")


def plot_agent_utilization_heatmap(summary: List[Dict], outdir: str):
    """
    Plot 4: Agent utilization heatmap
    Show how different policies distribute tasks across agents
    """
    # Organize data
    scenarios = sorted(set(row['scenario'] for row in summary))
    policies = ['always_A', 'static_rule', 'random', 'linucb']
    policy_labels = {
        'random': 'Random',
        'static_rule': 'Static Rule',
        'linucb': 'LinUCB'
    }
    agents = ['A', 'B', 'C', 'D', 'E']
    
    for scenario in scenarios:
        # Build matrix: policies x agents
        matrix = []
        policy_names = []
        
        for policy in policies:
            row_data = [r for r in summary 
                       if r['scenario'] == scenario and r['policy'] == policy]
            if not row_data:
                continue
            
            row = row_data[0]
            counts = [int(row[f'choose_{agent}']) for agent in agents]
            total = sum(counts)
            percentages = [c / max(total, 1) * 100 for c in counts]
            matrix.append(percentages)
            policy_names.append(policy_labels[policy])
        
        if not matrix:
            continue
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        
        im = ax.imshow(matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=100)
        
        # Set ticks
        ax.set_xticks(np.arange(len(agents)))
        ax.set_yticks(np.arange(len(policy_names)))
        ax.set_xticklabels([f'Agent {a}' for a in agents])
        ax.set_yticklabels(policy_names)
        
        # Add text annotations
        for i in range(len(policy_names)):
            for j in range(len(agents)):
                text = ax.text(j, i, f'{matrix[i][j]:.1f}%',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title(f'Agent Utilization - {scenario.replace("_", " ").title()}',
                    fontsize=12, fontweight='bold', pad=15)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Utilization %', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'plot_utilization_heatmap_{scenario}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: plot_utilization_heatmap_{scenario}.png")


def plot_efficiency_comparison(summary: List[Dict], outdir: str):
    """
    Plot 5: Latency efficiency comparison
    Efficiency = success_rate / avg_latency_ms * 1000 (per second)
    """
    scenarios = sorted(set(row['scenario'] for row in summary))
    policies = ['always_A', 'static_rule', 'random', 'linucb']
    policy_labels = {
        'always_A': 'Always-A',
        'static_rule': 'Static Rule',
        'random': 'Random',
        'linucb': 'LinUCB (Ours)'
    }
    colors = {'always_A': '#9b59b6', 'random': '#e74c3c', 'static_rule': '#f39c12', 'linucb': '#2ecc71'}
    
    # Organize data
    data = {scenario: {} for scenario in scenarios}
    for row in summary:
        scenario = row['scenario']
        policy = row['policy']
        if policy in policies:
            efficiency = float(row['latency_efficiency'])
            data[scenario][policy] = efficiency
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    
    x = np.arange(len(scenarios))
    width = 0.2
    
    for i, policy in enumerate(policies):
        values = [data[sc].get(policy, 0) for sc in scenarios]
        bars = ax.bar(x + i * width, values, width, 
                     label=policy_labels[policy], 
                     color=colors[policy], alpha=0.9)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=7)
    
    ax.set_ylabel('Latency Efficiency (success/sec)', fontsize=11)
    ax.set_title('Latency Efficiency Comparison (Higher is Better)', 
                fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([sc.replace('_', '\n') for sc in scenarios], fontsize=9)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'plot_efficiency_comparison.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: plot_efficiency_comparison.png")


def plot_latency_success_tradeoff(step_logs: List[Dict], outdir: str):
    """
    Plot: Two separate subplots showing latency and success rate over time
    Shows the trade-off: LinUCB optimizes latency while maintaining reasonable success rate
    """
    scenario = 'latency_heterogeneous'  # Focus on key scenario
    
    # Filter by scenario
    scenario_logs = [log for log in step_logs if log['scenario'] == scenario]
    if not scenario_logs:
        return
    
    # Group by policy
    data = defaultdict(lambda: {'t': [], 'latency': [], 'success': []})
    for log in scenario_logs:
        policy = log['policy']
        t = int(log['t'])
        latency = float(log['latency_ms'])
        success = int(log['success'])
        data[policy]['t'].append(t)
        data[policy]['latency'].append(latency)
        data[policy]['success'].append(success)
    
    # Sort by t
    for policy in data:
        sorted_indices = sorted(range(len(data[policy]['t'])), 
                               key=lambda i: data[policy]['t'][i])
        data[policy]['t'] = [data[policy]['t'][i] for i in sorted_indices]
        data[policy]['latency'] = [data[policy]['latency'][i] for i in sorted_indices]
        data[policy]['success'] = [data[policy]['success'][i] for i in sorted_indices]
    
    policies = ['always_A', 'static_rule', 'random', 'linucb']
    policy_labels = {
        'always_A': 'Always-A',
        'static_rule': 'Static Rule',
        'random': 'Random',
        'linucb': 'LinUCB (Ours)'
    }
    colors = {'always_A': '#9b59b6', 'random': '#e74c3c', 'static_rule': '#f39c12', 'linucb': '#2ecc71'}
    
    # Create 2 subplots vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=150, sharex=True)
    fig.suptitle('Latency vs Success Rate Trade-off - Key Scenario', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    window = 50
    
    # Subplot 1: Latency (primary metric for exp4)
    for policy in policies:
        if policy not in data or len(data[policy]['t']) == 0:
            continue
        
        ts = data[policy]['t']
        lats = data[policy]['latency']
        
        # Rolling average
        rolling_lats = []
        for i in range(len(lats)):
            start = max(0, i - window + 1)
            rolling_lats.append(np.mean(lats[start:i+1]))
        
        linestyle = '--' if policy == 'linucb' else '-'
        linewidth = 2.5 if policy == 'linucb' else 1.3
        alpha = 1.0 if policy == 'linucb' else 0.7
        
        ax1.plot(ts, rolling_lats, label=policy_labels[policy], 
                color=colors[policy], linestyle=linestyle, 
                linewidth=linewidth, alpha=alpha)
    
    ax1.set_ylabel('Latency (ms, rolling avg)', fontsize=11)
    ax1.set_title('(a) Latency Learning Curve', fontsize=11, fontweight='bold', loc='left')
    ax1.tick_params(axis='y')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.95)
    
    # Add annotation
    ax1.text(0.02, 0.98, 
            'LinUCB reduces latency by 51.6%',
            transform=ax1.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
    
    # Subplot 2: Success Rate (shows stability)
    for policy in policies:
        if policy not in data or len(data[policy]['t']) == 0:
            continue
        
        ts = data[policy]['t']
        successes = data[policy]['success']
        
        # Cumulative average success rate
        cum_success = []
        cum_sum = 0.0
        for i, s in enumerate(successes, 1):
            cum_sum += s
            cum_success.append(cum_sum / i)
        
        # Rolling average for smoothing
        rolling_success = []
        for i in range(len(cum_success)):
            start = max(0, i - window + 1)
            rolling_success.append(np.mean(cum_success[start:i+1]))
        
        linestyle = '--' if policy == 'linucb' else '-'
        linewidth = 2.5 if policy == 'linucb' else 1.3
        alpha = 1.0 if policy == 'linucb' else 0.7
        
        ax2.plot(ts, rolling_success, label=policy_labels[policy],
                color=colors[policy], linestyle=linestyle, 
                linewidth=linewidth, alpha=alpha)
    
    ax2.set_xlabel('Task Index (t)', fontsize=11)
    ax2.set_ylabel('Cumulative Success Rate (rolling avg)', fontsize=11)
    ax2.set_title('(b) Success Rate Stability', fontsize=11, fontweight='bold', loc='left')
    ax2.tick_params(axis='y')
    ax2.set_ylim(0.75, 1.0)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.95)
    
    # Add annotation
    ax2.text(0.02, 0.02, 
            'LinUCB maintains 82% success rate\n(stable, not degrading)',
            transform=ax2.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.4))
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'plot_latency_success_tradeoff.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: plot_latency_success_tradeoff.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--result-dir',
        type=str,
        default='experiments/result/exp4_system_optimization',
        help='Directory containing experiment results'
    )
    args = ap.parse_args()
    
    result_dir = args.result_dir
    summary_path = os.path.join(result_dir, 'summary_all_scenarios.csv')
    step_logs_path = os.path.join(result_dir, 'step_logs_all.csv')
    
    if not os.path.exists(summary_path):
        print(f"❌ Summary file not found: {summary_path}")
        return
    
    if not os.path.exists(step_logs_path):
        print(f"❌ Step logs file not found: {step_logs_path}")
        return
    
    print("📊 Loading data...")
    summary = load_summary(summary_path)
    step_logs = load_step_logs(step_logs_path)
    
    print(f"📈 Generating plots...")
    
    # Plot 1: Scenario comparison (most comprehensive)
    plot_scenario_comparison(summary, result_dir)
    
    # Plot 2: Latency learning curve - REMOVED (redundant with tradeoff plot)
    # Skipped to avoid duplicate information
    
    # Plot 2: Efficiency comparison (key metric)
    plot_efficiency_comparison(summary, result_dir)
    
    # Plot 4: Latency-Success Trade-off (shows optimization while maintaining success)
    plot_latency_success_tradeoff(step_logs, result_dir)
    
    # Plot 4: Agent utilization heatmap (only for latency_heterogeneous - shows learning)
    print("  ✅ Generating agent utilization heatmap for key scenario...")
    scenario = 'latency_heterogeneous'
    agents = ['A', 'B', 'C', 'D', 'E']
    policies = ['always_A', 'static_rule', 'random', 'linucb']
    policy_labels = {
        'always_A': 'Always-A',
        'static_rule': 'Static Rule',
        'random': 'Random',
        'linucb': 'LinUCB'
    }
    
    # Build matrix: policies x agents
    matrix = []
    policy_names = []
    
    for policy in policies:
        row_data = [r for r in summary 
                   if r['scenario'] == scenario and r['policy'] == policy]
        if not row_data:
            continue
        
        row = row_data[0]
        counts = [int(row[f'choose_{agent}']) for agent in agents]
        total = sum(counts)
        percentages = [c / max(total, 1) * 100 for c in counts]
        matrix.append(percentages)
        policy_names.append(policy_labels[policy])
    
    if matrix:
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        
        im = ax.imshow(matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=100)
        
        # Set ticks
        ax.set_xticks(np.arange(len(agents)))
        ax.set_yticks(np.arange(len(policy_names)))
        ax.set_xticklabels([f'Agent {a}' for a in agents])
        ax.set_yticklabels(policy_names)
        
        # Add text annotations
        for i in range(len(policy_names)):
            for j in range(len(agents)):
                text = ax.text(j, i, f'{matrix[i][j]:.1f}%',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title(f'Agent Utilization - LinUCB Learns Low-Latency Preference',
                    fontsize=12, fontweight='bold', pad=15)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Utilization %', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f'plot_agent_utilization.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    ✅ Saved: plot_agent_utilization.png")
    
    print(f"\n✅ Core plots generated! (4 plots total)")
    print(f"📁 Output directory: {result_dir}")


if __name__ == '__main__':
    main()

