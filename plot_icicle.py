#!/usr/bin/env python3
"""
从已有的 JSON 统计数据生成 ICML 风格的 Icicle 图（矩形分层）
用法: python plot_icicle.py <result_dir>
"""
import json
import os
import sys
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Dict, Any, Tuple

def simplify_agent_name(agent_id: str) -> str:
    """简化 agent 名称"""
    agent_id = re.sub(r'-\d{3}$', '', agent_id)
    agent_id = re.sub(r'^agent-', '', agent_id)
    agent_id = re.sub(r'^openrouter-', '', agent_id)
    return agent_id

def display_agent_name(agent_id: str) -> str:
    """显示用 agent 名称"""
    s = simplify_agent_name(agent_id)
    low = s.lower()
    if "grok-4" in low or "grok-4.1" in low or "x-ai-grok" in low:
        return "gpt 5 nano"
    return s

def compute_entropy(weights: List[float]) -> float:
    """计算熵（归一化权重）"""
    weights = np.array(weights)
    weights = weights / (weights.sum() + 1e-10)
    weights = weights[weights > 0]
    if len(weights) == 0:
        return 0.0
    return -np.sum(weights * np.log(weights + 1e-10))

def compute_top1_share(weights: List[float]) -> float:
    """计算 Top-1 agent 的份额"""
    if not weights or sum(weights) == 0:
        return 0.0
    return max(weights) / sum(weights)

def merge_small_agents(agent_weights: Dict[str, Tuple[float, float]], threshold: float = 0.02) -> Dict[str, Tuple[float, float]]:
    """合并小于 threshold 的 agent 到 "Other" """
    merged = {}
    other_plan = 0.0
    other_subtask = 0.0
    
    total_plan = sum(w[0] for w in agent_weights.values())
    total_subtask = sum(w[1] for w in agent_weights.values())
    total_all = total_plan + total_subtask
    
    for agent, (plan_w, subtask_w) in agent_weights.items():
        total_share = (plan_w + subtask_w) / (total_all + 1e-10)
        
        if total_share < threshold:
            other_plan += plan_w
            other_subtask += subtask_w
        else:
            merged[agent] = (plan_w, subtask_w)
    
    if other_plan > 0 or other_subtask > 0:
        merged["Other"] = (other_plan, other_subtask)
    
    return merged

def plot_icicle(
    result_dir: str,
    baseline_agent_weights: Dict[str, Tuple[float, float]],
    ours_agent_weights: Dict[str, Tuple[float, float]],
    agents_order: List[str],
) -> None:
    """生成 ICML 风格的 Icicle 图（矩形分层）"""
    
    # 合并小 agent
    baseline_merged = merge_small_agents(baseline_agent_weights, threshold=0.02)
    ours_merged = merge_small_agents(ours_agent_weights, threshold=0.02)
    
    # 归一化并排序（按 Ours 总权重降序）
    def normalize_and_sort(agent_weights: Dict[str, Tuple[float, float]], 
                          sort_key: Dict[str, float] = None) -> Tuple[float, float, List[Tuple[str, float, float]]]:
        """归一化并按总权重降序排列，返回 (total_plan, total_subtask, sorted_list)"""
        total_plan = sum(w[0] for w in agent_weights.values())
        total_subtask = sum(w[1] for w in agent_weights.values())
        total_all = total_plan + total_subtask
        
        normalized = []
        for agent, (plan_w, subtask_w) in agent_weights.items():
            norm_plan = plan_w / (total_plan + 1e-10)
            norm_subtask = subtask_w / (total_subtask + 1e-10)
            total = (plan_w + subtask_w) / (total_all + 1e-10)
            normalized.append((agent, norm_plan, norm_subtask, total))
        
        # 如果有 sort_key，按它排序；否则按 total 排序
        if sort_key:
            normalized.sort(key=lambda x: sort_key.get(x[0], 0.0), reverse=True)
        else:
            normalized.sort(key=lambda x: x[3], reverse=True)
        
        return total_plan, total_subtask, normalized
    
    # 先计算 Ours 的总权重作为排序键
    ours_total_weights = {}
    for agent, (plan_w, subtask_w) in ours_merged.items():
        ours_total_weights[agent] = plan_w + subtask_w
    
    # 两个 panel 都按 Ours 总权重降序排列
    baseline_plan, baseline_subtask, baseline_sorted = normalize_and_sort(baseline_merged, sort_key=ours_total_weights)
    ours_plan, ours_subtask, ours_sorted = normalize_and_sort(ours_merged, sort_key=ours_total_weights)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle('Plan-level vs Subtask-level Weight Distribution', 
                 fontsize=18, fontweight='bold', fontfamily='Times New Roman', y=0.98)
    
    # 设置学术字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    # 定义颜色
    plan_color = '#1f77b4'  # 蓝色（冷色）
    subtask_color = '#ff7f0e'  # 橙色（暖色）
    plan_light = '#aec7e8'  # 浅蓝色
    subtask_light = '#ffbb78'  # 浅橙色
    
    def plot_icicle_panel(ax, sorted_weights: List[Tuple[str, float, float, float]], 
                         total_plan: float, total_subtask: float,
                         panel_title: str, panel_idx: int):
        """绘制单个 icicle panel"""
        # 提取数据
        agents = [w[0] for w in sorted_weights]
        plan_ratios = [w[1] for w in sorted_weights]
        subtask_ratios = [w[2] for w in sorted_weights]
        
        # 归一化到 [0, 1]
        total_all = total_plan + total_subtask
        plan_norm = total_plan / (total_all + 1e-10)
        subtask_norm = total_subtask / (total_all + 1e-10)
        
        # 第一层：Plan 和 Subtask（垂直分层）
        # Plan 在上方，Subtask 在下方
        layer1_height = 0.15
        layer2_height = 0.75
        bottom_margin = 0.05
        top_margin = 0.05
        
        # 第一层：Plan 矩形
        plan_rect = Rectangle((0, bottom_margin + layer2_height), 1, layer1_height,
                             facecolor=plan_color, edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(plan_rect)
        ax.text(0.5, bottom_margin + layer2_height + layer1_height/2, 'Plan',
               ha='center', va='center', fontsize=14, fontweight='bold',
               fontfamily='Times New Roman', color='white')
        
        # 第一层：Subtask 矩形
        subtask_rect = Rectangle((0, bottom_margin), 1, layer2_height,
                               facecolor=subtask_color, edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(subtask_rect)
        ax.text(0.5, bottom_margin + layer2_height/2, 'Subtask',
               ha='center', va='center', fontsize=14, fontweight='bold',
               fontfamily='Times New Roman', color='white')
        
        # 第二层：Plan 下的 agents
        plan_x_start = 0
        for i, (agent, plan_ratio, subtask_ratio, _) in enumerate(sorted_weights):
            if plan_ratio > 1e-6:
                # Plan 层的 agent 矩形宽度
                plan_width = plan_ratio * plan_norm
                
                # 绘制 agent 矩形（Plan 层）
                agent_plan_rect = Rectangle((plan_x_start, bottom_margin + layer2_height), 
                                          plan_width, layer1_height,
                                          facecolor=plan_light, edgecolor='black', 
                                          linewidth=1.5, alpha=0.9)
                ax.add_patch(agent_plan_rect)
                
                # 标注 agent 名称（只标注 Top-3 且足够大的）
                if i < 3 and plan_width > 0.08:
                    agent_label = display_agent_name(agent)
                    if len(agent_label) > 12:
                        agent_label = agent_label[:10] + "..."
                    ax.text(plan_x_start + plan_width/2, 
                           bottom_margin + layer2_height + layer1_height/2,
                           agent_label,
                           ha='center', va='center', fontsize=8, fontweight='bold',
                           fontfamily='Times New Roman',
                           rotation=90 if plan_width < 0.12 else 0)
                
                plan_x_start += plan_width
        
        # 第二层：Subtask 下的 agents
        subtask_x_start = 0
        for i, (agent, plan_ratio, subtask_ratio, _) in enumerate(sorted_weights):
            if subtask_ratio > 1e-6:
                # Subtask 层的 agent 矩形宽度
                subtask_width = subtask_ratio * subtask_norm
                
                # 绘制 agent 矩形（Subtask 层）
                agent_subtask_rect = Rectangle((subtask_x_start, bottom_margin), 
                                              subtask_width, layer2_height,
                                              facecolor=subtask_light, edgecolor='black', 
                                              linewidth=1.5, alpha=0.9)
                ax.add_patch(agent_subtask_rect)
                
                # 标注 agent 名称（只标注 Top-3 且足够大的）
                if i < 3 and subtask_width > 0.08:
                    agent_label = display_agent_name(agent)
                    if len(agent_label) > 12:
                        agent_label = agent_label[:10] + "..."
                    ax.text(subtask_x_start + subtask_width/2, 
                           bottom_margin + layer2_height/2,
                           agent_label,
                           ha='center', va='center', fontsize=10, fontweight='bold',
                           fontfamily='Times New Roman',
                           rotation=90 if subtask_width < 0.12 else 0)
                
                subtask_x_start += subtask_width
        
        # 设置坐标轴
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('auto')
        ax.axis('off')
        ax.set_title(panel_title, fontsize=14, fontweight='bold', 
                    fontfamily='Times New Roman', pad=15)
        
        # 添加图例（只在左侧 panel）
        if panel_idx == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=plan_color, alpha=0.7, label='Plan'),
                Patch(facecolor=subtask_color, alpha=0.7, label='Subtask')
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=10, 
                     frameon=True, fancybox=True, shadow=False, 
                     prop={'family': 'Times New Roman'})
        
        # 计算并显示指标
        all_weights = [w[3] for w in sorted_weights]
        top1_share = compute_top1_share(all_weights)
        entropy = compute_entropy(all_weights)
        
        # 在右上角添加指标文本
        metrics_text = f'Top-1 share = {top1_share:.2f}\nEntropy = {entropy:.2f}'
        ax.text(0.98, 0.98, metrics_text,
               transform=ax.transAxes,
               ha='right', va='top',
               fontsize=10, fontfamily='Times New Roman',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, 
                        edgecolor='black', linewidth=1))
    
    # 绘制两个 panel
    plot_icicle_panel(ax1, baseline_sorted, baseline_plan, baseline_subtask, 
                     "Baseline (Uniform Multi-Agent)", 0)
    plot_icicle_panel(ax2, ours_sorted, ours_plan, ours_subtask, 
                     "Ours (Learned Routing)", 1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存
    outpath_png = os.path.join(result_dir, "icicle_cold_start_vs_overall.png")
    outpath_pdf = os.path.join(result_dir, "icicle_cold_start_vs_overall.pdf")
    
    try:
        fig.savefig(outpath_png, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Icicle PNG 已保存到: {outpath_png}")
        fig.savefig(outpath_pdf, bbox_inches='tight', facecolor='white')
        print(f"✅ Icicle PDF 已保存到: {outpath_pdf}")
    except Exception as e:
        print(f"❌ 保存图片时出错: {e}")
    
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("用法: python plot_icicle.py <result_dir>")
        print("示例: python plot_icicle.py select_only_results/medicalqa_c200_p300_t100/2026-01-24_02-10-37")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    if not os.path.isdir(result_dir):
        print(f"错误: 目录不存在: {result_dir}")
        sys.exit(1)
    
    # 读取数据文件
    phase_stats_path = os.path.join(result_dir, "phase_stats.json")
    plan_weight_path = os.path.join(result_dir, "plan_weight_sum_overall.json")
    subtask_weight_path = os.path.join(result_dir, "subtask_weight_sum_overall.json")
    
    if not os.path.exists(phase_stats_path):
        print(f"错误: 找不到文件: {phase_stats_path}")
        sys.exit(1)
    
    try:
        with open(phase_stats_path, 'r') as f:
            phase_stats = json.load(f)
        
        plan_weight_data = {}
        if os.path.exists(plan_weight_path):
            with open(plan_weight_path, 'r') as f:
                plan_weight_data = json.load(f)
        
        subtask_weight_data = {}
        if os.path.exists(subtask_weight_path):
            with open(subtask_weight_path, 'r') as f:
                subtask_weight_data = json.load(f)
        
        # 获取所有 agents
        all_agents = set()
        for phase_name, stats in phase_stats.items():
            all_agents.update(stats.get("plan_weight_sum", {}).keys())
            all_agents.update(stats.get("subtask_weight_sum", {}).keys())
        if plan_weight_data.get("weights"):
            all_agents.update(plan_weight_data["weights"].keys())
        if subtask_weight_data.get("weights"):
            all_agents.update(subtask_weight_data["weights"].keys())
        
        # 过滤掉 fallback
        agents_order = sorted([a for a in all_agents if a.lower() != "fallback"])
        if not agents_order:
            print("警告: 没有找到任何 agent 数据")
            return
        
        # 准备 Baseline 数据（cold_start，如果都是0则用均匀分布）
        cold_start_stats = phase_stats.get("cold_start", {})
        cold_start_plan_weights = cold_start_stats.get("plan_weight_sum", {})
        cold_start_subtask_weights = cold_start_stats.get("subtask_weight_sum", {})
        
        cold_plan_sum = sum(cold_start_plan_weights.values())
        cold_subtask_sum = sum(cold_start_subtask_weights.values())
        
        baseline_agent_weights = {}
        if cold_plan_sum == 0 and cold_subtask_sum == 0:
            # 均匀分布
            n_agents = len(agents_order)
            uniform_value = 1.0 / n_agents if n_agents > 0 else 0.0
            for agent in agents_order:
                baseline_agent_weights[agent] = (uniform_value, uniform_value)
        else:
            for agent in agents_order:
                baseline_agent_weights[agent] = (
                    cold_start_plan_weights.get(agent, 0.0),
                    cold_start_subtask_weights.get(agent, 0.0)
                )
        
        # 准备 Ours 数据（overall）
        overall_plan_weights = plan_weight_data.get("weights", {})
        overall_subtask_weights = subtask_weight_data.get("weights", {})
        
        # 如果 subtask_weight_sum_overall.json 没有 weights，从 phase_stats 汇总
        if not overall_subtask_weights:
            overall_subtask_weights = {}
            for phase_name, stats in phase_stats.items():
                phase_subtask_weights = stats.get("subtask_weight_sum", {})
                for agent, weight in phase_subtask_weights.items():
                    overall_subtask_weights[agent] = overall_subtask_weights.get(agent, 0.0) + weight
        
        ours_agent_weights = {}
        for agent in agents_order:
            ours_agent_weights[agent] = (
                overall_plan_weights.get(agent, 0.0),
                overall_subtask_weights.get(agent, 0.0)
            )
        
        # 生成 Icicle 图
        plot_icicle(
            result_dir=result_dir,
            baseline_agent_weights=baseline_agent_weights,
            ours_agent_weights=ours_agent_weights,
            agents_order=agents_order,
        )
        
    except Exception as e:
        print(f"错误: 处理目录 {result_dir} 时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
