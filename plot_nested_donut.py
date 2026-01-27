#!/usr/bin/env python3
"""
从已有的 JSON 统计数据生成 ICML 风格的 Nested Donut 图（内圈 Plan/Subtask，外圈 Agents）
用法: python plot_nested_donut.py <result_dir>
"""
import json
import os
import sys
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
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

def plot_nested_donut(
    result_dir: str,
    baseline_agent_weights: Dict[str, Tuple[float, float]],
    ours_agent_weights: Dict[str, Tuple[float, float]],
    agents_order: List[str],
) -> None:
    """生成 ICML 风格的 Nested Donut 图（内圈 Plan/Subtask，外圈 Agents）"""
    
    # 合并小 agent
    baseline_merged = merge_small_agents(baseline_agent_weights, threshold=0.02)
    ours_merged = merge_small_agents(ours_agent_weights, threshold=0.02)
    
    # 归一化并排序（按 Ours 总权重降序）
    def normalize_and_sort(agent_weights: Dict[str, Tuple[float, float]], 
                          sort_key: Dict[str, float] = None) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], float, float]:
        """归一化并按总权重降序排列，返回 (plan_agents, subtask_agents, plan_total, subtask_total)"""
        total_plan = sum(w[0] for w in agent_weights.values())
        total_subtask = sum(w[1] for w in agent_weights.values())
        total_all = total_plan + total_subtask
        
        plan_agents = []
        subtask_agents = []
        
        for agent, (plan_w, subtask_w) in agent_weights.items():
            norm_plan = plan_w / (total_plan + 1e-10)
            norm_subtask = subtask_w / (total_subtask + 1e-10)
            plan_agents.append((agent, norm_plan))
            subtask_agents.append((agent, norm_subtask))
        
        # 如果有 sort_key，按它排序；否则按 weight 排序
        if sort_key:
            plan_agents.sort(key=lambda x: sort_key.get(x[0], 0.0), reverse=True)
            subtask_agents.sort(key=lambda x: sort_key.get(x[0], 0.0), reverse=True)
        else:
            plan_agents.sort(key=lambda x: x[1], reverse=True)
            subtask_agents.sort(key=lambda x: x[1], reverse=True)
        
        return plan_agents, subtask_agents, total_plan / (total_all + 1e-10), total_subtask / (total_all + 1e-10)
    
    # 先计算 Ours 的总权重作为排序键
    ours_total_weights = {}
    for agent, (plan_w, subtask_w) in ours_merged.items():
        ours_total_weights[agent] = plan_w + subtask_w
    
    # 创建 agent 到字母的映射（A, B, C, D, E）
    # 按 Ours 总权重降序排列，然后分配字母
    sorted_agents = sorted(ours_merged.keys(), key=lambda x: ours_total_weights.get(x, 0.0), reverse=True)
    agent_to_letter = {}
    letters = ['A', 'B', 'C', 'D', 'E']
    for i, agent in enumerate(sorted_agents):
        if i < len(letters):
            agent_to_letter[agent] = letters[i]
        else:
            agent_to_letter[agent] = f"Agent{i+1}"
    
    # 定义 5 个固定颜色（用于 A~E）
    agent_colors_fixed = {
        'A': '#2E86AB',  # 蓝色
        'B': '#A23B72',  # 紫色
        'C': '#F18F01',  # 橙色
        'D': '#C73E1D',  # 红色
        'E': '#6A994E',  # 绿色
    }
    # 如果超过 5 个，使用灰度
    default_colors = plt.cm.Greys(np.linspace(0.3, 0.7, 10))
    
    # 两个 panel 都按 Ours 总权重降序排列
    baseline_plan, baseline_subtask, baseline_plan_ratio, baseline_subtask_ratio = normalize_and_sort(
        baseline_merged, sort_key=ours_total_weights)
    ours_plan, ours_subtask, ours_plan_ratio, ours_subtask_ratio = normalize_and_sort(
        ours_merged, sort_key=ours_total_weights)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Plan-level vs Subtask-level Weight Distribution', 
                 fontsize=18, fontweight='bold', fontfamily='Times New Roman', y=0.98)
    
    # 设置学术字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    # 定义颜色（Sunburst 风格：内层主色）
    plan_base_color = '#1f77b4'  # 蓝色（冷色）
    subtask_base_color = '#ff7f0e'  # 橙色（暖色）
    
    def plot_nested_donut_panel(ax, plan_agents: List[Tuple[str, float]], 
                                subtask_agents: List[Tuple[str, float]],
                                plan_ratio: float, subtask_ratio: float,
                                panel_title: str, panel_idx: int,
                                agent_to_letter: Dict[str, str],
                                agent_colors_fixed: Dict[str, str],
                                default_colors):
        """绘制单个 nested donut panel"""
        
        # 内圈：Plan vs Subtask（Sunburst 风格：从中心开始）
        inner_outer_radius = 0.45
        inner_inner_radius = 0.0  # 从中心开始，更像 Sunburst
        
        # 计算内圈角度
        plan_angle = 2 * np.pi * plan_ratio
        subtask_angle = 2 * np.pi * subtask_ratio
        
        # 绘制内圈 Plan 部分（Sunburst 风格：更深的颜色）
        if plan_angle > 0.01:
            wedge_plan = Wedge((0, 0), inner_outer_radius, 0, 
                               np.degrees(plan_angle),
                               width=inner_outer_radius - inner_inner_radius,
                               facecolor=plan_base_color, edgecolor='black', linewidth=2.0, alpha=0.85)
            ax.add_patch(wedge_plan)
            
            # Plan 标签
            label_angle = plan_angle / 2
            label_radius = (inner_outer_radius + inner_inner_radius) / 2
            label_x = label_radius * np.cos(label_angle)
            label_y = label_radius * np.sin(label_angle)
            ax.text(label_x, label_y, 'Plan',
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   fontfamily='Times New Roman', color='white')
        
        # 绘制内圈 Subtask 部分（Sunburst 风格：更深的颜色）
        if subtask_angle > 0.01:
            wedge_subtask = Wedge((0, 0), inner_outer_radius, 
                                 np.degrees(plan_angle),
                                 np.degrees(plan_angle + subtask_angle),
                                 width=inner_outer_radius - inner_inner_radius,
                                 facecolor=subtask_base_color, edgecolor='black', linewidth=2.0, alpha=0.85)
            ax.add_patch(wedge_subtask)
            
            # Subtask 标签
            label_angle = plan_angle + subtask_angle / 2
            label_radius = (inner_outer_radius + inner_inner_radius) / 2
            label_x = label_radius * np.cos(label_angle)
            label_y = label_radius * np.sin(label_angle)
            ax.text(label_x, label_y, 'Subtask',
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   fontfamily='Times New Roman', color='white')
        
        # 外圈：Agents（在 Plan 和 Subtask 区域内分别绘制，从内圈延伸）
        outer_outer_radius = 0.85
        outer_inner_radius = 0.45  # 从内圈外边缘开始
        
        # Plan 区域内的 agents
        if plan_angle > 0.01 and plan_agents:
            plan_total = sum(w for _, w in plan_agents)
            plan_angles = [plan_angle * w / (plan_total + 1e-10) for _, w in plan_agents]
            plan_cum_angles = np.cumsum([0] + plan_angles[:-1])
            
            for i, (agent, weight, start_angle, angle) in enumerate(
                zip([a for a, _ in plan_agents], [w for _, w in plan_agents], 
                    plan_cum_angles, plan_angles)
            ):
                if angle > 0.01:  # 只绘制足够大的扇区
                    # 获取 agent 对应的字母和颜色
                    letter = agent_to_letter.get(agent, f"Agent{i+1}")
                    agent_color = agent_colors_fixed.get(letter, default_colors[i % len(default_colors)])
                    
                    # 使用固定颜色（Plan 和 Subtask 区域相同 agent 用相同颜色）
                    wedge = Wedge((0, 0), outer_outer_radius, 
                                 np.degrees(start_angle), 
                                 np.degrees(start_angle + angle),
                                 width=outer_outer_radius - outer_inner_radius,
                                 facecolor=agent_color, 
                                 edgecolor='black', linewidth=1.2, alpha=0.75)
                    ax.add_patch(wedge)
                    
                    # 标注字母（只标注 Top-2 且角度足够大）
                    if i < 2 and angle > 0.15:
                        label_angle = start_angle + angle / 2
                        label_radius = (outer_outer_radius + outer_inner_radius) / 2
                        label_x = label_radius * np.cos(label_angle)
                        label_y = label_radius * np.sin(label_angle)
                        
                        ax.text(label_x, label_y, letter,
                               ha='center', va='center', fontsize=10, fontweight='bold',
                               fontfamily='Times New Roman',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                        alpha=0.9, edgecolor='black', linewidth=0.5))
        
        # Subtask 区域内的 agents
        if subtask_angle > 0.01 and subtask_agents:
            subtask_total = sum(w for _, w in subtask_agents)
            subtask_angles = [subtask_angle * w / (subtask_total + 1e-10) for _, w in subtask_agents]
            subtask_cum_angles = np.cumsum([plan_angle] + subtask_angles[:-1])
            
            for i, (agent, weight, start_angle, angle) in enumerate(
                zip([a for a, _ in subtask_agents], [w for _, w in subtask_agents], 
                    subtask_cum_angles, subtask_angles)
            ):
                if angle > 0.01:  # 只绘制足够大的扇区
                    # 获取 agent 对应的字母和颜色（与 Plan 区域相同）
                    letter = agent_to_letter.get(agent, f"Agent{i+1}")
                    agent_color = agent_colors_fixed.get(letter, default_colors[i % len(default_colors)])
                    
                    # 使用固定颜色（Plan 和 Subtask 区域相同 agent 用相同颜色）
                    wedge = Wedge((0, 0), outer_outer_radius, 
                                 np.degrees(start_angle), 
                                 np.degrees(start_angle + angle),
                                 width=outer_outer_radius - outer_inner_radius,
                                 facecolor=agent_color, 
                                 edgecolor='black', linewidth=1.2, alpha=0.75)
                    ax.add_patch(wedge)
                    
                    # 标注字母（只标注 Top-2 且角度足够大）
                    if i < 2 and angle > 0.15:
                        label_angle = start_angle + angle / 2
                        label_radius = (outer_outer_radius + outer_inner_radius) / 2
                        label_x = label_radius * np.cos(label_angle)
                        label_y = label_radius * np.sin(label_angle)
                        
                        ax.text(label_x, label_y, letter,
                               ha='center', va='center', fontsize=10, fontweight='bold',
                               fontfamily='Times New Roman',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                        alpha=0.9, edgecolor='black', linewidth=0.5))
        
        # 设置坐标轴
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(panel_title, fontsize=14, fontweight='bold', 
                    fontfamily='Times New Roman', pad=20)
        
        # 添加图例（只在左侧 panel）
        if panel_idx == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=plan_base_color, alpha=0.85, label='Plan'),
                Patch(facecolor=subtask_base_color, alpha=0.85, label='Subtask')
            ]
            # 添加 agent 颜色图例
            for letter in ['A', 'B', 'C', 'D', 'E']:
                if letter in agent_colors_fixed:
                    color = agent_colors_fixed[letter]
                    # 找到对应的 agent 名称
                    agent_name = None
                    for agent, let in agent_to_letter.items():
                        if let == letter:
                            agent_name = display_agent_name(agent)
                            break
                    if agent_name:
                        legend_elements.append(
                            Patch(facecolor=color, alpha=0.75, label=f'Agent {letter}: {agent_name}')
                        )
            
            ax.legend(handles=legend_elements, loc='lower left', fontsize=9, 
                     frameon=True, fancybox=True, shadow=False, 
                     prop={'family': 'Times New Roman'})
        
        # 计算并显示指标（基于所有 agent 的总权重）
        # 合并 plan 和 subtask 的权重
        agent_total_weights = {}
        for agent, weight in plan_agents:
            agent_total_weights[agent] = agent_total_weights.get(agent, 0.0) + weight
        for agent, weight in subtask_agents:
            agent_total_weights[agent] = agent_total_weights.get(agent, 0.0) + weight
        all_weights = list(agent_total_weights.values())
        
        top1_share = compute_top1_share(all_weights)
        entropy = compute_entropy(all_weights)
        
        # 在右上角添加指标文本
        metrics_text = f'Top-1 share = {top1_share:.2f}\nEntropy = {entropy:.2f}'
        ax.text(0.98, 0.98, metrics_text,
               transform=ax.transAxes,
               ha='right', va='top',
               fontsize=11, fontfamily='Times New Roman',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95, 
                        edgecolor='black', linewidth=1.5))
    
    # 绘制两个 panel
    plot_nested_donut_panel(ax1, baseline_plan, baseline_subtask, 
                            baseline_plan_ratio, baseline_subtask_ratio,
                            "Baseline (Uniform Multi-Agent)", 0,
                            agent_to_letter, agent_colors_fixed, default_colors)
    plot_nested_donut_panel(ax2, ours_plan, ours_subtask,
                            ours_plan_ratio, ours_subtask_ratio,
                            "Ours (Learned Routing)", 1,
                            agent_to_letter, agent_colors_fixed, default_colors)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存
    outpath_png = os.path.join(result_dir, "nested_donut_cold_start_vs_overall.png")
    outpath_pdf = os.path.join(result_dir, "nested_donut_cold_start_vs_overall.pdf")
    
    try:
        fig.savefig(outpath_png, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Nested Donut PNG 已保存到: {outpath_png}")
        fig.savefig(outpath_pdf, bbox_inches='tight', facecolor='white')
        print(f"✅ Nested Donut PDF 已保存到: {outpath_pdf}")
    except Exception as e:
        print(f"❌ 保存图片时出错: {e}")
    
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("用法: python plot_nested_donut.py <result_dir>")
        print("示例: python plot_nested_donut.py select_only_results/medicalqa_c200_p300_t100/2026-01-24_02-10-37")
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
        
        # 生成 Nested Donut
        plot_nested_donut(
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
