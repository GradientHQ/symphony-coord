#!/usr/bin/env python3
"""
从已有的 JSON 统计数据生成 ICML 风格的 Agent Distribution Donut 图（简化版）
只显示 agent 分布，不显示 Plan/Subtask 分割
用法: python plot_agent_donut.py <result_dir>
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

def merge_small_agents(agent_weights: Dict[str, float], threshold: float = 0.02) -> Dict[str, float]:
    """合并小于 threshold 的 agent 到 "Other" """
    merged = {}
    other_total = 0.0
    
    total_all = sum(agent_weights.values())
    
    for agent, weight in agent_weights.items():
        share = weight / (total_all + 1e-10)
        
        if share < threshold:
            other_total += weight
        else:
            merged[agent] = weight
    
    if other_total > 0:
        merged["Other"] = other_total
    
    return merged

def plot_agent_donut(
    result_dir: str,
    baseline_agent_weights: Dict[str, float],
    ours_agent_weights: Dict[str, float],
    agents_order: List[str],
) -> None:
    """生成 ICML 风格的 Agent Distribution Donut 图"""
    
    # 合并小 agent
    baseline_merged = merge_small_agents(baseline_agent_weights, threshold=0.02)
    ours_merged = merge_small_agents(ours_agent_weights, threshold=0.02)
    
    # 归一化并排序（按 Ours 总权重降序）
    def normalize_and_sort(agent_weights: Dict[str, float], 
                          sort_key: Dict[str, float] = None) -> List[Tuple[str, float]]:
        """归一化并按总权重降序排列，返回 (agent, normalized_weight)"""
        total_all = sum(agent_weights.values())
        
        normalized = []
        for agent, weight in agent_weights.items():
            norm_weight = weight / (total_all + 1e-10)
            normalized.append((agent, norm_weight))
        
        # 如果有 sort_key，按它排序；否则按 weight 排序
        if sort_key:
            normalized.sort(key=lambda x: sort_key.get(x[0], 0.0), reverse=True)
        else:
            normalized.sort(key=lambda x: x[1], reverse=True)
        
        return normalized
    
    # 先计算 Ours 的总权重作为排序键
    ours_total_weights = ours_merged.copy()
    
    # 两个 panel 都按 Ours 总权重降序排列
    baseline_sorted = normalize_and_sort(baseline_merged, sort_key=ours_total_weights)
    ours_sorted = normalize_and_sort(ours_merged, sort_key=ours_total_weights)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Agent Selection Distribution', 
                 fontsize=18, fontweight='bold', fontfamily='Times New Roman', y=0.98)
    
    # 设置学术字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    # 定义颜色（使用灰度色阶，更学术）
    n_agents = len(ours_sorted)
    base_colors = plt.cm.Greys(np.linspace(0.3, 0.8, n_agents))
    
    def plot_donut_panel(ax, sorted_weights: List[Tuple[str, float]], 
                        panel_title: str, panel_idx: int, is_ours: bool = False):
        """绘制单个 donut panel"""
        # 提取数据
        agents = [w[0] for w in sorted_weights]
        weights = [w[1] for w in sorted_weights]
        
        # 计算角度
        total_sum = sum(weights)
        angles = [2 * np.pi * w / total_sum for w in weights]
        cum_angles = np.cumsum([0] + angles[:-1])
        
        # Donut 半径
        outer_radius = 0.8
        inner_radius = 0.4
        
        # 绘制扇区
        for i, (agent, weight, start_angle, angle) in enumerate(
            zip(agents, weights, cum_angles, angles)
        ):
            # 判断是否是 Top-1
            is_top1 = (i == 0) and is_ours
            
            # 颜色：Top-1 用深色，其他用灰度
            if is_top1:
                face_color = '#2c3e50'  # 深蓝灰色
                edge_color = '#000000'  # 黑色
                edge_width = 2.5
                explode_offset = 0.05  # 轻微 explode
            else:
                face_color = base_colors[i % len(base_colors)]
                edge_color = '#333333'  # 深灰色
                edge_width = 1.0
                explode_offset = 0.0
            
            # 计算 explode 后的位置
            explode_x = explode_offset * np.cos(start_angle + angle / 2)
            explode_y = explode_offset * np.sin(start_angle + angle / 2)
            
            # 绘制扇区
            wedge = Wedge((explode_x, explode_y), outer_radius, 
                         np.degrees(start_angle), 
                         np.degrees(start_angle + angle),
                         width=outer_radius - inner_radius,
                         facecolor=face_color, edgecolor=edge_color, 
                         linewidth=edge_width, alpha=0.8)
            ax.add_patch(wedge)
            
            # 标注 agent 名称（只标注 Top-1/Top-2）
            if i < 2 and angle > 0.08:  # 只标注足够大的扇区
                # 计算标签位置（donut 中间位置）
                label_angle = start_angle + angle / 2
                label_radius = (outer_radius + inner_radius) / 2
                label_x = label_radius * np.cos(label_angle) + explode_x
                label_y = label_radius * np.sin(label_angle) + explode_y
                
                agent_label = display_agent_name(agent)
                # 如果名称太长，截断
                if len(agent_label) > 15:
                    agent_label = agent_label[:13] + "..."
                
                # Top-1 用白色文字，其他用黑色
                text_color = 'white' if is_top1 else 'black'
                
                ax.text(label_x, label_y, agent_label,
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       fontfamily='Times New Roman', color=text_color,
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white' if not is_top1 else 'black', 
                                alpha=0.9 if not is_top1 else 0.7, edgecolor='none'))
        
        # 设置坐标轴
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(panel_title, fontsize=14, fontweight='bold', 
                    fontfamily='Times New Roman', pad=20)
        
        # 计算并显示指标
        top1_share = compute_top1_share(weights)
        entropy = compute_entropy(weights)
        
        # 在右上角添加指标文本
        metrics_text = f'Top-1 share = {top1_share:.2f}\nEntropy = {entropy:.2f}'
        ax.text(0.98, 0.98, metrics_text,
               transform=ax.transAxes,
               ha='right', va='top',
               fontsize=11, fontfamily='Times New Roman',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95, 
                        edgecolor='black', linewidth=1.5))
    
    # 绘制两个 panel
    plot_donut_panel(ax1, baseline_sorted, "Baseline (Uniform Multi-Agent)", 0, is_ours=False)
    plot_donut_panel(ax2, ours_sorted, "Ours (Learned Routing)", 1, is_ours=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存
    outpath_png = os.path.join(result_dir, "agent_donut_cold_start_vs_overall.png")
    outpath_pdf = os.path.join(result_dir, "agent_donut_cold_start_vs_overall.pdf")
    
    try:
        fig.savefig(outpath_png, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Agent Donut PNG 已保存到: {outpath_png}")
        fig.savefig(outpath_pdf, bbox_inches='tight', facecolor='white')
        print(f"✅ Agent Donut PDF 已保存到: {outpath_pdf}")
    except Exception as e:
        print(f"❌ 保存图片时出错: {e}")
    
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("用法: python plot_agent_donut.py <result_dir>")
        print("示例: python plot_agent_donut.py select_only_results/medicalqa_c200_p300_t100/2026-01-24_02-10-37")
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
                baseline_agent_weights[agent] = uniform_value
        else:
            for agent in agents_order:
                baseline_agent_weights[agent] = (
                    cold_start_plan_weights.get(agent, 0.0) +
                    cold_start_subtask_weights.get(agent, 0.0)
                )
        
        # 准备 Ours 数据（overall，Plan + Subtask）
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
                overall_plan_weights.get(agent, 0.0) +
                overall_subtask_weights.get(agent, 0.0)
            )
        
        # 生成 Agent Donut
        plot_agent_donut(
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
