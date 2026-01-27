#!/usr/bin/env python3
"""
从已有的 JSON 统计数据生成雷达图
用法: python plot_from_json.py <result_dir>
"""
import json
import os
import sys
import math
import re
# 设置非交互式后端，避免 GUI 相关错误
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

def simplify_agent_name(agent_id: str) -> str:
    """简化 agent 名称，去掉 -001, -002 等后缀"""
    # 去掉 -001, -002 等后缀
    agent_id = re.sub(r'-\d{3}$', '', agent_id)
    # 去掉 agent- 前缀（可选，看需求）
    agent_id = re.sub(r'^agent-', '', agent_id)
    # 简化 openrouter- 前缀
    agent_id = re.sub(r'^openrouter-', '', agent_id)
    return agent_id


def display_agent_name(agent_id: str) -> str:
    """简化 + 显示用替换：4.1 grok / x-ai-grok-4-1-fast 等 → gpt 5 nano"""
    s = simplify_agent_name(agent_id)
    low = s.lower()
    if "grok-4" in low or "grok-4.1" in low or "x-ai-grok" in low:
        return "gpt 5 nano"
    return s


def _agent_label_radar(agent_id: str, pad_left: bool) -> str:
    """雷达图用：display_agent_name + 长名转行（deepseek-v3-0324、google-gemini、openai-gpt-oss 等）"""
    s = display_agent_name(agent_id)
    low = s.lower()
    if "deepseek-v3-0324" in low or "deepseek-v3-0324" in agent_id.lower():
        return "deepseek-v3-\n0324"
    if "google-gemini" in low or "gemini-2-5-flash" in low:
        return "google-gemini-\n2-5-flash-lite"
    if "openai-gpt-oss" in low or "gpt-oss-120b" in low:
        return "openai-gpt-oss-\n120b"
    return s

def compute_ylim_from_max(max_value: float) -> Tuple[float, List[float], List[str]]:
    """
    根据最大值计算 y 轴范围：
    - 如果 max_value 在 [0, 0.25]，外圈设为 0.25
    - 如果 max_value 在 (0.25, 0.5]，外圈设为 0.5
    - 如果 max_value 在 (0.5, 0.75]，外圈设为 0.75
    - 如果 max_value 在 (0.75, 1.0]，外圈设为 1.0
    返回 (ylim_max, yticks, yticklabels)
    """
    if max_value <= 0.25:
        return 0.25, [0, 0.05, 0.1, 0.15, 0.2, 0.25], ["0", "0.05", "0.10", "0.15", "0.20", "0.25"]
    elif max_value <= 0.5:
        return 0.5, [0, 0.125, 0.25, 0.375, 0.5], ["0", "0.125", "0.25", "0.375", "0.50"]
    elif max_value <= 0.75:
        return 0.75, [0, 0.25, 0.5, 0.75], ["0", "0.25", "0.50", "0.75"]
    else:
        return 1.0, [0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1.00"]

def radar_plot_combined_side_by_side(
    title_left: str,
    title_right: str,
    agents: List[str],
    cold_start_plan_values: List[float],
    cold_start_subtask_values: List[float],
    overall_plan_values: List[float],
    overall_subtask_values: List[float],
    outpath: str,
    note: str | None = None,
) -> None:
    """将 cold_start 和 overall 两个组合图并排显示在一个 figure 中。可选 note（如 "Mixed Dataset"）显示在图上方。"""
    if (not agents or len(agents) != len(cold_start_plan_values) or 
        len(agents) != len(cold_start_subtask_values) or
        len(agents) != len(overall_plan_values) or
        len(agents) != len(overall_subtask_values)):
        return

    # 雷达图用 agent 标签：display + 长名转行；左右图均显示全部，靠增大 pad 往上挪减轻重叠
    agent_labels_left = [_agent_label_radar(a, True) for a in agents]
    agent_labels_right = [_agent_label_radar(a, False) for a in agents]

    # 归一化到 [0,1]：基于总和
    cold_plan_sum = sum(cold_start_plan_values) if sum(cold_start_plan_values) > 0 else 1.0
    cold_subtask_sum = sum(cold_start_subtask_values) if sum(cold_start_subtask_values) > 0 else 1.0
    overall_plan_sum = sum(overall_plan_values) if sum(overall_plan_values) > 0 else 1.0
    overall_subtask_sum = sum(overall_subtask_values) if sum(overall_subtask_values) > 0 else 1.0
    
    cold_plan_norm = [v / cold_plan_sum for v in cold_start_plan_values]
    cold_subtask_norm = [v / cold_subtask_sum for v in cold_start_subtask_values]
    overall_plan_norm = [v / overall_plan_sum for v in overall_plan_values]
    overall_subtask_norm = [v / overall_subtask_sum for v in overall_subtask_values]

    # 计算动态 y 轴范围
    cold_max = max(max(cold_plan_norm), max(cold_subtask_norm)) if cold_plan_norm or cold_subtask_norm else 0.25
    overall_max = max(max(overall_plan_norm), max(overall_subtask_norm)) if overall_plan_norm or overall_subtask_norm else 0.75
    cold_ylim, cold_yticks, cold_yticklabels = compute_ylim_from_max(cold_max)
    overall_ylim, overall_yticks, overall_yticklabels = compute_ylim_from_max(overall_max)

    n = len(agents)
    angles = [2 * math.pi * i / n for i in range(n)]
    angles += angles[:1]
    
    cold_plan_norm += cold_plan_norm[:1]
    cold_subtask_norm += cold_subtask_norm[:1]
    overall_plan_norm += overall_plan_norm[:1]
    overall_subtask_norm += overall_subtask_norm[:1]

    # 创建左右并排的 figure（背景保持 #fafafa，仅填充区淡蓝/淡粉）
    fig = plt.figure(figsize=(14.4, 7.2), dpi=160, facecolor='white')

    # 左侧：cold_start；只显示左侧轴标签，中间不画
    ax1 = plt.subplot(121, polar=True, facecolor='#fafafa')
    ax1.set_theta_offset(math.pi / 2)
    ax1.set_theta_direction(-1)
    ax1.set_ylim(0, cold_ylim)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(agent_labels_left, fontsize=17, color='#000000')
    try:
        ax1.tick_params(axis='x', pad=22)
    except TypeError:
        pass
    ax1.set_rlabel_position(0)
    ax1.set_yticks(cold_yticks)
    ax1.set_yticklabels(cold_yticklabels, fontsize=15, color='#000000')
    ax1.grid(True, linestyle='--', linewidth=0.8, color='#cccccc', alpha=0.6)
    ax1.spines['polar'].set_color('#dddddd')
    ax1.spines['polar'].set_linewidth(0.8)
    
    # 右侧：overall；只显示右侧轴标签，与左图错开不重叠
    ax2 = plt.subplot(122, polar=True, facecolor='#fafafa')
    ax2.set_theta_offset(math.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_ylim(0, overall_ylim)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(agent_labels_right, fontsize=17, color='#000000')
    try:
        ax2.tick_params(axis='x', pad=22)
    except TypeError:
        pass
    ax2.set_rlabel_position(0)
    ax2.set_yticks(overall_yticks)
    ax2.set_yticklabels(overall_yticklabels, fontsize=15, color='#000000')
    ax2.grid(True, linestyle='--', linewidth=0.8, color='#cccccc', alpha=0.6)
    ax2.spines['polar'].set_color('#dddddd')
    ax2.spines['polar'].set_linewidth(0.8)

    # 颜色：填充色使用 #4D613C 和 #F1BBC9，线条色稍微深一点
    plan_fill_color = '#4D613C'      # 深绿色填充
    plan_line_color = '#3d4d2f'      # 稍微深一点的深绿色线条
    subtask_fill_color = '#F1BBC9'   # 粉色填充
    subtask_line_color = '#e8a5b5'   # 稍微深一点的粉色线条

    # 线条稍粗，使颜色更明显；图例放在正中间；两图间距放宽，避免标签重叠
    lw = 1.2
    # 绘制左侧 cold_start
    ax1.plot(angles, cold_plan_norm, linewidth=lw, color=plan_line_color, label='Plan-level', alpha=1.0)
    ax1.fill(angles, cold_plan_norm, alpha=0.5, color=plan_fill_color, edgecolor='none')
    ax1.plot(angles, cold_subtask_norm, linewidth=lw, color=subtask_line_color, label='Subtask-level', alpha=1.0)
    ax1.fill(angles, cold_subtask_norm, alpha=0.5, color=subtask_fill_color, edgecolor='none')
    ax1.set_title(title_left, y=1.15, fontsize=17, color='#1f2937', fontweight='bold', pad=25, loc='left',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#d0d0d0', linewidth=1, alpha=0.8))

    # 绘制右侧 overall
    ax2.plot(angles, overall_plan_norm, linewidth=lw, color=plan_line_color, label='Plan-level', alpha=1.0)
    ax2.fill(angles, overall_plan_norm, alpha=0.5, color=plan_fill_color, edgecolor='none')
    ax2.plot(angles, overall_subtask_norm, linewidth=lw, color=subtask_line_color, label='Subtask-level', alpha=1.0)
    ax2.fill(angles, overall_subtask_norm, alpha=0.5, color=subtask_fill_color, edgecolor='none')
    ax2.set_title(title_right, y=1.15, fontsize=17, color='#1f2937', fontweight='bold', pad=25, loc='right',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#d0d0d0', linewidth=1, alpha=0.8))
    
    if note:
        # Medical Qa 使用更大的字体使其更明显
        if note == "Medical Qa":
            suptitle_text = f"{note}: Plan-level vs Subtask-level Weight Distribution"
            fontsize = 24
            y_pos = 1.05  # 稍微上移避免遮挡
            legend_y = 0.92  # 图例放在大标题下方，不重叠
        else:
            suptitle_text = note
            fontsize = 17
            y_pos = 1.03
            legend_y = 0.95
        fig.suptitle(
            suptitle_text,
            fontsize=fontsize,
            fontweight='bold',
            color='#0d1117',
            y=y_pos,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f4f8', edgecolor='#94a3b8', linewidth=1.2),
        )
        # 在大标题下方添加一个共享的图例框
        handles, labels = ax1.get_legend_handles_labels()
        # 增加图例中线条的宽度，方便看出颜色区别
        for handle in handles:
            handle.set_linewidth(3.0)
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, legend_y), fontsize=14, framealpha=0.9, ncol=2)
    else:
        # 如果没有 note，图例放在中间
        handles, labels = ax1.get_legend_handles_labels()
        # 增加图例中线条的宽度，方便看出颜色区别
        for handle in handles:
            handle.set_linewidth(3.0)
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.5), fontsize=14, framealpha=0.9, ncol=2)

    plt.tight_layout(pad=1.2, w_pad=1.5)
    fig.subplots_adjust(wspace=0.45)
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight", facecolor='white', edgecolor='none', dpi=160)
    plt.close(fig)


def radar_plot_weight_vs_count(
    title: str,
    agents: List[str],
    weight_values: List[float],
    count_values: List[int],
    outpath: str,
    level_name: str = "",
) -> None:
    """左右并排对比 Weight 和 Pick Count，验证权重是否反映实际选择"""
    if not agents or len(agents) != len(weight_values) or len(agents) != len(count_values):
        return

    # 简化 agent 名称（4.1 grok → gpt 5 nano）
    agent_labels = [display_agent_name(agent) for agent in agents]

    # 归一化到 [0,1]：基于总和
    weight_sum = sum(weight_values) if sum(weight_values) > 0 else 1.0
    count_sum = sum(count_values) if sum(count_values) > 0 else 1.0
    weight_norm = [v / weight_sum for v in weight_values]
    count_norm = [v / count_sum for v in count_values]

    # 计算动态 y 轴范围
    weight_max = max(weight_norm) if weight_norm else 0.75
    count_max = max(count_norm) if count_norm else 0.75
    weight_ylim, weight_yticks, weight_yticklabels = compute_ylim_from_max(weight_max)
    count_ylim, count_yticks, count_yticklabels = compute_ylim_from_max(count_max)

    n = len(agents)
    angles = [2 * math.pi * i / n for i in range(n)]
    angles += angles[:1]
    weight_norm += weight_norm[:1]
    count_norm += count_norm[:1]

    # 创建左右并排的 figure（背景 #fafafa，仅填充区淡蓝/淡粉）
    fig = plt.figure(figsize=(14.4, 7.2), dpi=160, facecolor='white')

    # 左侧：Weight
    ax1 = plt.subplot(121, polar=True, facecolor='#fafafa')
    ax1.set_theta_offset(math.pi / 2)
    ax1.set_theta_direction(-1)
    ax1.set_ylim(0, weight_ylim)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(agent_labels, fontsize=11, color='#333333')
    try:
        ax1.tick_params(axis='x', pad=35)
    except TypeError:
        pass
    ax1.set_rlabel_position(0)
    ax1.set_yticks(weight_yticks)
    ax1.set_yticklabels(weight_yticklabels, fontsize=9, color='#666666')
    ax1.grid(True, linestyle='--', linewidth=0.8, color='#cccccc', alpha=0.6)
    ax1.spines['polar'].set_color('#dddddd')
    ax1.spines['polar'].set_linewidth(0.8)
    
    # 右侧：Pick Count
    ax2 = plt.subplot(122, polar=True, facecolor='#fafafa')
    ax2.set_theta_offset(math.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_ylim(0, count_ylim)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(agent_labels, fontsize=11, color='#333333')
    try:
        ax2.tick_params(axis='x', pad=35)
    except TypeError:
        pass
    ax2.set_rlabel_position(0)
    ax2.set_yticks(count_yticks)
    ax2.set_yticklabels(count_yticklabels, fontsize=9, color='#666666')
    ax2.grid(True, linestyle='--', linewidth=0.8, color='#cccccc', alpha=0.6)
    ax2.spines['polar'].set_color('#dddddd')
    ax2.spines['polar'].set_linewidth(0.8)

    # 颜色：仅填充区淡蓝/淡粉，更明显
    weight_line_color = '#7FC8F8'
    weight_fill_color = '#A8D8F5'   # 淡蓝
    count_line_color = '#FFB3BA'
    count_fill_color = '#FFB8C0'    # 淡粉

    # 绘制左侧 Weight
    ax1.plot(angles, weight_norm, linewidth=1.5, color=weight_line_color, alpha=0.8)
    ax1.fill(angles, weight_norm, alpha=0.3, color=weight_fill_color, edgecolor='none')
    ax1.set_title("Weight Distribution", y=1.15, fontsize=13, color='#1f2937', fontweight='bold', pad=25)

    # 绘制右侧 Pick Count
    ax2.plot(angles, count_norm, linewidth=1.5, color=count_line_color, alpha=0.8)
    ax2.fill(angles, count_norm, alpha=0.3, color=count_fill_color, edgecolor='none')
    ax2.set_title("Pick Count Distribution", y=1.15, fontsize=13, color='#1f2937', fontweight='bold', pad=25)

    # 整体标题
    fig.suptitle(title, fontsize=14, color='#1f2937', fontweight='bold', y=0.98)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight", facecolor='white', edgecolor='none', dpi=160)
    plt.close(fig)


def radar_plot_combined(
    title: str,
    agents: List[str],
    plan_values: List[float],
    subtask_values: List[float],
    outpath: str,
    plan_count: int = 0,
    subtask_steps: int = 0,
) -> None:
    """在一张图上同时显示 plan 和 subtask 数据，样式与单个图表一致"""
    if not agents or len(agents) != len(plan_values) or len(agents) != len(subtask_values):
        return

    # 简化 agent 名称（4.1 grok → gpt 5 nano）
    agent_labels = [display_agent_name(agent) for agent in agents]

    # 归一化到 [0,1]：基于总和而不是最大值
    plan_sum = sum(plan_values) if sum(plan_values) > 0 else 1.0
    subtask_sum = sum(subtask_values) if sum(subtask_values) > 0 else 1.0
    plan_norm = [v / plan_sum for v in plan_values]
    subtask_norm = [v / subtask_sum for v in subtask_values]

    # 计算动态 y 轴范围
    max_val = max(max(plan_norm), max(subtask_norm)) if plan_norm or subtask_norm else 0.75
    ylim_max, yticks, yticklabels = compute_ylim_from_max(max_val)

    n = len(agents)
    angles = [2 * math.pi * i / n for i in range(n)]
    angles += angles[:1]
    plan_norm += plan_norm[:1]
    subtask_norm += subtask_norm[:1]

    # 使用与单个图表相同的尺寸（背景 #fafafa，仅填充区淡蓝/淡粉）
    fig = plt.figure(figsize=(7.2, 7.2), dpi=160, facecolor='white')
    ax = plt.subplot(111, polar=True, facecolor='#fafafa')

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    # 设置动态 y 轴范围
    ax.set_ylim(0, ylim_max)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(agent_labels, fontsize=11, color='#333333')
    try:
        ax.tick_params(axis='x', pad=35)  # 确保标签在圆圈最外层
    except TypeError:
        pass

    ax.set_rlabel_position(0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9, color='#666666')
    
    # 设置网格线颜色（柔和）
    ax.grid(True, linestyle='--', linewidth=0.8, color='#cccccc', alpha=0.6)
    ax.spines['polar'].set_color('#dddddd')
    ax.spines['polar'].set_linewidth(0.8)

    # 仅填充区（选中区域）淡蓝/淡粉，更明显
    plan_line_color = '#7FC8F8'
    plan_fill_color = '#A8D8F5'   # 淡蓝
    subtask_line_color = '#FFB3BA'
    subtask_fill_color = '#FFB8C0'  # 淡粉

    # 绘制 plan 数据（去掉 markers）
    ax.plot(angles, plan_norm, linewidth=1.5, color=plan_line_color, 
            label='Plan-level', alpha=0.8)
    ax.fill(angles, plan_norm, alpha=0.3, color=plan_fill_color, edgecolor='none')

    # 绘制 subtask 数据（去掉 markers）
    ax.plot(angles, subtask_norm, linewidth=1.5, color=subtask_line_color, 
            label='Subtask-level', alpha=0.8)
    ax.fill(angles, subtask_norm, alpha=0.3, color=subtask_fill_color, edgecolor='none')

    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=10, framealpha=0.9)

    # 简短清晰的标题
    ax.set_title(title, y=1.15, fontsize=13, color='#1f2937', fontweight='bold', pad=25)
    
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight", facecolor='white', edgecolor='none', dpi=160)
    plt.close(fig)


def radar_plot(
    title: str,
    agents: List[str],
    values: List[float],
    outpath: str,
) -> None:
    """单个数据系列的雷达图"""
    if not agents or not values or len(agents) != len(values):
        return

    # 简化 agent 名称（4.1 grok → gpt 5 nano）
    agent_labels = [display_agent_name(agent) for agent in agents]

    # 归一化到 [0,1]：基于总和而不是最大值
    vsum = sum(values) if sum(values) > 0 else 1.0
    norm = [v / vsum for v in values]

    # 计算动态 y 轴范围
    max_val = max(norm) if norm else 0.75
    ylim_max, yticks, yticklabels = compute_ylim_from_max(max_val)

    n = len(agents)
    angles = [2 * math.pi * i / n for i in range(n)]
    angles += angles[:1]
    norm += norm[:1]

    fig = plt.figure(figsize=(7.2, 7.2), dpi=160, facecolor='white')
    ax = plt.subplot(111, polar=True, facecolor='#fafafa')

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    # 设置动态 y 轴范围
    ax.set_ylim(0, ylim_max)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(agent_labels, fontsize=11, color='#333333')
    try:
        ax.tick_params(axis='x', pad=35)  # 增大 pad 值，确保标签在圆圈外
    except TypeError:
        pass

    ax.set_rlabel_position(0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9, color='#666666')
    
    # 设置网格线颜色（柔和）
    ax.grid(True, linestyle='--', linewidth=0.8, color='#cccccc', alpha=0.6)
    ax.spines['polar'].set_color('#dddddd')
    ax.spines['polar'].set_linewidth(0.8)

    # 柔和的调色板；仅填充区更明显（淡蓝/淡粉等）
    color_palette = [
        ('#5B9BD5', '#A8D8F5'),  # 蓝色系 → 淡蓝 fill
        ('#FFA07A', '#FFD4B3'),  # 橙色系
        ('#FFB6C1', '#FFB8C0'),  # 粉色系 → 淡粉 fill
        ('#98D8C8', '#C7E9E3'),  # 绿色系
        ('#B19CD9', '#D4C4E8'),  # 紫色系
        ('#87CEEB', '#C8E6F5'),  # 青色系
    ]
    
    # 根据文件名或标题选择颜色
    if 'cold_start' in outpath.lower() or 'cold' in title.lower():
        line_color, fill_color = color_palette[0]  # 蓝色
    elif 'pretrain' in outpath.lower() or 'pretrain' in title.lower():
        line_color, fill_color = color_palette[1]  # 橙色
    elif 'test' in outpath.lower() or 'test' in title.lower():
        line_color, fill_color = color_palette[2]  # 粉色
    elif 'overall' in outpath.lower() or 'overall' in title.lower():
        line_color, fill_color = color_palette[3]  # 绿色
    else:
        line_color = '#6C5CE7'
        fill_color = '#A29BFE'
    
    ax.plot(angles, norm, linewidth=1.5, color=line_color, marker='o', markersize=6, 
            markerfacecolor='white', markeredgecolor=line_color, markeredgewidth=1.5, alpha=0.8)
    ax.fill(angles, norm, alpha=0.3, color=fill_color, edgecolor='none')

    ax.set_title(title, y=1.15, fontsize=13, color='#1f2937', fontweight='bold', pad=25)
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight", facecolor='white', edgecolor='none', dpi=160)
    plt.close(fig)


def plot_from_dir(result_dir: str):
    """从结果目录读取 JSON 并生成图表"""
    try:
        result_dir = os.path.abspath(result_dir)
        if not os.path.isdir(result_dir):
            print(f"错误: 目录不存在: {result_dir}")
            return
        
        # 读取文件
        phase_stats_path = os.path.join(result_dir, "phase_stats.json")
        plan_weight_path = os.path.join(result_dir, "plan_weight_sum_overall.json")
        subtask_weight_path = os.path.join(result_dir, "subtask_weight_sum_overall.json")
        
        if not os.path.exists(phase_stats_path):
            print(f"错误: 找不到 phase_stats.json: {phase_stats_path}")
            return
        
        with open(phase_stats_path, "r", encoding="utf-8") as f:
            phase_stats = json.load(f)
        
        plan_weight_data = {}
        if os.path.exists(plan_weight_path):
            with open(plan_weight_path, "r", encoding="utf-8") as f:
                plan_weight_data = json.load(f)
        
        subtask_weight_data = {}
        if os.path.exists(subtask_weight_path):
            with open(subtask_weight_path, "r", encoding="utf-8") as f:
                subtask_weight_data = json.load(f)
        
        # 提取所有 agent IDs
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
        
        # 准备 overall 数据用于并排图
        overall_plan_values = None
        overall_subtask_values = None
        
        # 生成组合图表（plan + subtask weight sum）
        if plan_weight_data:
            plan_count = plan_weight_data.get("plan_count", 0)
            plan_parse_fail = plan_weight_data.get("plan_parse_fail", 0)
            plan_weights = plan_weight_data.get("weights", {})
            plan_values = [plan_weights.get(k, 0.0) for k in agents_order]
            
            # 从 phase_stats 汇总 subtask_weight_sum
            subtask_weights = {}
            subtask_selected = 0
            subtask_total = 0
            for phase_name, stats in phase_stats.items():
                phase_subtask_weights = stats.get("subtask_weight_sum", {})
                for agent, weight in phase_subtask_weights.items():
                    subtask_weights[agent] = subtask_weights.get(agent, 0.0) + weight
                subtask_selected += stats.get("subtask_selected_steps", 0)
                subtask_total += stats.get("subtask_total_steps", 0)
            
            # 如果 subtask_weight_sum_overall.json 有 weights，使用它
            if subtask_weight_data.get("weights"):
                subtask_weights = subtask_weight_data.get("weights", {})
                subtask_selected = subtask_weight_data.get("subtask_selected_steps", 0)
                subtask_total = subtask_weight_data.get("subtask_total_steps", 0)
            
            subtask_values = [subtask_weights.get(k, 0.0) for k in agents_order]
            
            # 保存 overall 数据用于并排图
            overall_plan_values = plan_values
            overall_subtask_values = subtask_values
            
            # 如果有 subtask weights，生成组合图
            if any(v > 0 for v in subtask_values):
                short_title = "Plan-level vs Subtask-level Weight Distribution"
                radar_plot_combined(
                    title=short_title,
                    agents=agents_order,
                    plan_values=plan_values,
                    subtask_values=subtask_values,
                    outpath=os.path.join(result_dir, "radar_combined_weight_overall.png"),
                    plan_count=plan_count,
                    subtask_steps=subtask_selected,
                )
            else:
                # 如果没有 subtask weights，只画 plan
                academic_title = (
                    f"Plan-level Agent Selection Weight Distribution\n"
                    f"(N_plans={plan_count}, failed={plan_parse_fail})"
                )
                radar_plot(
                    title=academic_title,
                    agents=agents_order,
                    values=plan_values,
                    outpath=os.path.join(result_dir, "radar_plan_level_weight_overall.png"),
                )
        
        # 生成其他图表（保持原有功能）
        if plan_weight_data:
            plan_count = plan_weight_data.get("plan_count", 0)
            plan_parse_fail = plan_weight_data.get("plan_parse_fail", 0)
            weights = plan_weight_data.get("weights", {})
            pick_counts = plan_weight_data.get("pick_counts", {})
            
            v1_weight = [weights.get(k, 0.0) for k in agents_order]
            v1_count = [pick_counts.get(k, 0) for k in agents_order]
            
            academic_title = (
                f"Plan-level Agent Selection Weight Distribution\n"
                f"(N_plans={plan_count}, failed={plan_parse_fail})"
            )
            radar_plot(
                title=academic_title,
                agents=agents_order,
                values=v1_weight,
                outpath=os.path.join(result_dir, "radar_plan_level_weight_overall_old.png"),
            )
            
            academic_title = (
                f"Plan-level Agent Selection Frequency\n"
                f"(N_plans={plan_count}, failed={plan_parse_fail})"
            )
            radar_plot(
                title=academic_title,
                agents=agents_order,
                values=[float(c) for c in v1_count],
                outpath=os.path.join(result_dir, "radar_plan_level_count_overall.png"),
            )
            
            # 生成 Weight vs Pick Count 对比图（Plan-level）
            comparison_title = "Plan-level: Weight vs Pick Count Comparison"
            radar_plot_weight_vs_count(
                title=comparison_title,
                agents=agents_order,
                weight_values=v1_weight,
                count_values=[pick_counts.get(k, 0) for k in agents_order],
                outpath=os.path.join(result_dir, "radar_plan_level_weight_vs_count.png"),
                level_name="plan",
            )
        
        if subtask_weight_data:
            weights = subtask_weight_data.get("weights", {})
            pick_counts = subtask_weight_data.get("pick_counts", {})
            selected_steps = subtask_weight_data.get("subtask_selected_steps", 0)
            total_steps = subtask_weight_data.get("subtask_total_steps", 0)
            
            v2_weight = [weights.get(k, 0.0) for k in agents_order]
            v2_count = [pick_counts.get(k, 0) for k in agents_order]
            
            if weights:
                academic_title = (
                    f"Subtask-level Agent Selection Weight Distribution\n"
                    f"(N_selected={selected_steps}, N_total={total_steps})"
                )
                radar_plot(
                    title=academic_title,
                    agents=agents_order,
                    values=v2_weight,
                    outpath=os.path.join(result_dir, "radar_subtask_level_weight_overall.png"),
                )
            
            academic_title = (
                f"Subtask-level Agent Selection Frequency\n"
                f"(N_selected={selected_steps}, N_total={total_steps})"
            )
            radar_plot(
                title=academic_title,
                agents=agents_order,
                values=[float(c) for c in v2_count],
                outpath=os.path.join(result_dir, "radar_subtask_level_count_overall.png"),
            )
            
            # 生成 Weight vs Pick Count 对比图（Subtask-level，如果有 weights）
            if weights:
                comparison_title = "Subtask-level: Weight vs Pick Count Comparison"
                radar_plot_weight_vs_count(
                    title=comparison_title,
                    agents=agents_order,
                    weight_values=v2_weight,
                    count_values=[pick_counts.get(k, 0) for k in agents_order],
                    outpath=os.path.join(result_dir, "radar_subtask_level_weight_vs_count.png"),
                    level_name="subtask",
                )
        
        # 准备并排图的数据（在循环前初始化）
        cold_start_plan_values = None
        cold_start_subtask_values = None
        
        # 生成分阶段图表
        for phase_name, stats in phase_stats.items():
            phase_plan_weight = [stats["plan_weight_sum"].get(k, 0.0) for k in agents_order]
            phase_plan_count = [stats["plan_pick_count"].get(k, 0) for k in agents_order]
            phase_subtask_weight = [stats["subtask_weight_sum"].get(k, 0.0) for k in agents_order]
            phase_subtask_count = [stats["subtask_pick_count"].get(k, 0) for k in agents_order]
            
            plan_count = stats.get("plan_count", 0)
            subtask_selected = stats.get("subtask_selected_steps", 0)
            subtask_total = stats.get("subtask_total_steps", 0)
            
            phase_display = phase_name.replace("_", " ").title()
            
            # 为 cold_start 阶段生成组合图
            if phase_name == "cold_start":
                # 如果数据都是 0，设置为均匀分布（每个 agent 占比相同）
                plan_sum = sum(phase_plan_weight)
                subtask_sum = sum(phase_subtask_weight)
                
                if plan_sum == 0 and subtask_sum == 0:
                    # 均匀分布：每个 agent 占比 1/n
                    n_agents = len(agents_order)
                    uniform_value = 1.0 / n_agents if n_agents > 0 else 0.0
                    phase_plan_weight = [uniform_value] * n_agents
                    phase_subtask_weight = [uniform_value] * n_agents
                
                # 保存 cold_start 数据用于并排图
                cold_start_plan_values = phase_plan_weight
                cold_start_subtask_values = phase_subtask_weight
                
                # 生成单独的 cold_start 组合图
                if any(v > 0 for v in phase_plan_weight) or any(v > 0 for v in phase_subtask_weight):
                    short_title = "Cold Start: Plan-level vs Subtask-level Weight Distribution"
                    radar_plot_combined(
                        title=short_title,
                        agents=agents_order,
                        plan_values=phase_plan_weight,
                        subtask_values=phase_subtask_weight,
                        outpath=os.path.join(result_dir, "radar_combined_weight_cold_start.png"),
                        plan_count=plan_count,
                        subtask_steps=subtask_selected,
                    )
            
            academic_title = (
                f"{phase_display} Phase: Plan-level Agent Selection Weight Distribution\n"
                f"(N_plans={plan_count})"
            )
            radar_plot(
                title=academic_title,
                agents=agents_order,
                values=phase_plan_weight,
                outpath=os.path.join(result_dir, f"radar_plan_level_weight_{phase_name}.png"),
            )
            
            academic_title = (
                f"{phase_display} Phase: Plan-level Agent Selection Frequency\n"
                f"(N_plans={plan_count})"
            )
            radar_plot(
                title=academic_title,
                agents=agents_order,
                values=[float(c) for c in phase_plan_count],
                outpath=os.path.join(result_dir, f"radar_plan_level_count_{phase_name}.png"),
            )
            
            academic_title = (
                f"{phase_display} Phase: Subtask-level Agent Selection Weight Distribution\n"
                f"(N_selected={subtask_selected}, N_total={subtask_total})"
            )
            radar_plot(
                title=academic_title,
                agents=agents_order,
                values=phase_subtask_weight,
                outpath=os.path.join(result_dir, f"radar_subtask_level_weight_{phase_name}.png"),
            )
            
            academic_title = (
                f"{phase_display} Phase: Subtask-level Agent Selection Frequency\n"
                f"(N_selected={subtask_selected}, N_total={subtask_total})"
            )
            radar_plot(
                title=academic_title,
                agents=agents_order,
                values=[float(c) for c in phase_subtask_count],
                outpath=os.path.join(result_dir, f"radar_subtask_level_count_{phase_name}.png"),
            )
        
        # 生成并排图（cold_start 和 overall）
        if (cold_start_plan_values is not None and cold_start_subtask_values is not None and
            overall_plan_values is not None and overall_subtask_values is not None):
            _path = os.path.normpath(result_dir)
            _bn = os.path.basename(_path)
            _parent = os.path.basename(os.path.dirname(_path))
            _full = _path.lower()
            if "_all_" in _bn or "_all_" in _full:
                note = "Mixed Dataset"
            elif "gsm8k" in _bn or "gsm8k" in _full:
                note = "gsm8k"
            elif "medicalqa" in _full or "medical_qa" in _full or "medicalqa" in _parent.lower():
                note = "Medical Qa"
            else:
                note = None
            radar_plot_combined_side_by_side(
                title_left="Cold Start",
                title_right="Overall",
                agents=agents_order,
                cold_start_plan_values=cold_start_plan_values,
                cold_start_subtask_values=cold_start_subtask_values,
                overall_plan_values=overall_plan_values,
                overall_subtask_values=overall_subtask_values,
                outpath=os.path.join(result_dir, "radar_combined_cold_start_vs_overall.png"),
                note=note,
            )
        
        print(f"✅ 已生成图表到: {result_dir}")
    except Exception as e:
        print(f"错误: 处理目录 {result_dir} 时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python plot_from_json.py <result_dir1> [result_dir2] ...")
        print("示例: python plot_from_json.py select_only_results/bbh_c200_p300_t100/2026-01-24_02-13-04")
        sys.exit(1)
    
    for result_dir in sys.argv[1:]:
        print(f"\n处理目录: {result_dir}")
        plot_from_dir(result_dir)
