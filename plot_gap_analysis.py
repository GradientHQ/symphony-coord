#!/usr/bin/env python3
"""
Generalization Gap and Robustness Analysis：Train vs Test 泛化差距图。

用法:
  python plot_gap_analysis.py           # 仅生成 PDF/PNG
  python plot_gap_analysis.py --show   # 生成后并弹出窗口
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. 准备数据
labels = ['(3, 15)', '(3, 20)', '(5, 25)', '(5, 30)', '(7, 35)', '(9, 40)']
train_acc = np.array([77.00, 76.00, 76.67, 77.67, 75.67, 76.00])
test_acc = np.array([81.00, 80.00, 82.00, 80.00, 77.00, 79.00])
gap = test_acc - train_acc  # 泛化差距
x = np.arange(len(labels))

# 2. 设置绘图风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

# 3. 绘制连接线 (Vertical Gaps) — 泛化差距（加粗虚线）
ax.vlines(x, train_acc, test_acc, color='gray', linestyle='--', alpha=0.5, linewidth=2.5)

# 4. 绘制点
ax.scatter(x, train_acc, color='#1f77b4', s=100, label='Train Accuracy',
           marker='o', edgecolors='white', zorder=3)
ax.scatter(x, test_acc, color='#2ca02c', s=120, label='Test Accuracy',
           marker='D', edgecolors='white', zorder=3)

# 5. Gap trend：右轴显示 Test−Train 随参数变化（加粗线条）
ax2 = ax.twinx()
ax2.spines['top'].set_visible(False)
ax2.plot(x, gap, color='#d62728', linestyle='-', linewidth=3.0, marker='s', markersize=8,
         label='Gap trend', zorder=0, alpha=0.75, markerfacecolor='#d62728', 
         markeredgecolor='white', markeredgewidth=1.5)
ax2.set_ylabel('Gap (Test − Train) %', fontsize=15, color='black', fontweight='bold')  # 加粗并改为黑色
ax2.tick_params(axis='y', labelcolor='black', labelsize=12)  # 右轴刻度改为黑色
ax2.set_ylim(0, max(gap) * 1.15)
ax2.set_yticks(np.linspace(0, max(gap), 5))
ax2.grid(visible=False)

# 6. 添加数值标注（水平/垂直方向0.08，斜着的0.03）
for i in range(len(x)):
    # Train accuracy 标注（垂直下方，距离0.22）
    ax.text(x[i], train_acc[i] - 0.22, f'{train_acc[i]:.1f}', ha='center', va='top',
            fontsize=12, color='#1f77b4', fontweight='bold', zorder=5)
    
    # Test accuracy 标注（水平方向，距离0.08）
    if i == 0:  # (3,15) 对应的 test accuracy 81，写在点的右侧
        ax.text(x[i] + 0.08, test_acc[i], f'{test_acc[i]:.1f}', ha='left', va='center',
                fontsize=12, color='#2ca02c', fontweight='bold', zorder=5)
    else:  # 其他点放在左侧
        ax.text(x[i] - 0.08, test_acc[i], f'{test_acc[i]:.1f}', ha='right', va='center',
                fontsize=12, color='#2ca02c', fontweight='bold', zorder=5)
    
    # Gap 数值标注（斜着的对角线方向，距离0.03）
    if i == 5:  # (9,40) gap trend对应的3，写在点的右下方
        ax2.text(x[i] + 0.03, gap[i] - 0.03, f'{gap[i]:.2f}', ha='left', va='top',
                 fontsize=12, color='#d62728', fontweight='bold', zorder=5)
    else:
        gap_val = gap[i]
        if abs(gap_val - 4.00) < 0.01:  # 4.00 放在右下方
            ax2.text(x[i] + 0.03, gap[i] - 0.03, f'{gap[i]:.2f}', ha='left', va='top',
                     fontsize=12, color='#d62728', fontweight='bold', zorder=5)
        elif abs(gap_val - 2.33) < 0.01:  # 2.33 放在右上方
            ax2.text(x[i] + 0.03, gap[i] + 0.03, f'{gap[i]:.2f}', ha='left', va='bottom',
                     fontsize=12, color='#d62728', fontweight='bold', zorder=5)
        elif abs(gap_val - 1.33) < 0.01 or abs(gap_val - 1.22) < 0.01:  # 1.33/1.22 放在右下方
            ax2.text(x[i] + 0.03, gap[i] - 0.03, f'{gap[i]:.2f}', ha='left', va='top',
                     fontsize=12, color='#d62728', fontweight='bold', zorder=5)
        else:  # 其他值默认放在右上方
            ax2.text(x[i] + 0.03, gap[i] + 0.03, f'{gap[i]:.2f}', ha='left', va='bottom',
                     fontsize=12, color='#d62728', fontweight='bold', zorder=5)

# 7. 坐标轴与细节美化
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=15)
ax.set_ylabel('Accuracy (%)', fontsize=15, fontweight='bold')  # 加粗
ax.set_xlabel('Parameter Settings (Top-L, Pool)', fontsize=15, labelpad=10, fontweight='bold')  # 加粗
ax.set_title('Generalization Gap and Robustness Analysis', fontsize=19, pad=20, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.3)
ax.set_ylim(74, 84)
# 增加x轴边距，让(3,15)和左轴、(9,40)和右轴距离更大
ax.set_xlim(-0.3, len(x) - 1 + 0.3)
# 左轴刻度字体大小设为12，颜色为黑色
ax.tick_params(axis='y', labelsize=12, labelcolor='black')
ax.tick_params(axis='x', labelsize=15, labelcolor='black')  # x轴刻度也设为黑色

# 合并图例：左轴 + Gap trend（增大字体，向上移动0.02）
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
legend = ax.legend(lines1 + lines2, labels1 + labels2, frameon=True, facecolor='white',
          framealpha=0.8, loc='upper left', fontsize=15, prop={'family': 'serif', 'size': 15},
          bbox_to_anchor=(0, 1.02))  # 向上移动0.02（y坐标从1.0增加到1.02）

plt.tight_layout()

# 保存
for name in ['ICML_Gap_Analysis_Plot.pdf', 'ICML_Gap_Analysis_Plot.png']:
    try:
        kw = dict(bbox_inches='tight', facecolor='white')
        if name.endswith('.png'):
            kw['dpi'] = 150
        fig.savefig(name, **kw)
        print(f"图表已成功保存为 {name}")
    except Exception as e:
        print(f"保存 {name} 时发生错误: {e}")

if '--show' in sys.argv:
    plt.show()
else:
    plt.close(fig)
