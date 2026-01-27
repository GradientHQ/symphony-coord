#!/usr/bin/env python3
"""
Extended Robustness Results under Different (Top-L, Pool) Settings
柱状图：Train / Test Accuracy，数据标注于柱上，学术风格。

用法:
  python plot_robustness_bars.py           # 仅生成 PDF/PNG
  python plot_robustness_bars.py --show   # 生成后并弹出窗口
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 数据（与 parallel coordinates / 3D surface 一致）
settings = [(3, 15), (3, 20), (5, 25), (5, 30), (7, 35), (9, 40)]
train_acc = [77.00, 76.00, 76.67, 77.67, 75.67, 76.00]
test_acc = [81.00, 80.00, 82.00, 80.00, 77.00, 79.00]

x_labels = [f'({l}, {p})' for l, p in settings]
n = len(settings)
x = np.arange(n)
w = 0.36  # 柱宽

fig, ax = plt.subplots(figsize=(10, 6), dpi=150, facecolor='white')
ax.set_facecolor('white')

# 配色：柔和蓝 / 柔和绿，学术风格
c_train = '#5B8FA3'   # slate blue
c_test = '#7BA05B'    # sage green
bars_train = ax.bar(x - w / 2, train_acc, w, label='Train', color=c_train, edgecolor='white', linewidth=1.2)
bars_test = ax.bar(x + w / 2, test_acc, w, label='Test', color=c_test, edgecolor='white', linewidth=1.2)

# 数据标注：写在柱顶
def add_labels(bars, fmt='.2f'):
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:{fmt}}',
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='500',
                    color='#1a1a1a', fontfamily='serif')

add_labels(bars_train)
add_labels(bars_test)

ax.set_ylabel('Accuracy (%)', fontsize=13, fontfamily='serif')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Parameter Settings (Top-L, Pool)', fontsize=13, fontfamily='serif')
ax.set_title('Extended Robustness Results under Different (Top-L, Pool) Settings',
             fontsize=15, fontweight='bold', fontfamily='serif', pad=14)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=11, fontfamily='serif')
ax.set_ylim(72.5, 85)

# 在每组柱下方显式标注坐标 (Top-L, Pool)，避免被裁切
for i, (ll, pp) in enumerate(settings):
    ax.text(x[i], 72.6, f'({ll}, {pp})', ha='center', va='top', fontsize=11,
            fontweight='500', fontfamily='serif', color='#2c3e50')
ax.set_yticks([74, 76, 78, 80, 82, 84])
ax.tick_params(axis='both', labelsize=11)
ax.yaxis.grid(True, linestyle='--', color='#e0e0e0', alpha=0.9)
ax.set_axisbelow(True)
leg = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False,
                framealpha=0.98, edgecolor='#c8d4e0', fontsize=11, prop={'family': 'serif'})
leg.get_frame().set_facecolor('#f8fafc')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.tight_layout()

for name in ['ICML_Robustness_Bars.pdf', 'ICML_Robustness_Bars.png']:
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
