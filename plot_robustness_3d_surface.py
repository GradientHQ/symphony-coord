#!/usr/bin/env python3
"""
3D 响应曲面图：Robustness Response Surface Analysis
(Top-L, Pool Size) -> Test Accuracy (%)

用法:
  python plot_robustness_3d_surface.py           # 仅生成 PDF/PNG
  python plot_robustness_3d_surface.py --show   # 生成后并弹出交互式窗口
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata

# 1. 原始数据 (Top-L, Pool, Test Accuracy)
l_coords = np.array([3, 3, 5, 5, 7, 9])
pool_coords = np.array([15, 20, 25, 30, 35, 40])
test_acc = np.array([81.00, 80.00, 82.00, 80.00, 77.00, 79.00])

# 2. 创建网格用于插值 (让曲面平滑)
l_grid = np.linspace(l_coords.min(), l_coords.max(), 100)
pool_grid = np.linspace(pool_coords.min(), pool_coords.max(), 100)
L, P = np.meshgrid(l_grid, pool_grid)

# 使用三次元插值 (Cubic Interpolation) 生成曲面数据
Z = griddata((l_coords, pool_coords), test_acc, (L, P), method='cubic')

# 3. 开始绘图
fig = plt.figure(figsize=(12, 8), dpi=150)
ax = fig.add_subplot(111, projection='3d')

# 设置学术字体 (如果环境支持 LaTeX 请开启)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# 绘制 3D 表面
# cmap 使用 viridis 或 plasma，alpha 设置透明度以便观察投影
surf = ax.plot_surface(L, P, Z, cmap=cm.viridis, alpha=0.7,
                       linewidth=0, antialiased=True, shade=True)

# 4. 关键：在底部绘制等高线投影 (Contour Plot)
# 这能让审稿人一眼看到哪个参数区间是性能“甜点区”
offset = test_acc.min() - 2
ax.contourf(L, P, Z, zdir='z', offset=offset, cmap=cm.viridis, alpha=0.5)

# 5. 标注原始实验点 (散点)
# 这样能证明你的曲面是由真实数据支撑的
ax.scatter(l_coords, pool_coords, test_acc, color='red', s=40,
           label='Experimental Data', edgecolors='white')

# 6. 坐标轴美化
ax.set_xlabel('Top-L Setting', fontsize=12, labelpad=10)
ax.set_ylabel('Pool Size', fontsize=12, labelpad=10)
ax.set_zlabel('Test Accuracy (%)', fontsize=12, labelpad=10)
ax.set_title('Robustness Response Surface Analysis', fontsize=15, fontweight='bold', pad=20)

# 调整视角 (视角很重要，决定了图的“高级感”)
ax.view_init(elev=25, azim=-45)

# 添加颜色条
cbar = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label('Accuracy Scale')

# 导出
plt.tight_layout()
for name in ['ICML_Robustness_3D_Surface.pdf', 'ICML_Robustness_3D_Surface.png']:
    try:
        kw = dict(bbox_inches='tight')
        if name.endswith('.png'):
            kw['dpi'] = 150
        plt.savefig(name, **kw)
        print(f"图表已成功保存为 {name}")
    except Exception as e:
        print(f"保存 {name} 时发生错误: {e}")

if '--show' in sys.argv:
    plt.show()
else:
    plt.close(fig)
