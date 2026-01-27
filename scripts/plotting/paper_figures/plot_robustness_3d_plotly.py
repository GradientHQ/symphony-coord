#!/usr/bin/env python3
"""
Plotly 3D 响应曲面：Robustness Sensitivity Analysis，带顶部数值与 (Top-L, Pool) 坐标标注。
（Plotly 无 Bar3d，用 Surface + 散点模拟；样式按 ICML 柱状图风格。）

用法:
  python plot_robustness_3d_plotly.py           # 仅生成 PDF/PNG
  python plot_robustness_3d_plotly.py --show   # 生成后并弹出交互式窗口
"""
import sys
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

# 数据
l_vals = np.array([3, 3, 5, 5, 7, 9])
pool_vals = np.array([15, 20, 25, 30, 35, 40])
test_acc = np.array([81.00, 80.00, 82.00, 80.00, 77.00, 79.00])

# 插值得到平滑曲面
l_grid = np.linspace(l_vals.min(), l_vals.max(), 50)
pool_grid = np.linspace(pool_vals.min(), pool_vals.max(), 50)
L, P = np.meshgrid(l_grid, pool_grid)
Z = griddata((l_vals, pool_vals), test_acc, (L, P), method='cubic')

fig = go.Figure()

# 平滑曲面（移除底部投影等高线，保持简洁）
fig.add_trace(go.Surface(
    z=Z, x=l_grid, y=pool_grid,
    colorscale='Viridis',
    showscale=True,
    colorbar=dict(title='Accuracy (%)', thickness=20),
    opacity=0.9,
    # 完全移除等高线投影，保持底部平面干净
))

# 原始实验点（红点）
fig.add_trace(go.Scatter3d(
    x=l_vals, y=pool_vals, z=test_acc,
    mode='markers',
    marker=dict(size=8, color='red', opacity=0.95, symbol='circle', line=dict(color='white', width=1)),
    name='Data',
    hovertemplate='(Top-L, Pool) = (%{x:.0f}, %{y:.0f})<br>Test Acc = %{z:.2f}%<extra></extra>',
))

# 顶部数值标注 (如 81%)
for l, p, a in zip(l_vals, pool_vals, test_acc):
    fig.add_trace(go.Scatter3d(
        x=[l], y=[p], z=[a + 0.35],
        text=[f'{a}%'],
        mode='text',
        textfont=dict(color='black', size=12, family='Times New Roman'),
        showlegend=False,
    ))

# 底部坐标标注 (Top-L, Pool)
for l, p in zip(l_vals, pool_vals):
    fig.add_trace(go.Scatter3d(
        x=[l], y=[p], z=[74.2],
        text=[f'({int(l)}, {int(p)})'],
        mode='text',
        textfont=dict(color='#333', size=10, family='Times New Roman'),
        showlegend=False,
    ))

# ICML 风格（与 Bar3d 版布局一致）
fig.update_layout(
    title=dict(
        text='<b>Robustness Sensitivity Analysis</b>',
        font=dict(family='Times New Roman', size=22),
    ),
    scene=dict(
        xaxis=dict(title='Top-L', tickvals=[3, 5, 7, 9], gridcolor='lightgray'),
        yaxis=dict(title='Pool Size', tickvals=[15, 20, 25, 30, 35, 40], gridcolor='lightgray'),
        zaxis=dict(title='Accuracy (%)', range=[74, 83], dtick=2, gridcolor='lightgray'),
        xaxis_backgroundcolor='rgba(0,0,0,0)',
        yaxis_backgroundcolor='rgba(0,0,0,0)',
        zaxis_backgroundcolor='rgba(240,240,240,0.5)',
        camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2)),
    ),
    margin=dict(l=0, r=0, b=0, t=50),
    font=dict(family='Times New Roman', size=14),
    template='plotly_white',
    width=900,
    height=800,
)

# 保存
for name in ['ICML_3D_Surface_Final.pdf', 'ICML_3D_Surface_Final.png']:
    try:
        fig.write_image(name, scale=2)
        print(f"图表已成功保存为 {name}")
    except ImportError:
        print("请安装 kaleido: pip install -U kaleido")
        break
    except Exception as e:
        print(f"保存 {name} 时发生错误: {e}")

if '--show' in sys.argv:
    fig.show()
