#!/usr/bin/env python3
"""
平行坐标图：Extended Robustness Results under Different (Top-L, Pool) Settings

用法:
  python plot_parallel_coordinates.py           # 仅生成 PDF/PNG
  python plot_parallel_coordinates.py --show   # 生成后并弹出交互式图表
"""
import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

data = {
    'Top-L': [3, 3, 5, 5, 7, 9],
    'Pool Size': [15, 20, 25, 30, 35, 40],
    'Train Accuracy (%)': [77.00, 76.00, 76.67, 77.67, 75.67, 76.00],
    'Test Accuracy (%)': [81.00, 80.00, 82.00, 80.00, 77.00, 79.00]
}
df = pd.DataFrame(data)

fig = go.Figure(data=go.Parcoords(
    line=dict(
        color=df['Test Accuracy (%)'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(
            title=dict(text="Test Accuracy (%)", font=dict(family="Times New Roman, serif", size=14)),
            tickfont=dict(family="Times New Roman, serif", size=12),
            thickness=20,
            len=0.6,
        ),
    ),
    dimensions=list([
        dict(label='Top-L', values=df['Top-L'], tickvals=[3, 5, 7, 9], range=[2.5, 9.5]),
        dict(label='Pool Size', values=df['Pool Size'], tickvals=[15, 20, 25, 30, 35, 40], range=[12, 42]),
        dict(label='Train Accuracy (%)', values=df['Train Accuracy (%)'],
             tickvals=[75.67, 76.0, 76.5, 77.0, 77.5, 77.67], range=[75.5, 77.8], tickformat='.2f'),
        dict(label='Test Accuracy (%)', values=df['Test Accuracy (%)'],
             tickvals=[77, 78, 79, 80, 81, 82], range=[76.5, 82.5], tickformat='.0f'),
    ]),
))

fig.update_layout(
    title=dict(
        text="Extended Robustness Results under Different (Top-L, Pool) Settings",
        font=dict(family="Times New Roman, serif", size=20, color="black"),
        x=0.5, xanchor='center',
        y=0.98,  # 标题位置稍微上移
        pad=dict(t=10, b=30),  # 增加标题下方间距，让轴标签下移
    ),
    font=dict(family="Times New Roman, serif", size=14, color="black"),
    height=500,
    width=1000,
    margin=dict(l=60, r=80, t=120, b=50),  # 大幅增加顶部边距，确保轴标签在标题下方
    template='plotly_white',
)

for name in ['ICML_Parallel_Coordinates_Plot.pdf', 'ICML_Parallel_Coordinates_Plot.png']:
    try:
        fig.write_image(name, scale=3)
        print(f"图表已成功保存为 {name}")
    except ImportError:
        print("请安装 kaleido: pip install -U kaleido")
        break
    except Exception as e:
        print(f"保存 {name} 时发生错误: {e}")

if '--show' in sys.argv:
    fig.show()
