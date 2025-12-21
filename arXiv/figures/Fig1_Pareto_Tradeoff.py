import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon, Rectangle

# 设置学术出版级绘图风格
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'mathtext.fontset': 'stix',
    'font.size': 14,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'savefig.dpi': 300,
    'figure.autolayout': True,
})

# 定义核心配色
COLOR_OURS = '#D62728'  # 砖红色 - PhyCL-Net
COLOR_BASELINE = '#1F77B4' # 蓝色 - MSPA-FAA-PDK
COLOR_OTHERS = '#7F7F7F' # 灰色 - 其他竞品

def plot_fig1_pareto():
    # 数据录入 (来自 1.md)
    data = {
        'Model': ['PhyCL-Net (Ours)', 'MSPA-FAA-PDK', 'InceptionTime', 'TCN', 'Transformer', 'ResNet-Tiny', 'LSTM'],
        'Params_M': [1.049, 1.657, 0.041, 0.101, 0.200, 0.014, 0.532],
        'Accuracy': [98.20, 98.04, 97.91, 97.13, 95.48, 95.13, 95.02],
        'Latency_ms': [125.99, 184.31, 140.0, 160.0, 200.0, 110.5, 180.0],
        'Type': ['Ours', 'Base', 'Other', 'Other', 'Other', 'Other', 'Other']
    }
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 绘制散点
    for idx, row in df.iterrows():
        color = COLOR_OURS if row['Type'] == 'Ours' else (COLOR_BASELINE if row['Type'] == 'Base' else COLOR_OTHERS)
        marker = '*' if row['Type'] == 'Ours' else 'o'
        size = row['Latency_ms'] * 3
        
        ax.scatter(row['Params_M'], row['Accuracy'], s=size, c=color, alpha=0.8, edgecolors='k', marker=marker, zorder=10)
        
        y_offset = 0.15
        if row['Model'] == 'PhyCL-Net (Ours)': y_offset = 0.25
        if row['Model'] == 'InceptionTime': y_offset = -0.35
        ax.text(row['Params_M'], row['Accuracy'] + y_offset, row['Model'], fontsize=11, ha='center', fontweight='bold' if row['Type']=='Ours' else 'normal')
    
    ax.set_xscale('log')
    ax.set_xlabel('Model Parameters (Millions, Log Scale)', fontweight='bold')
    ax.set_ylabel('LOSO Accuracy (%)', fontweight='bold')
    ax.set_title('Accuracy-Efficiency Trade-off', fontweight='bold', pad=20)
    
    ax.add_patch(Rectangle((0.8, 98.0), 1.0, 0.5, color=COLOR_OURS, alpha=0.1, linestyle='--', linewidth=1))
    ax.text(1.1, 98.4, 'High Performance\nEdge-Optimized Zone', color=COLOR_OURS, fontsize=10, ha='left')
    
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_ylim(94.5, 98.8)
    
    ms = [100, 200]
    l1 = ax.scatter([],[], s=ms[0]*3, c='gray', alpha=0.5, label='Low Latency')
    l2 = ax.scatter([],[], s=ms[1]*3, c='gray', alpha=0.5, label='High Latency')
    ax.legend(handles=[l1, l2], title="Bubble Size $\propto$ Latency", loc='lower right')
    
    plt.tight_layout()
    plt.savefig('arXiv/figures/fig1_pareto_impact.pdf')
    plt.show()

if __name__ == '__main__':
    plot_fig1_pareto()
