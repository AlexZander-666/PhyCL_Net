"""
Publication-Quality Figure Generation for PhyCL-Net Paper
Target: Nature Electronics / IEEE TPAMI style

=== FOR GOOGLE COLAB ===
Copy and paste this entire script into a Colab cell and run it.
Figures will be saved to /content/figures/ and can be downloaded.
"""

# =============================================================================
# COLAB SETUP: Install fonts and dependencies
# =============================================================================
import subprocess
import sys
import platform

# Install Liberation Serif font (Times New Roman alternative for Colab)
# Only run on Linux (Colab)
if platform.system() == 'Linux':
    subprocess.run(['apt-get', 'update', '-qq'], check=False)
    subprocess.run(['apt-get', 'install', '-qq', '-y', 'fonts-liberation'], check=False)

# Clear matplotlib font cache to recognize new fonts
import matplotlib
import matplotlib.font_manager as fm
# Rebuild font cache (compatible with different matplotlib versions)
if platform.system() == 'Linux':
    try:
        fm._rebuild()
    except AttributeError:
        # For newer matplotlib versions
        fm.fontManager.__init__()
        fm._load_fontmanager(try_read_cache=False)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os

# =============================================================================
# SETUP: Font and Style Configuration
# =============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'DejaVu Serif', 'Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color Palette
HERO_COLOR = '#A50026'       # Deep Burgundy Red (PhyCL-Net)
HERO_ALT = '#D62728'         # Deep Crimson alternative
BASELINE_GREY = '#95A5A6'    # Cool Grey for baselines
SLATE = '#7F8C8D'            # Slate grey
TEAL = '#1ABC9C'             # Teal/Green for efficiency
BLUE_GRADIENT = ['#3498DB', '#5DADE2', '#85C1E9']  # Blue gradient for baselines

# Ensure figures directory exists
os.makedirs('figures', exist_ok=True)

# =============================================================================
# DATA
# =============================================================================
# Data A: Efficiency Frontier
models_efficiency = ['PhyCL-Net (Ours)', 'AMSNetV2', 'InceptionTime', 'TCN',
                     'Transformer', 'ResNet-Tiny', 'LSTM']
params = [1.05, 1.66, 0.04, 0.10, 0.20, 0.014, 0.53]
accuracy = [98.20, 98.04, 97.91, 97.13, 95.48, 95.13, 95.02]

# Data B: Radar Metrics
radar_metrics = ['Accuracy', 'Macro-F1', 'Sensitivity', 'Specificity']
phycl_radar = [98.20, 98.15, 98.07, 98.29]
amsnet_radar = [98.04, 97.98, 97.67, 98.30]
tcn_radar = [97.13, 97.04, 96.43, 97.63]

# Data C: Latency
latency_models = ['AMSNetV2', 'PhyCL-Net (Ours)']
latency_values = [184.31, 125.99]

# Data D: Safety
safety_models = ['PhyCL-Net', 'AMSNetV2', 'InceptionTime']
tpr_fpr1 = [96.32, 96.02, 95.52]


# =============================================================================
# FIGURE 1: Efficiency Frontier (Accuracy vs Parameters) - REDESIGNED
# =============================================================================
def create_figure1():
    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Define label positions manually for each model (in data coordinates)
    # Format: (x_offset_factor, y_offset, ha, va)
    label_config = {
        'AMSNetV2':      (1.15, 0.12, 'left', 'bottom'),
        'InceptionTime': (1.3, 0.08, 'left', 'bottom'),
        'TCN':           (1.3, -0.08, 'left', 'top'),
        'Transformer':   (1.3, 0.08, 'left', 'bottom'),
        'ResNet-Tiny':   (1.5, -0.15, 'left', 'top'),
        'LSTM':          (1.3, -0.08, 'left', 'top'),
    }

    # Add "High Performance Zone" shading (top-left quadrant)
    # Use axvspan + axhspan combination for better control
    ax.axvspan(0.008, 0.12, ymin=(97.8-96.5)/(98.5-96.5), ymax=1,
               color='#E8F8F5', alpha=0.6, zorder=0)
    ax.text(0.015, 98.35, 'High Efficiency\nZone', fontsize=9,
            color='#1E8449', alpha=0.8, style='italic', fontweight='medium',
            ha='left', va='top')

    # Plot baselines as grey circles with smart labels
    for model, p, acc in zip(models_efficiency, params, accuracy):
        if 'Ours' not in model:
            # Plot point
            ax.scatter(p, acc, s=150, c=BASELINE_GREY, alpha=0.75,
                      edgecolors='white', linewidths=2, zorder=5)

            # Get label config
            cfg = label_config.get(model, (1.3, 0.1, 'left', 'bottom'))
            x_off, y_off, ha, va = cfg

            # Calculate text position (multiplicative for log scale x)
            text_x = p * x_off
            text_y = acc + y_off

            # Draw label with subtle connecting line
            ax.annotate(model, xy=(p, acc), xytext=(text_x, text_y),
                       fontsize=9, color=SLATE, ha=ha, va=va,
                       arrowprops=dict(arrowstyle='-', color='#BDC3C7',
                                      lw=0.8, shrinkA=4, shrinkB=2))

    # Plot PhyCL-Net as prominent red star (HERO)
    phycl_idx = 0
    ax.scatter(params[phycl_idx], accuracy[phycl_idx], s=500, c=HERO_COLOR,
              marker='*', edgecolors='white', linewidths=1.5, zorder=10,
              label='PhyCL-Net (Ours)')

    # Add PhyCL-Net label with elegant curved arrow
    ax.annotate('PhyCL-Net (Ours)\nOptimal Trade-off',
                xy=(params[phycl_idx], accuracy[phycl_idx]),
                xytext=(0.25, 97.3),
                fontsize=11, fontweight='bold', color=HERO_COLOR,
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor=HERO_COLOR, alpha=0.9, linewidth=1.5),
                arrowprops=dict(arrowstyle='fancy,head_length=0.6,head_width=0.4',
                               color=HERO_COLOR, lw=1.5,
                               connectionstyle='arc3,rad=0.3',
                               shrinkA=0, shrinkB=8))

    # Add Pareto frontier hint (dashed curve connecting top performers)
    # Sort by params for frontier - only show models within visible range
    frontier_models = [(p, a) for p, a, m in zip(params, accuracy, models_efficiency)
                       if a > 96.8]  # Only high performers visible in narrowed range
    frontier_models.sort(key=lambda x: x[0])
    if len(frontier_models) > 2:
        fp, fa = zip(*frontier_models)
        ax.plot(fp, fa, '--', color='#E74C3C', alpha=0.3, lw=2, zorder=1)

    # Axis configuration
    # NARROWED Y-AXIS: 96.5-98.5 to emphasize PhyCL-Net's lead
    ax.set_xscale('log')
    ax.set_xlim(0.008, 4)
    ax.set_ylim(96.5, 98.5)
    ax.set_xlabel('Parameters (M)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')

    # Custom x-ticks for better readability
    ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2])
    ax.set_xticklabels(['0.01', '0.02', '0.05', '0.1', '0.2', '0.5', '1', '2'])

    # Custom y-ticks for narrowed range
    ax.set_yticks([96.5, 97.0, 97.5, 98.0, 98.5])
    ax.set_yticklabels(['96.5', '97.0', '97.5', '98.0', '98.5'])

    # Grid - subtle
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig('figures/fig1_accuracy_vs_params.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()
    print("✓ Figure 1 saved: figures/fig1_accuracy_vs_params.pdf")


# =============================================================================
# FIGURE 2: Diamond Radar Chart
# =============================================================================
def create_figure2():
    # Number of metrics
    num_vars = len(radar_metrics)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Close the radar plots
    phycl_values = phycl_radar + phycl_radar[:1]
    amsnet_values = amsnet_radar + amsnet_radar[:1]
    tcn_values = tcn_radar + tcn_radar[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    # Set the range (CRITICAL: 94-99, not starting at 0)
    ax.set_ylim(94, 99)

    # Plot baselines first (grey dashed)
    ax.plot(angles, tcn_values, 'o--', linewidth=1.5, color=BASELINE_GREY,
            alpha=0.7, label='TCN', markersize=5)
    ax.plot(angles, amsnet_values, 's--', linewidth=1.5, color=SLATE,
            alpha=0.7, label='AMSNetV2', markersize=5)

    # Plot PhyCL-Net (hero - thick red line with fill)
    ax.plot(angles, phycl_values, 'o-', linewidth=3, color=HERO_COLOR,
            label='PhyCL-Net (Ours)', markersize=8)
    ax.fill(angles, phycl_values, color=HERO_COLOR, alpha=0.1)

    # Set metric labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics, fontsize=12, fontweight='bold')

    # Y-axis ticks
    ax.set_yticks([95, 96, 97, 98, 99])
    ax.set_yticklabels(['95', '96', '97', '98', '99'], fontsize=10, color='grey')

    # Grid styling
    ax.yaxis.grid(True, linestyle=':', alpha=0.4)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3)

    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05), frameon=True,
              fancybox=True, shadow=False, framealpha=0.9)

    plt.tight_layout()
    plt.savefig('figures/fig2_radar_comparison.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()
    print("✓ Figure 2 saved: figures/fig2_radar_comparison.pdf")


# =============================================================================
# FIGURE 3: Latency Reduction (Horizontal Bar)
# =============================================================================
def create_figure3():
    fig, ax = plt.subplots(figsize=(8, 4))

    y_pos = np.arange(len(latency_models))
    colors = [BASELINE_GREY, TEAL]  # Grey for AMSNetV2, Teal for PhyCL-Net

    # Create horizontal bars
    bars = ax.barh(y_pos, latency_values, height=0.5, color=colors,
                   edgecolor='white', linewidth=2)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, latency_values)):
        ax.text(val + 3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f} ms', va='center', ha='left',
                fontsize=12, fontweight='bold',
                color=colors[i] if i == 1 else SLATE)

    # Speedup annotation
    speedup = (latency_values[0] - latency_values[1]) / latency_values[0] * 100
    mid_y = 0.5
    ax.annotate(f'-{speedup:.1f}% Speedup',
                xy=(latency_values[1], 1),
                xytext=(latency_values[1] + 20, mid_y),
                fontsize=14, fontweight='bold', color=TEAL,
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=TEAL, linewidth=2))

    # Draw comparison arrow
    ax.annotate('', xy=(latency_values[1], 0.15), xytext=(latency_values[0], 0.85),
                arrowprops=dict(arrowstyle='<->', color='#2C3E50', lw=1.5,
                               connectionstyle='arc3,rad=0'))

    # Axis configuration
    ax.set_yticks(y_pos)
    ax.set_yticklabels(latency_models, fontsize=12, fontweight='bold')
    ax.set_xlabel('Inference Latency (ms)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 220)

    # Grid
    ax.xaxis.grid(True, alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('figures/fig3_latency_reduction.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()
    print("✓ Figure 3 saved: figures/fig3_latency_reduction.pdf")


# =============================================================================
# FIGURE 4: Safety Criticality (TPR@FPR=1%)
# =============================================================================
def create_figure4():
    fig, ax = plt.subplots(figsize=(7, 5))

    x_pos = np.arange(len(safety_models))

    # Colors: Red for PhyCL-Net, gradient blue for baselines
    colors = [HERO_COLOR, BLUE_GRADIENT[0], BLUE_GRADIENT[1]]

    # Create vertical bars
    bars = ax.bar(x_pos, tpr_fpr1, width=0.6, color=colors,
                  edgecolor='white', linewidth=2)

    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, tpr_fpr1)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.08,
                f'{val:.2f}%', ha='center', va='bottom',
                fontsize=12, fontweight='bold',
                color=colors[i])

    # Draw horizontal dashed line at second-best (ceiling line)
    second_best = tpr_fpr1[1]  # AMSNetV2
    ax.axhline(y=second_best, color=SLATE, linestyle='--', linewidth=2,
               alpha=0.7, zorder=0)
    ax.text(2.35, second_best + 0.05, 'Previous Best', fontsize=10,
            color=SLATE, style='italic', va='bottom')

    # Highlight PhyCL-Net breaking the ceiling
    ax.annotate('', xy=(0, tpr_fpr1[0]), xytext=(0, second_best),
                arrowprops=dict(arrowstyle='->', color=HERO_COLOR, lw=2))

    # Axis configuration
    ax.set_xticks(x_pos)
    ax.set_xticklabels(safety_models, fontsize=12, fontweight='bold')
    ax.set_ylabel('TPR @ FPR=1% (%)', fontsize=14, fontweight='bold')
    ax.set_ylim(94, 97)

    # Grid
    ax.yaxis.grid(True, alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('figures/fig4_safety_metrics.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()
    print("✓ Figure 4 saved: figures/fig4_safety_metrics.pdf")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Generating Publication-Quality Figures for PhyCL-Net")
    print("Target Style: Nature Electronics / IEEE TPAMI")
    print("=" * 60)

    # Generate all figures
    create_figure1()
    create_figure2()
    create_figure3()
    create_figure4()

    print("=" * 60)
    print("✓ All figures generated successfully!")
    print("Output directory: figures/")
    print("Format: PDF (300 DPI)")
    print("=" * 60)

    # Download helper for Colab
    print("\n📥 To download all PDFs in Colab, run:")
    print("from google.colab import files")
    print("import glob")
    print("for f in glob.glob('figures/*.pdf'): files.download(f)")
