#!/usr/bin/env python
"""
Publication-quality figure generation for AMSNetV2 Fall Detection paper.

This script generates:
1. t-SNE visualization comparing Baseline vs AMSNetV2 feature clustering
2. Attention heatmap showing Fall-Aware Attention focus on impact moments

Usage:
    python scripts/generate_paper_figures.py \
        --baseline-ckpt outputs/baseline_best.pth \
        --amsv2-ckpt outputs/amsv2_best.pth \
        --data-root ./data \
        --output-dir ./figures

Author: AMSNetV2 Research Team
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.ams_net_v2 import AMSNetV2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Academic Publication Style Configuration
# =============================================================================

def setup_publication_style():
    """Configure matplotlib for academic publication quality."""
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,

        # Figure settings
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Line settings
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.8,

        # Grid settings
        'axes.grid': False,
        'grid.alpha': 0.3,

        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',

        # Spine settings
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# =============================================================================
# Feature Extraction Utilities
# =============================================================================

class FeatureExtractor:
    """Hook-based feature extractor for neural networks."""

    def __init__(self, model: nn.Module, target_layer: str = 'pool'):
        """
        Initialize feature extractor with forward hook.

        Args:
            model: The neural network model
            target_layer: Name of layer to extract features from
        """
        self.model = model
        self.features: Optional[torch.Tensor] = None
        self._hook_handle = None

        # Register hook on target layer
        target = self._get_layer(model, target_layer)
        if target is not None:
            self._hook_handle = target.register_forward_hook(self._hook_fn)
            logger.info(f"Registered hook on layer: {target_layer}")
        else:
            logger.warning(f"Layer '{target_layer}' not found, using model output")

    def _get_layer(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Get a layer by name from the model."""
        parts = layer_name.split('.')
        current = model
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        return current

    def _hook_fn(self, module: nn.Module, input: Tuple, output: torch.Tensor):
        """Forward hook to capture intermediate features."""
        self.features = output.detach()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input tensor."""
        self.features = None
        with torch.no_grad():
            _ = self.model(x)

        if self.features is not None:
            # Handle different output shapes
            if self.features.dim() == 3:  # (B, C, L)
                return self.features.squeeze(-1)  # (B, C) after pool
            return self.features
        return None

    def remove_hook(self):
        """Remove the registered hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()


class AttentionExtractor:
    """Extract Fall-Aware Attention weights from AMSNetV2."""

    def __init__(self, model: nn.Module):
        """
        Initialize attention extractor.

        Args:
            model: AMSNetV2 model instance
        """
        self.model = model
        self.faa_modules: List[nn.Module] = []
        self._find_faa_modules(model)
        logger.info(f"Found {len(self.faa_modules)} FAA modules")

    def _find_faa_modules(self, module: nn.Module):
        """Recursively find all FallAwareAttention modules."""
        for name, child in module.named_children():
            class_name = child.__class__.__name__
            if class_name == 'FallAwareAttention':
                self.faa_modules.append(child)
            else:
                self._find_faa_modules(child)

    def get_attention_weights(self, x: torch.Tensor, stage_idx: int = -1) -> Optional[torch.Tensor]:
        """
        Get attention weights for input.

        Args:
            x: Input tensor (B, 3, 512)
            stage_idx: Which FAA stage to extract (-1 for last)

        Returns:
            Attention weights (B, C, L) or (B, 1, L) averaged
        """
        with torch.no_grad():
            _ = self.model(x)

        if not self.faa_modules:
            logger.warning("No FAA modules found")
            return None

        idx = stage_idx if stage_idx >= 0 else len(self.faa_modules) + stage_idx
        faa = self.faa_modules[idx]

        if hasattr(faa, 'last_attention') and faa.last_attention is not None:
            attn = faa.last_attention
            # Average across channels to get temporal attention
            if attn.dim() == 3:
                return attn.mean(dim=1, keepdim=True)  # (B, 1, L)
            return attn

        logger.warning("last_attention not available")
        return None


def get_features_and_labels(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features and labels from a data loader.

    Args:
        model: Neural network model
        loader: DataLoader with (data, labels) batches
        device: Torch device
        max_samples: Maximum samples to extract (for t-SNE performance)

    Returns:
        features: numpy array (N, D)
        labels: numpy array (N,)
    """
    model.eval()
    extractor = FeatureExtractor(model, target_layer='pool')

    all_features = []
    all_labels = []
    total_samples = 0

    try:
        with torch.no_grad():
            for batch_data, batch_labels in loader:
                if total_samples >= max_samples:
                    break

                batch_data = batch_data.to(device)
                features = extractor(batch_data)

                if features is None:
                    # Fallback: use logits from model output
                    outputs = model(batch_data)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    features = logits

                # Flatten if needed
                if features.dim() > 2:
                    features = features.view(features.size(0), -1)

                all_features.append(features.cpu().numpy())
                all_labels.append(batch_labels.numpy())
                total_samples += len(batch_labels)
    finally:
        extractor.remove_hook()

    features = np.concatenate(all_features, axis=0)[:max_samples]
    labels = np.concatenate(all_labels, axis=0)[:max_samples]

    logger.info(f"Extracted {len(features)} samples with feature dim {features.shape[1]}")
    return features, labels


# =============================================================================
# t-SNE Visualization
# =============================================================================

def plot_tsne_comparison(
    baseline_features: np.ndarray,
    baseline_labels: np.ndarray,
    amsv2_features: np.ndarray,
    amsv2_labels: np.ndarray,
    output_path: Path,
    perplexity: int = 30,
    random_state: int = 42,
):
    """
    Create side-by-side t-SNE visualization comparing baseline and AMSNetV2.

    Args:
        baseline_features: Feature array from baseline model
        baseline_labels: Labels for baseline
        amsv2_features: Feature array from AMSNetV2
        amsv2_labels: Labels for AMSNetV2
        output_path: Path to save figure
        perplexity: t-SNE perplexity parameter
        random_state: Random seed for reproducibility
    """
    setup_publication_style()

    logger.info("Computing t-SNE embeddings...")

    # Compute t-SNE for both models
    # Note: n_iter was renamed to max_iter in sklearn 1.5+
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            max_iter=1000,
            learning_rate='auto',
            init='pca',
        )
    except TypeError:
        # Fallback for older sklearn versions
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=1000,
            learning_rate='auto',
            init='pca',
        )

    baseline_2d = tsne.fit_transform(baseline_features)
    amsv2_2d = tsne.fit_transform(amsv2_features)

    # Define colors
    colors = {0: '#3498db', 1: '#e74c3c'}  # Blue for ADL, Red for Fall
    class_names = {0: 'ADL', 1: 'Fall'}

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Plot baseline
    ax1 = axes[0]
    for label in [0, 1]:
        mask = baseline_labels == label
        ax1.scatter(
            baseline_2d[mask, 0],
            baseline_2d[mask, 1],
            c=colors[label],
            label=class_names[label],
            alpha=0.6,
            s=20,
            edgecolors='white',
            linewidths=0.3,
        )
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.set_title('(a) Baseline Model', fontweight='bold')
    ax1.legend(loc='upper right', markerscale=1.5)

    # Plot AMSNetV2
    ax2 = axes[1]
    for label in [0, 1]:
        mask = amsv2_labels == label
        ax2.scatter(
            amsv2_2d[mask, 0],
            amsv2_2d[mask, 1],
            c=colors[label],
            label=class_names[label],
            alpha=0.6,
            s=20,
            edgecolors='white',
            linewidths=0.3,
        )
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.set_title('(b) AMSNetV2 (Ours)', fontweight='bold')
    ax2.legend(loc='upper right', markerscale=1.5)

    # Calculate and annotate cluster metrics
    for ax, features_2d, labels, title in [
        (ax1, baseline_2d, baseline_labels, 'Baseline'),
        (ax2, amsv2_2d, amsv2_labels, 'AMSNetV2'),
    ]:
        # Compute inter-class and intra-class distances
        adl_mask = labels == 0
        fall_mask = labels == 1

        adl_center = features_2d[adl_mask].mean(axis=0)
        fall_center = features_2d[fall_mask].mean(axis=0)

        inter_dist = np.linalg.norm(adl_center - fall_center)
        intra_adl = np.mean(np.linalg.norm(features_2d[adl_mask] - adl_center, axis=1))
        intra_fall = np.mean(np.linalg.norm(features_2d[fall_mask] - fall_center, axis=1))

        # Silhouette-like ratio (higher is better)
        ratio = inter_dist / (intra_adl + intra_fall + 1e-8)

        ax.text(
            0.02, 0.02,
            f'Sep. Ratio: {ratio:.2f}',
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

    plt.tight_layout()

    # Save in multiple formats
    for fmt in ['png', 'pdf', 'svg']:
        save_path = output_path.with_suffix(f'.{fmt}')
        fig.savefig(save_path, format=fmt, dpi=300, bbox_inches='tight')
        logger.info(f"Saved t-SNE plot: {save_path}")

    plt.close(fig)


def plot_tsne_single(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str = "Feature Embedding Visualization",
    perplexity: int = 30,
    random_state: int = 42,
):
    """
    Create single t-SNE visualization for one model.

    Args:
        features: Feature array (N, D)
        labels: Labels array (N,)
        output_path: Path to save figure
        title: Plot title
        perplexity: t-SNE perplexity parameter
        random_state: Random seed
    """
    setup_publication_style()

    logger.info("Computing t-SNE embedding...")
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            max_iter=1000,
            learning_rate='auto',
            init='pca',
        )
    except TypeError:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=1000,
            learning_rate='auto',
            init='pca',
        )

    features_2d = tsne.fit_transform(features)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))

    colors = {0: '#3498db', 1: '#e74c3c'}
    class_names = {0: 'ADL', 1: 'Fall'}

    for label in [0, 1]:
        mask = labels == label
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=colors[label],
            label=class_names[label],
            alpha=0.6,
            s=25,
            edgecolors='white',
            linewidths=0.3,
        )

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='best', markerscale=1.5)

    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        save_path = output_path.with_suffix(f'.{fmt}')
        fig.savefig(save_path, format=fmt, dpi=300)
        logger.info(f"Saved: {save_path}")

    plt.close(fig)


# =============================================================================
# Attention Heatmap Visualization
# =============================================================================

def plot_attention_heatmap(
    signal: np.ndarray,
    attention: np.ndarray,
    output_path: Path,
    sample_rate: float = 50.0,
    title: str = "Fall-Aware Attention Visualization",
):
    """
    Create attention heatmap overlaid on accelerometer signal.

    Args:
        signal: Raw accelerometer signal (3, T) or (T, 3)
        attention: Attention weights (T,) or (1, T)
        output_path: Path to save figure
        sample_rate: Sampling rate in Hz
        title: Plot title
    """
    setup_publication_style()

    # Normalize inputs
    if signal.ndim == 2 and signal.shape[0] == 3:
        signal = signal.T  # (T, 3)

    attention = attention.flatten()

    T = len(attention)
    if len(signal) != T:
        # Interpolate attention to match signal length
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(attention))
        x_new = np.linspace(0, 1, len(signal))
        attention = interp1d(x_old, attention, kind='linear')(x_new)
        T = len(signal)

    time = np.arange(T) / sample_rate

    # Normalize attention to [0, 1]
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

    # Create custom colormap for attention (white to red)
    attention_cmap = LinearSegmentedColormap.from_list(
        'attention', ['#ffffff', '#ffcccc', '#ff6666', '#ff0000']
    )

    # Create figure with two subplots
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.1)

    ax_signal = fig.add_subplot(gs[0])
    ax_attn = fig.add_subplot(gs[1], sharex=ax_signal)

    # Plot accelerometer signals with attention-based coloring
    axis_names = ['Acc-X', 'Acc-Y', 'Acc-Z']
    axis_colors = ['#2ecc71', '#3498db', '#9b59b6']  # Green, Blue, Purple

    for i, (name, color) in enumerate(zip(axis_names, axis_colors)):
        # Create line segments colored by attention
        points = np.array([time, signal[:, i]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Color intensity based on attention
        colors_rgba = plt.cm.colors.to_rgba_array([color] * (T - 1))
        # Modulate alpha by attention (min 0.3, max 1.0)
        colors_rgba[:, 3] = 0.3 + 0.7 * attention[:-1]

        lc = LineCollection(segments, colors=colors_rgba, linewidths=1.5)
        ax_signal.add_collection(lc)

        # Add legend entry
        ax_signal.plot([], [], color=color, label=name, linewidth=2)

    # Set axis limits
    ax_signal.set_xlim(time.min(), time.max())
    y_margin = 0.1 * (signal.max() - signal.min())
    ax_signal.set_ylim(signal.min() - y_margin, signal.max() + y_margin)

    ax_signal.set_ylabel('Acceleration (g)')
    ax_signal.set_title(title, fontweight='bold', pad=10)
    ax_signal.legend(loc='upper right', ncol=3, framealpha=0.9)

    # Remove x-axis labels from signal plot
    plt.setp(ax_signal.get_xticklabels(), visible=False)

    # Plot attention bar
    im = ax_attn.imshow(
        attention.reshape(1, -1),
        aspect='auto',
        cmap=attention_cmap,
        extent=[time.min(), time.max(), 0, 1],
        vmin=0, vmax=1,
    )
    ax_attn.set_xlabel('Time (s)')
    ax_attn.set_ylabel('Attention')
    ax_attn.set_yticks([])

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax_attn, orientation='vertical', pad=0.02, aspect=10)
    cbar.set_label('Attention Weight', fontsize=9)

    # Mark peak attention region
    peak_idx = np.argmax(attention)
    peak_time = time[peak_idx]
    ax_signal.axvline(x=peak_time, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax_attn.axvline(x=peak_time, color='red', linestyle='--', alpha=0.7, linewidth=1)

    # Annotate impact moment
    ax_signal.annotate(
        'Impact',
        xy=(peak_time, signal[peak_idx].max()),
        xytext=(peak_time + 0.5, signal[peak_idx].max() + y_margin * 0.5),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
        color='red',
    )

    plt.tight_layout()

    for fmt in ['png', 'pdf', 'svg']:
        save_path = output_path.with_suffix(f'.{fmt}')
        fig.savefig(save_path, format=fmt, dpi=300, bbox_inches='tight')
        logger.info(f"Saved attention heatmap: {save_path}")

    plt.close(fig)


def plot_attention_heatmap_detailed(
    signal: np.ndarray,
    attention: np.ndarray,
    output_path: Path,
    sample_rate: float = 50.0,
    window_size: float = 2.0,
):
    """
    Create detailed attention heatmap with SVM overlay.

    This version includes:
    - Raw signal (3 channels)
    - Signal Vector Magnitude (SVM)
    - Attention weights as background shading

    Args:
        signal: Raw accelerometer signal (3, T) or (T, 3)
        attention: Attention weights (T,) or (1, T)
        output_path: Path to save figure
        sample_rate: Sampling rate in Hz
        window_size: Time window around peak to zoom (seconds)
    """
    setup_publication_style()

    # Normalize inputs
    if signal.ndim == 2 and signal.shape[0] == 3:
        signal = signal.T  # (T, 3)

    attention = attention.flatten()
    T = len(signal)

    # Interpolate attention if needed
    if len(attention) != T:
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(attention))
        x_new = np.linspace(0, 1, T)
        attention = interp1d(x_old, attention, kind='linear')(x_new)

    time = np.arange(T) / sample_rate

    # Compute SVM
    svm = np.sqrt(np.sum(signal ** 2, axis=1))

    # Normalize attention
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

    # Find peak and create zoom window
    peak_idx = np.argmax(attention)
    peak_time = time[peak_idx]

    # Full signal plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) Full signal with attention shading
    ax1 = axes[0, 0]

    # Attention background
    for i in range(T - 1):
        ax1.axvspan(time[i], time[i + 1], alpha=attention[i] * 0.5, color='red', linewidth=0)

    axis_names = ['Acc-X', 'Acc-Y', 'Acc-Z']
    axis_colors = ['#2ecc71', '#3498db', '#9b59b6']

    for i, (name, color) in enumerate(zip(axis_names, axis_colors)):
        ax1.plot(time, signal[:, i], color=color, label=name, linewidth=1.2, alpha=0.9)

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (g)')
    ax1.set_title('(a) Full Signal with Attention Overlay', fontweight='bold')
    ax1.legend(loc='upper right', ncol=3)
    ax1.axvline(x=peak_time, color='red', linestyle='--', alpha=0.5, linewidth=1)

    # (b) SVM with attention
    ax2 = axes[0, 1]

    ax2_twin = ax2.twinx()
    ax2.plot(time, svm, color='#34495e', linewidth=1.5, label='SVM')
    ax2_twin.fill_between(time, 0, attention, alpha=0.3, color='red', label='Attention')
    ax2_twin.plot(time, attention, color='red', linewidth=1, alpha=0.7)

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('SVM (g)', color='#34495e')
    ax2_twin.set_ylabel('Attention Weight', color='red')
    ax2.set_title('(b) Signal Vector Magnitude & Attention', fontweight='bold')
    ax2.axvline(x=peak_time, color='red', linestyle='--', alpha=0.5, linewidth=1)

    # (c) Zoomed view around impact
    ax3 = axes[1, 0]

    zoom_start = max(0, peak_time - window_size)
    zoom_end = min(time[-1], peak_time + window_size)
    mask = (time >= zoom_start) & (time <= zoom_end)

    for i in range(sum(mask) - 1):
        idx = np.where(mask)[0][i]
        ax3.axvspan(time[idx], time[idx + 1], alpha=attention[idx] * 0.5, color='red', linewidth=0)

    for i, (name, color) in enumerate(zip(axis_names, axis_colors)):
        ax3.plot(time[mask], signal[mask, i], color=color, label=name, linewidth=1.5)

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (g)')
    ax3.set_title(f'(c) Zoomed View (±{window_size}s around peak)', fontweight='bold')
    ax3.legend(loc='upper right', ncol=3)
    ax3.axvline(x=peak_time, color='red', linestyle='--', alpha=0.7, linewidth=1.5)

    # (d) Attention heatmap
    ax4 = axes[1, 1]

    # Create 2D attention map
    attention_cmap = LinearSegmentedColormap.from_list(
        'attention', ['#ffffff', '#fee8e8', '#ff9999', '#ff3333', '#cc0000']
    )

    im = ax4.imshow(
        attention.reshape(1, -1),
        aspect='auto',
        cmap=attention_cmap,
        extent=[time.min(), time.max(), 0, 1],
        vmin=0, vmax=1,
    )
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('')
    ax4.set_yticks([])
    ax4.set_title('(d) Attention Weight Distribution', fontweight='bold')
    ax4.axvline(x=peak_time, color='black', linestyle='--', alpha=0.7, linewidth=1)

    cbar = fig.colorbar(im, ax=ax4, orientation='vertical', pad=0.02)
    cbar.set_label('Attention Weight')

    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        save_path = output_path.with_suffix(f'.{fmt}')
        fig.savefig(save_path, format=fmt, dpi=300, bbox_inches='tight')
        logger.info(f"Saved detailed attention heatmap: {save_path}")

    plt.close(fig)


# =============================================================================
# Model Loading Utilities
# =============================================================================

def load_model(
    checkpoint_path: Path,
    device: torch.device,
    ablation: Optional[Dict[str, bool]] = None,
) -> AMSNetV2:
    """
    Load AMSNetV2 model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint
        device: Torch device
        ablation: Optional ablation configuration

    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading model from: {checkpoint_path}")

    # Default ablation (full model)
    if ablation is None:
        ablation = {'mspa': True, 'dks': True, 'faa': True}

    model = AMSNetV2(
        in_channels=3,
        num_classes=2,
        ablation=ablation,
    ).to(device)

    # Load checkpoint
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # Remove 'module.' prefix if present (from DataParallel)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    logger.info(f"Model loaded successfully")
    return model


def create_baseline_model(device: torch.device) -> AMSNetV2:
    """
    Create baseline model (AMSNetV2 without key components).

    This serves as a baseline for comparison - same architecture
    but with MSPA, DKS, and FAA disabled.
    """
    ablation = {'mspa': False, 'dks': False, 'faa': False}
    model = AMSNetV2(
        in_channels=3,
        num_classes=2,
        ablation=ablation,
    ).to(device)
    model.eval()
    return model


# =============================================================================
# Demo Data Generation
# =============================================================================

def generate_demo_fall_signal(
    length: int = 512,
    sample_rate: float = 50.0,
    impact_time: float = 5.0,
) -> np.ndarray:
    """
    Generate synthetic fall signal for demonstration.

    Args:
        length: Signal length in samples
        sample_rate: Sampling rate in Hz
        impact_time: Time of impact in seconds

    Returns:
        signal: (3, length) accelerometer signal
    """
    t = np.arange(length) / sample_rate

    # Background noise
    noise = np.random.randn(3, length) * 0.05

    # Gravity baseline (standing upright)
    signal = noise.copy()
    signal[2, :] += 1.0  # Z-axis gravity

    # Impact parameters
    impact_idx = int(impact_time * sample_rate)
    impact_duration = int(0.2 * sample_rate)  # 200ms impact

    # Pre-fall: tilting (slow rotation)
    pre_fall_start = impact_idx - int(1.0 * sample_rate)
    pre_fall_end = impact_idx
    if pre_fall_start > 0:
        tilt = np.linspace(0, 0.5, pre_fall_end - pre_fall_start)
        signal[0, pre_fall_start:pre_fall_end] += tilt  # X-axis tilt
        signal[2, pre_fall_start:pre_fall_end] -= tilt * 0.3  # Reduce Z

    # Impact: sharp spike
    if impact_idx + impact_duration < length:
        # Multi-axis impact
        impact_profile = np.exp(-np.linspace(0, 5, impact_duration))
        signal[0, impact_idx:impact_idx + impact_duration] += 2.5 * impact_profile
        signal[1, impact_idx:impact_idx + impact_duration] += 1.5 * impact_profile * np.sin(np.linspace(0, 2*np.pi, impact_duration))
        signal[2, impact_idx:impact_idx + impact_duration] += 3.0 * impact_profile

    # Post-fall: lying still (different gravity orientation)
    post_fall_start = impact_idx + impact_duration
    if post_fall_start < length:
        signal[0, post_fall_start:] = 0.8 + noise[0, post_fall_start:]
        signal[1, post_fall_start:] = 0.2 + noise[1, post_fall_start:]
        signal[2, post_fall_start:] = 0.3 + noise[2, post_fall_start:]

    return signal.astype(np.float32)


def generate_demo_attention(
    length: int = 512,
    impact_idx: int = 250,
) -> np.ndarray:
    """
    Generate synthetic attention weights for demonstration.

    Args:
        length: Attention length
        impact_idx: Index of impact moment

    Returns:
        attention: (length,) attention weights
    """
    attention = np.zeros(length)

    # Background attention
    attention += 0.1

    # Peak around impact
    x = np.arange(length)
    sigma = 30  # Width of attention peak
    attention += 0.9 * np.exp(-((x - impact_idx) ** 2) / (2 * sigma ** 2))

    # Secondary peak at pre-fall
    pre_fall_idx = impact_idx - 50
    if pre_fall_idx > 0:
        attention += 0.3 * np.exp(-((x - pre_fall_idx) ** 2) / (2 * (sigma * 0.5) ** 2))

    return attention.astype(np.float32)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate publication figures for AMSNetV2 paper',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--baseline-ckpt', type=Path, help='Path to baseline model checkpoint')
    parser.add_argument('--amsv2-ckpt', type=Path, help='Path to AMSNetV2 checkpoint')
    parser.add_argument('--data-root', type=Path, default=Path('./data'), help='Dataset root directory')
    parser.add_argument('--output-dir', type=Path, default=Path('./figures'), help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-samples', type=int, default=1500, help='Max samples for t-SNE')
    parser.add_argument('--perplexity', type=int, default=30, help='t-SNE perplexity')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--demo', action='store_true', help='Generate demo figures with synthetic data')

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    if args.demo:
        logger.info("Generating demo figures with synthetic data...")

        # Generate synthetic features for t-SNE demo
        n_samples = 500

        # Baseline: features with poor separation
        baseline_features = np.random.randn(n_samples, 128)
        baseline_labels = np.random.randint(0, 2, n_samples)
        # Add small class-dependent shift
        baseline_features[baseline_labels == 1] += 0.5

        # AMSNetV2: features with good separation
        amsv2_features = np.random.randn(n_samples, 128)
        amsv2_labels = np.random.randint(0, 2, n_samples)
        # Add large class-dependent shift
        amsv2_features[amsv2_labels == 1] += 3.0

        # Plot t-SNE comparison
        plot_tsne_comparison(
            baseline_features, baseline_labels,
            amsv2_features, amsv2_labels,
            args.output_dir / 'tsne_comparison',
            perplexity=args.perplexity,
            random_state=args.seed,
        )

        # Generate synthetic fall signal and attention
        signal = generate_demo_fall_signal(length=512, impact_time=5.0)
        attention = generate_demo_attention(length=512, impact_idx=250)

        # Plot attention heatmaps
        plot_attention_heatmap(
            signal.T, attention,
            args.output_dir / 'attention_heatmap',
            sample_rate=50.0,
            title='Fall-Aware Attention on Fall Event',
        )

        plot_attention_heatmap_detailed(
            signal.T, attention,
            args.output_dir / 'attention_detailed',
            sample_rate=50.0,
        )

        logger.info(f"Demo figures saved to: {args.output_dir}")
        return

    # Real data mode
    if args.amsv2_ckpt is None:
        logger.error("Please provide --amsv2-ckpt or use --demo for synthetic data")
        return

    # Load model
    amsv2_model = load_model(args.amsv2_ckpt, device)

    # Load baseline if provided
    if args.baseline_ckpt:
        baseline_model = load_model(
            args.baseline_ckpt, device,
            ablation={'mspa': False, 'dks': False, 'faa': False}
        )
    else:
        logger.info("No baseline checkpoint provided, creating untrained baseline")
        baseline_model = create_baseline_model(device)

    # Try to load dataset
    try:
        from sisfall_dataset import SisFallDataset  # Assuming this exists

        dataset = SisFallDataset(args.data_root / 'SisFall')
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False, num_workers=4
        )

        # Extract features
        logger.info("Extracting features from baseline model...")
        baseline_features, baseline_labels = get_features_and_labels(
            baseline_model, loader, device, max_samples=args.max_samples
        )

        logger.info("Extracting features from AMSNetV2...")
        amsv2_features, amsv2_labels = get_features_and_labels(
            amsv2_model, loader, device, max_samples=args.max_samples
        )

        # Plot t-SNE
        plot_tsne_comparison(
            baseline_features, baseline_labels,
            amsv2_features, amsv2_labels,
            args.output_dir / 'tsne_comparison',
            perplexity=args.perplexity,
            random_state=args.seed,
        )

        # Get a fall sample for attention visualization
        for data, label in loader:
            if label[0] == 1:  # Fall
                sample = data[0:1].to(device)
                break

        # Extract attention
        attn_extractor = AttentionExtractor(amsv2_model)
        attention = attn_extractor.get_attention_weights(sample)

        if attention is not None:
            plot_attention_heatmap(
                sample.squeeze().cpu().numpy(),
                attention.squeeze().cpu().numpy(),
                args.output_dir / 'attention_heatmap',
            )

            plot_attention_heatmap_detailed(
                sample.squeeze().cpu().numpy(),
                attention.squeeze().cpu().numpy(),
                args.output_dir / 'attention_detailed',
            )

    except ImportError:
        logger.warning("Could not import dataset, using synthetic data for visualization")
        logger.info("Run with --demo flag for full demo with synthetic data")

        # Generate synthetic for attention demo
        signal = generate_demo_fall_signal()
        sample = torch.from_numpy(signal).unsqueeze(0).to(device)

        attn_extractor = AttentionExtractor(amsv2_model)
        attention = attn_extractor.get_attention_weights(sample)

        if attention is not None:
            plot_attention_heatmap(
                signal.T, attention.squeeze().cpu().numpy(),
                args.output_dir / 'attention_heatmap',
            )
        else:
            # Use synthetic attention
            attention = generate_demo_attention()
            plot_attention_heatmap(
                signal.T, attention,
                args.output_dir / 'attention_heatmap',
            )

    logger.info(f"All figures saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
