from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap


# Allow local imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

from DMC_Net_experiments import (  # noqa: E402
    SisFallDataset,
    _resolve_sisfall_root,
    parse_ablation_config,
)
from models.ams_net_v2 import AMSNetV2  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


class AttentionExtractor:
    def __init__(self, model: nn.Module):
        self.model = model
        self.faa_modules: List[nn.Module] = []
        self._find_faa_modules(model)

    def _find_faa_modules(self, module: nn.Module) -> None:
        for _, child in module.named_children():
            if child.__class__.__name__ == "FallAwareAttention":
                self.faa_modules.append(child)
            else:
                self._find_faa_modules(child)

    def get_attention(self, x: torch.Tensor, stage_idx: int = -1) -> Optional[torch.Tensor]:
        with torch.no_grad():
            _ = self.model(x)
        if not self.faa_modules:
            return None
        idx = stage_idx if stage_idx >= 0 else len(self.faa_modules) + stage_idx
        faa = self.faa_modules[idx]
        if hasattr(faa, "last_attention") and faa.last_attention is not None:
            attn = faa.last_attention
            if attn.dim() == 3:
                return attn.mean(dim=1, keepdim=True)
            return attn
        return None


def load_args_from_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    text = config_path.read_text(encoding="utf-8", errors="replace")
    lines = []
    in_args = False
    for line in text.splitlines():
        if line.startswith("args:"):
            in_args = True
        if in_args:
            if line.startswith("dependencies:"):
                break
            lines.append(line)
    if not lines:
        return {}
    try:
        import yaml  # pylint: disable=import-error
    except ImportError:
        LOGGER.warning("PyYAML missing; skipping config parse.")
        return {}
    payload = "\n".join(lines)
    data = yaml.safe_load(payload)
    return data.get("args", {}) if isinstance(data, dict) else {}


def load_model(
    ckpt_path: Path,
    config_args: Dict[str, Any],
    ablation_spec: str,
    device: torch.device,
) -> AMSNetV2:
    ablation = parse_ablation_config(ablation_spec or config_args.get("ablation"))
    model = AMSNetV2(
        in_channels=int(config_args.get("in_channels", 3)),
        num_classes=int(config_args.get("num_classes", 2)),
        proj_dim=int(config_args.get("proj_dim", 128)),
        ablation=ablation,
        freq_method=config_args.get("freq_method", "fft"),
        sample_rate=float(config_args.get("sample_rate", 50.0)),
        time_attn=config_args.get("attn_time", "none"),
        freq_attn=config_args.get("attn_freq", "none"),
        fusion_attn=config_args.get("attn_fuse", "none"),
        fusion_variant=config_args.get("fusion_variant", "enhanced"),
        fusion_kernel_sizes=tuple(config_args.get("fusion_kernel_sizes", (3, 5, 7))),
        adaptive_bands=bool(config_args.get("adaptive_bands", True)),
        band_edges=tuple(config_args.get("band_edges")) if config_args.get("band_edges") else None,
        num_bands=int(config_args.get("num_bands", 4)),
        faa_axis_attn=bool(config_args.get("faa_axis_attn", True)),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def normalize_attention(attn: np.ndarray, target_len: int) -> np.ndarray:
    attn = attn.reshape(-1)
    if len(attn) != target_len:
        x_old = np.linspace(0, 1, len(attn))
        x_new = np.linspace(0, 1, target_len)
        attn = np.interp(x_new, x_old, attn)
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    return attn


def plot_samples(
    samples: List[Dict[str, Any]],
    output_path: Path,
    sample_rate: float,
) -> None:
    n = len(samples)
    fig_height = max(10, n * 3.2)
    fig = plt.figure(figsize=(10, fig_height))
    gs = fig.add_gridspec(nrows=n * 2, ncols=1, height_ratios=[4, 1] * n, hspace=0.45)

    attention_cmap = LinearSegmentedColormap.from_list(
        "attention", ["#ffffff", "#ffdddd", "#ff9999", "#ff3333", "#cc0000"]
    )
    axes = []

    for idx, sample in enumerate(samples):
        signal = sample["signal"]
        if signal.shape[0] == 3:
            signal = signal.T
        length = signal.shape[0]
        time = np.arange(length) / sample_rate
        attention = normalize_attention(sample["attention"], length)

        ax_sig = fig.add_subplot(gs[idx * 2])
        ax_att = fig.add_subplot(gs[idx * 2 + 1], sharex=ax_sig)
        axes.append(ax_sig)

        axis_names = ["Acc-X", "Acc-Y", "Acc-Z"]
        axis_colors = ["#2ecc71", "#3498db", "#e67e22"]
        for c in range(3):
            points = np.array([time, signal[:, c]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            colors_rgba = plt.cm.colors.to_rgba_array([axis_colors[c]] * (length - 1))
            colors_rgba[:, 3] = 0.3 + 0.7 * attention[:-1]
            lc = LineCollection(segments, colors=colors_rgba, linewidths=1.2)
            ax_sig.add_collection(lc)
            ax_sig.plot([], [], color=axis_colors[c], label=axis_names[c], linewidth=1.5)

        y_margin = 0.1 * (signal.max() - signal.min() + 1e-8)
        ax_sig.set_xlim(time.min(), time.max())
        ax_sig.set_ylim(signal.min() - y_margin, signal.max() + y_margin)
        ax_sig.set_ylabel("Accel (g)")
        ax_sig.legend(loc="upper right", ncol=3, framealpha=0.9, fontsize=8)
        ax_sig.set_title(sample["title"], fontweight="bold", fontsize=10)
        plt.setp(ax_sig.get_xticklabels(), visible=False)

        ax_att.imshow(
            attention.reshape(1, -1),
            aspect="auto",
            cmap=attention_cmap,
            extent=[time.min(), time.max(), 0, 1],
            vmin=0,
            vmax=1,
        )
        ax_att.set_yticks([])
        ax_att.set_ylabel("Attn")
        ax_att.set_xlabel("Time (s)")

    fig.suptitle("How the Model Detects Falls", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.98, 0.98])
    for ext in ["png", "pdf"]:
        save_path = output_path.with_suffix(f".{ext}")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        LOGGER.info("Saved figure: %s", save_path)
    plt.close(fig)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate attention-based explainability figure for Lite-AMSNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path.")
    parser.add_argument("--config", type=Path, default=None, help="Optional experiment_config.yaml path.")
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./figures"))
    parser.add_argument("--ablation", type=str, default="mspa:False")
    parser.add_argument("--channels-used", type=str, default="accel3")
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--num-tp", type=int, default=3)
    parser.add_argument("--num-fp", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_args: Dict[str, Any] = {}
    if args.config is not None:
        config_args = load_args_from_config(args.config)
    elif args.ckpt.parent.joinpath("experiment_config.yaml").exists():
        config_args = load_args_from_config(args.ckpt.parent / "experiment_config.yaml")

    sample_rate = float(config_args.get("sample_rate", 50.0))
    window_size = int(config_args.get("window_size", args.window_size))
    stride = int(config_args.get("stride", args.stride))
    channels_used = str(config_args.get("channels_used", args.channels_used))

    device = torch.device(args.device)
    model = load_model(args.ckpt, config_args, args.ablation, device)
    attn_extractor = AttentionExtractor(model)

    sisfall_root = _resolve_sisfall_root(str(args.data_root))
    dataset = SisFallDataset(
        sisfall_root,
        subjects=[],
        window_size=window_size,
        stride=stride,
        log_dir=str(args.output_dir),
        channels_used=channels_used,
        transform=None,
    )

    if len(dataset) == 0:
        raise RuntimeError("SisFallDataset is empty.")

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    tp_samples = []
    fp_samples = []

    with torch.no_grad():
        for x, y, subj in loader:
            x = x.to(device)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            for i in range(len(y)):
                label = int(y[i])
                pred = int(preds[i])
                prob_fall = float(probs[i, 1].detach().cpu())
                if label == 1 and pred == 1 and len(tp_samples) < args.num_tp:
                    tp_samples.append(
                        {
                            "signal": x[i].detach().cpu().numpy(),
                            "label": label,
                            "pred": pred,
                            "prob": prob_fall,
                            "subj": subj[i],
                        }
                    )
                if label == 0 and pred == 1 and len(fp_samples) < args.num_fp:
                    fp_samples.append(
                        {
                            "signal": x[i].detach().cpu().numpy(),
                            "label": label,
                            "pred": pred,
                            "prob": prob_fall,
                            "subj": subj[i],
                        }
                    )
            if len(tp_samples) >= args.num_tp and len(fp_samples) >= args.num_fp:
                break

    if len(tp_samples) < args.num_tp:
        raise RuntimeError("Not enough TP fall samples found.")
    if len(fp_samples) < args.num_fp:
        raise RuntimeError("Not enough FP ADL samples found.")

    samples: List[Dict[str, Any]] = []
    for idx, sample in enumerate(tp_samples, 1):
        x = torch.from_numpy(sample["signal"]).unsqueeze(0).to(device)
        attention = attn_extractor.get_attention(x)
        if attention is None:
            raise RuntimeError("Attention extraction failed.")
        sample["attention"] = attention.squeeze().cpu().numpy()
        sample["title"] = f"TP Fall #{idx} (GT: Fall, Pred: Fall, P={sample['prob']:.3f})"
        samples.append(sample)

    for sample in fp_samples:
        x = torch.from_numpy(sample["signal"]).unsqueeze(0).to(device)
        attention = attn_extractor.get_attention(x)
        if attention is None:
            raise RuntimeError("Attention extraction failed.")
        sample["attention"] = attention.squeeze().cpu().numpy()
        sample["title"] = f"Misclassified ADL (Diagnostic) (GT: ADL, Pred: Fall, P={sample['prob']:.3f})"
        samples.append(sample)

    output_path = args.output_dir / "how_model_detects_falls"
    plot_samples(samples, output_path, sample_rate=sample_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
