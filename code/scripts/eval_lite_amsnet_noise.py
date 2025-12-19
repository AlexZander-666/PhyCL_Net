"""
Noise robustness evaluation for Lite-AMSNet fall detection.

Implements the execution plan in docs/noise_robustness_execution_plan.md with:
- AWGN injection at target SNRs (dB) using variance (after de-meaning) as signal power.
- Repeated runs per SNR to report mean/std accuracy.
- Accuracy vs SNR plot with std shadow (fill_between), whitegrid background, dark line color, dpi=300.
- Raw results persisted to CSV for downstream plotting (Origin/LaTeX).
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score

try:
    import seaborn as sns
except Exception:
    sns = None

# Allow importing project modules (LiteAMSNet + SisFallDataset live in DMC_Net_experiments.py)
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DMC_Net_experiments import LiteAMSNet, SisFallDataset, _resolve_sisfall_root  # type: ignore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger("lite_amsnet_noise")


# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def build_dataloader(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    window_size: int,
    stride: int,
    channels_used: str,
    log_dir: Path,
    subjects: Sequence[str] | None = None,
) -> DataLoader:
    sis_root = _resolve_sisfall_root(str(data_root))
    dataset = SisFallDataset(
        sis_root,
        subjects=list(subjects) if subjects else [],
        window_size=window_size,
        stride=stride,
        log_dir=str(log_dir),
        channels_used=channels_used,
        transform=None,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No SisFall samples loaded from {sis_root}.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


# -----------------------------------------------------------------------------
# Noise injection
# -----------------------------------------------------------------------------
def _snr_to_noise_std(
    data: torch.Tensor,
    snr_db: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute per-sample noise std for the target SNR.

    Signal power P_s is defined as variance after de-meaning to remove gravity/DC:
        P_s = var(x) = mean((x - mean(x))^2)
    """
    if math.isinf(snr_db):
        return torch.zeros_like(data[:, :1, :])

    snr_linear = 10.0 ** (snr_db / 10.0)
    # Variance already de-means internally; keep explicit centering for clarity.
    centered = data - data.mean(dim=(1, 2), keepdim=True)
    signal_power = torch.mean(centered.pow(2), dim=(1, 2), keepdim=True)
    signal_power = torch.clamp(signal_power, min=eps)
    noise_power = signal_power / snr_linear
    return torch.sqrt(noise_power)


def add_awgn_for_snr(
    data: torch.Tensor,
    snr_db: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if math.isinf(snr_db):
        return data
    noise_std = _snr_to_noise_std(data, snr_db)
    # torch.randn_like on some PyTorch builds lacks the generator kwarg; fall back to randn
    noise = torch.randn(
        data.shape,
        device=data.device,
        dtype=data.dtype,
        generator=generator,
    ) * noise_std
    return data + noise


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def load_model(
    ckpt_path: Path,
    device: torch.device,
    in_channels: int,
    num_classes: int,
    channels: int,
    n_blocks: int,
    attn: str | None,
) -> LiteAMSNet:
    model = LiteAMSNet(
        in_channels=in_channels,
        num_classes=num_classes,
        channels=channels,
        n_blocks=n_blocks,
        attn=attn,
    ).to(device)

    LOGGER.info("Loading checkpoint: %s", ckpt_path)
    state = torch.load(ckpt_path, map_location=device)
    state_dict = None
    if isinstance(state, dict):
        # Handle multiple checkpoint formats from DMC_Net_experiments.py
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in state:
                state_dict = state[key]
                break
    if state_dict is None:
        # Assume the loaded object is already a state_dict
        state_dict = state

    # Strip DataParallel prefixes if needed
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        LOGGER.warning(
            "State dict loaded with missing_keys=%s unexpected_keys=%s",
            incompatible.missing_keys,
            incompatible.unexpected_keys,
        )
    model.eval()
    LOGGER.info("Checkpoint loaded.")
    return model


def evaluate_once(
    model: LiteAMSNet,
    loader: DataLoader,
    device: torch.device,
    snr_db: float,
    seed: int,
) -> Tuple[float, float]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    total = 0
    correct = 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                data, labels, _ = batch
            else:
                data, labels = batch
            data = data.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)

            noisy = add_awgn_for_snr(data, snr_db, generator=gen)
            outputs = model(noisy)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.numel()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    if total == 0:
        raise RuntimeError("No samples evaluated; check the dataloader.")
    acc = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, macro_f1


def run_repeats(
    model: LiteAMSNet,
    loader: DataLoader,
    device: torch.device,
    snrs_db: Iterable[float],
    repeats: int,
    base_seed: int,
) -> Tuple[List[Dict[str, float]], Dict[float, Dict[str, float]]]:
    per_run: List[Dict[str, float]] = []
    aggregated: Dict[float, Dict[str, float]] = {}

    for snr_db in snrs_db:
        acc_list: List[float] = []
        f1_list: List[float] = []
        for r in range(repeats):
            repeat_seed = base_seed + r if math.isinf(snr_db) else base_seed + int(snr_db * 1000) + r
            acc, macro_f1 = evaluate_once(model, loader, device, snr_db, seed=repeat_seed)
            acc_list.append(acc)
            f1_list.append(macro_f1)
            per_run.append({"snr_db": snr_db, "repeat": r, "accuracy": acc, "macro_f1": macro_f1})
            LOGGER.info("SNR=%s dB | repeat=%d | acc=%.4f | macro_f1=%.4f", snr_db, r, acc, macro_f1)

        mean_acc = float(np.mean(acc_list))
        std_acc = float(np.std(acc_list, ddof=0))
        mean_f1 = float(np.mean(f1_list))
        std_f1 = float(np.std(f1_list, ddof=0))
        aggregated[snr_db] = {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1,
        }
        LOGGER.info(
            "SNR=%s dB | mean_acc=%.4f | std_acc=%.4f | mean_f1=%.4f | std_f1=%.4f",
            snr_db,
            mean_acc,
            std_acc,
            mean_f1,
            std_f1,
        )

    return per_run, aggregated


# -----------------------------------------------------------------------------
# Persistence & plotting
# -----------------------------------------------------------------------------
def save_csv(
    rows: List[Dict[str, float]],
    summary: Dict[float, Dict[str, float]],
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "snr_db",
                "repeat",
                "accuracy",
                "macro_f1",
                "mean_accuracy",
                "std_accuracy",
                "mean_macro_f1",
                "std_macro_f1",
            ]
        )
        for row in rows:
            agg = summary[row["snr_db"]]
            writer.writerow(
                [
                    row["snr_db"],
                    row["repeat"],
                    f"{row['accuracy']:.6f}",
                    f"{row['macro_f1']:.6f}",
                    f"{agg['mean_accuracy']:.6f}",
                    f"{agg['std_accuracy']:.6f}",
                    f"{agg['mean_macro_f1']:.6f}",
                    f"{agg['std_macro_f1']:.6f}",
                ]
            )
    LOGGER.info("Saved raw + summary data to %s", csv_path)


def plot_accuracy_vs_snr(
    summary: Dict[float, Dict[str, float]],
    snr_order: Sequence[float],
    figure_path: Path,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    if sns:
        sns.set_theme(style="whitegrid")
    else:
        plt.style.use("seaborn-v0_8-whitegrid")

    snrs = list(snr_order)
    mean_acc = np.array([summary[s]["mean_accuracy"] for s in snrs]) * 100.0
    std_acc = np.array([summary[s]["std_accuracy"] for s in snrs]) * 100.0

    lower = np.clip(mean_acc - std_acc, 0.0, 100.0)
    upper = np.clip(mean_acc + std_acc, 0.0, 100.0)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(snrs, mean_acc, color="#2c3e50", marker="o", linewidth=2.2, label="Accuracy")
    ax.fill_between(snrs, lower, upper, color="#2c3e50", alpha=0.18, label="±1 std")

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Robustness to Sensor Noise")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    LOGGER.info("Saved plot to %s", figure_path)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Noise robustness evaluation for Lite-AMSNet (Accuracy vs SNR)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to Lite-AMSNet checkpoint (.pth)")
    parser.add_argument("--data-root", type=Path, default=Path("./data"), help="Path to SisFall root (or folder containing SisFall/)")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"), help="Directory to save CSV results")
    parser.add_argument("--figure-dir", type=Path, default=Path("./figures"), help="Directory to save the robustness plot")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--snrs",
        type=float,
        nargs="+",
        default=[40, 35, 30, 25, 20, 15, 10, 5],
        help="SNR values (dB)",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Repeats per SNR for mean/std estimation")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for noise generation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--channels-used", type=str, default="accel3", choices=["accel3", "accel6", "accel6+gyro", "accel9", "full"], help="Which sensor channels to load from SisFall")
    parser.add_argument("--window-size", type=int, default=512, help="Window length")
    parser.add_argument("--stride", type=int, default=256, help="Stride for windowing")
    parser.add_argument("--model-channels", type=int, default=32, help="Lite-AMSNet base channel width (must match training)")
    parser.add_argument("--model-blocks", type=int, default=2, help="Number of Lite-AMSNet blocks (must match training)")
    parser.add_argument("--attn-lite", type=str, default=None, help="Attention variant used during training (if any)")
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subject IDs (e.g., SA01,SA02); empty for all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    LOGGER.info("Using device: %s", device)
    set_seed(args.seed)

    subjects = [s.strip().upper() for s in args.subjects.split(",")] if args.subjects else None

    loader = build_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        window_size=args.window_size,
        stride=args.stride,
        channels_used=args.channels_used,
        log_dir=args.output_dir,
        subjects=subjects or [],
    )

    in_channels = {
        "accel3": 3,
        "accel6": 6,
        "accel6+gyro": 9,
        "accel9": 9,
        "full": 9,
    }.get(args.channels_used, 3)

    model = load_model(
        ckpt_path=args.ckpt,
        device=device,
        in_channels=in_channels,
        num_classes=2,
        channels=args.model_channels,
        n_blocks=args.model_blocks,
        attn=args.attn_lite,
    )

    per_run, summary = run_repeats(
        model=model,
        loader=loader,
        device=device,
        snrs_db=args.snrs,
        repeats=max(1, args.repeats),
        base_seed=args.seed,
    )

    csv_path = args.output_dir / "noise_robustness_results.csv"
    save_csv(per_run, summary, csv_path)

    figure_path = args.figure_dir / "Robustness_to_Sensor_Noise.png"
    plot_accuracy_vs_snr(summary, args.snrs, figure_path)

    LOGGER.info("Done. CSV: %s | Figure: %s", csv_path, figure_path)


if __name__ == "__main__":
    main()
