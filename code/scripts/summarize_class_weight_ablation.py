from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple


METRICS = {
    "accuracy": ("accuracy_mean_mean", "accuracy_mean_std"),
    "macro_f1": ("macro_f1_mean_mean", "macro_f1_mean_std"),
    "fall_recall": ("sensitivity_mean_mean", "sensitivity_mean_std"),
}


def read_summary(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(summary: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    metrics: Dict[str, Tuple[float, float]] = {}
    for name, (mean_key, std_key) in METRICS.items():
        if mean_key not in summary or std_key not in summary:
            raise KeyError(f"Missing keys for {name}: {mean_key}, {std_key}")
        metrics[name] = (float(summary[mean_key]), float(summary[std_key]))
    return metrics


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean * 100:.2f} ± {std * 100:.2f}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize class weighting ablation results into CSV and Markdown.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ablation-root",
        type=str,
        default="outputs/ablation/class_weighting",
        help="Root directory containing strategy subfolders.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/ablation/class_weight_ablation.csv",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="docs/class_weight_ablation.md",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["none", "auto", "sqrt_inv_freq", "effective_num"],
        help="Strategy folder names to include in the table.",
    )
    parser.add_argument("--effective-num-beta", type=float, default=0.999)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.ablation_root)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)

    rows = []
    for strategy in args.strategies:
        summary_path = root / strategy / "summary_results.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary file: {summary_path}")
        summary = read_summary(summary_path)
        metrics = extract_metrics(summary)
        rows.append(
            {
                "strategy": strategy,
                "accuracy_mean": metrics["accuracy"][0],
                "accuracy_std": metrics["accuracy"][1],
                "macro_f1_mean": metrics["macro_f1"][0],
                "macro_f1_std": metrics["macro_f1"][1],
                "fall_recall_mean": metrics["fall_recall"][0],
                "fall_recall_std": metrics["fall_recall"][1],
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "accuracy_mean",
                "accuracy_std",
                "macro_f1_mean",
                "macro_f1_std",
                "fall_recall_mean",
                "fall_recall_std",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Class Weighting Ablation (Auto vs Alternatives)")
    lines.append("")
    lines.append("Dataset: SisFall (LOSO). Model: Lite-AMSNet (MSPA removed).")
    lines.append("Strategies: none, auto (inverse frequency), sqrt_inv_freq, effective_num.")
    lines.append(f"effective_num beta = {args.effective_num_beta:.3f}.")
    lines.append("")
    lines.append("| Strategy | Accuracy (%) | Macro-F1 (%) | Fall Recall (%) |")
    lines.append("|---|---|---|---|")
    for row in rows:
        acc = format_mean_std(row["accuracy_mean"], row["accuracy_std"])
        macro_f1 = format_mean_std(row["macro_f1_mean"], row["macro_f1_std"])
        fall_recall = format_mean_std(row["fall_recall_mean"], row["fall_recall_std"])
        lines.append(f"| {row['strategy']} | {acc} | {macro_f1} | {fall_recall} |")
    lines.append("")
    lines.append("Notes:")
    lines.append("- All values are mean ± std across seeds.")
    lines.append("- Auto corresponds to the current inverse-frequency weighting.")

    output_md.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
