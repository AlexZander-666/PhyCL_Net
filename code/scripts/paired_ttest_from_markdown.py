"""
Read accuracy lists from a Markdown file and run paired t-tests + Cohen's d.

Inputs (must exist in the Markdown as Python assignments):
  - lite_amsnet_acc = [...]
  - lite_mspa_acc = [...]
  - inception_acc = [...]

Outputs:
  - Raw p-values + Cohen's d (paired, d_z) for:
      1) Lite-AMSNet vs Lite-AMSNet w/MSPA
      2) Lite-AMSNet vs InceptionTime
  - A paper-ready paragraph with conditional phrasing.
"""

from __future__ import annotations

import argparse
import ast
import re
import statistics
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy import stats


def extract_python_list(md_text: str, var_name: str) -> List[float]:
    pat = re.compile(
        rf"^\s*{re.escape(var_name)}\s*=\s*(\[[^\]]*\])\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    m = pat.search(md_text)
    if not m:
        raise ValueError(f"Could not find assignment for {var_name!r} in markdown.")

    list_text = m.group(1)
    try:
        parsed = ast.literal_eval(list_text)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Failed to parse list for {var_name!r}: {e}") from e

    if not isinstance(parsed, list):
        raise ValueError(f"Parsed {var_name!r} is not a list.")

    out: List[float] = []
    for idx, v in enumerate(parsed):
        if v is None:
            raise ValueError(f"{var_name}[{idx}] is None; cannot run paired test.")
        try:
            out.append(float(v))
        except (TypeError, ValueError) as e:
            raise ValueError(f"{var_name}[{idx}] is not numeric: {v!r}") from e
    return out


def cohen_dz(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        return float("nan")
    return float(np.mean(diff) / sd)


def paired_ttest_and_effect(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    t_stat, p_val = stats.ttest_rel(x_arr, y_arr, nan_policy="raise")
    d_z = cohen_dz(x_arr, y_arr)
    return float(t_stat), float(p_val), float(d_z)


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired t-tests from markdown accuracy lists.")
    parser.add_argument(
        "--md",
        default="docs/fold_test_accuracy_extraction_results.md",
        help="Path to markdown file containing the accuracy lists.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for report text (default: 0.05).",
    )
    args = parser.parse_args()

    md_path = Path(args.md)
    md_text = md_path.read_text(encoding="utf-8", errors="ignore")

    lite = extract_python_list(md_text, "lite_amsnet_acc")
    mspa = extract_python_list(md_text, "lite_mspa_acc")
    inc = extract_python_list(md_text, "inception_acc")

    if not (len(lite) == len(mspa) == len(inc)):
        raise ValueError(
            f"List lengths differ: lite={len(lite)}, mspa={len(mspa)}, inception={len(inc)}"
        )
    n = len(lite)
    if n < 2:
        raise ValueError(f"Need at least 2 paired samples, got n={n}.")

    t1, p1, d1 = paired_ttest_and_effect(lite, mspa)
    t2, p2, d2 = paired_ttest_and_effect(lite, inc)

    mean_diff_1 = statistics.mean([a - b for a, b in zip(lite, mspa)])
    mean_diff_2 = statistics.mean([a - b for a, b in zip(lite, inc)])

    print(f"n_pairs = {n}")
    print("")
    print("Pair 1 (Ablation): Lite-AMSNet vs Lite-AMSNet w/MSPA")
    print(f"  mean_diff (Proposed - MSPA) = {mean_diff_1:.4f} percentage points")
    print(f"  t = {t1:.6f}, p = {p1:.6g}, Cohen's d_z = {d1:.6f}")
    print("")
    print("Pair 2 (Baseline): Lite-AMSNet vs InceptionTime")
    print(f"  mean_diff (Proposed - Inception) = {mean_diff_2:.4f} percentage points")
    print(f"  t = {t2:.6f}, p = {p2:.6g}, Cohen's d_z = {d2:.6f}")
    print("")

    # Academic report generation (as requested)
    if p1 > args.alpha:
        ablation_text = (
            "In the ablation study, removing the MSPA module resulted in no significant "
            "performance degradation, validating the efficiency of the lightweight design."
        )
    else:
        ablation_text = (
            "In the ablation study, removing the MSPA module led to a statistically significant "
            "change in performance."
        )

    if p2 < args.alpha:
        baseline_text = "Compared with InceptionTime, the proposed Lite-AMSNet achieved a statistically significant improvement."
    else:
        baseline_text = (
            "Compared with InceptionTime, the proposed Lite-AMSNet achieved comparable performance "
            "with significantly lower computational cost."
        )

    paragraph = f"{ablation_text} {baseline_text}"
    print("Suggested paper paragraph:")
    print(paragraph)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

