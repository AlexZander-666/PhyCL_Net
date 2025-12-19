"""
Extract per-fold Test Accuracy values from experiment log files.

This script scans one or more root directories recursively, identifies log files
for three models (Lite-AMSNet, Lite-AMSNet w/MSPA, InceptionTime), extracts
fold/subject IDs and test accuracies, and prints three Python lists sorted by
fold ID.

Typical log patterns supported (case-insensitive):
- "Test Accuracy: 98.20%"
- "Accuracy: 0.9820"
- "best_acc: 98.2"
- "Test Results: {'accuracy': 0.9845, ...}"

Fold/subject ID is extracted from either filename/path (e.g., "fold_1",
"subject_1") or log content (e.g., "[loso_SA01]").
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


MODEL_LITE = "lite"
MODEL_MSPA = "mspa"
MODEL_INCEPTION = "inception"


DEFAULT_EXCLUDE_EXTS = {
    ".pth",
    ".pt",
    ".ckpt",
    ".npy",
    ".npz",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
}


DEFAULT_TEXT_EXTS = {
    ".log",
    ".txt",
    ".out",
    ".err",
}


FOLD_ID_PATTERNS = [
    re.compile(r"(?:^|[\\/_.-])fold[_ \\-]*(\d{1,4})(?:\D|$)", re.IGNORECASE),
    re.compile(r"(?:^|[\\/_.-])subject[_ \\-]*(\d{1,4})(?:\D|$)", re.IGNORECASE),
    re.compile(r"(?:^|[\\/_.-])subj(?:ect)?[_ \\-]*(\d{1,4})(?:\D|$)", re.IGNORECASE),
    re.compile(r"\bloso[_ \\-]*sa(\d{1,4})\b", re.IGNORECASE),
    re.compile(r"\bsa(\d{2})\b", re.IGNORECASE),
]


SEED_PATTERNS = [
    re.compile(r"\bseed set to\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bseed\s*(\d+)\b", re.IGNORECASE),
]


MODEL_FROM_LINE_PATTERNS = [
    re.compile(r"\bmodel\s*:\s*(\w+)\b", re.IGNORECASE),
]


MSPA_FLAG_PATTERNS = [
    re.compile(r"'mspa'\s*:\s*(true|false)", re.IGNORECASE),
    re.compile(r"mspa\s*=\s*(true|false)", re.IGNORECASE),
]


@dataclass(frozen=True)
class AccuracyCandidate:
    value_pct: float
    priority: int
    line_no: int
    raw_line: str
    fold_id: Optional[int]
    seed_id: Optional[int]


def _safe_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_accuracy_to_pct(value: float, has_percent_sign: bool) -> float:
    if has_percent_sign:
        return value
    if value <= 1.0:
        return value * 100.0
    return value


def extract_fold_id_from_path(path: str) -> Optional[int]:
    for pat in FOLD_ID_PATTERNS:
        m = pat.search(path)
        if not m:
            continue
        fold_id = _safe_float(m.group(1))
        if fold_id is None:
            continue
        return int(fold_id)
    return None


def extract_fold_id_from_line(line: str) -> Optional[int]:
    for pat in FOLD_ID_PATTERNS:
        m = pat.search(line)
        if not m:
            continue
        fold_id = _safe_float(m.group(1))
        if fold_id is None:
            continue
        return int(fold_id)
    return None


def extract_seed_id_from_line(line: str) -> Optional[int]:
    for pat in SEED_PATTERNS:
        m = pat.search(line)
        if not m:
            continue
        seed_id = _safe_float(m.group(1))
        if seed_id is None:
            continue
        return int(seed_id)
    return None


def extract_model_name_from_line(line: str) -> Optional[str]:
    for pat in MODEL_FROM_LINE_PATTERNS:
        m = pat.search(line)
        if m:
            return m.group(1).strip().lower()
    return None


def extract_mspa_flag_from_line(line: str) -> Optional[bool]:
    for pat in MSPA_FLAG_PATTERNS:
        m = pat.search(line)
        if not m:
            continue
        value = m.group(1).strip().lower()
        if value == "true":
            return True
        if value == "false":
            return False
    return None


def build_accuracy_patterns() -> List[Tuple[int, re.Pattern[str]]]:
    number = r"(?P<val>-?\d+(?:\.\d+)?)"
    pct = r"(?P<pct>%?)"

    patterns: List[Tuple[int, re.Pattern[str]]] = [
        # Highest priority: explicit best
        (
            3,
            re.compile(
                rf"\bbest[_\s-]*acc(?:uracy)?\b\s*[:=]\s*{number}\s*{pct}",
                re.IGNORECASE,
            ),
        ),
        # Common: explicit test accuracy
        (
            2,
            re.compile(
                rf"\btest\s*acc(?:uracy)?\b\s*[:=]\s*{number}\s*{pct}",
                re.IGNORECASE,
            ),
        ),
        # Common in this repo: "Test Results: {'accuracy': 0.9845, ...}"
        (
            2,
            re.compile(
                rf"\btest results\b.*?\baccuracy\b['\"]?\s*:\s*{number}\b",
                re.IGNORECASE,
            ),
        ),
        # Generic accuracy (lowest priority)
        (
            1,
            re.compile(
                rf"\bacc(?:uracy)?\b\s*[:=]\s*{number}\s*{pct}",
                re.IGNORECASE,
            ),
        ),
        # JSON/dict-like: "'accuracy': 0.9845"
        (
            1,
            re.compile(
                rf"\baccuracy\b['\"]?\s*:\s*{number}\b",
                re.IGNORECASE,
            ),
        ),
    ]
    return patterns


def extract_accuracy_candidates_from_line(
    line: str,
    line_no: int,
    fold_id: Optional[int],
    seed_id: Optional[int],
    patterns: List[Tuple[int, re.Pattern[str]]],
) -> List[AccuracyCandidate]:
    lowered = line.lower()
    candidates: List[AccuracyCandidate] = []

    # Avoid grabbing training accuracy from generic patterns unless it's clearly test/best.
    looks_like_train_only = ("train" in lowered) and ("test" not in lowered) and ("best" not in lowered)
    for priority, pat in patterns:
        if priority == 1 and looks_like_train_only:
            continue
        for m in pat.finditer(line):
            val_text = m.group("val")
            val = _safe_float(val_text)
            if val is None:
                continue
            has_pct = False
            if "pct" in m.groupdict():
                has_pct = (m.group("pct") or "").strip() == "%"
            value_pct = _normalize_accuracy_to_pct(val, has_pct)
            if not (0.0 <= value_pct <= 100.0):
                continue
            candidates.append(
                AccuracyCandidate(
                    value_pct=value_pct,
                    priority=priority,
                    line_no=line_no,
                    raw_line=line.rstrip("\n"),
                    fold_id=fold_id,
                    seed_id=seed_id,
                )
            )
    return candidates


def looks_like_text_file(path: str, exclude_exts: Iterable[str]) -> bool:
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext in set(e.lower() for e in exclude_exts):
        return False
    if ext in DEFAULT_TEXT_EXTS:
        return True
    # Also allow extensionless files (common for logs) and ".log.*".
    if ext == "":
        return True
    if ".log." in path.lower():
        return True
    return False


def iter_candidate_log_files(
    roots: List[str],
    exclude_exts: Iterable[str],
    max_mb: float,
) -> Iterable[str]:
    max_bytes = int(max_mb * 1024 * 1024)
    for root in roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                if not looks_like_text_file(path, exclude_exts):
                    continue
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size > max_bytes:
                    continue
                yield path


def classify_model_from_observations(
    path: str,
    model_name: Optional[str],
    mspa_enabled: Optional[bool],
) -> Optional[str]:
    path_l = path.lower()
    if model_name == "inceptiontime" or "inceptiontime" in path_l:
        return MODEL_INCEPTION

    # "Lite-AMSNet" can appear as "amsv2" in these logs; use the MSPA ablation flag if present.
    if model_name in {"amsv2", "lite-amsnet", "lite_amsnet", "liteamsnet", "amsnet"} or "amsv2" in path_l:
        if mspa_enabled is True:
            return MODEL_MSPA
        if mspa_enabled is False:
            return MODEL_LITE

    # Path-based fallback heuristics.
    if "no_mspa" in path_l or "ablation_no_mspa" in path_l:
        return MODEL_LITE
    if "mspa" in path_l or "full_mspa" in path_l:
        return MODEL_MSPA
    if "lite-amsnet" in path_l or "lite_amsnet" in path_l or "liteamsnet" in path_l:
        return MODEL_LITE

    return None


def aggregate(values: List[float], method: str) -> float:
    if not values:
        raise ValueError("Cannot aggregate empty list.")
    if method == "mean":
        return float(statistics.mean(values))
    if method == "median":
        return float(statistics.median(values))
    if method == "max":
        return float(max(values))
    if method == "last":
        return float(values[-1])
    raise ValueError(f"Unknown aggregation method: {method}")


def parse_log_file(
    path: str,
    patterns: List[Tuple[int, re.Pattern[str]]],
    pick_within_file: str,
) -> Tuple[Optional[str], Dict[Tuple[int, Optional[int]], float]]:
    """
    Returns:
      (model_key, {(fold_id, seed_id): accuracy_pct})
    """
    model_name: Optional[str] = None
    mspa_enabled: Optional[bool] = None
    last_fold_id: Optional[int] = extract_fold_id_from_path(path)
    last_seed_id: Optional[int] = None

    candidates_by_key: Dict[Tuple[int, Optional[int]], List[AccuracyCandidate]] = {}

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, start=1):
                fold_from_line = extract_fold_id_from_line(line)
                if fold_from_line is not None:
                    last_fold_id = fold_from_line

                seed_from_line = extract_seed_id_from_line(line)
                if seed_from_line is not None:
                    last_seed_id = seed_from_line

                maybe_model = extract_model_name_from_line(line)
                if maybe_model is not None:
                    model_name = maybe_model

                maybe_mspa = extract_mspa_flag_from_line(line)
                if maybe_mspa is not None:
                    mspa_enabled = maybe_mspa

                if last_fold_id is None:
                    continue

                for cand in extract_accuracy_candidates_from_line(
                    line=line,
                    line_no=line_no,
                    fold_id=last_fold_id,
                    seed_id=last_seed_id,
                    patterns=patterns,
                ):
                    key = (cand.fold_id, cand.seed_id)
                    candidates_by_key.setdefault(key, []).append(cand)
    except OSError:
        return None, {}

    model_key = classify_model_from_observations(path, model_name, mspa_enabled)
    if model_key is None:
        return None, {}

    picked: Dict[Tuple[int, Optional[int]], float] = {}
    for key, cands in candidates_by_key.items():
        if not cands:
            continue
        max_priority = max(c.priority for c in cands)
        top = [c for c in cands if c.priority == max_priority]
        if pick_within_file == "last":
            picked[key] = top[-1].value_pct
        elif pick_within_file == "max":
            picked[key] = max(c.value_pct for c in top)
        else:
            raise ValueError(f"Unknown --pick-within-file: {pick_within_file}")

    return model_key, picked


def format_python_list(values: List[Optional[float]], round_digits: Optional[int]) -> str:
    def fmt(v: Optional[float]) -> str:
        if v is None:
            return "None"
        if round_digits is None:
            return repr(float(v))
        return repr(round(float(v), round_digits))

    return "[" + ", ".join(fmt(v) for v in values) + "]"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract per-fold Test Accuracy values for paired t-tests.",
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=[],
        help="Root directories to scan recursively (default: logs, outputs, 大创/outputs if present).",
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=50.0,
        help="Skip files larger than this size in MB (default: 50).",
    )
    parser.add_argument(
        "--exclude-ext",
        nargs="*",
        default=sorted(DEFAULT_EXCLUDE_EXTS),
        help="File extensions to skip (default includes checkpoints/images/archives).",
    )
    parser.add_argument(
        "--include-ext",
        nargs="*",
        default=[],
        help="Additional file extensions to treat as text logs (e.g., .json .yaml).",
    )
    parser.add_argument(
        "--pick-within-file",
        choices=["max", "last"],
        default="max",
        help="If multiple matches in one file, pick the max or the last (default: max).",
    )
    parser.add_argument(
        "--aggregate-seeds",
        choices=["mean", "median", "max", "last"],
        default="mean",
        help="If multiple seeds per fold exist, aggregate them per fold (default: mean).",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=2,
        help="Round output accuracies to N decimals (default: 2). Use -1 for no rounding.",
    )
    args = parser.parse_args()

    roots = list(args.roots)
    if not roots:
        for candidate in ["logs/analysis", "logs", "outputs", os.path.join("大创", "outputs")]:
            if os.path.exists(candidate):
                roots.append(candidate)

    patterns = build_accuracy_patterns()

    for ext in args.include_ext:
        if not ext:
            continue
        normalized = ext.lower()
        if not normalized.startswith("."):
            normalized = "." + normalized
        DEFAULT_TEXT_EXTS.add(normalized)

    # results[model][fold][seed] = [acc1, acc2, ...]
    results: Dict[str, Dict[int, Dict[Optional[int], List[float]]]] = {
        MODEL_LITE: {},
        MODEL_MSPA: {},
        MODEL_INCEPTION: {},
    }

    scanned_files = 0
    used_files = 0
    for path in iter_candidate_log_files(roots, args.exclude_ext, args.max_mb):
        scanned_files += 1
        model_key, picked = parse_log_file(path, patterns, args.pick_within_file)
        if model_key is None or not picked:
            continue
        used_files += 1
        for (fold_id, seed_id), acc in picked.items():
            if fold_id is None:
                continue
            results.setdefault(model_key, {}).setdefault(fold_id, {}).setdefault(seed_id, []).append(acc)

    def fold_list_for(model_key: str) -> Tuple[List[int], List[Optional[float]]]:
        fold_ids = sorted(results.get(model_key, {}).keys())
        values: List[Optional[float]] = []
        for fold_id in fold_ids:
            seed_map = results[model_key][fold_id]
            per_seed_values: List[float] = []
            for _, vals in seed_map.items():
                per_seed_values.append(aggregate(vals, "max"))
            if not per_seed_values:
                values.append(None)
                continue
            values.append(aggregate(per_seed_values, args.aggregate_seeds))
        return fold_ids, values

    lite_fold_ids, lite_vals = fold_list_for(MODEL_LITE)
    mspa_fold_ids, mspa_vals = fold_list_for(MODEL_MSPA)
    inc_fold_ids, inc_vals = fold_list_for(MODEL_INCEPTION)

    all_folds = sorted(set(lite_fold_ids) | set(mspa_fold_ids) | set(inc_fold_ids))
    if not all_folds:
        print(
            "No fold accuracies found. Try adjusting --roots or check that logs contain 'Test Results'/'accuracy'.",
            file=sys.stderr,
        )
        return 2

    def align_to_all_folds(fold_ids: List[int], vals: List[Optional[float]]) -> List[Optional[float]]:
        mapping = {fid: vals[i] for i, fid in enumerate(fold_ids)}
        return [mapping.get(fid) for fid in all_folds]

    lite_aligned = align_to_all_folds(lite_fold_ids, lite_vals)
    mspa_aligned = align_to_all_folds(mspa_fold_ids, mspa_vals)
    inc_aligned = align_to_all_folds(inc_fold_ids, inc_vals)

    round_digits: Optional[int] = None if args.round < 0 else args.round

    print(f"# scanned_files={scanned_files}, used_files={used_files}, folds={all_folds}")
    print(f"fold_ids = {all_folds}")
    print(f"lite_amsnet_acc = {format_python_list(lite_aligned, round_digits)}")
    print(f"lite_mspa_acc = {format_python_list(mspa_aligned, round_digits)}")
    print(f"inception_acc = {format_python_list(inc_aligned, round_digits)}")

    missing = []
    for fid, a, b, c in zip(all_folds, lite_aligned, mspa_aligned, inc_aligned):
        if a is None or b is None or c is None:
            missing.append(fid)
    if missing:
        print(f"# WARNING: Missing values for folds: {missing}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
