import argparse
import io
import json
import logging
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


MOBIACT_GITHUB_ZIP_URL = (
    "https://github.com/yehowlong/MobiAct_Dataset_v2.0-MobiFall_Dataset_v2.0/archive/refs/heads/main.zip"
)
MOBIACT_FALL_CODES = {"FOL", "FKL", "BSC", "SDL"}
MOBIACT_ADL_CODES = {"STD", "WAL", "JOG", "JUM", "STU", "STN", "SCH", "CSI", "CSO"}
KFALL_FALL_TASK_IDS = set(range(20, 35))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def crop_or_pad_window(x: np.ndarray, target_len: int, center_idx: int) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array shaped (channels, length), got {x.shape}")
    length = x.shape[1]
    if length == 0:
        raise ValueError("Cannot crop an empty sequence.")
    if length == target_len:
        return x.astype(np.float32, copy=False)
    if length < target_len:
        pad_total = target_len - length
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(x, ((0, 0), (pad_left, pad_right)), mode="edge").astype(np.float32, copy=False)

    center_idx = int(np.clip(center_idx, 0, length - 1))
    start = center_idx - target_len // 2
    end = start + target_len
    if start < 0:
        start = 0
        end = target_len
    if end > length:
        end = length
        start = length - target_len
    return x[:, start:end].astype(np.float32, copy=False)


def resample_channels(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array shaped (channels, length), got {x.shape}")
    if x.shape[1] == target_len:
        return x.astype(np.float32, copy=False)
    if x.shape[1] < 2:
        return np.repeat(x.astype(np.float32, copy=False), target_len, axis=1)

    source_idx = np.linspace(0.0, 1.0, num=x.shape[1], dtype=np.float64)
    target_idx = np.linspace(0.0, 1.0, num=target_len, dtype=np.float64)
    out = np.empty((x.shape[0], target_len), dtype=np.float32)
    for channel in range(x.shape[0]):
        out[channel] = np.interp(target_idx, source_idx, x[channel].astype(np.float64))
    return out


def extract_fixed_center_window(x: np.ndarray, target_len: int) -> np.ndarray:
    return crop_or_pad_window(x, target_len=target_len, center_idx=x.shape[1] // 2)


def estimate_peak_center(x: np.ndarray) -> int:
    magnitude = np.linalg.norm(x.astype(np.float32, copy=False), axis=0)
    return int(np.argmax(magnitude))


def extract_kfall_window(
    x: np.ndarray,
    target_len: int,
    label: int,
    onset_frame: Optional[float] = None,
    impact_frame: Optional[float] = None,
) -> np.ndarray:
    if int(label) != 1:
        return extract_fixed_center_window(x, target_len)

    if onset_frame is not None and impact_frame is not None:
        center_idx = int(round((float(onset_frame) + float(impact_frame)) / 2.0))
    elif impact_frame is not None:
        center_idx = int(round(float(impact_frame)))
    elif onset_frame is not None:
        center_idx = int(round(float(onset_frame)))
    else:
        center_idx = x.shape[1] // 2
    return crop_or_pad_window(x, target_len=target_len, center_idx=center_idx)


def allocate_subject_splits(subjects: Sequence[int], seed: int) -> Dict[str, set]:
    unique_subjects = sorted({int(s) for s in subjects})
    if len(unique_subjects) < 3:
        raise ValueError("Need at least 3 unique subjects to create train/val/test splits.")

    rng = np.random.default_rng(seed)
    shuffled = list(unique_subjects)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_n = max(1, int(round(n * 0.7)))
    val_n = max(1, int(round(n * 0.15)))
    if train_n + val_n >= n:
        overflow = train_n + val_n - (n - 1)
        train_n = max(1, train_n - overflow)
    test_n = n - train_n - val_n
    if test_n <= 0:
        test_n = 1
        if train_n > val_n:
            train_n -= 1
        else:
            val_n -= 1

    train_subjects = set(shuffled[:train_n])
    val_subjects = set(shuffled[train_n : train_n + val_n])
    test_subjects = set(shuffled[train_n + val_n :])
    return {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }


def write_split_npz(
    out_dir: Path,
    split_name: str,
    xs: Sequence[np.ndarray],
    ys: Sequence[int],
    subjects: Sequence[int],
    sources: Sequence[str],
) -> None:
    if not xs:
        raise ValueError(f"Split {split_name} is empty; cannot write NPZ.")
    ensure_dir(out_dir)
    np.savez_compressed(
        out_dir / f"{split_name}.npz",
        x=np.stack(xs).astype(np.float32),
        y=np.asarray(ys, dtype=np.int64),
        subjects=np.asarray(subjects, dtype=np.int64),
        sources=np.asarray(list(sources), dtype=object),
    )


def persist_dataset(
    dataset_name: str,
    samples: Sequence[Tuple[np.ndarray, int, int, str]],
    out_root: Path,
    seed: int,
) -> Dict[str, object]:
    if not samples:
        raise ValueError(f"No samples were prepared for {dataset_name}.")

    split_subjects = allocate_subject_splits([subject for _, _, subject, _ in samples], seed)
    out_dir = out_root / dataset_name
    split_counts: Dict[str, int] = {}
    label_counts: Dict[str, int] = {}

    for split_name, allowed_subjects in split_subjects.items():
        split_samples = [sample for sample in samples if sample[2] in allowed_subjects]
        xs = [sample[0] for sample in split_samples]
        ys = [sample[1] for sample in split_samples]
        subjects = [sample[2] for sample in split_samples]
        sources = [sample[3] for sample in split_samples]
        write_split_npz(out_dir, split_name, xs, ys, subjects, sources)
        split_counts[split_name] = len(split_samples)

    for _, label, _, _ in samples:
        key = str(int(label))
        label_counts[key] = label_counts.get(key, 0) + 1

    summary = {
        "dataset": dataset_name,
        "num_samples": len(samples),
        "subjects": sorted({int(subject) for _, _, subject, _ in samples}),
        "split_counts": split_counts,
        "label_counts": label_counts,
        "output_dir": str(out_dir.resolve()),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_mobiact_acc_file(path: Path, target_len: int, label: int) -> np.ndarray:
    rows: List[List[float]] = []
    in_data = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("@DATA"):
            in_data = True
            continue
        if not in_data or line.startswith("#"):
            continue
        values = [part.strip() for part in line.split(",")]
        if len(values) < 4:
            continue
        rows.append([float(values[1]), float(values[2]), float(values[3])])
    if not rows:
        raise ValueError(f"No accelerometer rows found in {path}")
    x = np.asarray(rows, dtype=np.float32).T
    if int(label) == 1:
        return crop_or_pad_window(x, target_len=target_len, center_idx=estimate_peak_center(x))
    return extract_fixed_center_window(x, target_len)


def maybe_download_mobiact(source_root: Optional[Path], cache_root: Path) -> Path:
    if source_root and source_root.is_dir():
        return source_root

    ensure_dir(cache_root)
    extracted_root = cache_root / "MobiAct_Dataset_v2.0-MobiFall_Dataset_v2.0-main"
    if extracted_root.is_dir():
        return extracted_root

    archive_path = cache_root / "mobiact_main.zip"
    logging.info("Downloading MobiAct dataset from GitHub to %s", archive_path)
    urllib.request.urlretrieve(MOBIACT_GITHUB_ZIP_URL, archive_path)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(cache_root)
    return extracted_root


def prepare_mobiact_dataset(source_root: Path, out_root: Path, target_len: int, seed: int) -> Dict[str, object]:
    samples: List[Tuple[np.ndarray, int, int, str]] = []
    pattern = re.compile(r"^(?P<code>[A-Z]+)_acc_(?P<subject>\d+)_(?P<trial>\d+)\.txt$")

    for path in sorted(source_root.rglob("*_acc_*_*.txt")):
        match = pattern.match(path.name)
        if not match:
            continue
        code = match.group("code")
        if code in MOBIACT_FALL_CODES:
            label = 1
        elif code in MOBIACT_ADL_CODES:
            label = 0
        else:
            continue
        subject = int(match.group("subject"))
        samples.append((parse_mobiact_acc_file(path, target_len, label), label, subject, str(path)))

    return persist_dataset("mobiact", samples, out_root, seed)


def _load_npy_from_zip(zf: zipfile.ZipFile, member_name: str) -> np.ndarray:
    with zf.open(member_name) as f:
        return np.load(io.BytesIO(f.read()), allow_pickle=False)


def _resolve_unimib_arrays_from_dir(source_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    data_path = source_root / "two_classes_data.npy"
    label_path = source_root / "two_classes_labels.npy"
    if not data_path.is_file() or not label_path.is_file():
        raise FileNotFoundError("UniMiB source directory must contain two_classes_data.npy and two_classes_labels.npy")
    return np.load(data_path), np.load(label_path)


def _resolve_unimib_arrays_from_archive(archive_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with zipfile.ZipFile(archive_path) as zf:
        return (
            _load_npy_from_zip(zf, "SCI666/data/UniMiB_SHAR/two_classes_data.npy"),
            _load_npy_from_zip(zf, "SCI666/data/UniMiB_SHAR/two_classes_labels.npy"),
        )


def prepare_unimib_dataset(source: Path, out_root: Path, target_len: int, seed: int) -> Dict[str, object]:
    if source.is_dir():
        data, labels = _resolve_unimib_arrays_from_dir(source)
    else:
        data, labels = _resolve_unimib_arrays_from_archive(source)

    if data.ndim != 2:
        raise ValueError(f"Unexpected UniMiB data shape: {data.shape}")
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError(f"Unexpected UniMiB label shape: {labels.shape}")

    channels = data.reshape(data.shape[0], 3, -1)
    raw_labels = labels[:, 0].astype(np.int64)
    if set(np.unique(raw_labels).tolist()) == {1, 2}:
        ys = (raw_labels == 2).astype(np.int64)
    else:
        ys = raw_labels.astype(np.int64)
    subjects = labels[:, 1].astype(np.int64)

    samples = [
        (resample_channels(channels[idx], target_len), int(ys[idx]), int(subjects[idx]), f"unimib:{idx}")
        for idx in range(channels.shape[0])
    ]
    return persist_dataset("unimib", samples, out_root, seed)


def _parse_kfall_csv_bytes(blob: bytes) -> np.ndarray:
    df = pd.read_csv(io.BytesIO(blob))
    required = ["AccX", "AccY", "AccZ"]
    if not all(column in df.columns for column in required):
        raise ValueError(f"KFall csv missing required columns: {required}")
    return df[required].to_numpy(dtype=np.float32).T


def _parse_kfall_task_id(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    match = re.search(r"\((\d+)\)", str(value))
    if match:
        return int(match.group(1))
    return None


def _load_kfall_annotations_from_dir(source_root: Path) -> Dict[Tuple[int, int, int], Tuple[Optional[float], Optional[float]]]:
    mapping: Dict[Tuple[int, int, int], Tuple[Optional[float], Optional[float]]] = {}
    for path in sorted(source_root.rglob("*_label.xlsx")):
        match = re.search(r"SA(\d+)_label\.xlsx$", path.name)
        if not match:
            continue
        subject = int(match.group(1))
        df = pd.read_excel(path)
        df["Task Code (Task ID)"] = df["Task Code (Task ID)"].ffill()
        for _, row in df.iterrows():
            task_id = _parse_kfall_task_id(row.get("Task Code (Task ID)"))
            trial_id = row.get("Trial ID")
            if task_id is None or pd.isna(trial_id):
                continue
            mapping[(subject, task_id, int(trial_id))] = (
                None if pd.isna(row.get("Fall_onset_frame")) else float(row.get("Fall_onset_frame")),
                None if pd.isna(row.get("Fall_impact_frame")) else float(row.get("Fall_impact_frame")),
            )
    return mapping


def _load_kfall_annotations_from_archive(source: Path) -> Dict[Tuple[int, int, int], Tuple[Optional[float], Optional[float]]]:
    mapping: Dict[Tuple[int, int, int], Tuple[Optional[float], Optional[float]]] = {}
    with zipfile.ZipFile(source) as zf:
        members = [name for name in zf.namelist() if name.endswith("_label.xlsx") and "label_data/" in name]
        for member in members:
            match = re.search(r"SA(\d+)_label\.xlsx$", member)
            if not match:
                continue
            subject = int(match.group(1))
            with zf.open(member) as f:
                df = pd.read_excel(io.BytesIO(f.read()))
            df["Task Code (Task ID)"] = df["Task Code (Task ID)"].ffill()
            for _, row in df.iterrows():
                task_id = _parse_kfall_task_id(row.get("Task Code (Task ID)"))
                trial_id = row.get("Trial ID")
                if task_id is None or pd.isna(trial_id):
                    continue
                mapping[(subject, task_id, int(trial_id))] = (
                    None if pd.isna(row.get("Fall_onset_frame")) else float(row.get("Fall_onset_frame")),
                    None if pd.isna(row.get("Fall_impact_frame")) else float(row.get("Fall_impact_frame")),
                )
    return mapping


def _iter_kfall_dir_samples(source_root: Path, target_len: int) -> Iterable[Tuple[np.ndarray, int, int, str]]:
    pattern = re.compile(r"^S(?P<subject>\d+)T(?P<task>\d+)R(?P<trial>\d+)\.csv$")
    annotations = _load_kfall_annotations_from_dir(source_root)
    for path in sorted(source_root.rglob("*.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        task_id = int(match.group("task"))
        subject = int(match.group("subject"))
        trial_id = int(match.group("trial"))
        label = 1 if task_id in KFALL_FALL_TASK_IDS else 0
        onset_frame, impact_frame = annotations.get((subject, task_id, trial_id), (None, None))
        x = _parse_kfall_csv_bytes(path.read_bytes())
        yield (extract_kfall_window(x, target_len, label, onset_frame, impact_frame), label, subject, str(path))


def _iter_kfall_archive_samples(source: Path, target_len: int) -> Iterable[Tuple[np.ndarray, int, int, str]]:
    pattern = re.compile(r"sensor_data/SA(?P<subject>\d+)/S(?P=subject)T(?P<task>\d+)R(?P<trial>\d+)\.csv$")
    annotations = _load_kfall_annotations_from_archive(source)
    with zipfile.ZipFile(source) as zf:
        for member in sorted(zf.namelist()):
            if not member.endswith(".csv") or "sensor_data/" not in member:
                continue
            match = pattern.search(member)
            if not match:
                continue
            task_id = int(match.group("task"))
            subject = int(match.group("subject"))
            trial_id = int(match.group("trial"))
            label = 1 if task_id in KFALL_FALL_TASK_IDS else 0
            onset_frame, impact_frame = annotations.get((subject, task_id, trial_id), (None, None))
            with zf.open(member) as f:
                x = _parse_kfall_csv_bytes(f.read())
                yield (extract_kfall_window(x, target_len, label, onset_frame, impact_frame), label, subject, member)


def prepare_kfall_dataset(source: Path, out_root: Path, target_len: int, seed: int) -> Dict[str, object]:
    iterator = _iter_kfall_dir_samples(source, target_len) if source.is_dir() else _iter_kfall_archive_samples(source, target_len)
    samples = list(iterator)
    return persist_dataset("kfall", samples, out_root, seed)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare MobiAct / UniMiB / KFall into NPZ splits for PhyCL-Net.")
    parser.add_argument("--out-root", default="./data", help="Directory that will receive mobiact/unimib/kfall folders")
    parser.add_argument("--datasets", nargs="+", default=["mobiact", "unimib", "kfall"], choices=["mobiact", "unimib", "kfall"])
    parser.add_argument("--target-len", type=int, default=512, help="Length used after per-sample resampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subject-disjoint split allocation")
    parser.add_argument("--mobiact-root", default=None, help="Existing MobiAct root directory; download from GitHub when absent")
    parser.add_argument("--mobiact-cache-root", default=None, help="Cache directory for downloaded MobiAct archive")
    parser.add_argument("--unimib-root", default=None, help="UniMiB directory containing two_classes_data.npy / two_classes_labels.npy")
    parser.add_argument("--kfall-root", default=None, help="KFall extracted dataset root containing sensor_data/")
    parser.add_argument("--archive-path", default=None, help="Local archive containing SCI666/data/UniMiB_SHAR and/or KFall")
    parser.add_argument("--summary-path", default=None, help="Optional JSON path for the combined summary")
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out_root = Path(args.out_root).resolve()
    ensure_dir(out_root)

    summaries: Dict[str, object] = {}
    archive_path = Path(args.archive_path).resolve() if args.archive_path else None

    if "mobiact" in args.datasets:
        cache_root = Path(args.mobiact_cache_root).resolve() if args.mobiact_cache_root else out_root / "_sources" / "mobiact"
        source_root = Path(args.mobiact_root).resolve() if args.mobiact_root else None
        mobiact_root = maybe_download_mobiact(source_root, cache_root)
        summaries["mobiact"] = prepare_mobiact_dataset(mobiact_root, out_root, args.target_len, args.seed)

    if "unimib" in args.datasets:
        if args.unimib_root:
            unimib_source = Path(args.unimib_root).resolve()
        elif archive_path:
            unimib_source = archive_path
        else:
            parser.error("UniMiB preparation requires --unimib-root or --archive-path")
        summaries["unimib"] = prepare_unimib_dataset(unimib_source, out_root, args.target_len, args.seed)

    if "kfall" in args.datasets:
        if args.kfall_root:
            kfall_source = Path(args.kfall_root).resolve()
        elif archive_path:
            kfall_source = archive_path
        else:
            parser.error("KFall preparation requires --kfall-root or --archive-path")
        summaries["kfall"] = prepare_kfall_dataset(kfall_source, out_root, args.target_len, args.seed)

    summary_path = Path(args.summary_path).resolve() if args.summary_path else out_root / "prepared_dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
