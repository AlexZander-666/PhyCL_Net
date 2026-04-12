from pathlib import Path
import importlib.util
import io
import sys
import zipfile

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
SCRIPTS_DIR = CODE_DIR / "scripts"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def _load_script_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_mobiact_acc_file(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{idx}, {x}, {y}, {z}" for idx, (x, y, z) in enumerate(rows, start=1))
    path.write_text(
        "#Acceleration force along the x y z axes (including gravity).\n"
        "#timestamp(ns),x,y,z(m/s^2)\n"
        "\n"
        "@DATA\n"
        f"{body}\n",
        encoding="utf-8",
    )


def _write_kfall_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "TimeStamp(s),FrameCounter,AccX,AccY,AccZ,GyrX,GyrY,GyrZ,EulerX,EulerY,EulerZ\n"
    lines = [
        f"{i/100:.2f},{i},{x},{y},{z},0,0,0,0,0,0"
        for i, (x, y, z) in enumerate(rows, start=1)
    ]
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def test_prepare_mobiact_dataset_writes_binary_npz_splits(tmp_path):
    script_path = SCRIPTS_DIR / "prepare_cross_dataset_npz.py"
    module = _load_script_module("prepare_cross_dataset_npz_mobiact", script_path)

    raw_root = tmp_path / "mobiact_raw"
    subjects = {
        1: ("ADL", "WAL"),
        2: ("FALLS", "FOL"),
        3: ("ADL", "CSI"),
        4: ("FALLS", "SDL"),
        5: ("ADL", "STU"),
        6: ("FALLS", "FKL"),
    }
    for subject_id, (bucket, code) in subjects.items():
        rows = [(subject_id + i, subject_id + i + 1, subject_id + i + 2) for i in range(6)]
        name = f"{code}_acc_{subject_id}_1.txt"
        _write_mobiact_acc_file(raw_root / f"sub{subject_id}" / bucket / code / name, rows)

    out_dir = tmp_path / "prepared"
    summary = module.prepare_mobiact_dataset(
        source_root=raw_root,
        out_root=out_dir,
        target_len=8,
        seed=7,
    )

    assert summary["dataset"] == "mobiact"
    assert summary["num_samples"] == 6

    split_subjects = {}
    for split in ("train", "val", "test"):
        payload = np.load(out_dir / "mobiact" / f"{split}.npz")
        assert payload["x"].shape[1:] == (3, 8)
        assert set(np.unique(payload["y"]).tolist()).issubset({0, 1})
        split_subjects[split] = set(payload["subjects"].tolist())

    assert split_subjects["train"]
    assert split_subjects["val"]
    assert split_subjects["test"]
    assert split_subjects["train"].isdisjoint(split_subjects["val"])
    assert split_subjects["train"].isdisjoint(split_subjects["test"])
    assert split_subjects["val"].isdisjoint(split_subjects["test"])


def test_prepare_unimib_dataset_from_archive_uses_two_class_labels(tmp_path):
    script_path = SCRIPTS_DIR / "prepare_cross_dataset_npz.py"
    module = _load_script_module("prepare_cross_dataset_npz_unimib", script_path)

    archive_path = tmp_path / "datasets.zip"
    data = np.array(
        [
            np.arange(12, dtype=np.float32),
            np.arange(12, dtype=np.float32) + 10,
            np.arange(12, dtype=np.float32) + 20,
            np.arange(12, dtype=np.float32) + 30,
            np.arange(12, dtype=np.float32) + 40,
            np.arange(12, dtype=np.float32) + 50,
        ]
    )
    labels = np.array(
        [
            [1, 1, 1],
            [2, 2, 1],
            [1, 3, 1],
            [2, 4, 1],
            [1, 5, 1],
            [2, 6, 1],
        ],
        dtype=np.uint8,
    )
    with zipfile.ZipFile(archive_path, "w") as zf:
        data_buf = io.BytesIO()
        np.save(data_buf, data)
        zf.writestr("SCI666/data/UniMiB_SHAR/two_classes_data.npy", data_buf.getvalue())

        label_buf = io.BytesIO()
        np.save(label_buf, labels)
        zf.writestr("SCI666/data/UniMiB_SHAR/two_classes_labels.npy", label_buf.getvalue())

    out_dir = tmp_path / "prepared"
    summary = module.prepare_unimib_dataset(
        source=archive_path,
        out_root=out_dir,
        target_len=10,
        seed=11,
    )

    assert summary["dataset"] == "unimib"
    assert summary["num_samples"] == 6

    payload = np.load(out_dir / "unimib" / "test.npz")
    assert payload["x"].shape[1:] == (3, 10)
    assert set(np.unique(payload["y"]).tolist()).issubset({0, 1})


def test_prepare_kfall_dataset_from_archive_labels_fall_tasks(tmp_path):
    script_path = SCRIPTS_DIR / "prepare_cross_dataset_npz.py"
    module = _load_script_module("prepare_cross_dataset_npz_kfall", script_path)

    archive_path = tmp_path / "kfall.zip"
    label_df = pd.DataFrame(
        {
            "Task Code (Task ID)": ["F01 (20)"],
            "Description": ["Forward fall when trying to sit down"],
            "Trial ID": [1],
            "Fall_onset_frame": [3],
            "Fall_impact_frame": [5],
        }
    )

    with zipfile.ZipFile(archive_path, "w") as zf:
        for subject_id in range(6, 12):
            task_id = 20 if subject_id % 2 == 0 else 5
            csv_path = f"SCI666/data/KFall/KFall Dataset/sensor_data/SA{subject_id:02d}/S{subject_id:02d}T{task_id:02d}R01.csv"
            rows = "\n".join(
                f"{i/100:.2f},{i},{subject_id+i},{subject_id+i+1},{subject_id+i+2},0,0,0,0,0,0"
                for i in range(1, 7)
            )
            zf.writestr(
                csv_path,
                "TimeStamp(s),FrameCounter,AccX,AccY,AccZ,GyrX,GyrY,GyrZ,EulerX,EulerY,EulerZ\n" + rows + "\n",
            )

            if subject_id == 6:
                xlsx_buf = io.BytesIO()
                label_df.to_excel(xlsx_buf, index=False)
                zf.writestr(
                    "SCI666/data/KFall/KFall Dataset/label_data/SA06_label.xlsx",
                    xlsx_buf.getvalue(),
                )

    out_dir = tmp_path / "prepared"
    summary = module.prepare_kfall_dataset(
        source=archive_path,
        out_root=out_dir,
        target_len=9,
        seed=19,
    )

    assert summary["dataset"] == "kfall"
    assert summary["num_samples"] == 6

    total_positive = 0
    for split in ("train", "val", "test"):
        payload = np.load(out_dir / "kfall" / f"{split}.npz")
        assert payload["x"].shape[1:] == (3, 9)
        assert set(np.unique(payload["y"]).tolist()).issubset({0, 1})
        total_positive += int(payload["y"].sum())
    assert total_positive >= 1


def test_mobiact_fall_window_uses_acceleration_peak_as_center(tmp_path):
    script_path = SCRIPTS_DIR / "prepare_cross_dataset_npz.py"
    module = _load_script_module("prepare_cross_dataset_npz_mobiact_peak", script_path)

    path = tmp_path / "sub1" / "FALLS" / "FOL" / "FOL_acc_1_1.txt"
    rows = [(0.0, 0.0, 0.0)] * 20
    rows[15] = (50.0, 0.0, 0.0)
    _write_mobiact_acc_file(path, rows)

    window = module.parse_mobiact_acc_file(path, target_len=5, label=1)

    assert window.shape == (3, 5)
    assert np.argmax(window[0]) == 2
    assert float(window[0, 2]) == 50.0


def test_kfall_fall_window_uses_onset_impact_midpoint(tmp_path):
    script_path = SCRIPTS_DIR / "prepare_cross_dataset_npz.py"
    module = _load_script_module("prepare_cross_dataset_npz_kfall_center", script_path)

    rows = np.zeros((3, 20), dtype=np.float32)
    rows[0, 7] = 7.0
    rows[0, 8] = 8.0
    rows[0, 9] = 99.0
    rows[0, 10] = 10.0

    window = module.extract_kfall_window(
        rows,
        target_len=5,
        label=1,
        onset_frame=8,
        impact_frame=12,
    )

    assert window.shape == (3, 5)
    assert np.argmax(window[0]) == 1
    assert float(window[0, 1]) == 99.0
