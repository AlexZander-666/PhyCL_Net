import os
from glob import glob
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import signal
import torch
from torch.utils.data import Dataset


class BaseMotionDataset(Dataset):
    """Base dataset with resampling/padding helpers."""

    def __init__(self, root_dir: str, target_length: int, target_channels: int, transform=None, mode: str = "test"):
        self.root_dir = root_dir
        self.target_length = target_length
        self.target_channels = target_channels
        self.transform = transform
        self.data: List[np.ndarray] = []
        self.labels: List[int] = []  # 0 for ADL, 1 for Fall
        self.mode = mode
        self._load_data()

    def _load_data(self) -> None:
        raise NotImplementedError

    def _resample_and_pad(self, sequence: np.ndarray) -> np.ndarray:
        """Ensure each sample matches (target_length, target_channels)."""
        sequence = np.asarray(sequence, dtype=float)
        if sequence.shape[0] < sequence.shape[1]:
            sequence = sequence.T

        curr_len, curr_channels = sequence.shape

        if curr_channels < self.target_channels:
            pad_width = self.target_channels - curr_channels
            sequence = np.pad(sequence, ((0, 0), (0, pad_width)), "constant")
        elif curr_channels > self.target_channels:
            sequence = sequence[:, : self.target_channels]

        if curr_len != self.target_length:
            sequence = signal.resample(sequence, self.target_length, axis=0)

        return sequence

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        label = self.labels[idx]
        sample = self._resample_and_pad(sample)
        sample_tensor = torch.FloatTensor(sample).unsqueeze(0)  # [1, L, C]
        return sample_tensor, label


class MobiFallDataset(BaseMotionDataset):
    """
    MobiFall Dataset Loader (v2.0).
    Expected structure: root/subXX/{FALLS|ADL}/ACTIVITY/*.txt
    """

    def _load_signal(self, path: str) -> np.ndarray:
        try:
            df = pd.read_csv(
                path,
                comment="#",
                header=None,
                names=["ts", "x", "y", "z"],
                engine="python",
            )
            df = df.dropna()
            return df[["x", "y", "z"]].to_numpy(dtype=float)
        except Exception as exc:  # pragma: no cover
            print(f"[Warn] Failed to read {path}: {exc}")
            return np.empty((0, 3))

    def _load_pair(self, acc_path: str, gyro_path: str | None) -> np.ndarray:
        acc = self._load_signal(acc_path)
        gyro = self._load_signal(gyro_path) if gyro_path and os.path.exists(gyro_path) else np.empty((acc.shape[0], 0))
        length = min(len(acc), len(gyro)) if gyro.size else len(acc)
        if length == 0:
            return np.empty((0, 0))
        acc = acc[:length]
        gyro = gyro[:length] if gyro.size else np.empty((length, 0))
        return np.concatenate([acc, gyro], axis=1)

    def _load_data(self) -> None:
        print(f"Loading MobiFall from {self.root_dir}...")
        if not os.path.isdir(self.root_dir):
            print(f"[Warning] MobiFall path not found: {self.root_dir}. Skipping.")
            return

        subjects = sorted(glob(os.path.join(self.root_dir, "sub*")))
        for subject in subjects:
            for class_name, label in (("FALLS", 1), ("ADL", 0)):
                class_dir = os.path.join(subject, class_name)
                if not os.path.isdir(class_dir):
                    continue
                for activity_dir in sorted(glob(os.path.join(class_dir, "*"))):
                    acc_files = sorted(glob(os.path.join(activity_dir, "*_acc_*.txt")))
                    for acc_path in acc_files:
                        gyro_path = acc_path.replace("_acc_", "_gyro_")
                        sample = self._load_pair(acc_path, gyro_path)
                        if sample.size == 0:
                            continue
                        self.data.append(sample)
                        self.labels.append(label)


class UniMiBDataset(BaseMotionDataset):
    """UniMiB SHAR loader using the provided binary two-class npy files."""

    def _load_data(self) -> None:
        print(f"Loading UniMiB from {self.root_dir}...")
        data_path = os.path.join(self.root_dir, "two_classes_data.npy")
        labels_path = os.path.join(self.root_dir, "two_classes_labels.npy")

        if not (os.path.exists(data_path) and os.path.exists(labels_path)):
            print(f"[Warning] UniMiB files not found at {self.root_dir}. Skipping.")
            return

        data = np.load(data_path)
        labels = np.load(labels_path)
        # labels[:,0]: 1=ADL, 2=Fall
        for seq, lbl in zip(data, labels):
            length = seq.shape[0] // 3
            seq = seq.reshape(length, 3)
            label = 1 if int(lbl[0]) == 2 else 0
            self.data.append(seq)
            self.labels.append(label)


class KFallDataset(BaseMotionDataset):
    """KFall dataset loader using label Excel + sensor CSV."""

    def _load_data(self) -> None:
        print(f"Loading KFall from {self.root_dir}...")
        label_dir = os.path.join(self.root_dir, "KFall Dataset", "label_data")
        sensor_dir = os.path.join(self.root_dir, "KFall Dataset", "sensor_data")
        if not (os.path.isdir(label_dir) and os.path.isdir(sensor_dir)):
            print(f"[Warning] KFall path not found. Skipping.")
            return

        label_files = sorted(glob(os.path.join(label_dir, "SA*_label.xlsx")))
        for lbl_path in label_files:
            subject_id = os.path.splitext(os.path.basename(lbl_path))[0].replace("_label", "")
            try:
                df = pd.read_excel(lbl_path)
            except Exception as exc:  # pragma: no cover
                print(f"[Warn] Failed to read {lbl_path}: {exc}")
                continue

            df["Task Code (Task ID)"] = df["Task Code (Task ID)"].ffill()
            df["Description"] = df["Description"].ffill()

            for _, row in df.iterrows():
                task_str = str(row["Task Code (Task ID)"])
                if "(" not in task_str or ")" not in task_str:
                    continue
                task_code = task_str.split()[0]  # e.g., F01
                task_id = int(task_code.replace("F", ""))
                trial_id = int(row["Trial ID"])
                onset = row.get("Fall_onset_frame")
                onset = int(onset) if pd.notna(onset) else None

                csv_name = f"{subject_id.replace('SA','S')}T{task_id:02d}R{trial_id:02d}.csv"
                csv_path = os.path.join(sensor_dir, subject_id, csv_name)
                if not os.path.exists(csv_path):
                    continue

                try:
                    df_sig = pd.read_csv(csv_path)
                    sig = df_sig[["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]].to_numpy(dtype=float)
                except Exception as exc:  # pragma: no cover
                    print(f"[Warn] Failed to parse {csv_path}: {exc}")
                    continue

                if sig.shape[0] == 0:
                    continue

                # Pre-fall segment as ADL (0)
                if onset is not None and onset > 10:
                    pre = sig[:onset]
                    if pre.shape[0] > 0:
                        self.data.append(pre)
                        self.labels.append(0)

                # Full trial as fall (1)
                self.data.append(sig)
                self.labels.append(1)
