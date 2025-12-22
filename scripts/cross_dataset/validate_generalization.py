import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    nn = None
    DataLoader = None

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from datasets_cross_validation import KFallDataset, MobiFallDataset, UniMiBDataset
from sci_utils import (
    inspect_model_requirements,
    evaluate,
    fine_tune,
    print_sci_table,
)

MODEL_PATH = ROOT_DIR / "logs" / "best_model.pth"
DATA_ROOT = ROOT_DIR / "data"


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required. Please install torch before running this script.")


def _load_model(num_classes: int = 2) -> Any:
    """Load LiteAMSNet from model.py if available, otherwise use a lightweight placeholder."""
    _require_torch()
    try:
        from model import LiteAMSNet  # type: ignore

        model = LiteAMSNet(num_classes=num_classes)
        print("Loaded LiteAMSNet from model.py")
    except Exception as exc:
        print(f"[Warn] Could not import LiteAMSNet from model.py ({exc}). Using a placeholder model.")

        class LiteAMSNet(nn.Module):
            def __init__(self, num_classes: int = 2):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=(9, 3)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.classifier = nn.Linear(32, num_classes)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.features(x)
                x = x.flatten(1)
                return self.classifier(x)

        model = LiteAMSNet(num_classes=num_classes)

    return model


def run_validation() -> None:
    _require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    model = _load_model().to(device)
    if MODEL_PATH.exists():
        try:
            model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
            print(f"Loaded weights from {MODEL_PATH.as_posix()}")
        except Exception as exc:
            print(f"[Warn] Weight loading failed ({exc}). Using random weights for demo.")
    else:
        print("[Warn] No weights found. Using randomly initialized model.")

    target_length, target_channels = inspect_model_requirements(model)

    datasets: List[tuple[str, Any]] = [
        ("MobiFall", MobiFallDataset(str(DATA_ROOT / "MobiFall_Dataset_v2.0"), target_length, target_channels)),
        ("UniMiB", UniMiBDataset(str(DATA_ROOT / "UniMiB_SHAR"), target_length, target_channels)),
        ("KFall", KFallDataset(str(DATA_ROOT / "KFall"), target_length, target_channels)),
    ]

    results: List[Dict[str, Any]] = []

    for name, ds in datasets:
        if len(ds) == 0:
            print(f"Skipping {name} (No data loaded).")
            continue

        print(f"\nProcessing {name}...")
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        metrics = evaluate(model, loader, device, desc=f"{name} Zero-shot")
        metrics["Dataset"] = name
        metrics["Method"] = "Zero-shot"
        results.append(metrics)
        print(f"  Zero-shot F1: {metrics['F1-Score']}")

        if float(metrics["F1-Score"]) < 0.8 and len(ds) > 32:
            ft_metrics = fine_tune(copy.deepcopy(model), ds, device, epochs=5)
            ft_metrics["Dataset"] = name
            ft_metrics["Method"] = "Transfer (5-shot)"
            results.append(ft_metrics)
            print(f"  Transfer F1: {ft_metrics['F1-Score']}")

    if results:
        df = print_sci_table(results)
        logs_dir = ROOT_DIR / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        out_csv = logs_dir / "cross_dataset_results.csv"
        df.to_csv(str(out_csv), index=False)
        print(f"Results saved to {out_csv.as_posix()}")
    else:
        print("No datasets were processed. Check your data paths.")


if __name__ == "__main__":
    run_validation()
