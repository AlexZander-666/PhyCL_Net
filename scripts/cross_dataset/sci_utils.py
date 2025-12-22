import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)
from typing import Dict, List, Tuple, Any

# Optional torch imports are guarded so the module can be imported before torch is installed.
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
except ImportError:  # pragma: no cover - used for friendly erroring before torch is installed
    torch = None
    nn = None
    optim = None
    DataLoader = None
    random_split = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for model evaluation/fine-tuning. Please install torch first."
        )


def calculate_sci_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray) -> Dict[str, str]:
    """Compute the SCI-oriented metrics (acc, prec, rec, spec, F1, TPR@1%FPR)."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    target_fpr = 0.01
    idx = np.argmin(np.abs(fpr - target_fpr))
    tpr_at_1_fpr = tpr[idx]

    return {
        "Accuracy": f"{acc:.4f}",
        "Precision": f"{prec:.4f}",
        "Recall (Sens)": f"{rec:.4f}",
        "Specificity": f"{specificity:.4f}",
        "F1-Score": f"{f1:.4f}",
        "TPR@1%FPR": f"{tpr_at_1_fpr:.4f}",
    }


def print_sci_table(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Print a Markdown table ready for reporting and return the DataFrame."""
    df = pd.DataFrame(results_list)
    print("\n" + "=" * 50)
    print(" >>> CROSS-DATASET GENERALIZATION RESULTS <<<")
    print("=" * 50)
    print(df.to_markdown(index=False))
    print("=" * 50 + "\n")
    return df


def inspect_model_requirements(model: Any) -> Tuple[int, int]:
    """Inspect the first conv layer to infer expected (length, channels)."""
    _require_torch()
    print("Inspecting model architecture...")
    first_layer = None
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            first_layer = module
            break

    if first_layer is None:
        print("[Warning] Could not find Conv layer. Defaulting to Length=128, Channels=3.")
        return 128, 3

    w_shape = first_layer.weight.shape
    print(f"  First Layer Weight Shape: {w_shape}")

    if len(w_shape) == 4:
        req_channels = w_shape[1]
        req_length = 200
    else:
        req_channels = w_shape[1]
        req_length = 200

    # For single-channel convs, default to keeping full sensor data (acc+gyro) = 6.
    if req_channels <= 1:
        req_channels = 6

    print(f"  > Detected/Default Requirements: Length={req_length}, Channels={req_channels}")
    return req_length, req_channels


def evaluate(model: Any, dataloader: Any, device: Any, desc: str = "Evaluating") -> Dict[str, str]:
    """Run inference over a dataloader and compute metrics."""
    _require_torch()
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return calculate_sci_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))


def fine_tune(model: Any, dataset: Any, device: Any, epochs: int = 5) -> Dict[str, str]:
    """Freeze backbone, train head on a small support set, then evaluate."""
    _require_torch()
    if random_split is None or DataLoader is None:
        raise ImportError("torch.utils.data is required for fine-tuning.")

    print("  > Triggering Fine-tuning (Transfer Learning)...")
    train_size = max(1, int(0.2 * len(dataset)))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True
    elif hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

    return evaluate(model, test_loader, device, desc="Eval after FT")
