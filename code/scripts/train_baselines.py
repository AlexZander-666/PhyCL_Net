"""
train_baselines.py - Baseline Models Training Script for SCI Paper Comparison

Implements two lightweight baseline models for comparison with AMSNetV2:
1. Standard LSTM (2-layer, bidirectional)
2. ResNet-1D (simplified ResNet-18 structure)

Usage:
    python scripts/train_baselines.py --data-root ./data
    python scripts/train_baselines.py --data-root ./data --epochs 50
"""

import os
import sys
import logging
import argparse
import random
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


# ========================= Utility Functions =========================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")
    return device


# ========================= Model Definitions =========================

class StandardLSTM(nn.Module):
    """
    Standard LSTM Classifier for fall detection.

    Architecture:
    - Input: [Batch, 3, 512] -> permuted to [Batch, 512, 3]
    - 2-layer Bidirectional LSTM (hidden_dim=64)
    - Last hidden state -> Linear classifier
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64,
                 num_layers: int = 2, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Bidirectional -> hidden_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)

        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [Batch, 3, 512]
        Returns:
            logits: Output tensor of shape [Batch, num_classes]
        """
        # Permute: [B, C, L] -> [B, L, C]
        x = x.permute(0, 2, 1)

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state from both directions
        # h_n shape: [num_layers * 2, batch, hidden_dim]
        h_forward = h_n[-2, :, :]  # Last layer forward
        h_backward = h_n[-1, :, :]  # Last layer backward
        h_combined = torch.cat([h_forward, h_backward], dim=1)

        # Classifier
        logits = self.fc(h_combined)
        return logits


class ResidualBlock1D(nn.Module):
    """Basic residual block for 1D convolutions."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    """
    Simplified ResNet-18 for 1D time-series classification.

    Architecture:
    - Input: [Batch, 3, 512]
    - Stem: Conv1d(3 -> 64) + BN + ReLU + MaxPool
    - Layer1: 2x ResidualBlock (64 channels)
    - Layer2: 2x ResidualBlock (128 channels, stride=2)
    - Layer3: 2x ResidualBlock (256 channels, stride=2)
    - Global Average Pooling -> Linear classifier
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Residual layers
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

        self._init_weights()

    def _make_layer(self, in_channels: int, out_channels: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [Batch, 3, 512]
        Returns:
            logits: Output tensor of shape [Batch, num_classes]
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        logits = self.fc(x)
        return logits


# ========================= Dataset =========================

class SisFallDataset(Dataset):
    """
    SisFall dataset loader for accelerometer data.
    Falls back to synthetic data if real data is not found.
    """

    def __init__(self, data_root: str, window_size: int = 512, stride: int = 256,
                 split: str = 'train', val_ratio: float = 0.15, test_ratio: float = 0.15,
                 seed: int = 42):
        super().__init__()
        self.items: List[Tuple[np.ndarray, int]] = []
        self.window_size = window_size
        self.stride = stride
        self.split = split

        # Try to load real data
        sisfall_root = self._resolve_sisfall_root(data_root)

        if sisfall_root is not None:
            logging.info(f"Loading SisFall data from {sisfall_root}")
            self._load_sisfall(sisfall_root, val_ratio, test_ratio, seed)
        else:
            logging.warning(f"SisFall data not found at {data_root}, using synthetic data")
            self._generate_synthetic(seed)

    def _resolve_sisfall_root(self, data_root: str) -> Optional[str]:
        """Find SisFall directory with ADL/FALL subdirectories."""
        candidates = [
            os.path.abspath(data_root),
            os.path.join(os.path.abspath(data_root), 'SisFall'),
        ]

        for candidate in candidates:
            if not os.path.isdir(candidate):
                continue
            if all(os.path.isdir(os.path.join(candidate, sub)) for sub in ('ADL', 'FALL')):
                return candidate
        return None

    def _load_sisfall(self, root: str, val_ratio: float, test_ratio: float, seed: int):
        """Load real SisFall data."""
        all_items = []

        for folder, label in [('ADL', 0), ('FALL', 1)]:
            folder_path = os.path.join(root, folder)
            if not os.path.isdir(folder_path):
                continue

            for fname in os.listdir(folder_path):
                if not fname.lower().endswith('.txt'):
                    continue

                file_path = os.path.join(folder_path, fname)
                windows = self._parse_file(file_path, label)
                all_items.extend(windows)

        if not all_items:
            logging.warning("No valid data files found, using synthetic data")
            self._generate_synthetic(seed)
            return

        # Shuffle and split
        rng = random.Random(seed)
        rng.shuffle(all_items)

        n = len(all_items)
        test_start = int(n * (1 - test_ratio))
        val_start = int(n * (1 - test_ratio - val_ratio))

        if self.split == 'train':
            self.items = all_items[:val_start]
        elif self.split == 'val':
            self.items = all_items[val_start:test_start]
        else:  # test
            self.items = all_items[test_start:]

        logging.info(f"Loaded {len(self.items)} {self.split} samples "
                    f"(ADL: {sum(1 for _, l in self.items if l == 0)}, "
                    f"Fall: {sum(1 for _, l in self.items if l == 1)})")

    def _parse_file(self, file_path: str, label: int) -> List[Tuple[np.ndarray, int]]:
        """Parse a single SisFall txt file into windows."""
        windows = []
        try:
            data_rows = []
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip().rstrip(';')
                    if not line:
                        continue
                    line = line.replace(';', ',')
                    values = [v.strip() for v in line.split(',') if v.strip()]
                    if len(values) < 3:
                        continue
                    try:
                        nums = [float(v) for v in values[:3]]  # Use first 3 channels
                        data_rows.append(nums)
                    except ValueError:
                        continue

            if not data_rows:
                return windows

            # Convert to array: [L, C] -> [C, L]
            raw_data = np.array(data_rows, dtype=np.float32).T

            # Normalize per-channel
            mean = raw_data.mean(axis=1, keepdims=True)
            std = raw_data.std(axis=1, keepdims=True) + 1e-6
            norm_data = (raw_data - mean) / std

            # Extract windows
            c, l = norm_data.shape
            start = 0
            while start + self.window_size <= l:
                window = norm_data[:, start:start + self.window_size].copy()
                windows.append((window, label))
                start += self.stride

            # Pad last window if needed
            if l >= self.window_size // 2 and start < l:
                pad = np.zeros((c, self.window_size), dtype=np.float32)
                remaining = l - start
                pad[:, :remaining] = norm_data[:, start:]
                windows.append((pad, label))

        except Exception as e:
            logging.warning(f"Error parsing {file_path}: {e}")

        return windows

    def _generate_synthetic(self, seed: int):
        """Generate synthetic data for testing."""
        rng = np.random.default_rng(seed)

        if self.split == 'train':
            n_samples = 400
        elif self.split == 'val':
            n_samples = 100
        else:
            n_samples = 100

        for i in range(n_samples):
            label = i % 2  # Balanced classes
            x = rng.standard_normal((3, self.window_size)).astype(np.float32)

            # Add class-specific patterns
            if label == 1:  # Fall
                # Add spike pattern for falls
                spike_pos = rng.integers(self.window_size // 4, 3 * self.window_size // 4)
                spike_width = 20
                x[:, spike_pos:spike_pos + spike_width] += rng.uniform(2, 4)

            self.items.append((x, label))

        logging.info(f"Generated {len(self.items)} synthetic {self.split} samples")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x, y = self.items[idx]
        return torch.from_numpy(x), y


def get_dataloaders(data_root: str, batch_size: int = 32,
                    seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""

    train_ds = SisFallDataset(data_root, split='train', seed=seed)
    val_ds = SisFallDataset(data_root, split='val', seed=seed)
    test_ds = SisFallDataset(data_root, split='test', seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


# ========================= Training =========================

def train_and_evaluate(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    save_dir: str = 'outputs'
) -> float:
    """
    Train and evaluate a model.

    Args:
        model_name: Name for saving checkpoint
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device to train on
        epochs: Number of epochs
        lr: Learning rate
        save_dir: Directory to save checkpoints

    Returns:
        best_accuracy: Best test accuracy achieved
    """
    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"\n{'='*50}")
    logging.info(f"Training {model_name}")
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info(f"{'='*50}")

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += x.size(0)

        scheduler.step()

        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total

        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                test_correct += (preds == y).sum().item()
                test_total += x.size(0)

        test_acc = 100.0 * test_correct / test_total

        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            save_path = os.path.join(save_dir, f'{model_name}_baseline.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
            }, save_path)
            logging.info(f"Epoch {epoch:2d}/{epochs} | Loss: {train_loss:.4f} | "
                        f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% *BEST*")
        else:
            logging.info(f"Epoch {epoch:2d}/{epochs} | Loss: {train_loss:.4f} | "
                        f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    return best_accuracy


def main():
    parser = argparse.ArgumentParser(description='Train baseline models for SCI comparison')
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Root directory for SisFall data')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save-dir', type=str, default='outputs',
                        help='Directory to save checkpoints')

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()

    # Load data
    logging.info(f"Loading data from {args.data_root}")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        seed=args.seed
    )

    # Results storage
    results: Dict[str, float] = {}

    # ==================== Train LSTM ====================
    lstm_model = StandardLSTM(
        in_channels=3,
        hidden_dim=64,
        num_layers=2,
        num_classes=2,
        dropout=0.3
    )

    lstm_acc = train_and_evaluate(
        model_name='LSTM',
        model=lstm_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir
    )
    results['LSTM'] = lstm_acc

    # ==================== Train ResNet-1D ====================
    resnet_model = ResNet1D(
        in_channels=3,
        num_classes=2
    )

    resnet_acc = train_and_evaluate(
        model_name='ResNet',
        model=resnet_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir
    )
    results['ResNet'] = resnet_acc

    # ==================== Print Summary ====================
    print("\n")
    print("=" * 50)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 50)
    print(f"LSTM Best Accuracy:   {results['LSTM']:.2f}%")
    print(f"ResNet Best Accuracy: {results['ResNet']:.2f}%")
    print("=" * 50)
    print(f"\nCheckpoints saved to: {args.save_dir}/")
    print("  - LSTM_baseline.pth")
    print("  - ResNet_baseline.pth")


if __name__ == '__main__':
    main()
