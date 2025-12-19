#!/usr/bin/env python
"""
multi_dataset_loso.py - Multi-Dataset LOSO Cross-Validation Experiment

实验目标：通过融合 SisFall (80%) + MobiFall (20%) 数据集，
验证模型的跨数据集泛化能力（Zero-Shot Generalization）。

关键设计：
1. 采样率统一至 50Hz（与 Lite-AMSNet 主实验一致）
2. 12-fold LOSO 基于 SisFall 的 12 个核心受试者
3. MobiFall 始终作为辅助训练数据（不参与 LOSO 划分）
4. 标签映射为二分类：{Fall: 1, ADL: 0}

Usage:
    python code/scripts/multi_dataset_loso.py --data-root ./data --out-dir ./outputs/multi_dataset_loso --epochs 100 --seed 42
"""

import os
import sys
import json
import time
import random
import argparse
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy import signal

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

# ============================================================================
# Configuration Constants
# ============================================================================

# 12 核心受试者（与主实验一致）
CORE_SUBJECTS = [
    'SA01', 'SA02', 'SA04', 'SA05', 'SA06', 'SA09',
    'SA10', 'SA11', 'SA17', 'SA18', 'SA19', 'SA21'
]

# 目标采样率 (Hz)
TARGET_SAMPLE_RATE = 50

# SisFall 原始采样率 (Hz)
SISFALL_SAMPLE_RATE = 200

# MobiFall 原始采样率 (Hz) - 约 87Hz
MOBIFALL_SAMPLE_RATE = 87

# 窗口大小和步长（基于 50Hz）
WINDOW_SIZE = 256  # 约 5.12 秒
STRIDE = 128       # 50% 重叠


# ============================================================================
# Utility Functions
# ============================================================================

def setup_logging(out_dir: str):
    """配置日志系统"""
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(os.path.join(out_dir, 'experiment.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)


def set_seed(seed: int):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resample_signal(data: np.ndarray, orig_rate: float, target_rate: float) -> np.ndarray:
    """
    重采样信号到目标采样率
    
    Args:
        data: shape (channels, length) or (length, channels)
        orig_rate: 原始采样率
        target_rate: 目标采样率
    
    Returns:
        重采样后的数据
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    # 确保是 (channels, length) 格式
    if data.shape[0] > data.shape[1]:
        data = data.T
    
    if orig_rate == target_rate:
        return data
    
    num_samples = int(data.shape[1] * target_rate / orig_rate)
    resampled = signal.resample(data, num_samples, axis=1)
    return resampled.astype(np.float32)


# ============================================================================
# Dataset Classes
# ============================================================================

class SisFallLoader:
    """SisFall 数据集加载器（下采样至 50Hz）"""
    
    def __init__(self, data_root: str, window_size: int = WINDOW_SIZE, stride: int = STRIDE):
        self.data_root = self._resolve_root(data_root)
        self.window_size = window_size
        self.stride = stride
        self.samples = []  # List of (data, label, subject, dataset_name)
        self._load_all()
    
    def _resolve_root(self, data_root: str) -> str:
        """解析 SisFall 根目录"""
        candidates = [
            os.path.join(data_root, 'SisFall'),
            data_root,
        ]
        for c in candidates:
            if os.path.isdir(os.path.join(c, 'ADL')) and os.path.isdir(os.path.join(c, 'FALL')):
                return c
        raise FileNotFoundError(f"Cannot find SisFall dataset in {data_root}")
    
    def _parse_subject(self, filename: str) -> str:
        """从文件名提取受试者 ID"""
        import re
        match = re.search(r'SA(\d+)', filename, re.IGNORECASE)
        if match:
            return f"SA{int(match.group(1)):02d}"
        return "UNKNOWN"
    
    def _load_file(self, filepath: str) -> Optional[np.ndarray]:
        """加载单个 SisFall 文件"""
        try:
            data_rows = []
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip().rstrip(';')
                    if not line:
                        continue
                    line = line.replace(';', ',')
                    values = [v.strip() for v in line.split(',') if v.strip()]
                    if len(values) < 3:
                        continue
                    try:
                        nums = [float(v) for v in values[:3]]  # 只取加速度 xyz
                        data_rows.append(nums)
                    except ValueError:
                        continue
            
            if not data_rows:
                return None
            
            raw_data = np.array(data_rows, dtype=np.float32).T  # (3, length)
            
            # 下采样: 200Hz -> 50Hz
            resampled = resample_signal(raw_data, SISFALL_SAMPLE_RATE, TARGET_SAMPLE_RATE)
            return resampled
        except Exception as e:
            logging.warning(f"Failed to load {filepath}: {e}")
            return None
    
    def _load_all(self):
        """加载所有 SisFall 数据"""
        logging.info(f"Loading SisFall from {self.data_root}...")
        
        for folder, label in [('ADL', 0), ('FALL', 1)]:
            folder_path = os.path.join(self.data_root, folder)
            if not os.path.isdir(folder_path):
                continue
            
            for fname in os.listdir(folder_path):
                if not fname.lower().endswith('.txt'):
                    continue
                
                subject = self._parse_subject(fname)
                filepath = os.path.join(folder_path, fname)
                data = self._load_file(filepath)
                
                if data is None or data.shape[1] < self.window_size:
                    continue
                
                # 滑动窗口分割
                windows = self._sliding_window(data, label, subject)
                self.samples.extend(windows)
        
        logging.info(f"SisFall loaded: {len(self.samples)} windows")
    
    def _sliding_window(self, data: np.ndarray, label: int, subject: str) -> List[Tuple]:
        """滑动窗口分割"""
        windows = []
        length = data.shape[1]
        start = 0
        
        while start + self.window_size <= length:
            window = data[:, start:start + self.window_size].copy()
            # Z-score 标准化
            mean = window.mean(axis=1, keepdims=True)
            std = window.std(axis=1, keepdims=True) + 1e-6
            window = (window - mean) / std
            windows.append((window, label, subject, 'sisfall'))
            start += self.stride
        
        return windows
    
    def get_samples_by_subjects(self, subjects: List[str]) -> List[Tuple]:
        """获取指定受试者的样本"""
        subjects_set = set(s.upper() for s in subjects)
        return [s for s in self.samples if s[2].upper() in subjects_set]


class MobiFallLoader:
    """MobiFall 数据集加载器（下采样至 50Hz）"""
    
    def __init__(self, data_root: str, window_size: int = WINDOW_SIZE, stride: int = STRIDE):
        self.data_root = self._resolve_root(data_root)
        self.window_size = window_size
        self.stride = stride
        self.samples = []
        self._load_all()
    
    def _resolve_root(self, data_root: str) -> str:
        """解析 MobiFall 根目录"""
        candidates = [
            os.path.join(data_root, 'MobiFall_Dataset_v2.0'),
            os.path.join(data_root, 'MobiFall'),
            data_root,
        ]
        for c in candidates:
            if os.path.isdir(c) and any(d.startswith('sub') for d in os.listdir(c) if os.path.isdir(os.path.join(c, d))):
                return c
        raise FileNotFoundError(f"Cannot find MobiFall dataset in {data_root}")
    
    def _load_signal(self, filepath: str) -> Optional[np.ndarray]:
        """加载单个信号文件"""
        try:
            import pandas as pd
            df = pd.read_csv(filepath, comment='#', header=None, 
                           names=['ts', 'x', 'y', 'z'], engine='python')
            df = df.dropna()
            if len(df) < 10:
                return None
            return df[['x', 'y', 'z']].to_numpy(dtype=np.float32).T  # (3, length)
        except Exception as e:
            return None
    
    def _load_all(self):
        """加载所有 MobiFall 数据"""
        logging.info(f"Loading MobiFall from {self.data_root}...")
        
        from glob import glob
        subjects = sorted(glob(os.path.join(self.data_root, 'sub*')))
        
        for subject_dir in subjects:
            subject_id = os.path.basename(subject_dir)
            
            for class_name, label in [('FALLS', 1), ('ADL', 0)]:
                class_dir = os.path.join(subject_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                
                for activity_dir in glob(os.path.join(class_dir, '*')):
                    if not os.path.isdir(activity_dir):
                        continue
                    
                    acc_files = glob(os.path.join(activity_dir, '*_acc_*.txt'))
                    for acc_path in acc_files:
                        data = self._load_signal(acc_path)
                        if data is None:
                            continue
                        
                        # 下采样: 87Hz -> 50Hz
                        resampled = resample_signal(data, MOBIFALL_SAMPLE_RATE, TARGET_SAMPLE_RATE)
                        
                        if resampled.shape[1] < self.window_size:
                            continue
                        
                        # 滑动窗口
                        windows = self._sliding_window(resampled, label, subject_id)
                        self.samples.extend(windows)
        
        logging.info(f"MobiFall loaded: {len(self.samples)} windows")
    
    def _sliding_window(self, data: np.ndarray, label: int, subject: str) -> List[Tuple]:
        """滑动窗口分割"""
        windows = []
        length = data.shape[1]
        start = 0
        
        while start + self.window_size <= length:
            window = data[:, start:start + self.window_size].copy()
            mean = window.mean(axis=1, keepdims=True)
            std = window.std(axis=1, keepdims=True) + 1e-6
            window = (window - mean) / std
            windows.append((window, label, subject, 'mobifall'))
            start += self.stride
        
        return windows
    
    def get_all_samples(self) -> List[Tuple]:
        """获取所有样本"""
        return self.samples


class FusedDataset(Dataset):
    """融合数据集，支持加权采样"""
    
    def __init__(self, samples: List[Tuple], transform=None):
        """
        Args:
            samples: List of (data, label, subject, dataset_name)
            transform: 可选的数据增强
        """
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data, label, subject, dataset = self.samples[idx]
        data = np.array(data, copy=True)
        
        if self.transform is not None:
            data = self.transform(data)
        
        return torch.FloatTensor(data), label, subject, dataset
    
    def get_dataset_weights(self, sisfall_ratio: float = 0.8) -> torch.Tensor:
        """
        计算加权采样权重，使 SisFall:MobiFall = 80:20
        """
        sisfall_count = sum(1 for s in self.samples if s[3] == 'sisfall')
        mobifall_count = len(self.samples) - sisfall_count
        
        if sisfall_count == 0 or mobifall_count == 0:
            return torch.ones(len(self.samples))
        
        # 计算权重使采样比例为 80:20
        sisfall_weight = sisfall_ratio / sisfall_count
        mobifall_weight = (1 - sisfall_ratio) / mobifall_count
        
        weights = []
        for s in self.samples:
            if s[3] == 'sisfall':
                weights.append(sisfall_weight)
            else:
                weights.append(mobifall_weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_class_weights(self, num_classes: int = 2) -> torch.Tensor:
        """计算类别权重用于损失函数"""
        labels = [s[1] for s in self.samples]
        counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
        counts = np.maximum(counts, 1e-6)
        weights = len(labels) / (num_classes * counts)
        return torch.tensor(weights, dtype=torch.float32)


def collate_fn(batch):
    """自定义 collate 函数"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None, None, None, None
    
    data, labels, subjects, datasets = zip(*batch)
    data = torch.stack(data, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return data, labels, list(subjects), list(datasets)


# ============================================================================
# Model Definition (LiteAMSNet)
# ============================================================================

class SimplifiedSpectralBlock(nn.Module):
    """轻量级频域编码器"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            X = torch.fft.rfft(x.float(), dim=-1, norm='ortho')
            mag = torch.abs(X)
            mag = F.interpolate(mag, size=x.size(-1), mode='linear', align_corners=False)
        return self.conv(mag.to(dtype=x.dtype))


class SeparableConv1d(nn.Module):
    """深度可分离卷积"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 15):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, 
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class LiteAMSBlock(nn.Module):
    """轻量级 AMS 块"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.time_branch = SeparableConv1d(channels, channels, kernel_size=15)
        self.freq_branch = SimplifiedSpectralBlock(channels)
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = self.time_branch(x)
        x_f = self.freq_branch(x)
        alpha = torch.sigmoid(self.alpha)
        out = alpha * x_t + (1 - alpha) * x_f
        return x + out


class GhostConv1d(nn.Module):
    """Ghost 卷积（参数高效）"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        init_channels = out_channels // 2
        self.primary = nn.Conv1d(in_channels, init_channels, kernel_size, 
                                padding=kernel_size // 2)
        self.cheap = nn.Conv1d(init_channels, init_channels, 3, padding=1, 
                              groups=init_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
    
    def forward(self, x):
        x1 = self.primary(x)
        x2 = self.cheap(x1)
        out = torch.cat([x1, x2], dim=1)
        return self.act(self.bn(out))


class LiteAMSNet(nn.Module):
    """轻量级 AMS-Net（<100K 参数）"""
    
    def __init__(self, in_channels: int = 3, num_classes: int = 2, 
                 channels: int = 32, n_blocks: int = 2):
        super().__init__()
        self.stem = GhostConv1d(in_channels, channels, kernel_size=5)
        self.blocks = nn.ModuleList([LiteAMSBlock(channels) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, num_classes),
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        if batch[0] is None:
            continue
        
        data, labels, _, _ = batch
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs = model(data)
            loss = criterion(outputs, labels)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate(model, data_loader, device, dataset_filter: str = None) -> Dict[str, Any]:
    """
    评估模型
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        dataset_filter: 可选，只评估特定数据集 ('sisfall' 或 'mobifall')
    
    Returns:
        评估指标字典
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in data_loader:
        if batch[0] is None:
            continue
        
        data, labels, subjects, datasets = batch
        
        # 过滤特定数据集
        if dataset_filter is not None:
            mask = [d == dataset_filter for d in datasets]
            if not any(mask):
                continue
            indices = [i for i, m in enumerate(mask) if m]
            data = data[indices]
            labels = labels[indices]
        
        data = data.to(device)
        outputs = model(data)
        probs = F.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
    
    if len(all_labels) == 0:
        return {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds) * 100,
        'f1': f1_score(all_labels, all_preds, average='macro') * 100,
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100,
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100,
        'n_samples': len(all_labels),
    }
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def run_loso_fold(fold_idx: int, test_subject: str, sisfall_loader: SisFallLoader,
                  mobifall_loader: MobiFallLoader, config: Dict, device: torch.device) -> Dict:
    """
    运行单个 LOSO fold
    
    Args:
        fold_idx: fold 索引
        test_subject: 测试受试者 ID
        sisfall_loader: SisFall 数据加载器
        mobifall_loader: MobiFall 数据加载器
        config: 配置字典
        device: 计算设备
    
    Returns:
        fold 结果字典
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"Fold {fold_idx + 1}/12: Test Subject = {test_subject}")
    logging.info(f"{'='*60}")
    
    # 划分训练/验证集
    train_subjects = [s for s in CORE_SUBJECTS if s != test_subject]
    
    # 随机选择一个作为验证集
    rng = random.Random(config['seed'] + fold_idx)
    val_subject = rng.choice(train_subjects)
    train_subjects = [s for s in train_subjects if s != val_subject]
    
    # 获取 SisFall 样本
    train_sisfall = sisfall_loader.get_samples_by_subjects(train_subjects)
    val_sisfall = sisfall_loader.get_samples_by_subjects([val_subject])
    test_sisfall = sisfall_loader.get_samples_by_subjects([test_subject])
    
    # MobiFall 全部用于训练（作为辅助数据）
    train_mobifall = mobifall_loader.get_all_samples()
    
    # 融合训练数据
    train_samples = train_sisfall + train_mobifall
    
    logging.info(f"Train: {len(train_sisfall)} SisFall + {len(train_mobifall)} MobiFall = {len(train_samples)}")
    logging.info(f"Val: {len(val_sisfall)} SisFall")
    logging.info(f"Test: {len(test_sisfall)} SisFall (held-out subject)")
    
    # 创建数据集
    train_ds = FusedDataset(train_samples)
    val_ds = FusedDataset(val_sisfall)
    test_ds = FusedDataset(test_sisfall)
    
    # 加权采样器（80% SisFall, 20% MobiFall）
    sample_weights = train_ds.get_dataset_weights(sisfall_ratio=0.8)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    
    # 数据加载器
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], sampler=sampler,
                             num_workers=config['num_workers'], collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False,
                           num_workers=config['num_workers'], collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], collate_fn=collate_fn)
    
    # 创建模型
    model = LiteAMSNet(in_channels=3, num_classes=2, channels=32, n_blocks=2).to(device)
    
    # 类别权重
    class_weights = train_ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # AMP
    scaler = torch.amp.GradScaler('cuda') if config['amp'] and device.type == 'cuda' else None
    
    # 训练循环
    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, 
                                                optimizer, scaler, device, epoch)
        scheduler.step()
        
        # 验证
        val_metrics = evaluate(model, val_loader, device)
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logging.info(f"Epoch {epoch+1}/{config['epochs']}: "
                        f"Loss={train_loss:.4f}, TrainAcc={train_acc:.2f}%, "
                        f"ValF1={val_metrics['f1']:.2f}%")
        
        # Early stopping
        if patience_counter >= config['patience']:
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 测试评估
    test_metrics = evaluate(model, test_loader, device)
    
    # 跨数据集评估：在 MobiFall 上测试
    mobifall_test_ds = FusedDataset(mobifall_loader.get_all_samples())
    mobifall_test_loader = DataLoader(mobifall_test_ds, batch_size=config['batch_size'],
                                      shuffle=False, num_workers=config['num_workers'],
                                      collate_fn=collate_fn)
    mobifall_metrics = evaluate(model, mobifall_test_loader, device)
    
    logging.info(f"Fold {fold_idx + 1} Results:")
    logging.info(f"  SisFall Test Acc: {test_metrics['accuracy']:.2f}%")
    logging.info(f"  MobiFall Test Acc: {mobifall_metrics['accuracy']:.2f}%")
    logging.info(f"  Cross-Dataset F1: {mobifall_metrics['f1']:.2f}%")
    
    return {
        'fold': fold_idx,
        'test_subject': test_subject,
        'sisfall_metrics': test_metrics,
        'mobifall_metrics': mobifall_metrics,
        'best_val_f1': best_val_f1,
    }


def run_experiment(config: Dict) -> Dict:
    """运行完整的多数据集 LOSO 实验"""
    
    # 设置
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # 加载数据集
    logging.info("Loading datasets...")
    sisfall_loader = SisFallLoader(config['data_root'], WINDOW_SIZE, STRIDE)
    mobifall_loader = MobiFallLoader(config['data_root'], WINDOW_SIZE, STRIDE)
    
    # 统计信息
    sisfall_falls = sum(1 for s in sisfall_loader.samples if s[1] == 1)
    sisfall_adls = len(sisfall_loader.samples) - sisfall_falls
    mobifall_falls = sum(1 for s in mobifall_loader.samples if s[1] == 1)
    mobifall_adls = len(mobifall_loader.samples) - mobifall_falls
    
    logging.info(f"\nDataset Statistics:")
    logging.info(f"  SisFall: {len(sisfall_loader.samples)} windows (Falls: {sisfall_falls}, ADLs: {sisfall_adls})")
    logging.info(f"  MobiFall: {len(mobifall_loader.samples)} windows (Falls: {mobifall_falls}, ADLs: {mobifall_adls})")
    
    # 12-fold LOSO
    fold_results = []
    for fold_idx, test_subject in enumerate(CORE_SUBJECTS):
        result = run_loso_fold(fold_idx, test_subject, sisfall_loader, 
                              mobifall_loader, config, device)
        fold_results.append(result)
        
        # 保存中间结果
        interim_path = os.path.join(config['out_dir'], 'interim_results.json')
        with open(interim_path, 'w') as f:
            json.dump(fold_results, f, indent=2)
    
    # 汇总结果
    sisfall_accs = [r['sisfall_metrics']['accuracy'] for r in fold_results]
    sisfall_f1s = [r['sisfall_metrics']['f1'] for r in fold_results]
    mobifall_accs = [r['mobifall_metrics']['accuracy'] for r in fold_results]
    mobifall_f1s = [r['mobifall_metrics']['f1'] for r in fold_results]
    
    summary = {
        'experiment': 'Multi-Dataset LOSO (SisFall 80% + MobiFall 20%)',
        'seed': config['seed'],
        'n_folds': len(fold_results),
        'sisfall': {
            'accuracy_mean': np.mean(sisfall_accs),
            'accuracy_std': np.std(sisfall_accs),
            'f1_mean': np.mean(sisfall_f1s),
            'f1_std': np.std(sisfall_f1s),
            'per_fold_accuracy': sisfall_accs,
        },
        'mobifall': {
            'accuracy_mean': np.mean(mobifall_accs),
            'accuracy_std': np.std(mobifall_accs),
            'f1_mean': np.mean(mobifall_f1s),
            'f1_std': np.std(mobifall_f1s),
            'per_fold_accuracy': mobifall_accs,
        },
        'cross_dataset_f1': {
            'mean': np.mean(mobifall_f1s),
            'std': np.std(mobifall_f1s),
        },
        'fold_details': fold_results,
    }
    
    return summary


def save_results(summary: Dict, out_dir: str):
    """保存实验结果"""
    
    # JSON 结果
    json_path = os.path.join(out_dir, 'summary_results.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Results saved to {json_path}")
    
    # CSV 每折结果
    csv_path = os.path.join(out_dir, 'per_fold_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['Fold', 'Test_Subject', 'SisFall_Acc', 'SisFall_F1', 
                        'MobiFall_Acc', 'MobiFall_F1'])
        for r in summary['fold_details']:
            writer.writerow([
                r['fold'] + 1,
                r['test_subject'],
                f"{r['sisfall_metrics']['accuracy']:.2f}",
                f"{r['sisfall_metrics']['f1']:.2f}",
                f"{r['mobifall_metrics']['accuracy']:.2f}",
                f"{r['mobifall_metrics']['f1']:.2f}",
            ])
    logging.info(f"Per-fold metrics saved to {csv_path}")
    
    # 打印最终汇总
    logging.info("\n" + "="*60)
    logging.info("FINAL RESULTS")
    logging.info("="*60)
    logging.info(f"SisFall Accuracy: {summary['sisfall']['accuracy_mean']:.2f}% ± {summary['sisfall']['accuracy_std']:.2f}%")
    logging.info(f"SisFall F1: {summary['sisfall']['f1_mean']:.2f}% ± {summary['sisfall']['f1_std']:.2f}%")
    logging.info(f"MobiFall Accuracy: {summary['mobifall']['accuracy_mean']:.2f}% ± {summary['mobifall']['accuracy_std']:.2f}%")
    logging.info(f"MobiFall F1: {summary['mobifall']['f1_mean']:.2f}% ± {summary['mobifall']['f1_std']:.2f}%")
    logging.info(f"Cross-Dataset F1: {summary['cross_dataset_f1']['mean']:.2f}% ± {summary['cross_dataset_f1']['std']:.2f}%")
    logging.info("="*60)


def main():
    parser = argparse.ArgumentParser(description='Multi-Dataset LOSO Cross-Validation')
    parser.add_argument('--data-root', type=str, default='./data', help='数据集根目录')
    parser.add_argument('--out-dir', type=str, default='./outputs/multi_dataset_loso', help='输出目录')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--patience', type=int, default=15, help='早停耐心值')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num-workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--amp', action='store_true', help='启用混合精度训练')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 设置日志
    setup_logging(args.out_dir)
    
    # 配置
    config = vars(args)
    logging.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # 保存配置
    config_path = os.path.join(args.out_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 运行实验
    start_time = time.time()
    summary = run_experiment(config)
    elapsed = time.time() - start_time
    
    summary['elapsed_time_seconds'] = elapsed
    summary['elapsed_time_formatted'] = f"{elapsed/3600:.2f} hours"
    
    # 保存结果
    save_results(summary, args.out_dir)
    
    logging.info(f"\nExperiment completed in {elapsed/3600:.2f} hours")


if __name__ == '__main__':
    main()
