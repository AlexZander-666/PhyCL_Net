"""
Training and evaluation entrypoint for the PhyCL-Net project.

Historical file names in this repository still mention DMCNet or AMSNetV2. The
canonical manuscript-facing names are:
    - phycl:      PhyCL-Net without the spectral MSPA branch
    - phycl_full: matched spectral baseline used for the trade-off study
    - amsv2:      legacy internal name kept for backward compatibility
"""

import os
import sys
import json
import math
import time
import random
import argparse
import traceback
import logging
import inspect
import csv
import copy
import yaml
import subprocess
import multiprocessing
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

# [Fix Windows] 在Windows上使用spawn启动方法，避免DataLoader多进程死锁
if sys.platform == 'win32':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 已经设置过，忽略

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import amp as torch_amp
import re

from models import AMSNetV2
from models.ams_net_v2 import CrossGatedFusion
from models.modules.mspa import MultiScaleSpectralPyramidAttention, MultiScaleSpectralPyramid
from models.modules.dks import DynamicKernelBlock
from models.modules.spectral import MultiScaleSTFTBlock, WaveletSpectralBlock
from models.modules.efficient import GhostConv1d, SeparableConv1d
from models.modules.attention import build_attention
from losses import AMSNetLoss

# --- Safe Imports ---
try:
    import threadpoolctl  # type: ignore
    _HAS_THREADPOOLCTL = True
except ImportError:
    threadpoolctl = None
    _HAS_THREADPOOLCTL = False

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix, balanced_accuracy_score, matthews_corrcoef
    _HAS_SKLEARN = True
except ImportError:
    accuracy_score = f1_score = precision_score = recall_score = roc_auc_score = average_precision_score = confusion_matrix = balanced_accuracy_score = matthews_corrcoef = None
    _HAS_SKLEARN = False

try:
    from scipy import stats
except ImportError:
    stats = None
try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None
try:
    from thop import profile
except ImportError:
    profile = None

ALLOW_METRICS_FALLBACK = False

def ensure_metric_dependencies(allow_fallback: bool = False) -> bool:
    """
    Ensure sklearn metric dependencies are available. Fail fast by default to avoid silent metric drop.
    """
    missing = []
    if not _HAS_THREADPOOLCTL:
        missing.append('threadpoolctl')
    if not _HAS_SKLEARN:
        missing.append('scikit-learn')
    if missing:
        missing_str = ", ".join(missing)
        msg = (
            f"Missing dependencies for evaluation metrics: {missing_str}. "
            "Install via `pip install threadpoolctl scikit-learn` or rerun with "
            "`--allow-metrics-fallback` to use manual metrics only (not recommended)."
        )
        if allow_fallback:
            logging.warning(msg)
            return False
        raise RuntimeError(msg)
    return True

# -------------------------- Utilities --------------------------
def setup_logging(out_dir: str):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # Check existing file handlers for this out_dir
    has_file = False
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == os.path.join(out_dir, 'experiment.log'):
            has_file = True
            break
    if not has_file:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(os.path.join(out_dir, 'experiment.log'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter); logger.addHandler(fh)
    # Ensure a stream handler exists
    has_stream = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if not has_stream:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(sh)
    logging.info(f"Logging initialized. Output directory: {out_dir}")

def set_seed(seed: int, deterministic: bool = False):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    logging.info(f"Seed set to {seed} (deterministic={deterministic})")

def worker_init_fn(worker_id):
    """
    [Fix 2] 改进的 Worker 初始化：使用 get_worker_info 确保与 PyTorch 版本兼容性
    保证每个 worker 拥有确定性但不同的种子。
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    # 使用 worker_info.seed 确保跨 PyTorch 版本的稳健性
    seed = worker_info.seed % 2**32
    np.random.seed(seed)
    random.seed(seed)

def resolve_resume_path(resume_path: Optional[str], out_dir: str, seed: int, split_tag: str) -> Optional[str]:
    """
    Resolve the actual checkpoint path to use for resuming.
    - If resume_path is a directory, try per-seed/per-split ckpt_last inside it.
    - If resume_path is a file, use it directly (only if it exists).
    """
    if not resume_path:
        return None
    if os.path.isdir(resume_path):
        cand = os.path.join(resume_path, f"ckpt_last_seed{seed}_{split_tag}.pth")
        if os.path.exists(cand):
            return cand
        # fallback: generic last checkpoint without split suffix
        cand_generic = os.path.join(resume_path, f"ckpt_last_seed{seed}.pth")
        if os.path.exists(cand_generic):
            return cand_generic
        logging.warning(f"[{split_tag}] Resume dir '{resume_path}' has no checkpoint for seed {seed}; starting fresh.")
        return None
    if os.path.exists(resume_path):
        return resume_path
    logging.warning(f"[{split_tag}] Resume path '{resume_path}' not found; starting fresh.")
    return None

def get_rng_states():
    """获取所有随机数生成器的状态"""
    states = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states['torch_cuda'] = torch.cuda.get_rng_state_all()
    return states

def set_rng_states(states):
    """恢复随机数生成器状态"""
    try:
        random.setstate(states['python'])
        np.random.set_state(states['numpy'])
        torch.set_rng_state(states['torch'])
        if torch.cuda.is_available() and 'torch_cuda' in states:
            torch.cuda.set_rng_state_all(states['torch_cuda'])
        logging.info("RNG states restored successfully.")
    except Exception as e:
        logging.warning(f"Failed to restore RNG states: {e}")

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_run_info(out_dir, args):
    """保存实验配置"""
    info = {
        'args': vars(args),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(out_dir, 'run_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

def get_git_commit_hash() -> Optional[str]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return commit
    except Exception:
        return None

def get_pip_freeze() -> Optional[List[str]]:
    try:
        output = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL).decode().splitlines()
        return output
    except Exception:
        return None

def save_complete_experiment_config(args, out_dir: str):
    config = {
        'args': vars(args),
        'environment': {
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        'git_commit': get_git_commit_hash(),
        'dependencies': get_pip_freeze(),
        'timestamp': datetime.now().isoformat(),
    }
    with open(os.path.join(out_dir, 'experiment_config.yaml'), 'w') as f:
        yaml.dump(config, f)

AMP_DEVICE_TYPE = 'cuda'

try:
    GradScaler = torch_amp.GradScaler  # type: ignore[attr-defined]
    _GRAD_SCALER_SUPPORTS_DEVICE = True
except AttributeError:
    from torch.cuda.amp import GradScaler  # type: ignore
    _GRAD_SCALER_SUPPORTS_DEVICE = False


def _amp_autocast(enabled: bool, device_type: str = AMP_DEVICE_TYPE):
    """统一使用 torch.amp.autocast / GradScaler 的 device_type。"""
    return torch_amp.autocast(device_type=device_type, enabled=enabled)


def _make_grad_scaler(device_type: str):
    return GradScaler(device=device_type) if _GRAD_SCALER_SUPPORTS_DEVICE else GradScaler()

ABLATION_PRESETS = {
    'full': {'mspa': True, 'dks': True, 'faa': True, 'tfcl': True, 'center': True},
    'baseline': {'mspa': False, 'dks': False, 'faa': False, 'tfcl': False, 'center': False},
    'no_mspa': {'mspa': False, 'dks': True, 'faa': True, 'tfcl': True, 'center': True},
    'no_dks': {'mspa': True, 'dks': False, 'faa': True, 'tfcl': True, 'center': True},
    'no_faa': {'mspa': True, 'dks': True, 'faa': False, 'tfcl': True, 'center': True},
    'no_tfcl': {'mspa': True, 'dks': True, 'faa': True, 'tfcl': False, 'center': True},
    'no_center': {'mspa': True, 'dks': True, 'faa': True, 'tfcl': True, 'center': False},
    'time_only': {'mspa': False, 'dks': True, 'faa': False, 'tfcl': False, 'center': False},
    'freq_only': {'mspa': True, 'dks': False, 'faa': False, 'tfcl': False, 'center': False},
}

MODEL_ALIASES = {
    'phycl': ('amsv2', 'no_mspa'),
    'phycl_full': ('amsv2', 'full'),
}

HYPERPARAMETER_SENSITIVITY = {
    'num_bands': [2, 4, 6, 8],
    'kernel_sizes': [(7, 15, 31), (7, 15, 31, 63), (3, 7, 15, 31, 63)],
    'tfcl_temperature': [0.05, 0.1, 0.2, 0.5],
    'center_loss_beta': [0.001, 0.01, 0.1],
    'proj_dim': [64, 128, 256],
    'window_size': [256, 512, 1024],
}

ABLATION_MATRIX = [
    {"name": "full", "mspa": True, "dks": True, "faa": True, "tfcl": True, "center": True},
    {"name": "w/o_mspa", "mspa": False, "dks": True, "faa": True, "tfcl": True, "center": True},
    {"name": "w/o_dks", "mspa": True, "dks": False, "faa": True, "tfcl": True, "center": True},
    {"name": "w/o_faa", "mspa": True, "dks": True, "faa": False, "tfcl": True, "center": True},
    {"name": "w/o_tfcl", "mspa": True, "dks": True, "faa": True, "tfcl": False, "center": True},
    {"name": "w/o_center", "mspa": True, "dks": True, "faa": True, "tfcl": True, "center": False},
    {"name": "time_only", "mspa": False, "dks": True, "faa": False, "tfcl": False, "center": False},
    {"name": "freq_only", "mspa": True, "dks": False, "faa": False, "tfcl": False, "center": False},
]


def parse_ablation_config(spec: Optional[str]) -> Dict[str, bool]:
    """
    Parse ablation config string into module toggles.
    Accepts preset names or comma-separated key=value pairs (1/0, true/false).
    """
    if isinstance(spec, dict):
        return spec
    config = {'mspa': True, 'dks': True, 'faa': True, 'tfcl': True, 'center': True}
    if not spec:
        return config
    key = str(spec).lower()
    if key in ABLATION_PRESETS:
        return dict(ABLATION_PRESETS[key])
    for token in str(spec).split(','):
        if '=' in token:
            k, v = token.split('=', 1)
        elif ':' in token:
            k, v = token.split(':', 1)
        else:
            continue
        k = k.strip().lower()
        v = v.strip().lower()
        if k in config:
            config[k] = v not in ('0', 'false', 'no')
    if spec and all(config.values()):
        logging.warning("ablation spec unrecognized, got defaults")
    return config


def resolve_requested_model(model_name: str, ablation_spec: Optional[str]) -> Tuple[str, Optional[str], str]:
    """
    Map manuscript-facing model aliases to the legacy internal implementation.
    Returns (internal_model, effective_ablation_spec, display_name).
    """
    key = str(model_name).lower()
    if key in MODEL_ALIASES:
        internal_model, default_ablation = MODEL_ALIASES[key]
        effective_ablation = ablation_spec if ablation_spec else default_ablation
        display_name = "PhyCL-Net" if key == "phycl" else "PhyCL-Net + MSPA"
        return internal_model, effective_ablation, display_name
    return model_name, ablation_spec, model_name


def sensor_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None, None, None
    xs, ys, subs = zip(*batch)
    xs = [x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32) for x in xs]
    xs = torch.stack(xs, dim=0)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, ys, list(subs)

class AdaptiveEarlyStopping:
    """
    Patience shrinks after improvements to speed up convergence.
    """
    def __init__(self, initial_patience: int = 15, min_patience: int = 5, decay_factor: float = 0.9):
        self.patience = initial_patience
        self.min_patience = min_patience
        self.decay_factor = decay_factor
        self.counter = 0
    def step(self, improved: bool) -> bool:
        if improved:
            self.patience = max(self.min_patience, int(self.patience * self.decay_factor))
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(m.bias, -bound, bound)
    elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

_TORCH_LOAD_HAS_WEIGHTS_ONLY = 'weights_only' in inspect.signature(torch.load).parameters


def torch_load_full(path: str, map_location=None):
    kwargs = {}
    if map_location is not None:
        kwargs['map_location'] = map_location
    if _TORCH_LOAD_HAS_WEIGHTS_ONLY:
        kwargs['weights_only'] = False
    return torch.load(path, **kwargs)


def load_checkpoint_for_resume(ckpt_path, model, optimizer=None, scheduler=None, scaler=None, device=None):
    """
    [Patch B] 完整的断点恢复加载函数
    """
    logging.info(f"Resuming training from {ckpt_path}...")
    ckpt = torch_load_full(ckpt_path, map_location=device)
    
    # 1. Load Model
    model.load_state_dict(ckpt['model_state'])
    
    # 2. Load States if provided
    if optimizer is not None and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    if scheduler is not None and 'scheduler_state' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state'])
    if scaler is not None and 'scaler_state' in ckpt and ckpt['scaler_state'] is not None:
        scaler.load_state_dict(ckpt['scaler_state'])
    
    # 3. Restore RNG
    if 'rng_states' in ckpt:
        set_rng_states(ckpt['rng_states'])
        
    start_epoch = ckpt.get('epoch', 0) + 1
    best_f1 = ckpt.get('best_f1', 0.0)
    logging.info(f"Resumed from epoch {start_epoch-1}. Best F1 so far: {best_f1:.4f}")
    
    return start_epoch, best_f1

# -------------------------- Deployment Profiling --------------------------

def profile_model_efficiency(model, input_size=(1, 3, 512), device=torch.device('cpu')):
    model.eval()
    model.to(device)
    x = torch.randn(input_size).to(device)
    
    flops, params = 0.0, 0.0
    if profile is not None:
        try:
            macs, params = profile(model, inputs=(x,), verbose=False)
            flops = macs * 2.0 
        except Exception as e:
            logging.warning(f"THOP profiling warning: {e}")
            params = sum(p.numel() for p in model.parameters())
            flops = None # [Fix 9] Explicitly None if failed
    else:
        # [Fix 9] Fallback if thop is missing
        params = sum(p.numel() for p in model.parameters())
        logging.warning("Warning: 'thop' not installed. FLOPs not calculated. MACs set to None.")
        flops = None

    avg_latency_ms = 0.0
    try:
        with torch.no_grad():
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            for _ in range(20): _ = model(x) # Warmup
            if device.type == 'cuda': torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            runs = 100
            for _ in range(runs): _ = model(x)
            if device.type == 'cuda': torch.cuda.synchronize()
            end_time = time.perf_counter()
        
        avg_latency_ms = ((end_time - start_time) / runs) * 1000
    except Exception as e:
        logging.error(f"Latency measurement failed: {e}")

    logging.info("-" * 30)
    logging.info(f"[Efficiency Report] Input: {input_size}")
    logging.info(f"  - Params: {params / 1e6:.2f} M")
    flops_str = f"{flops / 1e9:.3f} G" if flops is not None else "N/A"
    logging.info(f"  - FLOPs: {flops_str}")
    logging.info(f"  - Latency: {avg_latency_ms:.3f} ms")
    logging.info("-" * 30)
    return {'params_M': params/1e6, 'flops_G': flops/1e9 if flops else None, 'latency_ms': avg_latency_ms}

# -------------------------- AMS-Net Modules --------------------------
class AdaptiveSpectralBlock(nn.Module):
    """
    [改进点 A] 自适应频域块 (Frequency Domain Branch)
    原理：利用 FFT 将信号转换到频域，通过可学习的参数进行滤波/加权，再转回时域。
    优势：相比 LSK，能更有效地捕捉传感器数据的全局周期性，并抑制高频噪声。
    适合 SCI 论点：Explicit Global Frequency Modeling (显式全局频率建模)
    """
    def __init__(self, dim, temp_init: float = 8.0):
        super().__init__()
        # 小型频率门控网络（逐通道深度卷积），生成平滑频率掩码
        self.freq_mlp = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True),
            nn.Sigmoid()
        )
        # 可学习温度，控制软门控陡峭度（初始化为较平缓）
        self.log_temp = nn.Parameter(torch.log(torch.ones(dim) * temp_init))
        self.act = nn.GELU()

    def forward(self, x):
        assert x.dim() == 3, "AdaptiveSpectralBlock expects (B, C, L)"
        assert x.dtype.is_floating_point, "Input must be floating point"
        B, C, L = x.shape

        # Hann window 减少谱泄露
        win = torch.hann_window(L, device=x.device, dtype=torch.float32).view(1, 1, L)

        # 禁用 autocast 以保证 FFT 的数值稳定性
        with _amp_autocast(False):
            xw = (x * win).to(torch.float32)
            X = torch.fft.rfft(xw, dim=-1, norm='ortho')  # (B, C, F)

        amp = torch.abs(X)  # 幅值用于门控

        # 生成平滑频率门：局部 conv + 全局温度平滑
        gate_local = self.freq_mlp(amp)  # (B, C, F) in [0,1]
        temp = torch.exp(self.log_temp).view(1, C, 1)
        gate = torch.sigmoid((amp - amp.mean(dim=-1, keepdim=True)) / (temp + 1e-6)) * gate_local

        gate_c = gate.to(X.dtype)
        X_mod = X * gate_c

        with _amp_autocast(False):
            x_out = torch.fft.irfft(X_mod, n=L, dim=-1, norm='ortho')

        return self.act(x_out)


def build_spectral_branch(
    method: str,
    channels: int,
    adaptive_bands: bool = True,
    band_edges: Optional[Tuple[float, ...]] = None,
    num_bands: int = 4,
) -> nn.Module:
    """
    Factory for selecting spectral encoder (FFT/STFT/CWT).
    """
    m = (method or "fft").lower()
    if m == "stft":
        return MultiScaleSTFTBlock(channels)
    if m in ("cwt", "wavelet"):
        return WaveletSpectralBlock(channels)
    if m == "adaptive_fft":
        return AdaptiveSpectralBlock(channels)
    if m == "fft_attn":
        return MultiScaleSpectralPyramidAttention(channels)
    return MultiScaleSpectralPyramid(
        channels,
        num_bands=num_bands,
        band_edges=band_edges,
        adaptive_bands=adaptive_bands,
    )
 
class ModernTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 31):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.dw = nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels)
        self.pw = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(1, channels)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.dw(x)
        y = self.norm(y)
        y = self.act(y)
        y = self.pw(y)
        return y

class ImprovedDMCBlock(nn.Module):
    """
    [最终整合] 时频双流块
    """
    def __init__(
        self,
        channels: int,
        tcn_kernel: int = 31,
        freq_method: str = "fft",
        use_dks: bool = True,
        use_freq_branch: bool = True,
        kernel_sizes: Tuple[int, ...] = (7, 15, 31, 63),
        sample_rate: float = 50.0,
        adaptive_bands: bool = True,
        band_edges: Optional[Tuple[float, ...]] = None,
        num_bands: int = 4,
        fusion_variant: str = "enhanced",
        fusion_kernel_sizes: Tuple[int, ...] = (3, 5, 7),
    ):
        super().__init__()
        # 分支 1: 时域 (Local/Trend) - 保持 ModernTCN (擅长捕捉局部趋势)
        self.time_branch = (
            DynamicKernelBlock(channels, kernel_sizes=kernel_sizes, sample_rate=sample_rate) if use_dks else ModernTCNBlock(channels, kernel_size=tcn_kernel)
        )
        
        # 分支 2: 频域 (Global/Periodicity) - 替换 LSK 为 SpectralBlock
        self.freq_branch = (
            build_spectral_branch(
                freq_method,
                channels,
                adaptive_bands=adaptive_bands,
                band_edges=band_edges,
                num_bands=num_bands,
            )
            if use_freq_branch
            else nn.Identity()
        )
        
        # 融合: 交叉门控
        self.fusion = CrossGatedFusion(channels, variant=fusion_variant, kernel_sizes=fusion_kernel_sizes)
        self.res_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        assert x.dim() == 3, "ImprovedDMCBlock expects (B, C, L)"
        # 双流并行处理
        x_t = self.time_branch(x)
        x_f = self.freq_branch(x)
        
        # 融合
        out = self.fusion(x_t, x_f)
        
        # 残差连接
        return x + self.res_scale * out


class DMCNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int = 64,
        n_blocks: int = 4,
        num_classes: int = 2,
        freq_method: str = "fft",
        use_dks: bool = True,
        use_freq_branch: bool = True,
        kernel_sizes: Tuple[int, ...] = (7, 15, 31, 63),
        sample_rate: float = 50.0,
        adaptive_bands: bool = True,
        band_edges: Optional[Tuple[float, ...]] = None,
        num_bands: int = 4,
        fusion_variant: str = "enhanced",
        fusion_kernel_sizes: Tuple[int, ...] = (3, 5, 7),
    ):
        super().__init__()
        self.stem = nn.Conv1d(in_channels, channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                ImprovedDMCBlock(
                    channels,
                    freq_method=freq_method,
                    use_dks=use_dks,
                    use_freq_branch=use_freq_branch,
                    kernel_sizes=kernel_sizes,
                    sample_rate=sample_rate,
                    adaptive_bands=adaptive_bands,
                    band_edges=band_edges,
                    num_bands=num_bands,
                    fusion_variant=fusion_variant,
                    fusion_kernel_sizes=fusion_kernel_sizes,
                )
                for _ in range(n_blocks)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)
        self.apply(init_weights)

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        x = self.pool(x).squeeze(-1)
        out = self.fc(x)
        return out, x, x


class SimplifiedSpectralBlock(nn.Module):
    """Lightweight frequency encoder for LiteAMSNet."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _amp_autocast(False):
            X = torch.fft.rfft(x.to(torch.float32), dim=-1, norm="ortho")
            mag = torch.abs(X)
            mag = F.interpolate(mag, size=x.size(-1), mode="linear", align_corners=False)
        return self.conv(mag.to(dtype=x.dtype))


class LiteAMSBlock(nn.Module):
    """Lightweight AMS block for edge deployment."""

    def __init__(self, channels: int, attn: Any = None):
        super().__init__()
        self.time_branch = SeparableConv1d(channels, channels, kernel_size=15)
        self.freq_branch = SimplifiedSpectralBlock(channels)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.attn = build_attention(attn, channels) if attn is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = self.time_branch(x)
        x_f = self.freq_branch(x)
        alpha = torch.sigmoid(self.alpha)
        out = alpha * x_t + (1 - alpha) * x_f
        if self.attn is not None:
            out = self.attn(out)
        return x + out


class LiteAMSNet(nn.Module):
    """
    Lightweight AMS-Net variant (<100K params target) for edge deployment.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2, channels: int = 32, n_blocks: int = 2, attn: Any = None):
        super().__init__()
        self.stem = GhostConv1d(in_channels, channels, kernel_size=5)
        self.blocks = nn.ModuleList([LiteAMSBlock(channels, attn=attn) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, num_classes),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        logits = self.head(x)
        return logits, x, x


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor, T: float = 4.0, alpha: float = 0.7) -> torch.Tensor:
    """
    Knowledge distillation loss combining soft teacher targets and hard labels.
    """
    soft_targets = F.softmax(teacher_logits / T, dim=-1)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        soft_targets,
        reduction='batchmean',
    ) * (T * T)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss

# -------------------------- Baseline Models --------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, in_channels, hidden_size=128, num_layers=2, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.apply(init_weights)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        return self.fc(out)

class ResNet1D(nn.Module):
    def __init__(self, in_channels, num_classes=2, channels=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, channels, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.layer1 = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=1), nn.BatchNorm1d(channels), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)
        self.apply(init_weights)
    def forward(self, x):
        x = self.pool(self.layer1(self.relu(self.bn1(self.conv1(x))))).squeeze(-1)
        return self.fc(x)

class TemporalConvNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2, channels: int = 64, depth: int = 4):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, channels, kernel_size=1) if in_channels != channels else nn.Identity()
        layers = []
        for i in range(depth):
            dilation = 2 ** i
            layers.append(
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                    nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                )
            )
        self.blocks = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)
        self.apply(init_weights)
    def forward(self, x):
        x = self.proj(x)
        for blk in self.blocks:
            x = x + blk(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class TransformerClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, num_classes)
        self.apply(init_weights)
    def forward(self, x):
        # x: (B, C, L) -> (B, L, d_model)
        x = self.proj(x).permute(0, 2, 1)
        h = self.encoder(x)
        h = h.mean(dim=1)
        return self.cls(h)

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernels=(9, 19, 39)):
        super().__init__()
        branches = []
        for k in kernels:
            pad = k // 2
            branches.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1),
                    nn.Conv1d(out_channels, out_channels, kernel_size=k, padding=pad, groups=out_channels),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.pool_branch = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self.proj = nn.Conv1d(out_channels * (len(kernels) + 1), out_channels, kernel_size=1)
        self.res_scale = nn.Parameter(torch.tensor(0.0))
        # Shortcut projection when in_channels != out_channels
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        outs = [b(x) for b in self.branches]
        outs.append(self.pool_branch(x))
        out = torch.cat(outs, dim=1)
        out = self.proj(out)
        return self.shortcut(x) + self.res_scale * out

class InceptionTime(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2, channels: int = 32, depth: int = 3):
        super().__init__()
        blocks = []
        for i in range(depth):
            blocks.append(InceptionBlock1D(in_channels if i == 0 else channels, channels))
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)
        self.apply(init_weights)
    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class TinyHAR(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2, channels: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)
        self.apply(init_weights)
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class DeepConvLSTM(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2, conv_channels: int = 64, lstm_hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(conv_channels, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)
        self.apply(init_weights)
    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        h, _ = self.lstm(x)
        h = h.mean(dim=1)
        return self.fc(h)

class RocketClassifier(nn.Module):
    """
    Lightweight ROCKET-style random convolutional kernels.
    Kernels are fixed at init; only classifier is trained.
    """
    def __init__(self, in_channels: int, num_kernels: int = 256, kernel_size: int = 9, num_classes: int = 2):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.register_buffer("kernels", torch.randn(num_kernels, in_channels, kernel_size))
        self.register_buffer("bias", torch.zeros(num_kernels))
        self.classifier = nn.Linear(num_kernels * 2, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape
        k = self.kernels  # (K, C, Ksz)
        conv_out = F.conv1d(x, k, bias=self.bias, padding=self.kernel_size // 2)
        feat = torch.cat([conv_out.max(dim=-1).values, conv_out.mean(dim=-1)], dim=1)
        return self.classifier(feat)

# -------------------------- Data Handling --------------------------

def compute_class_weights_by_strategy(
    dataset,
    strategy: str,
    num_classes: int = 2,
    eps: float = 1e-6,
    max_weight: float = 10.0,
    beta: float = 0.999,
) -> torch.Tensor:
    """
    Compute class weights using the requested strategy.
    """
    labels = [int(y) for _, y, _ in dataset.items]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, eps)
    total = counts.sum()

    strategy_l = (strategy or "none").lower()
    if strategy_l in ("auto", "inv_freq"):
        weights = total / (num_classes * counts)
    elif strategy_l == "sqrt_inv_freq":
        weights = total / (num_classes * np.sqrt(counts))
    elif strategy_l == "effective_num":
        effective = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective, eps)
        weights = weights * (num_classes / np.maximum(weights.sum(), eps))
    else:
        raise ValueError(f"Unknown class weighting strategy: {strategy}")

    weights = np.clip(weights, 0.1, max_weight)
    logging.info("Computed class weights (%s): %s", strategy_l, weights)
    return torch.tensor(weights, dtype=torch.float32)


def compute_class_weights(dataset, num_classes=2, eps=1e-6, max_weight=10.0):
    """
    Backward-compatible wrapper using the default auto (inverse-frequency) strategy.
    """
    try:
        return compute_class_weights_by_strategy(
            dataset,
            "auto",
            num_classes=num_classes,
            eps=eps,
            max_weight=max_weight,
        )
    except Exception as e:
        logging.warning(f"Failed to compute class weights ({e}). Using ones.")
        return torch.ones(num_classes, dtype=torch.float32)


class DummyDataset(Dataset):
    def __init__(self, items: List[Tuple[torch.Tensor, int, str]]):
        super().__init__()
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

class PreloadedHARDataset(Dataset):
    """
    Generic dataset loader for preprocessed NPZ files.
    Expect keys: x (N, C, L) or (N, L, C) and y (N,).
    """
    def __init__(self, npz_path: str, channels_used: Optional[str] = None):
        super().__init__()
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"Dataset file not found: {npz_path}")
        data = np.load(npz_path)
        x = data.get('x', data.get('X', None))
        y = data.get('y', data.get('Y', None))
        if x is None or y is None:
            raise ValueError(f"{npz_path} missing 'x'/'y' keys.")
        if x.ndim != 3:
            raise ValueError(f"{npz_path} expected 3D array, got {x.shape}.")
        if x.shape[1] != 3 and x.shape[2] == 3:  # (N, L, C) -> (N, C, L)
            x = np.transpose(x, (0, 2, 1))
        if channels_used:
            mode = channels_used.lower()
            if mode == "accel3" and x.shape[1] >= 3:
                x = x[:, :3, :]
            elif mode == "accel6" and x.shape[1] >= 6:
                x = x[:, :6, :]
        # per-sample standardization
        mean = x.mean(axis=2, keepdims=True)
        std = x.std(axis=2, keepdims=True) + 1e-6
        x = (x - mean) / std
        self.items = [(x[i].astype(np.float32), int(y[i]), "preload") for i in range(len(x))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

class SensorAugmentation:
    """
    Lightweight sensor augmentation for time-series windows.
    """
    def __init__(self, noise_std: float = 0.05, scale_range: Tuple[float, float] = (0.9, 1.1), time_shift_ratio: float = 0.1, drop_prob: float = 0.1):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.time_shift_ratio = time_shift_ratio
        self.drop_prob = drop_prob

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_aug = np.array(x, copy=True)
        if self.noise_std > 0 and random.random() < 0.5:
            x_aug = x_aug + np.random.randn(*x_aug.shape).astype(np.float32) * self.noise_std
        if random.random() < 0.5:
            scale = random.uniform(*self.scale_range)
            x_aug = x_aug * scale
        if self.time_shift_ratio > 0 and random.random() < 0.5:
            shift = int(x_aug.shape[1] * self.time_shift_ratio * random.uniform(-1, 1))
            if shift != 0:
                x_aug = np.roll(x_aug, shift, axis=1)
        if self.drop_prob > 0 and random.random() < 0.5:
            drop_len = max(1, int(x_aug.shape[1] * self.drop_prob))
            start = random.randint(0, max(0, x_aug.shape[1] - drop_len))
            x_aug[:, start:start+drop_len] = 0
        return x_aug

def _resolve_sisfall_root(data_root: str) -> str:
    """
    确保返回包含 ADL/FALL 子目录的实际 SisFall 根目录。
    """
    if not data_root:
        raise ValueError("data_root for SisFallDataset cannot be empty.")

    candidates = [
        os.path.abspath(data_root),
        os.path.join(os.path.abspath(data_root), 'SisFall'),
    ]

    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        if all(os.path.isdir(os.path.join(candidate, sub)) for sub in ('ADL', 'FALL')):
            return candidate

    raise FileNotFoundError(
        f"Could not resolve SisFall root inside '{data_root}'. "
        "Ensure the folder contains 'ADL/' and 'FALL/' subdirectories."
    )


def _parse_sisfall_subject_from_name(filename: str) -> str:
    """
    从 SisFall 文件名中提取 subject 编号。
    文件名格式: D01_SA01_R01.txt 或 F01_SA01_R01.txt，其中 SA01 是 subject ID。
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(r'SA(\d+)', basename, re.IGNORECASE)
    if match:
        return f"SA{int(match.group(1)):02d}"
    return basename.upper()

_CHANNEL_WARNED = set()

def _select_sisfall_channels(raw_data: np.ndarray, mode: str) -> np.ndarray:
    """
    Select channels from raw SisFall arrays.
    Assumes ordering: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, ...].
    """
    mode_l = (mode or "accel3").lower()
    c = raw_data.shape[0]
    if mode_l == "accel3" or c < 4:
        return raw_data[: min(3, c)]
    if mode_l == "accel6":
        if c >= 6:
            return raw_data[:6]
    if mode_l in ("accel6+gyro", "accel9", "full"):
        if c >= 9:
            return raw_data[:9]
    # Fallback to available channels with a one-time warning
    if mode_l not in _CHANNEL_WARNED:
        logging.warning(f"Requested channels '{mode_l}' not fully available (have {c}). Falling back to first {min(c, 9)} channels.")
        _CHANNEL_WARNED.add(mode_l)
    return raw_data[:min(c, 9)]

def _channels_from_mode(mode: str) -> int:
    m = (mode or "accel3").lower()
    if m == "accel6":
        return 6
    if m in ("accel6+gyro", "accel9", "full"):
        return 9
    return 3

class SisFallDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        subjects: List[str],
        window_size: int = 512,
        stride: int = 256,
        log_dir: str = './',
        channels_used: str = "accel3",
        transform: Optional[SensorAugmentation] = None,
    ):
        super().__init__()
        self.items = []
        self.subjects = set(s.upper() for s in subjects) if subjects else None
        self.transform = transform
        self.channels_used = channels_used
        root = _resolve_sisfall_root(data_root)
        self.corrupt_log_path = os.path.join(log_dir, 'corrupt_files.log')

        def add_file(file_path: str, label: int, subj: str):
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
                            nums = [float(v) for v in values]
                        except ValueError:
                            continue
                        data_rows.append(nums)

                if not data_rows:
                    raise ValueError("No valid data rows parsed.")

                raw_data = np.array(data_rows, dtype=np.float32).T
                raw_data = _select_sisfall_channels(raw_data, self.channels_used)
                if np.isnan(raw_data).any():
                    raise ValueError("Contains NaN")

                mean = np.mean(raw_data, axis=1, keepdims=True)
                std = np.std(raw_data, axis=1, keepdims=True) + 1e-6
                norm_data = (raw_data - mean) / std
                c, l = norm_data.shape
                if l < window_size:
                    pad = np.zeros((c, window_size), dtype=np.float32)
                    pad[:, :l] = norm_data
                    self.items.append((pad, label, subj))
                else:
                    start = 0
                    while start + window_size <= l:
                        w = norm_data[:, start:start+window_size]
                        self.items.append((w.copy(), label, subj))
                        start += stride
            except Exception as e:
                with open(self.corrupt_log_path, 'a') as f:
                    f.write(f"{file_path}: {repr(e)}\n")

        adl_dir, fall_dir = os.path.join(root, 'ADL'), os.path.join(root, 'FALL')
        for folder_path in [adl_dir, fall_dir]:
            if not os.path.isdir(folder_path): continue
            folder_name = os.path.basename(folder_path).upper()
            current_label = 1 if 'FALL' in folder_name else 0
            for fname in os.listdir(folder_path):
                if not fname.lower().endswith('.txt'): continue
                subj = _parse_sisfall_subject_from_name(fname)
                if self.subjects and subj not in self.subjects: continue
                add_file(os.path.join(folder_path, fname), current_label, subj)
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        x, y, subj = self.items[idx]
        x_out = np.array(x, copy=True)
        if self.transform is not None:
            x_out = self.transform(x_out)
        return x_out, y, subj

# -------------------------- Training Logic --------------------------

def unpack_model_output(output):
    """
    Normalize model outputs to (logits, z_time, z_freq).
    """
    if isinstance(output, tuple):
        logits = output[0]
        z_time = output[1] if len(output) > 1 else None
        z_freq = output[2] if len(output) > 2 else None
    else:
        logits, z_time, z_freq = output, None, None
    return logits, z_time, z_freq


def compute_loss_with_aux(criterion, logits, targets, z_time=None, z_freq=None):
    """
    Compute loss and auxiliary components, supporting AMSNetLoss as well as standard criterions.
    """
    if isinstance(criterion, AMSNetLoss):
        loss_raw, parts = criterion(logits, targets, z_time, z_freq)
    else:
        loss_raw = criterion(logits, targets)
        parts = {'ce': float(loss_raw.detach().cpu())}
    return loss_raw, parts


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    scaler: Optional[GradScaler],
    accum_steps: int = 1,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    steps = 0
    accum_count = 0
    aux_sums: Dict[str, float] = {}
    
    for i, (x, y, _) in enumerate(loader):
        if x is None: continue 
        x, y = x.to(device), y.to(device)
        
        with _amp_autocast(scaler is not None):
            logits, z_time, z_freq = unpack_model_output(model(x))
            loss_raw, parts = compute_loss_with_aux(criterion, logits, y, z_time, z_freq)
            loss = loss_raw / accum_steps
        
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        accum_count += 1
        
        if accum_count == accum_steps:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            optimizer.zero_grad()
            accum_count = 0
        
        total_loss += loss_raw.item() * x.size(0)
        steps += x.size(0)
        for k, v in parts.items():
            aux_sums[k] = aux_sums.get(k, 0.0) + v * x.size(0)
        
    # [Patch A] Fix leftover gradient update (Consistent unscale/clip logic)
    if accum_count > 0:
        if scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad()
        
    aux_means = {k: v / (steps + 1e-12) for k, v in aux_sums.items()}
    return total_loss / (steps + 1e-12), aux_means

def compute_detection_latency_stats(
    predictions: List[int],
    labels: List[int],
    subjects: Optional[List[str]],
    stride: Optional[int],
    sample_rate: float,
) -> Dict[str, float]:
    """
    Compute detection latency following 0->1 transitions with per-subject continuity.
    Returns NaNs when stride/sample_rate are invalid to avoid misleading metrics.
    """
    if stride is None or sample_rate <= 0:
        return {
            'mean_latency_ms': np.nan,
            'median_latency_ms': np.nan,
            'std_latency_ms': np.nan,
            'min_latency_ms': np.nan,
            'max_latency_ms': np.nan,
            'detection_rate': 0.0,
            'total_falls': 0,
            'detected_falls': 0,
            'all_latencies': [],
        }

    predictions = np.array(predictions, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)
    subjects = np.array(subjects) if subjects is not None else None

    latencies: List[float] = []
    detected_falls = 0
    total_falls = 0

    for idx in range(1, len(labels)):
        if subjects is not None and subjects[idx] != subjects[idx - 1]:
            continue
        if labels[idx - 1] == 0 and labels[idx] == 1:
            total_falls += 1
            fall_start = idx
            detected = False
            for t in range(fall_start, len(predictions)):
                if subjects is not None and subjects[t] != subjects[fall_start]:
                    break
                if predictions[t] == 1:
                    latency_windows = t - fall_start
                    latency_ms = latency_windows * stride * 1000.0 / sample_rate
                    latencies.append(latency_ms)
                    detected_falls += 1
                    detected = True
                    break
            if not detected:
                latencies.append(np.nan)

    if total_falls == 0:
        return {
            'mean_latency_ms': np.nan,
            'median_latency_ms': np.nan,
            'std_latency_ms': np.nan,
            'min_latency_ms': np.nan,
            'max_latency_ms': np.nan,
            'detection_rate': 0.0,
            'total_falls': 0,
            'detected_falls': 0,
            'all_latencies': [],
        }

    valid_latencies = [l for l in latencies if not np.isnan(l)]
    return {
        'mean_latency_ms': float(np.nanmean(latencies)) if valid_latencies else np.nan,
        'median_latency_ms': float(np.nanmedian(latencies)) if valid_latencies else np.nan,
        'std_latency_ms': float(np.nanstd(latencies)) if valid_latencies else np.nan,
        'min_latency_ms': float(np.min(valid_latencies)) if valid_latencies else np.nan,
        'max_latency_ms': float(np.max(valid_latencies)) if valid_latencies else np.nan,
        'detection_rate': detected_falls / float(total_falls) if total_falls > 0 else 0.0,
        'total_falls': total_falls,
        'detected_falls': detected_falls,
        'all_latencies': latencies,
    }


def eval_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 2,
    stride: Optional[int] = None,
    sample_rate: float = 50.0,
    perturb_fn: Optional[Any] = None,
) -> Tuple[Dict[str, float], List[int], List[int], List[List[float]], List[str]]:
    ensure_metric_dependencies(ALLOW_METRICS_FALLBACK)
    model.eval()
    ys, preds, probs, subs = [], [], [], []
    with torch.no_grad():
        for x, y, subj in loader:
            if x is None: continue
            x, y = x.to(device), y.to(device)
            if perturb_fn is not None:
                x = perturb_fn(x)
            logits, _, _ = unpack_model_output(model(x))
            prob = F.softmax(logits, dim=1).cpu().numpy()
            
            ys.extend(y.cpu().numpy().tolist())
            preds.extend(np.argmax(prob, axis=1).tolist())
            probs.extend(prob.tolist())
            subs.extend(subj)
    
    metrics = {}
    if len(ys) == 0: return metrics, [], [], [], []

    # DEBUG: Check prediction distribution
    unique_preds = np.unique(preds)
    unique_labels = np.unique(ys)
    logging.debug(f"Eval predictions unique: {unique_preds}, counts: {np.bincount(preds)}")
    logging.debug(f"Eval labels unique: {unique_labels}, counts: {np.bincount(ys)}")
    if confusion_matrix:
        cm = confusion_matrix(ys, preds)
        logging.debug(f"Eval confusion matrix:\n{cm}")

    # Manual metric calculation (fallback when sklearn unavailable)
    ys_arr = np.array(ys, dtype=np.int32)
    preds_arr = np.array(preds, dtype=np.int32)

    # Accuracy
    metrics['accuracy'] = float(np.mean(ys_arr == preds_arr))

    # Per-class metrics
    for cls in range(num_classes):
        tp = np.sum((ys_arr == cls) & (preds_arr == cls))
        fp = np.sum((ys_arr != cls) & (preds_arr == cls))
        fn = np.sum((ys_arr == cls) & (preds_arr != cls))

        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)

        if cls == 1:
            metrics['fall_f1'] = float(f1)
            metrics['sensitivity'] = float(rec)
        if cls == 0:
            metrics['specificity'] = float(rec)

    # Macro F1
    f1_scores = []
    for cls in range(num_classes):
        tp = np.sum((ys_arr == cls) & (preds_arr == cls))
        fp = np.sum((ys_arr != cls) & (preds_arr == cls))
        fn = np.sum((ys_arr == cls) & (preds_arr != cls))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1_scores.append(2 * prec * rec / (prec + rec + 1e-12))
    metrics['macro_f1'] = float(np.mean(f1_scores))

    logging.debug(f"Manual metrics - Acc: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}, Fall F1: {metrics['fall_f1']:.4f}")

    if accuracy_score:
        acc = float(accuracy_score(ys, preds))
        metrics['accuracy'] = acc
        logging.debug(f"Accuracy: {acc:.4f}")
    if f1_score:
        f1_macro = float(f1_score(ys, preds, average='macro', zero_division=0))
        f1_class0 = float(f1_score(ys, preds, pos_label=0, zero_division=0))
        f1_class1 = float(f1_score(ys, preds, pos_label=1, zero_division=0))
        metrics['macro_f1'] = f1_macro
        logging.debug(f"F1 scores - Macro: {f1_macro:.4f}, Class0: {f1_class0:.4f}, Class1: {f1_class1:.4f}")
        metrics['fall_f1'] = float(f1_score(ys, preds, pos_label=1, zero_division=0))
    if recall_score:
        metrics['recall_macro'] = float(recall_score(ys, preds, average='macro', zero_division=0))
        metrics['sensitivity'] = float(recall_score(ys, preds, pos_label=1, zero_division=0))
        metrics['specificity'] = float(recall_score(ys, preds, pos_label=0, zero_division=0))
    if precision_score:
        metrics['precision'] = float(precision_score(ys, preds, average='macro', zero_division=0))
    if balanced_accuracy_score:
        metrics['balanced_accuracy'] = float(balanced_accuracy_score(ys, preds))
    
    # [Patch] AUC logic fix
    probs_arr = np.array(probs)
    if roc_auc_score and len(set(ys)) >= 2: # Fix: needs at least 2 classes
        try:
            if num_classes == 2:
                metrics['roc_auc'] = float(roc_auc_score(ys, probs_arr[:, 1]))
            else:
                metrics['roc_auc'] = float(roc_auc_score(ys, probs_arr, multi_class='ovr'))
        except Exception as e:
            metrics['roc_auc'] = np.nan
    else:
        metrics['roc_auc'] = np.nan
    if average_precision_score and len(set(ys)) >= 2:
        try:
            if num_classes == 2:
                metrics['pr_auc'] = float(average_precision_score(ys, probs_arr[:, 1]))
            else:
                metrics['pr_auc'] = float(average_precision_score(ys, probs_arr, average='macro'))
        except Exception:
            metrics['pr_auc'] = np.nan
    else:
        metrics['pr_auc'] = np.nan
    if matthews_corrcoef:
        try:
            metrics['mcc'] = float(matthews_corrcoef(ys, preds))
        except Exception:
            metrics['mcc'] = np.nan
    if 'sensitivity' in metrics and 'specificity' in metrics:
        metrics['g_mean'] = float(np.sqrt(metrics['sensitivity'] * metrics['specificity']))
    if confusion_matrix is not None and len(ys) > 0:
        cm = confusion_matrix(ys, preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['false_alarm_rate'] = float(fp / (fp + tn + 1e-12))
        else:
            metrics['false_alarm_rate'] = np.nan
    latency_stats = compute_detection_latency_stats(preds, ys, subs, stride, sample_rate)
    metrics['detection_latency_ms'] = latency_stats['mean_latency_ms']
    metrics['detection_latency_median_ms'] = latency_stats['median_latency_ms']
    metrics['detection_latency_std_ms'] = latency_stats['std_latency_ms']
    metrics['detection_latency_min_ms'] = latency_stats['min_latency_ms']
    metrics['detection_latency_max_ms'] = latency_stats['max_latency_ms']
    metrics['detection_rate'] = latency_stats['detection_rate']
    metrics['detection_total_falls'] = latency_stats['total_falls']
    metrics['detection_detected_falls'] = latency_stats['detected_falls']
    return metrics, ys, preds, probs, subs


def _make_perturb_fn_noise(snr_db: float):
    def fn(x: torch.Tensor) -> torch.Tensor:
        # Estimate signal power
        sig_power = x.pow(2).mean(dim=(1, 2), keepdim=True)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = sig_power / snr_linear
        noise = torch.randn_like(x) * torch.sqrt(noise_power + 1e-8)
        return x + noise
    return fn


def _make_perturb_fn_channel_dropout(drop_channels: List[int]):
    def fn(x: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(x)
        for ch in drop_channels:
            if ch < x.size(1):
                mask[:, ch, :] = 0
        return x * mask
    return fn


def _make_perturb_fn_data_dropout(rate: float):
    def fn(x: torch.Tensor) -> torch.Tensor:
        if rate <= 0:
            return x
        drop_len = max(1, int(x.size(-1) * rate))
        start = torch.randint(low=0, high=max(1, x.size(-1) - drop_len + 1), size=(1,)).item()
        x_clone = x.clone()
        x_clone[:, :, start:start+drop_len] = 0
        return x_clone
    return fn


def _make_perturb_fn_time_shift(shift_ms: float, sample_rate: float):
    def fn(x: torch.Tensor) -> torch.Tensor:
        shift = int(shift_ms * sample_rate / 1000.0)
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=-1)
    return fn


def robustness_evaluation(model: nn.Module, loader: DataLoader, device: torch.device, args: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Evaluate robustness under noise, channel dropout, data loss, and time shift.
    """
    results: Dict[str, Dict[str, float]] = {}
    if loader is None:
        return results
    sample_rate = float(args.get('sample_rate', 50.0))
    stride = args.get('stride')

    noise_levels = [30, 20, 10, 5, 0]
    for snr in noise_levels:
        metrics, _, _, _, _ = eval_model(model, loader, device, args['num_classes'], stride=stride, sample_rate=sample_rate, perturb_fn=_make_perturb_fn_noise(snr))
        results[f'noise_snr{snr}dB'] = metrics

    drop_cases = [[0], [1], [2], [0, 1]]
    for chs in drop_cases:
        metrics, _, _, _, _ = eval_model(model, loader, device, args['num_classes'], stride=stride, sample_rate=sample_rate, perturb_fn=_make_perturb_fn_channel_dropout(chs))
        results[f'channel_drop_{",".join(map(str, chs))}'] = metrics

    drop_rates = [0.05, 0.1, 0.2, 0.3]
    for dr in drop_rates:
        metrics, _, _, _, _ = eval_model(model, loader, device, args['num_classes'], stride=stride, sample_rate=sample_rate, perturb_fn=_make_perturb_fn_data_dropout(dr))
        results[f'data_dropout_{dr}'] = metrics

    shifts = [50, 100, 200, 500]
    for shift_ms in shifts:
        metrics, _, _, _, _ = eval_model(model, loader, device, args['num_classes'], stride=stride, sample_rate=sample_rate, perturb_fn=_make_perturb_fn_time_shift(shift_ms, sample_rate))
        results[f'time_shift_{shift_ms}ms'] = metrics

    return results

def make_dataloader(ds, batch_size, seed, shuffle=True, num_workers=0):
    g = torch.Generator()
    g.manual_seed(seed)

    # [Fix Windows] Windows上强制使用num_workers=0避免多进程死锁
    # Windows的multiprocessing spawn模式与PyTorch DataLoader存在兼容性问题
    if sys.platform == 'win32' and num_workers > 0:
        logging.warning(f"Windows平台检测到num_workers={num_workers}，自动设置为0以避免多进程死锁")
        num_workers = 0

    # 当num_workers > 0时使用persistent_workers减少进程创建开销
    persistent = num_workers > 0

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=worker_init_fn,
        generator=g,
        num_workers=num_workers,
        collate_fn=sensor_collate,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=persistent if num_workers > 0 else False
    )

def statistical_comparison(method_a_results: List[float], method_b_results: List[float]):
    if stats is None or len(method_a_results) != len(method_b_results):
        return {"t_test_p": None, "wilcoxon_p": None, "cohens_d": None, "significant": None}
    try:
        t_stat, p_value = stats.ttest_rel(method_a_results, method_b_results)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(method_a_results, method_b_results)
        diff = np.array(method_a_results) - np.array(method_b_results)
        cohens_d = diff.mean() / (diff.std(ddof=1) + 1e-12)
        return {
            "t_test_p": float(p_value),
            "wilcoxon_p": float(wilcoxon_p),
            "cohens_d": float(cohens_d),
            "significant": bool(p_value < 0.05),
        }
    except Exception:
        return {"t_test_p": None, "wilcoxon_p": None, "cohens_d": None, "significant": None}

def compute_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    vals = np.array(values, dtype=np.float64)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n < 2:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / np.sqrt(n))
    if stats is None:
        return mean, se, np.nan
    t_critical = float(stats.t.ppf((1 + confidence) / 2, df=n - 1))
    ci_half = t_critical * se
    return mean, se, ci_half


def interpret_cohens_d(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def comprehensive_statistical_tests(method_a: List[float], method_b: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    if stats is None:
        return {
            "normality": None,
            "paired_ttest": None,
            "wilcoxon": None,
            "cohens_d": None,
            "recommended_p": None,
            "significant": None,
            "effect_size": None,
        }
    arr_a = np.array(method_a, dtype=np.float64)
    arr_b = np.array(method_b, dtype=np.float64)
    mask = ~np.isnan(arr_a) & ~np.isnan(arr_b)
    arr_a, arr_b = arr_a[mask], arr_b[mask]
    if arr_a.size == 0 or arr_a.size != arr_b.size:
        return {
            "normality": None,
            "paired_ttest": None,
            "wilcoxon": None,
            "cohens_d": None,
            "recommended_p": None,
            "significant": None,
            "effect_size": None,
        }

    _, p_normal_a = stats.shapiro(arr_a) if arr_a.size >= 3 else (None, np.nan)
    _, p_normal_b = stats.shapiro(arr_b) if arr_b.size >= 3 else (None, np.nan)
    is_normal = bool(p_normal_a > 0.05 and p_normal_b > 0.05) if not (np.isnan(p_normal_a) or np.isnan(p_normal_b)) else False

    t_stat, p_ttest = stats.ttest_rel(arr_a, arr_b)
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(arr_a, arr_b)
    except Exception:
        w_stat, p_wilcoxon = np.nan, np.nan

    diff = arr_a - arr_b
    cohens_d_val = diff.mean() / (diff.std(ddof=1) + 1e-12) if diff.size > 1 else np.nan

    recommended_p = p_ttest if is_normal else p_wilcoxon
    return {
        "normality": {"a": None if np.isnan(p_normal_a) else float(p_normal_a), "b": None if np.isnan(p_normal_b) else float(p_normal_b), "is_normal": is_normal},
        "paired_ttest": {"t": float(t_stat), "p": float(p_ttest)},
        "wilcoxon": {"W": None if np.isnan(w_stat) else float(w_stat), "p": None if np.isnan(p_wilcoxon) else float(p_wilcoxon)},
        "cohens_d": None if np.isnan(cohens_d_val) else float(cohens_d_val),
        "effect_size": None if np.isnan(cohens_d_val) else interpret_cohens_d(float(cohens_d_val)),
        "recommended_p": None if np.isnan(recommended_p) else float(recommended_p),
        "significant": None if np.isnan(recommended_p) else bool(recommended_p < alpha),
    }


def compare_multiple_methods(method_results: Dict[str, List[float]], baseline_name: str, alpha: float = 0.05) -> List[Dict[str, Any]]:
    if multipletests is None or stats is None:
        return []
    if baseline_name not in method_results:
        return []
    baseline = method_results[baseline_name]
    raw_p_values = []
    comparisons: List[Dict[str, Any]] = []
    for name, values in method_results.items():
        if name == baseline_name:
            continue
        test_res = comprehensive_statistical_tests(values, baseline, alpha)
        test_res["method"] = name
        comparisons.append(test_res)
        raw_p_values.append(test_res.get("recommended_p", np.nan))

    # Multi-test correction
    raw_p = np.array(raw_p_values, dtype=np.float64)
    raw_p = raw_p[~np.isnan(raw_p)]
    if raw_p.size == 0:
        return comparisons
    bonf = [min(p * len(raw_p_values), 1.0) for p in raw_p_values]
    _, p_fdr, _, _ = multipletests(raw_p_values, method="fdr_bh")
    for idx, comp in enumerate(comparisons):
        comp["p_bonferroni"] = None if np.isnan(bonf[idx]) else float(bonf[idx])
        comp["p_fdr"] = None if np.isnan(p_fdr[idx]) else float(p_fdr[idx])
        comp["significant_bonferroni"] = None if np.isnan(bonf[idx]) else bool(bonf[idx] < alpha)
        comp["significant_fdr"] = None if np.isnan(p_fdr[idx]) else bool(p_fdr[idx] < alpha)
    return comparisons


def compare_metric_with_baseline(metric_vals: List[float], baseline_vals: List[float]) -> Dict[str, Optional[float]]:
    if stats is None or len(metric_vals) != len(baseline_vals):
        return {"t_stat": None, "p_ttest": None, "p_wilcoxon": None, "cohens_d": None, "p_bonferroni": None, "p_fdr": None, "significant": None}
    vals_a = np.array(metric_vals, dtype=np.float64)
    vals_b = np.array(baseline_vals, dtype=np.float64)
    mask = ~np.isnan(vals_a) & ~np.isnan(vals_b)
    vals_a, vals_b = vals_a[mask], vals_b[mask]
    if len(vals_a) == 0:
        return {"t_stat": None, "p_ttest": None, "p_wilcoxon": None, "cohens_d": None, "p_bonferroni": None, "p_fdr": None, "significant": None}
    normal_a = stats.shapiro(vals_a)[1] if len(vals_a) >= 3 else np.nan
    normal_b = stats.shapiro(vals_b)[1] if len(vals_b) >= 3 else np.nan
    is_normal = (normal_a > 0.05) and (normal_b > 0.05) if not (np.isnan(normal_a) or np.isnan(normal_b)) else False

    t_stat, p_t = stats.ttest_rel(vals_a, vals_b)
    try:
        w_stat, p_w = stats.wilcoxon(vals_a, vals_b)
    except Exception:
        w_stat, p_w = np.nan, np.nan
    diff = vals_a - vals_b
    d_val = float(diff.mean() / (diff.std(ddof=1) + 1e-12)) if len(diff) > 1 else np.nan
    recommended_p = p_t if is_normal else p_w

    p_bonf, p_fdr = None, None
    if multipletests is not None and not np.isnan(recommended_p):
        _, p_fdr_vals, _, _ = multipletests([recommended_p], method='fdr_bh')
        p_bonf_vals = [min(recommended_p * 1, 1.0)]
        p_bonf = float(p_bonf_vals[0])
        p_fdr = float(p_fdr_vals[0])
    significant = (p_fdr if p_fdr is not None else recommended_p) < 0.05 if not np.isnan(recommended_p) else None
    return {
        "t_stat": float(t_stat),
        "p_ttest": float(p_t),
        "p_wilcoxon": float(p_w) if not np.isnan(p_w) else None,
        "cohens_d": d_val,
        "p_bonferroni": p_bonf,
        "p_fdr": p_fdr,
        "recommended_p": None if np.isnan(recommended_p) else float(recommended_p),
        "normality": {"a": None if np.isnan(normal_a) else float(normal_a), "b": None if np.isnan(normal_b) else float(normal_b), "is_normal": bool(is_normal)},
        "significant": bool(significant) if significant is not None else None,
    }

def aggregate_loso_results(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate LOSO metrics with t-based CIs (aligned with 2.md guidance).
    """
    if not fold_results:
        return {}
    metric_keys = set()
    for fr in fold_results:
        metric_keys.update(fr.get('metrics', {}).keys())
    summary: Dict[str, Any] = {}
    for key in sorted(metric_keys):
        values = np.array([fr.get('metrics', {}).get(key, np.nan) for fr in fold_results], dtype=np.float64)
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        base_mean = float(np.mean(values))
        mean, se, ci_half = compute_confidence_interval(values)
        summary[f"{key}_mean"] = base_mean if not np.isnan(base_mean) else mean
        summary[f"{key}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        summary[f"{key}_se"] = se
        summary[f"{key}_ci_half"] = None if np.isnan(ci_half) else float(ci_half)
        summary[f"{key}_ci_lower"] = None if np.isnan(ci_half) else float(base_mean - ci_half)
        summary[f"{key}_ci_upper"] = None if np.isnan(ci_half) else float(base_mean + ci_half)
        summary[f"{key}_n_folds"] = int(values.size)
    return summary


def save_loso_results(out_dir: str, seed: int, folds: List[Dict[str, Any]], predictions: Dict[str, List[Any]], summary: Dict[str, Any]):
    payload = {
        "seed": seed,
        "summary": summary,
        "folds": folds,
    }
    if predictions:
        payload["predictions"] = predictions
    path = os.path.join(out_dir, f"loso_results_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logging.info(f"Saved LOSO fold results to {path}")

def aggregate_runs(results, baseline: Optional[List[float]] = None):
    """
    [Patch D] Robust stats aggregation handling NaNs + optional significance vs baseline.
    """
    if not results: return {}
    all_keys = set().union(*(r['test_metrics'].keys() for r in results))
    summary = {}
    per_metric_values: Dict[str, List[float]] = {k: [] for k in all_keys}
    for k in sorted(all_keys):
        vals = np.array([r['test_metrics'].get(k, np.nan) for r in results], dtype=np.float64)
        # Filter NaNs
        valid = vals[~np.isnan(vals)]
        n = len(valid)
        if n > 0:
            base_mean = float(np.mean(valid))
            mean, se, ci_half = compute_confidence_interval(valid)
            summary[f"{k}_mean"] = base_mean if not np.isnan(base_mean) else mean
            summary[f"{k}_std"] = float(np.std(valid, ddof=0)) # Population std for simple view
            summary[f"{k}_se"] = se
            summary[f"{k}_n"] = n
            per_metric_values[k] = valid.tolist()
            summary[f"{k}_95ci"] = None if np.isnan(ci_half) else float(ci_half)
            summary[f"{k}_ci_lower"] = None if np.isnan(ci_half) else float(base_mean - ci_half)
            summary[f"{k}_ci_upper"] = None if np.isnan(ci_half) else float(base_mean + ci_half)
        else:
            summary[f"{k}_mean"] = np.nan
    # optional significance vs baseline for macro_f1
    if baseline and per_metric_values.get("macro_f1"):
        curr = per_metric_values["macro_f1"]
        if len(curr) == len(baseline):
            summary["macro_f1_significance"] = comprehensive_statistical_tests(curr, baseline)
            summary["macro_f1_significance_multi"] = compare_multiple_methods(
                {"current": curr, "baseline": baseline},
                baseline_name="baseline",
            )
        else:
            summary["macro_f1_significance"] = {"error": "baseline_length_mismatch", "current_n": len(curr), "baseline_n": len(baseline)}
    return summary

def summarize_split(dataset: Dataset) -> Dict[str, Dict]:
    """
    Summarize dataset distribution by label and subject.
    """
    label_counts: Dict[int, int] = {}
    subject_counts: Dict[str, int] = {}
    items = getattr(dataset, "items", [])
    for _, y, subj in items:
        label_counts[int(y)] = label_counts.get(int(y), 0) + 1
        subject_counts[subj] = subject_counts.get(subj, 0) + 1
    return {
        "total": len(dataset),
        "label_counts": label_counts,
        "subject_counts": subject_counts,
    }

def save_split_stats(out_dir: str, seed: int, split_stats: Dict[str, Dict]):
    path = os.path.join(out_dir, f"split_stats_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(split_stats, f, indent=2)
    logging.info(f"Saved split stats to {path}")

def save_confusion_and_errors(ys: List[int], preds: List[int], probs: List[List[float]], out_dir: str, seed: int, split_tag: str = "test", subjects: Optional[List[str]] = None):
    if confusion_matrix is not None and len(ys) > 0:
        cm = confusion_matrix(ys, preds)
        cm_path = os.path.join(out_dir, f"confusion_matrix_seed{seed}_{split_tag}.npy")
        np.save(cm_path, cm)
        logging.info(f"Saved confusion matrix to {cm_path}")
    errors_path = os.path.join(out_dir, f"errors_seed{seed}_{split_tag}.csv")
    with open(errors_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "subject", "y_true", "y_pred", "prob_fall"])
        for idx, (y, p, prob) in enumerate(zip(ys, preds, probs)):
            prob_fall = prob[1] if len(prob) > 1 else prob[0]
            subj = subjects[idx] if subjects and idx < len(subjects) else ""
            writer.writerow([idx, subj, y, p, prob_fall])
    logging.info(f"Saved error analysis to {errors_path}")

def save_efficiency_report(report: Dict[str, float], out_dir: str, seed: int):
    path = os.path.join(out_dir, f"efficiency_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"Saved efficiency report to {path}")

def generate_complexity_comparison_table(models_dict: Dict[str, nn.Module], input_size: Tuple[int, int, int]):
    rows = []
    for name, model in models_dict.items():
        eff = profile_model_efficiency(model, input_size=input_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        rows.append({
            'Model': name,
            'Params (M)': eff['params_M'],
            'FLOPs (G)': eff['flops_G'],
            'Latency (ms)': eff['latency_ms'],
        })
    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except ImportError:
        return rows

def visualize_for_paper(model: nn.Module, sample_input: torch.Tensor, z_time: Optional[np.ndarray] = None, z_freq: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None, out_dir: str = "./outputs"):
    """
    Helper visualizations: feature embedding, attention weights, freq response, confusion matrix/ROC.
    """
    ensure_dir(out_dir)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.manifold import TSNE
        from sklearn.metrics import confusion_matrix as sk_cm, roc_curve, precision_recall_curve, auc
    except ImportError as e:
        logging.warning(f"Visualization skipped (missing dependency): {e}")
        return

    if z_time is not None and z_freq is not None and labels is not None:
        emb = np.concatenate([z_time, z_freq], axis=1)
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto")
        proj = tsne.fit_transform(emb)
        plt.figure()
        sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=labels, palette="coolwarm", s=12)
        plt.title("Time-Freq Embedding (t-SNE)")
        plt.savefig(os.path.join(out_dir, "tsne_embedding.png"), dpi=300, bbox_inches='tight')
        plt.close()

    if hasattr(model, "mspa") and getattr(model.mspa, "last_band_weights", None) is not None:
        weights = model.mspa.last_band_weights.mean(dim=(0, 2, 3)).cpu().numpy()
        plt.figure()
        sns.barplot(x=np.arange(len(weights)), y=weights)
        plt.title("MSPA Band Weights")
        plt.savefig(os.path.join(out_dir, "mspa_band_weights.png"), dpi=300, bbox_inches='tight')
        plt.close()

    if hasattr(model, "faa") and getattr(model.faa, "last_attention", None) is not None:
        attn = model.faa.last_attention.mean(dim=0).cpu().numpy()
        plt.figure()
        sns.heatmap(attn, cmap="mako")
        plt.title("FAA Attention Map")
        plt.savefig(os.path.join(out_dir, "faa_attention.png"), dpi=300, bbox_inches='tight')
        plt.close()

    if labels is not None and z_time is not None:
        preds = np.argmax(z_time, axis=1) if z_time.ndim == 2 else None
        if preds is not None:
            cm = sk_cm(labels, preds)
            plt.figure()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.savefig(os.path.join(out_dir, "confusion_heatmap.png"), dpi=300, bbox_inches='tight')
            plt.close()

            fpr, tpr, _ = roc_curve(labels, preds)
            prec, rec, _ = precision_recall_curve(labels, preds)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("ROC")
            plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=300, bbox_inches='tight')
            plt.close()
            plt.figure()
            plt.plot(rec, prec)
            plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve")
            plt.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=300, bbox_inches='tight')
            plt.close()


class GradCAM1D:
    """
    1D Grad-CAM for time-series models.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[torch.Tensor] = None):
        self.model.eval()
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        if target_class is None:
            target_class = output.argmax(dim=1)
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        for i in range(output.size(0)):
            cls_idx = int(target_class[i]) if isinstance(target_class, torch.Tensor) else int(target_class)
            one_hot[i, cls_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured for Grad-CAM.")
        weights = self.gradients.mean(dim=-1, keepdim=True)  # (B, C, 1)
        cam = (weights * self.activations).sum(dim=1)  # (B, L)
        cam = F.relu(cam)
        cam = cam / (cam.max(dim=-1, keepdim=True)[0] + 1e-8)
        return cam.detach().cpu().numpy()


def visualize_kernel_routing(model: nn.Module, dataloader: DataLoader, save_path: str):
    """
    Visualize dynamic kernel routing weights for interpretability.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        logging.warning(f"Routing visualization skipped (matplotlib missing): {e}")
        return

    device = next(model.parameters()).device
    model.eval()
    all_weights = {"fall": [], "adl": []}
    kernel_sizes = None

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            x, labels, _ = batch
            if x is None:
                continue
            x = x.to(device)
            labels_np = labels.cpu().numpy()
            _ = model(x)
            for module in model.modules():
                weights = getattr(module, "last_routing_weights", None)
                if weights is None:
                    continue
                if kernel_sizes is None and hasattr(module, "kernel_sizes"):
                    kernel_sizes = list(module.kernel_sizes)
                w_np = weights.detach().cpu().numpy()
                for i, lbl in enumerate(labels_np):
                    key = "fall" if lbl == 1 else "adl"
                    all_weights[key].append(w_np[i])

    if not all_weights["fall"] and not all_weights["adl"]:
        logging.warning("No routing weights captured; ensure model uses DynamicKernelBlock.")
        return
    kernel_sizes = kernel_sizes or list(range(len(all_weights["fall"][0]) if all_weights["fall"] else len(all_weights["adl"][0])))
    fall_mean = np.mean(all_weights["fall"], axis=0) if all_weights["fall"] else np.zeros(len(kernel_sizes))
    adl_mean = np.mean(all_weights["adl"], axis=0) if all_weights["adl"] else np.zeros(len(kernel_sizes))

    x_pos = np.arange(len(kernel_sizes))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x_pos - width / 2, fall_mean, width, label="Fall", color="red", alpha=0.7)
    plt.bar(x_pos + width / 2, adl_mean, width, label="ADL", color="blue", alpha=0.7)
    plt.xlabel("Kernel Size")
    plt.ylabel("Average Routing Weight")
    plt.xticks(x_pos, kernel_sizes)
    plt.legend()
    plt.title("Dynamic Kernel Routing Weights by Activity Type")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def visualize_tsne(model: nn.Module, dataloader: DataLoader, save_path: str, stage: str = "final"):
    """
    t-SNE visualization for time/frequency embeddings.
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError as e:
        logging.warning(f"t-SNE visualization skipped (dependency missing): {e}")
        return

    device = next(model.parameters()).device
    model.eval()
    features = []
    labels_list: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            x, labels, _ = batch
            if x is None:
                continue
            x = x.to(device)
            labels_list.extend(labels.cpu().numpy().tolist())
            outputs = model(x)
            if not isinstance(outputs, tuple) or len(outputs) < 3:
                continue
            _, z_time_list, z_freq_list = outputs
            if isinstance(z_time_list, torch.Tensor):
                z_time_list = [z_time_list]
            if isinstance(z_freq_list, torch.Tensor):
                z_freq_list = [z_freq_list]
            if not z_time_list or not z_freq_list:
                continue
            if stage == "time":
                feat = z_time_list[-1]
            elif stage == "freq":
                feat = z_freq_list[-1]
            else:
                feat = torch.cat([z_time_list[-1], z_freq_list[-1]], dim=-1)
            if feat.dim() > 2:
                feat = feat.mean(dim=tuple(range(2, feat.dim())))
            features.append(feat.detach().cpu().numpy())

    if not features:
        logging.warning("No features collected for t-SNE visualization.")
        return

    features = np.concatenate(features, axis=0)
    labels_arr = np.array(labels_list)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    proj = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red']
    markers = ['o', '^']
    class_names = ['ADL', 'Fall']
    for cls in [0, 1]:
        mask = labels_arr == cls
        plt.scatter(proj[mask, 0], proj[mask, 1], c=colors[cls], marker=markers[cls], label=class_names[cls], alpha=0.6, s=50)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.title(f"t-SNE Visualization of {stage.capitalize()} Features")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def _select_gradcam_layer(model: nn.Module) -> Optional[nn.Module]:
    # Prefer last AMS block if available
    if hasattr(model, "stage3") and isinstance(model.stage3, nn.ModuleList) and len(model.stage3) > 0:
        return model.stage3[-1]
    if hasattr(model, "blocks") and isinstance(model.blocks, nn.ModuleList) and len(model.blocks) > 0:
        return model.blocks[-1]
    return None
def build_augmentation(config) -> Optional[SensorAugmentation]:
    if not config.get("augment", False):
        return None
    scale_range = config.get("scale_range", (0.9, 1.1))
    if isinstance(scale_range, (list, tuple)) and len(scale_range) == 2:
        scale_range = (float(scale_range[0]), float(scale_range[1]))
    else:
        scale_range = (0.9, 1.1)
    return SensorAugmentation(
        noise_std=float(config.get("noise_std", 0.05)),
        scale_range=scale_range,
        time_shift_ratio=float(config.get("time_shift_ratio", 0.1)),
        drop_prob=float(config.get("drop_prob", 0.1)),
    )

def load_preloaded_splits(root: str, name: str, seed: int, channels_used: str):
    """
    Load preprocessed splits for MobiAct/UniMiB/KFall style datasets.
    Expected files under root/name/: train.npz, val.npz, test.npz
    Fallback to single all.npz and random split.
    """
    base_dir = os.path.join(root, name)
    train_path = os.path.join(base_dir, "train.npz")
    val_path = os.path.join(base_dir, "val.npz")
    test_path = os.path.join(base_dir, "test.npz")
    if all(os.path.isfile(p) for p in [train_path, val_path, test_path]):
        train_ds = PreloadedHARDataset(train_path, channels_used)
        val_ds = PreloadedHARDataset(val_path, channels_used)
        test_ds = PreloadedHARDataset(test_path, channels_used)
        C = train_ds.items[0][0].shape[0]
        return train_ds, val_ds, test_ds, C
    all_path = os.path.join(base_dir, "data.npz")
    if not os.path.isfile(all_path):
        all_path = os.path.join(base_dir, f"{name}.npz")
    full_ds = PreloadedHARDataset(all_path, channels_used)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(full_ds))
    rng.shuffle(idx)
    n = len(idx)
    val_split = max(1, int(0.15 * n))
    test_split = max(1, int(0.15 * n))
    train_idx = idx[: n - val_split - test_split]
    val_idx = idx[n - val_split - test_split : n - test_split]
    test_idx = idx[n - test_split :]
    train_items = [full_ds.items[i] for i in train_idx]
    val_items = [full_ds.items[i] for i in val_idx]
    test_items = [full_ds.items[i] for i in test_idx]
    train_ds = DummyDataset(train_items)
    val_ds = DummyDataset(val_items)
    test_ds = DummyDataset(test_items)
    C = full_ds.items[0][0].shape[0]
    return train_ds, val_ds, test_ds, C


def evaluate_cross_datasets(
    model: nn.Module,
    config: Dict,
    device: torch.device,
    seed: int,
    base_dataset: str,
    in_channels: int,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a trained model on other datasets without retraining to measure cross-dataset generalization.
    """
    candidate_datasets = ['sisfall', 'mobiact', 'unimib', 'kfall']
    target_datasets = [d for d in candidate_datasets if d != base_dataset]
    results: Dict[str, Dict[str, float]] = {}
    for ds_name in target_datasets:
        try:
            if ds_name == 'sisfall':
                try:
                    sis_root = _resolve_sisfall_root(config['data_root'])
                except Exception as e:
                    logging.warning(f"[cross] Skip SisFall evaluation: {e}")
                    continue
                test_ds = SisFallDataset(
                    sis_root,
                    subjects=[],
                    window_size=config['window_size'],
                    stride=config['stride'],
                    log_dir=config['out_dir'],
                    channels_used=config.get('channels_used', 'accel3'),
                    transform=None,
                )
            else:
                _, _, test_ds, C_loaded = load_preloaded_splits(config['data_root'], ds_name, seed, config.get('channels_used', 'accel3'))
                if C_loaded != in_channels:
                    logging.warning(f"[cross] Skip {ds_name}: channel mismatch (model expects {in_channels}, dataset has {C_loaded}).")
                    continue
            if len(test_ds) == 0:
                logging.warning(f"[cross] Skip {ds_name}: empty dataset.")
                continue
            test_c = test_ds.items[0][0].shape[0] if getattr(test_ds, "items", None) else in_channels
            if test_c != in_channels:
                logging.warning(f"[cross] Skip {ds_name}: channel mismatch (model expects {in_channels}, dataset has {test_c}).")
                continue
            loader = make_dataloader(test_ds, config['batch_size'], seed, False, config['num_workers'])
            metrics, ys, preds, probs, subs = eval_model(
                model,
                loader,
                device,
                config['num_classes'],
                stride=config.get('stride'),
                sample_rate=float(config.get('sample_rate', 50.0)),
            )
            results[ds_name] = metrics
            save_confusion_and_errors(ys, preds, probs, config['out_dir'], seed, f"cross_{ds_name}", subs)
            logging.info(f"[cross] {base_dataset} -> {ds_name}: {metrics}")
        except Exception as e:
            logging.warning(f"[cross] Evaluation on {ds_name} failed: {e}")
    if results:
        path = os.path.join(config['out_dir'], f"cross_eval_seed{seed}.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Saved cross-dataset evaluation to {path}")
    return results

def run_one_experiment(config, seed, resume_path=None):
    set_seed(seed, config.get('deterministic', False))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Common settings ---
    channels_used = config.get('channels_used', 'accel3')
    eval_mode = config.get('eval_mode', 'holdout')
    L = config['window_size']
    C = config['in_channels']
    fusion_kernel_sizes = tuple(config.get('fusion_kernel_sizes', (3, 5, 7)))
    band_edges_cfg = tuple(config['band_edges']) if config.get('band_edges') else None
    num_bands_cfg = int(config.get('num_bands', 4))
    fusion_variant = config.get('fusion_variant', 'enhanced')
    adaptive_bands = config.get('adaptive_bands', True)
    faa_axis_attn = config.get('faa_axis_attn', True)
    if config['dataset'] == 'sisfall':
        C = _channels_from_mode(channels_used)
    augmentor = build_augmentation(config)
    sisfall_root = None
    if config['dataset'] == 'sisfall':
        sisfall_root = _resolve_sisfall_root(config['data_root'])

    def train_eval_split(train_ds, val_ds, test_ds, split_tag: str, resume_path_split: Optional[str]):
        train_dl = make_dataloader(train_ds, config['batch_size'], seed, True, config['num_workers'])
        val_dl = make_dataloader(val_ds, config['batch_size'], seed, False, config['num_workers'])
        test_dl = make_dataloader(test_ds, config['batch_size'], seed, False, config['num_workers'])

        # --- Model Setup ---
        ablation_cfg = config.get('ablation', {}) or {}
        internal_model = config.get('model_internal', config['model'])
        model_display_name = config.get('model_display_name', internal_model)
        if internal_model == 'dmc':
            model = DMCNet(
                in_channels=C,
                channels=config['channels'],
                n_blocks=config['n_blocks'],
                num_classes=config['num_classes'],
                freq_method=config.get('freq_method', 'fft'),
                use_dks=ablation_cfg.get('dks', True),
                use_freq_branch=ablation_cfg.get('mspa', True),
                kernel_sizes=tuple(config.get('kernel_sizes', (7, 15, 31, 63))),
                sample_rate=float(config.get('sample_rate', 50.0)),
                adaptive_bands=adaptive_bands,
                band_edges=band_edges_cfg,
                num_bands=num_bands_cfg,
                fusion_variant=fusion_variant,
                fusion_kernel_sizes=fusion_kernel_sizes,
            )
        elif internal_model == 'lstm':
            model = LSTMClassifier(in_channels=C, num_classes=config['num_classes'])
        elif internal_model == 'resnet':
            model = ResNet1D(in_channels=C, num_classes=config['num_classes'])
        elif internal_model == 'amsv2':
            model = AMSNetV2(
                in_channels=C,
                num_classes=config['num_classes'],
                proj_dim=config.get('proj_dim', 128),
                ablation=ablation_cfg,
                freq_method=config.get('freq_method', 'fft'),
                sample_rate=float(config.get('sample_rate', 50.0)),
                time_attn=config.get('attn_time'),
                freq_attn=config.get('attn_freq'),
                fusion_attn=config.get('attn_fuse'),
                fusion_variant=fusion_variant,
                fusion_kernel_sizes=fusion_kernel_sizes,
                adaptive_bands=adaptive_bands,
                band_edges=band_edges_cfg,
                num_bands=num_bands_cfg,
                faa_axis_attn=faa_axis_attn,
            )
        elif internal_model == 'liteams':
            model = LiteAMSNet(
                in_channels=C,
                num_classes=config['num_classes'],
                channels=max(16, config.get('channels', 32)),
                n_blocks=max(1, config.get('n_blocks', 2)),
                attn=config.get('attn_lite'),
            )
        elif internal_model == 'tcn':
            model = TemporalConvNet(in_channels=C, num_classes=config['num_classes'], channels=config['channels'], depth=config.get('n_blocks', 4))
        elif internal_model == 'transformer':
            model = TransformerClassifier(in_channels=C, num_classes=config['num_classes'], d_model=max(64, config['channels']), nhead=max(1, config['channels']//32), num_layers=config.get('n_blocks', 2))
        elif internal_model == 'inceptiontime':
            model = InceptionTime(in_channels=C, num_classes=config['num_classes'], channels=max(32, config['channels']//2), depth=config.get('n_blocks', 3))
        elif internal_model == 'rocket':
            model = RocketClassifier(in_channels=C, num_classes=config['num_classes'], num_kernels=config.get('rocket_kernels', 256), kernel_size=config.get('rocket_kernel_size', 9))
        elif internal_model == 'tinyhar':
            model = TinyHAR(in_channels=C, num_classes=config['num_classes'], channels=max(32, config['channels']//2))
        elif internal_model == 'deeplstm':
            model = DeepConvLSTM(in_channels=C, num_classes=config['num_classes'], conv_channels=config.get('channels', 64), lstm_hidden=config.get('lstm_hidden', 128))
        else:
            raise ValueError(f"Unknown model: {internal_model}")
            
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"[{split_tag}] Model: {model_display_name} (internal={internal_model})")
        logging.info(f"[{split_tag}]   Total params: {total_params/1e6:.2f}M")
        logging.info(f"[{split_tag}]   Trainable params: {trainable_params/1e6:.2f}M")

        # --- Optimizer & Loss ---
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
        scaler = _make_grad_scaler(AMP_DEVICE_TYPE) if config['amp'] and device.type == 'cuda' else None
        
        class_weights = None
        class_weighting = config.get('class_weighting')
        if class_weighting is not None:
            if str(class_weighting).lower() == 'none':
                logging.info(f"[{split_tag}] Class weighting disabled via strategy=none.")
            else:
                logging.info(f"[{split_tag}] Using class weighting strategy: {class_weighting}.")
                class_weights = compute_class_weights_by_strategy(
                    train_ds,
                    class_weighting,
                    num_classes=config['num_classes'],
                    beta=float(config.get('effective_num_beta', 0.999)),
                ).to(device)
        elif config.get('weighted_loss', False):
            logging.info(f"[{split_tag}] Using weighted CrossEntropyLoss based on training data balance.")
            class_weights = compute_class_weights(train_ds, num_classes=config['num_classes']).to(device)

        if internal_model == 'amsv2':
            use_tfcl_flag = config.get('use_tfcl', False) and ablation_cfg.get('tfcl', True)
            beta_val = config.get('loss_beta', 0.01) if ablation_cfg.get('center', True) else 0.0
            use_uw = config.get('uncertainty_weighting', False)
            if use_uw:
                logging.info(f"[{split_tag}] Using uncertainty weighting (Kendall et al.) for automatic loss balancing")
            criterion = AMSNetLoss(
                num_classes=config['num_classes'],
                feat_dim=config.get('proj_dim', 128),
                alpha=config.get('loss_alpha', 0.1),
                beta=beta_val,
                use_tfcl=use_tfcl_flag,
                hierarchical_tfcl=config.get('tf_hierarchical', True),
                tf_cross_weight=config.get('tf_cross_weight', 0.3),
                temperature=config.get('tf_temperature', 0.1),
                supervised_weight=config.get('tf_supervised_weight', 0.1),
                label_smoothing=config.get('label_smoothing', 0.0),
                class_weights=class_weights,
                use_uncertainty_weighting=use_uw,
            ).to(device)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        start_epoch = 1
        best_f1 = -1.0
        
        # [Fix 3] Resume Logic with Seed Check
        if resume_path_split and os.path.exists(resume_path_split):
            ckpt_meta = torch_load_full(resume_path_split, map_location='cpu')
            ckpt_seed = ckpt_meta.get('seed', None)
            if ckpt_seed is not None and ckpt_seed != seed:
                logging.warning(f"[{split_tag}] Resume Warning: Checkpoint seed {ckpt_seed} != Current seed {seed}. Skipping resume.")
            else:
                start_epoch, best_f1 = load_checkpoint_for_resume(resume_path_split, model, optimizer, scheduler, scaler, device)
        elif resume_path_split:
            logging.warning(f"[{split_tag}] Resume path provided but not found: {resume_path_split} (start fresh)")

        # --- Training Loop ---
        warmup_epochs = max(0, int(config.get('warmup_epochs', 0)))
        early = AdaptiveEarlyStopping(
            initial_patience=config.get('patience', 15),
            min_patience=config.get('min_patience', 5),
            decay_factor=config.get('patience_decay', 0.9),
        )

        logging.info(f"[{split_tag}] Starting training: epochs={config['epochs']}, batch_size={config['batch_size']}, "
                     f"num_workers={config['num_workers']}, train_samples={len(train_ds)}")

        for ep in range(start_epoch, config['epochs']+1):
            if warmup_epochs > 0 and ep <= warmup_epochs:
                warmup_factor = ep / float(warmup_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config['lr'] * warmup_factor
            loss, loss_parts = train_epoch(model, train_dl, optimizer, device, criterion, scaler, config['accum_steps'])
            if warmup_epochs == 0 or ep > warmup_epochs:
                scheduler.step()
            
            metrics, _, _, _, _ = eval_model(
                model,
                val_dl,
                device,
                config['num_classes'],
                stride=config.get('stride'),
                sample_rate=float(config.get('sample_rate', 50.0)),
            )
            val_f1 = metrics.get('macro_f1', 0)
            
            # Save Checkpoint
            state = {
                'seed': seed, # Save seed for validation
                'epoch': ep, 
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict() if scaler else None,
                'best_f1': max(best_f1, val_f1),
                'rng_states': get_rng_states()
            }
            ckpt_last = os.path.join(config['out_dir'], f"ckpt_last_seed{seed}_{split_tag}.pth")
            ckpt_best = os.path.join(config['out_dir'], f"ckpt_best_seed{seed}_{split_tag}.pth")
            torch.save(state, ckpt_last)
            
            improved = val_f1 > best_f1
            if improved:
                best_f1 = val_f1
                torch.save(state, ckpt_best)
            stop = early.step(improved)
                
            extra_loss = " ".join([f"{k}={v:.4f}" for k, v in loss_parts.items() if k != 'total'])
            loss_msg = f"Loss={loss:.4f}"
            if extra_loss:
                loss_msg += f" ({extra_loss})"
            logging.info(f"[{split_tag}] Seed {seed} | Ep {ep}: {loss_msg} Val_F1={val_f1:.4f} Best_F1={best_f1:.4f}")
            if stop: 
                logging.info(f"[{split_tag}] Early stopping at epoch {ep}")
                break

        # --- Final Test ---
        best_path = os.path.join(config['out_dir'], f"ckpt_best_seed{seed}_{split_tag}.pth")
        if os.path.exists(best_path):
            ckpt = torch_load_full(best_path, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            logging.info(f"[{split_tag}] Loaded best checkpoint from epoch {ckpt['epoch']}")
        
        test_metrics, ys_test, preds_test, probs_test, subs_test = eval_model(
            model,
            test_dl,
            device,
            config['num_classes'],
            stride=config.get('stride'),
            sample_rate=float(config.get('sample_rate', 50.0)),
        )
        logging.info(f"[{split_tag}] Seed {seed} Test Results: {test_metrics}")
        save_confusion_and_errors(ys_test, preds_test, probs_test, config['out_dir'], seed, split_tag, subs_test)
        robustness_results = {}
        if config.get('run_robustness', False):
            robustness_results = robustness_evaluation(model, test_dl, device, config)
            rob_path = os.path.join(config['out_dir'], f"robustness_seed{seed}_{split_tag}.json")
            with open(rob_path, "w") as f:
                json.dump(robustness_results, f, indent=2)
            logging.info(f"[{split_tag}] Robustness results saved to {rob_path}")
        if config.get('viz_tsne', False):
            tsne_path = os.path.join(config['out_dir'], f"tsne_seed{seed}_{split_tag}.png")
            visualize_tsne(model, test_dl, tsne_path, stage=config.get('viz_tsne_stage', 'final'))
            logging.info(f"[{split_tag}] t-SNE saved to {tsne_path}")
        if config.get('viz_routing', False):
            routing_path = os.path.join(config['out_dir'], f"routing_seed{seed}_{split_tag}.png")
            visualize_kernel_routing(model, test_dl, routing_path)
            logging.info(f"[{split_tag}] Routing visualization saved to {routing_path}")
        if config.get('viz_gradcam', False):
            target_layer = _select_gradcam_layer(model)
            if target_layer is not None:
                try:
                    gradcam = GradCAM1D(model, target_layer)
                    for batch in test_dl:
                        if batch is None:
                            continue
                        x_gc, labels_gc, _ = batch
                        if x_gc is None:
                            continue
                        x_gc = x_gc.to(device)
                        cam = gradcam.generate(x_gc[:1], None)
                        gc_path = os.path.join(config['out_dir'], f"gradcam_seed{seed}_{split_tag}.npy")
                        np.save(gc_path, cam)
                        logging.info(f"[{split_tag}] Grad-CAM saved to {gc_path}")
                        break
                except Exception as e:
                    logging.warning(f"[{split_tag}] Grad-CAM generation failed: {e}")
        
        # Efficiency Profile (Only once)
        if seed == config['seeds'][0] and config.get('profile', False) and split_tag == "main":
            eff = profile_model_efficiency(model, (1, C, L), device)
            save_efficiency_report(eff, config['out_dir'], seed)
        cross_eval_results = {}
        if config.get('cross_eval', False) and not split_tag.startswith("loso"):
            cross_eval_results = evaluate_cross_datasets(model, config, device, seed, config['dataset'], C)

        return {
            'seed': seed,
            'test_metrics': test_metrics,
            'split': split_tag,
            'cross_eval': cross_eval_results,
            'y_true': ys_test,
            'y_pred': preds_test,
            'y_prob': probs_test,
            'subjects': subs_test,
        }

    results = []
    split_stats_total: Dict[str, Dict] = {}

    if config['dataset'] == 'dryrun':
        def make_dummy(n_samples, rng_seed):
            rng = np.random.default_rng(rng_seed) 
            X = rng.standard_normal((n_samples, C, L), dtype=np.float32)
            y = rng.integers(0, 2, size=(n_samples,))
            return [(X[i], int(y[i]), "synth") for i in range(n_samples)]
        
        train_ds = DummyDataset(make_dummy(200, seed))
        val_ds = DummyDataset(make_dummy(50, seed+1))
        test_ds = DummyDataset(make_dummy(50, seed+2))
        split_stats_total["train"] = summarize_split(train_ds)
        split_stats_total["val"] = summarize_split(val_ds)
        split_stats_total["test"] = summarize_split(test_ds)
        resume_path_split = resolve_resume_path(resume_path, config['out_dir'], seed, "main")
        res = train_eval_split(train_ds, val_ds, test_ds, "main", resume_path_split)
        results.append(res)

    elif config['dataset'] == 'sisfall':
        subjects = []
        for folder in ('ADL', 'FALL'):
            folder_path = os.path.join(sisfall_root, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if not fname.lower().endswith('.txt'):
                    continue
                subj = _parse_sisfall_subject_from_name(fname)
                subjects.append(subj)
        unique_subjects = sorted(set(subjects))
        if len(unique_subjects) < 3:
            raise RuntimeError(
                "Found fewer than 3 unique subjects in SisFall data_root; "
                "unable to perform train/val/test split."
            )

        if eval_mode == 'loso':
            logging.info(f"Running LOSO across {len(unique_subjects)} subjects.")
            loso_records: List[Dict] = []
            loso_predictions: Dict[str, List[Any]] = {"y_true": [], "y_pred": [], "y_prob": [], "subject": [], "fold": []}
            max_folds = config.get('loso_max_folds')
            for fold_idx, test_subj in enumerate(unique_subjects):
                if max_folds is not None and fold_idx >= max_folds:
                    logging.info(f"[loso] Reached loso_max_folds={max_folds}, stopping early.")
                    break
                train_val_subjs = [s for s in unique_subjects if s != test_subj]
                if len(train_val_subjs) < 2:
                    logging.warning(f"Skipping LOSO fold for {test_subj} (insufficient remaining subjects).")
                    continue
                rng = random.Random(seed + fold_idx)
                rng.shuffle(train_val_subjs)
                split_point = max(1, int(0.8 * len(train_val_subjs)))
                train_subjs = train_val_subjs[:split_point]
                val_subjs = train_val_subjs[split_point:]
                if not val_subjs:
                    val_subjs = train_subjs[-1:]
                    train_subjs = train_subjs[:-1] or train_subjs

                train_ds = SisFallDataset(sisfall_root, train_subjs, config['window_size'], config['stride'], config['out_dir'], channels_used, augmentor)
                val_ds = SisFallDataset(sisfall_root, val_subjs, config['window_size'], config['stride'], config['out_dir'], channels_used, None)
                test_ds = SisFallDataset(sisfall_root, [test_subj], config['window_size'], config['stride'], config['out_dir'], channels_used, None)

                split_tag = f"loso_{test_subj}"
                if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
                    logging.warning(f"Skipping LOSO fold {split_tag} due to empty split (train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}).")
                    continue
                split_stats_total[split_tag] = {
                    "train": summarize_split(train_ds),
                    "val": summarize_split(val_ds),
                    "test": summarize_split(test_ds),
                }
                resume_path_split = resolve_resume_path(resume_path, config['out_dir'], seed, split_tag)
                res = train_eval_split(train_ds, val_ds, test_ds, split_tag, resume_path_split)
                results.append(res)
                loso_records.append(
                    {
                        "fold": fold_idx,
                        "test_subject": test_subj,
                        "n_train": len(train_ds),
                        "n_val": len(val_ds),
                        "n_test": len(test_ds),
                        "metrics": res.get("test_metrics", {}),
                    }
                )
                # collect predictions for latency / stats diagnostics
                y_true_fold = res.get("y_true", [])
                y_pred_fold = res.get("y_pred", [])
                y_prob_fold = res.get("y_prob", [])
                subs_fold = res.get("subjects", [])
                loso_predictions["y_true"].extend(y_true_fold)
                loso_predictions["y_pred"].extend(y_pred_fold)
                loso_predictions["y_prob"].extend([p[1] if len(p) > 1 else float("nan") for p in y_prob_fold])
                loso_predictions["subject"].extend(subs_fold)
                loso_predictions["fold"].extend([fold_idx] * len(y_true_fold))
            if loso_records:
                loso_summary = aggregate_loso_results(loso_records)
                save_loso_results(config['out_dir'], seed, loso_records, loso_predictions, loso_summary)
        else:
            rng = random.Random(seed)
            rng.shuffle(unique_subjects)
            n = len(unique_subjects)
            train_cut = min(max(1, int(0.7 * n)), n - 2)
            val_candidate = max(1, int(0.85 * n))
            val_cut = min(n - 1, max(train_cut + 1, val_candidate))

            train_subjs = unique_subjects[:train_cut]
            val_subjs = unique_subjects[train_cut:val_cut]
            test_subjs = unique_subjects[val_cut:]

            logging.info(f"Found {n} unique subjects at '{sisfall_root}': {unique_subjects}")
            logging.info(f"Train subjects ({len(train_subjs)}): {train_subjs}")
            logging.info(f"Val subjects ({len(val_subjs)}): {val_subjs}")
            logging.info(f"Test subjects ({len(test_subjs)}): {test_subjs}")

            train_ds = SisFallDataset(sisfall_root, train_subjs, config['window_size'], config['stride'], config['out_dir'], channels_used, augmentor)
            val_ds = SisFallDataset(sisfall_root, val_subjs, config['window_size'], config['stride'], config['out_dir'], channels_used, None)
            test_ds = SisFallDataset(sisfall_root, test_subjs, config['window_size'], config['stride'], config['out_dir'], channels_used, None)

            if len(train_ds) == 0:
                raise RuntimeError(
                    "SisFallDataset is empty! \n"
                    "Tip: Ensure 'data_root' is correct and parser logic is implemented.\n"
                    "If running purely for code verification, use '--dataset dryrun'."
                )
            split_stats_total["train"] = summarize_split(train_ds)
            split_stats_total["val"] = summarize_split(val_ds)
            split_stats_total["test"] = summarize_split(test_ds)
            res = train_eval_split(train_ds, val_ds, test_ds, "main", resume_path)
            results.append(res)
    elif config['dataset'] in ('mobiact', 'unimib', 'kfall'):
        train_ds, val_ds, test_ds, C_loaded = load_preloaded_splits(config['data_root'], config['dataset'], seed, channels_used)
        C = C_loaded
        L = train_ds.items[0][0].shape[-1] if getattr(train_ds, "items", None) else L
        split_stats_total["train"] = summarize_split(train_ds)
        split_stats_total["val"] = summarize_split(val_ds)
        split_stats_total["test"] = summarize_split(test_ds)
        res = train_eval_split(train_ds, val_ds, test_ds, config['dataset'], resume_path)
        results.append(res)
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")

    if split_stats_total:
        save_split_stats(config['out_dir'], seed, split_stats_total)

    if eval_mode == 'loso' and results:
        agg = aggregate_runs(results)
        return {'seed': seed, 'test_metrics': agg}
    return results[0] if results else {'seed': seed, 'test_metrics': {}}


def run_ablation_suite(base_config: Dict[str, Any], seeds: List[int], resume_path: Optional[str] = None):
    """
    Run predefined ablation matrix sequentially; each ablation gets its own out_dir subfolder.
    """
    suite_records: List[Dict[str, Any]] = []
    for spec in ABLATION_MATRIX:
        ab_cfg = copy.deepcopy(base_config)
        ab_cfg['run_ablation_suite'] = False
        ab_cfg['ablation'] = {k: v for k, v in spec.items() if k in ('mspa', 'dks', 'faa', 'tfcl', 'center')}
        ab_name = spec.get("name", "ablation")
        safe_name = ab_name.replace("/", "_")
        ab_out = os.path.join(base_config['out_dir'], f"ablation_{safe_name}")
        ensure_dir(ab_out)
        ab_cfg['out_dir'] = ab_out
        logging.info(f"[ablation] Running preset {ab_name} -> {ab_out}")
        for s in seeds:
            try:
                res = run_one_experiment(ab_cfg, s, resume_path)
                suite_records.append({"name": ab_name, "seed": s, "result": res})
            except Exception as e:
                logging.error(f"[ablation] preset {ab_name} seed {s} failed: {e}")
                logging.error(traceback.format_exc())
                suite_records.append({"name": ab_name, "seed": s, "error": str(e)})
    return suite_records

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='dryrun', choices=['dryrun', 'sisfall', 'mobiact', 'unimib', 'kfall'])
    p.add_argument('--data-root', default='./data', help="Root path for dataset")
    p.add_argument('--out-dir', default='./outputs')
    p.add_argument('--model', default='phycl', choices=['phycl', 'phycl_full', 'dmc', 'lstm', 'resnet', 'amsv2', 'liteams', 'tcn', 'transformer', 'inceptiontime', 'rocket', 'tinyhar', 'deeplstm'])
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--warmup-epochs', type=int, default=5, help="Number of warmup epochs")
    p.add_argument('--accum-steps', type=int, default=1)
    p.add_argument('--channels', type=int, default=64)
    p.add_argument('--n-blocks', type=int, default=4)
    p.add_argument('--kernel-sizes', type=int, nargs='+', default=[7, 15, 31, 63], help="Kernel sizes for physics-aware dynamic kernels")
    p.add_argument('--freq-method', type=str, choices=['fft', 'fft_attn', 'stft', 'cwt', 'adaptive_fft'], default='fft', help="Frequency encoder: fft (adaptive bands)|fft_attn|stft|cwt|adaptive_fft")
    p.add_argument('--fusion-variant', type=str, default='enhanced', choices=['baseline', 'enhanced'], help="Cross-gated fusion variant")
    p.add_argument('--fusion-kernel-sizes', type=int, nargs='+', default=[3, 5, 7], help="Kernel sizes for enhanced fusion gating")
    p.add_argument('--num-bands', type=int, default=4, help="Number of spectral bands for MSPA modules")
    p.add_argument('--band-edges', type=float, nargs='+', default=None, help="Initial normalized band edges (len=num_bands+1)")
    p.add_argument('--adaptive-bands', action='store_true', help="Enable learnable MSPA band edges")
    p.add_argument('--no-adaptive-bands', action='store_false', dest='adaptive_bands', help="Disable learnable MSPA band edges")
    attn_choices = ['none', 'eca', 'cbam', 'ema', 'ca', 'simam', 'aspp', 'mca']
    p.add_argument('--attn-time', type=str, default='none', choices=attn_choices, help="Attention for AMSNetV2 time branch")
    p.add_argument('--attn-freq', type=str, default='none', choices=attn_choices, help="Attention for AMSNetV2 frequency branch")
    p.add_argument('--attn-fuse', type=str, default='none', choices=attn_choices, help="Attention after CrossGatedFusion in AMSNetV2")
    p.add_argument('--attn-lite', type=str, default='none', choices=attn_choices, help="Attention inside LiteAMSNet blocks")
    p.add_argument('--disable-faa-axis-attn', action='store_false', dest='faa_axis_attn', help="Disable cross-axis attention inside FallAwareAttention")
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--num-classes', type=int, default=2)
    p.add_argument('--in-channels', type=int, default=3)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--window-size', type=int, default=512)
    p.add_argument('--stride', type=int, default=256)
    p.add_argument('--channels-used', type=str, default='accel3', help="SisFall channels to use: accel3|accel6|accel6+gyro|full")
    p.add_argument('--augment', action='store_true', help="Enable sensor augmentations for training")
    p.add_argument('--noise-std', type=float, default=0.05, help="Gaussian noise std for augmentation")
    p.add_argument('--scale-range', type=float, nargs=2, default=[0.9, 1.1], help="Scale range for augmentation")
    p.add_argument('--time-shift-ratio', type=float, default=0.1, help="Max relative time shift for augmentation")
    p.add_argument('--drop-prob', type=float, default=0.1, help="Portion of window to zero for augmentation")
    p.add_argument('--eval-mode', choices=['holdout', 'loso'], default='holdout', help="Evaluation protocol for SisFall")
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--amp', action='store_true')
    p.add_argument('--weighted-loss', action='store_true')
    p.add_argument(
        '--class-weighting',
        type=str,
        default=None,
        choices=['none', 'auto', 'inv_freq', 'sqrt_inv_freq', 'effective_num'],
        help="Class weighting strategy (overrides --weighted-loss when set).",
    )
    p.add_argument(
        '--effective-num-beta',
        type=float,
        default=0.999,
        help="Beta for effective_num class weighting.",
    )
    p.add_argument('--profile', action='store_true', help="Run efficiency profiling")
    p.add_argument('--run-robustness', action='store_true', help="Run robustness evaluation on the test split")
    p.add_argument('--run-ablation-suite', action='store_true', help="Run predefined ablation suite (outputs saved under out_dir/ablation_*)")
    p.add_argument('--allow-metrics-fallback', action='store_true', help="Allow manual metric fallback when scikit-learn/threadpoolctl are missing (default: fail fast)")
    p.add_argument('--viz-tsne', action='store_true', help="Generate t-SNE visualization on test split")
    p.add_argument('--viz-tsne-stage', type=str, default='final', choices=['final', 'time', 'freq'], help="Stage for t-SNE visualization")
    p.add_argument('--viz-routing', action='store_true', help="Visualize dynamic kernel routing weights")
    p.add_argument('--viz-gradcam', action='store_true', help="Generate Grad-CAM for one batch")
    p.add_argument('--loso-max-folds', type=int, default=None, help="Optional cap on number of LOSO folds for quick smoke tests")
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--use-tfcl', action='store_true', help="Enable time-frequency contrastive loss")
    p.add_argument('--ablation', type=str, default=None, help="Ablation preset name or comma-separated toggles")
    p.add_argument('--proj-dim', type=int, default=128, help="Projection head dimension for contrastive learning")
    p.add_argument('--tf-temperature', type=float, default=0.1, help="Temperature for TF contrastive loss")
    p.add_argument('--tf-supervised-weight', type=float, default=0.1, help="Weight for supervised TF contrastive term")
    p.add_argument('--tf-cross-weight', type=float, default=0.3, help="Cross-layer weight for hierarchical TF contrastive loss")
    p.add_argument('--disable-tf-hierarchical', action='store_false', dest='tf_hierarchical', help="Disable hierarchical TF contrastive; fallback to single-level TFCL")
    p.add_argument('--label-smoothing', type=float, default=0.0, help="Label smoothing for CE loss")
    p.add_argument('--min-patience', type=int, default=5, help="Minimum patience for adaptive early stopping")
    p.add_argument('--patience-decay', type=float, default=0.9, help="Decay factor for patience after improvement")
    p.add_argument('--loss-alpha', type=float, default=0.1, help="Weight for TF contrastive loss term (ignored if --uncertainty-weighting)")
    p.add_argument('--loss-beta', type=float, default=0.01, help="Weight for center loss term (ignored if --uncertainty-weighting)")
    p.add_argument('--uncertainty-weighting', action='store_true', help="Use learned uncertainty weighting (Kendall et al.) instead of fixed alpha/beta")
    p.add_argument('--rocket-kernels', type=int, default=256, help="Number of random kernels for ROCKET baseline")
    p.add_argument('--rocket-kernel-size', type=int, default=9, help="Kernel size for ROCKET baseline")
    p.add_argument('--lstm-hidden', type=int, default=128, help="Hidden size for DeepConvLSTM baseline")
    p.add_argument('--stat-baseline', type=str, default=None, help="Comma-separated baseline metric values for significance testing (e.g., prior macro_f1 per seed)")
    p.add_argument('--sample-rate', type=float, default=50.0, help="Sampling rate (Hz) used to estimate detection latency")
    p.add_argument('--cross-eval', action='store_true', help="Evaluate trained model across other datasets without retraining")
    # [Patch B] Added resume argument
    p.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    p.set_defaults(tf_hierarchical=True, adaptive_bands=True, faa_axis_attn=True)
    
    args = p.parse_args()
    ensure_dir(args.out_dir)
    setup_logging(args.out_dir)

    global ALLOW_METRICS_FALLBACK
    ALLOW_METRICS_FALLBACK = bool(args.allow_metrics_fallback)
    ensure_metric_dependencies(ALLOW_METRICS_FALLBACK)

    save_complete_experiment_config(args, args.out_dir)
    
    internal_model, effective_ablation_spec, model_display_name = resolve_requested_model(args.model, args.ablation)
    config = vars(args)
    config['model_internal'] = internal_model
    config['model_display_name'] = model_display_name
    config['requested_model'] = args.model
    if config.get('band_edges'):
        config['band_edges'] = tuple(config['band_edges'])
    else:
        config['band_edges'] = None
    config['fusion_kernel_sizes'] = tuple(config.get('fusion_kernel_sizes', (3, 5, 7)))
    config['ablation'] = parse_ablation_config(effective_ablation_spec)
    logging.info(f"Optimized Config: {config}")
    
    results = []
    baseline_vals = None
    if args.stat_baseline:
        baseline_vals = [float(v) for v in args.stat_baseline.split(',') if v.strip()]
    for s in args.seeds:
        try:
            # Pass resume path only if provided
            res = run_one_experiment(config, s, args.resume)
            results.append(res)
        except Exception as e:
            logging.error(f"Run failed for seed {s}: {e}")
            logging.error(traceback.format_exc())

    summary = aggregate_runs(results, baseline_vals)
    logging.info("Final Summary: " + json.dumps(summary, indent=2))
    
    # Save Summary
    with open(os.path.join(args.out_dir, 'summary_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    if args.run_ablation_suite:
        ablation_records = run_ablation_suite(config, args.seeds, args.resume)
        ab_path = os.path.join(args.out_dir, 'ablation_summary.json')
        with open(ab_path, "w") as f:
            json.dump(ablation_records, f, indent=2)
        logging.info(f"Ablation suite completed. Saved to {ab_path}")

if __name__ == '__main__':
    main()
