# CLAUDE.md

Guidance for Claude Code when working with this fall detection research project.

## Project Overview

**TDFNet (Time-Domain Fusion Network)** - A lightweight time-domain dominant fall detection network using accelerometer data, targeted for **SCI Q4 journal publication**. The architecture integrates three core components:

1. **Physics-Aware Dynamic Kernel Selection (DKS)**: Adaptive kernels based on biomechanical priors (SVM, Jerk, impact duration)
2. **Hierarchical Time-Frequency Contrastive Learning (TFCL)**: Multi-layer time↔frequency embedding alignment
3. **Fall-Aware Attention (FAA)**: Multi-physics cue attention (SVM magnitude, Jerk impacts, Jerk rate)

**Note**: TDFNet uses basic FFT for frequency features (via `--ablation mspa:False`), creating a simpler and more efficient architecture compared to the full DKS-FAA-MSPA-TFCL variant while maintaining strong performance through time-domain features and contrastive learning.

---

## Quick Start

### Training Commands

```bash
# Quick test with synthetic data (recommended first)
python DMC_Net_experiments.py --dataset dryrun --profile --epochs 2 --batch-size 4 --ablation mspa:False

# Full LOSO training for SCI paper (NO_MSPA variant - recommended)
python DMC_Net_experiments.py --dataset sisfall --data-root ./data --model amsv2 \
  --eval-mode loso --seeds 42 123 456 789 1024 --epochs 100 --amp --weighted-loss --use-tfcl \
  --ablation mspa:False --out-dir ./outputs/ablation_no_mspa

# Holdout validation (quick)
python DMC_Net_experiments.py --dataset sisfall --data-root ./data --model amsv2 \
  --batch-size 16 --epochs 100 --weighted-loss --amp --use-tfcl --ablation mspa:False

# Resume from checkpoint
python DMC_Net_experiments.py --dataset sisfall --model amsv2 --ablation mspa:False \
  --resume ./outputs/ablation_no_mspa/ckpt_last_seed42.pth

# Full AMSNetV2 with MSPA (for comparison only)
python DMC_Net_experiments.py --dataset sisfall --data-root ./data --model amsv2 \
  --eval-mode loso --seeds 42 123 456 --epochs 100 --amp --weighted-loss --use-tfcl
```

### Ablation Studies

```bash
# Disable specific components
python DMC_Net_experiments.py --dataset sisfall --model amsv2 --eval-mode loso \
  --seeds 42 123 456 --epochs 100 --ablation [time_only|freq_only|mspa:False|dks:False|faa:False]
```

### Baseline Models

Available models: `amsv2`, `lstm`, `resnet`, `tcn`, `transformer`, `inceptiontime`, `rocket`, `tinyhar`

```bash
python DMC_Net_experiments.py --dataset sisfall --model [MODEL] --eval-mode loso --seeds 42 123 456 --epochs 100
```

### Automation Scripts

```bash
# Windows: Auto-manage ablation experiments
start_ablation_queue.bat

# Windows: Auto-launch next training jobs
start_auto_monitor.bat
```

---

## Architecture

### TDFNet: Three-Stage Time-Dominant Network (DKS + FAA + TFCL)

```
Input (B, 3, 512) → Stem (48-ch) →
  Stage1 (2×TDFBlock, 48-ch) → Downsample (96-ch) →
  Stage2 (2×TDFBlock, 96-ch) → Downsample (192-ch) →
  Stage3 (2×TDFBlock, 192-ch) → GlobalAvgPool → Classifier (2 classes)
```

**TDFBlock Structure** (DKS + FAA + FFT):
```
Input → Time Branch (DKS) → [TimeAttn] → t_feat
     └─ Freq Branch (FFT only) → FAA → [FreqAttn] → f_feat

t_feat + f_feat → CrossGatedFusion → [FusionAttn] → output
```

**Architecture Variants**:
- **TDFNet (DKS + FAA + TFCL)**: Time-domain dominant, uses basic FFT for frequency features (`--ablation mspa:False`)
- **Full variant (DKS + FAA + MSPA + TFCL)**: Includes Multi-Scale Spectral Pyramid for frequency band decomposition

**TDFNet Advantages**:
- Simpler frequency branch (basic FFT instead of pyramid)
- Faster training, lower computational cost
- Similar performance to full variant

**Hierarchical TF-Contrastive**: Each TDFBlock outputs time/freq features; 6 projection heads (3 stages × 2 branches) align embeddings via InfoNCE loss.

---

## Key CLI Arguments

**Training**:
- `--epochs N` - Training epochs (default: 10)
- `--batch-size N` - Batch size (default: 32)
- `--lr FLOAT` - Learning rate (default: 1e-3)
- `--weighted-loss` - Enable for imbalanced datasets
- `--amp` - Enable mixed precision (~30% speedup, CUDA only)
- `--accum-steps N` - Gradient accumulation (default: 1)

**Model**:
- `--model {amsv2,lstm,resnet,tcn,...}` - Model architecture
- `--kernel-sizes [K1 K2 ...]` - DKS kernels (default: [7,15,31,63])
- `--freq-method {fft,stft,cwt}` - Frequency encoder (default: fft)
- `--attn-time/freq/fusion {none,eca,cbam,ema,...}` - Attention modules

**Evaluation**:
- `--eval-mode {holdout,loso}` - Use `loso` for SCI paper
- `--seeds [S1 S2 ...]` - Multiple seeds (e.g., `42 123 456 789 1024`)
- `--deterministic` - Full reproducibility (slower)

**Other**:
- `--use-tfcl` - Enable hierarchical contrastive learning
- `--ablation PRESET` - Component ablation (e.g., `mspa:False,dks:False`)
- `--profile` - FLOPs/latency profiling
- `--resume PATH` - Resume from checkpoint

---

## Core Modules (Brief)

### 1. DynamicKernelBlock (models/modules/dks.py) - **ACTIVE**
Physics-aware kernel selection using SVM, Jerk, Jerk Rate, ZCR, Impact Duration, Post-Stillness. Kernels [7,15,31,63] cover 140ms-1.26s @ 50Hz.

### 2. FallAwareAttention (models/modules/faa.py) - **ACTIVE**
Multi-cue attention: SVM magnitude (L2 norm), Jerk (1st derivative), Jerk Rate (2nd derivative). Uses depthwise convs + global context gating.

### 3. CrossGatedFusion (models/ams_net_v2.py) - **ACTIVE**
Symmetric gating: freq→time and time→freq. Channel-wise SE-style fusion with learnable residual scale.

### 4. Hierarchical TF-Contrastive Loss (losses/tfcl.py) - **ACTIVE**
InfoNCE alignment: intra-layer (time↔freq), cross-layer (time_i↔freq_{i+1}), supervised contrastive on final embeddings.

### 5. MultiScaleSpectralPyramid (models/modules/mspa.py) - **DISABLED in TDFNet**
Frequency bands: 0-2Hz (postural), 2-8Hz (walking), 8-20Hz (fall impacts), >20Hz (noise). FFT-based amplitude enhancement, preserves phase. **Not used in TDFNet** - frequency branch uses basic FFT instead.

---

## Critical Implementation Notes

### AMP + Gradient Accumulation
```python
# CRITICAL: Unscale before clipping to avoid NaN gradients
if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### FFT Operations
```python
# Always use FP32 for FFT to avoid numerical errors
from models.modules.mspa import _amp_autocast

with _amp_autocast(False):
    X = torch.fft.rfft(x.to(torch.float32), norm='ortho')
```

### Checkpoint Compatibility
```python
def torch_load_full(path):
    try:
        return torch.load(path, weights_only=False)  # PyTorch ≥2.0
    except TypeError:
        return torch.load(path)  # PyTorch <2.0
```

### Logging Policy
- **Always use `logging` module, NEVER `print()`**
- Levels: INFO (progress), WARNING (non-critical), ERROR (failures)

### Ablation Flags (Counter-intuitive)
- `--ablation freq_only` → removes freq branch (time-only model)
- `--ablation time_only` → removes time branch (freq-only model)

---

## Dataset

### SisFall Structure
```
data/SisFall/
├── ADL/*.txt       # Activities of Daily Living (label 0)
└── FALL/*.txt      # Fall events (label 1)
```

**Format**: 3-column space-separated (X, Y, Z acceleration in LSB)
**Sampling**: 200Hz raw, downsampled to 50Hz
**Window**: 512 samples (10.24s @ 50Hz), stride 256 (50% overlap)
**LOSO**: Subject ID extracted via `r'[SD](\d+)_'` regex

---

## Monitoring & Outputs

**Live Logs**:
```bash
tail -f outputs/experiment.log                 # Main training log
tail -f ablation_queue_manager.log             # Ablation queue status
tail -f auto_train_monitor.log                 # Auto-monitor status
nvidia-smi                                     # GPU status
```

**Output Files** (TDFNet variant):
- `outputs/ablation_no_mspa/summary_results.json` - Aggregated mean/std/CI across seeds
- `outputs/ablation_no_mspa/experiment.log` - Full training history
- `outputs/ablation_no_mspa/ckpt_best_seed{N}.pth` - Best checkpoint (by F1)
- `outputs/ablation_no_mspa/ckpt_last_seed{N}.pth` - Last checkpoint (for resume)
- `outputs/ablation_no_mspa/loso_results_seed{N}.json` - Per-fold LOSO results
- `outputs/ablation_no_mspa/efficiency_seed{N}.json` - FLOPs/params/latency

**Key Metrics**: Accuracy, Precision, Recall, F1 (macro), Confusion Matrix, Detection Latency

---

## SCI Paper Requirements

### Must-Have Experiments
- [x] LOSO Cross-Validation (5 seeds)
- [x] Multi-Seed Statistical Validation (95% CI)
- [ ] Ablation Studies (DKS, FAA, MSPA, TFCL)
- [ ] Baseline Comparisons (≥5 SOTA methods)
- [ ] Cross-Dataset Validation (SisFall→UniMiB/KFall)
- [ ] Robustness Tests (noise, sensor failure)
- [ ] Efficiency Analysis (FLOPs, params, latency)

### Expected Results (for SCI Q4)
- LOSO F1-Score: >93% on SisFall (vs SOTA ~90%)
- Cross-Dataset: <10% performance drop
- Efficiency: <500K params, <50 GFLOPs
- Detection Latency: <200ms median

### Statistical Rigor
- Paired t-test with Bonferroni correction (α=0.05/n_comparisons)
- Report Cohen's d for effect size
- Use t-distribution (not z) for small n

---

## Troubleshooting

**NaN loss during AMP**: Ensure `scaler.unscale_()` before clipping (already fixed in codebase)

**OOM during LOSO**: Reduce `--batch-size` to 8/4 or use `--accum-steps 2`

**No data files found**: Check `--data-root` points to parent of `SisFall/` directory

**Slow LOSO training**: Enable `--amp`; use `--loso-max-folds 3` for smoke testing

**Ablation queue not launching**: Check GPU memory (≥6GB free), review `ablation_queue_manager.log`

**Detection latency NaN**: Known issue in metric computation (line ~1160), requires fix

---

## Development Best Practices

1. **Always test with `--dataset dryrun --epochs 2` first** before long LOSO runs
2. **Use automation scripts** (ablation queue, auto-monitor) for multi-stage workflows
3. **Monitor GPU memory** with `nvidia-smi` before concurrent experiments
4. **Preserve RNG states**: Resume only with matching seeds
5. **Log everything**: Use `logging` module (never `print()`)
6. **FFT requires FP32**: Always wrap with `_amp_autocast(False)`
7. **Gradient accumulation**: Call `scaler.unscale_()` before clipping

---

## Project Structure (Key Files)

```
D:\666\大创/
├── DMC_Net_experiments.py           # Main training script
├── ablation_queue_manager.py        # Auto ablation manager
├── auto_train_next.py               # Auto training launcher
├── models/
│   ├── ams_net_v2.py                # AMSNetV2 architecture
│   └── modules/
│       ├── dks.py                   # Dynamic Kernel Selection
│       ├── faa.py                   # Fall-Aware Attention
│       ├── mspa.py                  # Multi-Scale Spectral Pyramid
│       └── spectral.py              # FFT/STFT/CWT encoders
├── losses/
│   ├── tfcl.py                      # Hierarchical TF-Contrastive Loss
│   └── contrastive.py               # InfoNCE loss
├── data/SisFall/                    # SisFall dataset
├── outputs/                         # Logs, checkpoints, results
├── CLAUDE.md                        # This file
├── ABLATION_QUEUE_CONFIG.md         # Ablation queue config
├── AUTO_MONITOR_README.md           # Auto-monitor usage
└── requirements.txt                 # Python dependencies
```

**Related Documentation**:
- `AGENTS.md` - Experiment tracking guidelines
- `STAGE_2_TRAINING_PLAN.md` - Stage 2 experimental plan
- `GPU_OPTIMIZATION_CONFIG.md` - GPU optimization strategies
- `SCI_论文改进综合方案_完整审查版.md` - SCI paper improvement plan

---

**Version**: 2.2 (TDFNet - Time-Domain Fusion Network)
**Last Updated**: 2025-12-17

**Note**: TDFNet (DKS + FAA + TFCL) uses `--ablation mspa:False` to disable the Multi-Scale Spectral Pyramid, creating a simpler and more efficient architecture while maintaining competitive performance through time-domain features and contrastive learning.
