# PhyCL-Net

PhyCL-Net is an edge-oriented fall-detection repository centered on a physics-guided time-domain model for wearable inertial signals. The public repository is intended to support manuscript review by exposing the training entrypoint, model definitions, analysis scripts, and reproducibility notes. Datasets, checkpoints, and generated outputs remain local and are therefore not versioned.

## What This Repository Contains
- A single training/evaluation entrypoint: `code/DMC_Net_experiments.py`
- The manuscript-facing model and its matched spectral baseline
- Baseline training and analysis scripts used for tables and figures
- Reproducibility notes, manifests, and submission-facing documentation

## Canonical Model Names
- `phycl`: manuscript model, i.e. the time-domain PhyCL-Net configuration without the spectral MSPA branch
- `phycl_full`: matched spectral baseline used for the accuracy-efficiency comparison
- `amsv2`: legacy internal name kept for backward compatibility

The repository still contains some legacy file names from earlier iterations. The canonical reviewer-facing terminology is now `PhyCL-Net` for the manuscript model and `PhyCL-Net + MSPA` for the matched spectral baseline.

## Environment
The original local workflow referenced a private `SCI666` environment. That environment is not required by the repository itself. Use any Python environment that satisfies `requirements.txt`.

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Quick Start
Smoke test:

```bash
python code/DMC_Net_experiments.py --dataset dryrun --model phycl --epochs 2 --batch-size 4 --profile
```

SisFall LOSO, manuscript model:

```bash
python code/DMC_Net_experiments.py --dataset sisfall --data-root ./data --model phycl --eval-mode loso --seeds 42 123 456 789 1024 --epochs 50 --batch-size 256 --lr 0.004 --warmup-epochs 10 --weighted-loss --amp --use-tfcl --out-dir ./outputs/phycl_sisfall_loso
```

SisFall LOSO, matched spectral baseline:

```bash
python code/DMC_Net_experiments.py --dataset sisfall --data-root ./data --model phycl_full --eval-mode loso --seeds 42 123 --epochs 50 --batch-size 256 --lr 0.004 --warmup-epochs 10 --weighted-loss --amp --use-tfcl --out-dir ./outputs/phycl_full_sisfall_loso
```

Baselines:

```bash
python code/scripts/train_baselines.py --data-root ./data --epochs 50
```

Noise robustness check:

```bash
python code/scripts/eval_noise_robustness.py --ckpt outputs/phycl_sisfall_loso/ckpt_best_seed42_loso_SA01.pth --data-root ./data --figure-dir ./figures/noise
```

## Project Layout
- `code/`: training entrypoint, model modules, losses, and analysis scripts
- `automation/`: queue helpers for controlled training sweeps
- `scripts/`: utility scripts for profiling and cross-dataset checks
- `docs/`: reproducibility notes, manifests, experiment logs, and manuscript support files
- `paper/`: manuscript-related assets retained for archival context

## Reproducibility Notes
- The authoritative run protocol is documented in `docs/REPRODUCIBILITY.md`.
- `docs/REPRODUCIBILITY_MANIFEST.json` records the canonical reviewer-facing commands and expected artifacts.
- `data/`, `outputs/`, and generated figures are intentionally excluded from git. You must provide them locally to reproduce the reported numbers.
- If a manuscript statement and a legacy script comment disagree, follow the manuscript-facing names and commands in this README and `docs/REPRODUCIBILITY.md`.

Repository-specific working rules are in `AGENTS.md`.
