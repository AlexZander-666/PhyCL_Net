# PhyCL-Net

PhyCL-Net is a reviewer-facing reproducibility repository for the manuscript's physics-guided time-domain fall-detection model. It keeps only the code, supporting scripts, and documentation needed to inspect the main experimental pipeline and the paper-aligned auxiliary checks. Datasets, checkpoints, generated outputs, and the submitted manuscript package are intentionally not versioned here.

## What This Repository Contains
- A single training/evaluation entrypoint: `code/phycl_net_experiments.py`
- The manuscript-facing model and its matched spectral baseline
- Minimal supporting scripts for baseline comparison, CPU complexity measurement, and noise robustness evaluation
- Reproducibility notes and reviewer-facing documentation

## Canonical Model Names
- `phycl`: manuscript model, i.e. the time-domain PhyCL-Net configuration without the spectral MSPA branch
- `phycl_full`: matched spectral baseline used for the accuracy-efficiency comparison
- `amsv2`: deprecated legacy CLI alias kept only for backward compatibility

The canonical reviewer-facing terminology is `PhyCL-Net` for the manuscript model and `PhyCL-Net + MSPA` for the matched spectral baseline.

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
python code/phycl_net_experiments.py --dataset dryrun --model phycl --epochs 2 --batch-size 4 --profile
```

SisFall LOSO, manuscript model:

```bash
python code/phycl_net_experiments.py --dataset sisfall --data-root ./data --model phycl --eval-mode loso --seeds 42 123 456 789 1024 --epochs 50 --batch-size 256 --lr 0.004 --warmup-epochs 10 --weighted-loss --amp --use-tfcl --out-dir ./outputs/phycl_sisfall_loso
```

SisFall LOSO, matched spectral baseline:

```bash
python code/phycl_net_experiments.py --dataset sisfall --data-root ./data --model phycl_full --eval-mode loso --seeds 42 123 --epochs 50 --batch-size 256 --lr 0.004 --warmup-epochs 10 --weighted-loss --amp --use-tfcl --out-dir ./outputs/phycl_full_sisfall_loso
```

Baselines:

```bash
python code/scripts/run_baseline_comparison.py --data-root ./data --epochs 50
```

CPU complexity check:

```bash
python scripts/profile_phycl_complexity.py --device cpu
```

Noise robustness check:

```bash
python code/scripts/evaluate_noise_robustness.py --ckpt outputs/phycl_sisfall_loso/ckpt_best_seed42_loso_SA01.pth --data-root ./data --figure-dir ./figures/noise
```

## Project Layout
- `code/`: training entrypoint, model modules, losses, and retained reviewer-facing helper scripts
- `scripts/`: standalone utility scripts kept for paper-aligned checks
- `docs/`: reproducibility notes, manifest, and reviewer-facing explanatory materials

## Reproducibility Notes
- The authoritative run protocol is documented in `docs/REPRODUCIBILITY.md`.
- `docs/REPRODUCIBILITY_MANIFEST.json` records the canonical reviewer-facing commands and expected artifacts.
- `docs/paper/REVIEWER_RESPONSE_MAPPING.md` explains how this repository relates to the submitted revision without duplicating the manuscript source tree.
- `data/`, `outputs/`, and generated figures are intentionally excluded from git. You must provide them locally to reproduce the reported numbers.
- The LaTeX manuscript and journal submission materials were submitted separately and are therefore not mirrored in this repository.
- If a manuscript statement and a legacy script comment disagree, follow the manuscript-facing names and commands in this README and `docs/REPRODUCIBILITY.md`.

Repository-specific working rules are in `AGENTS.md`.

