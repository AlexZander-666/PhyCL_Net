# PhyCL-Net

PhyCL-Net is the reviewer-facing code and protocol package for the revised manuscript. It is not a mirror of the full local workspace and it is not a second copy of the journal submission package.

This repository keeps only the code, commands, and minimal documentation needed to inspect the reported protocol. Datasets, checkpoints, generated outputs, and manuscript submission materials are intentionally not versioned here.

## What This Repository Contains
- A single training/evaluation entrypoint: `code/phycl_net_experiments.py`
- The manuscript-facing model and its matched spectral baseline
- The retained reviewer-facing support scripts:
  - `code/scripts/run_baseline_comparison.py`
  - `code/scripts/evaluate_noise_robustness.py`
  - `scripts/profile_phycl_complexity.py`
- The minimal reviewer-facing documentation:
  - `docs/REPRODUCIBILITY.md`
  - `docs/REPRODUCIBILITY_MANIFEST.json`
  - `docs/paper/REVIEWER_RESPONSE_MAPPING.md`

## Canonical Model Names
- `phycl`: manuscript model, i.e. the time-domain PhyCL-Net configuration without the spectral MSPA branch
- `phycl_full`: matched spectral baseline used for the accuracy-efficiency comparison
- `dual_branch_baseline`: reviewer-facing key for the matched two-branch comparison baseline
- `compact_comparison_baseline`: reviewer-facing key for the compact comparison baseline

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

Expected reviewer-facing checkpoints from the baseline comparison script:
- `outputs/lstm_checkpoint.pth`
- `outputs/resnet_checkpoint.pth`

CPU complexity check:

```bash
python scripts/profile_phycl_complexity.py --device cpu
```

Noise robustness check:

```bash
python code/scripts/evaluate_noise_robustness.py --ckpt outputs/phycl_sisfall_loso/ckpt_best_seed42_loso_SA01.pth --data-root ./data --figure-dir ./figures/noise
```

Expected reviewer-facing noise artifacts:
- `outputs/noise/noise_robustness_results.json` with summary keys `clean_accuracy` and `clean_f1`
- `figures/noise/noise_robustness_curve.png`
- `figures/noise/noise_robustness_curve.pdf`

## Project Layout
- `code/`: training entrypoint, model modules, losses, and retained reviewer-facing helper scripts
- `scripts/`: standalone utility scripts kept for paper-aligned checks
- `docs/`: reproducibility notes, manifest, and reviewer-facing explanatory materials

## Reproducibility Notes
- The canonical run protocol is defined in `docs/REPRODUCIBILITY.md`.
- `docs/REPRODUCIBILITY_MANIFEST.json` records the canonical reviewer-facing commands and artifact examples.
- `docs/paper/REVIEWER_RESPONSE_MAPPING.md` defines the repository boundary relative to the submitted revision.
- `data/`, `outputs/`, checkpoints, and generated figures are intentionally excluded from git and must be provided locally.
- The LaTeX manuscript and journal submission materials were submitted separately and are not mirrored in this repository.
- If any README text, script comment, or local note conflicts with the manuscript-facing commands, follow `docs/REPRODUCIBILITY.md`.

Repository-specific working rules are in `AGENTS.md`.

