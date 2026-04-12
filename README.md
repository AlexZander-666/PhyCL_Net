# PhyCL-Net

PhyCL-Net is the reviewer-facing code and protocol package for the revised manuscript. It is not a mirror of the full local workspace and it is not a second copy of the journal submission package.

This repository keeps only the code, commands, and minimal documentation needed to inspect the reported protocol. Datasets, checkpoints, generated outputs, and manuscript submission materials are intentionally not versioned here.

## What This Repository Contains
- A single training/evaluation entrypoint: `code/phycl_net_experiments.py`
- The manuscript-facing model and its matched spectral baseline
- The reviewer-facing executable scripts:
  - `code/scripts/run_baseline_comparison.py`
  - `code/scripts/evaluate_noise_robustness.py`
  - `code/scripts/export_model_for_edge.py`
  - `code/scripts/benchmark_on_orangepi.py`
  - `code/scripts/prepare_cross_dataset_npz.py`
  - `code/scripts/run_cross_dataset_evaluation.py`
  - `scripts/profile_phycl_complexity.py`
- The canonical reviewer-facing documents:
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

Edge export bundle for embedded benchmarking:

```bash
python code/scripts/export_model_for_edge.py --checkpoint outputs/phycl_full_sisfall_loso/ckpt_best_seed42_loso_SA01.pth --out-dir ./outputs/edge_bundle --model phycl_full --prepared-npz ./prepared/edge_windows.npz
```

Expected reviewer-facing edge export artifacts:
- `outputs/edge_bundle/phycl_full_edge.ts`
- `outputs/edge_bundle/phycl_full_edge_manifest.json`
- `outputs/edge_bundle/phycl_full_edge_samples.npz`

Orange Pi CPU benchmark:

```bash
python code/scripts/benchmark_on_orangepi.py --model-path ./outputs/edge_bundle/phycl_full_edge.ts --out-json ./outputs/orangepi/orangepi_cpu.json --input-shape 1 3 512 --warmup 50 --repeats 200 --runtime-backend torchscript --execution-mode CPU --board-model "Orange Pi AI Pro 20T 24G" --npz-path ./outputs/edge_bundle/phycl_full_edge_samples.npz
```

Expected reviewer-facing Orange Pi artifacts:
- `outputs/orangepi/orangepi_cpu.json`

Cross-dataset NPZ preparation:

```bash
python code/scripts/prepare_cross_dataset_npz.py --dataset mobiact --source ./raw/MobiFall --out-root ./prepared --target-len 200
python code/scripts/prepare_cross_dataset_npz.py --dataset unimib --source ./raw/unimib.zip --out-root ./prepared --target-len 200
python code/scripts/prepare_cross_dataset_npz.py --dataset kfall --source ./raw/kfall.zip --out-root ./prepared --target-len 200
```

Cross-dataset evaluation:

```bash
python code/scripts/run_cross_dataset_evaluation.py --checkpoint outputs/phycl_sisfall_loso/ckpt_best_seed42_loso_SA01.pth --data-root ./prepared --out-dir ./outputs/cross_dataset --base-dataset sisfall --targets mobiact unimib kfall --model phycl
```

Expected reviewer-facing cross-dataset artifacts:
- `prepared/mobiact/train.npz`, `prepared/mobiact/val.npz`, `prepared/mobiact/test.npz`
- `prepared/unimib/train.npz`, `prepared/unimib/val.npz`, `prepared/unimib/test.npz`
- `prepared/kfall/train.npz`, `prepared/kfall/val.npz`, `prepared/kfall/test.npz`
- `outputs/cross_dataset/cross_dataset_summary.json`

## Project Layout
- `code/`: canonical training entrypoint, model modules, losses, and reviewer-facing comparison and evaluation scripts, including export, embedded benchmarking, and cross-dataset support surfaces
- `scripts/`: standalone reviewer-facing profiling utility
- `docs/`: canonical run protocol, artifact manifest, and repository boundary note

## Reproducibility Notes
- The canonical run protocol is defined in `docs/REPRODUCIBILITY.md`.
- `docs/REPRODUCIBILITY_MANIFEST.json` records the canonical reviewer-facing commands and artifact examples.
- `docs/paper/REVIEWER_RESPONSE_MAPPING.md` defines the repository boundary relative to the submitted revision.
- `data/`, `outputs/`, checkpoints, and generated figures are intentionally excluded from git and must be provided locally.
- The Orange Pi AI Pro 20T 24G CPU benchmark and the MobiFall, UniMiB, and KFall auxiliary transfer workflow are documented as reviewer-facing support surfaces rather than as replacements for the main SisFall LOSO claim.
- The LaTeX manuscript and journal submission materials were submitted separately and are not mirrored in this repository.
- If any README text, script comment, or local note conflicts with the manuscript-facing commands, follow `docs/REPRODUCIBILITY.md`.

Repository-specific working rules are in `AGENTS.md`.

