# Reproducibility Guide

This repository is organized around a single canonical entrypoint, `code/phycl_net_experiments.py`. Reviewer-facing commands below use the manuscript model names `phycl` and `phycl_full`. Legacy local aliases, if any, are outside the reviewer-facing interface. The manuscript source tree itself is not duplicated here because it is part of the separate journal submission package.

Additional reviewer-facing baseline keys exposed by the CLI are `dual_branch_baseline` and `compact_comparison_baseline`.

## Environment
Create any Python environment that satisfies `requirements.txt`. The repository does not require a private environment name such as `SCI666`.

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Data
- Place datasets under `data/` and pass `--data-root ./data`.
- This public repository does not commit datasets, checkpoints, or generated outputs.
- The current manuscript results are reported on SisFall under LOSO evaluation.
- Auxiliary transfer support for MobiFall, UniMiB, and KFall is exposed through separate preparation and evaluation scripts. These support surfaces broaden inspection of the revised manuscript, but they do not replace the main SisFall LOSO claim.

## Canonical Commands
Smoke test:

```bash
python code/phycl_net_experiments.py --dataset dryrun --model phycl --epochs 2 --batch-size 4 --profile
```

PhyCL-Net on SisFall LOSO:

```bash
python code/phycl_net_experiments.py --dataset sisfall --data-root ./data --model phycl --eval-mode loso --seeds 42 123 456 789 1024 --epochs 50 --batch-size 256 --lr 0.004 --warmup-epochs 10 --weighted-loss --amp --use-tfcl --out-dir ./outputs/phycl_sisfall_loso
```

Matched spectral baseline:

```bash
python code/phycl_net_experiments.py --dataset sisfall --data-root ./data --model phycl_full --eval-mode loso --seeds 42 123 --epochs 50 --batch-size 256 --lr 0.004 --warmup-epochs 10 --weighted-loss --amp --use-tfcl --out-dir ./outputs/phycl_full_sisfall_loso
```

Baselines:

```bash
python code/scripts/run_baseline_comparison.py --data-root ./data --epochs 50
```

Expected checkpoints from this helper script:
- `outputs/lstm_checkpoint.pth`
- `outputs/resnet_checkpoint.pth`

CPU complexity check:

```bash
python scripts/profile_phycl_complexity.py --device cpu
python scripts/profile_phycl_complexity.py --device cpu --ablation-mspa
```

Optional noise robustness check for the discussion section:

```bash
python code/scripts/evaluate_noise_robustness.py --ckpt outputs/phycl_sisfall_loso/ckpt_best_seed42_loso_SA01.pth --data-root ./data --output-dir ./outputs/noise --figure-dir ./figures/noise
```

Edge export bundle:

```bash
python code/scripts/export_model_for_edge.py --checkpoint outputs/phycl_full_sisfall_loso/ckpt_best_seed42_loso_SA01.pth --out-dir ./outputs/edge_bundle --model phycl_full --prepared-npz ./prepared/edge_windows.npz
```

Orange Pi CPU benchmark:

```bash
python code/scripts/benchmark_on_orangepi.py --model-path ./outputs/edge_bundle/phycl_full_edge.ts --out-json ./outputs/orangepi/orangepi_cpu.json --input-shape 1 3 512 --warmup 50 --repeats 200 --runtime-backend torchscript --execution-mode CPU --board-model "Orange Pi AI Pro 20T 24G" --npz-path ./outputs/edge_bundle/phycl_full_edge_samples.npz
```

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

## Expected Artifacts
- `summary_results.json`: aggregate metrics for the run
- `loso_records_seed*.json`: fold-level LOSO metrics
- `efficiency_report_seed*.json`: parameter, FLOP, and latency profiling
- `experiment.log`: training and evaluation log
- `outputs/lstm_checkpoint.pth` and `outputs/resnet_checkpoint.pth`: optional checkpoints emitted by `run_baseline_comparison.py`
- `noise_robustness_results.json`: optional robustness sweep output when the noise script is used; its summary block reports `clean_accuracy` and `clean_f1` for the sigma=0 reference run
- `noise_robustness_curve.png` and `noise_robustness_curve.pdf`: optional reviewer-facing plots emitted by the noise robustness script
- `outputs/edge_bundle/phycl_full_edge.ts`, `outputs/edge_bundle/phycl_full_edge_manifest.json`, and `outputs/edge_bundle/phycl_full_edge_samples.npz`: optional export artifacts for embedded benchmarking
- `outputs/orangepi/orangepi_cpu.json`: optional Orange Pi AI Pro 20T 24G CPU benchmark report with board metadata and p50/p95 latency
- `prepared/mobiact/*.npz`, `prepared/unimib/*.npz`, and `prepared/kfall/*.npz`: optional two-class preparation outputs for auxiliary transfer checks
- `outputs/cross_dataset/cross_dataset_summary.json`: optional auxiliary transfer summary across MobiFall, UniMiB, and KFall

## Notes on Scope
- The repository documents algorithmic reproducibility under the reported desktop CPU/GPU protocol.
- It also exposes the reviewer-facing support scripts used to prepare an edge export bundle, measure Orange Pi AI Pro 20T 24G CPU latency, and stage auxiliary transfer checks on MobiFall, UniMiB, and KFall.
- It does not claim direct validation on commercial wearables or medical alarm systems.
- Data availability should be read from the manuscript and any linked release statement, not inferred from this repository alone.
- Auxiliary internal logs, manuscript build trees, and submission packing utilities are intentionally excluded from this reviewer-facing repository.

