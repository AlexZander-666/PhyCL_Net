# Reproducibility Guide

This repository is organized around a single canonical entrypoint, `code/phycl_net_experiments.py`. Reviewer-facing commands below use the manuscript model names `phycl` and `phycl_full`; hidden compatibility aliases remain in the codebase only to avoid breaking older local commands. The manuscript source tree itself is not duplicated here because it is part of the separate journal submission package.

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
- The current manuscript results are reported on SisFall under LOSO evaluation. Additional datasets may be explored with separate scripts, but they are not part of the main reviewer-facing claim.

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

## Expected Artifacts
- `summary_results.json`: aggregate metrics for the run
- `loso_records_seed*.json`: fold-level LOSO metrics
- `efficiency_report_seed*.json`: parameter, FLOP, and latency profiling
- `experiment.log`: training and evaluation log
- `outputs/lstm_checkpoint.pth` and `outputs/resnet_checkpoint.pth`: optional checkpoints emitted by `run_baseline_comparison.py`
- `noise_robustness_results.json`: optional robustness sweep output when the noise script is used; its summary block reports `clean_accuracy` and `clean_f1` for the sigma=0 reference run
- `noise_robustness_curve.png` and `noise_robustness_curve.pdf`: optional reviewer-facing plots emitted by the noise robustness script

## Notes on Scope
- The repository documents algorithmic reproducibility under the reported desktop CPU/GPU protocol.
- It does not claim direct validation on commercial wearables or medical alarm systems.
- Data availability and repository release statements should be read together with the current manuscript revision.
- Auxiliary internal logs, manuscript build trees, and submission packing utilities are intentionally excluded from this reviewer-facing repository.

