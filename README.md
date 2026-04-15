# PhyCL-Net

PhyCL-Net is the reviewer-facing code and protocol package for the revised manuscript. It is not a mirror of the full local workspace and it is not a second copy of the journal submission package. This repository keeps only the code, commands, and minimal documentation needed to inspect the reported protocol.

For the revised manuscript, the reviewer-facing package also includes auxiliary cross-dataset transfer support for MobiFall, UniMiB, and KFall under a separate protocol, together with the Orange Pi AI Pro 20T 24G CPU benchmarking path. These materials are provided as reviewer-facing support surfaces rather than as replacements for the main SisFall LOSO claim.

Datasets, checkpoints, generated outputs, and manuscript submission materials are intentionally not versioned here.

## What This Repository Contains
- A single training/evaluation entrypoint: `code/phycl_net_experiments.py`
- The manuscript-facing model and its matched spectral baseline
- Auxiliary cross-dataset transfer support for MobiFall, UniMiB, and KFall under a separate reviewer-facing protocol
- The reviewer-facing executable scripts:
  - `code/scripts/run_baseline_comparison.py`
  - `code/scripts/run_cross_dataset_evaluation.py`
  - `code/scripts/evaluate_noise_robustness.py`
  - `code/scripts/prepare_cross_dataset_npz.py`
  - `code/scripts/export_model_for_edge.py`
  - `code/scripts/benchmark_on_orangepi.py`
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
Use any Python environment that satisfies `requirements.txt`.

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Verified Command Surface
The command blocks below are the exact Phase 3 and Phase 5 commands that were executed to verify the public interface, artifact contract, and the CPU-only Orange Pi closure. They are not manuscript-scale retraining recipes.

Smoke test:

```bash
python code/phycl_net_experiments.py --dataset dryrun --model phycl --epochs 1 --batch-size 4 --out-dir ./outputs/smoke_ckpt_config_phase3
```

Reviewer-facing dryrun for the manuscript model:

```bash
python code/phycl_net_experiments.py --dataset dryrun --model phycl --epochs 1 --batch-size 4 --out-dir ./outputs/repro_phycl_dryrun_phase3
```

Reviewer-facing dryrun for the matched spectral baseline:

```bash
python code/phycl_net_experiments.py --dataset dryrun --model phycl_full --epochs 1 --batch-size 4 --out-dir ./outputs/repro_phycl_full_dryrun_phase3
```

Baselines:

```bash
python code/scripts/run_baseline_comparison.py --data-root ./data --epochs 1 --save-dir ./outputs/baseline_smoke_phase3
```

Prepare `MobiAct` into reviewer-facing `train.npz / val.npz / test.npz` splits:

```bash
python code/scripts/prepare_cross_dataset_npz.py --out-root ./outputs/prepare_cross_dataset_smoke --archive-path "D:/BaiduNetdiskDownload/PhyCL_Net所有训练数据以及权重.zip" --datasets mobiact --target-len 512 --seed 42 --summary-path ./outputs/prepare_cross_dataset_smoke/summary.json
```

Cross-dataset evaluation from a trained checkpoint:

```bash
python code/scripts/run_cross_dataset_evaluation.py --checkpoint ./outputs/archive_ckpt_best_seed42_main.pth --data-root ./data --out-dir ./outputs/repro_cross_eval_phase3 --base-dataset sisfall --model phycl_full --targets mobiact unimib kfall --device cpu --batch-size 128
```

CPU complexity check:

```bash
python scripts/profile_phycl_complexity.py --device cpu > ./outputs/complexity_cpu_phase3.txt
```

Noise robustness demo:

```bash
python code/scripts/evaluate_noise_robustness.py --demo --output-dir ./outputs/noise_demo_phase3 --figure-dir ./outputs/noise_demo_phase3/figures --device cpu
```

Edge export bundle for the board run:

```bash
python code/scripts/export_model_for_edge.py --checkpoint ./outputs/archive_ckpt_best_seed42_main.pth --model phycl_full --out-dir ./outputs/orangepi_edge_export_phase5 --input-shape 1 3 512 --prepared-npz ./data/mobiact/test.npz --sample-count 32
```

On-device CPU benchmark on `Orange Pi AI Pro 20T 24G` with fixed input:

```bash
python code/scripts/benchmark_on_orangepi.py --model-path /home/HwHiAiUser/phycl_phase5/phycl_edge_model.ts --input-shape 1 3 512 --warmup 50 --repeats 200 --out-json /home/HwHiAiUser/phycl_phase5/orangepi_cpu_fixed_benchmark.json --runtime-backend torchscript --execution-mode CPU --board-model "Orange Pi AI Pro 20T 24G" --device cpu
```

On-device CPU benchmark on `Orange Pi AI Pro 20T 24G` with prepared `MobiAct` windows:

```bash
python code/scripts/benchmark_on_orangepi.py --model-path /home/HwHiAiUser/phycl_phase5/phycl_edge_model.ts --input-shape 1 3 512 --warmup 50 --repeats 200 --npz-path /home/HwHiAiUser/phycl_phase5/edge_benchmark_windows.npz --out-json /home/HwHiAiUser/phycl_phase5/orangepi_cpu_real_windows_benchmark.json --runtime-backend torchscript --execution-mode CPU --board-model "Orange Pi AI Pro 20T 24G" --device cpu
```

Expected reviewer-facing checkpoints from the baseline comparison script:
- `outputs/baseline_smoke_phase3/lstm_checkpoint.pth`
- `outputs/baseline_smoke_phase3/resnet_checkpoint.pth`

Expected reviewer-facing noise artifacts:
- `outputs/noise_demo_phase3/noise_robustness_results.json` with summary keys `clean_accuracy` and `clean_f1`
- `outputs/noise_demo_phase3/figures/noise_robustness_curve.png`
- `outputs/noise_demo_phase3/figures/noise_robustness_curve.pdf`

Expected Orange Pi Phase 5 artifacts:
- `outputs/orangepi_edge_export_phase5/edge_export_manifest.json`
- `outputs/orangepi_cpu_fixed_benchmark_phase5.json`
- `outputs/orangepi_cpu_real_windows_benchmark_phase5.json`
- `outputs/orangepi_phase5_table.csv`
- `outputs/orangepi_phase5_status.json`

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
- Public artifact groups and release-only evidence are documented in `docs/PUBLIC_ARTIFACTS.md`.
- `docs/paper/REVIEWER_RESPONSE_MAPPING.md` defines the repository boundary relative to the submitted revision.
- `data/`, `outputs/`, checkpoints, and generated figures are intentionally excluded from git and must be provided locally.
- The Orange Pi AI Pro 20T 24G CPU benchmark and the MobiFall, UniMiB, and KFall auxiliary transfer workflow are documented as reviewer-facing support surfaces rather than as replacements for the main SisFall LOSO claim.
- The LaTeX manuscript and journal submission materials were submitted separately and are not mirrored in this repository.
- The verified edge-device evidence in this repository is limited to an `on-device CPU benchmark on Orange Pi AI Pro 20T 24G`; it is not an NPU claim.
- If any README text, script comment, or local note conflicts with the manuscript-facing commands, follow `docs/REPRODUCIBILITY.md`.

Repository-specific working rules are in `AGENTS.md`.
