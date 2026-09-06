# Reproducibility guide

## Evidence verification (no dependencies)

Use Python 3.10+ from the repository root:

```bash
python scripts/verify_reviewer_evidence.py
```

This is the verified entry point for the published evidence package. It needs no trained model or dataset. See [the reviewer guide](REVIEWER_GUIDE.md) for estimators, units and interpretation.

## Model execution environment

The retained historical implementation uses PyTorch, NumPy, SciPy, scikit-learn and the packages in [requirements.txt](../requirements.txt). A private environment name is not required. For example on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python code/phycl_net_experiments.py --help
```

Training should use an appropriate CUDA-enabled PyTorch environment. Creating the environment or a synthetic smoke run does not reproduce manuscript metrics. This publication update performs no accuracy inference or hardware benchmark.

## Data and model inputs

SisFall is the cited public dataset: Sucerquia et al., *SisFall: A Fall and Movement Dataset*, [Sensors 17, 198 (2017)](https://doi.org/10.3390/s17010198). Acquire it from the dataset authors under their terms, keep original files locally under `data/SisFall`, and pass `--data-root ./data`. Review `SisFallDataset` for accepted discovery paths. Public raw files and all trained `.pth`/`.ckpt` files are excluded from Git.

The main manuscript specifies subjects SA01, SA02, SA04, SA05, SA06, SA09, SA10, SA11, SA17, SA18, SA19 and SA21; 200→50Hz, a fourth-order 5Hz low-pass filter, 512-sample windows and 256 stride, three accelerometer channels. It reports 21,678 windows (12,678 ADL; 9,000 fall). Historical score files have these counts but bind to a native-200Hz loader, not the stated filtered pipeline. Do not assert the filtered loader reproduces the historical counts. The paired FAA report has a different subject set, SA01–SA12.

## Existing implementation and controls

The public entrypoint is `code/phycl_net_experiments.py`. `--model phycl` selects no MSPA; `--model phycl_full` selects MSPA. `--seeds` is plural. Inspect `--help` before running auxiliary tools; their checkpoint/input flags differ.

The paper specifies 50 epochs, batch size 256, learning rate .004, 10 warmup epochs, AdamW weight decay 1e-4, cosine minimum LR 1e-6, AMP, gradient clip 1, inverse-frequency CE and contrastive weight .1 with temperature .1. These are **paper specifications**, not a claim that the inherited executable and historical args are fully aligned. [Code correspondence](MANUSCRIPT_CODE_MAPPING.md) documents known differences. No manuscript-exact rerun command is asserted until those differences and the source-run lineage are resolved.

For an interface-only synthetic smoke run in a configured environment, with a new output directory:

```bash
python code/phycl_net_experiments.py --dataset dryrun --model phycl --epochs 1 --batch-size 4 --out-dir outputs/interface_smoke
```

Do not use its outputs as evidence for the manuscript's accuracy or preprocessing.

## Reviewer-facing executable scripts

| Script | Input / purpose | Limitation |
| --- | --- | --- |
| `scripts/verify_reviewer_evidence.py` | Published CSV/JSON; standard-library recalculation | Saved evidence only |
| `code/scripts/run_baseline_comparison.py` | Local SisFall dataset | New runs require controlled settings and provenance |
| `scripts/profile_phycl_complexity.py` | CPU model profiling, `--device cpu` | Hardware/runtime/graph must match before comparing numbers |
| `code/scripts/evaluate_noise_robustness.py` | Real checkpoint and held-out input; see `--help` | A demo is not robustness evidence |
| `code/scripts/prepare_cross_dataset_npz.py` | Locally obtained MobiFall, UniMiB, KFall data | Preparation helper does not define the paper's unresolved five-shot protocol |
| `code/scripts/run_cross_dataset_evaluation.py` | Matching checkpoint, prepared data and input shape | Not proof of the paper's separate six-channel transfer table |
| `code/scripts/export_model_for_edge.py` | Matching checkpoint, architecture and optional windows | Save and verify export manifest before board attribution |
| `code/scripts/benchmark_on_orangepi.py` | Exported model, board/runtime configuration | Preserve model hash, threads, warmup, repeats and timing boundary |

Original helper output names include `lstm_checkpoint.pth`, `resnet_checkpoint.pth`, `noise_robustness_curve.png`, and noise-summary keys `clean_accuracy` and `clean_f1`. Locally generated runs may contain `summary_results.json`, `loso_results_seed*.json`, `split_stats_seed*.json`, checkpoint files and logs. File existence alone is not proof of run completion or correspondence to the manuscript.

## Hardware rerun requirements

The paper's desktop measurement is single-thread CPU, shape 1×3×512, p50/p95; reported software is Python 3.10.11/PyTorch 2.5.1/Windows 10. A rerun must record the actual CPU, thread count, inference graph and measurement scope. Parameter/FLOP measurements use forced CPU and must identify whether projection heads are counted.

Orange Pi AI Pro 20T 24G evidence uses TorchScript CPU, shape 1×3×512, 50 warmup iterations, 200 measurements, fixed inputs and 32 prepared windows. Original records report Python 3.9.2/PyTorch 2.1.0 on aarch64, four CPU cores. The original model, prepared bundle and matching export manifest are not included; the JSON files alone cannot perform a full rerun. Pi/Apollo source firmware/models and raw instrumentation records are likewise not included.

See [public artifacts](PUBLIC_ARTIFACTS.md), [reviewer mapping](paper/REVIEWER_RESPONSE_MAPPING.md), and [evidence boundaries](EVIDENCE_BOUNDARIES.md).
