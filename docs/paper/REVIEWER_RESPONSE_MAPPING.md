# Reviewer-Facing Notes

This repository is a compact reproducibility package for the revised PhyCL-Net manuscript. It is intentionally narrower than the full local working directory.

## What Is Kept Here

- The canonical training and evaluation entrypoint: `code/phycl_net_experiments.py`
- The manuscript model (`phycl`) and its matched spectral baseline (`phycl_full`)
- The retained reviewer-facing support scripts:
  - `code/scripts/run_baseline_comparison.py`
  - `code/scripts/evaluate_noise_robustness.py`
  - `scripts/profile_phycl_complexity.py`
- The minimal documentation required to rerun or inspect the reported protocol:
  - `README.md`
  - `docs/REPRODUCIBILITY.md`
  - `docs/REPRODUCIBILITY_MANIFEST.json`

## What Is Intentionally Not Duplicated Here

- The LaTeX manuscript source tree
- Journal-specific submission bundles
- Internal experiment logs, packing scripts, queue automation, and drafting materials
- Datasets, checkpoints, generated figures, and other large local outputs

These materials were either submitted through the journal system or are not part of the main reviewer-facing reproducibility chain.

## How This Repository Maps to the Revision

- Main accuracy and LOSO results:
  - Reproduced from `code/phycl_net_experiments.py` with the commands listed in `README.md` and `docs/REPRODUCIBILITY.md`
- Matched spectral baseline comparison:
  - Reproduced from the same entrypoint by switching from `--model phycl` to `--model phycl_full`
- Baseline comparisons against general-purpose architectures:
  - Reproduced with `code/scripts/run_baseline_comparison.py`
- CPU complexity and efficiency checks discussed in the paper:
  - Supported by `scripts/profile_phycl_complexity.py` and the profiling outputs emitted by `code/phycl_net_experiments.py`
- Noise robustness discussion:
  - Supported by `code/scripts/evaluate_noise_robustness.py`

## Scope Reminder

This repository supports code and protocol inspection for the revised manuscript. It does not serve as a second copy of the submitted paper package, and it should not be interpreted as evidence for claims beyond the paper's stated scope, such as commercial wearable deployment or cross-dataset external validation.

