# Reviewer-Facing Scope

This repository is the reviewer-facing code and protocol package for the revised PhyCL-Net manuscript. It is not a mirror of the full local workspace and it is not a second copy of the journal submission package.

## Kept Here

- The canonical training and evaluation entrypoint: `code/phycl_net_experiments.py`
- The manuscript model (`phycl`) and the matched spectral baseline (`phycl_full`)
- The reviewer-facing executable scripts:
  - `code/scripts/run_baseline_comparison.py`
  - `code/scripts/evaluate_noise_robustness.py`
  - `scripts/profile_phycl_complexity.py`
- The canonical reviewer-facing documents:
  - `README.md`
  - `docs/REPRODUCIBILITY.md`
  - `docs/REPRODUCIBILITY_MANIFEST.json`

## Not Kept Here

- The LaTeX manuscript source tree
- Journal-specific submission bundles
- Internal logs, packing scripts, queue automation, and drafting materials
- Datasets, checkpoints, generated figures, and other local run outputs

## Why The Paper Files Are Not Here

The manuscript materials were already submitted through the journal system. Duplicating them in this repository would create a second paper package and blur the boundary between submission materials and reproducibility materials. This repository is therefore limited to the code, commands, and artifact descriptions needed to inspect the reported protocol.

## Revision Mapping

- Main LOSO results: reproduce with `code/phycl_net_experiments.py` using the commands in `README.md` and `docs/REPRODUCIBILITY.md`
- Matched spectral comparison: rerun the same entrypoint with `--model phycl_full`
- General-purpose comparison models: inspect or rerun `code/scripts/run_baseline_comparison.py`
- CPU complexity checks: inspect or rerun `scripts/profile_phycl_complexity.py`
- Noise robustness discussion: inspect or rerun `code/scripts/evaluate_noise_robustness.py`

## Scope Boundary

This repository supports inspection of the code path behind the revised manuscript. It should not be read as a claim of separate manuscript hosting, commercial wearable validation, or external-dataset validation beyond the paper's stated scope.
