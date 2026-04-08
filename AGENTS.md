# Repository Guidelines

## Project Structure & Module Organization
- `code/`: PhyCL-Net training entry (`phycl_net_experiments.py`), model blocks in `models/`, losses in `losses/`, retained reviewer-facing helper scripts in `code/scripts/`.
- `data/`: local SisFall/KFall/UniMiB_SHAR/MobiFall datasets; treat as read-only and exclude from commits.
- `outputs/`, `figures/`: checkpoints, metrics, plots from runs; clean or redirect when starting new sweeps.
- `docs/`: reviewer-facing reproducibility notes, manifest, and manuscript-response mapping.
- `scripts/`: standalone reviewer-facing utilities such as CPU complexity measurement.

## Build, Test, and Development Commands
- Setup: `python -m venv .venv && .\\.venv\\Scripts\\activate && pip install -r requirements.txt`.
- Smoke check: `python code/phycl_net_experiments.py --dataset dryrun --model phycl --epochs 2 --batch-size 4 --profile` (fast env validation).
- Full SisFall: `python code/phycl_net_experiments.py --dataset sisfall --data-root ./data --model phycl --eval-mode loso --seeds 42 123 456 789 1024 --epochs 50 --batch-size 256 --lr 0.004 --warmup-epochs 10 --weighted-loss --amp --use-tfcl`.
- Baselines: `python code/scripts/run_baseline_comparison.py --data-root ./data --epochs 50`.
- CPU complexity: `python scripts/profile_phycl_complexity.py --device cpu`.
- Noise robustness: `python code/scripts/evaluate_noise_robustness.py --ckpt outputs/phycl_sisfall_loso/ckpt_best_seed42_loso_SA01.pth --data-root ./data --figure-dir ./figures/demo`.

## Coding Style & Naming Conventions
- PEP8, 4-space indent; add type hints when clear; keep functions small and deterministic.
- PascalCase for classes; snake_case for modules/functions/CLI flags (`--data-root`, `--batch-size`); reuse existing arg names.
- Prefer `logging` over prints; keep messages short and actionable.
- Align file naming with current artifacts (`ckpt_best_seed123_loso_SA01.pth`, `summary_results.json`, `experiment_config.yaml`); set seeds via `--seed`/`set_seed`.
- Use `phycl` and `phycl_full` in user-facing docs and commands; retain `amsv2` only for backward compatibility inside code.

## Testing Guidelines
- No dedicated unit suite for `code/`; run the dryrun plus one LOSO fold before long sweeps, and spot-check JSON/CSV/plots in `outputs/`.
- For loss/metric edits, rerun `evaluate_noise_robustness.py` on a single checkpoint to confirm curves.
- The reviewer-facing repo intentionally omits manuscript build trees, queue automation, and internal submission-packaging utilities.

## Commit & Pull Request Guidelines
- Commits: imperative subject, <=72 chars, optional scope (`fix: guard empty SisFall split`); never commit datasets or checkpoints.
- PRs: describe intent, commands run, before/after metrics or figure diffs; link issue/task id; flag new CLI options or breaking changes; keep changes reviewable.

## Environment Notes
- Historical local work used a private `SCI666` conda environment. Do not assume it exists on another machine.
- The portable requirement is simpler: use any Python environment that can install `requirements.txt`.
- Prefer GPU execution for training when available, but do not hard-code a private environment name into reviewer-facing instructions.

