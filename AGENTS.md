# Repository Guidelines

## Project Structure & Module Organization
- `code/`: PhyCL-Net training entry (`phycl_net_experiments.py`), model blocks in `models/`, losses in `losses/`, retained reviewer-facing helper scripts in `code/scripts/`.
- `data/`: local SisFall/KFall/UniMiB_SHAR/MobiFall datasets; treat as read-only and exclude from commits.
- `outputs/`, `figures/`: checkpoints, metrics, plots from runs; preserve existing results and use a new output directory.
- `docs/`: reviewer-facing reproducibility notes, manifest, and manuscript-response mapping.
- `scripts/`: standalone reviewer-facing utilities such as CPU complexity measurement.
- Complete manuscripts, manuscript source, response letters and submission assets remain local and must not be uploaded to this repository. Keep only code, selected experimental evidence and accompanying inspection documentation public.
- `artifacts/revision_20260906/`: source-traced report transcriptions and complete historical prediction exports. Do not relabel historical configurations or alter reported values.
- `docs/REVIEWER_GUIDE.md`, `docs/MANUSCRIPT_CODE_MAPPING.md`, and `docs/EVIDENCE_BOUNDARIES.md` distinguish paper specifications, historical execution and available verification.

## Build, Test, and Development Commands
- Setup: `python -m venv .venv && .\\.venv\\Scripts\\activate && pip install -r requirements.txt`.
- Smoke check: `python code/phycl_net_experiments.py --dataset dryrun --model phycl --epochs 2 --batch-size 4 --profile` (fast env validation).
- Evidence check (Python 3.10+, standard library only): `python scripts/verify_reviewer_evidence.py`. No manuscript-exact full rerun is claimed; first read the code/protocol differences in `docs/REPRODUCIBILITY.md`.
- Baselines: `python code/scripts/run_baseline_comparison.py --data-root ./data --epochs 50`.
- CPU complexity: `python scripts/profile_phycl_complexity.py --device cpu`.
- Noise robustness: `python code/scripts/evaluate_noise_robustness.py --ckpt outputs/phycl_sisfall_loso/ckpt_best_seed42_loso_SA01.pth --data-root ./data --figure-dir ./figures/demo`.

## Coding Style & Naming Conventions
- PEP8, 4-space indent; add type hints when clear; keep functions small and deterministic.
- PascalCase for classes; snake_case for modules/functions/CLI flags (`--data-root`, `--batch-size`); reuse existing arg names.
- Prefer `logging` over prints; keep messages short and actionable.
- Align file naming with current artifacts (`ckpt_best_seed123_loso_SA01.pth`, `summary_results.json`, `experiment_config.yaml`); set seeds via `--seed`/`set_seed`.
- Use `phycl` and `phycl_full` in user-facing docs and commands.

## Testing Guidelines
- Tests: `python -m pytest -q tests`. For model changes, additionally run a dryrun and one LOSO fold before long sweeps, using new output directories.
- For loss/metric edits, rerun `evaluate_noise_robustness.py` on a single checkpoint to confirm curves.
- The reviewer-facing repo excludes full manuscripts, submission assets, build trees, queue automation, and internal submission-packaging utilities. Refresh checksum manifests after an authorized package change; hashes attest bytes, not scientific validity.

## Commit & Pull Request Guidelines
- Commits: imperative subject, <=72 chars, optional scope (`fix: guard empty SisFall split`); never commit datasets or checkpoints.
- PRs: describe intent, commands run, before/after metrics or figure diffs; link issue/task id; flag new CLI options or breaking changes; keep changes reviewable.

## Environment Notes
- Historical local work used a private `SCI666` conda environment. Do not assume it exists on another machine.
- The portable requirement is simpler: use any Python environment that can install `requirements.txt`.
- Prefer GPU execution for training when available, but do not hard-code a private environment name into reviewer-facing instructions.

