# PhyCL-Net Fall Detection

Physics-inspired contrastive lightweight network for wearable fall detection. This repo bundles training code, analysis scripts, and manuscript assets; datasets and checkpoints stay local to keep runs reproducible.

## Environment
- Activate env before any install/run: `conda activate SCI666`
- Use Tsinghua PyPI mirror: `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`
- Install deps: `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
- GPU check: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`

## Quick Start
- Smoke test: `python code/DMC_Net_experiments.py --dataset dryrun --epochs 2 --batch-size 4 --profile`
- Full SisFall LOSO example: `python code/DMC_Net_experiments.py --dataset sisfall --data-root ./data --model dmc --epochs 100 --weighted-loss --amp --seed 42`
- Baselines: `python code/scripts/train_baselines.py --data-root ./data --epochs 50`
- Noise robustness check: `python code/scripts/eval_noise_robustness.py --ckpt outputs/stage1_amsv2_final/ckpt_best_seed42_loso_SA01.pth --data-root ./data --figure-dir ./figures/demo`

## Project Layout
- `code/` - training entry (`DMC_Net_experiments.py`), models, losses, and analysis scripts.
- `data/` - local datasets (SisFall/KFall/UniMiB_SHAR/MobiFall); read-only, not versioned.
- `outputs/`, `figures/`, `logs/` - checkpoints, metrics, and plots from runs (kept locally, ignored by git).
- `docs/` - reproducibility manifest, submission checklist, analysis plans, and experiment log (`docs/experiments/1.md`).
- `paper/arXiv/` - LaTeX manuscript and bib; figures in `paper/arXiv/figures/`.
- `automation/` - queue helpers for training sweeps; `scripts/` - misc utilities.
- `tools/` - auxiliary MCP servers/tools (see their READMEs).

## Reproducibility Notes
- Keep `data/`, `outputs/`, `figures/`, and checkpoints intact; they are ignored by git but required to reproduce reported numbers.
- Use seeds/config paths from `docs/REPRODUCIBILITY_MANIFEST.json`.
- Step-by-step guide: `docs/REPRODUCIBILITY.md`.
- Prefer `python code/DMC_Net_experiments.py ...` entrypoints over ad-hoc scripts to stay aligned with logged runs.
- For manuscript assets, rely on `paper/arXiv/main.tex` plus archived variants in `paper/arXiv/figures/archive/`.

For detailed ground rules, see `AGENTS.md`.
