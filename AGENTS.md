# Repository Guidelines

## Project Structure & Module Organization
- `code/`: AMSNetV2 training entry (`DMC_Net_experiments.py`), model blocks in `models/`, losses in `losses/`, utilities in `scripts/`.
- `data/`: local SisFall/KFall/UniMiB_SHAR/MobiFall datasets; treat as read-only and exclude from commits.
- `outputs/`, `figures/`: checkpoints, metrics, plots from runs; clean or redirect when starting new sweeps.
- `docs/`: training summary, reproducibility manifest, submission checklist; `automation/` queue helpers; `paper-search-mcp-main/` standalone MCP server with its own README/tests.

## Build, Test, and Development Commands
- Setup: `python -m venv .venv && .\\.venv\\Scripts\\activate && pip install -r requirements.txt`.
- Smoke check: `python code/DMC_Net_experiments.py --dataset dryrun --epochs 2 --batch-size 4 --profile` (fast env validation).
- Full SisFall: `python code/DMC_Net_experiments.py --dataset sisfall --data-root ./data --model dmc --epochs 100 --weighted-loss --amp --seed 42`.
- Baselines: `python code/scripts/train_baselines.py --data-root ./data --epochs 50`.
- Noise robustness: `python code/scripts/eval_noise_robustness.py --ckpt outputs/stage1_amsv2_final/ckpt_best_seed42_loso_SA01.pth --data-root ./data --figure-dir ./figures/demo`.
- Submission bundle: `python code/scripts/pack_sci_submission.py --output-dir ./submission_package --include-checkpoints`.
- MCP server: `(cd paper-search-mcp-main && uv run pytest && uv run -m paper_search_mcp.server)`.

## Coding Style & Naming Conventions
- PEP8, 4-space indent; add type hints when clear; keep functions small and deterministic.
- PascalCase for classes; snake_case for modules/functions/CLI flags (`--data-root`, `--batch-size`); reuse existing arg names.
- Prefer `logging` over prints; keep messages short and actionable.
- Align file naming with current artifacts (`ckpt_best_seed123_loso_SA01.pth`, `summary_results.json`, `experiment_config.yaml`); set seeds via `--seed`/`set_seed`.

## Testing Guidelines
- No dedicated unit suite for `code/`; run the dryrun plus one LOSO fold before long sweeps, and spot-check JSON/CSV/plots in `outputs/`.
- For loss/metric edits, rerun `eval_noise_robustness.py` on a single checkpoint to confirm curves.
- MCP changes: extend `paper-search-mcp-main/tests` and run `uv run pytest`.

## Commit & Pull Request Guidelines
- Commits: imperative subject, <=72 chars, optional scope (`fix: guard empty SisFall split`); never commit datasets or checkpoints.
- PRs: describe intent, commands run, before/after metrics or figure diffs; link issue/task id; flag new CLI options or breaking changes; keep changes reviewable.

## GPU Environment Setup (SCI666 Conda) - 🚨 MANDATORY / 强制执行

> **⚠️ CRITICAL RULE - MUST FOLLOW / 关键规则 - 必须遵守：**
> 
> **Kiro 在执行本项目的任何 Python 脚本或安装依赖时，必须：**
> 1. **必须先激活 SCI666 conda 环境** (`conda activate SCI666`)
> 2. **必须使用国内镜像源安装依赖** (清华源: `https://pypi.tuna.tsinghua.edu.cn/simple`)
> 3. **必须使用 GPU 进行训练和推理**
> 
> **违反以上任何一条规则都是不可接受的。在执行任何命令前，Kiro 必须确认已激活正确的 conda 环境。**

**重要：在 SCI666 的 conda 环境中使用 GPU，用国内镜像源安装本项目需要的所有依赖以及执行本项目的所有脚本。**

### 环境激活与依赖安装
- Activate environment: `conda activate SCI666`
- Configure Tsinghua mirror for pip (永久配置):
  ```bash
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  ```
- Install all dependencies with GPU support (使用国内镜像):
  ```bash
  conda activate SCI666
  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```
- Verify GPU availability: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`

### 执行本项目脚本
- Run training with GPU:
  ```bash
  python code/DMC_Net_experiments.py --dataset sisfall --data-root ./data --model dmc --epochs 100 --weighted-loss --amp --seed 42
  ```
- Run analysis scripts:
  ```bash
  python code/scripts/fine_grained_analysis.py --output-dir outputs/stage1_amsv2_final --figure-dir figures/fine_grained
  python code/scripts/eval_noise_robustness.py --ckpt outputs/stage1_amsv2_final/ckpt_best_seed42_loso_SA01.pth --data-root ./data
  ```
- Run baseline training:
  ```bash
  python code/scripts/train_baselines.py --data-root ./data --epochs 50
  ```
- Run noise robustness evaluation:
  ```bash
  python code/scripts/eval_noise_robustness.py --ckpt outputs/stage1_amsv2_final/ckpt_best_seed42_loso_SA01.pth --data-root ./data --figure-dir ./figures/demo
  ```
