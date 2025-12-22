# Reproducibility Guide

## Environment (Required)
```bash
conda activate SCI666
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## Data (Local / Not Versioned)
- Put datasets under `data/` and pass `--data-root ./data`.
- This repo does not commit datasets or checkpoints; keep your local `data/` and `outputs/` intact.

## Smoke Check
```bash
python code/DMC_Net_experiments.py --dataset dryrun --epochs 2 --batch-size 4 --profile
```

## SisFall LOSO (Example)
```bash
python code/DMC_Net_experiments.py --dataset sisfall --data-root ./data --model dmc --epochs 100 --weighted-loss --amp --seed 42
```

## Where Numbers Come From
- Experiment log (source of truth): `docs/experiments/1.md`
- Repro manifest (commands/config snapshot): `docs/REPRODUCIBILITY_MANIFEST.json`
