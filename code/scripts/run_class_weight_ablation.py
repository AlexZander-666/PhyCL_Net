from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_STRATEGIES = ["none", "auto", "sqrt_inv_freq", "effective_num"]


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "launcher.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
    )


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def build_train_cmd(
    *,
    repo_root: Path,
    strategy: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        "code/DMC_Net_experiments.py",
        "--dataset",
        args.dataset,
        "--data-root",
        str(args.data_root),
        "--model",
        "amsv2",
        "--eval-mode",
        args.eval_mode,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--ablation",
        args.ablation,
        "--seeds",
        *[str(s) for s in args.seeds],
        "--class-weighting",
        strategy,
        "--out-dir",
        str(out_dir),
    ]
    if args.amp:
        cmd.append("--amp")
    if args.use_tfcl:
        cmd.append("--use-tfcl")
    if args.loso_max_folds is not None:
        cmd.extend(["--loso-max-folds", str(args.loso_max_folds)])
    if strategy == "effective_num":
        cmd.extend(["--effective-num-beta", str(args.effective_num_beta)])
    if args.extra_args:
        cmd.extend(args.extra_args)
    return cmd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run class weighting ablation for Lite-AMSNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="sisfall")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--eval-mode", default="loso", choices=["holdout", "loso"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    parser.add_argument("--ablation", type=str, default="mspa:False")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--use-tfcl", action="store_true", default=True)
    parser.add_argument("--no-tfcl", action="store_false", dest="use_tfcl")
    parser.add_argument("--loso-max-folds", type=int, default=None)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--effective-num-beta", type=float, default=0.999)
    parser.add_argument("--out-root", type=str, default="outputs/ablation/class_weighting")
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args forwarded to DMC_Net_experiments.py (prefix with --).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = repo_root_from_here()
    log_dir = repo_root / "logs" / "class_weight_ablation" / datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(log_dir)

    out_root = repo_root / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        logging.info("Set CUDA_VISIBLE_DEVICES=%s", args.cuda_visible_devices)

    strategies = [s.strip() for s in args.strategies if s.strip()]
    if not strategies:
        logging.error("No strategies provided.")
        return 2

    exit_code = 0
    for strategy in strategies:
        strategy_dir = out_root / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_train_cmd(
            repo_root=repo_root,
            strategy=strategy,
            out_dir=strategy_dir,
            args=args,
        )
        log_path = log_dir / f"{strategy}.log"
        logging.info("Launching strategy=%s -> %s", strategy, log_path)
        logging.info("Command: %s", " ".join(cmd))

        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd,
                cwd=repo_root,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            ret = proc.wait()
        if ret != 0:
            logging.error("Strategy %s failed with exit code %s.", strategy, ret)
            exit_code = ret
        else:
            logging.info("Strategy %s completed successfully.", strategy)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
