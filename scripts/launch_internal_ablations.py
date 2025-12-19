from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, TextIO


@dataclass(frozen=True)
class AblationJob:
    tag: str
    ablation_spec: str
    out_dir_name: str


@dataclass
class RunningJob:
    job: AblationJob
    process: subprocess.Popen[bytes]
    log_file: TextIO


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "launcher.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
    )


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parent.parent


def get_free_gpu_memory_mb(gpu_index: int) -> Optional[int]:
    """
    Returns free VRAM (MiB) for the given physical GPU index, or None if query fails.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        logging.warning("nvidia-smi query failed (%s). Proceeding without VRAM gating.", exc)
        return None

    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        logging.warning("nvidia-smi returned no GPU lines. Proceeding without VRAM gating.")
        return None
    if gpu_index < 0 or gpu_index >= len(lines):
        logging.warning(
            "Requested gpu_index=%s but only %s GPUs reported. Proceeding without VRAM gating.",
            gpu_index,
            len(lines),
        )
        return None
    try:
        return int(lines[gpu_index])
    except ValueError:
        logging.warning("Unexpected nvidia-smi free-memory value: %r", lines[gpu_index])
        return None


def wait_for_vram(
    gpu_index: int,
    min_free_mb: int,
    poll_seconds: int,
    max_wait_seconds: int,
) -> bool:
    """
    Wait until free VRAM >= min_free_mb. Returns True if condition met, else False on timeout.
    If nvidia-smi is unavailable, returns True immediately.
    """
    start = time.time()
    while True:
        free_mb = get_free_gpu_memory_mb(gpu_index)
        if free_mb is None:
            return True
        if free_mb >= min_free_mb:
            logging.info("GPU %s free VRAM: %s MiB (>= %s MiB).", gpu_index, free_mb, min_free_mb)
            return True

        waited = int(time.time() - start)
        if waited >= max_wait_seconds:
            logging.error(
                "Timeout waiting for VRAM on GPU %s: free=%s MiB < %s MiB after %ss.",
                gpu_index,
                free_mb,
                min_free_mb,
                waited,
            )
            return False

        logging.info(
            "Waiting for VRAM on GPU %s: free=%s MiB < %s MiB (waited %ss).",
            gpu_index,
            free_mb,
            min_free_mb,
            waited,
        )
        time.sleep(poll_seconds)


def build_train_cmd(
    *,
    dataset: str,
    data_root: Path,
    out_dir: Path,
    eval_mode: str,
    epochs: int,
    batch_size: int,
    seeds: Iterable[int],
    ablation_spec: str,
    amp: bool,
    weighted_loss: bool,
    loso_max_folds: Optional[int],
    extra_args: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        "DMC_Net_experiments.py",
        "--dataset",
        dataset,
        "--data-root",
        str(data_root),
        "--out-dir",
        str(out_dir),
        "--model",
        "amsv2",
        "--eval-mode",
        eval_mode,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--ablation",
        ablation_spec,
        "--seeds",
        *[str(s) for s in seeds],
    ]
    if amp:
        cmd.append("--amp")
    if weighted_loss:
        cmd.append("--weighted-loss")
    if loso_max_folds is not None:
        cmd.extend(["--loso-max-folds", str(loso_max_folds)])
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def launch_job(
    *,
    job: AblationJob,
    repo_root: Path,
    code_dir: Path,
    log_dir: Path,
    env: dict[str, str],
    args: argparse.Namespace,
) -> tuple[AblationJob, subprocess.Popen[bytes], TextIO]:
    out_dir = repo_root / "outputs" / job.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_train_cmd(
        dataset=args.dataset,
        data_root=repo_root / args.data_root,
        out_dir=out_dir,
        eval_mode=args.eval_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seeds=args.seeds,
        ablation_spec=job.ablation_spec,
        amp=args.amp,
        weighted_loss=args.weighted_loss,
        loso_max_folds=args.loso_max_folds,
        extra_args=args.extra_args,
    )

    log_path = log_dir / f"{job.tag}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    logging.info("Launching %s -> %s", job.tag, log_path)
    logging.info("Command: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        cwd=code_dir,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return job, proc, log_file


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch internal AMSV2 component ablations with GPU VRAM gating.")

    p.add_argument("--dataset", default="sisfall", choices=["dryrun", "sisfall", "mobiact", "unimib", "kfall"])
    p.add_argument("--data-root", default="data", help="Dataset root (repo-relative).")
    p.add_argument("--eval-mode", default="loso", choices=["holdout", "loso"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123], help="Seeds to run per ablation.")

    p.add_argument("--amp", action="store_true", default=True, help="Enable AMP (default: on).")
    p.add_argument(
        "--no-amp",
        action="store_false",
        dest="amp",
        help="Disable AMP.",
    )
    p.add_argument("--weighted-loss", action="store_true", default=True, help="Enable weighted loss (default: on).")
    p.add_argument(
        "--no-weighted-loss",
        action="store_false",
        dest="weighted_loss",
        help="Disable weighted loss.",
    )
    p.add_argument("--loso-max-folds", type=int, default=None, help="Optional cap for quick runs.")

    p.add_argument("--startup-delay", type=int, default=90, help="Delay before launching ablations (seconds).")
    p.add_argument("--gpu-index", type=int, default=0, help="Physical GPU index for nvidia-smi VRAM check.")
    p.add_argument("--min-free-mb", type=int, default=4000, help="Minimum free VRAM required to launch a job.")
    p.add_argument("--poll-seconds", type=int, default=30, help="VRAM polling interval while waiting.")
    p.add_argument("--max-wait-seconds", type=int, default=7200, help="Max wait for VRAM per job (seconds).")
    p.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="If set, exported as CUDA_VISIBLE_DEVICES for launched jobs (e.g. '0' or '0,1').",
    )
    p.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args forwarded to DMC_Net_experiments.py (prefix with --).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = repo_root_from_here()
    code_dir = repo_root / "code"
    if not code_dir.exists():
        raise FileNotFoundError(f"Could not find code directory at: {code_dir}")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = repo_root / "logs" / "internal_ablations" / run_stamp
    setup_logging(log_dir)

    logging.info("Startup delay: %ss", args.startup_delay)
    time.sleep(args.startup_delay)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        logging.info("Set CUDA_VISIBLE_DEVICES=%s", args.cuda_visible_devices)

    jobs = [
        AblationJob(
            tag="no_dks",
            ablation_spec="mspa:False,dks:False",
            out_dir_name="ablation_no_dks",
        ),
        AblationJob(
            tag="no_faa",
            ablation_spec="mspa:False,faa:False",
            out_dir_name="ablation_no_faa",
        ),
    ]

    processes: list[RunningJob] = []
    for job in jobs:
        ok = wait_for_vram(
            args.gpu_index,
            args.min_free_mb,
            args.poll_seconds,
            args.max_wait_seconds,
        )
        if not ok:
            logging.error("Skipping %s due to VRAM timeout.", job.tag)
            continue
        launched_job, proc, log_file = launch_job(
            job=job,
            repo_root=repo_root,
            code_dir=code_dir,
            log_dir=log_dir,
            env=env,
            args=args,
        )
        processes.append(RunningJob(job=launched_job, process=proc, log_file=log_file))

    if not processes:
        logging.error("No ablation jobs launched.")
        return 2

    exit_code = 0
    try:
        for running in processes:
            ret = running.process.wait()
            try:
                running.log_file.close()
            except Exception:
                pass
            if ret == 0:
                logging.info("%s finished successfully.", running.job.tag)
            else:
                logging.error("%s failed with exit code %s.", running.job.tag, ret)
                exit_code = ret
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt: terminating ablation jobs...")
        for running in processes:
            running.process.terminate()
            try:
                running.log_file.close()
            except Exception:
                pass
        return 130

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
