from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# --- Configuration ---
SEEDS = [456, 789, 1024]
DELAY_SECONDS = 30
GPU_ID = "0"
# ---------------------

def main() -> int:
    # 1. Locate the code directory robustly
    # Script is likely in ROOT/scripts/, so parent.parent is ROOT
    repo_root = Path(__file__).resolve().parent.parent
    code_dir = repo_root / "code"
    
    # Validation
    if not code_dir.exists():
        # Fallback: Maybe the script is in ROOT?
        repo_root = Path(__file__).resolve().parent
        code_dir = repo_root / "code"
        if not code_dir.exists():
             raise FileNotFoundError(f"Could not find code directory. Expected at: {code_dir}")

    # 2. Setup Logs Directory
    # Create a 'logs' folder in the repo root so outputs don't clutter scripts/
    log_dir = repo_root / "logs" / "parallel_runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Logs will be saved to: {log_dir}")

    # 3. Environment Setup
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID

    processes: list[tuple[int, subprocess.Popen,  TextIO]] = []

    print(f"🚀 Launching {len(SEEDS)} parallel training jobs on GPU {GPU_ID}...")

    for idx, seed in enumerate(SEEDS):
        if idx > 0:
            print(f"⏳ Waiting {DELAY_SECONDS}s before launching next seed to stabilize GPU...", flush=True)
            time.sleep(DELAY_SECONDS)

        cmd = [
            sys.executable,
            "DMC_Net_experiments.py",
            "--dataset", "sisfall",
            "--model", "amsv2",
            "--eval-mode", "loso",
            "--epochs", "100",
            "--ablation", "mspa:False", # 核心：Lite模式
            "--seed", str(seed),
        ]

        # Log file per seed
        log_file_path = log_dir / f"train_seed_{seed}.log"
        log_file = open(log_file_path, "w", encoding="utf-8")

        print(f"   ▶️ Starting Seed {seed} -> Log: {log_file_path.name}", flush=True)
        
        # Popen with stdout/stderr redirection
        process = subprocess.Popen(
            cmd,
            cwd=code_dir,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT, # Merge stderr into stdout file
        )
        processes.append((seed, process, log_file))

    # 4. Monitor Loop
    print("\n✅ All jobs launched. Waiting for completion...")
    exit_code = 0
    
    try:
        for seed, process, log_file in processes:
            ret = process.wait()
            log_file.close() # Important: Close file handle
            
            if ret == 0:
                print(f"   🎉 Seed {seed} finished SUCCESSFULLY.")
            else:
                print(f"   ❌ Seed {seed} FAILED with exit code {ret}. Check logs!")
                exit_code = ret
    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt! Killing all background processes...")
        for _, p, f in processes:
            p.terminate()
            f.close()
        return 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())