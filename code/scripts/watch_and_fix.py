#!/usr/bin/env python3
import subprocess
import time
import sys
import re
from pathlib import Path
from datetime import datetime

class TaskWatcher:
    def __init__(self, task_name, train_cmd, output_dir, max_retries=3):
        self.task_name = task_name
        self.train_cmd = train_cmd
        self.output_dir = Path(output_dir)
        self.max_retries = max_retries
        self.retry_count = 0
        self.process = None
        self.last_log_size = 0

    def start_training(self, resume=False):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cmd = self.train_cmd
        if resume:
            ckpt = self.find_checkpoint()
            if ckpt:
                cmd += f" --resume {ckpt}"

        print(f"[{datetime.now()}] Starting: {self.task_name}")
        self.process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def find_checkpoint(self):
        ckpts = list(self.output_dir.glob("ckpt_last_*.pth"))
        return ckpts[0] if ckpts else None

    def check_status(self):
        log_file = self.output_dir / "experiment.log"
        if not log_file.exists():
            return "running", None

        current_size = log_file.stat().st_size
        if current_size == self.last_log_size:
            return "running", None

        self.last_log_size = current_size

        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Check for errors
        for line in reversed(lines[-50:]):
            if 'RuntimeError' in line or 'ERROR' in line or 'Traceback' in line:
                error_context = '\n'.join(lines[-20:])
                return "error", error_context

        # Check for completion
        if any('Final Summary' in line for line in lines[-10:]):
            return "completed", None

        return "running", None

    def auto_fix(self, error_msg):
        print(f"[{datetime.now()}] Error detected in {self.task_name}")
        print(error_msg[:300])

        # Pattern-based fixes
        if re.search(r"size of tensor a \((\d+)\) must match.*tensor b \((\d+)\)", error_msg):
            print("Fix: Adding --in-channels 3")
            self.train_cmd += " --in-channels 3"
            return "RESTART_REQUIRED"

        if "CUDA out of memory" in error_msg or "OutOfMemoryError" in error_msg:
            match = re.search(r"--batch-size (\d+)", self.train_cmd)
            if match:
                old_bs = int(match.group(1))
                new_bs = max(4, old_bs // 2)
                self.train_cmd = self.train_cmd.replace(f"--batch-size {old_bs}", f"--batch-size {new_bs}")
                print(f"Fix: Reduced batch size {old_bs} -> {new_bs}")
                return "RESTART_REQUIRED"

        if "size mismatch" in error_msg and "checkpoint" in error_msg.lower():
            print("Fix: Checkpoint incompatible, restarting from scratch")
            return "RESTART_REQUIRED"

        return "RESUME_OK"

    def run(self):
        self.start_training()

        while True:
            if self.process and self.process.poll() is not None:
                status, error = self.check_status()

                if status == "completed":
                    print(f"[{datetime.now()}] {self.task_name} completed successfully!")
                    break

                if status == "error":
                    self.retry_count += 1

                    if self.retry_count > self.max_retries:
                        print(f"[{datetime.now()}] {self.task_name} failed after {self.max_retries} retries")
                        break

                    fix_result = self.auto_fix(error)

                    time.sleep(5)

                    if fix_result == "RESTART_REQUIRED":
                        self.start_training(resume=False)
                    else:
                        self.start_training(resume=True)
                else:
                    print(f"[{datetime.now()}] {self.task_name} stopped unexpectedly, restarting...")
                    self.start_training(resume=True)

            time.sleep(30)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python watch_and_fix.py <task_name> <train_cmd> <output_dir>")
        sys.exit(1)

    task_name = sys.argv[1]
    train_cmd = sys.argv[2]
    output_dir = sys.argv[3]

    watcher = TaskWatcher(task_name, train_cmd, output_dir)
    watcher.run()
