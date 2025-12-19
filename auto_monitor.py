import os
import subprocess
import sys
import time

SEEDS = [456, 789, 1024]
PYTHON_EXEC = sys.executable  # 自动使用 SCI666 环境的 python
GPU_ID = "0"
OUTPUT_DIR = "outputs/ablation_no_mspa"  # 统一输出目录


def resolve_dir(seed: int) -> str:
    """所有 seed 共用同一个输出目录"""
    return OUTPUT_DIR


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check_and_run() -> bool:
    all_done = True

    for seed in SEEDS:
        dir_path = resolve_dir(seed)
        ckpt_path = os.path.join(dir_path, f"ckpt_best_seed{seed}_loso_SA01.pth")
        loso_results_path = os.path.join(dir_path, f"loso_results_seed{seed}.json")
        done_flag = os.path.join(dir_path, f"post_process_done_seed{seed}.txt")

        # 等待训练完成后再触发后处理 (检查 loso_results_seed{seed}.json)
        if not os.path.exists(loso_results_path) or not os.path.exists(ckpt_path):
            all_done = False
            continue

        # 已处理则跳过
        if os.path.exists(done_flag):
            continue

        log(f"[Seed {seed}] 训练完成，开始运行补充实验...")

        try:
            # 1. 运行噪声测试 (Noise Test)
            log(f"   [Seed {seed}] Running Noise Robustness Test...")
            cmd_noise = [
                PYTHON_EXEC,
                "test.py",
                "--checkpoint",
                ckpt_path,
                "--add-noise",
                "--snr-levels",
                "10",
                "20",
                "30",
            ]
            subprocess.run(cmd_noise, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_ID))

            # 2. 运行速度测试 (Latency Test)
            log(f"   [Seed {seed}] Running Latency Test...")
            cmd_speed = [
                PYTHON_EXEC,
                "speed_test.py",
                "--checkpoint",
                ckpt_path,
            ]
            with open(os.path.join(dir_path, "speed_results.txt"), "w") as f:
                subprocess.run(cmd_speed, stdout=f, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_ID))

            # 3. 标记为已完成
            with open(done_flag, "w") as f:
                f.write("Done")

            log(f"[Seed {seed}] 补充实验已完成。")

        except subprocess.CalledProcessError as e:
            log(f"[Seed {seed}] 实验出错: {e}")
        except Exception as e:
            log(f"[Seed {seed}] 未知错误: {e}")

    return all_done


if __name__ == "__main__":
    log(f"Watchdog 启动，监控 Seeds: {SEEDS}")
    log(f"   目标目录: {OUTPUT_DIR}")
    log(f"   目标 Python: {PYTHON_EXEC}")

    while True:
        log("Monitoring loop tick...")
        finished = check_and_run()
        if finished:
            if all(os.path.exists(os.path.join(resolve_dir(s), f"post_process_done_seed{s}.txt")) for s in SEEDS):
                log("所有 Seed 的训练和测试都结束了。")
                break

        # 每 60 秒检查一次
        time.sleep(60)
