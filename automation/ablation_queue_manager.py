#!/usr/bin/env python3
"""
Queue manager: six ablations, batch=256, lr=0.0016, seeds=42/123, num_workers=4,
max_concurrent=2, full LOSO.
Ablations: No_DKS, No_MSPA, No_FAA, No_TFCL, Time_Only, Freq_Only.
"""
import os, time, subprocess, logging, glob

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ablation_queue_manager.log"), logging.StreamHandler()],
)

BATCH_SIZE = "256"
LR = "0.0016"
SEEDS = ["42", "123"]
NUM_WORKERS = "4"
MIN_GPU_FREE_MB = 3000
MAX_CONCURRENT = 2
CHECK_INTERVAL = 15

ABLATION_QUEUE = [
    {
        "name": "No_DKS",
        "cmd": [
            "python", "DMC_Net_experiments.py",
            "--dataset", "sisfall",
            "--data-root", "./data",
            "--model", "amsv2",
            "--eval-mode", "loso",
            "--seeds", *SEEDS,
            "--epochs", "50",
            "--batch-size", BATCH_SIZE,
            "--lr", LR,
            "--num-workers", NUM_WORKERS,
            "--amp",
            "--weighted-loss",
            "--use-tfcl",
            "--ablation", "dks:False",
            "--out-dir", "./outputs/ablation_no_dks",
        ],
        "log_file": "outputs/ablation_no_dks.log",
        "min_gpu_free": MIN_GPU_FREE_MB,
    },
    {
        "name": "No_MSPA",
        "cmd": [
            "python", "DMC_Net_experiments.py",
            "--dataset", "sisfall",
            "--data-root", "./data",
            "--model", "amsv2",
            "--eval-mode", "loso",
            "--seeds", *SEEDS,
            "--epochs", "50",
            "--batch-size", BATCH_SIZE,
            "--lr", LR,
            "--num-workers", NUM_WORKERS,
            "--amp",
            "--weighted-loss",
            "--use-tfcl",
            "--ablation", "mspa:False",
            "--out-dir", "./outputs/ablation_no_mspa",
        ],
        "log_file": "outputs/ablation_no_mspa.log",
        "min_gpu_free": MIN_GPU_FREE_MB,
    },
    {
        "name": "No_FAA",
        "cmd": [
            "python", "DMC_Net_experiments.py",
            "--dataset", "sisfall",
            "--data-root", "./data",
            "--model", "amsv2",
            "--eval-mode", "loso",
            "--seeds", *SEEDS,
            "--epochs", "50",
            "--batch-size", BATCH_SIZE,
            "--lr", LR,
            "--num-workers", NUM_WORKERS,
            "--amp",
            "--weighted-loss",
            "--use-tfcl",
            "--ablation", "faa:False",
            "--out-dir", "./outputs/ablation_no_faa",
        ],
        "log_file": "outputs/ablation_no_faa.log",
        "min_gpu_free": MIN_GPU_FREE_MB,
    },
    {
        "name": "No_TFCL",
        "cmd": [
            "python", "DMC_Net_experiments.py",
            "--dataset", "sisfall",
            "--data-root", "./data",
            "--model", "amsv2",
            "--eval-mode", "loso",
            "--seeds", *SEEDS,
            "--epochs", "50",
            "--batch-size", BATCH_SIZE,
            "--lr", LR,
            "--num-workers", NUM_WORKERS,
            "--amp",
            "--weighted-loss",
            "--out-dir", "./outputs/ablation_no_tfcl",
        ],
        "log_file": "outputs/ablation_no_tfcl.log",
        "min_gpu_free": MIN_GPU_FREE_MB,
    },
    {
        "name": "Time_Only",
        "cmd": [
            "python", "DMC_Net_experiments.py",
            "--dataset", "sisfall",
            "--data-root", "./data",
            "--model", "amsv2",
            "--eval-mode", "loso",
            "--seeds", *SEEDS,
            "--epochs", "50",
            "--batch-size", BATCH_SIZE,
            "--lr", LR,
            "--num-workers", NUM_WORKERS,
            "--amp",
            "--weighted-loss",
            "--use-tfcl",
            "--ablation", "freq_only",
            "--out-dir", "./outputs/ablation_time_only",
        ],
        "log_file": "outputs/ablation_time_only.log",
        "min_gpu_free": MIN_GPU_FREE_MB,
    },
    {
        "name": "Freq_Only",
        "cmd": [
            "python", "DMC_Net_experiments.py",
            "--dataset", "sisfall",
            "--data-root", "./data",
            "--model", "amsv2",
            "--eval-mode", "loso",
            "--seeds", *SEEDS,
            "--epochs", "50",
            "--batch-size", BATCH_SIZE,
            "--lr", LR,
            "--num-workers", NUM_WORKERS,
            "--amp",
            "--weighted-loss",
            "--use-tfcl",
            "--ablation", "time_only",
            "--out-dir", "./outputs/ablation_freq_only",
        ],
        "log_file": "outputs/ablation_freq_only.log",
        "min_gpu_free": MIN_GPU_FREE_MB,
    },
]


def get_gpu_memory():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            used, free = result.stdout.strip().split(", ")
            return int(used), int(free)
    except Exception as e:
        logging.error(f"Error getting GPU memory: {e}")
    return None, None


def get_out_dir_from_cmd(cmd):
    for i, arg in enumerate(cmd):
        if arg == "--out-dir" and i + 1 < len(cmd):
            return cmd[i + 1]
    return None


def has_summary(out_dir):
    return os.path.isfile(os.path.join(out_dir, "summary_results.json"))


def has_checkpoint(out_dir):
    return bool(glob.glob(os.path.join(out_dir, "ckpt_last_seed*.pth")))


def start_ablation_experiment(config, cmd_override=None):
    try:
        cmd = cmd_override or config["cmd"]
        logging.info("=" * 80)
        logging.info(f"Starting: {config['name']}")
        logging.info(f"Cmd: {' '.join(cmd)}")
        out_dir = get_out_dir_from_cmd(cmd)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        log_file = open(config["log_file"], "w", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
        )
        logging.info(f"? Launched {config['name']} (PID: {process.pid})")
        logging.info(f"Log file: {config['log_file']}")
        logging.info("=" * 80)
        return process
    except Exception as e:
        logging.error(f"? Failed to launch {config['name']}: {e}")
        return None


def monitor_and_queue():
    logging.info("=" * 80)
    logging.info("Ablation queue started")
    logging.info(f"Queue: {[exp['name'] for exp in ABLATION_QUEUE]}")
    logging.info("=" * 80)

    launched = []
    queue = ABLATION_QUEUE.copy()

    while True:
        running = []
        for item in launched:
            proc = item["process"]
            if proc.poll() is None:
                running.append(item)
            else:
                logging.info(f"? Finished: {item['name']} (PID {item['pid']})")
                out_dir_done = get_out_dir_from_cmd(item["start_cmd"])
                if out_dir_done and not has_summary(out_dir_done):
                    logging.warning(f"{item['name']} finished without summary_results.json; re-queue to resume.")
                    queue.append(item["config"])
        launched = running

        while len(launched) < MAX_CONCURRENT and queue:
            cfg = queue[0]
            out_dir = get_out_dir_from_cmd(cfg["cmd"])
            if out_dir and has_summary(out_dir):
                logging.info(f"Skip {cfg['name']} - summary_results.json found in {out_dir}")
                queue.pop(0)
                continue
            used, free = get_gpu_memory()
            if free is not None and free < cfg["min_gpu_free"]:
                logging.warning(f"GPU free {free} MiB < {cfg['min_gpu_free']} MiB, wait {CHECK_INTERVAL}s")
                break
            launch_cmd = cfg["cmd"]
            if out_dir and has_checkpoint(out_dir):
                logging.info(f"Resuming {cfg['name']} from checkpoints in {out_dir}")
                launch_cmd = cfg["cmd"] + ["--resume", out_dir]
            proc = start_ablation_experiment(cfg, launch_cmd)
            if proc:
                launched.append({"name": cfg["name"], "pid": proc.pid, "process": proc, "config": cfg, "start_cmd": launch_cmd})
                queue.pop(0)
                time.sleep(10)
            else:
                break

        if not queue and not launched:
            logging.info("=" * 80)
            logging.info("?? All ablations completed")
            logging.info("=" * 80)
            break

        logging.info(f"Running: {len(launched)}/{MAX_CONCURRENT} | Queue: {len(queue)}")
        if queue:
            logging.info(f"Next: {queue[0]['name']}")
        time.sleep(CHECK_INTERVAL)


def main():
    if not os.path.exists("DMC_Net_experiments.py"):
        logging.error("Missing DMC_Net_experiments.py")
        return
    os.makedirs("outputs", exist_ok=True)
    monitor_and_queue()


if __name__ == "__main__":
    main()
