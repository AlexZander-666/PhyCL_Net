#!/usr/bin/env python3
"""
自动训练监控脚本 - 当前训练完成后自动启动下一批模型
Monitors current training and automatically starts Transformer and InceptionTime
when any of the current models (ResNet/LSTM/AMSNetV2/TCN) completes.
"""

import os
import time
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_train_monitor.log'),
        logging.StreamHandler()
    ]
)

# 当前正在训练的模型日志路径
CURRENT_MODELS = {
    'ResNet': './outputs/stage1_resnet_final/experiment.log',
    'LSTM': './outputs/stage1_lstm_final/experiment.log',
    'AMSNetV2': './outputs/stage1_amsv2_final/experiment.log',
    'TCN': './outputs/stage1_tcn_final/experiment.log',
}

# 待启动的新模型配置
NEW_MODELS = {
    'Transformer': {
        'cmd': [
            'python', 'DMC_Net_experiments.py',
            '--dataset', 'sisfall',
            '--data-root', './data',
            '--model', 'transformer',
            '--eval-mode', 'loso',
            '--seeds', '42', '123',
            '--epochs', '50',
            '--batch-size', '128',  # 与 AMSNetV2 相同
            '--lr', '0.003',
            '--num-workers', '4',
            '--amp',
            '--weighted-loss',
            '--out-dir', './outputs/stage1_transformer_final'
        ],
        'log_file': 'outputs/transformer_final.log'
    },
    'InceptionTime': {
        'cmd': [
            'python', 'DMC_Net_experiments.py',
            '--dataset', 'sisfall',
            '--data-root', './data',
            '--model', 'inceptiontime',
            '--eval-mode', 'loso',
            '--seeds', '42', '123',
            '--epochs', '50',
            '--batch-size', '96',  # 与 LSTM 相同
            '--lr', '0.0025',
            '--num-workers', '4',
            '--amp',
            '--weighted-loss',
            '--out-dir', './outputs/stage1_inceptiontime_final'
        ],
        'log_file': 'outputs/inceptiontime_final.log'
    }
}

# 完成标志（检查日志中是否出现这些关键字）
COMPLETION_MARKERS = [
    'All experiments completed',
    'Summary results saved to',
    'LOSO validation complete'
]


def check_training_complete(log_path):
    """
    检查训练是否完成
    通过检查日志文件中的完成标志或 fold 完成数量
    """
    if not os.path.exists(log_path):
        return False

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 方法1: 检查完成标志
        for marker in COMPLETION_MARKERS:
            if marker in content:
                return True

        # 方法2: 检查是否完成了所有12个LOSO folds × 2 seeds = 24个训练任务
        # 统计 "Fold XX completed" 或类似的完成标志
        fold_complete_count = content.count('Best model saved')
        # 每个fold完成会保存一次best model，12 folds × 2 seeds = 24
        if fold_complete_count >= 24:
            return True

        # 方法3: 检查 summary_results.json 是否生成
        log_dir = os.path.dirname(log_path)
        summary_file = os.path.join(log_dir, 'summary_results.json')
        if os.path.exists(summary_file):
            return True

    except Exception as e:
        logging.error(f"Error reading log {log_path}: {e}")

    return False


def get_gpu_memory():
    """获取当前GPU内存使用情况"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            used, free = result.stdout.strip().split(', ')
            return int(used), int(free)
    except Exception as e:
        logging.error(f"Error getting GPU memory: {e}")
    return None, None


def start_new_training(model_name, config):
    """启动新的训练任务"""
    try:
        logging.info(f"🚀 Starting {model_name} training...")
        logging.info(f"Command: {' '.join(config['cmd'])}")

        # 创建输出目录
        out_dir = None
        for i, arg in enumerate(config['cmd']):
            if arg == '--out-dir' and i + 1 < len(config['cmd']):
                out_dir = config['cmd'][i + 1]
                break

        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            logging.info(f"Created output directory: {out_dir}")

        # 启动训练进程（后台运行）
        log_file = open(config['log_file'], 'w', encoding='utf-8')
        process = subprocess.Popen(
            config['cmd'],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
        )

        logging.info(f"✅ {model_name} training started (PID: {process.pid})")
        logging.info(f"Log file: {config['log_file']}")
        return process

    except Exception as e:
        logging.error(f"❌ Failed to start {model_name}: {e}")
        return None


def monitor_and_launch():
    """
    主监控循环
    """
    logging.info("=" * 80)
    logging.info("自动训练监控脚本已启动 (Auto Training Monitor Started)")
    logging.info("=" * 80)
    logging.info(f"Monitoring models: {list(CURRENT_MODELS.keys())}")
    logging.info(f"Will launch when any completes: {list(NEW_MODELS.keys())}")
    logging.info("=" * 80)

    # 跟踪新模型的启动状态
    launched = {model: False for model in NEW_MODELS.keys()}
    check_interval = 60  # 每60秒检查一次

    while True:
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 检查每个当前正在训练的模型
            completed_models = []
            for model_name, log_path in CURRENT_MODELS.items():
                if check_training_complete(log_path):
                    completed_models.append(model_name)

            # 如果有模型完成，检查是否可以启动新模型
            if completed_models:
                logging.info(f"\n[{current_time}] ✅ Detected completed models: {completed_models}")

                # 检查GPU内存
                gpu_used, gpu_free = get_gpu_memory()
                if gpu_used is not None:
                    logging.info(f"GPU Memory: {gpu_used} MiB used, {gpu_free} MiB free")

                # 启动所有尚未启动的新模型
                for model_name, config in NEW_MODELS.items():
                    if not launched[model_name]:
                        # 检查GPU内存是否足够（至少需要4GB空闲）
                        if gpu_free is not None and gpu_free < 4000:
                            logging.warning(f"⚠️  GPU memory low ({gpu_free} MiB free), waiting...")
                            time.sleep(300)  # 等待5分钟后重试
                            continue

                        process = start_new_training(model_name, config)
                        if process:
                            launched[model_name] = True
                            logging.info(f"✅ {model_name} launched successfully")
                            time.sleep(30)  # 启动间隔30秒，避免同时启动

                # 检查是否所有新模型都已启动
                if all(launched.values()):
                    logging.info("\n" + "=" * 80)
                    logging.info("🎉 All new models (Transformer & InceptionTime) have been launched!")
                    logging.info("=" * 80)
                    logging.info("Monitor script will now exit.")
                    break

            else:
                # 没有模型完成，继续等待
                logging.info(f"[{current_time}] ⏳ No models completed yet. Checking again in {check_interval}s...")

            # 等待下一次检查
            time.sleep(check_interval)

        except KeyboardInterrupt:
            logging.info("\n\n⚠️  Monitor script interrupted by user. Exiting...")
            break
        except Exception as e:
            logging.error(f"Error in monitoring loop: {e}")
            time.sleep(check_interval)


if __name__ == '__main__':
    # 检查必要的文件是否存在
    if not os.path.exists('DMC_Net_experiments.py'):
        logging.error("❌ DMC_Net_experiments.py not found in current directory!")
        exit(1)

    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)

    # 启动监控
    monitor_and_launch()
