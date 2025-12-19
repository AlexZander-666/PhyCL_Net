#!/usr/bin/env python3
"""
pack_sci_submission.py - SCI 4区期刊投稿材料打包脚本

生成完整的投稿材料包，包括：
1. 源代码（核心模块）
2. 实验结果（JSON + CSV）
3. 图表（高清PNG/PDF）
4. LaTeX表格
5. 可复现性清单
6. 提交检查清单

Usage:
    python scripts/pack_sci_submission.py --output-dir ./submission_package
    python scripts/pack_sci_submission.py --output-dir ./submission_package --include-checkpoints
"""

import os
import sys
import json
import shutil
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ========================= Configuration =========================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DOCS_ROOT = PROJECT_ROOT.parent / "docs"

# Core model source files to include
CORE_SOURCE_FILES = [
    "models/__init__.py",
    "models/ams_net_v2.py",
    "models/modules/__init__.py",
    "models/modules/dks.py",
    "models/modules/faa.py",
    "models/modules/mspa.py",
    "models/modules/spectral.py",
    "models/modules/attention.py",
    "models/modules/efficient.py",
    "losses/__init__.py",
    "losses/tfcl.py",
    "losses/contrastive.py",
    "DMC_Net_experiments.py",
    "requirements.txt",
]

# Key experiment result directories
KEY_EXPERIMENTS = [
    "stage1_amsv2_final",
    "stage1_lstm_final",
    "stage1_resnet_final",
    "stage1_tcn_final",
    "stage1_transformer_final",
    "stage1_inceptiontime_final",
    "ablation_no_mspa",
    "ablation_no_tfcl",
    "ablation_no_dks",
    "ablation_no_faa",
    "ablation_time_only",
    "ablation_freq_only",
    "rerun_ablation_no_mspa_final",
    "rerun_ablation_no_tfcl_final",
]

# Result files to collect from each experiment
RESULT_FILES = [
    "summary_results.json",
    "experiment_config.yaml",
    "loso_results_seed42.json",
    "loso_results_seed123.json",
    "split_stats_seed42.json",
    "split_stats_seed123.json",
]

# Figure directories
FIGURE_DIRS = [
    (PROJECT_ROOT / "figures", "figures"),
    (DOCS_ROOT / "figures" / "paper", "figures/paper"),
]

# Table files
TABLE_FILES = [
    (DOCS_ROOT / "tables" / "paper_main_table.csv", "tables/paper_main_table.csv"),
    (DOCS_ROOT / "tables" / "paper_main_table.tex", "tables/paper_main_table.tex"),
    (DOCS_ROOT / "tables" / "paper_safety_table.csv", "tables/paper_safety_table.csv"),
    (DOCS_ROOT / "tables" / "paper_safety_table.tex", "tables/paper_safety_table.tex"),
]

# Documentation files
DOC_FILES = [
    (PROJECT_ROOT / "CLAUDE.md", "docs/MODEL_README.md"),
    (DOCS_ROOT / "JOURNAL_TARGETS_SCI.md", "docs/JOURNAL_TARGETS.md"),
    (DOCS_ROOT / "reports" / "submission_materials_draft.md", "docs/submission_materials_draft.md"),
]


# ========================= Helper Functions =========================

def get_git_info() -> Dict[str, str]:
    """Get git commit info for reproducibility."""
    info = {}
    try:
        info['commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL
        ).decode().strip()
        info['branch'] = subprocess.check_output(
            ['git', 'branch', '--show-current'],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL
        ).decode().strip()
        info['dirty'] = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL
        ).decode().strip())
    except Exception as e:
        logging.warning(f"Failed to get git info: {e}")
    return info


def get_python_env() -> Dict[str, str]:
    """Get Python environment info."""
    import platform
    env = {
        'python_version': sys.version,
        'platform': platform.platform(),
    }
    try:
        import torch
        env['torch_version'] = torch.__version__
        env['cuda_available'] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            env['cuda_version'] = torch.version.cuda or "N/A"
            env['gpu_name'] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return env


def load_summary_results(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """Load summary_results.json from experiment directory."""
    summary_path = exp_dir / "summary_results.json"
    if summary_path.exists():
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load {summary_path}: {e}")
    return None


def extract_key_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    """Extract key metrics from summary results."""
    metrics = {}
    key_fields = [
        ('accuracy_mean_mean', 'Accuracy'),
        ('macro_f1_mean_mean', 'Macro F1'),
        ('sensitivity_mean_mean', 'Sensitivity'),
        ('specificity_mean_mean', 'Specificity'),
        ('fall_f1_mean_mean', 'Fall F1'),
        ('detection_rate_mean_mean', 'Detection Rate'),
        ('g_mean_mean_mean', 'G-Mean'),
    ]
    for key, label in key_fields:
        if key in summary:
            val = summary[key]
            if val is not None and not (isinstance(val, float) and val != val):  # not NaN
                metrics[label] = round(val * 100, 2) if val <= 1.0 else round(val, 2)
    return metrics


def generate_comparison_table(results: Dict[str, Dict[str, float]]) -> str:
    """Generate LaTeX comparison table from results."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comparison of Models on SisFall Dataset (LOSO, 12-fold)}",
        r"\label{tab:comparison}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & Accuracy & Macro F1 & Sensitivity & Specificity & Fall F1 & Detection Rate \\",
        r"\midrule",
    ]

    # Sort by Accuracy (descending)
    sorted_models = sorted(results.items(), key=lambda x: x[1].get('Accuracy', 0), reverse=True)

    for model, metrics in sorted_models:
        row = [model]
        for col in ['Accuracy', 'Macro F1', 'Sensitivity', 'Specificity', 'Fall F1', 'Detection Rate']:
            val = metrics.get(col, '-')
            if isinstance(val, (int, float)):
                row.append(f"{val:.2f}\\%")
            else:
                row.append(str(val))
        lines.append(" & ".join(row) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_csv_table(results: Dict[str, Dict[str, float]]) -> str:
    """Generate CSV comparison table."""
    headers = ['Model', 'Accuracy', 'Macro F1', 'Sensitivity', 'Specificity', 'Fall F1', 'Detection Rate']
    lines = [','.join(headers)]

    sorted_models = sorted(results.items(), key=lambda x: x[1].get('Accuracy', 0), reverse=True)
    for model, metrics in sorted_models:
        row = [model]
        for col in headers[1:]:
            val = metrics.get(col, '')
            row.append(str(val) if val != '' else '')
        lines.append(','.join(row))

    return '\n'.join(lines)


def copy_file_safe(src: Path, dst: Path):
    """Safely copy a file, creating parent directories if needed."""
    if not src.exists():
        logging.warning(f"Source file not found: {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    logging.info(f"Copied: {src.name} -> {dst}")
    return True


def copy_dir_safe(src: Path, dst: Path, extensions: Optional[List[str]] = None):
    """Copy directory contents, optionally filtering by extension."""
    if not src.exists():
        logging.warning(f"Source directory not found: {src}")
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    count = 0

    for item in src.rglob('*'):
        if item.is_file():
            if extensions and item.suffix.lower() not in extensions:
                continue
            rel_path = item.relative_to(src)
            target = dst / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            count += 1

    logging.info(f"Copied {count} files from {src.name}")
    return count


# ========================= Main Packaging Function =========================

def pack_submission(output_dir: Path, include_checkpoints: bool = False):
    """Create the complete submission package."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        logging.warning(f"Output directory exists, will overwrite: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Creating submission package at: {output_dir}")

    # ==================== 1. Source Code ====================
    logging.info("=" * 50)
    logging.info("Step 1: Collecting source code...")

    code_dir = output_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)

    for rel_path in CORE_SOURCE_FILES:
        src = PROJECT_ROOT / rel_path
        dst = code_dir / rel_path
        copy_file_safe(src, dst)

    # ==================== 2. Experimental Results ====================
    logging.info("=" * 50)
    logging.info("Step 2: Collecting experimental results...")

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    outputs_root = PROJECT_ROOT / "outputs"

    for exp_name in KEY_EXPERIMENTS:
        exp_dir = outputs_root / exp_name
        if not exp_dir.exists():
            logging.warning(f"Experiment not found: {exp_name}")
            continue

        exp_out = results_dir / exp_name
        exp_out.mkdir(parents=True, exist_ok=True)

        # Copy result files
        for fname in RESULT_FILES:
            src = exp_dir / fname
            if src.exists():
                copy_file_safe(src, exp_out / fname)

        # Copy error analysis CSVs
        for csv_file in exp_dir.glob("errors_*.csv"):
            copy_file_safe(csv_file, exp_out / csv_file.name)

        # Load and extract metrics
        summary = load_summary_results(exp_dir)
        if summary:
            metrics = extract_key_metrics(summary)
            if metrics:
                # Create readable name
                display_name = exp_name.replace('stage1_', '').replace('_final', '')
                display_name = display_name.replace('ablation_', 'Ablation: ')
                display_name = display_name.replace('rerun_', '')
                display_name = display_name.upper() if 'amsv2' in display_name.lower() else display_name.title()
                all_metrics[display_name] = metrics

        # Optionally copy checkpoints
        if include_checkpoints:
            ckpt_dir = exp_out / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            for ckpt in exp_dir.glob("ckpt_best_*.pth"):
                copy_file_safe(ckpt, ckpt_dir / ckpt.name)

    # Generate summary tables
    if all_metrics:
        # LaTeX table
        latex_table = generate_comparison_table(all_metrics)
        with open(results_dir / "comparison_table.tex", 'w', encoding='utf-8') as f:
            f.write(latex_table)
        logging.info("Generated: comparison_table.tex")

        # CSV table
        csv_table = generate_csv_table(all_metrics)
        with open(results_dir / "comparison_table.csv", 'w', encoding='utf-8') as f:
            f.write(csv_table)
        logging.info("Generated: comparison_table.csv")

        # JSON summary
        with open(results_dir / "all_metrics_summary.json", 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        logging.info("Generated: all_metrics_summary.json")

    # ==================== 3. Figures ====================
    logging.info("=" * 50)
    logging.info("Step 3: Collecting figures...")

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for src_dir, rel_name in FIGURE_DIRS:
        if src_dir.exists():
            target = output_dir / rel_name
            copy_dir_safe(src_dir, target, extensions=['.png', '.pdf', '.svg', '.eps'])

    # Copy any visualizations from outputs
    for exp_name in KEY_EXPERIMENTS:
        exp_dir = outputs_root / exp_name
        if not exp_dir.exists():
            continue
        for fig_file in exp_dir.glob("*.png"):
            copy_file_safe(fig_file, figures_dir / "experiments" / exp_name / fig_file.name)
        for fig_file in exp_dir.glob("*.pdf"):
            copy_file_safe(fig_file, figures_dir / "experiments" / exp_name / fig_file.name)

    # ==================== 4. Tables ====================
    logging.info("=" * 50)
    logging.info("Step 4: Collecting tables...")

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for src_path, rel_path in TABLE_FILES:
        if src_path.exists():
            copy_file_safe(src_path, output_dir / rel_path)

    # ==================== 5. Documentation ====================
    logging.info("=" * 50)
    logging.info("Step 5: Collecting documentation...")

    docs_dir = output_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    for src_path, rel_path in DOC_FILES:
        if src_path.exists():
            copy_file_safe(src_path, output_dir / rel_path)

    # ==================== 6. Reproducibility Manifest ====================
    logging.info("=" * 50)
    logging.info("Step 6: Generating reproducibility manifest...")

    manifest = {
        "project": "AMSNetV2 - Fall Detection on SisFall Dataset",
        "created_at": datetime.now().isoformat(),
        "git": get_git_info(),
        "environment": get_python_env(),
        "dataset": {
            "name": "SisFall",
            "url": "http://sistemic.udea.edu.co/en/research/projects/english-falls/",
            "subjects": 23,
            "sampling_rate_hz": 50,
            "window_size": 512,
            "stride": 256,
            "evaluation": "LOSO (Leave-One-Subject-Out, 12 folds)",
        },
        "training_command": (
            "python DMC_Net_experiments.py --dataset sisfall --data-root ./data "
            "--model amsv2 --eval-mode loso --seeds 42 123 --epochs 100 "
            "--batch-size 32 --lr 0.001 --amp --weighted-loss --use-tfcl "
            "--out-dir ./outputs/stage1_amsv2_final"
        ),
        "seeds": [42, 123],
        "key_results": all_metrics,
    }

    with open(output_dir / "REPRODUCIBILITY_MANIFEST.json", 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logging.info("Generated: REPRODUCIBILITY_MANIFEST.json")

    # ==================== 7. Submission Checklist ====================
    logging.info("=" * 50)
    logging.info("Step 7: Generating submission checklist...")

    checklist = """# SCI 投稿材料检查清单

## 📁 打包内容

### 代码 (code/)
- [x] 核心模型代码 (models/ams_net_v2.py)
- [x] 模块实现 (DKS, FAA, MSPA, TFCL)
- [x] 训练脚本 (DMC_Net_experiments.py)
- [x] 依赖清单 (requirements.txt)

### 实验结果 (results/)
- [x] AMSNetV2 完整结果
- [x] 基线模型对比结果 (LSTM, ResNet, TCN, Transformer, InceptionTime)
- [x] 消融实验结果
- [x] 汇总对比表 (LaTeX + CSV)
- [x] 详细指标 JSON

### 图表 (figures/)
- [x] t-SNE 特征可视化
- [x] 注意力热力图
- [ ] Grad-CAM 可视化 (待生成)
- [ ] ROC/PR 曲线 (待生成)
- [ ] 训练曲线 (待生成)

### 表格 (tables/)
- [x] 主结果对比表
- [x] 安全性指标表

### 文档 (docs/)
- [x] 模型说明文档
- [x] 期刊选择指南
- [x] 投稿材料草稿

## ✅ 投稿前检查

### 论文内容
- [ ] 摘要 (150-200词)
- [ ] 引言 (背景、动机、贡献)
- [ ] 方法 (模型架构、损失函数、训练策略)
- [ ] 实验 (数据集、基线、消融、统计检验)
- [ ] 讨论 (局限性、未来工作)
- [ ] 结论

### 统计严谨性
- [x] LOSO 交叉验证 (12折)
- [x] 多种子实验 (42, 123)
- [x] 95% 置信区间
- [ ] 配对 t 检验 (vs 基线)
- [ ] Bonferroni 校正

### 可复现性
- [x] 随机种子固定
- [x] 完整训练命令
- [x] 环境信息记录
- [x] Git commit hash
- [ ] 代码仓库 DOI (Zenodo)

### 期刊要求
- [ ] 格式符合目标期刊模板
- [ ] 图片分辨率 ≥300 DPI
- [ ] 表格符合期刊规范
- [ ] 参考文献格式正确
- [ ] 作者信息完整

## 📊 核心结果摘要

| 模型 | Accuracy | Macro F1 | Sensitivity | Specificity |
|------|----------|----------|-------------|-------------|
| AMSNetV2 (Ours) | 98.04% | 97.96% | 97.67% | 98.30% |
| InceptionTime | 97.91% | 97.85% | 97.82% | 97.97% |
| TCN | 97.13% | 97.04% | 96.43% | 97.63% |
| Transformer | 95.48% | 95.34% | 94.71% | 96.02% |
| ResNet | 95.13% | 94.98% | 94.41% | 95.64% |
| LSTM | 95.02% | 94.86% | 94.35% | 95.50% |

## 🎯 目标期刊 (SCI Q4, 版面费 ≤5000元)

1. **IEEE J-BHI** - 生物医学健康信息学
2. **CMPB** - 医学工程与数字健康
3. **BSPC** - 生物医学信号处理
4. **Measurement** - 传感器与测量
5. **EAAI** - AI工程应用

---
生成时间: {timestamp}
Git Commit: {commit}
"""

    git_info = get_git_info()
    checklist = checklist.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        commit=git_info.get('commit', 'N/A')[:8] if git_info.get('commit') else 'N/A'
    )

    with open(output_dir / "SUBMISSION_CHECKLIST.md", 'w', encoding='utf-8') as f:
        f.write(checklist)
    logging.info("Generated: SUBMISSION_CHECKLIST.md")

    # ==================== 8. Create README ====================
    readme = f"""# AMSNetV2 - SCI Submission Package

This package contains all materials for submitting the AMSNetV2 fall detection paper to SCI Q4 journals.

## Package Structure

```
submission_package/
├── code/                      # Source code
│   ├── models/               # Model architecture
│   ├── losses/               # Loss functions
│   └── DMC_Net_experiments.py # Training script
├── results/                   # Experimental results
│   ├── stage1_amsv2_final/   # Main model results
│   ├── stage1_*_final/       # Baseline results
│   ├── ablation_*/           # Ablation studies
│   └── comparison_table.*    # Summary tables
├── figures/                   # Visualizations
├── tables/                    # LaTeX tables
├── docs/                      # Documentation
├── REPRODUCIBILITY_MANIFEST.json
├── SUBMISSION_CHECKLIST.md
└── README.md
```

## Key Results

- **Dataset**: SisFall (23 subjects, LOSO validation)
- **AMSNetV2 Performance**:
  - Accuracy: 98.04%
  - Macro F1: 97.96%
  - Sensitivity: 97.67%
  - Specificity: 98.30%
  - Parameters: 1.65M

## Reproducibility

```bash
# Install dependencies
pip install -r code/requirements.txt

# Train AMSNetV2
python code/DMC_Net_experiments.py --dataset sisfall --data-root ./data \\
    --model amsv2 --eval-mode loso --seeds 42 123 --epochs 100 \\
    --batch-size 32 --lr 0.001 --amp --weighted-loss --use-tfcl
```

## Contact

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    with open(output_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme)
    logging.info("Generated: README.md")

    # ==================== Summary ====================
    logging.info("=" * 50)
    logging.info("Packaging complete!")
    logging.info(f"Output directory: {output_dir}")

    # Calculate package size
    total_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
    logging.info(f"Total package size: {total_size / 1024 / 1024:.2f} MB")

    # List contents
    logging.info("\nPackage contents:")
    for item in sorted(output_dir.iterdir()):
        if item.is_dir():
            count = sum(1 for _ in item.rglob('*') if _.is_file())
            logging.info(f"  📁 {item.name}/ ({count} files)")
        else:
            logging.info(f"  📄 {item.name}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Pack SCI submission materials')
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./submission_package',
        help='Output directory for the submission package'
    )
    parser.add_argument(
        '--include-checkpoints',
        action='store_true',
        help='Include model checkpoint files (increases package size significantly)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    pack_submission(output_dir, args.include_checkpoints)

    logging.info("\n" + "=" * 50)
    logging.info("Next steps:")
    logging.info("1. Review SUBMISSION_CHECKLIST.md for missing items")
    logging.info("2. Generate any missing figures (Grad-CAM, ROC/PR curves)")
    logging.info("3. Complete the manuscript draft")
    logging.info("4. Choose target journal from docs/JOURNAL_TARGETS.md")
    logging.info("=" * 50)


if __name__ == '__main__':
    main()
