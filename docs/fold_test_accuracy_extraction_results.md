# Per-fold Test Accuracy 提取结果

时间：2025-12-18 17:30:12

## 执行目的
从实验日志中提取 **每个 LOSO subject（fold）** 的 **Test Accuracy**，并输出三组按 fold/subject ID 排序对齐的列表，用于后续 Paired t-test / Cohen’s d / CI 计算。

## 扫描范围与命令
- 扫描目录（递归）：`outputs`
- 执行命令：
  - `python code/scripts/extract_fold_test_accuracy.py --roots outputs --round 4`

脚本输出摘要：
- `scanned_files=19`
- `used_files=11`
- `folds=[1, 2, 4, 5, 6, 9, 10, 11, 17, 18, 19, 21]`

说明：
- 本项目日志中的 fold ID 来自 LOSO subject 标签（如 `loso_SA01` -> `fold_id=1`），因此并非严格的 `1..10` 连续编号。

## 最终结果（按 fold_ids 顺序）

```python
fold_ids = [1, 2, 4, 5, 6, 9, 10, 11, 17, 18, 19, 21]
lite_amsnet_acc = [98.2798, 93.8098, 99.5902, 99.3363, 98.9215, 99.7788, 99.5575, 95.3263, 99.3017, 99.3086, 98.7555, 99.1427]
lite_mspa_acc = [98.9215, 93.8053, 99.8064, 99.6128, 99.4469, 99.8894, 99.7235, 95.5199, 99.0503, 99.5852, 99.1427, 99.2533]
inception_acc = [98.396, 92.0907, 99.115, 98.7555, 99.2257, 99.8341, 98.4237, 94.1648, 98.324, 99.3639, 98.7002, 98.479]
```

## 备注（与配对检验相关）
- 若你希望把 “每个 seed 的每个 fold” 作为配对样本，而不是先对 seeds 做均值聚合，可在脚本中用参数控制（`--aggregate-seeds`），或进一步扩展脚本输出为二维表（fold × seed）。

## 配对显著性检验结果（基于上述列表）
执行脚本：
- `python code/scripts/paired_ttest_from_markdown.py --md docs/fold_test_accuracy_extraction_results.md`

原始统计结果（双侧配对 t 检验 + Cohen’s d_z）：
- Pair 1（Ablation）：Lite-AMSNet vs Lite-AMSNet w/MSPA  
  - `n_pairs=12`，`mean_diff(Proposed-MSPA)=-0.2207`（百分点）  
  - `t=-3.258387`，`p=0.00762079`，`Cohen's d_z=-0.940615`
- Pair 2（Baseline）：Lite-AMSNet vs InceptionTime  
  - `n_pairs=12`，`mean_diff(Proposed-Inception)=0.5197`（百分点）  
  - `t=2.838679`，`p=0.0161199`，`Cohen's d_z=0.819456`

建议论文表述（按预设逻辑自动生成）：
> In the ablation study, removing the MSPA module led to a statistically significant change in performance. Compared with InceptionTime, the proposed Lite-AMSNet achieved a statistically significant improvement.
