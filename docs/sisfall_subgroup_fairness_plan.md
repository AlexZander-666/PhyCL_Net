# SisFall 子群体公平性分析执行方案（Age × Gender × Fall Type）

## 1. 目标与范围

对 SisFall 数据集进行子群体（subgroup）分析，按以下维度分组并对比三类指标：

- 年龄：Youth（23 人） vs Elderly（15 人）
- 性别：Male vs Female
- 跌倒类型：15 种 Fall Type

需要为每个子群体计算并比较：

- Accuracy（准确率）
- Sensitivity（灵敏度/召回率/TPR）
- Specificity（特异度/TNR）

并生成：

- 分层（stratified）指标表格
- 指标分布对比的箱线图（box plot）

## 2. 数据输入与准备

### 2.1 预测结果（Predictions）

优先使用已有测试阶段输出；若没有，则对指定 checkpoint 在 SisFall 测试划分上做一次推理并落盘，保证“每条被评估样本”一行记录。

建议最小字段（CSV/Parquet 均可）：

- `subject_id`：受试者 ID（用于 subject-level 聚合）
- `y_true`：真实标签（建议二分类：0=ADL，1=Fall）
- `y_score`：模型输出分数（概率/logit，便于后续阈值一致性说明）
- `y_pred`：基于固定阈值的预测标签（0/1）
- `trial_id` / `event_id`：可选，用于事件级聚合或溯源
- `fall_type`：对跌倒样本标注 1–15；ADL 样本可置为 `ADL`/`None`
- `window_idx`/时间戳：可选，用于窗口级分析与排错

### 2.2 元数据（Metadata）

建立/校验一张以 `subject_id` 为主键的元数据表：

- `age_group` ∈ {`Youth`, `Elderly`}（总计 23 vs 15 人）
- `gender` ∈ {`Male`, `Female`}
- （可选）`age_years`：便于透明披露

> 注意：年龄与性别信息必须来源于 SisFall 官方/随附说明或项目现有映射文件，避免手工误配。

## 3. 分组方案（Grouping）

### 3.1 基础分组键

所有分析都至少按以下键分组：

- `age_group`
- `gender`

### 3.2 跌倒类型维度（Fall Type）

由于 `fall_type` 仅对“跌倒正类”有意义，采用如下策略：

- **Sensitivity by fall_type**：仅在 `y_true=1`（falls）上计算，并按 `age_group × gender × fall_type` 分组。
- **Specificity by fall_type**：负样本（ADL）不具备 `fall_type`。因此对每个 `age_group × gender × fall_type` 子群体，使用同一 `age_group × gender` 下的 ADL 负样本作为“匹配负类池（matched negatives）”来计算 specificity，使不同 fall_type 的 specificity 可比较且负样本来源一致。
- **Accuracy by fall_type**：与 specificity 同理，正类使用该 fall_type 的 falls，负类使用匹配 ADL 池。

### 3.3 公平性关键：聚合粒度

为避免“窗口数/样本数更多的受试者”主导结果，采用：

- **Subject-level 先聚合，再在子群体内汇总**。

做法：先对每个 `subject_id`（在对应子群体过滤后的数据上）计算混淆矩阵计数（TP/FP/TN/FN），再在子群体内合并（例如求和计数后再算指标，或先算每人指标再取均值；两者需选定一种并全程一致，同时建议附带计数与样本数透明披露）。

## 4. 指标计算方法（Accuracy / Sensitivity / Specificity）

### 4.1 混淆矩阵计数

对任一给定分析集合（例如某个子群体过滤后的样本集合）：

- `TP = sum(y_true=1 & y_pred=1)`
- `FN = sum(y_true=1 & y_pred=0)`
- `TN = sum(y_true=0 & y_pred=0)`
- `FP = sum(y_true=0 & y_pred=1)`

### 4.2 指标公式

- **Accuracy** = `(TP + TN) / (TP + TN + FP + FN)`
- **Sensitivity (TPR)** = `TP / (TP + FN)`（仅当 `TP+FN>0`）
- **Specificity (TNR)** = `TN / (TN + FP)`（仅当 `TN+FP>0`）

### 4.3 边界情况处理

若某子群体缺少正类或负类：

- 对应指标置为 `NaN`（或空值），在表格中明确标注原因（例如 “no positives” / “no negatives”）。
- 箱线图中可选择剔除缺失值，并在图注/说明中声明剔除规则。

### 4.4 不确定性估计（建议）

为支撑“公平性差异”的可靠性说明，建议：

- 对每个子群体做 **subject-level bootstrap**（按受试者重采样），得到 Accuracy/Sensitivity/Specificity 的 95% 置信区间（CI）。

## 5. 分层表格（Stratified Table）输出

生成一张分层表，行索引建议为：

- `age_group`, `gender`, `fall_type`
- 并额外提供每个 `age_group × gender` 的 `fall_type=ALL` 汇总行

列建议包含：

- 样本规模：`n_subjects`, `n_samples`（或 `n_events`）
- （可选但推荐）混淆计数：`TP`, `FP`, `TN`, `FN`
- 指标：`accuracy`, `sensitivity`, `specificity`
- （可选）CI：`*_ci_low`, `*_ci_high`

输出形式：

- 机器可读：`CSV`（用于复现实验与绘图）
- 可阅读：`Markdown`（或 `LaTeX`，用于论文/报告）

## 6. 箱线图（Box Plot）可视化

将表格转换为“长表（long-form）”结构：每行对应一个子群体与一个指标：

- `group_type`：例如 `Age` / `Gender` / `Age×Gender` / `FallType`
- `group_label`：例如 `Youth`、`Elderly`、`Male`、`Female`、`Youth-Male`、`F01`…`F15`
- `metric`：`accuracy` / `sensitivity` / `specificity`
- `value`：指标值（可用 subject-level bootstrap 的样本分布，或 subject-level 指标分布）

建议生成 3 张图（保证可读性）：

1. **按年龄组**：Youth vs Elderly 的三指标箱线图（可分面或颜色区分指标）
2. **按性别**：Male vs Female 的三指标箱线图
3. **按跌倒类型**：15 种 fall type 的三指标箱线图（建议按指标分面；如需再分 age/gender，优先使用分面避免过密）

图输出到 `figures/`，并使用固定、可追溯的文件名。

## 7. 公平性分析额外注意事项（Fairness Considerations）

- **统一阈值**：不对不同子群体单独调阈值；阈值应来自全局验证集策略或固定 0.5，并在文档中说明。
- **同一评估划分**：子群体分析必须严格基于最终报告用的测试划分（例如 LOSO folds），避免数据泄漏或分布不一致。
- **权重策略透明**：明确使用 subject-level 聚合（每人同权）还是 sample-level 聚合（按窗口加权），并建议两者都输出以便对比。
- **样本量披露**：对子群体样本过少的结果进行标注（例如 n_subjects 很低时提示不稳定）。
- **多重比较**：如要做显著性结论，需要多重比较校正；否则建议保持描述性对比（差异大小 + CI）。

## 8. 实施前需要确认的问题

1. 任务是 **二分类（Fall vs ADL）** 吗？还是需要扩展到多分类（15 类 fall type + ADL）？
2. 指标计算基于 **窗口级（window-level）** 预测还是 **事件级（event-level，按 trial 聚合）**？两者会显著影响 Accuracy/Specificity。
3. 预测结果是否已存在于 `outputs/`（如有请指定路径），还是需要我从某个 checkpoint 运行推理生成？

