# 模型可解释性分析 + 类别不均衡处理对比：执行方案

本文档固化两项分析的执行计划与产出规范，待确认后按此方案在仓库内实现脚本、运行实验并生成图表与消融结果表。

---

## 1. 模型可解释性分析（“How the Model Detects Falls”）

### 1.1 可视化方法选择（Attention 优先，Grad-CAM 兜底）
1. 检查训练入口 `code/DMC_Net_experiments.py` 所用模型（通常位于 `code/models/`），确认是否包含：
   - 可直接导出的 Attention 结构（例如 self-attention/attention pooling，且能拿到 attention map），或
   - 主要为卷积/时序卷积模块（更适合 Grad-CAM）。
2. 选择主方法：
   - **若可获取 Attention 权重：**以 Attention 可视化为主（模型“原生解释”）。
   - **否则：**使用 **Grad-CAM** 对模型最后一个关键卷积块进行解释。

### 1.2 Grad-CAM 生成方法（适用于 1D/2D 输入）
**解释目标：**对每个样本解释“fall”类别的 logit/概率来源（聚焦于 fall 预测）。

**单样本计算流程：**
1. 将模型置为 `eval()`，开启梯度（允许反向传播到目标层激活）。
2. 通过 forward hook 捕获目标层激活 `A`（建议为最后一个语义最强的 conv block 输出）。
3. 对 “fall” logit 做反向传播，获取梯度 `∂Y_fall/∂A`。
4. 对梯度在空间/时间维做全局平均池化得到通道权重 `w_k`：
   - 1D：沿时间维池化；
   - 2D（如 time×freq）：沿 (time,freq) 池化。
5. 计算 CAM：`CAM = ReLU(Σ_k w_k * A_k)`，并归一化至 `[0,1]`。
6. 将 CAM 上采样到输入分辨率（时间或 time×freq），用于叠加显示。

### 1.3 Attention 权重可视化方法（若模型支持）
1. 为模型增加或复用 “返回 attention” 的 forward 路径（例如 `return_attention=True`）。
2. 对多头/多层进行聚合以便展示（例如：头平均；展示最后一层；或层间平均）。
3. 输出形式：
   - `time→time` 的 attention heatmap；或
   - 每个时间步的重要性曲线（attention 归约后得到 per-timestep score）。

### 1.4 图像呈现规范（最终产出）
**图名（标题）：**`How the Model Detects Falls`

**推荐布局（单图多子图）：**
- 3–6 个代表性样本（默认：真阳性 fall；可选：加入 1 个 FP/FN 用于诊断）
- 每个样本包含：
  - 原始输入信号（按通道/轴绘制）；
  - CAM/Attention 作为热力图或透明度叠加到时间轴上；
  - 标注 `GT` 与 `Pred`（含概率/logit）用于解释一致性。

**输出文件：**
- `figures/how_model_detects_falls.png`
- `figures/how_model_detects_falls.pdf`
-（可选）`figures/explainability_samples/` 下按样本导出单独图片用于附录

### 1.5 代码交付物（实现后）
- 新脚本：`code/scripts/explainability_visualize.py`
  - 关键参数建议：
    - `--ckpt`（模型权重路径）
    - `--dataset` / `--data-root`
    - `--split {train,val,test}`（默认 `test`）
    - `--num-samples`（默认 6）
    - `--method {gradcam,attention}`（默认自动选择：attention 可用则 attention，否则 gradcam）
  - 产出：上述 `figures/` 下的最终图文件
- 若需要最小化改动模型：仅为 attention 导出增加返回值或缓存，不改动默认训练行为

---

## 2. 类别不均衡处理对比（Class Weighting Ablation）

### 2.1 对比策略集合（仅改变“权重策略”，其余保持一致）
建议比较下列权重策略（均在 loss 层面处理）：
1. `none`：不使用 class weight（基线）
2. `auto`：当前仓库已实现的自动权重策略（对照组/主方法）
3. `inv_freq`：`w_c ∝ 1 / count_c`
4. `sqrt_inv_freq`：`w_c ∝ 1 / sqrt(count_c)`（更温和）
5. `effective_num`：基于“有效样本数”的权重（Cui et al.），常用 `beta≈0.999`

> 可选扩展（如需）：`balanced_softmax` / `logit_adjustment`。默认先不引入以保持消融聚焦。

### 2.2 公平对比实验协议（避免泄漏，确保可复现）
1. **数据与划分**：以主数据集为准（建议 `sisfall` + LOSO；或你指定的基准设置）。
2. **固定不变项**（所有策略一致）：
   - 模型结构、数据增强、优化器、学习率策略、epoch、batch size、AMP、随机种子机制等。
3. **唯一变量**：`--class-weighting`（权重策略）。
4. **权重计算原则**：每折仅用该折的 **训练集** 统计 `count_c` 计算权重，避免验证/测试信息泄漏。
5. **重复次数**：建议至少 3 个种子（`--seed`），用于报告均值±标准差。

### 2.3 指标与记录（用于表格）
每个 run（或每折/每种子）记录：
- Accuracy
- Macro-F1（强烈建议：衡量不均衡）
- Fall Recall / Sensitivity（关键安全指标）
-（可选）Precision、Specificity
-（可选）Confusion Matrix（作为附录或补充材料）

同时将“实际使用的 class weights”写入配置/日志，便于审计与复现实验。

### 2.4 实现与运行组织（自动化消融）
1. 增加/复用训练参数：
   - `--class-weighting {auto,none,inv_freq,sqrt_inv_freq,effective_num}`
2. 增加一个消融驱动脚本（或在现有 `code/scripts/` 中扩展）：
   - 对策略 × 种子（× 折）组合批量运行；
   - 统一输出到 `outputs/ablation/class_weighting/`（建议分目录存放）。
3. 增加汇总脚本：
   - 读取每个 run 的 metrics（JSON/CSV）；
   - 输出聚合结果（均值±标准差），并保存为可用于画图的 CSV。

### 2.5 表格产出规范（最终产出）
**表名：**“Class Weighting Ablation (Auto vs Alternatives)”

**表结构：**
- 行：权重策略（`none`, `auto`, `inv_freq`, `sqrt_inv_freq`, `effective_num`）
- 列：Accuracy / Macro-F1 / Fall Recall（可选再加 Precision、Specificity）
- 单元格：`mean ± std`（跨 seeds；若跨 folds 再做二级聚合或先 fold 内聚合）

**输出文件：**
- `docs/class_weight_ablation.md`（论文/报告可直接引用）
- `outputs/ablation/class_weight_ablation.csv`（便于二次分析与绘图）
-（可选）`figures/class_weight_ablation_bar.png`（辅助可视化）

---

## 3. 最终交付清单（与任务要求对齐）
- Figure：`figures/how_model_detects_falls.png`（How the Model Detects Falls）
- Table：`docs/class_weight_ablation.md`（证明当前 `auto` 权重策略优于其他策略的消融对比）
- 可复现实验脚本（解释性可视化 + 权重策略消融 + 汇总）

---

## 4. 待确认项（执行前需要你确认）
1. “消融表”以哪个数据集/划分为主：`sisfall`（LOSO）还是其他？
2. 解释性图中样本选择：只展示真阳性 fall，还是加入 FP/FN 诊断样本？
3. 解释性图使用的 checkpoint：自动选择 `outputs/` 下 best，还是你指定路径？

