# SCI Q4 冲刺计划（3–4 天）— 基于 Gemini/PubMate 方案系统化重构

> 面向“可投稿”的最小闭环：补齐 **N=2**、澄清 **Latency**、完成 **叙述转向（Lite-AMSNet）**，再用跨数据集/可视化/鲁棒性做加分项。

## 0. 一页结论（TL;DR）

**Phase 1（必须完成）**
- 补齐种子：把主结果从 `n=2` 补到 `n=5`（新增 `456/789/1024`），并输出 `mean±std` + `95% CI`
- 解决延迟矛盾：用“可复现”的脚本测 **推理时延（Inference time）**，明确是 CPU/GPU、Batch=1、同步与预热
- 叙述转向：把“最优配置关闭 MSPA”包装为 **Lite-AMSNet（减法换效率）**，用 “With vs Without MSPA” 表支撑

**Phase 2（推荐完成）**
- 跨数据集：UniMiB（若实现困难，启用后备策略）
- 可视化：混淆矩阵（聚合 5 seeds）、t-SNE、注意力/关键片段可解释图
- 鲁棒性：噪声/缺失/时间偏移（代码已有时优先复用现成入口）

---

## 0.1 进度看板（建议每天更新一次）

**Phase 1（必须完成）**
- [ ] P1-Seed：Lite-AMSNet 补齐 `456/789/1024`（完成后主结果 `n=5`）
- [ ] P1-Latency：按口径测推理时延，并填完表格（至少 Lite；对比需 Full）
- [ ] P1-Pivot：完成 “With vs Without MSPA” 支撑表（精度 + 效率 + 延迟）
- [ ] P1-Stats：生成 `mean±std` + `95% CI`（主结果以 Lite 为准）

**Phase 2（推荐完成）**
- [ ] P2-Cross：跨数据集（UniMiB 或后备方案）
- [ ] P2-Vis：混淆矩阵聚合 + t-SNE/注意力图
- [ ] P2-Robust：鲁棒性曲线（噪声/缺失/偏移）

---

## 1. 问题定义（Gemini 方案中的“致命漏洞”）

### 1.1 N=2 统计陷阱（必须修）
- 现状：只有 `seed 42/123`
- 风险：无法可靠报告均值/方差/置信区间，审稿人会质疑“挑结果”
- 目标：至少 `n=5`（新增 `456/789/1024`），报告 `mean±std` 与 `95% CI`

### 1.2 Latency 数据矛盾（必须澄清）
- 现状：存在“0ms（测量问题）”与“~126ms（CPU？Batch？）”的叙述冲突
- 目标：将延迟拆分并分别给出：
  - **推理时延**（runtime，forward pass）
  - **检测延迟**（detection delay，窗口/stride 导致的检测滞后，单位 ms）

### 1.3 叙述逻辑脱节（必须转向）
- 现状：论文故事强调“频域/时频模块”，但最佳结果来自 `--ablation mspa:False`
- 目标：把“移除 MSPA”变成贡献：**Lite-AMSNet（更少计算、更低延迟、精度不降/略升）**

---

## 2. 约束与统一约定

- 工作目录：仓库根目录（`D:\SCI666`）
- 数据：`./data`（只读，不纳入提交）
- 输出：统一写入 `./outputs/<exp_name>`；图表写入 `./figures/<paper_or_sprint>`  
- 命名（建议固定，减少写作混乱）：
  - **Full**：AMSNetV2（MSPA=On）
  - **Lite**：Lite-AMSNet（即 AMSNetV2, `mspa:False`）

---

## 3. Phase 1（Must-Haves / 严格必做）

### 3.1 修复 N=2：补齐 5 seeds（主结果以 Lite-AMSNet 为准）

> Gemini 建议：主结果的补种子要和“最优配置”一致（即 `mspa:False`）。

- [ ] **Lite-AMSNet（主结果）**：新增 seeds `456/789/1024`（LOSO）
```powershell
python code/DMC_Net_experiments.py `
  --dataset sisfall `
  --data-root ./data `
  --model amsv2 `
  --eval-mode loso `
  --seeds 456 789 1024 `
  --epochs 100 `
  --batch-size 128 `
  --lr 1e-3 `
  --weighted-loss `
  --amp `
  --use-tfcl `
  --ablation mspa:False `
  --out-dir ./outputs/ablation_no_mspa
```

- [ ] **Full（对照）**：仅在需要“paired 对比”时补齐同样 seeds  
  - 目的：支撑“Lite 更高效/更快且精度不降”的主张
  - 若时间不足：至少保证 `seed 42/123` 的 Full 与 Lite 都存在，用于趋势表 + 讨论
```powershell
python code/DMC_Net_experiments.py `
  --dataset sisfall `
  --data-root ./data `
  --model amsv2 `
  --eval-mode loso `
  --seeds 456 789 1024 `
  --epochs 100 `
  --batch-size 128 `
  --lr 1e-3 `
  --weighted-loss `
  --amp `
  --use-tfcl `
  --out-dir ./outputs/stage1_amsv2_final
```

**验收标准（硬门槛）**
- [ ] `./outputs/ablation_no_mspa/summary_results.json` 显示 `n=5`
- [ ] （可选）`./outputs/stage1_amsv2_final/summary_results.json` 显示 `n=5`（用于 Full vs Lite 的 paired 对比）
- [ ] 论文表格可报告 `mean±std` 与 `95% CI`（CI 宽度建议 < 0.02 作为目标）

**快速自检（可直接复制执行）**
```powershell
python -c "import json; print('Lite n=', json.load(open('outputs/ablation_no_mspa/summary_results.json','r',encoding='utf-8'))['macro_f1_mean_n'])"
python -c "import json; print('Full n=', json.load(open('outputs/stage1_amsv2_final/summary_results.json','r',encoding='utf-8'))['macro_f1_mean_n'])"
```

---

### 3.2 解决 Latency Mystery：建立可复现的推理时延测量

**必须明确并在文稿中固定口径**
- [ ] 设备：CPU or GPU（可两套都给）
- [ ] Batch：`1`（必须），并说明样本长度（如 512）
- [ ] 预热：≥100 iterations
- [ ] 计时：≥1000 iterations；GPU 需 `torch.cuda.synchronize()`
- [ ] 报告：`mean±std`（ms/sample）

**Latency 口径表（填这里，避免口径漂移）**

| Model | Device | Batch | Warmup | Runs | mean (ms) | std (ms) | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| Lite-AMSNet (mspa:False) |  | 1 |  |  |  |  |  |
| Full (MSPA On) |  | 1 |  |  |  |  |  |

**推荐直接复用现成脚本（Gemini 目录）**
- [ ] 运行 `gemini/Latency_Measurement_Script.py`，记录 CPU/GPU（若脚本支持）  
```powershell
python gemini/Latency_Measurement_Script.py
```

**后备方案（若脚本参数需适配）**
- [ ] 用 `code/DMC_Net_experiments.py --profile` 输出 params/FLOPs/latency（并在文稿说明测量方式与 batch）
```powershell
python code/DMC_Net_experiments.py --dataset dryrun --epochs 1 --batch-size 4 --profile
```

**验收标准（硬门槛）**
- [ ] 上表填完整（至少 Lite 的一套结果；若要主张“Lite 更快”，需同时填 Full）
- [ ] 文稿中不再出现“0ms”这类无法解释的延迟结论

---

### 3.3 叙述转向（Narrative Pivot）：把 Lite-AMSNet 作为主贡献

**写作主线（建议固定）**
- [ ] 旧叙事：复杂时频融合（容易被“关闭 MSPA 最优”击穿）
- [ ] 新叙事：在低维加速度数据上，**频谱注意力带来 MAC/内存访问开销**，Lite-AMSNet 用 DKS/时域表征实现更优的效率-精度权衡

**必须产出 1 张“支撑表”**
- [ ] 表 A：With MSPA vs Without MSPA（Full vs Lite），包含：
  - Macro F1（`mean±std`）
  - Params / FLOPs（复用 `--profile`）
  - 推理时延（按 3.2 的口径）

---

### 3.4 统计严谨性（最低配：均值/方差/CI；加分：显著性与效应量）

**最低配（必须）**
- [ ] 5 seeds 的 `mean±std` + `95% CI`（主结果以 Lite 为准）

**加分项（推荐）**
- [ ] paired t-test（同 seeds）+ 多重比较校正（Bonferroni）+ Cohen’s d  
  - 对照建议：Lite vs Full、Lite vs 关键基线（不必补全所有基线 seeds）

> 说明：如果 Full 未补齐到 5 seeds，则显著性检验仅能做在“已对齐 seeds 的子集”，需在文稿如实声明。

---

## 4. Phase 2（Nice-to-Haves / 论文完整性加分）

### 4.1 跨数据集验证（优先：UniMiB；后备：简化方案）

**目标**
- [ ] 给出“SisFall → UniMiB”的迁移评估（哪怕是简单 holdout/zero-shot）

**优先路径（UniMiB）**
- [ ] 实现/复用 `--dataset unimib` 加载逻辑（若已有 CLI 却无分支，补齐分支）
- [ ] 评估命令（建议先 `--epochs 0` 做纯推理，确认 pipeline）：
```powershell
python code/DMC_Net_experiments.py `
  --dataset unimib `
  --data-root ./data `
  --model amsv2 `
  --eval-mode holdout `
  --resume ./outputs/ablation_no_mspa/ckpt_best_seed42_loso_SA01.pth `
  --epochs 0 `
  --out-dir ./outputs/cross_sisfall_to_unimib
```

> 备注：`--resume` 这里用的是示例 ckpt，你也可以替换为 `./outputs/ablation_no_mspa/` 下任意一个 `ckpt_best_seed42_loso_*.pth`。

**后备策略（任选其一，避免卡死在数据加载）**
- [ ] 方案 B：选用更容易接入的数据集（如 MobiFall 的子集）
- [ ] 方案 C：明确把 LOSO（用户无关）作为主要泛化论据，并将跨数据集作为 future work（需在讨论写得“合理且诚实”）

---

### 4.2 可视化（让 Q4 审稿人“看见工作量”）

- [ ] **混淆矩阵**：每个 seed 的 `confusion_matrix_seed*.npy` 已可落盘；做“跨 seed 聚合”并画热力图
- [ ] **t-SNE + 注意力图**：复用 `code/scripts/generate_paper_figures.py`
```powershell
python code/scripts/generate_paper_figures.py --help
```

可选加分：
- [ ] 类别分组柱状图：动态活动 vs 静态活动（如果标签允许分组）
- [ ] “关键片段”可解释：展示跌倒冲击段与模型响应（替代复杂 Grad-CAM 也可）

---

### 4.3 鲁棒性（若代码已有入口，优先直接跑）

- [ ] 优先复用：`code/scripts/eval_noise_robustness.py`（单 ckpt 出图更快）
```powershell
python code/scripts/eval_noise_robustness.py --help
```

或：
- [ ] 使用实验入口：`--run-robustness`（若已在训练脚本中实现）

---

### 4.4 对比表（保证“公平比较”的写作底座）

- [ ] 对照表只比较同一评估协议（建议统一 `--eval-mode loso`），避免把 holdout 与 LOSO 混在一张表里
- [ ] 基线模型可先用现有 `seed 42/123` 做趋势对比；若审稿压力增大，再按时间补齐关键基线 seeds

---

## 5. 交付物清单（投稿所需最小闭环）

**必须交付**
- [ ] `./outputs/ablation_no_mspa/summary_results.json`（n=5）
- [ ] Lite-AMSNet 主表：`mean±std` + `95% CI`
- [ ] Latency 复现记录（脚本/口径/结果）
- [ ] With vs Without MSPA 对照表（效率 + 精度）

**推荐交付**
- [ ] 跨数据集结果：`./outputs/cross_sisfall_to_unimib/holdout_results.json`
- [ ] 图：混淆矩阵（聚合）、t-SNE、注意力可解释图、鲁棒性曲线

---

## 6. 3–4 天执行时间线（按优先级排程）

### Day 1
- [ ] 跑 `--dataset dryrun` 做环境自检
- [ ] 启动 Lite-AMSNet seeds 456/789/1024（优先把训练挂满）
- [ ] 同步做 Latency 口径确认与一次测量（避免最后才发现“126ms”说不清）

### Day 2
- [ ] 收集 Lite 的三组 seed 结果（加上已有 42/123 形成 n=5）
- [ ] 生成主表（mean±std + CI）
- [ ] 若时间允许：补 Full 的 456/789/1024（用于 paired 对比）

### Day 3
- [ ] 叙述转向落地：With vs Without MSPA 表 + 写作要点
- [ ] 可视化：混淆矩阵聚合 + t-SNE/注意力图

### Day 4（缓冲/加分）
- [ ] 跨数据集（或启用后备策略）
- [ ] 鲁棒性曲线/效率 Pareto（可选）
- [ ] 整理输出、生成投稿包（如需）

---

## 7. 快速命令模板（Windows / PowerShell）

```powershell
# 0) 环境自检（建议先跑）
python code/DMC_Net_experiments.py --dataset dryrun --epochs 2 --batch-size 4 --profile

# 1) 主结果：Lite-AMSNet (w/o MSPA) 补 seeds
python code/DMC_Net_experiments.py `
  --dataset sisfall --data-root ./data `
  --model amsv2 --eval-mode loso `
  --seeds 456 789 1024 --epochs 100 --batch-size 128 `
  --weighted-loss --amp --use-tfcl --ablation mspa:False `
  --out-dir ./outputs/ablation_no_mspa

# 2) 推理时延测量（Gemini 脚本）
python gemini/Latency_Measurement_Script.py
```

---

## 8. 风险与后备策略（写在计划里，执行时更稳）

- **跨数据集卡住**：优先保证 Phase 1 闭环；跨数据集启用后备策略（4.1）
- **训练时间爆炸**：先保 Lite 的 n=5；Full 的补齐改为可选
- **Latency 数值不好看**：明确口径（CPU vs GPU / Batch），并用 Params/FLOPs 做补充论证；避免用“4090 延迟”推断 MCU
- **显著性不强**：把重点放在 CI、效应量与效率提升（诚实写作），避免过度宣称
