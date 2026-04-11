# ST-SRI 实验优化计划（基于审稿意见 + 改进建议）

> 最后更新：2026-04-11
> 状态：IJCAI 审稿后修订 / TNNLS 投稿准备

---

## 一、审计总结

基于三位 IJCAI 审稿人（PC#1/PC#2/PC#3）的意见与"改进建议.pdf"交叉对比，
将仍需改进的实验工作按优先级排列如下。

### 已完成项（无需再改）

| 项目 | 证据 |
|------|------|
| 数据划分防泄漏 | `common.py` 中 `blocked_time_split()` 已全面替代 `random_split`，含 gap + assert |
| 环境锁定文件 | `requirements.txt` + `environment.yml` 已存在且包含 `shap==0.51.0` |
| E3 峰值与受试者名单统一 | `good_subjects.json`（39人）与 `subject_peaks_e3.json`（39人）完全一致 |
| E12 噪声鲁棒性 | 已跑完全部 39 subjects |
| 多 Baseline 分布对比框架 | `e13_baseline_comparison.py` 已有 4 种 baseline，初步结果存在 |
| 手势类型 EMD 差异分析框架 | `e16_gesture_emd.py` 已有初步结果 |
| AOPC 框架 | `e15_aopc.py` 已实现 |
| IG / DeepLIFT 框架 | `e14_xai_baselines.py` 已用 captum 实现 |

---

## 二、待完成任务清单

### P0 — 必须完成（审稿人核心攻击点）

#### P0-1. 补充 TimeSHAP + TSR 时序 XAI 基线

**审稿依据**：
- PC#1: "no head-to-head with TimeSHAP (KDD'21) or TSR (2010.13924)"
- PC#3: "No comparison against TimeShap... its absence is a notable gap"
- 改进建议: "还需对比 TimeSHAP、WindowSHAP"

**现状**：`e14_xai_baselines.py` 已有 ST-SRI vs SHAP vs IG vs DeepLIFT，
但代码中**零引用** TimeSHAP / TSR / WindowSHAP。

**行动项**：
1. 安装 `timeshap` 包（`pip install timeshap`），实现 TimeSHAP 在 LSTM 上的归因
2. 实现 TSR（Temporal Saliency Rescaling）：
   - 先用 IG/DeepLIFT 获得 element-wise saliency
   - 按 TSR 论文做两步 rescaling：时间维聚合 → 特征维 rescaling
3. 统一对比指标：Stability（跨样本一致性）、Peak-in-EMD ratio、Fragmentation、AOPC
4. 扩大样本量：从当前 5 subjects → 全部 39 subjects
5. 输出：`results/xai_baselines/` 下生成对比表 + 对比图

**文件**：修改 `experiments/advanced/e14_xai_baselines.py`

---

#### P0-2. 补充 TCN / Transformer 架构验证

**审稿依据**：
- PC#3: "Only LSTMs are evaluated... unclear whether generalizes to TCN, Transformer"
- 改进建议: "需证明通用性，从'工具改进'到'范式提出'"

**现状**：`common.py` 仅有 `LSTMModel`，全项目无 TCN/Transformer 代码。

**行动项**：
1. 在 `common.py` 中新增：
   - `TCNModel`：基于 `torch.nn.Conv1d` 的因果卷积网络，kernel_size=7，3 层
   - `TransformerModel`：基于 `torch.nn.TransformerEncoder`，4 层，8 heads
2. 用 `e0_train.py` 的框架为每个架构训练 per-subject 模型
3. 对 3 种架构分别运行 ST-SRI 解释，对比：
   - 协同峰位置是否一致（核心：证明 ST-SRI 发现的 EMD 是数据特性而非模型特性）
   - Faithfulness Ratio (Rf) 在不同架构下是否稳定
4. 输出：`results/multi_arch/` 下生成跨架构对比表

**文件**：修改 `common.py`，新增 `experiments/advanced/e17_multi_arch.py`

---

#### P0-3. E11 LOSO 完整重跑

**审稿依据**：审稿人均要求完整的跨受试者泛化证据。

**现状**：
- `results/loso/loso_results.json` 仅含 3 个 subjects（S1/S2/S3）
- README 声称 "63.5%±1.5%" 与实际 JSON（mean=54.0%）不符
- 完整重跑已在后台启动（2026-04-11）

**行动项**：
1. 等待当前后台任务完成
2. 验证结果 JSON 完整性（应有 39 个 fold）
3. 更新 README 中的数值

**文件**：`experiments/advanced/e11.py`，`results/loso/`

---

### P1 — 应该完成（审稿人明确提出）

#### P1-1. τ_max 消融扫描

**审稿依据**：
- PC#3: "τ_max=150ms implicitly tuned to find peaks in 30-100ms;
  a more agnostic τ_max sweep would strengthen trust"

**现状**：`scan_fast()` 中 `max_lag_ms=150` 是硬编码默认值，无扫描实验。

**行动项**：
1. 新建实验：对 τ_max = {100, 150, 200, 300, 500} ms 分别运行 E3 协同谱扫描
2. 统计每组的协同峰位置分布，验证峰值位置不随 τ_max 改变
3. 记录计算时间随 τ_max 的增长趋势
4. 输出：τ_max 消融图 + 峰值稳定性统计表

**文件**：新增 `experiments/advanced/e18_tau_max_ablation.py`

---

#### P1-2. 修复 AOPC 异常值 + 失败案例分层分析

**审稿依据**：
- PC#3: "7 subjects (18%) have faithfulness ratio <1.0;
  S1 (81.25% accuracy) is not explained"
- PC#1: "reported p-values in text vs figure differ"

**现状**：
- `e15_aopc.py` 结果中 S1 的 SHAP AOPC = 0.0（完全无效），疑似 bug
- 无按模型精度分层的失败案例分析

**行动项**：
1. 排查 SHAP AOPC = 0 的根因（可能是 SHAP 归因集中在少数时间步，AOPC 扰动未覆盖）
2. 扩大 AOPC 评估样本量（当前 NUM_SAMPLES=15，建议增至 50）
3. 新增分层分析：
   - 按模型准确率分组（高>85% / 中70-85% / 低<70%）
   - 对 Rf < 1.0 的 subjects 逐个分析原因
4. 统一论文中 p 值口径（使用 Bonferroni 校正后数值）

**文件**：修改 `experiments/advanced/e15_aopc.py`，新增分析脚本

---

#### P1-3. Shapley 交互指数理论严谨性补充

**审稿依据**：
- PC#1: "not the full Shapley-Taylor interaction averaged over all coalitions;
  theoretical link is therefore incomplete"
- 改进建议: "从 PID 角度论证；讨论二阶 vs 高阶交互"

**现状**：纯论文写作问题，但可通过实验辅助。

**行动项**：
1. 论文 Methodology 节补充：
   - 明确说明采用的是 Shapley Interaction Index 的简化形式（固定 context），
     而非完整的 Shapley-Taylor interaction
   - 从 PID（Partial Information Decomposition）角度论证
     Synergy/Redundancy 的信息论意义
   - 讨论为何二阶交互已足够捕捉 sEMG 的主要生理特性
2. 实验辅助（可选）：新增 context sampling 消融
   - 在不同随机 context subsets 下计算交互指数，验证结果稳定性

**文件**：论文正文修改 + 可选新增实验

---

### P2 — 建议完成（提升论文质量）

#### P2-1. E13 Baseline 分布对比扩大样本量

**现状**：已跑 S1-S9（9 subjects），4 种 baseline 结果基本一致。

**行动项**：
- 扩大到全部 39 subjects
- 在论文 Appendix 中报告结果

**文件**：`experiments/advanced/e13_baseline_comparison.py`

---

#### P2-2. E14 XAI 基线对比扩大样本量

**现状**：已跑 5 subjects。

**行动项**：
- 配合 P0-1（补 TimeSHAP/TSR 后）一并扩大到全部 39 subjects

**文件**：`experiments/advanced/e14_xai_baselines.py`

---

#### P2-3. 更新 README 数值一致性

**现状**：README 中部分数值与实际 JSON 结果不一致（如 E11）。

**行动项**：
- 待所有实验重跑完成后统一更新
- 确保每个数值有 JSON 文件可追溯

**文件**：`README.md`

---

### P3 — 长期目标（改进建议提出，审稿人未要求）

#### P3-1. 多数据集验证

**改进建议**："增加 CapgMyo 或 CSL-HDEMG 数据集"

**行动项**：
- 下载 CapgMyo（高密度 sEMG）数据集
- 适配预处理流程
- 运行 E3 + E5 验证 ST-SRI 在不同数据集上的表现

**时间**：投 TNNLS 前完成

---

#### P3-2. 因果性实验

**PC#1**："causal interpretation of positive synergy is suggestive but not identified;
synergy remains correlational"

**行动项**：
- 设计因果干预实验：在特定 lag 注入人工延迟/取消延迟，
  观察 ST-SRI 协同峰是否相应移动
- 与 E2（合成信号验证）类似但更严格

---

## 三、执行优先级与时间估算

```
Week 1:  P0-3(E11重跑等待) + P0-2(TCN/Transformer实现+训练)
Week 2:  P0-1(TimeSHAP+TSR实现+全量运行)
Week 3:  P1-1(τ_max消融) + P1-2(AOPC修复+分层分析)
Week 4:  P2-*(扩大样本量) + P1-3(论文理论补充)
Week 5:  论文修订 + 数值统一 + 提交
```

| 阶段 | 任务 | 预计工作量 | 产出 |
|------|------|-----------|------|
| Week 1 | TCN + Transformer 模型实现与训练 | 3 天 | `common.py` 新模型 + 40×2 checkpoints |
| Week 1 | E11 重跑验证 | 等待 | `results/loso/loso_results.json` |
| Week 2 | TimeSHAP + TSR 集成 | 3 天 | `e14_xai_baselines.py` 扩展版 |
| Week 2 | 全量 XAI 基线对比运行 | 2 天 | 39 subjects × 5 methods |
| Week 3 | τ_max 消融实验 | 1 天 | `e18_tau_max_ablation.py` |
| Week 3 | AOPC 修复 + 分层分析 | 2 天 | `e15_aopc.py` 修订版 |
| Week 4 | 扩大 E13/E14 样本量 | 1 天 | 全量结果 |
| Week 4 | 论文 Methodology 理论补充 | 2 天 | PID 论证 + Shapley 简化说明 |
| Week 5 | 数值统一 + README 更新 + 提交 | 2 天 | 最终版论文 |

---

## 四、审稿人关键引用（便于 Rebuttal）

### PC#1 核心要求
> "No head-to-head with strong time-series explainers like TimeSHAP (KDD'21) or
> Temporal Saliency Rescaling (TSR, 2010.13924)"

→ 对应 **P0-1**

> "The Shapley interaction index used is a single four-term inclusion-exclusion quantity...
> the theoretical link to Shapley interactions is therefore incomplete"

→ 对应 **P1-3**

### PC#2 核心要求
> "The paper seems to be rather tailored for the specific application area"

→ 对应 **P0-2**（多架构证明通用性）+ **P3-1**（多数据集）

### PC#3 核心要求
> "Only LSTMs are evaluated... unclear whether the synergy decomposition
> generalizes to other sequence models (TCN, Transformer)"

→ 对应 **P0-2**

> "τ_max=150ms... the paper does not fully rule out that the method is designed
> to find peaks in a range it was already tuned to look at"

→ 对应 **P1-1**

> "The 1.62× accuracy drop ratio is numerically modest...
> For 7 subjects (18%), the faithfulness ratio is below 1.0"

→ 对应 **P1-2**

---

## 五、文件变更清单（预期）

| 操作 | 文件 | 说明 |
|------|------|------|
| 修改 | `common.py` | 新增 TCNModel + TransformerModel |
| 修改 | `experiments/advanced/e14_xai_baselines.py` | 集成 TimeSHAP + TSR |
| 修改 | `experiments/advanced/e15_aopc.py` | 修复异常值 + 分层分析 |
| 新增 | `experiments/advanced/e17_multi_arch.py` | 跨架构 ST-SRI 验证 |
| 新增 | `experiments/advanced/e18_tau_max_ablation.py` | τ_max 消融扫描 |
| 更新 | `requirements.txt` | 添加 `timeshap` 依赖 |
| 更新 | `environment.yml` | 添加 `timeshap` 依赖 |
| 更新 | `README.md` | 数值一致性修正 |
| 更新 | `results/` | 各实验全量结果 |
