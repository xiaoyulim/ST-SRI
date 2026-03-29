# ST-SRI 研究参考文档

> 课题：面向 EMD 时序错位的可解释早期 sEMG 运动意图识别与自适应对齐
> 最后更新：2026-03-24

---

## 一、项目现状总览

### 数据与模型

| 项目 | 参数 |
|------|------|
| 数据集 | NinaPro DB2（40 受试者，12 通道 sEMG，2000 Hz） |
| 任务 | 17 手势 + 静息，共 18 类 |
| 模型 | 3 层 LSTM（256 hidden，dropout=0.3） |
| 窗口 | 300 ms，stride=50 ms |
| 训练 | 80/20 划分，early stopping（patience=25） |
| 已训练模型 | `checkpoints_2000hz/best_model_S{1-40}.pth` |

### 已有实验与核心结论

| 实验 | 文件 | 核心结论 |
|------|------|---------|
| E1 消融（向量级 vs 元素级遮挡） | `e1.py` | 向量级遮挡更平滑，标准差更低 |
| E2 机理验证（合成 50 ms 注入） | `e2.py` | ST-SRI 可精确恢复注入延迟 |
| E3 生理验证（EMD 检测） | `e3.py` | 56.4%（22/39）受试者峰值落入 30–100 ms |
| E4 稳定性对比（ST-SRI vs SHAP） | `e4.py` | ST-SRI 跨样本峰值方差更小 |
| E5 忠实度（遮挡实验） | `e5.py` | EMD 窗口遮挡下降 3.67% vs 近期 2.27%，p=7.3e-07，Cohen's d=0.625 |
| E6 敏感性（block_size=1/2/5） | `e6.py` | 方法对掩码粒度鲁棒 |
| **实验二（多Δt提前预测）** | `exp_anticipation.py` | **Δt*=250ms；LSTM在现有模型下所有Δt均满足F1≥0.80** |
| **实验四（自适应对齐）** | `exp_alignment.py` | Eval: individual≈fixed_55ms（p=0.676，d=-0.003）；Train模式进行中 |

### 关键数字

- E3 有效峰值均值：**55.6 ± 26.5 ms**（生理合理范围 30–100 ms）
- E5：**p = 7.3e-07**，**Cohen's d = 0.625**（中等效应），Wilcoxon p = 3.3e-09
- E5 影响比：**Rf = 1.61**（EMD 窗口影响是近期窗口的 1.61 倍）
- **实验二（Eval）**：Δt*=250ms；Δt=250ms时 F1=0.836±0.080，Acc=0.862
- **实验四（Eval）**：individual(F1=0.873) vs fixed_55ms(F1=0.874)，p=0.676（等价）

---

## 二、完整实验设计方案（提案）与现有代码对照

### 总体映射

```
提案完整证据链：
  [EMD存在] → [提前预测可行] → [SRI能定位窗口] → [定位窗口能改善预测]
      ↑              ↑                 ↑                    ↑
   E3间接覆盖    ✅ exp_anticipation    E1-E6基本覆盖     ✅ exp_alignment(train进行中)
   缺直接量化    Δt*=250ms(eval)                         train结果待定
```

| 提案实验 | 内容 | 现有代码 | 覆盖状态 | 优先级 |
|---------|------|---------|---------|--------|
| **实验一** | sEMG onset vs 机械 onset 直接 EMD 量化 | E3（间接） | 部分覆盖 | P1 |
| **实验二** | 多 Δt（0/50/100/150/200/250 ms）提前预测 | `exp_anticipation.py` | ✅ **已完成** | P0 |
| **实验三** | ST-SRI 协同谱验证、稳定性、faithfulness | E1~E6 | 基本覆盖 | 已有 |
| **实验四** | 解释驱动自适应对齐补偿 | `exp_alignment.py` | ⏳ **train进行中** | P0 |
| **实验五** | SRI 通道筛选 | 无 | 缺失 | P2 |
| **实验六** | 跨受试者/跨天泛化，噪声鲁棒性 | 无 | 缺失 | P2 |

---

## 三、各实验详细方案

### 实验一：确认 EMD 错位（补充直接量化）

**目标**：直接对每个 trial 计算 sEMG onset 与 force/kinematic onset 的时间差，证明"EMD 不是常数"。

**方法**：
- sEMG onset：带通滤波（20–450 Hz）→ 整流 → 平滑（50 ms RMS）→ 阈值法或 Teager-Kaiser 能量法
- 机械 onset：DB2 中的 force 或 kinematic 信号，同样阈值检测
- 计算每个 trial 的 EMD = 机械 onset − sEMG onset
- 统计：每动作、每受试者的均值 ± 标准差

**输出**：EMD 分布表 + 个体差异图 + 动作差异图

**与 E3 关系**：E3 是 XAI 视角下的间接验证，此实验是信号处理层面的直接 ground truth，两者互补。

---

### 实验二：多 Δt 提前预测 baseline（**首要开发目标**）

**目标**：建立"给定提前量 Δt，预测未来动作意图"的标准任务框架。

**任务设计**：
- 提前量：Δt = 0, 50, 100, 150, 200, 250 ms
- 输入：当前时刻之前 300 ms 的 sEMG 窗口（固定长度）
- 标签：当前时刻 **之后 Δt ms** 的动作类别
- 实现：`NinaProDataset` 增加 `anticipation_ms` 参数，label 取 `window_end + Δt` 处的类别

**对比模型**：
- SVM（时域/频域特征）
- 1D-CNN / TCN
- LSTM（现有）

**核心评价指标**：
- Accuracy、Macro-F1
- **最早稳定提前量 Δt\***：Macro-F1 > 0.80 且跨受试者标准差可接受时，对应最大 Δt

**关键改动（`common.py`）**：
```python
class NinaProDataset(Dataset):
    def __init__(self, ..., anticipation_ms=0):
        self.anticipation_steps = int(anticipation_ms * self.fs / 1000)

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_len
        label_pos = end + self.anticipation_steps  # 标签前移
        if label_pos >= len(self.labels):
            label_pos = len(self.labels) - 1
        return self.data[start:end, :], self.labels[label_pos]
```

---

### 实验三：SRI 解释验证（已基本完成）

**已完成**：E1（消融）、E2（机理）、E3（生理）、E4（稳定性）、E5（faithfulness）、E6（敏感性）

**待完善**：
- E5 目前对照是"最近 0–20 ms 窗口"，可补充**真随机位置窗口**作为第三对照，使 Rf 计算更严格
- 统计方案：已有 paired t-test + Cohen's d + Bootstrap CI + Wilcoxon，符合要求

**Faithfulness Ratio（Rf）定义**：
$$R_f = \frac{\Delta\text{Perf}_{\text{SRI-window}}}{\Delta\text{Perf}_{\text{control-window}}}$$

现有结果：Rf = 3.67% / 2.27% = **1.61**（> 1，SRI 窗口更关键）

---

### 实验四：解释驱动的自适应对齐补偿（**核心创新目标**）

**目标**：将 E3 检测到的协同峰位置（存于 `subject_peaks_e3.json`）用于窗口前移，形成"解释 → 补偿 → 提升性能"的闭环。

**三种补偿策略对比**：

| 策略 | 方法 | 实现 |
|------|------|------|
| 固定补偿 | 统一前移 50 ms 或 70 ms | anticipation_ms 设为固定值 |
| 个体级补偿 | 每受试者用训练集协同峰均值 | 从 subject_peaks_e3.json 读取 |
| 动作级/自适应补偿 | 按动作类别或 trial 动态估计 | 对每类动作单独计算协同峰 |

**评价指标**：
- 各 Δt 下的 Accuracy、F1
- 最早稳定提前量 Δt*（相同准确率门槛下，自适应是否比固定多提前 30–50 ms）
- 端到端反应延迟代理指标

---

### 实验五：通道筛选与机制

**目标**：用 SRI 的独立性 + 协同性筛选通道，比较 12→8→6→4 通道时的性能下降曲线。

**筛选策略对比**：随机筛选 / 单独 SHAP 排序 / 传统特征选择 / SRI 组合贡献筛选

---

### 实验六：泛化与鲁棒性

**目标**：跨受试者（DB2 Leave-one-out 或 DB6 跨天）+ 模拟噪声扰动（电极偏移、掉道、幅值漂移、SNR 下降）。

**核心关注点**：在时序扰动下，协同峰位置是否稳定（vs 标准 SHAP 解释漂移）。

---

## 四、统计方案

| 因变量 | 检验方法 |
|--------|---------|
| Accuracy、Macro-F1、AUC | 重复测量 ANOVA（方法 × Δt × 受试者） |
| 协同峰时滞 τ* | Wilcoxon 符号秩检验（分布不正态时） |
| Faithfulness 比较 | Paired t-test + Cohen's d + Bootstrap CI |
| 通道筛选性能 | Friedman 检验（多组非参数） |

统一报告效应量：Cohen's d 或 Cliff's delta。

---

## 五、优先级行动计划

### P0（当前任务）

1. **修改 `common.py`**：NinaProDataset 增加 `anticipation_ms` 参数
2. **创建 `exp_anticipation.py`**：多 Δt × 多模型的提前预测实验框架
3. **创建 `exp_alignment.py`**：读取 `subject_peaks_e3.json`，实现三种对齐补偿策略，评估提前预测性能提升

### P1

4. **补充实验一**：用 DB2 force/kinematic 数据做 sEMG onset vs 机械 onset 直接量化
5. **强化 E5**：增加真随机位置对照组，完善 Rf 计算

### P2

6. **实验五**：通道筛选
7. **实验六**：泛化鲁棒性（DB6 或跨受试者）

---

## 六、已知问题与建议

### E3 相关

- 改进版（平衡采样）失败（成功率 10.3%），已回退使用原始版本（56.4%）
- 根本原因：ST-SRI 对 baseline 敏感，block_size=2（1 ms）粒度过细
- **论文策略**：诚实报告 56% 并分析失败原因，定位为 proof-of-concept

### ST-SRI 方法局限性

1. **Baseline 依赖**：当前使用随机 20 样本均值，建议改为 rest 类别样本均值
2. **Block size**：2000 Hz 下 block_size=2 仅遮挡 1 ms，EMD 是 30–100 ms 现象，建议增大到 10–20
3. **因果方向不明确**：遮挡导致的性能下降可能来自真实时序依赖或模型记忆效应

---

## 七、投稿建议

| 会议 | 适合度 | 原因 |
|------|--------|------|
| **EMBC** | ⭐⭐⭐⭐⭐ | 生物医学工程，接受探索性工作 |
| **ICASSP** | ⭐⭐⭐⭐ | 信号处理，重视方法创新 |
| NeurIPS XAI Workshop | ⭐⭐⭐⭐ | 探索性方法理想平台 |
| IJCNN | ⭐⭐⭐ | 接受范围广 |
| NeurIPS/ICML 主会 | ❌ | 56% 成功率不够 |

---

## 八、论文框架（完成 P0 后）

**标题建议**：
*Interpretable Early Motion Intent Recognition from sEMG via Synergy-Guided Temporal Alignment*

**Abstract 核心 claim**（目标状态）：
1. ST-SRI 能稳定定位个体化 EMD 窗口（E3，56.4%，55.6 ms）
2. 该窗口对提前预测模型决策具有显著 faithfulness（E5，p<0.001，d=0.625）
3. 将该窗口用于自适应对齐后，在相同准确率门槛下比固定补偿额外提前 X ms（实验四结果待填）

---

## 九、参考文献（关键）

- Cavanagh & Komi (1979). Electromechanical delay in human skeletal muscle. *Eur J Appl Physiol*.
- Atzori et al. (2014). NinaPro DB2. *Scientific Data*, 1, 140053.
- Lundberg & Lee (2017). SHAP. *NeurIPS*.
- Sundararajan et al. (2017). Integrated Gradients. *ICML*.
