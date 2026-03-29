# ST-SRI 项目说明

> Spatio-Temporal Synergistic Redundancy Interaction for sEMG Gesture Recognition

---

## 📁 项目结构

```
ST_SRI_Project/
├── README.md                      # 本文件
├── EXPERIMENT_DESIGN_SUGGESTIONS.md # 论文实验设计建议
├── CLEANUP_PLAN.md                # 清理计划（已执行）
│
├── data/                          # NinaPro DB2 数据
│   ├── S1_data.npy
│   ├── S1_label.npy
│   └── ...
│
├── checkpoints_2000hz/            # 训练好的模型
│   ├── best_model_S1.pth
│   └── ...
│
├── 01_preproccess.py              # 数据预处理脚本
├── common.py                      # 核心模块（模型、ST-SRI、统计工具）
├── check_acc.py                   # 模型准确率检查
│
├── e1.py                          # 实验1: 消融实验
├── e2.py                          # 实验2: 机理验证
├── e3.py                          # 实验3: 生理验证 ⭐
├── e4.py                          # 实验4: 稳定性对比
├── e5.py                          # 实验5: 忠实度评估 ⭐
├── e6.py                          # 实验6: 敏感性分析
│
├── good_subjects.json             # 合格受试者列表 (39人)
├── subject_peaks_e3.json          # E3 峰值检测结果
│
└── results/                       # 实验结果
    ├── e1.png
    ├── e2.png
    ├── e3.png                     # ⭐ 论文主图
    ├── e4.png
    ├── e5_faithfulness/           # ⭐ 论文主要结果
    │   ├── e5_detailed_results.csv
    │   ├── e5_final_plot.png
    │   └── e5_summary.json
    └── e6_sensitivity.png
```

---

## 🎯 核心实验结果

### E3: 生理验证（EMD 延迟检测）
- **成功率**: 56.4% (22/39 subjects)
- **有效峰值**: 55.6 ± 26.5 ms
- **结论**: 在半数以上受试者中检测到生理合理的 EMD 延迟

### E5: 忠实度评估（遮挡实验）
- **统计显著性**: p = 7.3e-07 (高度显著)
- **效应量**: Cohen's d = 0.625 (中等)
- **影响比**: 遮挡 EMD 位置导致准确率下降 3.67%，而遮挡近期仅下降 2.27%
- **结论**: ST-SRI 检测到的时滞位置确实对模型性能有显著影响

---

## 🚀 快速开始

### 1. 环境配置
```bash
pip install torch numpy scipy matplotlib pandas
```

### 2. 数据预处理（如果需要）
```bash
python 01_preproccess.py
```

### 3. 检查模型准确率
```bash
python check_acc.py
```

### 4. 运行实验
```bash
# 运行所有实验
python e1.py
python e2.py
python e3.py
python e4.py
python e5.py
python e6.py

# 或单独运行关键实验
python e3.py  # 生理验证
python e5.py  # 忠实度评估
```

---

## 📊 实验说明

### E1: 消融实验
- **目的**: 对比 Vector-wise 和 Element-wise 遮挡策略
- **结果**: Vector-wise 更平滑，Element-wise 噪声大
- **输出**: `results/e1.png`

### E2: 机理验证
- **目的**: 在合成信号上验证 ST-SRI 能否检测已知延迟
- **方法**: 人工注入 50ms 延迟
- **结果**: 成功检测到峰值在 50ms 附近
- **输出**: `results/e2.png`

### E3: 生理验证 ⭐
- **目的**: 在真实 sEMG 数据上检测 EMD 延迟
- **方法**: 分析 39 个受试者的动作 onset 时刻
- **结果**: 56.4% 受试者峰值在生理范围 (30-100ms)
- **输出**: `results/e3.png`, `subject_peaks_e3.json`

### E4: 稳定性对比
- **目的**: 对比 ST-SRI 和 SHAP 的稳定性
- **方法**: 在同一受试者的多个样本上计算重要性
- **结果**: ST-SRI 曲线更平滑，SHAP 波动大
- **输出**: `results/e4.png`

### E5: 忠实度评估 ⭐
- **目的**: 验证 ST-SRI 检测到的时滞是否真实影响模型
- **方法**: 遮挡检测到的 EMD 位置，测量准确率下降
- **结果**:
  - 遮挡 EMD: -3.67%
  - 遮挡近期: -2.27%
  - p < 0.001, Cohen's d = 0.625
- **输出**: `results/e5_faithfulness/`

### E6: 敏感性分析
- **目的**: 测试不同 block_size 对结果的影响
- **方法**: 对比 block_size = 1, 2, 5
- **结果**: 结果对 block_size 相对鲁棒
- **输出**: `results/e6_sensitivity.png`

---

## 📝 论文写作建议

### 适合投稿的会议/期刊
- ✅ **EMBC** (IEEE Engineering in Medicine and Biology Conference)
- ✅ **ICASSP** (International Conference on Acoustics, Speech and Signal Processing)
- ✅ **NeurIPS XAI Workshop** (探索性工作)
- ⚠️ **IJCNN** (International Joint Conference on Neural Networks)

### 核心 Claim
1. **方法新颖性**: 首次将 Synergy-Redundancy 分解应用于时序归因
2. **大规模验证**: 在 39 个受试者上验证（NinaPro DB2）
3. **统计严谨**: Cohen's d, Bootstrap CI, Wilcoxon 检验
4. **生理合理性**: 56% 受试者检测到合理的 EMD 延迟

### 写作策略
- ✅ **诚实报告**: 56% 成功率是探索性研究的合理结果
- ✅ **强调新颖性**: Synergy + Redundancy 分解的概念
- ✅ **深入分析**: 成功和失败案例的对比
- ✅ **讨论局限性**: Baseline 敏感性、个体差异

### 不要过度 Claim
- ❌ "准确检测 EMD 延迟"
- ✅ "探索性框架，在半数以上受试者中检测到生理合理的延迟"

---

## 🔧 核心模块说明

### common.py
包含：
- `NinaProDataset`: 数据加载器
- `LSTMModel`: LSTM 分类模型
- `ST_SRI_Interpreter`: ST-SRI 解释器
- `calculate_cohens_d`: Cohen's d 计算
- `bootstrap_ci`: Bootstrap 置信区间
- `interpret_cohens_d`: 效应量解释

### 关键参数
```python
FS = 2000              # 采样率 (Hz)
WINDOW_MS = 300        # 窗口长度 (ms)
BLOCK_SIZE = 2         # 遮挡块大小 (时间点)
MAX_LAG_MS = 150       # 最大扫描延迟 (ms)
```

---

## 📚 数据集信息

- **数据集**: NinaPro DB2
- **受试者**: 40 人（S1-S40，S21 缺失）
- **动作**: Exercise 1, 17 个手势 + rest
- **通道**: 12 个 sEMG 通道
- **采样率**: 2000 Hz
- **数据格式**: `.npy` (预处理后)

---

## ⚠️ 注意事项

1. **模型已训练**: `checkpoints_2000hz/` 包含所有受试者的模型，无需重新训练
2. **数据已预处理**: `data/` 包含 `.npy` 格式数据，无需重新预处理
3. **结果可复现**: 所有实验使用固定随机种子
4. **计算资源**: 建议使用 GPU，CPU 也可运行但较慢

---

## 📞 问题排查

### 问题 1: CUDA out of memory
```python
# 在 common.py 中修改
DEVICE = torch.device("cpu")  # 强制使用 CPU
```

### 问题 2: 找不到数据文件
```bash
# 确保数据文件存在
ls data/S1_data.npy
ls data/S1_label.npy
```

### 问题 3: 模型加载失败
```bash
# 检查模型文件
ls checkpoints_2000hz/best_model_S1.pth
```

---

## 📖 参考文献

### 生理学基础
- Cavanagh & Komi (1979). Electromechanical delay in human skeletal muscle.

### 数据集
- Atzori et al. (2014). Electromyography data for non-invasive naturally-controlled robotic hand prostheses. Scientific Data.

### 可解释性方法
- Lundberg & Lee (2017). A unified approach to interpreting model predictions. NeurIPS.
- Shapley (1953). A value for n-person games. Contributions to the Theory of Games.

---

**最后更新**: 2026-03-07
**项目状态**: 实验完成，准备论文写作
**联系方式**: [Your Email]
