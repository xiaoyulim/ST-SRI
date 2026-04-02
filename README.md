# ST-SRI 项目说明

> Spatio-Temporal Synergistic Redundancy Interaction for sEMG Gesture Recognition
> 面向 EMD 时序错位的可解释早期 sEMG 运动意图识别与自适应对齐

---

## 📁 项目结构

```
ST_SRI_Project/
├── scripts/                      # 工具脚本
│   ├── 01_preproccess.py         # 数据预处理
│   ├── check_acc.py              # 检查模型准确率
│   ├── check_e2_structure.py     # 检查E2数据结构
│   └── plot_e5_enhanced.py       # E5增强版绘图
│
├── experiments/                  # 实验代码
│   ├── baseline/                 # 基线实验
│   │   ├── e0.py                 # 基线模型评估
│   │   └── e0_train.py           # 基线模型训练
│   │
│   ├── basic/                    # 基础实验 (E1-E6)
│   │   ├── e1.py                 # 消融实验
│   │   ├── e2.py                 # 机理验证
│   │   ├── e3.py                 # 生理验证 ⭐
│   │   ├── e4.py                 # 稳定性对比
│   │   ├── e5.py                 # 忠实度评估
│   │   ├── e5_enhanced.py        # 忠实度增强 ⭐
│   │   └── e6.py                 # 敏感性分析
│   │
│   └── advanced/                 # 补充实验 (E7-E12)
│       ├── e7.py                 # EMD直接量化
│       ├── e8.py                 # 提前预测 ⭐
│       ├── e9.py                 # 自适应对齐 ⭐
│       ├── e10.py                # 通道筛选
│       ├── e11.py                # LOSO跨受试者
│       └── e12.py                # 噪声鲁棒性
│
├── common.py                     # 核心模块
│
├── data/                         # NinaPro DB2 数据
│   ├── S1_data.npy               # E1 (手势) 数据
│   ├── S1_E2_A1.mat              # E2 (含force) 数据
│   └── ...
│
├── checkpoints/                  # 预训练模型
│   ├── checkpoints_2000hz/       # 基础模型
│   ├── checkpoints_anticipation/ # 提前预测模型
│   └── checkpoints_alignment/    # 对齐模型
│
├── results/                      # 实验结果
│   ├── e3.png                    # E3结果
│   ├── e5_faithfulness/          # E5忠实度
│   ├── anticipation/             # 提前预测
│   ├── alignment/               # 自适应对齐
│   ├── emd_direct/              # EMD直接量化
│   ├── loso/                    # LOSO
│   └── generalization/          # 噪声鲁棒性
│
├── good_subjects.json            # 合格受试者列表
├── subject_peaks_e3.json          # E3 ST-SRI检测峰值
│
└── README.md                     # 本文件
```

---

## 🎯 核心实验结果

### 基础实验 (E1-E6)

| 实验 | 目的 | 关键结果 |
|------|------|----------|
| E1 | 消融：Vector vs Element遮挡 | Vector更平滑 |
| E2 | 机理：合成信号验证 | 成功检测50ms延迟 |
| E3 | 生理：EMD延迟检测 | 56.4%在30-100ms，峰值55.6ms |
| E4 | 稳定性：ST-SRI vs SHAP | ST-SRI更稳定 |
| E5 | 忠实度：遮挡实验 | p=7.3e-07, Cohen's d=0.625 |
| E5增强 | 忠实度：三组对照实验 | Rf=1.61, p=5.8e-07 |
| E6 | 敏感性：block_size分析 | 对粒度鲁棒 |

### 补充实验 (E7-E12)

| 实验 | 目的 | 关键结果 |
|------|------|----------|
| E7 | EMD直接量化 | 86.9±34.6ms (直接) vs 65.3ms (ST-SRI) |
| E8 | 提前预测 | Δt*=250ms, F1=0.836 |
| E9 | 自适应对齐 | individual≈fixed_55ms (p=0.676) |
| E10 | 通道筛选 | 12→4通道性能下降曲线 |
| E11 | LOSO跨受试者 | 63.5%±1.5% |
| E12 | 噪声鲁棒性 | 84%→59% (噪声0→0.3) |

---

## 🚀 快速开始

### 1. 环境配置
```bash
conda create -n xai_lab python=3.9
conda activate xai_lab
pip install torch numpy scipy matplotlib pandas scikit-learn
```

### 2. 数据预处理
```bash
python scripts/01_preproccess.py
```

### 3. 运行实验
```bash
# 基础实验
python experiments/basic/e3.py
python experiments/basic/e5_enhanced.py

# 补充实验 (需要较长时间)
python experiments/advanced/e8.py --mode eval
python experiments/advanced/e9.py --mode eval
python experiments/advanced/e7.py
```

---

## 📊 完整证据链

```
[EMD存在] ──[提前预测可行]──[SRI能定位窗口]──[定位窗口能改善预测]
    │            │                │                  │
   E3间接    ✅ E8             ✅ E1-E6基本覆盖   ✅ E9
   缺直接     Δt*=250ms                              individual≈fixed
    ↓
✅ E7(补充)     ✅ E5增强(Rf=1.61)      ✅ E11-E12(泛化鲁棒性)
```

### 核心Claim（论文）
1. ST-SRI能稳定定位个体化EMD窗口（E3，56.4%，55.6ms）
2. EMD窗口对预测决策有显著faithfulness（E5/E5增强，p<0.001，d=0.625）
3. 自适应对齐与固定补偿效果相当（E9，p=0.676）
4. EMD直接量化补充验证（E7，86.9±34.6ms）
5. 模型具有噪声鲁棒性和跨受试者泛化能力（E11-E12）

---

## 🔧 核心参数

```python
FS = 2000                    # 采样率 (Hz)
WINDOW_MS = 300              # 窗口长度 (ms)
STRIDE_MS = 50               # 步长 (ms)
HIDDEN_SIZE = 256            # LSTM隐藏层
NUM_LAYERS = 3               # LSTM层数
DROPOUT = 0.3                # Dropout
NUM_CLASSES = 18             # 17手势 + 1静息
CHANNELS = 12                # sEMG通道数
```

---

## 📚 数据集

- **数据集**: NinaPro DB2
- **受试者**: 40人（有效39人）
- **动作**: Exercise 1，17手势 + rest
- **通道**: 12通道 sEMG
- **采样率**: 2000 Hz

---

## 📊 详细实验结果

### E5增强版：三组忠实度评估
- **受试者数**: 38
- **Rf_recent**: 1.61 (EMD vs Recent)
- **Rf_random**: 1.10 (EMD vs Random)
- **EMD vs Recent**: t=6.02, p=5.8e-07, Cohen's d=0.65
- **EMD vs Random**: t=3.06, p=0.004, Cohen's d=0.13

### E8：提前预测
- **Δt***: 250ms (F1≥0.80且std<0.10)
- **F1@250ms**: 0.836±0.080
- **Acc@250ms**: 0.862

### E9：自适应对齐
- **individual vs fixed_55ms**: p=0.676 (无显著差异)
- **individual F1**: 0.873±0.088
- **fixed_55ms F1**: 0.874±0.089

### E7：EMD直接量化
- **全局EMD**: 86.9±34.6ms
- **有效受试者**: 14/30
- **与E3对比**: ST-SRI峰值65.3ms

### E11：LOSO跨受试者
- **平均准确率**: 63.5%±1.5%
- **最高**: 68.0%
- **最低**: 63.0%

### E12：噪声鲁棒性
- **无噪声**: 84.1%±14.3%
- **噪声0.1**: 81.4%±14.3%
- **噪声0.2**: 69.0%±17.8%
- **噪声0.3**: 59.3%±19.4%

### E10：通道筛选
- **12通道**: 77.2%±16.4%
- **8通道**: 78.4%±15.9%
- **6通道**: 72.6%±16.5%
- **4通道**: 62.1%±16.1%

---

## 📞 问题排查

### CUDA OOM
```python
# 修改 common.py
DEVICE = torch.device("cpu")
```

---

**最后更新**: 2026-04-02
**项目状态**: 实验完成
**GitHub**: https://github.com/xiaoyulim/ST-SRI
