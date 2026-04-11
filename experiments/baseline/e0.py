"""
基线实验（E0）：SHAP 基线评估
===============================
目标：使用标准 SHAP（GradientExplainer）对已训练的 LSTM 模型
      生成基线归因结果，作为 ST-SRI 的对比基准。

方法：
  - 对每个受试者的模型运行 SHAP 归因
  - 生成时间维归因热力图

输出：
  - 每个受试者的 SHAP 归因结果图
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import os
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d
from common import NinaProDataset, LSTMModel, DEVICE, FS

# 1. 修复 OMP 报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================= 配置 =================
SUB_ID = 1
WINDOW_MS = 300
BIN_SIZE = 15
CHECKPOINT_PATH = f"./checkpoints_2000hz/best_model_S{SUB_ID}.pth"
SAVE_PATH = "results/e0_blue_line.png" # 换个名字以防覆盖
# ========================================

def run_e0_single_subject():
    os.makedirs("./results", exist_ok=True)
    print(f">>> [E0] 正在生成 S{SUB_ID} 的时序归因图 (蓝色主线版)...")

    # 1. 准备数据 (代码逻辑不变)
    try:
        ds = NinaProDataset("./data", SUB_ID, window_ms=WINDOW_MS, target_fs=FS)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    loader = DataLoader(ds, batch_size=100, shuffle=True)
    test_x, test_y = next(iter(loader))
    samples = test_x[test_y != 0][:20].to(DEVICE)
    bg_data = ds.data[:ds.window_len].unsqueeze(0).to(DEVICE)

    # 2. 加载模型
    model = LSTMModel().to(DEVICE)
    if not os.path.exists(CHECKPOINT_PATH): return
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False))
    model.eval()

    # 3. 计算 SHAP
    explainer = shap.GradientExplainer(model, bg_data)
    with torch.backends.cudnn.flags(enabled=False):
        shap_values = explainer.shap_values(samples)

    # 4. 维度修复 (代码逻辑不变)
    if isinstance(shap_values, list):
        arr = np.array(shap_values)
        arr = np.abs(arr)
        arr = arr.sum(axis=0)
    else:
        arr = np.abs(shap_values)

    if arr.ndim == 3:
        shap_matrix = arr.sum(axis=-1)
    elif arr.ndim == 2:
        shap_matrix = arr
    else:
        shap_matrix = arr.reshape(len(samples), -1)

    S, T = shap_matrix.shape

    # 5. 分桶聚合
    T_new = T // BIN_SIZE
    binned = shap_matrix[:, :T_new * BIN_SIZE].reshape(S, T_new, BIN_SIZE).mean(axis=2)

    for i in range(S):
        _min, _max = binned[i].min(), binned[i].max()
        binned[i] = (binned[i] - _min) / (_max - _min + 1e-9)

    # ================= 6. 绘图 (改色部分) =================
    plt.figure(figsize=(12, 6), dpi=150)
    time_axis = np.linspace(-WINDOW_MS, 0, T_new)

    # --- 细线 (个体现): 保持红色，代表混乱 ---
    first_flag = True
    for i in range(S):
        label_str = "Individual Sample (Chaotic)" if first_flag else None
        # 这里的颜色改成了浅红色，对比更强
        plt.plot(time_axis, binned[i], color='#e74c3c', alpha=0.15, linewidth=0.8, label=label_str)
        first_flag = False

    # --- 主线 (均值): 改为深蓝色 ---
    mean_imp = np.mean(binned, axis=0)
    mean_smooth = gaussian_filter1d(mean_imp, sigma=1.2)
    # 【改动】color='#1f77b4' (深蓝)
    plt.plot(time_axis, mean_smooth, color='#1f77b4', linewidth=3, label='Mean SHAP Importance')

    # 修饰
    plt.title(f"Motivation: Temporal Instability of Standard SHAP (Subject {SUB_ID})", fontsize=14, weight='bold')
    plt.xlabel("Time relative to current action (ms)", fontsize=12)
    plt.ylabel("Normalized Importance", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(-WINDOW_MS, 0)
    plt.ylim(-0.05, 1.1)

    plt.legend(loc='upper left', frameon=True, fontsize=10)

    # 标注
    plt.annotate('Fragmented & Random Spikes\n(No consistent pattern)',
                 xy=(-180, 0.6), xytext=(-250, 0.8),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                 fontsize=11)

    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"✅ E0 (Blue Line) Generated: {SAVE_PATH}")

if __name__ == "__main__":
    run_e0_single_subject()