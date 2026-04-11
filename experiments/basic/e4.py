"""
实验四（E4）：稳定性对比 — ST-SRI vs SHAP
==========================================
目标：比较 ST-SRI 与标准 SHAP 在跨样本归因的一致性。

方法：
  - 对同一受试者的多个不同样本分别运行 ST-SRI 和 SHAP
  - 比较归因峰值位置的跨样本方差

评价指标：
  - 峰值位置标准差（越低越稳定）
  - 归因曲线形状一致性
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import os
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from common import NinaProDataset, LSTMModel, ST_SRI_Interpreter, DEVICE, FS

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================= 配置 =================
target_sub = 31
WINDOW_MS = 300
SAVE_PATH = "./results/e4.png"
NUM_SAMPLES = 15  # 样本数
SCAN_STRIDE = 3  # 扫描步长


# ========================================

def run_e4_final_fix():
    print(f">>> [E4] 正在运行单人对比 (图例强制修复版)...")
    os.makedirs("./results", exist_ok=True)

    # 1. 准备数据
    ds = NinaProDataset("./data", target_sub, window_ms=WINDOW_MS, target_fs=FS)
    model = LSTMModel().to(DEVICE)
    model.load_state_dict(
        torch.load(f"./checkpoints_2000hz/best_model_S{target_sub}.pth", map_location=DEVICE, weights_only=False))
    model.eval()

    active_indices = np.where(ds.labels.numpy() != 0)[0]
    np.random.seed(42)
    indices = np.random.choice(len(active_indices) - 600, NUM_SAMPLES, replace=False)
    samples_tensor = torch.stack([ds.data[i:i + 600] for i in indices]).to(DEVICE)
    bg_data = ds.data[:600].unsqueeze(0).to(DEVICE)

    # 2. 计算 Standard SHAP
    print("   1. 计算 Standard SHAP...")
    explainer = shap.GradientExplainer(model, bg_data)
    with torch.backends.cudnn.flags(enabled=False):
        shap_vals = explainer.shap_values(samples_tensor)

    if isinstance(shap_vals, list):
        shap_res = np.abs(np.array(shap_vals)).sum(axis=0).sum(axis=-1)
    else:
        shap_res = np.abs(shap_vals).sum(axis=-1)
    shap_lines = shap_res[:, -300:][:, ::-1]

    # 3. 计算 ST-SRI
    print("   2. 计算 ST-SRI...")
    interpreter = ST_SRI_Interpreter(model, bg_data)
    st_sri_lines = []
    for i in range(NUM_SAMPLES):
        _, syn, _ = interpreter.scan_fast(samples_tensor[i], max_lag_ms=150, stride=SCAN_STRIDE)
        st_sri_lines.append(syn[::-1])
    st_sri_mean = np.mean(st_sri_lines, axis=0)

    # 4. 绘图 (强制修复图例逻辑)
    print("   3. 绘图...")
    plt.figure(figsize=(10, 6), dpi=150)
    plt.axvspan(-100, -30, color='#e5f5e0', alpha=0.5, label='_nolegend_')  # 这里的 label 设为 _nolegend_ 隐藏

    time_axis_shap = np.linspace(-150, 0, 300)
    time_axis_st = np.linspace(-150, 0, len(st_sri_mean))

    # --- A. 画所有灰线 (全部不加标签！) ---
    for line in shap_lines:
        # 归一化 + 平滑
        y = (line - line.min()) / (line.max() - line.min() + 1e-9)
        y = gaussian_filter1d(y, sigma=1.5)
        # 关键：label='_nolegend_' 告诉 matplotlib 绝对不要显示它
        plt.plot(time_axis_shap, y, color='gray', alpha=0.3, linewidth=1.0, label='_nolegend_')

    # --- B. 画 ST-SRI 蓝线 ---
    y_st = (st_sri_mean - st_sri_mean.min()) / (st_sri_mean.max() - st_sri_mean.min() + 1e-9)
    y_st = gaussian_filter1d(y_st, sigma=2.0)
    plt.plot(time_axis_st, y_st, color='#1f77b4', linewidth=3.5, label='ST-SRI (Mean Trend)')

    # --- C. 手动添加图例项 (Proxy Artist) ---
    # 这是一条看不见的线，只为了在图例里显示“灰色 = Standard SHAP”
    # 这样图例里永远只有这一行灰色，绝不会重复
    plt.plot([], [], color='gray', alpha=0.5, linewidth=1.5, label='Standard SHAP (Individual Samples)')

    # 手动添加生理区间图例
    import matplotlib.patches as mpatches
    green_patch = mpatches.Patch(color='#e5f5e0', alpha=0.5, label='Physiological EMD Zone')

    # --- D. 标峰值 ---
    peak_idx = np.argmax(y_st)
    peak_time = time_axis_st[peak_idx]
    if -110 < peak_time < -20:
        plt.axvline(peak_time, color='#d62728', linestyle='--', linewidth=2,
                    label=f'Detected Peak ({abs(peak_time):.1f}ms)')

    plt.title(f"E4: Stability Comparison (S{target_sub})", fontsize=14, weight='bold')
    plt.xlabel("Time relative to current action (ms)", fontsize=12)
    plt.ylabel("Normalized Importance", fontsize=12)
    plt.xlim(-150, 0)
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle=':', alpha=0.4)

    # 强制重新整理图例句柄，确保顺序和数量正确
    handles, labels = plt.gca().get_legend_handles_labels()
    # 把绿色背景加进去
    handles.insert(0, green_patch)
    labels.insert(0, 'Physiological EMD Zone')

    plt.legend(handles=handles, labels=labels, loc='upper left', frameon=True)

    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"✅ 修复完成，图例已强制合并: {SAVE_PATH}")


if __name__ == "__main__":
    run_e4_final_fix()