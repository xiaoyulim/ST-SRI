import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d
from common import NinaProDataset, LSTMModel, ST_SRI_Interpreter, DEVICE, FS

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================= 配置 =================
DATA_ROOT = "./data"
CHECKPOINT_DIR = "./checkpoints_2000hz"
RESULT_DIR = "./results/e3_analysis"

WINDOW_MS = 300
NUM_CLASSES = 18

TARGET_COUNT = 30  # 每个受试者取的样本数
SMOOTH_SIGMA = 2.0
# =======================================

os.makedirs(RESULT_DIR, exist_ok=True)


def analyze_one_subject(sub_id):
    """分析单个受试者并缓存结果"""
    synergy_save_path = f"{RESULT_DIR}/S{sub_id}_synergy.npy"
    if os.path.exists(synergy_save_path):
        return np.load(synergy_save_path, allow_pickle=True)

    print(f">>> Analyzing S{sub_id} (Onset Detection Mode)...")

    try:
        ds = NinaProDataset(DATA_ROOT, sub_id, window_ms=WINDOW_MS, target_fs=FS)
    except:
        return None

    model_path = f"{CHECKPOINT_DIR}/best_model_S{sub_id}.pth"
    if not os.path.exists(model_path): return None

    model = LSTMModel(input_size=12, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
    model.eval()

    # 文档建议：baseline 改为 rest 类（label=0）样本均值，减少对随机采样的依赖
    rest_loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    all_x, all_y = next(iter(rest_loader))
    rest_mask = (all_y == 0)
    if rest_mask.sum() >= 20:
        bg_data = all_x[rest_mask][:200].to(DEVICE)   # 最多取 200 个 rest 样本
    else:
        # rest 样本不足时退回随机采样
        bg_loader = DataLoader(ds, batch_size=20, shuffle=True)
        bg_data, _ = next(iter(bg_loader))
        bg_data = bg_data.to(DEVICE)
    interpreter = ST_SRI_Interpreter(model, bg_data)

    seq_loader = DataLoader(ds, batch_size=1, shuffle=False)

    synergies = []
    lags_ms = []
    prev_label = 0
    collected_count = 0

    for x, y in seq_loader:
        curr_label = y.item()
        is_onset = (prev_label == 0) and (curr_label != 0)
        prev_label = curr_label

        if is_onset:
            x_tensor = x[0].to(DEVICE)
            try:
                # 文档建议：block_size 从 2 调大到 10（遮挡 5ms，更适合 30-100ms EMD 现象）
                l, s, _ = interpreter.scan_fast(x_tensor, max_lag_ms=150, stride=1, block_size=10)
                synergies.append(s)
                lags_ms = l
                collected_count += 1
                print(f"  S{sub_id} Onsets Found: {collected_count}/{TARGET_COUNT}", end='\r')
            except:
                continue
        if collected_count >= TARGET_COUNT: break

    if not synergies:
        return None

    mean_syn = np.mean(synergies, axis=0)
    np.save(synergy_save_path, mean_syn)
    np.save(f"{RESULT_DIR}/lags.npy", np.array(lags_ms))
    return mean_syn


def plot_e3_final(lags, all_syn_list):
    """执行最终的绘图逻辑：反转轴 + 动态量程优化"""
    print("\n>>> Plotting Final Optimized Spectrum...")

    # 1. 样本级归一化
    norm_syn = []
    for s in all_syn_list:
        s_min, s_max = s.min(), s.max()
        norm_syn.append((s - s_min) / (s_max - s_min + 1e-9))
    norm_syn = np.array(norm_syn)

    # 2. 计算均值和标准差
    mean_syn = np.mean(norm_syn, axis=0)
    std_syn = np.std(norm_syn, axis=0)

    # 3. 翻转轴：从 (0 to 150) 变为 (-150 to 0)
    time_axis = -lags[::-1]
    mean_plot = mean_syn[::-1]
    std_plot = std_syn[::-1]

    # 平滑处理 (Sigma=2.0 保持趋势)
    mean_smooth = gaussian_filter1d(mean_plot, sigma=2.0)
    std_smooth = gaussian_filter1d(std_plot, sigma=2.0)

    # 4. 开始绘图
    plt.figure(figsize=(10, 5.5), dpi=200)
    plt.style.use('seaborn-v0_8-ticks')  # 使用更清爽的样式

    # 标注生理区间
    plt.axvspan(-100, -30, color='#e5f5e0', alpha=0.5, label='Physiological EMD Zone (-100 to -30ms)')

    # 绘制波动范围 (阴影调淡)
    plt.fill_between(time_axis,
                     mean_smooth - std_smooth * 0.2,
                     mean_smooth + std_smooth * 0.2,
                     color='#1f77b4', alpha=0.1, label='Inter-subject Variance')

    # 绘制均值主线
    plt.plot(time_axis, mean_smooth, color='#1f77b4', linewidth=2.5, label='ST-SRI Synergy Spectrum')

    # 5. 自动寻找生理区间内的峰值并标注
    emd_mask = (time_axis >= -100) & (time_axis <= -30)
    if np.any(emd_mask):
        peak_idx_in_mask = np.argmax(mean_smooth[emd_mask])
        peak_time = time_axis[emd_mask][peak_idx_in_mask]
        peak_val = mean_smooth[emd_mask][peak_idx_in_mask]

        # 仅保留红色虚线标示位置
        plt.axvline(peak_time, color='#d62728', linestyle='--', linewidth=1.5, alpha=0.8,
                    label=f'Detected Peak ({abs(peak_time):.1f}ms)')

    # 6. 【核心优化】动态调整 Y 轴量程
    # 找到除 0ms 附近冗余点外的最大值，用于拉伸波形
    # 我们主要关注 -150 到 -10 ms 之间的起伏
    visible_mask = (time_axis < -5)
    y_limit_upper = np.max(mean_smooth[visible_mask]) * 1.3  # 留出 30% 头部空间
    plt.ylim(0, y_limit_upper)

    # 7. 修饰
    plt.title(f"Physiological Validation: ST-SRI Spectrum (N={len(all_syn_list)})", fontsize=13, weight='bold')
    plt.xlabel("Time relative to current action (ms)", fontsize=11)
    plt.ylabel("Normalized Synergy Index", fontsize=11)
    plt.xlim(-150, 0)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(loc='upper left', fontsize=9, frameon=True)

    plt.tight_layout()
    save_path = "./results/e3.png"
    plt.savefig(save_path)
    print(f"✅ Success! Plot saved to: {save_path}")
    print(f"   Y-axis range optimized to [0, {y_limit_upper:.2f}]")


def run_e3_main():
    # 1. 读取受试者名单
    if os.path.exists("good_subjects.json"):
        with open("good_subjects.json", "r") as f:
            target_subjects = json.load(f)
    else:
        target_subjects = list(range(1, 11))  # 默认跑前10个

    all_syn_list = []
    print(f"Target Subjects: {target_subjects}")

    # 2. 收集数据
    for sub in target_subjects:
        res = analyze_one_subject(sub)
        if res is not None:
            all_syn_list.append(res)

    if not all_syn_list:
        print("Error: No data collected.")
        return

    # 3. 加载时间轴
    lags = np.load(f"{RESULT_DIR}/lags.npy")

    # 4. 【核心修复】正式调用绘图函数
    plot_e3_final(lags, all_syn_list)


if __name__ == '__main__':
    run_e3_main()