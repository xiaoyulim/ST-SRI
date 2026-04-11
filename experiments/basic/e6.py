import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader
from common import NinaProDataset, LSTMModel, ST_SRI_Interpreter, DEVICE, FS

# 配置
BLOCK_SIZES = [1, 2, 5]
COLORS = ['#2ca02c', '#1f77b4', '#d62728']


def run_e6():
    print(">>> [E6] Sensitivity Analysis (Time Axis Fixed)...")

    if os.path.exists("good_subjects.json"):
        with open("good_subjects.json", "r") as f:
            # 取前 5 个跑得快一点，也可以全跑
            SUB_LIST = json.load(f)[:8]
    else:
        SUB_LIST = [1, 2, 3, 4, 5]

    results = {bs: [] for bs in BLOCK_SIZES}

    for sub in SUB_LIST:
        print(f"Processing S{sub}...", end='\r')
        try:
            ds = NinaProDataset("./data", sub, window_ms=300, target_fs=FS)
            model_path = f"./checkpoints_2000hz/best_model_S{sub}.pth"
            if not os.path.exists(model_path): continue

            sample_x, _ = ds[0]
            input_dim = sample_x.shape[1]
            model = LSTMModel(input_size=input_dim, num_classes=18).to(DEVICE)

            try:
                model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            except:
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()

            # 准备解释器
            bg_loader = DataLoader(ds, batch_size=20, shuffle=True)
            bg, _ = next(iter(bg_loader))
            bg = bg.to(DEVICE).mean(0)

            interpreter = ST_SRI_Interpreter(model, bg.unsqueeze(0))

            # 找动作样本 (Onset模式)
            target_samples = []
            loader = DataLoader(ds, batch_size=1, shuffle=False)
            prev_label = 0
            for x, y in loader:
                curr = y.item()
                if prev_label == 0 and curr != 0:
                    target_samples.append(x[0].to(DEVICE))
                prev_label = curr
                if len(target_samples) >= 5: break

            if not target_samples: continue

            # 扫描
            for bs in BLOCK_SIZES:
                sub_syns = []
                for x in target_samples:
                    # 调用 common.py 里的 scan_fast (支持 block_size)
                    _, s, _ = interpreter.scan_fast(x, max_lag_ms=150, block_size=bs)
                    sub_syns.append(s)

                if sub_syns:
                    avg_s = np.mean(sub_syns, axis=0)
                    # 归一化
                    _min, _max = avg_s.min(), avg_s.max()
                    norm_s = (avg_s - _min) / (_max - _min + 1e-9)
                    results[bs].append(norm_s)

        except Exception as e:
            continue

    print("\nData collection finished.")

    if len(results[BLOCK_SIZES[0]]) == 0:
        print("❌ Error: No data collected!")
        return

    # === 绘图 ===
    plt.figure(figsize=(10, 6))

    # 画 EMD 区域 (-100ms 到 -30ms)
    # 因为时间轴是负的，所以范围要反过来写
    plt.axvspan(-100, -30, color='#e5f5e0', alpha=0.4, label='EMD Range')

    # 原始 Lags 是正数 [0, 10, 20...]
    # 我们要把它变成负数 [-0, -10, -20...]
    raw_lags = np.arange(len(results[BLOCK_SIZES[0]][0])) * (1000 / FS)
    time_axis = -raw_lags  # 变成负数时间轴
    # ... (前面的代码不变)
    for i, bs in enumerate(BLOCK_SIZES):
        data = results[bs]
        if not data: continue

        data_arr = np.array(data)
        mean_curve = np.mean(data_arr, axis=0)

        # 加大一点平滑力度，让图更好看 (sigma 2.5 -> 3.0)
        mean_smooth = gaussian_filter1d(mean_curve, 3.0)

        # 【修复点】使用浮点数计算，保留一位小数
        ms_val = bs * (1000 / FS)
        label_str = f'Mask Size = {ms_val:.1f} ms'  # 例如 0.5 ms

        plt.plot(time_axis, mean_smooth, color=COLORS[i], linewidth=2.5, label=label_str)
    plt.title("Sensitivity Analysis: Robustness to Masking Granularity")
    plt.xlabel("Time Relative to Action Onset (ms)")  # 标签改了
    plt.ylabel("Normalized Synergy")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.5)

    # 设置 X 轴范围 (-150 到 0)
    plt.xlim(-150, 0)

    save_path = "./results/e6_sensitivity1.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ E6 Plot Saved to {save_path}")


if __name__ == "__main__":
    run_e6()