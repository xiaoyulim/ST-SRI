import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader
import torch

from common import (
    NinaProDataset,
    LSTMModel,
    ST_SRI_Interpreter,
    DEVICE,
    FS,
    EMD_VALID_MIN_MS,
    EMD_VALID_MAX_MS,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================= 配置 =================
DATA_ROOT = "./data"
CHECKPOINT_DIR = "./checkpoints_2000hz"
RESULT_DIR = "./results/e3_analysis"
WINDOW_MS = 300
NUM_CLASSES = 18
TARGET_COUNT = 30
SMOOTH_SIGMA = 2.0
GOOD_SUBJECTS_PATH = "./good_subjects.json"
PEAKS_PATH = "./subject_peaks_e3.json"
# =======================================

os.makedirs(RESULT_DIR, exist_ok=True)


def load_target_subjects():
    if os.path.exists(GOOD_SUBJECTS_PATH):
        with open(GOOD_SUBJECTS_PATH, "r") as f:
            return json.load(f)
    return list(range(1, 11))


def analyze_one_subject(sub_id):
    """分析单个受试者并缓存结果"""
    synergy_save_path = f"{RESULT_DIR}/S{sub_id}_synergy.npy"
    if os.path.exists(synergy_save_path):
        return np.load(synergy_save_path, allow_pickle=True)

    print(f">>> Analyzing S{sub_id} (Onset Detection Mode)...")

    try:
        ds = NinaProDataset(DATA_ROOT, sub_id, window_ms=WINDOW_MS, target_fs=FS)
    except Exception:
        return None

    model_path = f"{CHECKPOINT_DIR}/best_model_S{sub_id}.pth"
    if not os.path.exists(model_path):
        return None

    model = LSTMModel(input_size=12, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
    model.eval()

    rest_loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    all_x, all_y = next(iter(rest_loader))
    rest_mask = (all_y == 0)
    if rest_mask.sum() >= 20:
        bg_data = all_x[rest_mask][:200].to(DEVICE)
    else:
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
                l, s, _ = interpreter.scan_fast(x_tensor, max_lag_ms=150, stride=1, block_size=10)
                synergies.append(s)
                lags_ms = l
                collected_count += 1
                print(f"  S{sub_id} Onsets Found: {collected_count}/{TARGET_COUNT}", end='\r')
            except Exception:
                continue
        if collected_count >= TARGET_COUNT:
            break

    if not synergies:
        return None

    mean_syn = np.mean(synergies, axis=0)
    np.save(synergy_save_path, mean_syn)
    np.save(f"{RESULT_DIR}/lags.npy", np.array(lags_ms))
    return mean_syn


def compute_subject_peak_ms(lags, subject_syn):
    norm = (subject_syn - subject_syn.min()) / (subject_syn.max() - subject_syn.min() + 1e-9)
    time_axis = -lags[::-1]
    syn_axis = gaussian_filter1d(norm[::-1], sigma=SMOOTH_SIGMA)
    mask = (np.abs(time_axis) >= EMD_VALID_MIN_MS) & (np.abs(time_axis) <= EMD_VALID_MAX_MS)
    if not np.any(mask):
        return None
    peak_idx = np.argmax(syn_axis[mask])
    return float(np.abs(time_axis[mask][peak_idx]))


def export_subject_peaks(target_subjects, lags, all_syn_list):
    peak_map = {}
    for sub_id, syn in zip(target_subjects, all_syn_list):
        peak_ms = compute_subject_peak_ms(lags, syn)
        if peak_ms is not None:
            peak_map[str(sub_id)] = round(peak_ms, 1)

    with open(PEAKS_PATH, "w") as f:
        json.dump(peak_map, f, indent=2)

    print(f"\nSaved normalized peak map to {PEAKS_PATH}")
    print(f"Peak subjects exported: {len(peak_map)}/{len(target_subjects)}")


def plot_e3_final(lags, all_syn_list):
    print("\n>>> Plotting Final Optimized Spectrum...")

    norm_syn = []
    for s in all_syn_list:
        s_min, s_max = s.min(), s.max()
        norm_syn.append((s - s_min) / (s_max - s_min + 1e-9))
    norm_syn = np.array(norm_syn)

    mean_syn = np.mean(norm_syn, axis=0)
    std_syn = np.std(norm_syn, axis=0)

    time_axis = -lags[::-1]
    mean_plot = mean_syn[::-1]
    std_plot = std_syn[::-1]

    mean_smooth = gaussian_filter1d(mean_plot, sigma=SMOOTH_SIGMA)
    std_smooth = gaussian_filter1d(std_plot, sigma=SMOOTH_SIGMA)

    plt.figure(figsize=(10, 5.5), dpi=200)
    plt.style.use('seaborn-v0_8-ticks')

    plt.axvspan(-EMD_VALID_MAX_MS, -EMD_VALID_MIN_MS, color='#e5f5e0', alpha=0.5,
                label=f'Physiological EMD Zone (-{EMD_VALID_MAX_MS} to -{EMD_VALID_MIN_MS}ms)')
    plt.fill_between(time_axis,
                     mean_smooth - std_smooth * 0.2,
                     mean_smooth + std_smooth * 0.2,
                     color='#1f77b4', alpha=0.1, label='Inter-subject Variance')
    plt.plot(time_axis, mean_smooth, color='#1f77b4', linewidth=2.5, label='ST-SRI Synergy Spectrum')

    emd_mask = (np.abs(time_axis) >= EMD_VALID_MIN_MS) & (np.abs(time_axis) <= EMD_VALID_MAX_MS)
    if np.any(emd_mask):
        peak_idx_in_mask = np.argmax(mean_smooth[emd_mask])
        peak_time = time_axis[emd_mask][peak_idx_in_mask]
        plt.axvline(peak_time, color='#d62728', linestyle='--', linewidth=1.5, alpha=0.8,
                    label=f'Detected Peak ({abs(peak_time):.1f}ms)')

    visible_mask = (time_axis < -5)
    y_limit_upper = np.max(mean_smooth[visible_mask]) * 1.3
    plt.ylim(0, y_limit_upper)

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
    target_subjects = load_target_subjects()
    all_syn_list = []
    valid_subjects = []
    print(f"Target Subjects: {target_subjects}")

    for sub in target_subjects:
        res = analyze_one_subject(sub)
        if res is not None:
            all_syn_list.append(res)
            valid_subjects.append(sub)

    if not all_syn_list:
        print("Error: No data collected.")
        return

    lags = np.load(f"{RESULT_DIR}/lags.npy")
    export_subject_peaks(valid_subjects, lags, all_syn_list)
    plot_e3_final(lags, all_syn_list)


if __name__ == '__main__':
    run_e3_main()
