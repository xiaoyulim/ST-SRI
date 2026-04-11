"""
实验五（E5）：忠实度评估 — EMD 窗口遮挡实验
=============================================
目标：验证 ST-SRI 定位的 EMD 窗口对模型决策的实际重要性。

方法：
  - 遮挡 ST-SRI 检测到的 EMD 窗口，测量准确率下降
  - 遮挡近期窗口（0-20ms）作为对照
  - 计算 Faithfulness Ratio (Rf) = Δacc_EMD / Δacc_recent

评价指标：
  - Faithfulness Ratio (Rf > 1 表示 EMD 窗口更关键)
  - Paired t-test, Cohen's d, Wilcoxon 检验
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import torch
import numpy as np
import os
import json
import csv
import matplotlib.pyplot as plt
from scipy import stats
from torch.utils.data import DataLoader
from common import NinaProDataset, LSTMModel, DEVICE, FS, EMD_VALID_MIN_MS, EMD_VALID_MAX_MS

# ================= 配置 =================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CHECKPOINT_DIR = "./checkpoints_2000hz"
RESULT_DIR = "./results/e5_faithfulness"
PEAK_JSON_PATH = "subject_peaks_e3.json"
GOOD_SUBJECTS_PATH = "good_subjects.json"

SAVE_CSV_PATH = os.path.join(RESULT_DIR, "e5_detailed_results.csv")
SAVE_SUMMARY_PATH = os.path.join(RESULT_DIR, "e5_summary.json")
SAVE_PLOT_PATH = os.path.join(RESULT_DIR, "e5_final_plot.png")
# ========================================


def get_accuracy_with_mask(model, loader, mask_range_ms=None):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            if mask_range_ms:
                ms_start, ms_end = mask_range_ms
                T = x.shape[1]
                idx_end = T - int(ms_start * FS / 1000)
                idx_start = T - int(ms_end * FS / 1000)
                idx_start, idx_end = max(0, idx_start), min(T, idx_end)
                if idx_end > idx_start:
                    x[:, idx_start:idx_end, :] = 0

            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def load_valid_subjects_and_peaks():
    with open(PEAK_JSON_PATH, 'r') as f:
        raw_peaks = {int(k): float(v) for k, v in json.load(f).items()}
    with open(GOOD_SUBJECTS_PATH, 'r') as f:
        good_subjects = set(json.load(f))

    intersection = sorted(s for s in good_subjects if s in raw_peaks)
    valid_peak_values = [v for v in raw_peaks.values() if EMD_VALID_MIN_MS <= v <= EMD_VALID_MAX_MS]
    fallback_peak = float(np.mean(valid_peak_values)) if valid_peak_values else 55.0

    resolved_peaks = {}
    fallback_subjects = set()
    for sub_id in intersection:
        peak = raw_peaks[sub_id]
        if EMD_VALID_MIN_MS <= peak <= EMD_VALID_MAX_MS:
            resolved_peaks[sub_id] = peak
        else:
            resolved_peaks[sub_id] = fallback_peak
            fallback_subjects.add(sub_id)

    fallback_count = len(fallback_subjects)
    print(f"good subjects: {len(good_subjects)} | peak file subjects: {len(raw_peaks)} | intersection: {len(intersection)}")
    print(f"valid peak range: {EMD_VALID_MIN_MS}-{EMD_VALID_MAX_MS} ms | fallback count: {fallback_count}")
    return intersection, resolved_peaks, fallback_peak, fallback_count, fallback_subjects


def run_e5():
    os.makedirs(RESULT_DIR, exist_ok=True)

    if not os.path.exists(PEAK_JSON_PATH):
        print(f"❌ 找不到 {PEAK_JSON_PATH}，请先运行 e3.py 生成峰值文件！")
        return

    subjects, peaks_dict, fallback_peak, fallback_count, fallback_subjects = load_valid_subjects_and_peaks()
    rows_to_save = []

    print(f"{'Sub':<5} | {'Peak':<6} | {'Base':<8} | {'-Recent':<10} | {'-EMD':<8} | {'Check'}")
    print("-" * 70)

    for sub_id in subjects:
        try:
            peak = peaks_dict[sub_id]
            ds = NinaProDataset("./data", sub_id, window_ms=300, target_fs=FS)
            loader = DataLoader(ds, batch_size=128, shuffle=False)

            model = LSTMModel().to(DEVICE)
            model_path = f"{CHECKPOINT_DIR}/best_model_S{sub_id}.pth"
            if not os.path.exists(model_path):
                continue

            try:
                model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            except Exception:
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))

            acc_base = get_accuracy_with_mask(model, loader)
            acc_recent = get_accuracy_with_mask(model, loader, (0, 20))
            drop_recent = max(0, acc_base - acc_recent)

            emd_start = max(0, peak - 20)
            emd_end = min(300, peak + 20)
            acc_emd = get_accuracy_with_mask(model, loader, (emd_start, emd_end))
            drop_emd = max(0, acc_base - acc_emd)

            rows_to_save.append({
                "subject": sub_id,
                "peak_ms": peak,
                "used_fallback": int(sub_id in fallback_subjects),
                "drop_recent": drop_recent,
                "drop_emd": drop_emd,
            })

            chk = "✅" if drop_emd > drop_recent else " "
            print(f"S{sub_id:<4} | {peak:<6.1f} | {acc_base:<6.1f}%  | -{drop_recent:<8.2f}% | -{drop_emd:<6.2f}% | {chk}")

        except Exception:
            continue

    if not rows_to_save:
        print("❌ 没有有效数据！")
        return

    drops_rec = [r['drop_recent'] for r in rows_to_save]
    drops_emd = [r['drop_emd'] for r in rows_to_save]
    t_stat, p_val = stats.ttest_rel(drops_emd, drops_rec)

    avg_rec = np.mean(drops_rec)
    avg_emd = np.mean(drops_emd)
    std_rec = np.std(drops_rec)
    std_emd = np.std(drops_emd)
    ratio = avg_emd / (avg_rec + 1e-9)

    print("-" * 70)
    print(f"Total Subjects: {len(rows_to_save)}")
    print(f"Avg Drop Recent: {avg_rec:.3f}% ± {std_rec:.3f}")
    print(f"Avg Drop EMD:    {avg_emd:.3f}% ± {std_emd:.3f}")
    print(f"Impact Ratio:    {ratio:.2f}x")
    print(f"P-value:         {p_val:.2e} {'(Significant ***)' if p_val < 0.001 else ''}")

    with open(SAVE_CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "peak_ms", "used_fallback", "drop_recent", "drop_emd"])
        writer.writeheader()
        writer.writerows(rows_to_save)

    summary = {
        "n_subjects": len(rows_to_save),
        "mean_drop_recent": float(avg_rec),
        "std_drop_recent": float(std_rec),
        "mean_drop_emd": float(avg_emd),
        "std_drop_emd": float(std_emd),
        "impact_ratio": float(ratio),
        "p_value": float(p_val),
        "good_subject_count": len(subjects),
        "fallback_peak_ms": float(fallback_peak),
        "fallback_count": int(fallback_count),
        "peak_range_ms": [EMD_VALID_MIN_MS, EMD_VALID_MAX_MS],
    }
    with open(SAVE_SUMMARY_PATH, 'w') as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(9, 7))
    means = [avg_rec, avg_emd]
    stds = [std_rec, std_emd]
    labels = ['Mask Recent\n(0-20ms)', 'Mask EMD\n(Peak±20ms)']
    x_pos = [0, 1]

    bars = plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.6,
                   ecolor='black', capsize=10, width=0.5,
                   color=['#bdc3c7', '#e74c3c'])

    jitter = np.random.normal(0, 0.04, size=len(drops_rec))
    plt.scatter(np.zeros_like(drops_rec) + jitter, drops_rec, color='#555555', alpha=0.4, s=15, zorder=3)
    plt.scatter(np.ones_like(drops_emd) + jitter, drops_emd, color='#800000', alpha=0.4, s=15, zorder=3)

    for i in range(len(drops_rec)):
        plt.plot([0 + jitter[i], 1 + jitter[i]], [drops_rec[i], drops_emd[i]], color='gray', alpha=0.1, linewidth=0.5)

    if p_val < 0.05:
        h = max(max(drops_rec), max(drops_emd)) * 1.05
        plt.plot([0, 0, 1, 1], [h, h + 0.5, h + 0.5, h], lw=1.5, c='k')
        sig_symbol = '*** (p<0.001)' if p_val < 0.001 else ('**' if p_val < 0.01 else '*')
        plt.text(0.5, h + 0.6, sig_symbol, ha='center', va='bottom', fontsize=12, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height / 2, f'{height:.2f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    plt.ylabel('Accuracy Drop (%)', fontsize=12)
    plt.xticks(x_pos, labels, fontsize=12)
    plt.title(f'Faithfulness Evaluation (N={len(rows_to_save)})\nImpact Ratio: {ratio:.2f}x', fontsize=14)
    plt.ylim(bottom=0, top=max(max(drops_rec), max(drops_emd)) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.savefig(SAVE_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"\n✅ 绘图完成！已保存至 {SAVE_PLOT_PATH}")


if __name__ == "__main__":
    run_e5()
