"""
E5 增强版：三组忠实度评估（Faithfulness with Random Control）
=============================================================
在原 E5 基础上新增"随机位置"对照组，使 Faithfulness Ratio (Rf) 更严格：

  组别：
    Mask Recent  (0–20ms)           — 对照1：最近时段
    Mask Random  (随机 40ms 窗口)   — 对照2：随机位置（真随机基线）
    Mask EMD     (peak ± 20ms)      — 实验组：ST-SRI 检测到的协同峰

  增强后 Rf：
    Rf_recent  = ΔPerf_EMD / ΔPerf_Recent
    Rf_random  = ΔPerf_EMD / ΔPerf_Random   ← 更严格的无偏估计

输出：
  results/e5_faithfulness/e5_enhanced_results.csv
  results/e5_faithfulness/e5_enhanced_summary.json
  results/e5_faithfulness/e5_enhanced_plot.png
"""

import torch
import numpy as np
import os
import json
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from scipy import stats
from torch.utils.data import DataLoader

from common import NinaProDataset, LSTMModel, DEVICE, FS, calculate_cohens_d, interpret_cohens_d, bootstrap_ci

# ================= 配置 =================
CHECKPOINT_DIR = "./checkpoints_2000hz"
RESULT_DIR = "./results/e5_faithfulness"
PEAK_JSON_PATH = "subject_peaks_e3.json"
GOOD_SUBJECTS_PATH = "./good_subjects.json"

SAVE_CSV_PATH = os.path.join(RESULT_DIR, "e5_enhanced_results.csv")
SAVE_SUMMARY_PATH = os.path.join(RESULT_DIR, "e5_enhanced_summary.json")
SAVE_PLOT_PATH = os.path.join(RESULT_DIR, "e5_enhanced_plot.png")

MASK_HALF_WIDTH_MS = 20    # 遮挡窗口半宽：peak ± 20ms（共 40ms）
N_RANDOM_DRAWS = 10        # 每受试者随机遮挡重复次数（取均值以降低方差）
WINDOW_MS = 300
RANDOM_EXCL_MARGIN_MS = 5  # 随机位置避开 EMD 峰值和 recent 的安全边距


def get_accuracy_with_mask(model, loader, mask_range_ms=None):
    """mask_range_ms: (start_ms, end_ms) 从窗口末尾倒数，None 表示不遮挡"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            if mask_range_ms is not None:
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


def sample_random_mask_center(peak_ms, rng, window_ms=WINDOW_MS,
                               half_width=MASK_HALF_WIDTH_MS,
                               margin=RANDOM_EXCL_MARGIN_MS):
    """
    在 [half_width, window_ms - half_width] 范围内随机选取遮挡中心，
    排除 recent (0–20ms) 和 EMD (peak ± half_width + margin) 两个区域。
    """
    excl_recent = (0, 20 + half_width + margin)                        # 排除近期区
    excl_emd = (peak_ms - half_width - margin, peak_ms + half_width + margin)  # 排除 EMD 区

    valid_centers = []
    for c in range(half_width, window_ms - half_width + 1):
        in_recent = excl_recent[0] <= c <= excl_recent[1]
        in_emd = excl_emd[0] <= c <= excl_emd[1]
        if not in_recent and not in_emd:
            valid_centers.append(c)

    if not valid_centers:
        # 极少数情况下无有效位置（peak 很靠近边界），退而取中间
        return window_ms // 2
    return rng.choice(valid_centers)


def run_e5_enhanced():
    os.makedirs(RESULT_DIR, exist_ok=True)
    rng = np.random.RandomState(42)

    with open(PEAK_JSON_PATH) as f:
        peaks_dict = json.load(f)
    with open(GOOD_SUBJECTS_PATH) as f:
        good_subjects = set(json.load(f))

    subjects = sorted([int(k) for k in peaks_dict.keys() if int(k) in good_subjects])
    rows = []

    header = f"{'Sub':<5} | {'Peak':>6} | {'Base':>8} | {'-Recent':>10} | {'-Random':>10} | {'-EMD':>10} | {'EMD>Rnd?'}"
    print(header)
    print("-" * 75)

    for sub_id in subjects:
        try:
            peak = float(peaks_dict[str(sub_id)])
            model_path = f"{CHECKPOINT_DIR}/best_model_S{sub_id}.pth"
            if not os.path.exists(model_path):
                continue

            ds = NinaProDataset("./data", sub_id, window_ms=WINDOW_MS, target_fs=FS)
            loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

            model = LSTMModel().to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))

            # 1. Baseline
            acc_base = get_accuracy_with_mask(model, loader)

            # 2. Recent control (0–20ms)
            acc_recent = get_accuracy_with_mask(model, loader, (0, 20))
            drop_recent = max(0.0, acc_base - acc_recent)

            # 3. Random control (平均 N_RANDOM_DRAWS 次)
            acc_random_list = []
            for _ in range(N_RANDOM_DRAWS):
                c = sample_random_mask_center(peak, rng)
                mask = (c - MASK_HALF_WIDTH_MS, c + MASK_HALF_WIDTH_MS)
                acc_random_list.append(get_accuracy_with_mask(model, loader, mask))
            acc_random = float(np.mean(acc_random_list))
            drop_random = max(0.0, acc_base - acc_random)

            # 4. EMD (peak ± 20ms)
            emd_start = max(0.0, peak - MASK_HALF_WIDTH_MS)
            emd_end = min(float(WINDOW_MS), peak + MASK_HALF_WIDTH_MS)
            acc_emd = get_accuracy_with_mask(model, loader, (emd_start, emd_end))
            drop_emd = max(0.0, acc_base - acc_emd)

            rows.append({
                "subject": sub_id, "peak_ms": peak,
                "acc_base": acc_base,
                "drop_recent": drop_recent,
                "drop_random": drop_random,
                "drop_emd": drop_emd,
            })

            marker = "✅" if drop_emd > drop_random else "  "
            print(f"S{sub_id:<4} | {peak:>6.1f} | {acc_base:>6.1f}%  | "
                  f"-{drop_recent:>8.2f}% | -{drop_random:>8.2f}% | -{drop_emd:>8.2f}% | {marker}")

        except Exception as e:
            print(f"S{sub_id}: 错误 - {e}")
            continue

    if not rows:
        print("❌ 无有效数据")
        return

    # ===== 统计汇总 =====
    drops_rec = np.array([r['drop_recent'] for r in rows])
    drops_rnd = np.array([r['drop_random'] for r in rows])
    drops_emd = np.array([r['drop_emd'] for r in rows])
    N = len(rows)

    # 配对 t 检验
    t_er, p_er = stats.ttest_rel(drops_emd, drops_rec)
    t_rand, p_rand = stats.ttest_rel(drops_emd, drops_rnd)

    # Wilcoxon
    w_er, pw_er = stats.wilcoxon(drops_emd - drops_rec)
    w_rand, pw_rand = stats.wilcoxon(drops_emd - drops_rnd)

    # Cohen's d
    d_er = calculate_cohens_d(drops_emd.tolist(), drops_rec.tolist())
    d_rand = calculate_cohens_d(drops_emd.tolist(), drops_rnd.tolist())

    # Bootstrap CI on mean diff
    ci_er = bootstrap_ci((drops_emd - drops_rec).tolist(), n_bootstrap=5000)
    ci_rand = bootstrap_ci((drops_emd - drops_rnd).tolist(), n_bootstrap=5000)

    rf_recent = float(np.mean(drops_emd)) / (float(np.mean(drops_rec)) + 1e-9)
    rf_random = float(np.mean(drops_emd)) / (float(np.mean(drops_rnd)) + 1e-9)

    print("\n" + "=" * 75)
    print(f"受试者数 N = {N}")
    print(f"\n{'组别':<20} {'均值 (%)':>10} {'标准差':>10}")
    print(f"{'Mask Recent':.<20} {np.mean(drops_rec):>10.3f} {np.std(drops_rec):>10.3f}")
    print(f"{'Mask Random':.<20} {np.mean(drops_rnd):>10.3f} {np.std(drops_rnd):>10.3f}")
    print(f"{'Mask EMD':.<20} {np.mean(drops_emd):>10.3f} {np.std(drops_emd):>10.3f}")
    print(f"\nRf_recent  = {rf_recent:.3f}  (EMD vs Recent)")
    print(f"Rf_random  = {rf_random:.3f}  (EMD vs Random, 更严格)")
    print(f"\nEMD vs Recent: t={t_er:.3f}, p={p_er:.3e}, W={w_er:.0f}, pw={pw_er:.3e}, d={d_er:.3f} ({interpret_cohens_d(d_er)})")
    print(f"  95% CI diff: [{ci_er[0]:+.3f}, {ci_er[1]:+.3f}]%")
    print(f"EMD vs Random: t={t_rand:.3f}, p={p_rand:.3e}, W={w_rand:.0f}, pw={pw_rand:.3e}, d={d_rand:.3f} ({interpret_cohens_d(d_rand)})")
    print(f"  95% CI diff: [{ci_rand[0]:+.3f}, {ci_rand[1]:+.3f}]%")

    # ===== 保存 CSV =====
    with open(SAVE_CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  CSV → {SAVE_CSV_PATH}")

    # ===== 保存 JSON =====
    summary = {
        "n_subjects": N,
        "mean_drop_recent": float(np.mean(drops_rec)),
        "std_drop_recent": float(np.std(drops_rec)),
        "mean_drop_random": float(np.mean(drops_rnd)),
        "std_drop_random": float(np.std(drops_rnd)),
        "mean_drop_emd": float(np.mean(drops_emd)),
        "std_drop_emd": float(np.std(drops_emd)),
        "Rf_recent": rf_recent,
        "Rf_random": rf_random,
        "emd_vs_recent": {
            "t_stat": float(t_er), "p_ttest": float(p_er),
            "wilcoxon_W": float(w_er), "p_wilcoxon": float(pw_er),
            "cohens_d": float(d_er), "effect_size": interpret_cohens_d(d_er),
            "ci_95_lower": float(ci_er[0]), "ci_95_upper": float(ci_er[1]),
        },
        "emd_vs_random": {
            "t_stat": float(t_rand), "p_ttest": float(p_rand),
            "wilcoxon_W": float(w_rand), "p_wilcoxon": float(pw_rand),
            "cohens_d": float(d_rand), "effect_size": interpret_cohens_d(d_rand),
            "ci_95_lower": float(ci_rand[0]), "ci_95_upper": float(ci_rand[1]),
        },
        "n_random_draws": N_RANDOM_DRAWS,
    }
    with open(SAVE_SUMMARY_PATH, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON → {SAVE_SUMMARY_PATH}")

    # ===== 绘图：三组对比 =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # —— 左：Bar + Strip（三组均值 + 个体散点）——
    ax = axes[0]
    groups = [drops_rec, drops_rnd, drops_emd]
    labels = ['Mask Recent\n(0–20ms)', f'Mask Random\n(avg N={N_RANDOM_DRAWS})', 'Mask EMD\n(Peak±20ms)']
    colors_bar = ['#9e9e9e', '#64b5f6', '#e53935']
    colors_dot = ['#424242', '#1565c0', '#7f0000']
    x_pos = [0, 1, 2]

    bars = ax.bar(x_pos, [np.mean(g) for g in groups],
                  yerr=[np.std(g) / np.sqrt(N) for g in groups],
                  align='center', alpha=0.65,
                  ecolor='black', capsize=8, width=0.5,
                  color=colors_bar)

    rng_jit = np.random.RandomState(7)
    for xi, (g, col) in enumerate(zip(groups, colors_dot)):
        jitter = rng_jit.normal(0, 0.06, len(g))
        ax.scatter(np.full(len(g), xi) + jitter, g,
                   color=col, alpha=0.5, s=20, zorder=3)

    # 配对连线（EMD vs Random）
    for i in range(N):
        ax.plot([1, 2], [drops_rnd[i], drops_emd[i]],
                color='gray', alpha=0.08, linewidth=0.7)

    # 显著性括号（EMD vs Random）
    y_sig = max(np.max(drops_emd), np.max(drops_rnd)) * 1.08
    ax.plot([1, 1, 2, 2], [y_sig, y_sig + 0.3, y_sig + 0.3, y_sig], lw=1.5, c='k')
    sig_sym = '***' if p_rand < 0.001 else ('**' if p_rand < 0.01 else ('*' if p_rand < 0.05 else 'n.s.'))
    ax.text(1.5, y_sig + 0.4, f'{sig_sym}\n(p={p_rand:.2e})', ha='center', fontsize=9)

    # 数值标注
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h / 2,
                f'{h:.2f}%', ha='center', va='center',
                color='white', fontweight='bold', fontsize=11)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title(f'Faithfulness Evaluation (N={N})\n'
                 f'Rf_recent={rf_recent:.2f}x  |  Rf_random={rf_random:.2f}x', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # —— 右：每受试者 drop_emd - drop_random（差异分布）——
    ax2 = axes[1]
    diff_emd_rnd = drops_emd - drops_rnd
    colors_sub = ['#e53935' if d > 0 else '#1565c0' for d in diff_emd_rnd]
    sub_labels = [f"S{r['subject']}" for r in rows]
    bars2 = ax2.bar(range(N), diff_emd_rnd, color=colors_sub, alpha=0.7, width=0.7)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.axhline(np.mean(diff_emd_rnd), color='red', linewidth=1.5, linestyle='--',
                label=f'Mean = {np.mean(diff_emd_rnd):+.2f}%')
    ax2.set_xticks(range(N))
    ax2.set_xticklabels(sub_labels, rotation=45, ha='right', fontsize=7)
    ax2.set_ylabel('Δ(drop_EMD − drop_Random) (%)', fontsize=11)
    ax2.set_title('Per-Subject: EMD Drop − Random Drop\n'
                  f'(+Red = EMD more critical, −Blue = Random more critical)', fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"\n  图像 → {SAVE_PLOT_PATH}")
    plt.close()


if __name__ == "__main__":
    run_e5_enhanced()
