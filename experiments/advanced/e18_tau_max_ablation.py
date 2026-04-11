"""
实验十八：τ_max 消融扫描
=========================
目标：证明 ST-SRI 检测到的协同峰位置不随 τ_max 的选择而改变，
      排除"方法被调参到特定区间"的循环论证风险。

审稿人关注点（PC#3）：
  "τ_max=150ms implicitly tuned to find peaks in 30-100ms;
   a more agnostic τ_max sweep would strengthen trust"

方法：
  - 对 τ_max = {100, 150, 200, 300, 500} ms 分别运行 E3 协同谱扫描
  - 统计每组的协同峰位置分布
  - 验证峰值位置不随 τ_max 改变
  - 记录计算时间
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import json
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d
from scipy import stats

from common import (
    NinaProDataset, LSTMModel, ST_SRI_Interpreter,
    DEVICE, FS, EMD_VALID_MIN_MS, EMD_VALID_MAX_MS,
)

# ================= Configuration =================
DATA_ROOT = "./data"
CHECKPOINT_DIR = "./checkpoints_2000hz"
GOOD_SUBJECTS_PATH = "./good_subjects.json"
RESULT_DIR = "./results/tau_max_ablation"

TAU_MAX_VALUES = [100, 150, 200, 300, 500]  # ms
NUM_SAMPLES = 30
SMOOTH_SIGMA = 2.0
SCAN_STRIDE = 1
BLOCK_SIZE = 2
# =================================================

os.makedirs(RESULT_DIR, exist_ok=True)


def load_subjects(limit=None):
    with open(GOOD_SUBJECTS_PATH, "r") as f:
        subjects = json.load(f)
    return subjects[:limit] if limit is not None else subjects


def load_model(sub_id):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_model_S{sub_id}.pth")
    if not os.path.exists(ckpt_path):
        return None
    model = LSTMModel(input_size=12).to(DEVICE)
    try:
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def scan_subject(model, ds, tau_max_ms, num_samples=NUM_SAMPLES):
    """对单个 subject 运行 ST-SRI 扫描，返回协同峰位置和计时。"""
    bg_loader = DataLoader(ds, batch_size=min(20, len(ds)), shuffle=True)
    bg_batch = next(iter(bg_loader))[0].to(DEVICE)
    interp = ST_SRI_Interpreter(model, bg_batch)

    indices = np.linspace(0, len(ds) - 1, num_samples, dtype=int)

    all_synergy = []
    t_start = time.time()

    for idx in indices:
        x, _ = ds[idx]
        x = x.to(DEVICE)
        lags_ms, synergy, _ = interp.scan_fast(
            x, max_lag_ms=tau_max_ms, stride=SCAN_STRIDE, block_size=BLOCK_SIZE
        )
        if len(synergy) > 0:
            all_synergy.append(synergy)

    elapsed = time.time() - t_start

    if not all_synergy:
        return None, None, elapsed

    lags_ms = np.array(lags_ms)
    # 对齐长度
    min_len = min(len(s) for s in all_synergy)
    all_synergy = np.array([s[:min_len] for s in all_synergy])
    lags_ms = lags_ms[:min_len]

    mean_synergy = np.mean(all_synergy, axis=0)
    smooth_synergy = gaussian_filter1d(mean_synergy, sigma=SMOOTH_SIGMA)

    peak_idx = np.argmax(smooth_synergy)
    peak_ms = float(lags_ms[peak_idx])

    return peak_ms, smooth_synergy, elapsed


def main():
    parser = argparse.ArgumentParser(description="E18: tau_max ablation sweep")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only use first N subjects")
    parser.add_argument("--tau-values", type=int, nargs="+", default=TAU_MAX_VALUES,
                        help="tau_max values to sweep")
    args = parser.parse_args()

    subjects = load_subjects(limit=args.limit)
    tau_values = args.tau_values

    print("=" * 65)
    print("E18: tau_max Ablation Sweep")
    print("=" * 65)
    print(f"Subjects: {len(subjects)}")
    print(f"tau_max values: {tau_values} ms")
    print()

    # 收集结果
    results = {str(tau): {} for tau in tau_values}
    timing = {str(tau): [] for tau in tau_values}

    for sub_id in subjects:
        model = load_model(sub_id)
        if model is None:
            continue

        ds = NinaProDataset(DATA_ROOT, sub_id, window_ms=300, target_fs=2000, step_ms=50)
        print(f"  S{sub_id}:", end="")

        for tau_max in tau_values:
            peak_ms, _, elapsed = scan_subject(model, ds, tau_max)
            if peak_ms is not None:
                in_emd = EMD_VALID_MIN_MS <= peak_ms <= EMD_VALID_MAX_MS
                results[str(tau_max)][str(sub_id)] = {
                    'peak_ms': peak_ms,
                    'in_emd': in_emd,
                }
                timing[str(tau_max)].append(elapsed)
                print(f"  τ={tau_max}→{peak_ms:.0f}ms", end="")
        print()

    # 统计汇总
    print("\n" + "=" * 65)
    print(f"{'tau_max':>8} | {'Peak Mean':>10} | {'Peak Std':>9} | {'Median':>8} | {'In-EMD%':>8} | {'Time(s)':>8}")
    print("-" * 65)

    summary = {}
    for tau in tau_values:
        tau_key = str(tau)
        if not results[tau_key]:
            continue

        peaks = [v['peak_ms'] for v in results[tau_key].values()]
        in_emd = sum(1 for v in results[tau_key].values() if v['in_emd'])
        n = len(peaks)
        avg_time = np.mean(timing[tau_key]) if timing[tau_key] else 0

        summary[tau_key] = {
            'tau_max_ms': tau,
            'n_subjects': n,
            'peak_mean': float(np.mean(peaks)),
            'peak_std': float(np.std(peaks)),
            'peak_median': float(np.median(peaks)),
            'in_emd_ratio': in_emd / n,
            'avg_time_s': float(avg_time),
        }

        print(f"{tau:>8} | {np.mean(peaks):>10.1f} | {np.std(peaks):>9.1f} | "
              f"{np.median(peaks):>8.1f} | {in_emd/n*100:>7.1f}% | {avg_time:>8.1f}")

    # 稳定性检验：峰值位置在不同 τ_max 之间是否一致
    print("\n--- Stability Test (Kruskal-Wallis) ---")
    peak_groups = []
    for tau in tau_values:
        tau_key = str(tau)
        if tau_key in results and results[tau_key]:
            peak_groups.append([v['peak_ms'] for v in results[tau_key].values()])

    if len(peak_groups) >= 2:
        h_stat, p_value = stats.kruskal(*peak_groups)
        print(f"  H-statistic: {h_stat:.4f}")
        print(f"  p-value:     {p_value:.4f}")
        if p_value > 0.05:
            print(f"  Conclusion:  Peak positions are STABLE across tau_max (p={p_value:.3f} > 0.05)")
        else:
            print(f"  Conclusion:  Peak positions DIFFER across tau_max (p={p_value:.3f} < 0.05)")
        summary['kruskal_wallis'] = {'h_stat': float(h_stat), 'p_value': float(p_value)}

    # 保存
    payload = {
        'per_subject': results,
        'summary': summary,
        'config': {
            'tau_values': tau_values,
            'num_samples': NUM_SAMPLES,
            'scan_stride': SCAN_STRIDE,
            'block_size': BLOCK_SIZE,
        }
    }
    with open(os.path.join(RESULT_DIR, 'tau_max_ablation.json'), 'w') as f:
        json.dump(payload, f, indent=2)

    # 绘图
    plot_ablation(results, summary, tau_values)
    print(f"\nResults saved to {RESULT_DIR}/")


def plot_ablation(results, summary, tau_values):
    """生成 τ_max 消融图。"""
    taus = [t for t in tau_values if str(t) in summary]
    if len(taus) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. 峰值位置分布 (box plot)
    peak_data = []
    for tau in taus:
        peaks = [v['peak_ms'] for v in results[str(tau)].values()]
        peak_data.append(peaks)

    bp = axes[0].boxplot(peak_data, labels=[f'{t}ms' for t in taus],
                         patch_artist=True, widths=0.5)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(taus)))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    axes[0].axhspan(EMD_VALID_MIN_MS, EMD_VALID_MAX_MS, alpha=0.15, color='green',
                    label='EMD range')
    axes[0].set_xlabel('tau_max')
    axes[0].set_ylabel('Synergy Peak (ms)')
    axes[0].set_title('Peak Position vs tau_max')
    axes[0].legend(fontsize=8)

    # 2. In-EMD ratio
    emd_ratios = [summary[str(t)]['in_emd_ratio'] * 100 for t in taus]
    axes[1].plot(taus, emd_ratios, 'o-', linewidth=2, markersize=8, color='#2196F3')
    axes[1].set_xlabel('tau_max (ms)')
    axes[1].set_ylabel('Subjects in EMD range (%)')
    axes[1].set_title('EMD Detection Rate vs tau_max')
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.3)

    # 3. 计算时间
    times = [summary[str(t)]['avg_time_s'] for t in taus]
    axes[2].plot(taus, times, 's-', linewidth=2, markersize=8, color='#FF5722')
    axes[2].set_xlabel('tau_max (ms)')
    axes[2].set_ylabel('Avg Time per Subject (s)')
    axes[2].set_title('Computation Cost vs tau_max')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'tau_max_ablation.png'), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
