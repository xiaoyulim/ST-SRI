"""
实验十五：AOPC 忠实度评估（修复版 + 分层分析）
================================================
修复项：
  1. SHAP AOPC=0 bug：原因是 SHAP 归因维度 (T=600) 与 ST-SRI (T_lag≈100)
     不一致，导致排序后的遮挡区域无法正确定位。现统一在原始时间维度上操作。
  2. 样本量从 15 → 50
  3. 受试者从 5 → 全部 39
  4. 新增按模型精度分层分析（高/中/低）

审稿人关注点：
  PC#3: "7 subjects (18%) have faithfulness ratio <1.0;
         S1 (81.25% accuracy) is not explained"
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import shap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import argparse
from torch.utils.data import DataLoader, Dataset
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from common import (
    NinaProDataset, LSTMModel, ST_SRI_Interpreter,
    DEVICE, FS, EMD_VALID_MIN_MS, EMD_VALID_MAX_MS,
    create_blocked_split,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================= Configuration =================
DATA_ROOT = "./data"
CHECKPOINT_DIR = "./checkpoints_2000hz"
GOOD_SUBJECTS_PATH = "./good_subjects.json"
RESULT_DIR = "./results/aopc"

NUM_SAMPLES = 50           # 每 subject 用于 AOPC 的样本数
PERTURBATION_STEPS = 20    # 扰动步数
WINDOW_LEN = 600           # 300ms @ 2kHz

# 分层阈值
ACC_HIGH = 85.0
ACC_LOW = 70.0
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


def get_baseline_accuracy(model, ds):
    """计算模型在验证集上的无遮挡基线准确率。"""
    _, val_ds = create_blocked_split(ds, train_ratio=0.8)
    loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


# ================= 归因计算（统一输出 T=600 维时间归因） =================

def compute_st_sri_temporal(model, samples, bg_data):
    """
    ST-SRI 归因：先在 lag 空间得到协同谱，再映射回原始时间维度。
    输出: (T=600,) 归一化归因向量。
    """
    interpreter = ST_SRI_Interpreter(model, bg_data)
    all_synergy = []

    for i in range(len(samples)):
        try:
            lags_ms, synergy, _ = interpreter.scan_fast(
                samples[i], max_lag_ms=150, stride=1, block_size=2
            )
            if len(synergy) > 0:
                all_synergy.append(synergy)
        except Exception:
            continue

    if not all_synergy:
        return None

    # lag 空间平均
    min_len = min(len(s) for s in all_synergy)
    mean_synergy = np.mean([s[:min_len] for s in all_synergy], axis=0)
    lags_ms = np.array(lags_ms[:min_len])

    # 映射回 T=600 的时间维度
    T = WINDOW_LEN
    temporal = np.zeros(T)
    for j, lag_ms in enumerate(lags_ms):
        lag_samples = int(lag_ms * FS / 1000)
        t_idx = T - 1 - lag_samples
        if 0 <= t_idx < T:
            temporal[t_idx] = mean_synergy[j]

    # 平滑 + 归一化
    temporal = gaussian_filter1d(temporal, sigma=3.0)
    s = temporal.sum()
    if s > 1e-9:
        temporal /= s
    return temporal


def compute_shap_temporal(model, samples, bg_data):
    """
    Standard SHAP (GradientExplainer)。
    输出: (T=600,) 归一化归因向量。
    """
    try:
        explainer = shap.GradientExplainer(model, bg_data)
        with torch.backends.cudnn.flags(enabled=False):
            shap_vals = explainer.shap_values(samples)

        if isinstance(shap_vals, list):
            # multi-class: (n_classes, N, T, C)
            attr = np.abs(np.array(shap_vals)).sum(axis=0).sum(axis=-1)  # (N, T)
        else:
            attr = np.abs(shap_vals).sum(axis=-1)  # (N, T)

        mean_attr = np.mean(attr, axis=0)  # (T,)
        s = mean_attr.sum()
        if s > 1e-9:
            mean_attr /= s
        return mean_attr
    except Exception as e:
        print(f"      SHAP error: {e}")
        return None


def compute_random_temporal():
    """随机归因基线。"""
    attr = np.random.rand(WINDOW_LEN)
    return attr / attr.sum()


# ================= AOPC 计算 =================

class TensorDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.y[i]


def compute_aopc(model, samples, labels, attribution, steps=PERTURBATION_STEPS):
    """
    计算 AOPC：按归因从高到低逐步遮挡，记录准确率下降曲线。
    """
    T = len(attribution)
    sorted_indices = np.argsort(attribution)[::-1]  # 最重要的排前面
    step_size = max(1, T // steps)

    eval_ds = TensorDataset(samples, labels)
    loader = DataLoader(eval_ds, batch_size=64, shuffle=False)

    # 基线准确率
    model.eval()
    def eval_with_mask(mask_idx):
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE).long()
                if mask_idx:
                    x = x.clone()
                    x[:, mask_idx, :] = 0
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        return 100.0 * correct / total if total > 0 else 0.0

    acc_base = eval_with_mask(None)
    curve = [acc_base]
    masked = []

    for step in range(1, steps + 1):
        start = (step - 1) * step_size
        end = min(step * step_size, T)
        masked.extend(sorted_indices[start:end].tolist())
        curve.append(eval_with_mask(masked))

    curve = np.array(curve)
    aopc = float(np.mean(acc_base - curve))
    return aopc, curve.tolist()


# ================= 单 Subject 分析 =================

def analyze_subject(sub_id):
    """对单个 subject 计算 AOPC。"""
    model = load_model(sub_id)
    if model is None:
        return None

    try:
        ds = NinaProDataset(DATA_ROOT, sub_id, window_ms=300, target_fs=2000, step_ms=50)
    except Exception:
        return None

    # 基线准确率
    base_acc = get_baseline_accuracy(model, ds)

    # 采样
    np.random.seed(42)
    n = min(NUM_SAMPLES, len(ds))
    indices = np.random.choice(len(ds), n, replace=False)
    samples = torch.stack([ds[i][0] for i in indices])
    labels = torch.tensor([ds[i][1] for i in indices])

    # 背景数据
    bg_loader = DataLoader(ds, batch_size=min(20, len(ds)), shuffle=True)
    bg_data = next(iter(bg_loader))[0].to(DEVICE)

    samples_device = samples.to(DEVICE)

    # 归因
    sri_attr = compute_st_sri_temporal(model, samples_device, bg_data)
    shap_attr = compute_shap_temporal(model, samples_device, bg_data)
    rand_attr = compute_random_temporal()

    if sri_attr is None:
        return None

    # AOPC
    aopc_sri, curve_sri = compute_aopc(model, samples, labels, sri_attr)

    result = {
        'subject': sub_id,
        'base_acc': base_acc,
        'aopc_st_sri': aopc_sri,
        'curve_st_sri': curve_sri,
    }

    if shap_attr is not None:
        aopc_shap, curve_shap = compute_aopc(model, samples, labels, shap_attr)
        result['aopc_shap'] = aopc_shap
        result['curve_shap'] = curve_shap
    else:
        result['aopc_shap'] = 0.0

    aopc_rand, curve_rand = compute_aopc(model, samples, labels, rand_attr)
    result['aopc_random'] = aopc_rand
    result['curve_random'] = curve_rand

    # Faithfulness ratio: ST-SRI / Random
    result['rf_vs_random'] = aopc_sri / aopc_rand if aopc_rand > 1e-6 else float('inf')

    return result


# ================= 主流程 =================

def main():
    parser = argparse.ArgumentParser(description="E15: AOPC Faithfulness (Fixed)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    subjects = load_subjects(limit=args.limit)

    print("=" * 70)
    print("E15: AOPC Faithfulness Evaluation (Fixed + Stratified)")
    print("=" * 70)
    print(f"Subjects: {len(subjects)}, Samples/subject: {NUM_SAMPLES}")
    print()

    results = []
    for sub_id in subjects:
        print(f"  S{sub_id}...", end=" ")
        res = analyze_subject(sub_id)
        if res is not None:
            results.append(res)
            print(f"acc={res['base_acc']:.1f}% | AOPC: SRI={res['aopc_st_sri']:.2f}, "
                  f"SHAP={res['aopc_shap']:.2f}, Rand={res['aopc_random']:.2f} | "
                  f"Rf={res['rf_vs_random']:.2f}")
        else:
            print("SKIP")

    if not results:
        print("No results!")
        return

    # === 总体汇总 ===
    print("\n" + "=" * 70)
    print("Overall AOPC Summary (higher = more faithful)")
    print("-" * 70)
    for key, label in [('aopc_st_sri', 'ST-SRI'), ('aopc_shap', 'SHAP'), ('aopc_random', 'Random')]:
        vals = [r[key] for r in results]
        print(f"  {label:<10}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}")

    # === 分层分析 ===
    print("\n" + "=" * 70)
    print("Stratified Analysis by Model Accuracy")
    print("-" * 70)

    tiers = {
        f'High (>{ACC_HIGH}%)': [r for r in results if r['base_acc'] > ACC_HIGH],
        f'Mid ({ACC_LOW}-{ACC_HIGH}%)': [r for r in results if ACC_LOW <= r['base_acc'] <= ACC_HIGH],
        f'Low (<{ACC_LOW}%)': [r for r in results if r['base_acc'] < ACC_LOW],
    }

    for tier_name, tier_results in tiers.items():
        if not tier_results:
            print(f"\n  {tier_name}: 0 subjects")
            continue

        sri_vals = [r['aopc_st_sri'] for r in tier_results]
        rf_vals = [r['rf_vs_random'] for r in tier_results if r['rf_vs_random'] < 100]
        n_rf_below_1 = sum(1 for r in tier_results if r['rf_vs_random'] < 1.0)
        subs = [r['subject'] for r in tier_results]

        print(f"\n  {tier_name}: {len(tier_results)} subjects {subs}")
        print(f"    AOPC(SRI): {np.mean(sri_vals):.3f} +/- {np.std(sri_vals):.3f}")
        print(f"    Rf(mean):  {np.mean(rf_vals):.2f}" if rf_vals else "    Rf: N/A")
        print(f"    Rf < 1.0:  {n_rf_below_1}/{len(tier_results)}")

    # === 失败案例分析 ===
    failures = [r for r in results if r['rf_vs_random'] < 1.0]
    if failures:
        print(f"\n{'=' * 70}")
        print(f"Failure Cases (Rf < 1.0): {len(failures)} subjects")
        print("-" * 70)
        for r in sorted(failures, key=lambda x: x['rf_vs_random']):
            print(f"  S{r['subject']}: acc={r['base_acc']:.1f}%, "
                  f"AOPC(SRI)={r['aopc_st_sri']:.2f}, "
                  f"AOPC(Rand)={r['aopc_random']:.2f}, "
                  f"Rf={r['rf_vs_random']:.2f}")

    # 保存
    payload = {
        'results': results,
        'config': {
            'num_samples': NUM_SAMPLES,
            'perturbation_steps': PERTURBATION_STEPS,
            'acc_high': ACC_HIGH,
            'acc_low': ACC_LOW,
        },
        'summary': {
            'n_subjects': len(results),
            'aopc_sri_mean': float(np.mean([r['aopc_st_sri'] for r in results])),
            'aopc_shap_mean': float(np.mean([r['aopc_shap'] for r in results])),
            'aopc_random_mean': float(np.mean([r['aopc_random'] for r in results])),
            'n_rf_below_1': len(failures),
        }
    }
    with open(os.path.join(RESULT_DIR, 'aopc_results.json'), 'w') as f:
        json.dump(payload, f, indent=2, default=str)

    # 绘图
    plot_results(results, tiers)
    print(f"\nResults saved to {RESULT_DIR}/")


def plot_results(results, tiers):
    """生成 AOPC 对比图 + 分层散点图。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 总体 AOPC 柱状图
    methods = ['ST-SRI', 'SHAP', 'Random']
    keys = ['aopc_st_sri', 'aopc_shap', 'aopc_random']
    colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c']
    means = [np.mean([r[k] for r in results]) for k in keys]
    stds = [np.std([r[k] for r in results]) for k in keys]

    x = np.arange(len(methods))
    axes[0].bar(x, means, yerr=stds, color=colors_bar, alpha=0.7, capsize=5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylabel('AOPC Score')
    axes[0].set_title(f'AOPC Comparison (N={len(results)})', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # 2. Rf vs 模型精度散点图
    accs = [r['base_acc'] for r in results]
    rfs = [min(r['rf_vs_random'], 5.0) for r in results]  # 截断极端值
    axes[1].scatter(accs, rfs, c='#1f77b4', alpha=0.6, edgecolors='k', linewidths=0.5, s=40)
    axes[1].axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Rf=1.0')
    axes[1].axvline(ACC_HIGH, color='gray', linestyle=':', alpha=0.4)
    axes[1].axvline(ACC_LOW, color='gray', linestyle=':', alpha=0.4)
    axes[1].set_xlabel('Model Accuracy (%)')
    axes[1].set_ylabel('Faithfulness Ratio (Rf)')
    axes[1].set_title('Rf vs Model Accuracy (Stratified)', fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    # 3. 分层 AOPC 柱状图
    tier_names = list(tiers.keys())
    tier_sri = []
    tier_n = []
    for tn in tier_names:
        tr = tiers[tn]
        if tr:
            tier_sri.append(np.mean([r['aopc_st_sri'] for r in tr]))
        else:
            tier_sri.append(0)
        tier_n.append(len(tr))

    colors_tier = ['#4CAF50', '#FF9800', '#F44336']
    bars = axes[2].bar(range(len(tier_names)), tier_sri,
                       color=colors_tier, alpha=0.7)
    axes[2].set_xticks(range(len(tier_names)))
    axes[2].set_xticklabels([f'{tn}\n(n={n})' for tn, n in zip(tier_names, tier_n)],
                            fontsize=8)
    axes[2].set_ylabel('AOPC (ST-SRI)')
    axes[2].set_title('AOPC by Accuracy Tier', fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'aopc_comparison.png'), dpi=200)
    plt.close()


if __name__ == '__main__':
    main()
