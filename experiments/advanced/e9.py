"""
实验四：解释驱动的自适应对齐补偿
================================
将 E3 检测到的个体化协同峰位置（subject_peaks_e3.json）用于窗口前移，
形成"解释 → 补偿 → 提升性能"的闭环。

四种策略对比：
  - baseline     : 无提前（Δt = 0 ms）
  - fixed_50ms   : 统一固定前移 50 ms
  - fixed_mean   : 统一固定前移（所有受试者协同峰均值）
  - individual   : 每受试者使用其自身的 ST-SRI 协同峰（subject_peaks_e3.json）

运行模式：
  --mode eval    使用已训练的 Δt=0 模型直接评估（快速，无需重训）
  --mode train   为每种策略独立训练专用模型（完整实验，需较长时间）

输出：
  results/alignment/alignment_results_{mode}.json    各策略 × 受试者的 acc、f1
  results/alignment/alignment_summary_{mode}.png     策略对比图（Violin / Box）
  results/alignment/alignment_stats_{mode}.json      统计检验（配对 t 检验 + Cohen's d）
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy import stats
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from common import NinaProDataset, LSTMModel, DEVICE, calculate_cohens_d, interpret_cohens_d, bootstrap_ci, create_blocked_split, EMD_VALID_MIN_MS, EMD_VALID_MAX_MS

# ============================= 配置 =============================
DATA_ROOT = "./data"
CHECKPOINT_DIR = "./checkpoints_2000hz"
RESULT_DIR = "./results/alignment"
GOOD_SUBJECTS_PATH = "./good_subjects.json"
PEAKS_PATH = "./subject_peaks_e3.json"

# 生理有效范围：仅将该范围内的峰值纳入 individual 策略
# (uses common global constants for consistency with E3/E5)
EMD_LOW_MS = EMD_VALID_MIN_MS
EMD_HIGH_MS = EMD_VALID_MAX_MS

# 训练模式配置（仅 --mode train 使用）
TRAIN_EPOCHS = 40          # --fast 时覆盖为 20
TRAIN_PATIENCE = 15        # --fast 时覆盖为 6
TRAIN_LR = 0.001
TRAIN_BATCH = 64
ALIGN_CHECKPOINT_DIR = "./checkpoints_alignment"
# 热启动：从 Δt=0 模型加载初始权重，加速收敛
WARM_START = True          # 默认开启
WARM_LR_SCALE = 0.3        # 热启动时学习率缩放（避免过快破坏已有特征）


# ============================= 工具函数 =============================

def load_good_subjects():
    with open(GOOD_SUBJECTS_PATH) as f:
        return json.load(f)


def load_subject_peaks():
    with open(PEAKS_PATH) as f:
        raw = json.load(f)
    return {int(k): float(v) for k, v in raw.items()}


def build_strategies(subject_peaks, subjects):
    """
    构建4种策略的 anticipation_ms 映射 {strategy_name: {sub_id: ms}}。
    individual 策略仅使用生理有效范围内的受试者；其余受试者回退到 fixed_mean。
    """
    # 仅使用生理有效范围内的峰值来计算均值
    valid_peaks = [v for v in subject_peaks.values() if EMD_LOW_MS <= v <= EMD_HIGH_MS]
    global_mean = float(np.mean(valid_peaks)) if valid_peaks else 55.0
    fixed_mean_ms = round(global_mean)

    strategies = {
        "baseline": {s: 0 for s in subjects},
        "fixed_50ms": {s: 50 for s in subjects},
        f"fixed_{fixed_mean_ms}ms": {s: fixed_mean_ms for s in subjects},
        "individual": {},
    }

    for s in subjects:
        peak = subject_peaks.get(s, None)
        if peak is not None and EMD_LOW_MS <= peak <= EMD_HIGH_MS:
            # 裁剪到合理区间
            strategies["individual"][s] = int(round(peak))
        else:
            # 回退到 fixed_mean
            strategies["individual"][s] = fixed_mean_ms

    return strategies, fixed_mean_ms, global_mean


def build_dataset(sub_id, anticipation_ms):
    return NinaProDataset(DATA_ROOT, sub_id, window_ms=300,
                          target_fs=2000, step_ms=50,
                          anticipation_ms=anticipation_ms)


def split_dataset(ds, train_ratio=0.8, batch_size=64):
    train_ds, val_ds = create_blocked_split(ds, train_ratio)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def evaluate_model(model, loader):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = torch.argmax(model(x), dim=1)
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(y.cpu().numpy())
    acc = float(np.mean(np.array(all_pred) == np.array(all_true)))
    f1 = float(f1_score(all_true, all_pred, average='macro', zero_division=0))
    return acc, f1


# ============================= 模式一：评估已有模型 =============================

def run_eval_mode(subjects, strategies):
    """使用已训练的 Δt=0 模型，以不同 anticipation_ms 构建标签后评估。"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    results = {name: {} for name in strategies}

    for strategy_name, dt_map in strategies.items():
        print(f"\n=== 策略: {strategy_name} (eval mode) ===")
        for sub_id in subjects:
            ckpt = os.path.join(CHECKPOINT_DIR, f"best_model_S{sub_id}.pth")
            if not os.path.exists(ckpt):
                continue

            dt = dt_map[sub_id]
            try:
                ds = build_dataset(sub_id, dt)
                if len(ds) < 10:
                    continue
                _, val_loader = split_dataset(ds)
                model = LSTMModel().to(DEVICE)
                model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
                acc, f1 = evaluate_model(model, val_loader)
                results[strategy_name][sub_id] = {"acc": acc, "f1": f1, "dt_ms": dt}
                print(f"  S{sub_id} Δt={dt}ms: Acc={acc:.3f}, F1={f1:.3f}")
            except Exception as e:
                print(f"  S{sub_id}: 错误 - {e}")

    return results


# ============================= 模式二：独立训练每种策略 =============================

def train_one(sub_id, anticipation_ms, save_path):
    """训练一个模型（指定 anticipation_ms），返回 (best_acc, best_f1)。
    若 WARM_START=True，从已有 Δt=0 基线模型热启动，加速收敛。"""
    if os.path.exists(save_path):
        ds = build_dataset(sub_id, anticipation_ms)
        _, val_loader = split_dataset(ds)
        model = LSTMModel().to(DEVICE)
        model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
        acc, f1 = evaluate_model(model, val_loader)
        print(f"  S{sub_id} Δt={anticipation_ms}ms: 已有模型 Acc={acc:.3f}, F1={f1:.3f}")
        return acc, f1

    ds = build_dataset(sub_id, anticipation_ms)
    if len(ds) < 50:
        return None, None

    train_loader, val_loader = split_dataset(ds, batch_size=TRAIN_BATCH)
    model = LSTMModel().to(DEVICE)

    # 热启动：从 Δt=0 基线模型加载初始权重
    warm_ckpt = os.path.join(CHECKPOINT_DIR, f"best_model_S{sub_id}.pth")
    effective_lr = TRAIN_LR
    if WARM_START and os.path.exists(warm_ckpt) and anticipation_ms != 0:
        model.load_state_dict(torch.load(warm_ckpt, map_location=DEVICE, weights_only=True))
        effective_lr = TRAIN_LR * WARM_LR_SCALE  # 缩小学习率，微调而非重训

    optimizer = optim.Adam(model.parameters(), lr=effective_lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5)

    best_acc, best_f1, patience_cnt = 0.0, 0.0, 0

    for epoch in range(TRAIN_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE).long()
                val_loss += criterion(model(x), y).item()
        scheduler.step(val_loss / len(val_loader))

        acc, f1 = evaluate_model(model, val_loader)
        print(f"  S{sub_id} Δt={anticipation_ms}ms Ep{epoch+1}: Acc={acc:.3f}", end='\r')

        if acc > best_acc:
            best_acc, best_f1 = acc, f1
            torch.save(model.state_dict(), save_path)
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= TRAIN_PATIENCE:
            break

    print(f"  S{sub_id} Δt={anticipation_ms}ms 完成: Acc={best_acc:.3f}, F1={best_f1:.3f}     ")
    return best_acc, best_f1


def run_train_mode(subjects, strategies):
    """为每种策略 × 受试者训练专用模型。"""
    os.makedirs(ALIGN_CHECKPOINT_DIR, exist_ok=True)
    results = {name: {} for name in strategies}

    for strategy_name, dt_map in strategies.items():
        print(f"\n=== 策略: {strategy_name} (train mode) ===")
        for sub_id in subjects:
            dt = dt_map[sub_id]
            save_path = os.path.join(ALIGN_CHECKPOINT_DIR,
                                     f"model_S{sub_id}_{strategy_name}_dt{dt}.pth")
            acc, f1 = train_one(sub_id, dt, save_path)
            if acc is not None:
                results[strategy_name][sub_id] = {"acc": acc, "f1": f1, "dt_ms": dt}

    return results


# ============================= 统计分析 =============================

def statistical_analysis(results, strategy_names):
    """
    对 individual 策略与各 baseline/fixed 策略进行配对比较。
    返回统计检验结果字典。
    """
    stats_results = {}
    individual_key = "individual"

    if individual_key not in results:
        return stats_results

    # 找到所有策略都有数据的公共受试者
    common_subjects = set(results[individual_key].keys())
    for name in strategy_names:
        if name != individual_key and name in results:
            common_subjects &= set(results[name].keys())
    common_subjects = sorted(common_subjects)

    if len(common_subjects) < 5:
        print(f"  警告：公共受试者不足（{len(common_subjects)}），统计检验可能不可靠")

    ind_f1 = np.array([results[individual_key][s]["f1"] for s in common_subjects])

    for name in strategy_names:
        if name == individual_key or name not in results:
            continue

        cmp_f1 = np.array([results[name][s]["f1"] for s in common_subjects])
        diff = ind_f1 - cmp_f1

        t_stat, p_ttest = stats.ttest_rel(ind_f1, cmp_f1)
        w_stat, p_wilcox = stats.wilcoxon(diff) if len(diff) >= 10 else (np.nan, np.nan)
        d = calculate_cohens_d(ind_f1.tolist(), cmp_f1.tolist())
        ci_lo, ci_hi = bootstrap_ci(diff, n_bootstrap=5000)

        stats_results[f"individual_vs_{name}"] = {
            "n_subjects": len(common_subjects),
            "individual_mean_f1": float(np.mean(ind_f1)),
            "individual_std_f1": float(np.std(ind_f1)),
            f"{name}_mean_f1": float(np.mean(cmp_f1)),
            f"{name}_std_f1": float(np.std(cmp_f1)),
            "mean_diff_f1": float(np.mean(diff)),
            "t_stat": float(t_stat),
            "p_ttest": float(p_ttest),
            "w_stat": float(w_stat) if not np.isnan(w_stat) else None,
            "p_wilcoxon": float(p_wilcox) if not np.isnan(p_wilcox) else None,
            "cohens_d": float(d),
            "effect_size": interpret_cohens_d(d),
            "ci_95_lower": float(ci_lo),
            "ci_95_upper": float(ci_hi),
        }

        print(f"\n  individual vs {name}:")
        print(f"    Individual F1: {np.mean(ind_f1):.3f} ± {np.std(ind_f1):.3f}")
        print(f"    {name} F1:     {np.mean(cmp_f1):.3f} ± {np.std(cmp_f1):.3f}")
        print(f"    Mean diff:     {np.mean(diff):+.4f}  "
              f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"    Paired t-test: t={t_stat:.3f}, p={p_ttest:.4e}")
        if not np.isnan(p_wilcox):
            print(f"    Wilcoxon:      W={w_stat:.1f}, p={p_wilcox:.4e}")
        print(f"    Cohen's d:     {d:.3f} ({interpret_cohens_d(d)})")

    return stats_results


# ============================= 绘图 =============================

def plot_results(results, strategy_names, fixed_mean_ms, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # —— 左：每个受试者 individual 补偿量 Δt 分布 ——
    ax = axes[0]
    ind_dts = sorted([v["dt_ms"] for v in results["individual"].values()])
    ax.hist(ind_dts, bins=20, color='steelblue', alpha=0.75, edgecolor='white')
    ax.axvline(np.mean(ind_dts), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean = {np.mean(ind_dts):.1f} ms')
    ax.axvline(fixed_mean_ms, color='darkorange', linestyle=':', linewidth=1.5,
               label=f'fixed_mean = {fixed_mean_ms} ms')
    ax.set_xlabel('Individual Δt (ms)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Individual Compensation (ST-SRI Peak)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # —— 右：各策略 Macro-F1 对比（Box + Strip）——
    ax = axes[1]
    f1_data = []
    labels = []
    colors = ['#9e9e9e', '#f4a261', '#e76f51', '#2a9d8f']

    for i, name in enumerate(strategy_names):
        if name not in results:
            continue
        f1_vals = [v["f1"] for v in results[name].values()]
        f1_data.append(f1_vals)
        labels.append(name.replace('_', '\n'))

    bplot = ax.boxplot(f1_data, patch_artist=True, notch=False,
                       widths=0.45, medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bplot['boxes'], colors[:len(f1_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # 添加散点（每个受试者）
    for i, f1_vals in enumerate(f1_data):
        jitter = np.random.RandomState(42).normal(0, 0.06, len(f1_vals))
        ax.scatter(np.full(len(f1_vals), i + 1) + jitter, f1_vals,
                   alpha=0.5, s=18, color=colors[i], zorder=3)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Macro-F1', fontsize=12)
    ax.set_title('Alignment Strategy Comparison (Macro-F1)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # 标注均值
    for i, f1_vals in enumerate(f1_data):
        ax.text(i + 1, max(f1_vals) + 0.01, f'{np.mean(f1_vals):.3f}',
                ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  图像已保存至 {save_path}")
    plt.close()


def print_summary_table(results, strategy_names):
    print("\n" + "=" * 70)
    print(f"{'策略':<22} {'受试者数':>8} {'Mean F1':>10} {'Std F1':>10} {'Mean Acc':>10}")
    print("-" * 70)
    for name in strategy_names:
        if name not in results:
            continue
        accs = [v["acc"] for v in results[name].values()]
        f1s = [v["f1"] for v in results[name].values()]
        n = len(f1s)
        print(f"{name:<22} {n:>8} {np.mean(f1s):>10.3f} {np.std(f1s):>10.3f} {np.mean(accs):>10.3f}")
    print("=" * 70)


# ============================= 主入口 =============================

def main():
    parser = argparse.ArgumentParser(description="解释驱动的自适应对齐补偿实验")
    parser.add_argument('--mode', choices=['eval', 'train'], default='eval',
                        help='eval: 使用已有 Δt=0 模型评估（快速）；train: 独立训练（完整）')
    parser.add_argument('--subjects', type=int, nargs='+', default=None,
                        help='指定受试者 ID（默认使用 good_subjects.json 全部）')
    parser.add_argument('--fast', action='store_true',
                        help='快速模式：减少 epoch/patience，适合验证性实验')
    args = parser.parse_args()

    # 快速模式覆盖训练参数
    global TRAIN_EPOCHS, TRAIN_PATIENCE
    if args.fast:
        TRAIN_EPOCHS = 20
        TRAIN_PATIENCE = 6
        print("  [FAST MODE] TRAIN_EPOCHS=20, TRAIN_PATIENCE=6, WARM_START=True")

    subjects = args.subjects if args.subjects else load_good_subjects()
    subject_peaks = load_subject_peaks()

    strategies, fixed_mean_ms, global_mean = build_strategies(subject_peaks, subjects)
    strategy_names = list(strategies.keys())

    print(f"受试者数量：{len(subjects)}")
    print(f"运行模式：{args.mode}")
    print(f"策略列表：{strategy_names}")
    print(f"有效峰值均值（生理范围 {EMD_LOW_MS}–{EMD_HIGH_MS} ms）：{global_mean:.1f} ms → fixed_mean = {fixed_mean_ms} ms")

    # 统计 individual 策略中落在生理范围内的比例
    valid_cnt = sum(
        1 for s in subjects
        if subject_peaks.get(s) is not None and EMD_LOW_MS <= subject_peaks[s] <= EMD_HIGH_MS
    )
    print(f"individual 策略使用真实峰值的受试者：{valid_cnt}/{len(subjects)}")

    os.makedirs(RESULT_DIR, exist_ok=True)

    if args.mode == 'eval':
        results = run_eval_mode(subjects, strategies)
    else:
        results = run_train_mode(subjects, strategies)

    # 保存原始结果
    result_path = os.path.join(RESULT_DIR, f"alignment_results_{args.mode}.json")
    with open(result_path, 'w') as f:
        json.dump(
            {name: {str(s): v for s, v in subs.items()} for name, subs in results.items()},
            f, indent=2
        )
    print(f"\n原始结果已保存至 {result_path}")

    # 汇总打印
    print_summary_table(results, strategy_names)

    # 统计检验（individual vs 其他策略）
    print("\n--- 统计检验 ---")
    stats_out = statistical_analysis(results, strategy_names)
    stats_path = os.path.join(RESULT_DIR, f"alignment_stats_{args.mode}.json")
    with open(stats_path, 'w') as f:
        json.dump(stats_out, f, indent=2)
    print(f"\n统计结果已保存至 {stats_path}")

    # 绘图
    plot_path = os.path.join(RESULT_DIR, f"alignment_summary_{args.mode}.png")
    plot_results(results, strategy_names, fixed_mean_ms, plot_path)

    print(f"\n实验完成。结果目录：{RESULT_DIR}")


if __name__ == "__main__":
    main()
