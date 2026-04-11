"""
实验十七：跨架构 ST-SRI 验证
==============================
目标：证明 ST-SRI 发现的协同峰是数据的生理特性，而非特定模型架构的伪影。

方法：
1. 在 LSTM / TCN / Transformer 三种架构上分别训练 per-subject 模型
2. 对每种架构运行 ST-SRI 协同谱扫描
3. 对比三种架构的协同峰位置分布和 Faithfulness Ratio

审稿人关注点（PC#3）：
  "Only LSTMs are evaluated... unclear whether the synergy decomposition
   generalizes to other sequence models (TCN, Transformer)"
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from common import (
    NinaProDataset, LSTMModel, TCNModel, TransformerModel,
    build_model, MODEL_REGISTRY,
    ST_SRI_Interpreter, DEVICE, FS,
    EMD_VALID_MIN_MS, EMD_VALID_MAX_MS,
    create_blocked_split,
)

# ================= Configuration =================
DATA_ROOT = "./data"
GOOD_SUBJECTS_PATH = "./good_subjects.json"
RESULT_DIR = "./results/multi_arch"

ARCHITECTURES = ['lstm', 'tcn', 'transformer']
CHECKPOINT_DIRS = {
    'lstm': './checkpoints_2000hz',
    'tcn': './checkpoints_tcn',
    'transformer': './checkpoints_transformer',
}

# Training config
TRAIN_EPOCHS = 40
TRAIN_PATIENCE = 15
TRAIN_BATCH = 64
TRAIN_LR = 1e-3

# ST-SRI scan config
NUM_SAMPLES = 30       # 每个 subject 采样多少个窗口做 ST-SRI
SMOOTH_SIGMA = 2.0
# =================================================

os.makedirs(RESULT_DIR, exist_ok=True)
for d in CHECKPOINT_DIRS.values():
    os.makedirs(d, exist_ok=True)


def load_subjects(limit=None):
    with open(GOOD_SUBJECTS_PATH, "r") as f:
        subjects = json.load(f)
    return subjects[:limit] if limit is not None else subjects


# ================= 训练 =================

def train_one_subject(sub_id, arch='lstm'):
    """训练单个 subject 的单个架构模型。"""
    ckpt_dir = CHECKPOINT_DIRS[arch]
    save_path = os.path.join(ckpt_dir, f"best_model_S{sub_id}.pth")

    if os.path.exists(save_path):
        print(f"  [{arch}] S{sub_id} checkpoint exists, skip training.")
        return save_path

    print(f"  [{arch}] Training S{sub_id}...")

    ds = NinaProDataset(DATA_ROOT, sub_id, window_ms=300, target_fs=2000, step_ms=50)
    train_ds, val_ds = create_blocked_split(ds, train_ratio=0.8)

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_BATCH, shuffle=False, num_workers=0)

    model = build_model(arch, input_size=12, num_classes=18).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    patience_cnt = 0

    for epoch in range(TRAIN_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE).long()
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= TRAIN_PATIENCE:
                break

    if best_state is not None:
        torch.save(best_state, save_path)
    print(f"    [{arch}] S{sub_id} best_val_acc={best_acc:.4f}")
    return save_path


def load_model(sub_id, arch='lstm'):
    """加载已训练的模型。"""
    ckpt_dir = CHECKPOINT_DIRS[arch]
    ckpt_path = os.path.join(ckpt_dir, f"best_model_S{sub_id}.pth")
    if not os.path.exists(ckpt_path):
        return None
    model = build_model(arch, input_size=12, num_classes=18).to(DEVICE)
    try:
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


# ================= ST-SRI 扫描 =================

def run_sri_scan(model, ds, num_samples=NUM_SAMPLES):
    """对一个模型运行 ST-SRI 协同谱扫描，返回平均协同谱和峰值。"""
    bg_loader = DataLoader(ds, batch_size=20, shuffle=True)
    bg_batch = next(iter(bg_loader))[0].to(DEVICE)
    interp = ST_SRI_Interpreter(model, bg_batch)

    all_synergy = []
    indices = np.linspace(0, len(ds) - 1, num_samples, dtype=int)

    for idx in indices:
        x, _ = ds[idx]
        x = x.to(DEVICE)
        lags_ms, synergy, _ = interp.scan_fast(x, max_lag_ms=150, stride=1, block_size=2)
        if len(synergy) > 0:
            all_synergy.append(synergy)

    if not all_synergy:
        return None, None, None

    lags_ms = np.array(lags_ms)
    mean_synergy = np.mean(all_synergy, axis=0)
    smooth_synergy = gaussian_filter1d(mean_synergy, sigma=SMOOTH_SIGMA)

    # 检测峰值
    peak_idx = np.argmax(smooth_synergy)
    peak_ms = float(lags_ms[peak_idx])

    return lags_ms, smooth_synergy, peak_ms


def evaluate_faithfulness(model, ds, peak_ms, num_samples=50):
    """
    Faithfulness 评估：比较遮挡 EMD 窗口 vs 遮挡近期窗口的准确率下降。
    """
    loader = DataLoader(ds, batch_size=128, shuffle=False)

    # 基线准确率
    correct_base, total = 0, 0
    correct_emd, correct_recent = 0, 0

    # EMD 窗口参数
    emd_center = int(peak_ms * FS / 1000)
    emd_half = int(10 * FS / 1000)  # ±10ms 窗口

    # 近期窗口: 0-20ms
    recent_end = int(20 * FS / 1000)

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            T = x.size(1)
            t_end = T

            # 基线
            pred_base = model(x).argmax(1)
            correct_base += (pred_base == y).sum().item()

            # 遮挡 EMD 窗口
            x_emd = x.clone()
            mask_start = max(0, t_end - 1 - emd_center - emd_half)
            mask_end = min(T, t_end - 1 - emd_center + emd_half)
            x_emd[:, mask_start:mask_end, :] = 0
            pred_emd = model(x_emd).argmax(1)
            correct_emd += (pred_emd == y).sum().item()

            # 遮挡近期窗口
            x_recent = x.clone()
            x_recent[:, -recent_end:, :] = 0
            pred_recent = model(x_recent).argmax(1)
            correct_recent += (pred_recent == y).sum().item()

            total += y.size(0)

    acc_base = correct_base / total
    acc_emd = correct_emd / total
    acc_recent = correct_recent / total

    drop_emd = acc_base - acc_emd
    drop_recent = acc_base - acc_recent
    rf = drop_emd / drop_recent if drop_recent > 1e-6 else float('inf')

    return {
        'acc_base': acc_base,
        'acc_emd_masked': acc_emd,
        'acc_recent_masked': acc_recent,
        'drop_emd': drop_emd,
        'drop_recent': drop_recent,
        'rf': rf,
    }


# ================= 主流程 =================

def main():
    parser = argparse.ArgumentParser(description="Multi-architecture ST-SRI comparison")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only use first N subjects (for smoke test)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, only run ST-SRI scan")
    parser.add_argument("--archs", nargs="+", default=ARCHITECTURES,
                        choices=ARCHITECTURES, help="Architectures to evaluate")
    args = parser.parse_args()

    subjects = load_subjects(limit=args.limit)
    print(f"=== Multi-Architecture ST-SRI Comparison ===")
    print(f"Architectures: {args.archs}")
    print(f"Subjects: {len(subjects)}")

    # Phase 1: Train models
    if not args.skip_train:
        print("\n--- Phase 1: Training models ---")
        for arch in args.archs:
            if arch == 'lstm':
                # LSTM checkpoints already exist in checkpoints_2000hz
                print(f"\n[{arch}] Using existing checkpoints in {CHECKPOINT_DIRS[arch]}")
                continue
            print(f"\n[{arch}] Training {len(subjects)} subjects...")
            for sub_id in subjects:
                train_one_subject(sub_id, arch=arch)

    # Phase 2: ST-SRI scan for each architecture
    print("\n--- Phase 2: ST-SRI Synergy Scan ---")
    results = {arch: {} for arch in args.archs}

    for arch in args.archs:
        print(f"\n[{arch}] Running ST-SRI scan...")
        for sub_id in subjects:
            model = load_model(sub_id, arch=arch)
            if model is None:
                print(f"  [{arch}] S{sub_id}: no checkpoint, skip.")
                continue

            ds = NinaProDataset(DATA_ROOT, sub_id, window_ms=300, target_fs=2000, step_ms=50)
            lags_ms, smooth_synergy, peak_ms = run_sri_scan(model, ds)

            if peak_ms is None:
                continue

            # Faithfulness
            _, val_ds = create_blocked_split(ds, train_ratio=0.8)
            faith = evaluate_faithfulness(model, val_ds, peak_ms)

            in_emd = EMD_VALID_MIN_MS <= peak_ms <= EMD_VALID_MAX_MS

            results[arch][str(sub_id)] = {
                'peak_ms': peak_ms,
                'in_emd_range': in_emd,
                'acc_base': faith['acc_base'],
                'rf': faith['rf'],
                'drop_emd': faith['drop_emd'],
                'drop_recent': faith['drop_recent'],
            }

            print(f"  [{arch}] S{sub_id}: peak={peak_ms:.1f}ms "
                  f"{'✓' if in_emd else '✗'} | "
                  f"acc={faith['acc_base']:.3f} | Rf={faith['rf']:.2f}")

    # Phase 3: Aggregate and save
    print("\n--- Phase 3: Summary ---")
    summary = {}
    for arch in args.archs:
        if not results[arch]:
            continue
        peaks = [v['peak_ms'] for v in results[arch].values()]
        in_emd_count = sum(1 for v in results[arch].values() if v['in_emd_range'])
        rfs = [v['rf'] for v in results[arch].values() if v['rf'] < 100]
        accs = [v['acc_base'] for v in results[arch].values()]

        summary[arch] = {
            'n_subjects': len(results[arch]),
            'peak_mean_ms': float(np.mean(peaks)),
            'peak_std_ms': float(np.std(peaks)),
            'peak_median_ms': float(np.median(peaks)),
            'in_emd_ratio': in_emd_count / len(results[arch]),
            'rf_mean': float(np.mean(rfs)) if rfs else 0.0,
            'rf_std': float(np.std(rfs)) if rfs else 0.0,
            'acc_mean': float(np.mean(accs)),
        }

        print(f"\n  [{arch}]")
        print(f"    Peak: {summary[arch]['peak_mean_ms']:.1f} ± {summary[arch]['peak_std_ms']:.1f} ms "
              f"(median={summary[arch]['peak_median_ms']:.1f})")
        print(f"    In EMD range: {summary[arch]['in_emd_ratio']*100:.1f}%")
        print(f"    Rf: {summary[arch]['rf_mean']:.2f} ± {summary[arch]['rf_std']:.2f}")
        print(f"    Accuracy: {summary[arch]['acc_mean']*100:.1f}%")

    # Save
    payload = {
        'per_subject': results,
        'summary': summary,
    }
    with open(os.path.join(RESULT_DIR, 'multi_arch_results.json'), 'w') as f:
        json.dump(payload, f, indent=2)

    # Phase 4: Plot
    plot_comparison(results, summary)
    print(f"\nResults saved to {RESULT_DIR}/")


def plot_comparison(results, summary):
    """生成跨架构对比图。"""
    archs = [a for a in ARCHITECTURES if a in summary]
    if len(archs) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Peak position distribution (box plot)
    peak_data = [[v['peak_ms'] for v in results[arch].values()] for arch in archs]
    bp = axes[0].boxplot(peak_data, labels=[a.upper() for a in archs],
                         patch_artist=True, widths=0.5)
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    for patch, color in zip(bp['boxes'], colors[:len(archs)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].axhspan(EMD_VALID_MIN_MS, EMD_VALID_MAX_MS, alpha=0.15, color='green',
                    label='EMD range (30-100ms)')
    axes[0].set_ylabel('Synergy Peak (ms)')
    axes[0].set_title('Synergy Peak Position by Architecture')
    axes[0].legend(fontsize=8)

    # 2. In-EMD ratio (bar)
    emd_ratios = [summary[a]['in_emd_ratio'] * 100 for a in archs]
    bars = axes[1].bar([a.upper() for a in archs], emd_ratios,
                       color=colors[:len(archs)], alpha=0.7)
    axes[1].set_ylabel('Subjects in EMD range (%)')
    axes[1].set_title('EMD Detection Rate')
    axes[1].set_ylim(0, 100)
    for bar, val in zip(bars, emd_ratios):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val:.0f}%', ha='center', fontsize=10)

    # 3. Faithfulness Ratio (bar + error)
    rf_means = [summary[a]['rf_mean'] for a in archs]
    rf_stds = [summary[a]['rf_std'] for a in archs]
    bars = axes[2].bar([a.upper() for a in archs], rf_means, yerr=rf_stds,
                       capsize=5, color=colors[:len(archs)], alpha=0.7)
    axes[2].axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Rf=1.0')
    axes[2].set_ylabel('Faithfulness Ratio (Rf)')
    axes[2].set_title('Faithfulness by Architecture')
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'multi_arch_comparison.png'), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
