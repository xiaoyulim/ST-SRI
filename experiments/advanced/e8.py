"""
实验二：多 Δt 提前预测基线实验
=============================
任务：给定提前量 Δt，用当前 300ms sEMG 窗口预测 Δt ms 后的动作意图。

运行模式：
  --mode eval    使用已训练的 Δt=0 模型，仅改变标签评估提前预测能力（快速，不需重训）
  --mode train   为每个 Δt 独立训练模型（完整实验，需要较长时间）

输出：
  results/anticipation/anticipation_results.json    各 Δt × 受试者的准确率、F1
  results/anticipation/anticipation_summary.png     性能随 Δt 变化的折线图
  results/anticipation/delta_t_star.json            最早稳定提前量 Δt*
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from common import NinaProDataset, LSTMModel, DEVICE, create_blocked_split

# ============================= 配置 =============================
DATA_ROOT = "./data"
CHECKPOINT_DIR = "./checkpoints_2000hz"
RESULT_DIR = "./results/anticipation"
GOOD_SUBJECTS_PATH = "./good_subjects.json"

ANTICIPATION_LIST = [0, 50, 100, 150, 200, 250]  # ms
F1_THRESHOLD = 0.80        # Δt* 的 F1 门槛
STD_THRESHOLD = 0.10       # Δt* 的跨受试者标准差上限

# 训练模式配置（仅 --mode train 使用）
TRAIN_EPOCHS = 40
TRAIN_PATIENCE = 25
TRAIN_LR = 0.001
TRAIN_BATCH = 64
TRAIN_CHECKPOINT_DIR = "./checkpoints_anticipation"  # 各 Δt 的独立模型存放位置


# ============================= 工具函数 =============================

def load_good_subjects():
    with open(GOOD_SUBJECTS_PATH) as f:
        return json.load(f)


def evaluate_model(model, loader, num_classes=18):
    """返回 (accuracy, macro_f1)"""
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = torch.argmax(model(x), dim=1)
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(y.cpu().numpy())
    acc = np.mean(np.array(all_pred) == np.array(all_true))
    f1 = f1_score(all_true, all_pred, average='macro', zero_division=0)
    return float(acc), float(f1)


def build_dataset(sub_id, anticipation_ms):
    return NinaProDataset(DATA_ROOT, sub_id, window_ms=300,
                          target_fs=2000, step_ms=50,
                          anticipation_ms=anticipation_ms)


def split_dataset(ds, train_ratio=0.8, batch_size=64):
    train_ds, val_ds = create_blocked_split(ds, train_ratio)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


# ============================= 模式一：评估已有模型 =============================

def run_eval_mode(subjects):
    """
    使用已训练的 Δt=0 模型，以不同 anticipation_ms 构建数据集后直接评估。
    揭示"现有模型的特征是否已包含提前预测信息"。
    """
    os.makedirs(RESULT_DIR, exist_ok=True)
    results = {}  # {delta_t: {sub_id: {acc, f1}}}

    for delta_t in ANTICIPATION_LIST:
        print(f"\n=== Δt = {delta_t} ms (eval mode) ===")
        results[delta_t] = {}

        for sub_id in subjects:
            ckpt = os.path.join(CHECKPOINT_DIR, f"best_model_S{sub_id}.pth")
            if not os.path.exists(ckpt):
                print(f"  S{sub_id}: 模型文件缺失，跳过")
                continue

            try:
                ds = build_dataset(sub_id, delta_t)
                if len(ds) < 10:
                    print(f"  S{sub_id}: 数据集过小（{len(ds)} 样本），跳过")
                    continue

                _, val_loader = split_dataset(ds)

                model = LSTMModel().to(DEVICE)
                model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))

                acc, f1 = evaluate_model(model, val_loader)
                results[delta_t][sub_id] = {"acc": acc, "f1": f1}
                print(f"  S{sub_id}: Acc={acc:.3f}, F1={f1:.3f}")

            except Exception as e:
                print(f"  S{sub_id}: 错误 - {e}")
                continue

    return results


# ============================= 模式二：独立训练每个 Δt =============================

def train_one(sub_id, delta_t, save_dir):
    """为指定受试者和 Δt 训练一个模型，返回 (best_acc, save_path)"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"model_S{sub_id}_dt{delta_t}.pth")

    if os.path.exists(save_path):
        # 加载并直接评估
        ds = build_dataset(sub_id, delta_t)
        _, val_loader = split_dataset(ds)
        model = LSTMModel().to(DEVICE)
        model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
        acc, f1 = evaluate_model(model, val_loader)
        print(f"  S{sub_id} Δt={delta_t}ms: 已有模型 Acc={acc:.3f}, F1={f1:.3f}")
        return acc, f1

    ds = build_dataset(sub_id, delta_t)
    if len(ds) < 50:
        return None, None

    train_loader, val_loader = split_dataset(ds, batch_size=TRAIN_BATCH)
    sample_x, _ = ds[0]
    model = LSTMModel(input_size=sample_x.shape[1]).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=TRAIN_LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5)

    best_acc, patience_cnt = 0.0, 0
    best_f1 = 0.0

    for epoch in range(TRAIN_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE).long()
                val_loss += criterion(model(x), y).item()
        scheduler.step(val_loss / len(val_loader))

        acc, f1 = evaluate_model(model, val_loader)
        if acc > best_acc:
            best_acc, best_f1 = acc, f1
            torch.save(model.state_dict(), save_path)
            patience_cnt = 0
        else:
            patience_cnt += 1

        print(f"  S{sub_id} Δt={delta_t}ms Ep{epoch+1}: Acc={acc:.3f}", end='\r')
        if patience_cnt >= TRAIN_PATIENCE:
            break

    print(f"  S{sub_id} Δt={delta_t}ms 完成: Acc={best_acc:.3f}, F1={best_f1:.3f}     ")
    return best_acc, best_f1


def run_train_mode(subjects):
    """为每个 Δt 独立训练模型并评估。"""
    results = {}

    for delta_t in ANTICIPATION_LIST:
        print(f"\n=== Δt = {delta_t} ms (train mode) ===")
        results[delta_t] = {}

        for sub_id in subjects:
            acc, f1 = train_one(sub_id, delta_t, TRAIN_CHECKPOINT_DIR)
            if acc is not None:
                results[delta_t][sub_id] = {"acc": acc, "f1": f1}

    return results


# ============================= 结果分析与绘图 =============================

def compute_delta_t_star(results, f1_thresh=F1_THRESHOLD, std_thresh=STD_THRESHOLD):
    """
    计算最早稳定提前量 Δt*：
    在满足 Macro-F1 > f1_thresh 且跨受试者 F1 标准差 < std_thresh 的前提下，最大的 Δt。
    """
    delta_t_star = 0
    summary = {}

    for delta_t in sorted(results.keys()):
        f1_vals = [v["f1"] for v in results[delta_t].values() if v]
        if not f1_vals:
            continue
        mean_f1 = np.mean(f1_vals)
        std_f1 = np.std(f1_vals)
        summary[delta_t] = {"mean_f1": mean_f1, "std_f1": std_f1,
                             "mean_acc": np.mean([v["acc"] for v in results[delta_t].values() if v]),
                             "n_subjects": len(f1_vals)}
        if mean_f1 >= f1_thresh and std_f1 <= std_thresh:
            delta_t_star = delta_t

    return delta_t_star, summary


def plot_results(summary, delta_t_star, mode_label, save_path):
    delta_ts = sorted(summary.keys())
    mean_f1s = [summary[d]["mean_f1"] for d in delta_ts]
    std_f1s = [summary[d]["std_f1"] for d in delta_ts]
    mean_accs = [summary[d]["mean_acc"] for d in delta_ts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # F1 曲线
    ax1.errorbar(delta_ts, mean_f1s, yerr=std_f1s, fmt='o-', color='steelblue',
                 capsize=5, linewidth=2, label='Macro-F1 (mean ± std)')
    ax1.axhline(F1_THRESHOLD, color='gray', linestyle='--', linewidth=1,
                label=f'F1 threshold = {F1_THRESHOLD}')
    if delta_t_star > 0:
        ax1.axvline(delta_t_star, color='red', linestyle=':', linewidth=1.5,
                    label=f'Δt* = {delta_t_star} ms')
    ax1.set_xlabel('Anticipation Δt (ms)', fontsize=12)
    ax1.set_ylabel('Macro-F1', fontsize=12)
    ax1.set_title(f'Early Prediction F1 vs Δt ({mode_label})', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # Accuracy 曲线
    ax2.plot(delta_ts, mean_accs, 's-', color='darkorange', linewidth=2, label='Accuracy (mean)')
    ax2.set_xlabel('Anticipation Δt (ms)', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Early Prediction Accuracy vs Δt ({mode_label})', fontsize=13)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  图像已保存至 {save_path}")


def print_summary_table(summary, delta_t_star):
    print("\n" + "=" * 60)
    print(f"{'Δt (ms)':<10} {'Mean F1':>10} {'Std F1':>10} {'Mean Acc':>10} {'N':>5}")
    print("-" * 60)
    for dt in sorted(summary.keys()):
        s = summary[dt]
        marker = " ← Δt*" if dt == delta_t_star else ""
        print(f"{dt:<10} {s['mean_f1']:>10.3f} {s['std_f1']:>10.3f} "
              f"{s['mean_acc']:>10.3f} {s['n_subjects']:>5}{marker}")
    print("=" * 60)
    print(f"\n最早稳定提前量 Δt* = {delta_t_star} ms")
    print(f"（判据：Macro-F1 ≥ {F1_THRESHOLD} 且跨受试者 F1 std < {STD_THRESHOLD}）")


# ============================= 主入口 =============================

def main():
    parser = argparse.ArgumentParser(description="多 Δt 提前预测实验")
    parser.add_argument('--mode', choices=['eval', 'train'], default='eval',
                        help='eval: 使用已有模型评估（快速）；train: 为每个 Δt 重训练（完整）')
    parser.add_argument('--subjects', type=int, nargs='+', default=None,
                        help='指定受试者 ID（默认使用 good_subjects.json 中的全部）')
    args = parser.parse_args()

    subjects = args.subjects if args.subjects else load_good_subjects()
    print(f"受试者数量：{len(subjects)}")
    print(f"运行模式：{args.mode}")
    print(f"提前量列表：{ANTICIPATION_LIST} ms")

    os.makedirs(RESULT_DIR, exist_ok=True)

    if args.mode == 'eval':
        results = run_eval_mode(subjects)
        mode_label = "Eval (pretrained Δt=0 model)"
    else:
        results = run_train_mode(subjects)
        mode_label = "Train (per-Δt model)"

    # 保存原始结果
    result_path = os.path.join(RESULT_DIR, f"anticipation_results_{args.mode}.json")
    with open(result_path, 'w') as f:
        json.dump({str(k): {str(sk): sv for sk, sv in v.items()}
                   for k, v in results.items()}, f, indent=2)
    print(f"\n原始结果已保存至 {result_path}")

    # 计算 Δt* 和汇总统计
    delta_t_star, summary = compute_delta_t_star(results)
    print_summary_table(summary, delta_t_star)

    # 保存 Δt* 结果
    star_path = os.path.join(RESULT_DIR, f"delta_t_star_{args.mode}.json")
    with open(star_path, 'w') as f:
        json.dump({"delta_t_star_ms": delta_t_star, "summary": summary}, f, indent=2)

    # 绘图
    plot_path = os.path.join(RESULT_DIR, f"anticipation_summary_{args.mode}.png")
    plot_results(summary, delta_t_star, mode_label, plot_path)

    print(f"\n实验完成。结果目录：{RESULT_DIR}")


if __name__ == "__main__":
    main()
