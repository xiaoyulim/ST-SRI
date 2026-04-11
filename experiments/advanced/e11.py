"""
实验六：LOSO跨受试者泛化 - 修正版
=================================
真正的 Leave-One-Subject-Out：
- held-out subject 只用于测试
- 训练侧每个 subject 独立滑窗，再做 blocked split 取 train/val
- 不再拼接原始序列，避免跨受试者边界窗口
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from common import (
    NinaProDataset,
    LSTMModel,
    DEVICE,
    create_blocked_split,
    describe_blocked_split,
)

DATA_ROOT = "./data"
RESULT_DIR = "./results/loso"
GOOD_SUBJECTS_PATH = "./good_subjects.json"

TRAIN_EPOCHS = 40
TRAIN_PATIENCE = 12
TRAIN_BATCH = 64
TRAIN_LR = 1e-3

os.makedirs(RESULT_DIR, exist_ok=True)


def load_subjects(limit=None):
    with open(GOOD_SUBJECTS_PATH, "r") as f:
        subjects = json.load(f)
    return subjects[:limit] if limit is not None else subjects


def evaluate_model(model, loader):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).long()
            pred = model(x).argmax(1)
            all_true.extend(y.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
    if not all_true:
        return {"acc": 0.0, "f1": 0.0}
    return {
        "acc": float(accuracy_score(all_true, all_pred)),
        "f1": float(f1_score(all_true, all_pred, average="macro", zero_division=0)),
    }


def build_fold_datasets(train_subjects, test_subject, batch_size=TRAIN_BATCH):
    train_parts = []
    val_parts = []
    split_report = {}

    for sub_id in train_subjects:
        ds = NinaProDataset(DATA_ROOT, sub_id, window_ms=300, target_fs=2000, step_ms=50)
        tr, val = create_blocked_split(ds, train_ratio=0.8)
        train_parts.append(tr)
        val_parts.append(val)
        split_report[str(sub_id)] = describe_blocked_split(ds, train_ratio=0.8)

    test_ds = NinaProDataset(DATA_ROOT, test_subject, window_ms=300, target_fs=2000, step_ms=50)
    split_report[str(test_subject)] = {
        "test": {
            "window_range": [0, len(test_ds) - 1],
            "raw_range": [0, len(test_ds.data) - 1],
        }
    }

    train_ds = ConcatDataset(train_parts)
    val_ds = ConcatDataset(val_parts)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, split_report


def train_one_fold(train_loader, val_loader, epochs=TRAIN_EPOCHS):
    model = LSTMModel(input_size=12).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val_acc = -1.0
    patience = 0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).long()
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        metrics = evaluate_model(model, val_loader)
        val_acc = metrics["acc"]
        print(f"    epoch {epoch + 1:02d}: val_acc={val_acc:.4f}, val_f1={metrics['f1']:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= TRAIN_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def loso_experiment(subjects):
    print("\n=== LOSO 跨受试者泛化实验（修正版）===")
    results = {}
    split_reports = {}

    for fold_idx, test_sub in enumerate(subjects, start=1):
        train_subjects = [s for s in subjects if s != test_sub]
        print(f"\n>>> Fold {fold_idx}/{len(subjects)} | held-out S{test_sub}")
        print(f"    train subjects: {train_subjects}")

        train_loader, val_loader, test_loader, split_report = build_fold_datasets(train_subjects, test_sub)
        split_reports[str(test_sub)] = split_report

        print(f"    train batches={len(train_loader)}, val batches={len(val_loader)}, test batches={len(test_loader)}")
        model = train_one_fold(train_loader, val_loader)
        test_metrics = evaluate_model(model, test_loader)
        results[str(test_sub)] = test_metrics
        print(f"    test S{test_sub}: acc={test_metrics['acc']:.4f}, f1={test_metrics['f1']:.4f}")

    return results, split_reports


def save_outputs(results, split_reports):
    accs = np.array([v["acc"] for v in results.values()])
    f1s = np.array([v["f1"] for v in results.values()])
    summary = {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "n_subjects": len(results),
    }

    payload = {
        "results": results,
        "summary": summary,
        "split_reports": split_reports,
    }

    with open(os.path.join(RESULT_DIR, "loso_results.json"), "w") as f:
        json.dump(payload, f, indent=2)

    sub_ids = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(range(len(sub_ids)), [results[s]["acc"] for s in sub_ids], color="#2196F3", alpha=0.8)
    axes[0].axhline(summary["mean_accuracy"], color="red", linestyle="--", label=f"mean={summary['mean_accuracy']:.3f}")
    axes[0].set_title("LOSO Accuracy")
    axes[0].set_xticks(range(len(sub_ids)))
    axes[0].set_xticklabels([f"S{s}" for s in sub_ids], rotation=45)
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    axes[1].bar(range(len(sub_ids)), [results[s]["f1"] for s in sub_ids], color="#4CAF50", alpha=0.8)
    axes[1].axhline(summary["mean_f1"], color="red", linestyle="--", label=f"mean={summary['mean_f1']:.3f}")
    axes[1].set_title("LOSO Macro-F1")
    axes[1].set_xticks(range(len(sub_ids)))
    axes[1].set_xticklabels([f"S{s}" for s in sub_ids], rotation=45)
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "loso_results.png"), dpi=300)
    plt.close()

    return summary


def main():
    parser = argparse.ArgumentParser(description="LOSO cross-subject experiment")
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None, help="仅运行前 N 个 good subjects")
    args = parser.parse_args()

    subjects = args.subjects if args.subjects else load_subjects(limit=args.limit)
    print(f"LOSO subjects: {subjects}")
    results, split_reports = loso_experiment(subjects)
    summary = save_outputs(results, split_reports)

    print("\n=== LOSO summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
