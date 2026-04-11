import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
"""
实验十二：噪声鲁棒性 - 修正版
============================
目标：定义清晰、可复现的 noise robustness 实验。
- 使用 good_subjects.json 或其子集
- 基于已训练的 per-subject 模型评估 noise=0 基线与噪声退化
- 支持高斯噪声与电极掉道
- 使用 blocked split 保持与训练/验证口径一致
"""
import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from common import NinaProDataset, LSTMModel, DEVICE, create_blocked_split

DATA_ROOT = "./data"
RESULT_DIR = "./results/generalization"
GOOD_SUBJECTS_PATH = "./good_subjects.json"
CHECKPOINT_DIR = "./checkpoints_2000hz"

GAUSSIAN_LEVELS = [0.0, 0.1, 0.2, 0.3]
DROP_PATTERNS = {
    "none": [],
    "drop_ch_0": [0],
    "drop_ch_0_1": [0, 1],
    "drop_ch_0_1_2": [0, 1, 2],
}

os.makedirs(RESULT_DIR, exist_ok=True)


class NoisyDataset(Dataset):
    def __init__(self, original_ds, noise_level=0.0, drop_channels=None):
        self.ds = original_ds
        self.noise_level = float(noise_level)
        self.drop_channels = list(drop_channels or [])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        x = x.clone()

        if self.noise_level > 0:
            x = x + torch.randn_like(x) * self.noise_level

        if self.drop_channels:
            x[:, self.drop_channels] = 0

        return x, y


def load_subjects(limit=None):
    with open(GOOD_SUBJECTS_PATH, "r") as f:
        subjects = json.load(f)
    return subjects[:limit] if limit is not None else subjects


def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).long()
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return float(correct / total) if total > 0 else 0.0


def build_eval_subset(sub_id):
    ds = NinaProDataset(DATA_ROOT, sub_id, window_ms=300, target_fs=2000, step_ms=50)
    _, val_ds = create_blocked_split(ds, train_ratio=0.8)
    return val_ds


def load_model_for_subject(sub_id):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_model_S{sub_id}.pth")
    if not os.path.exists(ckpt_path):
        return None
    model = LSTMModel().to(DEVICE)
    try:
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def run_noise_experiment(subjects):
    print("\n=== Gaussian noise robustness ===")
    results = {str(level): {} for level in GAUSSIAN_LEVELS}

    for sub_id in subjects:
        model = load_model_for_subject(sub_id)
        if model is None:
            continue
        base_ds = build_eval_subset(sub_id)

        for level in GAUSSIAN_LEVELS:
            noisy_ds = NoisyDataset(base_ds, noise_level=level)
            loader = DataLoader(noisy_ds, batch_size=128, shuffle=False)
            acc = evaluate_model(model, loader)
            results[str(level)][str(sub_id)] = acc
            print(f"  S{sub_id} noise={level:.1f}: acc={acc:.4f}")

    return results


def run_drop_experiment(subjects):
    print("\n=== Channel dropout robustness ===")
    results = {name: {} for name in DROP_PATTERNS}

    for sub_id in subjects:
        model = load_model_for_subject(sub_id)
        if model is None:
            continue
        base_ds = build_eval_subset(sub_id)

        for pattern_name, channels in DROP_PATTERNS.items():
            noisy_ds = NoisyDataset(base_ds, noise_level=0.0, drop_channels=channels)
            loader = DataLoader(noisy_ds, batch_size=128, shuffle=False)
            acc = evaluate_model(model, loader)
            results[pattern_name][str(sub_id)] = acc
            print(f"  S{sub_id} {pattern_name}: acc={acc:.4f}")

    return results


def summarize_results(per_setting):
    summary = {}
    for key, subject_map in per_setting.items():
        vals = list(subject_map.values())
        summary[key] = {
            "mean_accuracy": float(np.mean(vals)) if vals else 0.0,
            "std_accuracy": float(np.std(vals)) if vals else 0.0,
            "n_subjects": len(vals),
        }
    return summary


def plot_results(noise_results, drop_results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    noise_levels = [float(k) for k in noise_results.keys()]
    noise_means = [np.mean(list(v.values())) if v else 0.0 for v in noise_results.values()]
    noise_stds = [np.std(list(v.values())) if v else 0.0 for v in noise_results.values()]
    axes[0].errorbar(noise_levels, noise_means, yerr=noise_stds, fmt='o-', capsize=5, linewidth=2)
    axes[0].set_title("Gaussian Noise Robustness")
    axes[0].set_xlabel("Noise std")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    drop_names = list(drop_results.keys())
    drop_means = [np.mean(list(drop_results[k].values())) if drop_results[k] else 0.0 for k in drop_names]
    drop_stds = [np.std(list(drop_results[k].values())) if drop_results[k] else 0.0 for k in drop_names]
    axes[1].bar(range(len(drop_names)), drop_means, yerr=drop_stds, capsize=5, color="#FF7043", alpha=0.8)
    axes[1].set_title("Channel Dropout Robustness")
    axes[1].set_xticks(range(len(drop_names)))
    axes[1].set_xticklabels(drop_names, rotation=20)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "generalization_results.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Noise robustness experiment")
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fast", action="store_true", help="只跑少量 subject 做 smoke test")
    args = parser.parse_args()

    subjects = args.subjects if args.subjects else load_subjects(limit=args.limit)
    if args.fast:
        subjects = subjects[:3]

    print(f"noise robustness subjects: {subjects}")
    noise_results = run_noise_experiment(subjects)
    drop_results = run_drop_experiment(subjects)

    payload = {
        "subjects": subjects,
        "noise": noise_results,
        "noise_summary": summarize_results(noise_results),
        "channel_drop": drop_results,
        "channel_drop_summary": summarize_results(drop_results),
    }

    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(payload, f, indent=2)

    plot_results(noise_results, drop_results)
    print(f"\nresults saved to {RESULT_DIR}")


if __name__ == "__main__":
    main()
