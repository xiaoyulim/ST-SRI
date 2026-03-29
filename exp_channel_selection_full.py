"""
实验五：通道筛选 - 完整训练
============================
比较不同通道数(12,8,6,4)下的识别性能
使用完整的训练配置（不是简化版）
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ========== 配置 ==========
DATA_ROOT = "./data"
RESULT_DIR = "./results/channel_selection"
GOOD_SUBJECTS_PATH = "./good_subjects.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS = 2000

# 完整训练配置
CHANNEL_COUNTS = [12, 8, 6, 4]
N_SUBJECTS = 10  # 受试者数量
N_REPEATS = 3    # 随机重复次数
TRAIN_EPOCHS = 40
TRAIN_PATIENCE = 25
TRAIN_BATCH = 64
TRAIN_LR = 0.001
TRAIN_RATIO = 0.8

os.makedirs(RESULT_DIR, exist_ok=True)


# ========== 数据集 ==========
class NinaProDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, subject_id, window_ms=300, step_ms=50):
        self.fs = FS
        self.window_len = int(window_ms * self.fs / 1000)
        self.stride = int(step_ms * self.fs / 1000)
        
        d_path = os.path.join(root_dir, f"S{subject_id}_data.npy")
        l_path = os.path.join(root_dir, f"S{subject_id}_label.npy")
        
        self.data = np.load(d_path)
        self.labels = np.load(l_path)
        
        # 标准化
        self.data = (self.data - np.mean(self.data, axis=0)) / (np.std(self.data, axis=0) + 1e-6)
        
        self.num_samples = (len(self.data) - self.window_len) // self.stride + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_len
        x = torch.from_numpy(self.data[start:end, :]).float()
        labels = self.labels[start:end]
        label = int(np.bincount(labels).argmax())
        return x, label


class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, original_ds, channel_indices):
        self.ds = original_ds
        self.channel_indices = channel_indices

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        return x[:, self.channel_indices], y


# ========== 模型 ==========
class LSTMModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=256, num_layers=3, num_classes=18, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ========== 工具函数 ==========
def build_dataset(subject_id, channel_indices=None):
    ds = NinaProDataset(DATA_ROOT, subject_id)
    if channel_indices is not None:
        ds = FilteredDataset(ds, channel_indices)
    return ds


def split_dataset(ds, train_ratio=TRAIN_RATIO, batch_size=TRAIN_BATCH):
    train_len = int(train_ratio * len(ds))
    val_len = len(ds) - train_len
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def train_and_evaluate(subject_id, channel_indices, verbose=False):
    """完整训练模型并评估"""
    ds = build_dataset(subject_id, channel_indices)
    
    if len(ds) < 50:
        return None, None
    
    train_loader, val_loader = split_dataset(ds)
    input_size = len(channel_indices) if channel_indices is not None else 12
    
    # 模型
    model = LSTMModel(input_size=input_size, hidden_size=256, num_layers=3, dropout=0.3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5)
    
    best_acc = 0.0
    best_f1 = 0.0
    patience_cnt = 0
    
    for epoch in range(TRAIN_EPOCHS):
        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{TRAIN_EPOCHS}", end='\r')
        # 训练
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # 验证
        model.eval()
        all_pred, all_true = [], []
        val_loss = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE).long()
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                
                pred = out.argmax(1)
                all_pred.extend(pred.cpu().numpy())
                all_true.extend(y.cpu().numpy())
        
        scheduler.step(val_loss / len(val_loader))
        
        acc = np.mean(np.array(all_pred) == np.array(all_true))
        f1 = f1_score(all_true, all_pred, average='macro', zero_division=0)
        
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            patience_cnt = 0
        else:
            patience_cnt += 1
        
        if patience_cnt >= TRAIN_PATIENCE:
            break
    
    if verbose:
        print(f"    S{subject_id}: Acc={best_acc:.3f}, F1={best_f1:.3f}")
    
    return best_acc, best_f1


# ========== 主实验 ==========
def run_experiment():
    """运行通道筛选完整实验"""
    with open(GOOD_SUBJECTS_PATH) as f:
        subjects = json.load(f)
    
    # 限制受试者数量以控制运行时间
    subjects = subjects[:N_SUBJECTS]
    
    print("=" * 60)
    print("实验五：通道筛选（完整训练）")
    print("=" * 60)
    print(f"受试者: {subjects}")
    print(f"通道数: {CHANNEL_COUNTS}")
    print(f"重复次数: {N_REPEATS}")
    print(f"训练轮数: {TRAIN_EPOCHS}")
    print("=" * 60)
    
    # 结果存储
    results = {}
    for n_ch in CHANNEL_COUNTS:
        results[n_ch] = []
    
    # 对每个通道数进行实验
    for n_ch in CHANNEL_COUNTS:
        print(f"\n>>> 通道数: {n_ch}")
        
        for sub_id in subjects:
            if n_ch < 12:
                # 随机选择通道，重复N次
                for rep in range(N_REPEATS):
                    np.random.seed(sub_id * 1000 + n_ch * 100 + rep)
                    channel_indices = sorted(np.random.choice(12, n_ch, replace=False).tolist())
                    
                    acc, f1 = train_and_evaluate(sub_id, channel_indices, verbose=True)
                    
                    if acc is not None:
                        results[n_ch].append({
                            "subject": sub_id,
                            "rep": rep,
                            "n_channels": n_ch,
                            "accuracy": acc,
                            "f1": f1,
                            "channels": channel_indices
                        })
                        print(f"  S{sub_id} Rep{rep+1}: Acc={acc:.3f}, F1={f1:.3f}")
            else:
                # 12通道基线（无通道筛选）
                acc, f1 = train_and_evaluate(sub_id, None, verbose=False)
                
                if acc is not None:
                    results[n_ch].append({
                        "subject": sub_id,
                        "rep": 0,
                        "n_channels": n_ch,
                        "accuracy": acc,
                        "f1": f1,
                        "channels": list(range(12))
                    })
                    print(f"  S{sub_id}: Acc={acc:.3f}, F1={f1:.3f}")
    
    return results


def analyze_and_plot(results):
    """分析并绘图"""
    summary = {}
    
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)
    
    for n_ch in CHANNEL_COUNTS:
        accs = [r['accuracy'] for r in results[n_ch]]
        f1s = [r['f1'] for r in results[n_ch]]
        
        if accs:
            summary[n_ch] = {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "mean_f1": float(np.mean(f1s)),
                "std_f1": float(np.std(f1s)),
                "n_samples": len(accs)
            }
            print(f"通道数 {n_ch}: Acc={np.mean(accs):.3f}±{np.std(accs):.3f}, F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}")
    
    # 计算相对于12通道的性能下降
    baseline_acc = summary.get(12, {}).get("mean_accuracy", 0)
    baseline_f1 = summary.get(12, {}).get("mean_f1", 0)
    
    print("\n性能下降（相对于12通道）:")
    for n_ch in CHANNEL_COUNTS:
        if n_ch != 12 and n_ch in summary:
            acc_drop = (baseline_acc - summary[n_ch]["mean_accuracy"]) / baseline_acc * 100
            f1_drop = (baseline_f1 - summary[n_ch]["mean_f1"]) / baseline_f1 * 100
            print(f"  {n_ch}通道: Acc下降 {acc_drop:.1f}%, F1下降 {f1_drop:.1f}%")
    
    # 保存结果
    output = {
        "summary": summary,
        "detailed_results": results,
        "config": {
            "channel_counts": CHANNEL_COUNTS,
            "n_subjects": N_SUBJECTS,
            "n_repeats": N_REPEATS,
            "train_epochs": TRAIN_EPOCHS,
            "train_patience": TRAIN_PATIENCE,
            "train_batch": TRAIN_BATCH,
            "train_lr": TRAIN_LR
        }
    }
    
    with open(os.path.join(RESULT_DIR, "channel_selection_results.json"), 'w') as f:
        json.dump(output, f, indent=2)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    channel_counts = sorted(summary.keys())
    
    # Accuracy 曲线
    ax1 = axes[0]
    means = [summary[nc]['mean_accuracy'] for nc in channel_counts]
    stds = [summary[nc]['std_accuracy'] for nc in channel_counts]
    
    ax1.errorbar(channel_counts, means, yerr=stds, fmt='o-', capsize=5, 
                 linewidth=2, markersize=8, color='#2196F3')
    ax1.fill_between(channel_counts, 
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.2, color='#2196F3')
    ax1.set_xlabel('Number of Channels', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Channel Selection: Accuracy vs Channel Count', fontsize=13)
    ax1.set_xticks(channel_counts)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # F1 曲线
    ax2 = axes[1]
    means_f1 = [summary[nc]['mean_f1'] for nc in channel_counts]
    stds_f1 = [summary[nc]['std_f1'] for nc in channel_counts]
    
    ax2.errorbar(channel_counts, means_f1, yerr=stds_f1, fmt='s-', capsize=5,
                 linewidth=2, markersize=8, color='#FF5722')
    ax2.fill_between(channel_counts,
                     [m - s for m, s in zip(means_f1, stds_f1)],
                     [m + s for m, s in zip(means_f1, stds_f1)],
                     alpha=0.2, color='#FF5722')
    ax2.set_xlabel('Number of Channels', fontsize=12)
    ax2.set_ylabel('Macro F1', fontsize=12)
    ax2.set_title('Channel Selection: F1 vs Channel Count', fontsize=13)
    ax2.set_xticks(channel_counts)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "channel_selection_full.png"), dpi=300)
    print(f"\n图像已保存至 {RESULT_DIR}/channel_selection_full.png")
    plt.close()
    
    return summary


def main():
    results = run_experiment()
    summary = analyze_and_plot(results)
    
    print("\n" + "=" * 60)
    print("实验完成")
    print("=" * 60)
    print(f"结果目录: {RESULT_DIR}")


if __name__ == "__main__":
    main()
