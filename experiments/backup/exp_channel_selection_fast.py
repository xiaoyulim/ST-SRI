"""
实验五：通道筛选 - 快速版
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

DATA_ROOT = "./data"
RESULT_DIR = "./results/channel_selection"
GOOD_SUBJECTS_PATH = "./good_subjects.json"
CHECKPOINT_DIR = "./checkpoints_2000hz"
FS = 2000
DEVICE = torch.device("cpu")

CHANNEL_COUNTS = [12, 8, 6, 4]
N_SUBJECTS = 5

os.makedirs(RESULT_DIR, exist_ok=True)

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_classes=18):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class NinaProDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, subject_id):
        self.window_len = int(300 * FS / 1000)
        self.stride = int(100 * FS / 1000)  # 增大步长减少样本
        self.data = np.load(os.path.join(root_dir, f"S{subject_id}_data.npy"))
        self.labels = np.load(os.path.join(root_dir, f"S{subject_id}_label.npy"))
        self.data = (self.data - np.mean(self.data, axis=0)) / (np.std(self.data, axis=0) + 1e-6)
        self.num_samples = (len(self.data) - self.window_len) // self.stride + 1

    def __len__(self):
        return min(self.num_samples, 150)  # 限制样本数

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_len
        x = torch.from_numpy(self.data[start:end, :]).float()
        labels = self.labels[start:end]
        y = int(np.bincount(labels).argmax()) if len(labels) > 0 else 0
        return x, torch.tensor(y)

class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, original_ds, channels):
        self.ds = original_ds
        self.channels = channels
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        x, y = self.ds[idx]
        return x[:, self.channels], y

def train_channel_count(sub_id, n_channels, channels=None):
    ds = NinaProDataset(DATA_ROOT, sub_id)
    if channels is not None:
        ds = FilteredDataset(ds, channels)
    
    # 分割数据
    train_len = int(0.7 * len(ds))
    val_len = len(ds) - train_len
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    input_size = n_channels
    model = SimpleLSTM(input_size=input_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(3):  # 快速训练
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    return correct / total if total > 0 else 0

def main():
    with open(GOOD_SUBJECTS_PATH) as f:
        subjects = json.load(f)[:N_SUBJECTS]
    
    print("=" * 50)
    print("实验五：通道筛选")
    print("=" * 50)
    
    results = {}
    for n_ch in CHANNEL_COUNTS:
        results[n_ch] = []
        print(f"\n>>> 通道数: {n_ch}")
        
        for sub_id in subjects:
            if n_ch < 12:
                np.random.seed(sub_id * 100 + n_ch)
                channels = sorted(np.random.choice(12, n_ch, replace=False).tolist())
            else:
                channels = None
            
            acc = train_channel_count(sub_id, n_ch, channels)
            results[n_ch].append(acc)
            print(f"  S{sub_id}: {acc:.3f}")
        
        print(f"  Mean: {np.mean(results[n_ch]):.3f}")
    
    # 打印汇总
    print("\n" + "=" * 50)
    print("结果汇总")
    print("=" * 50)
    for n_ch in CHANNEL_COUNTS:
        print(f"通道数 {n_ch}: {np.mean(results[n_ch]):.3f} +/- {np.std(results[n_ch]):.3f}")
    
    # 保存JSON
    with open(os.path.join(RESULT_DIR, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [np.mean(results[n]) for n in CHANNEL_COUNTS]
    stds = [np.std(results[n]) for n in CHANNEL_COUNTS]
    ax.errorbar(CHANNEL_COUNTS, means, yerr=stds, fmt='o-', capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Channels')
    ax.set_ylabel('Accuracy')
    ax.set_title('Channel Selection: Accuracy vs Channel Count')
    ax.set_xticks(CHANNEL_COUNTS)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "performance_curve.png"), dpi=300)
    print(f"\n图像已保存至 {RESULT_DIR}/performance_curve.png")

if __name__ == "__main__":
    main()
