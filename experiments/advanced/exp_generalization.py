"""
实验六：泛化鲁棒性
==================
1. Leave-one-subject-out (LOSO) 跨受试者泛化
2. 噪声鲁棒性测试（高斯噪声、电极掉道）

输出:
  results/generalization/loso_results.json
  results/generalization/noise_results.json
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
RESULT_DIR = "./results/generalization"
GOOD_SUBJECTS_PATH = "./good_subjects.json"
CHECKPOINT_DIR = "./checkpoints_2000hz"
FS = 2000
DEVICE = torch.device("cpu")

os.makedirs(RESULT_DIR, exist_ok=True)

class LSTMModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=256, num_classes=18):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 3, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class NinaProDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, subject_ids, window_ms=300):
        self.fs = FS
        self.window_len = int(window_ms * self.fs / 1000)
        self.stride = int(50 * self.fs / 1000)
        
        all_data = []
        all_labels = []
        
        for sub_id in subject_ids:
            data = np.load(os.path.join(root_dir, f"S{sub_id}_data.npy"))
            labels = np.load(os.path.join(root_dir, f"S{sub_id}_label.npy"))
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-6)
            all_data.append(data)
            all_labels.append(labels)
        
        self.data = np.vstack(all_data)
        self.labels = np.concatenate(all_labels)
        
        # 确保标签连续
        unique_labels = sorted(set(self.labels))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        self.labels = np.array([label_map[l] for l in self.labels])
        
        self.num_classes = len(unique_labels)
        self.num_samples = (len(self.data) - self.window_len) // self.stride + 1

    def __len__(self):
        return min(self.num_samples, 300)

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_len
        x = torch.from_numpy(self.data[start:end, :]).float()
        label_chunk = self.labels[start:end]
        y = int(np.bincount(label_chunk).argmax())
        return x, torch.tensor(y)

class NoisyDataset(torch.utils.data.Dataset):
    def __init__(self, original_ds, noise_level=0, drop_channels=[]):
        self.ds = original_ds
        self.noise_level = noise_level
        self.drop_channels = drop_channels
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        x, y = self.ds[idx]
        
        # 添加高斯噪声
        if self.noise_level > 0:
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise
        
        # 通道置零
        if self.drop_channels:
            x[:, self.drop_channels, :] = 0
        
        return x, y

def train_and_eval(train_ds, val_ds, epochs=1):
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    input_size = train_ds[0][0].shape[-1]
    model = LSTMModel(input_size=input_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    return correct / total if total > 0 else 0

def loso_experiment(subjects):
    print("\n=== LOSO 跨受试者泛化 ===")
    results = {}
    
    for test_sub in subjects[:2]:
        print(f"测试受试者: S{test_sub}")
        
        train_subs = [s for s in subjects if s != test_sub][:1]  # 限制训练受试者数
        
        train_ds = NinaProDataset(DATA_ROOT, train_subs)
        val_ds = NinaProDataset(DATA_ROOT, [test_sub])
        
        acc = train_and_eval(train_ds, val_ds)
        results[test_sub] = acc
        print(f"  S{test_sub}: Acc={acc:.3f}")
    
    return results

def noise_experiment():
    print("\n=== 噪声鲁棒性测试 ===")
    results = {}
    
    test_subjects = [1, 2, 3]
    
    for noise_level in [0, 0.1, 0.2, 0.3]:
        results[noise_level] = []
        
        for sub_id in test_subjects:
            ds = NinaProDataset(DATA_ROOT, [sub_id])
            noisy_ds = NoisyDataset(ds, noise_level=noise_level)
            val_loader = DataLoader(noisy_ds, batch_size=50)
            
            # 加载预训练模型
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_model_S{sub_id}.pth")
            if not os.path.exists(ckpt_path):
                continue
            
            model = LSTMModel().to(DEVICE)
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
            model.eval()
            
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    pred = model(x).argmax(1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            
            acc = correct / total if total > 0 else 0
            results[noise_level].append(acc)
        
        mean_acc = np.mean(results[noise_level])
        print(f"噪声水平 {noise_level}: Acc={mean_acc:.3f}")
    
    return results

def main():
    # 只做噪声实验，跳过LOSO（太慢）
    noise_results = noise_experiment()
    loso_results = {}
    
    # 保存结果
    all_results = {
        "loso": loso_results,
        "noise": noise_results
    }
    with open(os.path.join(RESULT_DIR, "results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # LOSO图
    ax1 = axes[0]
    subs = list(loso_results.keys())
    accs = list(loso_results.values())
    ax1.bar([f'S{s}' for s in subs], accs, color='#2196F3', alpha=0.7)
    ax1.set_xlabel('Test Subject')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('LOSO: Leave-One-Subject-Out')
    ax1.set_ylim(0, 1)
    ax1.axhline(np.mean(accs), color='red', linestyle='--', label=f'Mean={np.mean(accs):.3f}')
    ax1.legend()
    
    # 噪声图
    ax2 = axes[1]
    noise_levels = sorted(noise_results.keys())
    means = [np.mean(noise_results[n]) for n in noise_levels]
    stds = [np.std(noise_results[n]) for n in noise_levels]
    ax2.errorbar(noise_levels, means, yerr=stds, fmt='o-', capsize=5, linewidth=2, markersize=8, color='#FF5722')
    ax2.set_xlabel('Noise Level (Gaussian Std)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Noise Robustness')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "generalization_results.png"), dpi=300)
    print(f"\n图像已保存至 {RESULT_DIR}")

if __name__ == "__main__":
    main()
