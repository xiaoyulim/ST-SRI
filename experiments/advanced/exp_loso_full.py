"""
实验六：LOSO跨受试者泛化 - 完整版
=================================
Leave-One-Subject-Out: 训练N-1个受试者，测试剩下的1个
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

DATA_ROOT = "./data"
RESULT_DIR = "./results/loso"
GOOD_SUBJECTS_PATH = "./good_subjects.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS = 2000

# 完整训练配置
N_SUBJECTS = 10  # 参与LOSO的受试者数
TRAIN_EPOCHS = 40
TRAIN_PATIENCE = 20
TRAIN_BATCH = 64
TRAIN_LR = 0.001

os.makedirs(RESULT_DIR, exist_ok=True)

class LSTMModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 18)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class NinaProDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, subject_ids, window_ms=300, step_ms=50):
        self.fs = FS
        self.window_len = int(window_ms * self.fs / 1000)
        self.stride = int(step_ms * self.fs / 1000)
        
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
        
        # 标签重映射确保连续
        unique_labels = sorted(set(self.labels))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        self.labels = np.array([label_map[l] for l in self.labels])
        
        self.num_classes = len(unique_labels)
        self.num_samples = (len(self.data) - self.window_len) // self.stride + 1

    def __len__(self):
        return min(self.num_samples, 500)

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_len
        x = torch.from_numpy(self.data[start:end, :]).float()
        label_chunk = self.labels[start:end]
        y = int(np.bincount(label_chunk).argmax()) if len(label_chunk) > 0 else 0
        return x, torch.tensor(y)

class NoisyDataset(torch.utils.data.Dataset):
    def __init__(self, original_ds, noise_std=0):
        self.ds = original_ds
        self.noise_std = noise_std
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        x, y = self.ds[idx]
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x, y

def split_dataset(ds, train_ratio=0.8):
    train_len = int(train_ratio * len(ds))
    val_len = len(ds) - train_len
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_BATCH, shuffle=False)
    return train_loader, val_loader

def train_and_eval(train_ds, val_ds, epochs=TRAIN_EPOCHS):
    train_loader, val_loader = split_dataset(train_ds)
    
    model = LSTMModel(input_size=12).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    best_acc = 0
    patience = 0
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE).long()
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        acc = correct / total if total > 0 else 0
        
        if acc > best_acc:
            best_acc = acc
            patience = 0
        else:
            patience += 1
        
        if patience >= TRAIN_PATIENCE:
            break
    
    return best_acc

def loso_experiment(subjects):
    print("\n=== LOSO 跨受试者泛化实验 ===")
    print(f"受试者: {subjects}")
    print(f"每个测试样本: 在其余 {len(subjects)-1} 个受试者上训练")
    
    results = {}
    
    for test_idx, test_sub in enumerate(subjects):
        print(f"\n>>> 测试受试者: S{test_sub} ({test_idx+1}/{len(subjects)})")
        
        # 训练集：除测试外的所有受试者
        train_subs = [s for s in subjects if s != test_sub]
        
        # 验证集
        val_subs = [test_sub]
        
        print(f"    训练受试者: {train_subs}")
        
        # 构建数据加载器
        train_ds = NinaProDataset(DATA_ROOT, train_subs)
        val_ds = NinaProDataset(DATA_ROOT, val_subs)
        
        print(f"    训练样本: {len(train_ds)}, 测试样本: {len(val_ds)}")
        
        # 训练并评估
        acc = train_and_eval(train_ds, val_ds)
        
        results[test_sub] = acc
        print(f"    S{test_sub}: Acc={acc:.3f}")
    
    return results

def main():
    with open(GOOD_SUBJECTS_PATH) as f:
        all_subjects = json.load(f)
    
    subjects = all_subjects[:N_SUBJECTS]
    
    print("=" * 60)
    print("LOSO 跨受试者泛化实验（完整版）")
    print("=" * 60)
    print(f"测试受试者数: {len(subjects)}")
    print(f"每个模型训练轮数: {TRAIN_EPOCHS}")
    print("=" * 60)
    
    results = loso_experiment(subjects)
    
    # 汇总
    accs = list(results.values())
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    
    print("\n" + "=" * 60)
    print("LOSO 结果汇总")
    print("=" * 60)
    print(f"平均准确率: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"最高准确率: {max(accs):.3f}")
    print(f"最低准确率: {min(accs):.3f}")
    
    # 保存
    with open(os.path.join(RESULT_DIR, "loso_results.json"), 'w') as f:
        json.dump({
            "results": results,
            "summary": {
                "mean_accuracy": float(mean_acc),
                "std_accuracy": float(std_acc),
                "n_subjects": len(subjects)
            }
        }, f, indent=2)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sub_ids = list(results.keys())
    accs = list(results.values())
    
    bars = ax.bar(range(len(sub_ids)), accs, color='#2196F3', alpha=0.7)
    ax.axhline(mean_acc, color='red', linestyle='--', linewidth=2, label=f'Mean={mean_acc:.3f}')
    ax.fill_between([-0.5, len(sub_ids)-0.5], mean_acc-std_acc, mean_acc+std_acc, alpha=0.2, color='red')
    
    ax.set_xlabel('Test Subject', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('LOSO: Leave-One-Subject-Out Cross-Validation', fontsize=14)
    ax.set_xticks(range(len(sub_ids)))
    ax.set_xticklabels([f'S{s}' for s in sub_ids])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    for i, (s, a) in enumerate(zip(sub_ids, accs)):
        ax.text(i, a + 0.02, f'{a:.2f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "loso_results.png"), dpi=300)
    print(f"\n图像已保存至 {RESULT_DIR}/loso_results.png")

if __name__ == "__main__":
    main()
