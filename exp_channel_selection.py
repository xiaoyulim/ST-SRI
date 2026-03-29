"""
实验五：通道筛选 - 基于预训练模型推理
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
import random

DATA_ROOT = "./data"
RESULT_DIR = "./results/channel_selection"
GOOD_SUBJECTS_PATH = "./good_subjects.json"
CHECKPOINT_DIR = "./checkpoints_2000hz"
FS = 2000
CHANNEL_COUNTS = [12]
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
    def __init__(self, root_dir, subject_id):
        self.window_len = int(300 * FS / 1000)
        self.stride = int(50 * FS / 1000)
        self.data = np.load(os.path.join(root_dir, f"S{subject_id}_data.npy"))
        self.labels = np.load(os.path.join(root_dir, f"S{subject_id}_label.npy"))
        self.data = (self.data - np.mean(self.data, axis=0)) / (np.std(self.data, axis=0) + 1e-6)
        self.num_samples = (len(self.data) - self.window_len) // self.stride + 1

    def __len__(self):
        return min(self.num_samples, 200)

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_len
        x = torch.from_numpy(self.data[start:end, :]).float()
        y = int(np.bincount(self.labels[start:end]).argmax())
        return x, torch.tensor(y)

def main():
    with open(GOOD_SUBJECTS_PATH) as f:
        subjects = json.load(f)[:3]
    
    print("实验五: 通道筛选 - 预训练模型推理")
    
    results = {}
    for n_ch in CHANNEL_COUNTS:
        results[n_ch] = []
        
        for sub_id in subjects:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_model_S{sub_id}.pth")
            if not os.path.exists(ckpt_path):
                continue
            
            model = LSTMModel().to(DEVICE)
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
            model.eval()
            
            ds = NinaProDataset(DATA_ROOT, sub_id)
            loader = DataLoader(ds, batch_size=100)
            
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in loader:
                    pred = model(x).argmax(1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            
            acc = correct / total if total > 0 else 0
            results[n_ch].append(acc)
            print(f"S{sub_id}: Acc={acc:.3f}")
    
    print(f"\n通道数 12: {np.mean(results[12]):.3f}")
    
    with open(os.path.join(RESULT_DIR, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print("完成")

if __name__ == "__main__":
    main()
