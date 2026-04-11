"""
基线训练（E0-Train）：Per-Subject LSTM 模型训练
=================================================
目标：为每个受试者训练独立的 LSTM 分类模型，作为后续所有实验的基础。

模型：3 层 LSTM（hidden=256, dropout=0.3, 18 类）
数据：NinaPro DB2, 300ms 窗口, stride=50ms, 2000Hz
划分：80/20 blocked split（无泄漏）
训练：Adam, lr=1e-3, 早停 patience=25, 最多 40 epochs

输出：./checkpoints_2000hz/best_model_S{1-40}.pth
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
# 文件名: 02_train_factory.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from common import NinaProDataset, LSTMModel, DEVICE, create_blocked_split

# === 训练配置 ===
DATA_ROOT = "./data"
SAVE_DIR = "./checkpoints_2000hz"  # 专门存放高精度模型
TARGET_FS = 2000  # 2000Hz 原始采样率
BATCH_SIZE = 64  # 如果显存不够，改小到 32
EPOCHS = 40  # 每个人跑 60 轮
PATIENCE = 25  # 早停
LR = 0.001


def train_one_subject(sub_id):
    # 路径检查
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    save_path = os.path.join(SAVE_DIR, f"best_model_S{sub_id}.pth")

    # 断点续传
    if os.path.exists(save_path):
        print(f"S{sub_id} 模型已存在，跳过。")
        return

    print(f"\n>>> [开始训练 S{sub_id}] 2000Hz 高精度模式")

    # 1. 加载数据
    try:
        # step_ms=50 表示每隔 50ms 取一个样本（83% 的重叠率）
        # 这是论文中非常常用的设置
        ds = NinaProDataset(DATA_ROOT, sub_id, window_ms=300, target_fs=TARGET_FS, step_ms=50)
    except FileNotFoundError:
        print(f"  [Error] S{sub_id} 数据文件缺失，请先运行 01_preprocess.py")
        return

    # 无泄漏划分 80% / 20%（连续时间块 + gap）
    train_ds, val_ds = create_blocked_split(ds, train_ratio=0.8)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. 初始化模型
    sample_x, _ = ds[0]
    input_dim = sample_x.shape[1]  # 自动获取通道数 (12)
    model = LSTMModel(input_size=input_dim, hidden_size=256, num_layers=3, num_classes=18).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5)

    # 3. 训练循环
    best_acc = 0.0
    patience_cnt = 0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            # 梯度裁剪 (关键！防止长序列梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Val
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE).long()
                out = model(x)
                val_loss += criterion(out, y).item()
                _, pred = torch.max(out, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()

        acc = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"  Ep {epoch + 1}/{EPOCHS} | Val Acc: {acc:.2f}% | Loss: {avg_val_loss:.4f}", end='\r')

        # 保存最佳
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= PATIENCE:
            print(f"\n  [早停] S{sub_id} 训练结束。最佳准确率: {best_acc:.2f}%")
            break

    if patience_cnt < PATIENCE:
        print(f"\n  S{sub_id} 完成。最佳准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    # 循环训练所有 40 个受试者
    for i in range(1, 41):
        train_one_subject(i)