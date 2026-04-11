import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import os
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from common import NinaProDataset, LSTMModel, DEVICE, create_blocked_split

# 修复 OMP 报错 (防止在某些环境下绘图崩溃)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================= 配置 =================
DATA_ROOT = "./data"
CHECKPOINT_DIR = "./checkpoints_2000hz"
RESULT_DIR = "./results/model_evaluation"  # 图片保存路径
TARGET_FS = 2000
WINDOW_MS = 300
STEP_MS = 50  # 与 e0_train.py 和 checkpoints_2000hz 保持一致
NUM_CLASSES = 18
ACC_THRESHOLD = 75.0  # 达标阈值
# =======================================

os.makedirs(RESULT_DIR, exist_ok=True)


def evaluate_saved_model(sub_id):
    model_path = os.path.join(CHECKPOINT_DIR, f"best_model_S{sub_id}.pth")
    if not os.path.exists(model_path): return None

    try:
        ds = NinaProDataset(DATA_ROOT, sub_id, window_ms=WINDOW_MS, target_fs=TARGET_FS, step_ms=STEP_MS)
    except:
        return None

    # 无泄漏划分：使用连续时间块划分替代 random_split
    _, val_ds = create_blocked_split(ds, train_ratio=0.8)

    # 稍微增大 batch_size 加速评估
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    # 加载模型
    sample_x, _ = ds[0]
    model = LSTMModel(input_size=sample_x.shape[1], hidden_size=256, num_layers=3, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))  # 兼容新旧pytorch
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            out = model(x)
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    if total == 0: return 0.0
    return 100 * correct / total


def save_table_image(df, save_path):
    """绘制并保存表格图片"""
    # 设置图形大小，根据行数动态调整高度
    rows = len(df)
    plt.figure(figsize=(8, rows * 0.35 + 2), dpi=200)
    ax = plt.gca()
    ax.axis('off')

    # 绘制表格
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    # 美化表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    # 设置颜色和字体粗细
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            # 表头
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')  # 深蓝色表头
            cell.set_edgecolor('white')
        else:
            # 数据行
            cell.set_edgecolor('#dddddd')
            if row % 2 == 0:
                cell.set_facecolor('#f5f5f5')  # 隔行变色

            # 特殊高亮：如果 Status 是 ❌ 或 Accuracy < Threshold
            # 注意：DataFrame 的数据是字符串，需要简单判断逻辑
            row_data = df.iloc[row - 1]
            acc_str = str(row_data['Accuracy'])

            if "❌" in str(row_data['Status']) or "⚠️" in str(row_data['Status']):
                cell.set_text_props(color='#d62728')  # 红色字体
            elif acc_str != "--":
                try:
                    val = float(acc_str.replace('%', ''))
                    if val < ACC_THRESHOLD:
                        cell.set_text_props(color='#ff7f0e')  # 橙色字体表示未达标
                    else:
                        cell.set_text_props(color='#2ca02c', weight='bold')  # 绿色表示优秀
                except:
                    pass

    plt.title(f"Model Evaluation Summary (Threshold > {ACC_THRESHOLD}%)", fontsize=14, weight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"📊 表格图片已保存至: {save_path}")


if __name__ == "__main__":
    print(f"{'Subject':<10} | {'Status':<15} | {'Accuracy':<20}")
    print("-" * 50)

    good_subjects = []
    results_data = []  # 用于存储表格数据
    valid_accuracies = []

    for i in range(1, 41):
        try:
            acc = evaluate_saved_model(i)

            if acc is not None:
                acc_val = float(acc)
                acc_str = f"{acc_val:.2f}%"

                if acc_val >= ACC_THRESHOLD:
                    status = "✅ Qualified"
                    good_subjects.append(i)
                else:
                    status = "⚠️ Low Acc"

                valid_accuracies.append(acc_val)
            else:
                status = "❌ Missing"
                acc_str = "--"

            print(f"S{i:<9} | {status:<15} | {acc_str}")

            # 存入列表
            results_data.append([f"S{i}", status, acc_str])

        except Exception as e:
            print(f"S{i:<9} | {'⚠️ Error':<15} | {str(e)}")
            results_data.append([f"S{i}", "⚠️ Error", "Error"])

    print("-" * 50)

    # 计算平均值
    if valid_accuracies:
        avg_acc = np.mean(valid_accuracies)
        print(f"平均准确率 (Valid Models): {avg_acc:.2f}%")
        results_data.append(["AVERAGE", "---", f"{avg_acc:.2f}%"])

    print(f"合格模型 (Acc > {ACC_THRESHOLD}%): {len(good_subjects)} 个")

    # 1. 保存 JSON
    with open("good_subjects.json", "w") as f:
        json.dump(good_subjects, f)
    print(">>> 已保存名单到 'good_subjects.json'")

    # 2. 生成并保存表格图片
    if results_data:
        df = pd.DataFrame(results_data, columns=["Subject", "Status", "Accuracy"])
        save_path = os.path.join(RESULT_DIR, "model_accuracy_table.png")
        save_table_image(df, save_path)
