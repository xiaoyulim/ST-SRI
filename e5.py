import torch
import numpy as np
import os
import json
import csv
import matplotlib.pyplot as plt
from scipy import stats  # 新增：用于统计检验
from torch.utils.data import DataLoader
from common import NinaProDataset, LSTMModel, DEVICE, FS

# ================= 配置 =================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CHECKPOINT_DIR = "./checkpoints_2000hz"
RESULT_DIR = "./results/e5_faithfulness"
PEAK_JSON_PATH = "subject_peaks_e3.json"

SAVE_CSV_PATH = os.path.join(RESULT_DIR, "e5_detailed_results.csv")
SAVE_SUMMARY_PATH = os.path.join(RESULT_DIR, "e5_summary.json")
SAVE_PLOT_PATH = os.path.join(RESULT_DIR, "e5_final_plot.png")


# ========================================

def get_accuracy_with_mask(model, loader, mask_range_ms=None):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            if mask_range_ms:
                ms_start, ms_end = mask_range_ms
                T = x.shape[1]
                idx_end = T - int(ms_start * FS / 1000)
                idx_start = T - int(ms_end * FS / 1000)
                idx_start, idx_end = max(0, idx_start), min(T, idx_end)
                if idx_end > idx_start:
                    x[:, idx_start:idx_end, :] = 0

            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def run_e5():
    os.makedirs(RESULT_DIR, exist_ok=True)

    if not os.path.exists(PEAK_JSON_PATH):
        print(f"❌ 找不到 {PEAK_JSON_PATH}，请先运行 e3.py 生成峰值文件！")
        return

    with open(PEAK_JSON_PATH, 'r') as f:
        peaks_dict = json.load(f)

    rows_to_save = []

    # 提取所有受试者ID并排序
    subjects = sorted([int(k) for k in peaks_dict.keys()])

    print(f"{'Sub':<5} | {'Peak':<6} | {'Base':<8} | {'-Recent':<10} | {'-EMD':<8} | {'Check'}")
    print("-" * 70)

    for sub_id in subjects:
        try:
            peak = peaks_dict[str(sub_id)]

            # 加载数据
            ds = NinaProDataset("./data", sub_id, window_ms=300, target_fs=FS)
            # 使用全量数据测更准
            loader = DataLoader(ds, batch_size=128, shuffle=False)

            # 加载模型
            model = LSTMModel().to(DEVICE)
            model_path = f"{CHECKPOINT_DIR}/best_model_S{sub_id}.pth"
            if not os.path.exists(model_path): continue

            # 兼容 weights_only
            try:
                model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            except:
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))

            # 1. 基准
            acc_base = get_accuracy_with_mask(model, loader)

            # 2. 屏蔽近期 (0-20ms) - 对照组
            acc_recent = get_accuracy_with_mask(model, loader, (0, 20))
            drop_recent = max(0, acc_base - acc_recent)

            # 3. 屏蔽动态 EMD (Peak ± 20ms) - 实验组
            # 这里的区间宽度(40ms)应与Recent(20ms)有可比性，或者略宽以覆盖不确定性
            # 为了公平，我们可以设为 ±15ms (共30ms) 或 ±20ms (共40ms)
            # 只要 EMD 的单位密度下降更高即可，这里保持 ±20ms
            emd_start = max(0, peak - 20)
            emd_end = min(300, peak + 20)
            acc_emd = get_accuracy_with_mask(model, loader, (emd_start, emd_end))
            drop_emd = max(0, acc_base - acc_emd)

            # 记录
            rows_to_save.append({
                "subject": sub_id,
                "drop_recent": drop_recent,
                "drop_emd": drop_emd
            })

            chk = "✅" if drop_emd > drop_recent else " "
            print(
                f"S{sub_id:<4} | {peak:<6.1f} | {acc_base:<6.1f}%  | -{drop_recent:<8.2f}% | -{drop_emd:<6.2f}% | {chk}")

        except Exception as e:
            # print(f"S{sub_id} Error: {e}")
            continue

    if not rows_to_save:
        print("❌ 没有有效数据！")
        return

    # === 数据处理 ===
    drops_rec = [r['drop_recent'] for r in rows_to_save]
    drops_emd = [r['drop_emd'] for r in rows_to_save]

    # 统计检验 (Paired T-Test)
    t_stat, p_val = stats.ttest_rel(drops_emd, drops_rec)

    avg_rec = np.mean(drops_rec)
    avg_emd = np.mean(drops_emd)
    std_rec = np.std(drops_rec)
    std_emd = np.std(drops_emd)

    ratio = avg_emd / (avg_rec + 1e-9)

    print("-" * 70)
    print(f"Total Subjects: {len(rows_to_save)}")
    print(f"Avg Drop Recent: {avg_rec:.3f}% ± {std_rec:.3f}")
    print(f"Avg Drop EMD:    {avg_emd:.3f}% ± {std_emd:.3f}")
    print(f"Impact Ratio:    {ratio:.2f}x")
    print(f"P-value:         {p_val:.2e} {'(Significant ***)' if p_val < 0.001 else ''}")

    # === 保存 CSV ===
    with open(SAVE_CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "drop_recent", "drop_emd"])
        writer.writeheader()
        writer.writerows(rows_to_save)

    # ==========================================
    # 📊 终极绘图 (Rich Visualization)
    # ==========================================
    plt.figure(figsize=(9, 7))

    # 1. 准备数据
    means = [avg_rec, avg_emd]
    stds = [std_rec, std_emd]  # 或者用标准误: std / np.sqrt(N)
    labels = ['Mask Recent\n(0-20ms)', 'Mask EMD\n(Peak±20ms)']
    x_pos = [0, 1]

    # 2. 绘制柱状图 (Bar)
    bars = plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.6,
                   ecolor='black', capsize=10, width=0.5,
                   color=['#bdc3c7', '#e74c3c'])  # 灰 vs 红

    # 3. 绘制个体散点 (Swarm/Strip Plot Effect)
    # 给每个点加一点点随机抖动，防止重叠
    jitter = np.random.normal(0, 0.04, size=len(drops_rec))
    plt.scatter(np.zeros_like(drops_rec) + jitter, drops_rec, color='#555555', alpha=0.4, s=15, zorder=3)
    plt.scatter(np.ones_like(drops_emd) + jitter, drops_emd, color='#800000', alpha=0.4, s=15, zorder=3)

    # 4. 连接线 (Pairing Lines) - 可选，展示同一个人的变化
    # 如果想图看起来更整洁，可以注释掉这部分；如果想强调配对关系，保留它
    for i in range(len(drops_rec)):
        plt.plot([0 + jitter[i], 1 + jitter[i]], [drops_rec[i], drops_emd[i]],
                 color='gray', alpha=0.1, linewidth=0.5)

    # 5. 绘制显著性标记 (Significance Bracket)
    if p_val < 0.05:
        h = max(max(drops_rec), max(drops_emd)) * 1.05  # 横线高度
        plt.plot([0, 0, 1, 1], [h, h + 0.5, h + 0.5, h], lw=1.5, c='k')  # 画括弧

        # 确定星号数量
        if p_val < 0.001:
            sig_symbol = '*** (p<0.001)'
        elif p_val < 0.01:
            sig_symbol = '**'
        else:
            sig_symbol = '*'

        plt.text(0.5, h + 0.6, sig_symbol, ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 6. 数值标注
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height / 2,
                 f'{height:.2f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    # 装饰
    plt.ylabel('Accuracy Drop (%)', fontsize=12)
    plt.xticks(x_pos, labels, fontsize=12)
    plt.title(f'Faithfulness Evaluation (N={len(rows_to_save)})\nImpact Ratio: {ratio:.2f}x', fontsize=14)
    plt.ylim(bottom=0, top=max(max(drops_rec), max(drops_emd)) * 1.2)  # 留出顶部空间给星号
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # 保存
    plt.savefig(SAVE_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"\n✅ 绘图完成！已保存至 {SAVE_PLOT_PATH}")
    # plt.show()


if __name__ == "__main__":
    run_e5()