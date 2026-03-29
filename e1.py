import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from common import NinaProDataset, LSTMModel, ST_SRI_Interpreter, DEVICE, FS

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================= 配置 =================
SUB_ID = 1
WINDOW_MS = 300
CHECKPOINT_PATH = f"./checkpoints_2000hz/best_model_S{SUB_ID}.pth"
SAVE_PATH = "results/e1.png"


# ========================================

class ElementWiseInterpreter(ST_SRI_Interpreter):
    """用于消融实验的单元素掩码解释器"""

    def scan_element_wise(self, x, target_channel=0, max_lag_ms=150):
        T, C = x.shape
        max_lag = int(max_lag_ms * FS / 1000)
        lags = np.arange(1, max_lag)
        t_curr = T - 1

        with torch.no_grad():
            logits = self.model(x.unsqueeze(0))
            target_cls = torch.argmax(logits[0]).item()
            f_both = torch.softmax(logits, dim=1)[0, target_cls].item()

        batch_input = []
        for tau in lags:
            t_lag = t_curr - tau
            # 单通道遮挡
            xt = x.clone();
            xt[t_curr, target_channel] = self.baseline[t_curr, target_channel]
            xtau = x.clone();
            xtau[t_lag, target_channel] = self.baseline[t_lag, target_channel]
            xnone = x.clone()
            xnone[t_curr, target_channel] = self.baseline[t_curr, target_channel]
            xnone[t_lag, target_channel] = self.baseline[t_lag, target_channel]
            batch_input.extend([xt, xtau, xnone])

        batch_tensor = torch.stack(batch_input).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(batch_tensor), dim=1)[:, target_cls].cpu().numpy()

        sii = f_both - probs.reshape(-1, 3)[:, 0] - probs.reshape(-1, 3)[:, 1] + probs.reshape(-1, 3)[:, 2]
        return lags * (1000 / FS), np.maximum(sii, 0)


def run_e1_final():
    os.makedirs("./results", exist_ok=True)
    print(">>> [E1] 正在进行坐标轴反转消融实验...")

    ds = NinaProDataset("./data", SUB_ID, window_ms=WINDOW_MS, target_fs=FS)
    model = LSTMModel().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False))
    model.eval()

    # 寻找激活样本
    x_sample = ds[500][0].to(DEVICE)  # 选取一个样本
    bg_data = ds.data[:ds.window_len].unsqueeze(0).to(DEVICE)

    # 1. 计算两种策略
    interpreter = ElementWiseInterpreter(model, bg_data)
    lags, syn_v, _ = interpreter.scan_fast(x_sample, max_lag_ms=150)  # 注意解构 3 个值
    _, syn_e = interpreter.scan_element_wise(x_sample, target_channel=0, max_lag_ms=150)

    # 2. 坐标轴反转逻辑：将 Lag 转换为 相对于 0 的负值 (过去 -> 现在)
    # 加上 np.array() 以避免 list 负号报错
    time_rel = -np.array(lags)[::-1]
    v_plot = syn_v[::-1]
    e_plot = syn_e[::-1]

    # 归一化
    v_plot = (v_plot - v_plot.min()) / (v_plot.max() - v_plot.min() + 1e-9)
    e_plot = (e_plot - e_plot.min()) / (e_plot.max() - e_plot.min() + 1e-9)

    # 3. 统计指标
    def calc_smooth(arr): return np.std(np.diff(arr))

    smooth_v = calc_smooth(v_plot)
    smooth_e = calc_smooth(e_plot)

    # 4. 绘图
    plt.figure(figsize=(10, 6), dpi=150)

    # 增加平滑系数以展示趋势
    plt.plot(time_rel, gaussian_filter1d(v_plot, 3),
             color='#1f77b4', linewidth=3, label='Vector-wise (ST-SRI)')

    # 注意：这里使用了修复后的长度切片逻辑 time_rel[:len(e_plot)]
    plt.plot(time_rel[:len(e_plot)], gaussian_filter1d(e_plot, sigma=3),
             color='#ff7f0e', linestyle='--', linewidth=2, alpha=0.7, label='Element-wise')

    # --- 样式调整区 ---
    # 1. 标题和轴标签字体增大
    plt.title("E1: Ablation Study - Past-to-Present Causality Tracking", fontsize=18, weight='bold')
    plt.xlabel("Time relative to current action (ms)", fontsize=15)
    plt.ylabel("Normalized Synergy Index", fontsize=15)

    # 2. 坐标刻度字体增大
    plt.tick_params(axis='both', which='major', labelsize=13)

    plt.axvline(0, color='black', linestyle='-', alpha=0.3)  # 标注当前时刻
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(-150, 0)

    # 3. 图例移到左上角，字体增大
    plt.legend(loc='upper left', fontsize=13, framealpha=0.9)

    # 4. 指标框移到左上角（图例正下方），字体增大
    text_str = f"Smoothness (Std Diff):\nVector: {smooth_v:.2e}\nElement: {smooth_e:.2e}"
    # transform=plt.gca().transAxes 坐标系中 (0,1)是左上角
    # 这里设置为 (0.02, 0.75) 大约在图例下方
    plt.text(0.02, 0.75, text_str, transform=plt.gca().transAxes,
             fontsize=13, bbox=dict(facecolor='white', alpha=0.8, edgecolor='#cccccc'))

    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"✅ E1 已保存至: {SAVE_PATH} (样式调整完成)")


if __name__ == "__main__":
    run_e1_final()