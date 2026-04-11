import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d
from common import ST_SRI_Interpreter, DEVICE

# 修复 OMP 报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================= 配置 =================
FS = 2000  # 与项目一致的采样率
WINDOW_LEN = 600  # 300ms
TARGET_DELAY_MS = 50  # 我们人工注入 50ms 的延迟
TARGET_DELAY_POINTS = int(TARGET_DELAY_MS * FS / 1000)  # 对应 100 个点
SAVE_PATH = "results/e2.png"


# ========================================

# 1. 构造一个“假模型” (Mock Model)
# 逻辑：只有当当前时刻 t 和 50ms 前的信号同时为高时，模型才输出高概率
class MockSynergyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x 形状: (Batch, Time, Channels)
        # 取当前时刻 t (最后一点) 和 t - 100 (50ms前)
        t_curr = x[:, -1, 0]
        t_lag = x[:, -1 - TARGET_DELAY_POINTS, 0]

        # 构造协同非线性关系: y = sigmoid(t_curr * t_lag)
        # 这种乘法关系是标准的“协同（Synergy）”
        score = torch.sigmoid(t_curr * t_lag * 10)

        # 模拟输出 18 个类别的 logits
        out = torch.zeros(x.shape[0], 18).to(x.device)
        out[:, 1] = score * 5  # 假设动作类索引为 1
        return out


def run_e2_simulation():
    os.makedirs("./results", exist_ok=True)
    print(f">>> [E2] 正在运行机理验证：尝试找回人工注入的 {TARGET_DELAY_MS}ms 信号...")

    # 2. 构造仿真信号 (高斯噪声背景)
    # 创建一个 (600, 12) 的样本
    x_sim = torch.randn(WINDOW_LEN, 12).to(DEVICE) * 0.1
    # 在当前点和 50ms 前的点注入强信号
    x_sim[-1, 0] = 2.0
    x_sim[-1 - TARGET_DELAY_POINTS, 0] = 2.0

    # 构造背景数据（全 0 或弱噪声，用于解释器基准）
    bg_data = torch.zeros(1, WINDOW_LEN, 12).to(DEVICE)

    # 3. 初始化解释器
    model = MockSynergyModel().to(DEVICE).eval()
    interpreter = ST_SRI_Interpreter(model, bg_data)

    # 4. 扫描时滞
    # 扫描 0 到 150ms
    lags, syn, _ = interpreter.scan_fast(x_sim, max_lag_ms=150)

    # 5. 绘图
    plt.figure(figsize=(9, 5), dpi=150)

    # 转换坐标轴：过去 -> 现在
    time_rel = -lags[::-1]
    plot_syn = syn[::-1]

    # 归一化
    plot_syn = (plot_syn - plot_syn.min()) / (plot_syn.max() - plot_syn.min() + 1e-9)
    # 轻微平滑
    plot_syn_smooth = gaussian_filter1d(plot_syn, sigma=1.0)

    plt.plot(time_rel, plot_syn_smooth, color='#1f77b4', linewidth=2.5, label='ST-SRI Synergy Index')

    # 标注 Ground Truth (人工注入的 50ms)
    plt.axvline(-TARGET_DELAY_MS, color='#d62728', linestyle='--', linewidth=2,
                label=f'Ground Truth Delay ({TARGET_DELAY_MS}ms)')

    plt.title("E2: Mechanism Validation (Synthetic Synergy)", fontsize=13, weight='bold')
    plt.xlabel("Injected Causal Lag (ms)", fontsize=11)
    plt.ylabel("Normalized Synergy Strength", fontsize=11)
    plt.xlim(-150, 0)
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"✅ E2 仿真完成！图表已保存至: {SAVE_PATH}")
    print(f"   观察波峰是否精准出现在 -{TARGET_DELAY_MS}ms 处。")


if __name__ == "__main__":
    run_e2_simulation()