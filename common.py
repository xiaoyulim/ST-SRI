import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS = 2000


class NinaProDataset(Dataset):
    def __init__(self, root_dir, subject_id, window_ms=300, target_fs=2000, step_ms=50,
                 anticipation_ms=0):
        self.root = root_dir
        self.fs = target_fs
        self.window_len = int(window_ms * self.fs / 1000)
        self.stride = int(step_ms * self.fs / 1000)
        self.anticipation_steps = int(anticipation_ms * self.fs / 1000)
        d_path = os.path.join(root_dir, f"S{subject_id}_data.npy")
        l_path = os.path.join(root_dir, f"S{subject_id}_label.npy")
        raw_data = np.load(d_path)
        raw_labels = np.load(l_path)
        self.data = (raw_data - np.mean(raw_data, axis=0)) / (np.std(raw_data, axis=0) + 1e-6)
        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(raw_labels).long()
        # 有提前量时，末尾需要预留 anticipation_steps 个样本作为标签参考
        self.num_samples = (len(self.data) - self.window_len - self.anticipation_steps) // self.stride + 1

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_len
        if self.anticipation_steps == 0:
            # 原始行为：取窗口内众数标签
            label = torch.mode(self.labels[start:end]).values
        else:
            # 提前预测：取窗口结束后 anticipation_steps 处的标签
            label_pos = end - 1 + self.anticipation_steps
            label = self.labels[label_pos]
        return self.data[start:end, :], label


class LSTMModel(nn.Module):
    # ... (这部分保持不变)
    def __init__(self, input_size=12, hidden_size=256, num_layers=3, num_classes=18, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ================= 3. ST-SRI 解释器 (Interpreter) =================
class ST_SRI_Interpreter:
    def __init__(self, model, background_data):
        self.model = model.to(DEVICE).eval()
        self.baseline = torch.mean(background_data, dim=0).to(DEVICE)
        self.T, self.C = self.baseline.shape

    def get_score_batch(self, x_batch, target_cls=None):
        """
        专用批量打分函数
        target_cls: 如果为 None，自动选择预测概率最高的类
        """
        with torch.no_grad():
            logits = self.model(x_batch)
            probs = torch.softmax(logits, dim=1)

            # 如果没有指定目标类，就取第一个样本预测最高的类
            if target_cls is None:
                target_cls = torch.argmax(probs[0]).item()

            # 返回该类别的概率 (B, )
            return probs[:, target_cls].cpu().numpy()

    def scan_fast(self, x, max_lag_ms=150, stride=1, block_size=2):
        """
        极速版扫描：利用 GPU 的并行能力，一次算完所有 Lag
        """
        # 1. 预计算所有参数
        max_lag_points = int(max_lag_ms * (FS / 1000))
        curr_t = self.T - 1
        max_lag_points = min(max_lag_points, curr_t)

        lags = list(range(stride, max_lag_points + 1, stride))
        if not lags: return [], [], []

        N_lags = len(lags)

        # 2. 构造超级 Batch
        x_base = x.unsqueeze(0).repeat(N_lags, 1, 1)

        x_lag = x_base.clone()
        x_curr = x_base.clone()
        x_none = x_base.clone()

        # 构建 Mask (注意：这里正确使用了 block_size)
        for i, tau in enumerate(lags):
            # 遮挡区间计算
            # 我们希望遮挡 [t - block_size + 1, t] 这一段，包含 t 本身
            t_end = curr_t + 1
            t_start = t_end - block_size

            # 边界保护
            t_start = max(0, t_start)

            # 遮挡 t (curr)
            x_lag[i, t_start:t_end, :] = self.baseline[t_start:t_end, :]
            x_none[i, t_start:t_end, :] = self.baseline[t_start:t_end, :]

            # 遮挡 t-tau (lag)
            t_prev_end = curr_t - tau + 1
            t_prev_start = t_prev_end - block_size

            if t_prev_start >= 0:
                x_curr[i, t_prev_start:t_prev_end, :] = self.baseline[t_prev_start:t_prev_end, :]
                x_none[i, t_prev_start:t_prev_end, :] = self.baseline[t_prev_start:t_prev_end, :]

        # 3. 确定目标类别 (Target Class)
        # 我们基于原始输入确定模型想预测什么，确保所有 Batch 关注同一个类
        with torch.no_grad():
            orig_logits = self.model(x.unsqueeze(0))
            target_cls = torch.argmax(orig_logits[0]).item()

        # 4. 批量推理 (传入 target_cls，解决报错！)
        f_both = self.get_score_batch(x.unsqueeze(0), target_cls)[0]

        s_lag = self.get_score_batch(x_lag, target_cls)
        s_curr = self.get_score_batch(x_curr, target_cls)
        s_none = self.get_score_batch(x_none, target_cls)

        # 5. SII 计算
        interactions = f_both - s_lag - s_curr + s_none

        synergy = np.maximum(interactions, 0)
        redundancy = np.minimum(interactions, 0)

        lags_ms = [l * (1000 / FS) for l in lags]

        return lags_ms, synergy, redundancy


# ================= 统计工具函数 (从 experiments_improved 提取) =================
def calculate_cohens_d(group1, group2):
    """
    计算 Cohen's d 效应量

    Args:
        group1, group2: 两组数据（列表或数组）

    Returns:
        float: Cohen's d 值
    """
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    # 合并标准差
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d


def interpret_cohens_d(d):
    """
    解释 Cohen's d 效应量大小

    Args:
        d: Cohen's d 值

    Returns:
        str: 效应量解释
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def bootstrap_ci(data, n_bootstrap=10000, ci=95, statistic=np.mean):
    """
    计算 Bootstrap 置信区间

    Args:
        data: 原始数据
        n_bootstrap: Bootstrap 采样次数
        ci: 置信水平（百分比）
        statistic: 统计量函数（默认均值）

    Returns:
        tuple: (lower_bound, upper_bound)
    """
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic(sample))

    lower = np.percentile(bootstrap_stats, (100 - ci) / 2)
    upper = np.percentile(bootstrap_stats, 100 - (100 - ci) / 2)

    return lower, upper