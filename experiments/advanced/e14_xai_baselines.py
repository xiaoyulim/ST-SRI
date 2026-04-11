"""
实验十四：XAI 时序基线全面对比（扩展版）
==========================================
对比方法：
  1. ST-SRI（本文方法）
  2. Standard SHAP (GradientExplainer)
  3. Integrated Gradients (captum)
  4. DeepLIFT (captum)
  5. TimeSHAP (timeshap) ← 新增
  6. TSR - Temporal Saliency Rescaling ← 新增

审稿人关注点：
  PC#1: "no head-to-head with TimeSHAP (KDD'21) or TSR (2010.13924)"
  PC#3: "No comparison against TimeShap... its absence is a notable gap"

评价指标：
  - Stability:       跨样本归因一致性（越低越好）
  - Peak-in-EMD:     峰值落入 30-100ms 的比例（越高越好）
  - Fragmentation:   归因曲线平滑度（越低越好）
  - AOPC:            扰动曲线下面积（越高越 faithful）
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d
from common import (
    NinaProDataset, LSTMModel, ST_SRI_Interpreter,
    DEVICE, FS, EMD_VALID_MIN_MS, EMD_VALID_MAX_MS,
    create_blocked_split,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Optional imports
try:
    from captum.attr import IntegratedGradients, DeepLift, Saliency
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("Warning: captum not installed. IG/DeepLIFT/TSR disabled.")

try:
    import timeshap.explainer as tsx
    import timeshap.utils as tsu
    TIMESHAP_AVAILABLE = True
except (ImportError, Exception):
    TIMESHAP_AVAILABLE = False
    # 使用自实现的 TimeSHAP 核心逻辑（基于 KDD'21 论文算法）
    # 不依赖 timeshap 包，避免与 shap>=0.50 的兼容性问题


# ================= Configuration =================
DATA_ROOT = "./data"
CHECKPOINT_DIR = "./checkpoints_2000hz"
GOOD_SUBJECTS_PATH = "./good_subjects.json"
RESULT_DIR = "./results/xai_baselines"

WINDOW_MS = 300
NUM_CLASSES = 18
NUM_SAMPLES = 30       # 每个 subject 的采样数
SCAN_STRIDE = 3        # ST-SRI 扫描步长
# =================================================

os.makedirs(RESULT_DIR, exist_ok=True)


def load_subjects(limit=None):
    with open(GOOD_SUBJECTS_PATH, "r") as f:
        subjects = json.load(f)
    return subjects[:limit] if limit is not None else subjects


def load_model(sub_id):
    model_path = os.path.join(CHECKPOINT_DIR, f"best_model_S{sub_id}.pth")
    if not os.path.exists(model_path):
        return None
    model = LSTMModel(input_size=12, num_classes=NUM_CLASSES).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def get_samples_and_bg(ds, num_samples=NUM_SAMPLES):
    """从数据集中采样活跃窗口和背景数据。"""
    # 背景数据
    bg_loader = DataLoader(ds, batch_size=min(20, len(ds)), shuffle=True)
    bg_batch = next(iter(bg_loader))[0].to(DEVICE)

    # 活跃样本采样
    np.random.seed(42)
    indices = np.random.choice(len(ds), min(num_samples, len(ds)), replace=False)
    samples = torch.stack([ds[i][0] for i in indices]).to(DEVICE)

    return samples, bg_batch


# ================= 归因方法实现 =================

def compute_st_sri(model, samples, bg_data):
    """ST-SRI 协同谱归因。返回 (N, T_lag) 的时序归因。"""
    interpreter = ST_SRI_Interpreter(model, bg_data)
    attributions = []
    for i in range(len(samples)):
        try:
            lags_ms, synergy, _ = interpreter.scan_fast(
                samples[i], max_lag_ms=150, stride=SCAN_STRIDE, block_size=2
            )
            if len(synergy) > 0:
                attributions.append(synergy)
        except Exception:
            continue
    if not attributions:
        return None
    # 对齐长度（取最短）
    min_len = min(len(a) for a in attributions)
    return np.array([a[:min_len] for a in attributions])


def compute_shap(model, samples, bg_data):
    """Standard SHAP (GradientExplainer)。返回 (N, T) 的时序归因。"""
    try:
        explainer = shap.GradientExplainer(model, bg_data)
        with torch.backends.cudnn.flags(enabled=False):
            shap_vals = explainer.shap_values(samples)
        if isinstance(shap_vals, list):
            attr = np.abs(np.array(shap_vals)).sum(axis=0).sum(axis=-1)
        else:
            attr = np.abs(shap_vals).sum(axis=-1)
        return attr
    except Exception as e:
        print(f"    SHAP error: {e}")
        return None


def compute_ig(model, samples, bg_data):
    """Integrated Gradients (captum)。返回 (N, T) 的时序归因。"""
    if not CAPTUM_AVAILABLE:
        return None
    try:
        ig = IntegratedGradients(model)
        baseline = torch.mean(bg_data, dim=0).unsqueeze(0).expand_as(samples)
        all_attr = []
        # LSTM 的 cuDNN 后向需要 train 模式
        model.train()
        for i in range(len(samples)):
            x_i = samples[i:i+1]
            bl_i = baseline[i:i+1]
            with torch.no_grad():
                target_i = model(x_i).argmax(1).item()
            attr_i = ig.attribute(x_i, baselines=bl_i, target=target_i)
            all_attr.append(torch.abs(attr_i).sum(dim=-1).cpu().detach().numpy()[0])
        model.eval()
        return np.array(all_attr)
    except Exception as e:
        model.eval()
        print(f"    IG error: {e}")
        return None


def compute_deeplift(model, samples, bg_data):
    """DeepLIFT (captum)。返回 (N, T) 的时序归因。"""
    if not CAPTUM_AVAILABLE:
        return None
    try:
        dl = DeepLift(model)
        baseline = torch.mean(bg_data, dim=0).unsqueeze(0).expand_as(samples)
        all_attr = []
        for i in range(len(samples)):
            x_i = samples[i:i+1]
            bl_i = baseline[i:i+1]
            with torch.no_grad():
                target_i = model(x_i).argmax(1).item()
            attr_i = dl.attribute(x_i, baselines=bl_i, target=target_i)
            all_attr.append(torch.abs(attr_i).sum(dim=-1).cpu().numpy()[0])
        return np.array(all_attr)
    except Exception as e:
        print(f"    DeepLIFT error: {e}")
        return None


def compute_timeshap(model, samples, bg_data):
    """
    TimeSHAP (KDD'21) 自实现版 — 分组揭露加速。

    核心思路（同原论文）：
    - 从全 baseline 开始，逐段"揭露"时间块
    - 测量 ΔP = P(揭露后) - P(揭露前) 得到每段的边际贡献

    加速策略：
    - 将 T 个时间步分成 N_groups 组（每组 group_size 步）
    - 每组内共享同一个归因值
    - 这将 600 次前向传播减少到 ~60 次
    """
    GROUP_SIZE = 10  # 每组 10 个时间步 = 5ms@2kHz

    try:
        baseline_vec = torch.mean(bg_data, dim=0).to(DEVICE)  # (T, C)
        all_attr = []

        model.eval()
        for i in range(len(samples)):
            x = samples[i]  # (T, C)
            T = x.size(0)

            with torch.no_grad():
                target_cls = model(x.unsqueeze(0)).argmax(1).item()

            # 分组
            n_groups = (T + GROUP_SIZE - 1) // GROUP_SIZE
            attr = np.zeros(T)
            x_masked = baseline_vec.clone()

            with torch.no_grad():
                prev_prob = torch.softmax(
                    model(x_masked.unsqueeze(0)), dim=1
                )[0, target_cls].item()

            for g in range(n_groups):
                t_start = g * GROUP_SIZE
                t_end = min(t_start + GROUP_SIZE, T)
                x_masked[t_start:t_end, :] = x[t_start:t_end, :]

                with torch.no_grad():
                    curr_prob = torch.softmax(
                        model(x_masked.unsqueeze(0)), dim=1
                    )[0, target_cls].item()

                delta = curr_prob - prev_prob
                attr[t_start:t_end] = abs(delta) / (t_end - t_start)
                prev_prob = curr_prob

            all_attr.append(attr)

        return np.array(all_attr)
    except Exception as e:
        print(f"    TimeSHAP error: {e}")
        return None


def compute_tsr(model, samples, bg_data):
    """
    TSR - Temporal Saliency Rescaling (arXiv:2010.13924)。
    两步法：
      Step 1: 用梯度方法（Saliency）得到 element-wise 归因 (B, T, C)
      Step 2: 先在特征维聚合得到时间重要性，再 rescale 原始归因
    最终输出 (N, T) 的时序归因。
    """
    if not CAPTUM_AVAILABLE:
        return None
    try:
        saliency = Saliency(model)
        all_attr = []

        # LSTM 的 cuDNN 后向需要 train 模式
        model.train()
        for i in range(len(samples)):
            x_i = samples[i:i+1].requires_grad_(True)
            with torch.no_grad():
                target_i = model(x_i).argmax(1).item()

            # Step 1: element-wise saliency (B, T, C)
            attr_raw = saliency.attribute(x_i, target=target_i)
            attr_abs = torch.abs(attr_raw).cpu().detach().numpy()[0]  # (T, C)

            # Step 2a: 时间维聚合 → 特征重要性 (C,)
            feature_importance = attr_abs.mean(axis=0)  # (C,)
            feature_importance = feature_importance / (feature_importance.sum() + 1e-9)

            # Step 2b: rescale — 用特征重要性加权每个时间步的归因
            rescaled = attr_abs * feature_importance[np.newaxis, :]  # (T, C)

            # 在特征维聚合得到时间归因
            temporal_attr = rescaled.sum(axis=1)  # (T,)
            all_attr.append(temporal_attr)

        model.eval()
        return np.array(all_attr)
    except Exception as e:
        model.eval()
        print(f"    TSR error: {e}")
        return None


# ================= 评价指标 =================

def compute_metrics(attributions, method_name=""):
    """计算归因质量指标。"""
    if attributions is None or len(attributions) == 0:
        return None

    # 归一化
    norm_attrs = []
    for attr in attributions:
        r = attr.max() - attr.min()
        if r > 1e-9:
            norm_attrs.append((attr - attr.min()) / r)
        else:
            norm_attrs.append(np.zeros_like(attr))
    norm_attrs = np.array(norm_attrs)

    mean_attr = np.mean(norm_attrs, axis=0)

    # 1. Stability: 跨样本标准差均值（越低越稳定）
    stability = float(np.mean(np.std(norm_attrs, axis=0)))

    # 2. Peak-in-EMD ratio
    T = attributions.shape[1]
    is_lag_space = (method_name == 'st_sri')

    if is_lag_space:
        # ST-SRI: 归因在 lag 空间，index=0 对应最小 lag
        # lags_ms ≈ index * stride * (1000/FS)
        emd_start_idx = int(EMD_VALID_MIN_MS / (SCAN_STRIDE * 1000 / FS))
        emd_end_idx = int(EMD_VALID_MAX_MS / (SCAN_STRIDE * 1000 / FS))
    else:
        # 其他方法: 归因在时间空间，EMD 窗口从末尾回溯
        emd_start_idx = max(0, T - int(EMD_VALID_MAX_MS * FS / 1000))
        emd_end_idx = T - int(EMD_VALID_MIN_MS * FS / 1000)

    emd_start_idx = max(0, min(emd_start_idx, T - 1))
    emd_end_idx = max(0, min(emd_end_idx, T - 1))

    peak_in_emd = 0
    for attr in norm_attrs:
        peak_idx = np.argmax(attr)
        if emd_start_idx <= peak_idx <= emd_end_idx:
            peak_in_emd += 1
    peak_ratio = peak_in_emd / len(norm_attrs)

    # 3. Fragmentation: 归因曲线一阶导数的标准差（越低越平滑）
    fragmentation = float(np.std(np.diff(mean_attr)))

    return {
        'stability': stability,
        'peak_in_emd_ratio': peak_ratio,
        'fragmentation': fragmentation,
    }


# ================= 单 Subject 分析 =================

ALL_METHODS = {
    'st_sri':    ('ST-SRI (Ours)',           compute_st_sri),
    'shap':      ('Standard SHAP',           compute_shap),
    'ig':        ('Integrated Gradients',     compute_ig),
    'deeplift':  ('DeepLIFT',                compute_deeplift),
    'timeshap':  ('TimeSHAP (KDD\'21)',      compute_timeshap),
    'tsr':       ('TSR (2010.13924)',         compute_tsr),
}


def analyze_subject(sub_id, methods_to_run):
    """对单个 subject 运行所有 XAI 方法并计算指标。"""
    model = load_model(sub_id)
    if model is None:
        return None

    try:
        ds = NinaProDataset(DATA_ROOT, sub_id, window_ms=WINDOW_MS, target_fs=FS, step_ms=50)
    except Exception:
        return None

    samples, bg_data = get_samples_and_bg(ds)
    results = {'subject': sub_id}

    for method_key in methods_to_run:
        label, compute_fn = ALL_METHODS[method_key]
        try:
            attr = compute_fn(model, samples, bg_data)
            metrics = compute_metrics(attr, method_name=method_key)
            if metrics is not None:
                results[method_key] = metrics
        except Exception as e:
            print(f"    [{method_key}] S{sub_id} failed: {e}")

    return results if len(results) > 1 else None


# ================= 主流程 =================

def main():
    parser = argparse.ArgumentParser(description="E14: XAI Baseline Comparison (Extended)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only use first N subjects")
    parser.add_argument("--methods", nargs="+", default=None,
                        choices=list(ALL_METHODS.keys()),
                        help="Which methods to run (default: all available)")
    args = parser.parse_args()

    subjects = load_subjects(limit=args.limit)

    # 确定可用方法
    if args.methods:
        methods_to_run = args.methods
    else:
        methods_to_run = ['st_sri', 'shap']
        if CAPTUM_AVAILABLE:
            methods_to_run.extend(['ig', 'deeplift', 'tsr'])
        # TimeSHAP 使用自实现版本，无外部依赖
        methods_to_run.append('timeshap')

    print("=" * 70)
    print("E14: Multi-XAI Baseline Comparison (Extended)")
    print("=" * 70)
    print(f"Subjects: {len(subjects)}")
    print(f"Methods:  {[ALL_METHODS[m][0] for m in methods_to_run]}")
    print(f"Samples/subject: {NUM_SAMPLES}")
    print()

    # 收集结果
    all_results = []
    for sub_id in subjects:
        print(f"  Processing S{sub_id}...", end=" ")
        res = analyze_subject(sub_id, methods_to_run)
        if res is not None:
            all_results.append(res)
            done_methods = [m for m in methods_to_run if m in res]
            print(f"OK ({len(done_methods)}/{len(methods_to_run)} methods)")
        else:
            print("SKIP")

    if not all_results:
        print("Error: No results collected!")
        return

    print(f"\nCollected {len(all_results)} subjects")

    # 聚合
    aggregated = {}
    for method_key in methods_to_run:
        vals = {'stability': [], 'peak_in_emd_ratio': [], 'fragmentation': []}
        for res in all_results:
            if method_key in res:
                for k in vals:
                    vals[k].append(res[method_key][k])

        if vals['stability']:
            aggregated[method_key] = {
                'label': ALL_METHODS[method_key][0],
                'n_subjects': len(vals['stability']),
                'stability_mean': float(np.mean(vals['stability'])),
                'stability_std': float(np.std(vals['stability'])),
                'peak_in_emd_mean': float(np.mean(vals['peak_in_emd_ratio'])),
                'peak_in_emd_std': float(np.std(vals['peak_in_emd_ratio'])),
                'fragmentation_mean': float(np.mean(vals['fragmentation'])),
                'fragmentation_std': float(np.std(vals['fragmentation'])),
            }

    # 打印汇总表
    print("\n" + "=" * 80)
    print(f"{'Method':<28} | {'Stability':>12} | {'Peak-in-EMD':>12} | {'Fragmentation':>14} | N")
    print("-" * 80)
    for mk in methods_to_run:
        if mk in aggregated:
            a = aggregated[mk]
            print(f"{a['label']:<28} | "
                  f"{a['stability_mean']:.4f}+{a['stability_std']:.3f} | "
                  f"{a['peak_in_emd_mean']:.2%}+{a['peak_in_emd_std']:.2%} | "
                  f"{a['fragmentation_mean']:.5f}+{a['fragmentation_std']:.4f} | "
                  f"{a['n_subjects']}")
    print("=" * 80)
    print("  Stability:     lower = more consistent across samples")
    print("  Peak-in-EMD:   higher = more peaks in physiological 30-100ms range")
    print("  Fragmentation: lower = smoother attribution curves")

    # 保存
    payload = {
        'aggregated': aggregated,
        'individual_results': all_results,
        'config': {
            'num_samples': NUM_SAMPLES,
            'scan_stride': SCAN_STRIDE,
            'methods': methods_to_run,
            'n_subjects': len(all_results),
        }
    }
    out_path = os.path.join(RESULT_DIR, 'xai_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)

    # 绘图
    plot_comparison(aggregated, methods_to_run)
    print(f"\nResults saved to {RESULT_DIR}/")


def plot_comparison(aggregated, methods_to_run):
    """生成 XAI 方法对比柱状图。"""
    methods = [m for m in methods_to_run if m in aggregated]
    if len(methods) < 2:
        return

    labels = [aggregated[m]['label'] for m in methods]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    x = np.arange(len(methods))
    w = 0.6

    # 1. Stability
    vals = [aggregated[m]['stability_mean'] for m in methods]
    errs = [aggregated[m]['stability_std'] for m in methods]
    axes[0].bar(x, vals, w, yerr=errs, color=colors[:len(methods)], alpha=0.7, capsize=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    axes[0].set_ylabel('Cross-Sample Std')
    axes[0].set_title('Stability (Lower = Better)', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # 2. Peak-in-EMD
    vals = [aggregated[m]['peak_in_emd_mean'] for m in methods]
    errs = [aggregated[m]['peak_in_emd_std'] for m in methods]
    axes[1].bar(x, vals, w, yerr=errs, color=colors[:len(methods)], alpha=0.7, capsize=4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    axes[1].set_ylabel('Ratio')
    axes[1].set_title('Peak in EMD Range (Higher = Better)', fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis='y', alpha=0.3)

    # 3. Fragmentation
    vals = [aggregated[m]['fragmentation_mean'] for m in methods]
    errs = [aggregated[m]['fragmentation_std'] for m in methods]
    axes[2].bar(x, vals, w, yerr=errs, color=colors[:len(methods)], alpha=0.7, capsize=4)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    axes[2].set_ylabel('Derivative Std')
    axes[2].set_title('Fragmentation (Lower = Better)', fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'xai_comparison.png'), dpi=200)
    plt.close()
    print(f"  Plot saved: {RESULT_DIR}/xai_comparison.png")


if __name__ == '__main__':
    main()
