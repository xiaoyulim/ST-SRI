"""
实验一：EMD直接量化
===================
直接计算 sEMG onset vs glove(机械运动) onset 的时间差

需要 scipy.io 读取 .mat 文件
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

# ========== 配置 ==========
DATA_ROOT = "./data"
RESULT_DIR = "./results/emd_direct"
PEAKS_E3_PATH = "./subject_peaks_e3.json"

FS = 2000  # 采样率
EMG_BAND = (20, 450)  # sEMG 带通滤波
GLOVE_BAND = (0.5, 10)  # 手套低频带通（运动相关）
WINDOW_MS = 50  # RMS平滑窗口
THRESHOLD_RATIO = 2.0  # EMG检测阈值
GLOVE_THRESHOLD_RATIO = 1.5  # 手套检测阈值
EMD_MIN_MS, EMD_MAX_MS = 20, 150  # 合理EMD范围

os.makedirs(RESULT_DIR, exist_ok=True)


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def smooth_rms(signal, fs, window_ms):
    window_pts = int(window_ms * fs / 1000)
    if window_pts < 2:
        return signal
    return np.sqrt(uniform_filter1d(signal ** 2, window_pts))


def detect_onset(signal, fs, threshold_ratio, smooth_sigma=2.0):
    sig_smooth = gaussian_filter1d(signal.astype(float), smooth_sigma)
    
    baseline = np.median(sig_smooth[:int(2 * fs)])
    std = np.std(sig_smooth[:int(2 * fs)])
    threshold = baseline + threshold_ratio * std
    
    above = sig_smooth > threshold
    crossings = np.where(np.diff(above.astype(int)) == 1)[0]
    
    min_gap = int(150 * fs / 1000)
    if len(crossings) == 0:
        return np.array([])
    
    merged = [crossings[0]]
    for c in crossings[1:]:
        if c - merged[-1] > min_gap:
            merged.append(c)
    return np.array(merged)


def process_subject(sub_id):
    import scipy.io
    
    e2_path = os.path.join(DATA_ROOT, f"S{sub_id}_E2_A1.mat")
    if not os.path.exists(e2_path):
        return None
    
    print(f">>> 处理 S{sub_id}...")
    
    try:
        mat = scipy.io.loadmat(e2_path)
    except Exception as e:
        print(f"  S{sub_id}: 加载失败 - {e}")
        return None
    
    emg = mat['emg']
    glove = mat['glove'][:, :22]
    stimulus = mat['stimulus'].flatten()
    restimulus = mat['restimulus'].flatten()
    
    # 滤波 EMG
    b, a = butter_bandpass(EMG_BAND[0], EMG_BAND[1], FS, order=4)
    try:
        emg_filt = filtfilt(b, a, emg, axis=0)
    except:
        emg_filt = emg
    
    # 滤波 Glove
    b_g, a_g = butter_bandpass(GLOVE_BAND[0], GLOVE_BAND[1], FS, order=2)
    try:
        glove_filt = filtfilt(b_g, a_g, glove, axis=0)
    except:
        glove_filt = glove
    
    emg_energy = np.mean(emg_filt ** 2, axis=1)
    glove_energy = np.mean(glove_filt ** 2, axis=1)
    emg_energy = smooth_rms(emg_energy, FS, WINDOW_MS)
    glove_energy = smooth_rms(glove_energy, FS, WINDOW_MS)
    
    results = []
    unique_labels = np.unique(stimulus)
    unique_labels = unique_labels[unique_labels > 0]
    
    for label in unique_labels:
        mask = (stimulus == label)
        indices = np.where(mask)[0]
        if len(indices) < 10:
            continue
        
        for rep in np.unique(restimulus[mask]):
            if rep == 0:
                continue
            rep_mask = (restimulus == rep)
            rep_indices = np.where(rep_mask)[0]
            if len(rep_indices) < 50:
                continue
            
            start_idx = max(0, rep_indices[0] - 500)
            end_idx = min(len(emg), rep_indices[-1] + 500)
            
            emg_seg = emg_energy[start_idx:end_idx]
            glove_seg = glove_energy[start_idx:end_idx]
            
            emg_onsets = detect_onset(emg_seg, FS, THRESHOLD_RATIO)
            glove_onsets = detect_onset(glove_seg, FS, GLOVE_THRESHOLD_RATIO)
            
            if len(emg_onsets) == 0 or len(glove_onsets) == 0:
                continue
            
            emd_ms = (glove_onsets[0] - emg_onsets[0]) * 1000 / FS
            
            if EMD_MIN_MS < emd_ms < EMD_MAX_MS:
                results.append({
                    "subject": sub_id,
                    "label": int(label),
                    "repetition": int(rep),
                    "emg_onset_ms": float(emg_onsets[0] * 1000 / FS),
                    "glove_onset_ms": float(glove_onsets[0] * 1000 / FS),
                    "emd_ms": float(emd_ms)
                })
    
    return results


def analyze_and_plot(all_results):
    if not all_results:
        print("无有效数据")
        return
    
    subject_emds = {}
    for r in all_results:
        sub = r['subject']
        if sub not in subject_emds:
            subject_emds[sub] = []
        subject_emds[sub].append(r['emd_ms'])
    
    subject_means = {s: np.mean(vals) for s, vals in subject_emds.items()}
    valid_subs = [s for s, vals in subject_emds.items() 
                  if len(vals) >= 3 and EMD_MIN_MS < np.mean(vals) < EMD_MAX_MS]
    
    all_emds = np.array([r['emd_ms'] for r in all_results])
    global_mean = np.mean(all_emds)
    global_std = np.std(all_emds)
    
    print("\n" + "=" * 60)
    print("EMD 直接量化结果")
    print("=" * 60)
    print(f"总 trial 数: {len(all_results)}")
    print(f"有效受试者: {len(valid_subs)}/{len(subject_emds)}")
    print(f"全局 EMD: {global_mean:.1f} +/- {global_std:.1f} ms")
    
    # 对比 E3
    e3_peaks = {}
    if os.path.exists(PEAKS_E3_PATH):
        with open(PEAKS_E3_PATH) as f:
            e3_peaks = {int(k): float(v) for k, v in json.load(f).items()}
    
    common_subs = sorted(set(subject_means.keys()) & set(e3_peaks.keys()))
    if common_subs:
        e1_means = [subject_means[s] for s in common_subs]
        e3_vals = [e3_peaks[s] for s in common_subs]
        corr = np.corrcoef(e1_means, e3_vals)[0, 1]
        print(f"\n与 E3 ST-SRI 对比 (N={len(common_subs)}):")
        print(f"  直接 EMD: {np.mean(e1_means):.1f} ms")
        print(f"  ST-SRI: {np.mean(e3_vals):.1f} ms")
        print(f"  相关系数: r = {corr:.3f}")
    
    # 保存
    summary = {
        "n_trials": len(all_results),
        "n_subjects": len(subject_emds),
        "valid_subjects": len(valid_subs),
        "global_mean_ms": float(global_mean),
        "global_std_ms": float(global_std),
        "subject_means": {str(k): float(v) for k, v in subject_means.items()},
    }
    with open(os.path.join(RESULT_DIR, "emd_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.hist(all_emds, bins=30, alpha=0.7, color='#2196F3', edgecolor='black')
    ax1.axvline(global_mean, color='red', linestyle='--', label=f'Mean={global_mean:.1f} ms')
    ax1.axvspan(global_mean - global_std, global_mean + global_std, alpha=0.2, color='red')
    ax1.set_xlabel('EMD (ms)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'EMD Distribution (N={len(all_results)})', fontsize=13)
    ax1.legend()
    
    ax2 = axes[1]
    sub_labels = sorted(subject_means.keys())
    means = [subject_means[s] for s in sub_labels]
    stds = [np.std(subject_emds[s]) for s in sub_labels]
    colors = ['#4CAF50' if EMD_MIN_MS < m < EMD_MAX_MS else '#F44336' for m in means]
    ax2.barh(range(len(sub_labels)), means, xerr=stds, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(sub_labels)))
    ax2.set_yticklabels([f'S{s}' for s in sub_labels], fontsize=8)
    ax2.set_xlabel('EMD (ms)', fontsize=12)
    ax2.set_title('Per-Subject EMD', fontsize=13)
    ax2.axvline(EMD_MIN_MS, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(EMD_MAX_MS, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "emd_distribution.png"), dpi=300)
    print(f"\n图像已保存至 {RESULT_DIR}")
    plt.close()


def main():
    print("=" * 40)
    print("实验一：EMD 直接量化")
    print("=" * 40)
    
    all_results = []
    for sub_id in range(1, 41):
        result = process_subject(sub_id)
        if result:
            all_results.extend(result)
            print(f"  S{sub_id}: {len(result)} trials")
    
    print(f"\n处理完成，总 {len(all_results)} trials")
    
    with open(os.path.join(RESULT_DIR, "emd_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    analyze_and_plot(all_results)


if __name__ == "__main__":
    main()
