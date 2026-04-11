import os
import sys
# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d
from common import NinaProDataset, LSTMModel, ST_SRI_Interpreter, DEVICE, FS

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================= Configuration =================
DATA_ROOT = "./data"
CHECKPOINT_DIR = "./checkpoints_2000hz"
RESULT_DIR = "./results/baseline_comparison"

WINDOW_MS = 300
NUM_CLASSES = 18
TARGET_COUNT = 30  # Samples per subject
SMOOTH_SIGMA = 2.0
# =================================================

os.makedirs(RESULT_DIR, exist_ok=True)


class MultiBaselineInterpreter(ST_SRI_Interpreter):
    """Extended ST-SRI interpreter supporting multiple baseline strategies"""

    def __init__(self, model, background_data, baseline_type='channel_mean'):
        """
        Args:
            baseline_type: 'channel_mean', 'zero', 'gaussian_noise', 'global_mean'
        """
        self.model = model.to(DEVICE).eval()
        self.background_data = background_data
        self.baseline_type = baseline_type

        # Compute baseline based on strategy
        if baseline_type == 'channel_mean':
            # Current default: mean across samples for each (time, channel)
            self.baseline = torch.mean(background_data, dim=0).to(DEVICE)
        elif baseline_type == 'zero':
            # Zero vector
            self.baseline = torch.zeros_like(background_data[0]).to(DEVICE)
        elif baseline_type == 'gaussian_noise':
            # Gaussian noise with same std as background data
            std = torch.std(background_data, dim=0)
            self.baseline = torch.randn_like(background_data[0]).to(DEVICE) * std.to(DEVICE)
        elif baseline_type == 'global_mean':
            # Global mean across all samples, time, and channels
            global_mean = torch.mean(background_data)
            self.baseline = torch.ones_like(background_data[0]).to(DEVICE) * global_mean
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")

        self.T, self.C = self.baseline.shape


def analyze_one_subject_with_baseline(sub_id, baseline_type):
    """Analyze single subject with specified baseline strategy"""
    cache_path = f"{RESULT_DIR}/S{sub_id}_{baseline_type}_synergy.npy"
    if os.path.exists(cache_path):
        return np.load(cache_path, allow_pickle=True)

    print(f"  Analyzing S{sub_id} with {baseline_type}...", end='\r')

    try:
        ds = NinaProDataset(DATA_ROOT, sub_id, window_ms=WINDOW_MS, target_fs=FS)
    except:
        return None

    model_path = f"{CHECKPOINT_DIR}/best_model_S{sub_id}.pth"
    if not os.path.exists(model_path):
        return None

    model = LSTMModel(input_size=12, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
    model.eval()

    # Get background data (rest samples)
    rest_loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    all_x, all_y = next(iter(rest_loader))
    rest_mask = (all_y == 0)
    if rest_mask.sum() >= 20:
        bg_data = all_x[rest_mask][:200].to(DEVICE)
    else:
        bg_loader = DataLoader(ds, batch_size=20, shuffle=True)
        bg_data, _ = next(iter(bg_loader))
        bg_data = bg_data.to(DEVICE)

    # Create interpreter with specified baseline
    interpreter = MultiBaselineInterpreter(model, bg_data, baseline_type=baseline_type)

    seq_loader = DataLoader(ds, batch_size=1, shuffle=False)

    synergies = []
    lags_ms = []
    prev_label = 0
    collected_count = 0

    for x, y in seq_loader:
        curr_label = y.item()
        is_onset = (prev_label == 0) and (curr_label != 0)
        prev_label = curr_label

        if is_onset:
            x_tensor = x[0].to(DEVICE)
            try:
                l, s, _ = interpreter.scan_fast(x_tensor, max_lag_ms=150, stride=1, block_size=10)
                synergies.append(s)
                lags_ms = l
                collected_count += 1
            except:
                continue
        if collected_count >= TARGET_COUNT:
            break

    if not synergies:
        return None

    mean_syn = np.mean(synergies, axis=0)
    np.save(cache_path, mean_syn)
    return mean_syn


def compute_spectrum_metrics(synergy, lags_ms):
    """Compute metrics for synergy spectrum"""
    # Find peak in EMD range (-100 to -30 ms)
    time_axis = -np.array(lags_ms)[::-1]
    synergy_reversed = synergy[::-1]

    emd_mask = (time_axis >= -100) & (time_axis <= -30)
    if not np.any(emd_mask):
        return None

    # Peak position and amplitude
    peak_idx = np.argmax(synergy_reversed[emd_mask])
    peak_time = time_axis[emd_mask][peak_idx]
    peak_amplitude = synergy_reversed[emd_mask][peak_idx]

    # Smoothness (lower derivative std = smoother)
    derivative = np.diff(synergy_reversed)
    smoothness = np.std(derivative)

    return {
        'peak_time_ms': float(peak_time),
        'peak_amplitude': float(peak_amplitude),
        'smoothness': float(smoothness)
    }


def run_e13():
    """Main experiment: compare baseline strategies"""
    print("=" * 70)
    print("E13: Multi-Baseline Masking Strategy Comparison")
    print("=" * 70)

    # Load subject list
    if os.path.exists("good_subjects.json"):
        with open("good_subjects.json", "r") as f:
            target_subjects = json.load(f)[:10]  # Use first 10 subjects
    else:
        target_subjects = list(range(1, 11))

    baseline_types = ['channel_mean', 'zero', 'gaussian_noise', 'global_mean']
    baseline_labels = {
        'channel_mean': 'Channel Mean (Current)',
        'zero': 'Zero Vector',
        'gaussian_noise': 'Gaussian Noise',
        'global_mean': 'Global Mean'
    }

    results = {bt: {'synergies': [], 'metrics': []} for bt in baseline_types}

    # Collect data for each baseline type
    for baseline_type in baseline_types:
        print(f"\n>>> Processing baseline: {baseline_labels[baseline_type]}")
        for sub in target_subjects:
            syn = analyze_one_subject_with_baseline(sub, baseline_type)
            if syn is not None:
                results[baseline_type]['synergies'].append(syn)
        print(f"    Collected {len(results[baseline_type]['synergies'])} subjects")

    # Load time axis (should be same for all)
    lags_ms = None
    for bt in baseline_types:
        if results[bt]['synergies']:
            # Reconstruct lags from synergy length
            lags_ms = np.arange(1, len(results[bt]['synergies'][0]) + 1) * (1000 / FS)
            break

    if lags_ms is None:
        print("Error: No data collected")
        return

    # Compute metrics for each baseline
    print("\n" + "=" * 70)
    print("Metrics Summary:")
    print("=" * 70)
    print(f"{'Baseline':<25} | {'Peak (ms)':<12} | {'Amplitude':<12} | {'Smoothness':<12}")
    print("-" * 70)

    for baseline_type in baseline_types:
        synergies = results[baseline_type]['synergies']
        if not synergies:
            continue

        # Normalize each synergy
        norm_syn = []
        for s in synergies:
            s_min, s_max = s.min(), s.max()
            norm_syn.append((s - s_min) / (s_max - s_min + 1e-9))

        mean_syn = np.mean(norm_syn, axis=0)

        # Compute metrics
        metrics = compute_spectrum_metrics(mean_syn, lags_ms)
        if metrics:
            results[baseline_type]['metrics'].append(metrics)
            print(f"{baseline_labels[baseline_type]:<25} | "
                  f"{metrics['peak_time_ms']:>10.1f}ms | "
                  f"{metrics['peak_amplitude']:>12.4f} | "
                  f"{metrics['smoothness']:>12.4f}")

    # Save metrics to JSON
    metrics_summary = {}
    for bt in baseline_types:
        if results[bt]['metrics']:
            metrics_summary[bt] = results[bt]['metrics'][0]

    with open(f"{RESULT_DIR}/metrics_summary.json", 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    # Plot comparison
    plot_baseline_comparison(results, lags_ms, baseline_labels)

    print(f"\n✅ E13 completed! Results saved to {RESULT_DIR}/")


def plot_baseline_comparison(results, lags_ms, baseline_labels):
    """Plot synergy spectra for all baseline types"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    axes = axes.flatten()

    baseline_types = ['channel_mean', 'zero', 'gaussian_noise', 'global_mean']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    time_axis = -lags_ms[::-1]

    for idx, baseline_type in enumerate(baseline_types):
        ax = axes[idx]
        synergies = results[baseline_type]['synergies']

        if not synergies:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(baseline_labels[baseline_type])
            continue

        # Normalize and compute mean
        norm_syn = []
        for s in synergies:
            s_min, s_max = s.min(), s.max()
            norm_syn.append((s - s_min) / (s_max - s_min + 1e-9))

        mean_syn = np.mean(norm_syn, axis=0)
        std_syn = np.std(norm_syn, axis=0)

        # Reverse for plotting
        mean_plot = mean_syn[::-1]
        std_plot = std_syn[::-1]

        # Smooth
        mean_smooth = gaussian_filter1d(mean_plot, sigma=SMOOTH_SIGMA)
        std_smooth = gaussian_filter1d(std_plot, sigma=SMOOTH_SIGMA)

        # Plot EMD zone
        ax.axvspan(-100, -30, color='#e5f5e0', alpha=0.5, label='EMD Zone')

        # Plot variance
        ax.fill_between(time_axis,
                        mean_smooth - std_smooth * 0.2,
                        mean_smooth + std_smooth * 0.2,
                        color=colors[idx], alpha=0.2)

        # Plot mean
        ax.plot(time_axis, mean_smooth, color=colors[idx], linewidth=2.5,
                label=f'N={len(synergies)}')

        # Mark peak in EMD zone
        emd_mask = (time_axis >= -100) & (time_axis <= -30)
        if np.any(emd_mask):
            peak_idx = np.argmax(mean_smooth[emd_mask])
            peak_time = time_axis[emd_mask][peak_idx]
            ax.axvline(peak_time, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                      label=f'Peak: {abs(peak_time):.1f}ms')

        ax.set_title(baseline_labels[baseline_type], fontsize=12, weight='bold')
        ax.set_xlabel('Time relative to action (ms)', fontsize=10)
        ax.set_ylabel('Normalized Synergy', fontsize=10)
        ax.set_xlim(-150, 0)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/baseline_comparison.png", dpi=200)
    print(f"   Plot saved: {RESULT_DIR}/baseline_comparison.png")


if __name__ == '__main__':
    run_e13()
