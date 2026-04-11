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
RESULT_DIR = "./results/gesture_emd"

WINDOW_MS = 300
NUM_CLASSES = 18
SAMPLES_PER_GESTURE = 20  # Samples per gesture class
SMOOTH_SIGMA = 2.0

# Gesture grouping based on NinaPro DB1 taxonomy
GESTURE_GROUPS = {
    'Power Grasps': [1, 2, 3, 4, 5, 6],  # Large cylindrical, tip, lateral, etc.
    'Precision Pinches': [7, 8, 9, 10],  # Small diameter, tripod, etc.
    'Wrist Motions': [11, 12, 13, 14],   # Flexion, extension, radial/ulnar deviation
    'Finger Motions': [15, 16, 17, 18]   # Pointing, adduction, abduction, etc.
}

GESTURE_NAMES = {
    1: "Thumb-Index Pinch", 2: "Thumb-Middle Pinch", 3: "Thumb-Ring Pinch",
    4: "Thumb-Little Pinch", 5: "Fist", 6: "Pointing",
    7: "Tripod Pinch", 8: "Palmar Grasp", 9: "Lateral Grasp",
    10: "Cylindrical Grasp", 11: "Hook Grasp", 12: "Spherical Grasp",
    13: "Wrist Flexion", 14: "Wrist Extension", 15: "Wrist Supination",
    16: "Wrist Pronation", 17: "Hand Open", 18: "Adduction"
}
# =================================================

os.makedirs(RESULT_DIR, exist_ok=True)


def analyze_gesture_for_subject(sub_id, gesture_class):
    """Analyze synergy spectrum for a specific gesture class"""
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

    interpreter = ST_SRI_Interpreter(model, bg_data)

    # Find samples of this gesture class
    gesture_mask = (all_y == gesture_class)
    gesture_indices = torch.where(gesture_mask)[0].numpy()

    if len(gesture_indices) < SAMPLES_PER_GESTURE:
        return None

    # Sample random instances
    np.random.seed(42 + gesture_class)
    selected_indices = np.random.choice(gesture_indices, SAMPLES_PER_GESTURE, replace=False)

    synergies = []
    lags_ms = []

    for idx in selected_indices:
        x_tensor = all_x[idx].to(DEVICE)
        try:
            l, s, _ = interpreter.scan_fast(x_tensor, max_lag_ms=150, stride=1, block_size=10)
            synergies.append(s)
            lags_ms = l
        except:
            continue

    if not synergies:
        return None

    mean_syn = np.mean(synergies, axis=0)
    return mean_syn, lags_ms


def compute_peak_in_range(synergy, lags_ms, time_range=(-100, -30)):
    """Find peak position and amplitude in specified time range"""
    time_axis = -np.array(lags_ms)[::-1]
    synergy_reversed = synergy[::-1]

    # Normalize
    s_min, s_max = synergy_reversed.min(), synergy_reversed.max()
    synergy_norm = (synergy_reversed - s_min) / (s_max - s_min + 1e-9)

    # Find peak in range
    mask = (time_axis >= time_range[0]) & (time_axis <= time_range[1])
    if not np.any(mask):
        return None, None

    peak_idx = np.argmax(synergy_norm[mask])
    peak_time = time_axis[mask][peak_idx]
    peak_amplitude = synergy_norm[mask][peak_idx]

    return peak_time, peak_amplitude


def run_e16():
    """Main experiment: Gesture-specific EMD analysis"""
    print("=" * 70)
    print("E16: Gesture-Specific EMD Analysis")
    print("=" * 70)

    # Load subject list
    if os.path.exists("good_subjects.json"):
        with open("good_subjects.json", "r") as f:
            target_subjects = json.load(f)[:10]  # Use first 10 subjects
    else:
        target_subjects = list(range(1, 11))

    print(f"Target subjects: {target_subjects}")
    print(f"Analyzing {NUM_CLASSES} gesture classes...")

    # Collect data for each gesture
    gesture_results = {}

    for gesture_class in range(1, NUM_CLASSES + 1):
        print(f"\n>>> Gesture {gesture_class}: {GESTURE_NAMES.get(gesture_class, 'Unknown')}")
        synergies = []
        lags_ms = None

        for sub in target_subjects:
            result = analyze_gesture_for_subject(sub, gesture_class)
            if result is not None:
                syn, lags = result
                synergies.append(syn)
                lags_ms = lags
                print(f"    S{sub}: ✓", end='')

        if synergies:
            mean_syn = np.mean(synergies, axis=0)
            std_syn = np.std(synergies, axis=0)

            # Compute peak
            peak_time, peak_amp = compute_peak_in_range(mean_syn, lags_ms)

            gesture_results[gesture_class] = {
                'name': GESTURE_NAMES.get(gesture_class, f'Gesture {gesture_class}'),
                'mean_synergy': mean_syn,
                'std_synergy': std_syn,
                'lags_ms': lags_ms,
                'peak_time': float(peak_time) if peak_time is not None else None,
                'peak_amplitude': float(peak_amp) if peak_amp is not None else None,
                'n_subjects': len(synergies)
            }

            print(f"\n    Peak: {abs(peak_time):.1f}ms" if peak_time else "\n    No peak found")
        else:
            print(f"\n    No data collected")

    if not gesture_results:
        print("Error: No data collected")
        return

    # Group by gesture type and analyze
    print("\n" + "=" * 70)
    print("Peak Analysis by Gesture Group:")
    print("=" * 70)

    group_peaks = {}
    for group_name, gesture_ids in GESTURE_GROUPS.items():
        peaks = []
        for gid in gesture_ids:
            if gid in gesture_results and gesture_results[gid]['peak_time'] is not None:
                peaks.append(abs(gesture_results[gid]['peak_time']))

        if peaks:
            group_peaks[group_name] = peaks
            print(f"{group_name:20s}: {np.mean(peaks):6.1f} ± {np.std(peaks):5.1f} ms  (N={len(peaks)})")

    # Save results
    save_data = {
        'gesture_results': {
            str(k): {
                'name': v['name'],
                'peak_time': v['peak_time'],
                'peak_amplitude': v['peak_amplitude'],
                'n_subjects': v['n_subjects']
            }
            for k, v in gesture_results.items()
        },
        'group_peaks': group_peaks
    }

    with open(f"{RESULT_DIR}/gesture_peaks.json", 'w') as f:
        json.dump(save_data, f, indent=2)

    # Plot results
    plot_gesture_comparison(gesture_results, group_peaks)

    print(f"\n✅ E16 completed! Results saved to {RESULT_DIR}/")


def plot_gesture_comparison(gesture_results, group_peaks):
    """Plot gesture-specific synergy spectra and peak distribution"""

    # Plot 1: Peak distribution by group
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Bar plot of group peaks
    group_names = list(group_peaks.keys())
    group_means = [np.mean(group_peaks[g]) for g in group_names]
    group_stds = [np.std(group_peaks[g]) for g in group_names]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    x_pos = np.arange(len(group_names))
    ax1.bar(x_pos, group_means, yerr=group_stds, color=colors, alpha=0.6, capsize=5)

    # Add individual gesture points
    for i, group_name in enumerate(group_names):
        peaks = group_peaks[group_name]
        jitter = np.random.normal(0, 0.05, len(peaks))
        ax1.scatter(np.ones(len(peaks)) * i + jitter, peaks, color='black', alpha=0.4, s=30)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(group_names, rotation=15, ha='right')
    ax1.set_ylabel('Peak Latency (ms)', fontsize=12)
    ax1.set_title('EMD Peak by Gesture Type', fontsize=13, weight='bold')
    ax1.axhspan(30, 100, color='#e5f5e0', alpha=0.3, label='Physiological EMD')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # Plot 2: Selected gesture spectra
    # Show one representative from each group
    representative_gestures = [
        (5, 'Power Grasps', colors[0]),      # Fist
        (7, 'Precision Pinches', colors[1]), # Tripod Pinch
        (13, 'Wrist Motions', colors[2]),    # Wrist Flexion
        (17, 'Finger Motions', colors[3])    # Hand Open
    ]

    for gesture_id, group_name, color in representative_gestures:
        if gesture_id not in gesture_results:
            continue

        result = gesture_results[gesture_id]
        lags_ms = result['lags_ms']
        mean_syn = result['mean_synergy']

        # Normalize and reverse
        time_axis = -np.array(lags_ms)[::-1]
        s_min, s_max = mean_syn.min(), mean_syn.max()
        synergy_norm = (mean_syn - s_min) / (s_max - s_min + 1e-9)
        synergy_plot = synergy_norm[::-1]

        # Smooth
        synergy_smooth = gaussian_filter1d(synergy_plot, sigma=SMOOTH_SIGMA)

        # Plot
        label = f"{result['name']} ({group_name})"
        ax2.plot(time_axis, synergy_smooth, color=color, linewidth=2.5, label=label)

        # Mark peak
        if result['peak_time'] is not None:
            peak_time = result['peak_time']
            ax2.axvline(peak_time, color=color, linestyle='--', linewidth=1.5, alpha=0.5)

    ax2.axvspan(-100, -30, color='#e5f5e0', alpha=0.3)
    ax2.set_xlabel('Time relative to action (ms)', fontsize=12)
    ax2.set_ylabel('Normalized Synergy', fontsize=12)
    ax2.set_title('Representative Synergy Spectra', fontsize=13, weight='bold')
    ax2.set_xlim(-150, 0)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/gesture_comparison.png", dpi=200)
    print(f"   Plot saved: {RESULT_DIR}/gesture_comparison.png")

    # Plot 3: Heatmap of all gestures
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    # Prepare data for heatmap
    gesture_ids = sorted(gesture_results.keys())
    n_gestures = len(gesture_ids)

    if n_gestures == 0:
        return

    # Get time axis from first gesture
    lags_ms = gesture_results[gesture_ids[0]]['lags_ms']
    time_axis = -np.array(lags_ms)[::-1]

    # Build matrix
    synergy_matrix = []
    gesture_labels = []

    for gid in gesture_ids:
        result = gesture_results[gid]
        mean_syn = result['mean_synergy']

        # Normalize
        s_min, s_max = mean_syn.min(), mean_syn.max()
        synergy_norm = (mean_syn - s_min) / (s_max - s_min + 1e-9)
        synergy_plot = synergy_norm[::-1]

        # Smooth
        synergy_smooth = gaussian_filter1d(synergy_plot, sigma=SMOOTH_SIGMA)

        synergy_matrix.append(synergy_smooth)
        gesture_labels.append(f"G{gid}: {result['name'][:15]}")

    synergy_matrix = np.array(synergy_matrix)

    # Plot heatmap
    im = ax.imshow(synergy_matrix, aspect='auto', cmap='YlOrRd', interpolation='bilinear')

    # Set ticks
    ax.set_yticks(np.arange(n_gestures))
    ax.set_yticklabels(gesture_labels, fontsize=8)

    # X-axis: show time in ms
    n_time_points = len(time_axis)
    tick_positions = np.linspace(0, n_time_points - 1, 7).astype(int)
    tick_labels = [f"{time_axis[i]:.0f}" for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('Time relative to action (ms)', fontsize=11)

    # Mark EMD zone
    emd_start_idx = np.argmin(np.abs(time_axis - (-100)))
    emd_end_idx = np.argmin(np.abs(time_axis - (-30)))
    ax.axvspan(emd_start_idx, emd_end_idx, color='cyan', alpha=0.1, linewidth=2, edgecolor='cyan')

    ax.set_title('Synergy Spectrum Heatmap by Gesture', fontsize=13, weight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Synergy', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/gesture_heatmap.png", dpi=200)
    print(f"   Heatmap saved: {RESULT_DIR}/gesture_heatmap.png")


if __name__ == '__main__':
    run_e16()
