import os
import sys
# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import json
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d
from common import NinaProDataset, LSTMModel, ST_SRI_Interpreter, DEVICE, FS

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Try to import captum (optional)
try:
    from captum.attr import IntegratedGradients, DeepLift
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("⚠️  Warning: captum not installed. Install with: python3 -m pip install captum")
    print("    Only ST-SRI and SHAP will be evaluated.")

# ================= Configuration =================
DATA_ROOT = "./data"
CHECKPOINT_DIR = "./checkpoints_2000hz"
RESULT_DIR = "./results/xai_baselines"

WINDOW_MS = 300
NUM_CLASSES = 18
NUM_SAMPLES = 30  # Samples per subject
SCAN_STRIDE = 3
# =================================================

os.makedirs(RESULT_DIR, exist_ok=True)


def compute_st_sri_attribution(model, samples, bg_data):
    """Compute ST-SRI temporal attribution"""
    interpreter = ST_SRI_Interpreter(model, bg_data)

    attributions = []
    for i in range(len(samples)):
        try:
            lags_ms, synergy, _ = interpreter.scan_fast(samples[i], max_lag_ms=150, stride=SCAN_STRIDE, block_size=10)
            # Reverse to get temporal attribution
            attr = synergy[::-1]
            attributions.append(attr)
        except:
            continue

    if not attributions:
        return None

    return np.array(attributions)


def compute_shap_attribution(model, samples, bg_data):
    """Compute Standard SHAP temporal attribution"""
    explainer = shap.GradientExplainer(model, bg_data)

    with torch.backends.cudnn.flags(enabled=False):
        shap_vals = explainer.shap_values(samples)

    # Process SHAP values
    if isinstance(shap_vals, list):
        shap_res = np.abs(np.array(shap_vals)).sum(axis=0).sum(axis=-1)
    else:
        shap_res = np.abs(shap_vals).sum(axis=-1)

    return shap_res


def compute_integrated_gradients_attribution(model, samples, bg_data):
    """Compute Integrated Gradients temporal attribution"""
    if not CAPTUM_AVAILABLE:
        return None

    # Create wrapper for model output
    def model_forward(x):
        return model(x)

    ig = IntegratedGradients(model_forward)

    # Use mean of background as baseline
    baseline = torch.mean(bg_data, dim=0).unsqueeze(0).repeat(len(samples), 1, 1)

    # Compute attributions
    attributions = ig.attribute(samples, baselines=baseline, target=None)

    # Sum over channels to get temporal attribution
    attr_temporal = torch.abs(attributions).sum(dim=-1).cpu().numpy()

    return attr_temporal


def compute_deeplift_attribution(model, samples, bg_data):
    """Compute DeepLIFT temporal attribution"""
    if not CAPTUM_AVAILABLE:
        return None

    # Create wrapper for model output
    def model_forward(x):
        return model(x)

    dl = DeepLift(model_forward)

    # Use mean of background as baseline
    baseline = torch.mean(bg_data, dim=0).unsqueeze(0).repeat(len(samples), 1, 1)

    # Compute attributions
    attributions = dl.attribute(samples, baselines=baseline, target=None)

    # Sum over channels to get temporal attribution
    attr_temporal = torch.abs(attributions).sum(dim=-1).cpu().numpy()

    return attr_temporal


def compute_stability_metrics(attributions):
    """
    Compute stability metrics for attribution methods.

    Args:
        attributions: (N_samples, T) array of temporal attributions

    Returns:
        dict with metrics
    """
    if attributions is None or len(attributions) == 0:
        return None

    # 1. Cross-sample stability (lower std = more stable)
    # Normalize each sample first
    norm_attrs = []
    for attr in attributions:
        attr_norm = (attr - attr.min()) / (attr.max() - attr.min() + 1e-9)
        norm_attrs.append(attr_norm)
    norm_attrs = np.array(norm_attrs)

    mean_attr = np.mean(norm_attrs, axis=0)
    std_attr = np.std(norm_attrs, axis=0)
    stability = np.mean(std_attr)  # Lower = more stable

    # 2. Peak in EMD range (30-100ms from end)
    # Assuming attributions are in forward time (0 to T)
    T = attributions.shape[1]
    emd_start_idx = max(0, T - int(100 * FS / 1000))
    emd_end_idx = T - int(30 * FS / 1000)

    peak_in_emd = 0
    for attr in norm_attrs:
        peak_idx = np.argmax(attr)
        if emd_start_idx <= peak_idx <= emd_end_idx:
            peak_in_emd += 1
    peak_in_emd_ratio = peak_in_emd / len(norm_attrs)

    # 3. Fragmentation (derivative std - lower = smoother)
    derivatives = np.diff(mean_attr)
    fragmentation = np.std(derivatives)

    return {
        'stability': float(stability),
        'peak_in_emd_ratio': float(peak_in_emd_ratio),
        'fragmentation': float(fragmentation)
    }


def analyze_subject_xai(sub_id):
    """Analyze XAI methods for a single subject"""
    print(f"  Analyzing S{sub_id}...", end='\r')

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

    # Get active samples
    active_indices = np.where(ds.labels.numpy() != 0)[0]
    if len(active_indices) < NUM_SAMPLES:
        return None

    # Sample random active windows
    np.random.seed(42)
    sample_indices = np.random.choice(active_indices, NUM_SAMPLES, replace=False)
    samples = torch.stack([ds.data[i:i+600] for i in sample_indices]).to(DEVICE)

    # Get background data
    bg_data = ds.data[:600].unsqueeze(0).to(DEVICE)

    # Compute attributions for each method
    results = {}

    # 1. ST-SRI
    st_sri_attr = compute_st_sri_attribution(model, samples, bg_data)
    if st_sri_attr is not None:
        results['st_sri'] = compute_stability_metrics(st_sri_attr)

    # 2. SHAP
    shap_attr = compute_shap_attribution(model, samples, bg_data)
    if shap_attr is not None:
        results['shap'] = compute_stability_metrics(shap_attr)

    # 3. Integrated Gradients (if available)
    if CAPTUM_AVAILABLE:
        try:
            ig_attr = compute_integrated_gradients_attribution(model, samples, bg_data)
            if ig_attr is not None:
                results['integrated_gradients'] = compute_stability_metrics(ig_attr)
        except Exception as e:
            print(f"\n    Warning: IG failed for S{sub_id}: {e}")

    # 4. DeepLIFT (if available)
    if CAPTUM_AVAILABLE:
        try:
            dl_attr = compute_deeplift_attribution(model, samples, bg_data)
            if dl_attr is not None:
                results['deeplift'] = compute_stability_metrics(dl_attr)
        except Exception as e:
            print(f"\n    Warning: DeepLIFT failed for S{sub_id}: {e}")

    if not results:
        return None

    results['subject'] = sub_id
    return results


def run_e14():
    """Main experiment: XAI baseline comparison"""
    print("=" * 70)
    print("E14: Multi-XAI Baseline Comparison")
    print("=" * 70)

    if not CAPTUM_AVAILABLE:
        print("\n⚠️  Note: captum not available. Only ST-SRI and SHAP will be compared.")
        print("    To enable Integrated Gradients and DeepLIFT:")
        print("    Install captum: python3 -m pip install captum\n")

    # Load subject list
    if os.path.exists("good_subjects.json"):
        with open("good_subjects.json", "r") as f:
            target_subjects = json.load(f)[:5]  # Use first 5 subjects (simplified)
    else:
        target_subjects = list(range(1, 6))

    print(f"Target subjects: {target_subjects}")

    # Collect results
    all_results = []
    for sub in target_subjects:
        res = analyze_subject_xai(sub)
        if res is not None:
            all_results.append(res)
            print(f"  S{sub}: ✓")

    if not all_results:
        print("Error: No data collected")
        return

    print(f"\n✅ Collected {len(all_results)} subjects")

    # Aggregate metrics
    methods = ['st_sri', 'shap']
    if CAPTUM_AVAILABLE:
        methods.extend(['integrated_gradients', 'deeplift'])

    method_labels = {
        'st_sri': 'ST-SRI',
        'shap': 'Standard SHAP',
        'integrated_gradients': 'Integrated Gradients',
        'deeplift': 'DeepLIFT'
    }

    aggregated = {}
    for method in methods:
        stability_vals = []
        peak_vals = []
        frag_vals = []

        for res in all_results:
            if method in res:
                stability_vals.append(res[method]['stability'])
                peak_vals.append(res[method]['peak_in_emd_ratio'])
                frag_vals.append(res[method]['fragmentation'])

        if stability_vals:
            aggregated[method] = {
                'stability_mean': np.mean(stability_vals),
                'stability_std': np.std(stability_vals),
                'peak_in_emd_mean': np.mean(peak_vals),
                'peak_in_emd_std': np.std(peak_vals),
                'fragmentation_mean': np.mean(frag_vals),
                'fragmentation_std': np.std(frag_vals),
                'n_subjects': len(stability_vals)
            }

    # Print summary
    print("\n" + "=" * 70)
    print("Metrics Summary:")
    print("=" * 70)
    print(f"{'Method':<25} | {'Stability':<15} | {'Peak in EMD':<15} | {'Fragmentation':<15}")
    print("-" * 70)

    for method in methods:
        if method in aggregated:
            agg = aggregated[method]
            print(f"{method_labels[method]:<25} | "
                  f"{agg['stability_mean']:.4f}±{agg['stability_std']:.4f}  | "
                  f"{agg['peak_in_emd_mean']:.2%}±{agg['peak_in_emd_std']:.2%}  | "
                  f"{agg['fragmentation_mean']:.4f}±{agg['fragmentation_std']:.4f}")

    print("\nNote: Lower stability = more consistent across samples")
    print("      Higher peak_in_emd = more peaks in physiological range")
    print("      Lower fragmentation = smoother attribution curves")

    # Save results
    with open(f"{RESULT_DIR}/xai_comparison.json", 'w') as f:
        json.dump({
            'aggregated': aggregated,
            'individual_results': all_results
        }, f, indent=2)

    # Plot results
    plot_xai_comparison(aggregated, method_labels)

    print(f"\n✅ E14 completed! Results saved to {RESULT_DIR}/")


def plot_xai_comparison(aggregated, method_labels):
    """Plot XAI method comparison"""
    methods = list(aggregated.keys())
    if not methods:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    x_pos = np.arange(len(methods))

    # Plot 1: Stability (lower is better)
    stability_means = [aggregated[m]['stability_mean'] for m in methods]
    stability_stds = [aggregated[m]['stability_std'] for m in methods]

    axes[0].bar(x_pos, stability_means, yerr=stability_stds, color=colors[:len(methods)], alpha=0.6, capsize=5)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([method_labels[m] for m in methods], rotation=15, ha='right')
    axes[0].set_ylabel('Cross-Sample Std', fontsize=11)
    axes[0].set_title('Stability (Lower = Better)', fontsize=12, weight='bold')
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)

    # Plot 2: Peak in EMD ratio (higher is better)
    peak_means = [aggregated[m]['peak_in_emd_mean'] for m in methods]
    peak_stds = [aggregated[m]['peak_in_emd_std'] for m in methods]

    axes[1].bar(x_pos, peak_means, yerr=peak_stds, color=colors[:len(methods)], alpha=0.6, capsize=5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([method_labels[m] for m in methods], rotation=15, ha='right')
    axes[1].set_ylabel('Ratio', fontsize=11)
    axes[1].set_title('Peak in EMD Range (Higher = Better)', fontsize=12, weight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)

    # Plot 3: Fragmentation (lower is better)
    frag_means = [aggregated[m]['fragmentation_mean'] for m in methods]
    frag_stds = [aggregated[m]['fragmentation_std'] for m in methods]

    axes[2].bar(x_pos, frag_means, yerr=frag_stds, color=colors[:len(methods)], alpha=0.6, capsize=5)
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels([method_labels[m] for m in methods], rotation=15, ha='right')
    axes[2].set_ylabel('Derivative Std', fontsize=11)
    axes[2].set_title('Fragmentation (Lower = Better)', fontsize=12, weight='bold')
    axes[2].grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/xai_comparison.png", dpi=200)
    print(f"   Plot saved: {RESULT_DIR}/xai_comparison.png")


if __name__ == '__main__':
    run_e14()
