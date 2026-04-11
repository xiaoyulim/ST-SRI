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
from common import NinaProDataset, LSTMModel, ST_SRI_Interpreter, DEVICE, FS

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================= Configuration =================
DATA_ROOT = "./data"
CHECKPOINT_DIR = "./checkpoints_2000hz"
RESULT_DIR = "./results/aopc"

WINDOW_MS = 300
NUM_CLASSES = 18
NUM_SAMPLES = 15  # Samples per subject for AOPC evaluation
PERTURBATION_STEPS = 10  # Number of perturbation steps
# =================================================

os.makedirs(RESULT_DIR, exist_ok=True)


def get_accuracy_with_mask(model, loader, mask_indices=None):
    """
    Compute accuracy with optional masking of specific time indices.

    Args:
        model: The model to evaluate
        loader: DataLoader
        mask_indices: List of time indices to mask (set to 0)

    Returns:
        Accuracy percentage
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()

            if mask_indices is not None and len(mask_indices) > 0:
                x_masked = x.clone()
                x_masked[:, mask_indices, :] = 0
                x = x_masked

            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total if total > 0 else 0.0


def compute_st_sri_attribution(model, samples, bg_data):
    """Compute ST-SRI temporal attribution for samples"""
    interpreter = ST_SRI_Interpreter(model, bg_data)

    attributions = []
    for i in range(len(samples)):
        try:
            lags_ms, synergy, _ = interpreter.scan_fast(samples[i], max_lag_ms=150, stride=1, block_size=10)
            # Convert to time-step attribution (reverse and normalize)
            attr = synergy[::-1]
            attr = attr / (attr.sum() + 1e-9)  # Normalize to sum to 1
            attributions.append(attr)
        except:
            continue

    if not attributions:
        return None

    # Average across samples
    mean_attr = np.mean(attributions, axis=0)
    return mean_attr


def compute_shap_attribution(model, samples, bg_data):
    """Compute SHAP temporal attribution for samples"""
    explainer = shap.GradientExplainer(model, bg_data)

    with torch.backends.cudnn.flags(enabled=False):
        shap_vals = explainer.shap_values(samples)

    # Process SHAP values
    if isinstance(shap_vals, list):
        shap_res = np.abs(np.array(shap_vals)).sum(axis=0).sum(axis=-1)
    else:
        shap_res = np.abs(shap_vals).sum(axis=-1)

    # Average across samples and normalize
    mean_attr = np.mean(shap_res, axis=0)
    mean_attr = mean_attr / (mean_attr.sum() + 1e-9)

    return mean_attr


def compute_aopc(model, dataset, attribution, perturbation_steps=20):
    """
    Compute Area Over Perturbation Curve (AOPC).

    Args:
        model: The model to evaluate
        dataset: Dataset containing samples
        attribution: Temporal attribution scores (length = time steps)
        perturbation_steps: Number of perturbation steps

    Returns:
        AOPC score and accuracy curve
    """
    T = len(attribution)

    # Sort time indices by attribution importance (high to low)
    sorted_indices = np.argsort(attribution)[::-1]

    # Compute step size
    step_size = max(1, T // perturbation_steps)

    # Create DataLoader for evaluation
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Baseline accuracy (no masking)
    acc_baseline = get_accuracy_with_mask(model, loader, mask_indices=None)

    # Compute accuracy curve
    accuracies = [acc_baseline]
    masked_so_far = []

    for step in range(1, perturbation_steps + 1):
        # Add next most important indices to mask
        start_idx = (step - 1) * step_size
        end_idx = min(step * step_size, T)
        new_mask = sorted_indices[start_idx:end_idx].tolist()
        masked_so_far.extend(new_mask)

        # Compute accuracy with current mask
        acc = get_accuracy_with_mask(model, loader, mask_indices=masked_so_far)
        accuracies.append(acc)

    # Compute AOPC (area between baseline and curve)
    accuracies = np.array(accuracies)
    aopc = np.mean(acc_baseline - accuracies)

    return aopc, accuracies


def analyze_subject_aopc(sub_id):
    """Analyze AOPC for a single subject"""
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

    # Get active samples (non-rest)
    active_indices = np.where(ds.labels.numpy() != 0)[0]
    if len(active_indices) < NUM_SAMPLES:
        return None

    # Sample random active windows
    np.random.seed(42)
    sample_indices = np.random.choice(active_indices, NUM_SAMPLES, replace=False)
    samples = torch.stack([ds.data[i:i+600] for i in sample_indices]).to(DEVICE)

    # Get background data
    bg_data = ds.data[:600].unsqueeze(0).to(DEVICE)

    # Compute attributions
    st_sri_attr = compute_st_sri_attribution(model, samples, bg_data)
    shap_attr = compute_shap_attribution(model, samples, bg_data)

    if st_sri_attr is None or shap_attr is None:
        return None

    # Create random and reverse attributions
    T = len(st_sri_attr)
    random_attr = np.random.rand(T)
    random_attr = random_attr / random_attr.sum()

    reverse_attr = st_sri_attr[::-1]  # Least important first

    # Create a small dataset with these samples for AOPC evaluation
    class SampleDataset(torch.utils.data.Dataset):
        def __init__(self, samples, labels):
            self.samples = samples
            self.labels = labels

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx], self.labels[idx]

    # Get labels for samples
    sample_labels = torch.stack([ds.labels[i] for i in sample_indices])
    eval_ds = SampleDataset(samples.cpu(), sample_labels)

    # Compute AOPC for each attribution method
    aopc_st_sri, curve_st_sri = compute_aopc(model, eval_ds, st_sri_attr, PERTURBATION_STEPS)
    aopc_shap, curve_shap = compute_aopc(model, eval_ds, shap_attr, PERTURBATION_STEPS)
    aopc_random, curve_random = compute_aopc(model, eval_ds, random_attr, PERTURBATION_STEPS)
    aopc_reverse, curve_reverse = compute_aopc(model, eval_ds, reverse_attr, PERTURBATION_STEPS)

    return {
        'subject': sub_id,
        'aopc_st_sri': float(aopc_st_sri),
        'aopc_shap': float(aopc_shap),
        'aopc_random': float(aopc_random),
        'aopc_reverse': float(aopc_reverse),
        'curve_st_sri': curve_st_sri.tolist(),
        'curve_shap': curve_shap.tolist(),
        'curve_random': curve_random.tolist(),
        'curve_reverse': curve_reverse.tolist()
    }


def run_e15():
    """Main experiment: AOPC faithfulness evaluation"""
    print("=" * 70)
    print("E15: AOPC Faithfulness Evaluation")
    print("=" * 70)

    # Load subject list
    if os.path.exists("good_subjects.json"):
        with open("good_subjects.json", "r") as f:
            target_subjects = json.load(f)[:5]  # Use first 5 subjects
    else:
        target_subjects = list(range(1, 11))

    print(f"Target subjects: {len(target_subjects)}")

    # Collect results
    results = []
    for sub in target_subjects:
        res = analyze_subject_aopc(sub)
        if res is not None:
            results.append(res)
            print(f"  S{sub}: ST-SRI={res['aopc_st_sri']:.3f}, SHAP={res['aopc_shap']:.3f}, "
                  f"Random={res['aopc_random']:.3f}, Reverse={res['aopc_reverse']:.3f}")

    if not results:
        print("Error: No data collected")
        return

    print(f"\n✅ Collected {len(results)} subjects")

    # Compute statistics
    aopc_st_sri = [r['aopc_st_sri'] for r in results]
    aopc_shap = [r['aopc_shap'] for r in results]
    aopc_random = [r['aopc_random'] for r in results]
    aopc_reverse = [r['aopc_reverse'] for r in results]

    print("\n" + "=" * 70)
    print("AOPC Summary (higher = more faithful):")
    print("=" * 70)
    print(f"ST-SRI:   {np.mean(aopc_st_sri):.3f} ± {np.std(aopc_st_sri):.3f}")
    print(f"SHAP:     {np.mean(aopc_shap):.3f} ± {np.std(aopc_shap):.3f}")
    print(f"Random:   {np.mean(aopc_random):.3f} ± {np.std(aopc_random):.3f}")
    print(f"Reverse:  {np.mean(aopc_reverse):.3f} ± {np.std(aopc_reverse):.3f}")

    # Save results
    with open(f"{RESULT_DIR}/aopc_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Plot results
    plot_aopc_results(results)

    print(f"\n✅ E15 completed! Results saved to {RESULT_DIR}/")


def plot_aopc_results(results):
    """Plot AOPC comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Extract data
    aopc_st_sri = [r['aopc_st_sri'] for r in results]
    aopc_shap = [r['aopc_shap'] for r in results]
    aopc_random = [r['aopc_random'] for r in results]
    aopc_reverse = [r['aopc_reverse'] for r in results]

    # Plot 1: Bar chart with individual points
    methods = ['ST-SRI', 'SHAP', 'Random', 'Reverse']
    means = [np.mean(aopc_st_sri), np.mean(aopc_shap), np.mean(aopc_random), np.mean(aopc_reverse)]
    stds = [np.std(aopc_st_sri), np.std(aopc_shap), np.std(aopc_random), np.std(aopc_reverse)]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    x_pos = np.arange(len(methods))
    ax1.bar(x_pos, means, yerr=stds, color=colors, alpha=0.6, capsize=5)

    # Add individual points
    for i, data in enumerate([aopc_st_sri, aopc_shap, aopc_random, aopc_reverse]):
        jitter = np.random.normal(0, 0.05, len(data))
        ax1.scatter(np.ones(len(data)) * i + jitter, data, color='black', alpha=0.3, s=20)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods)
    ax1.set_ylabel('AOPC Score', fontsize=12)
    ax1.set_title(f'AOPC Comparison (N={len(results)})', fontsize=13, weight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # Plot 2: Perturbation curves
    # Average curves across subjects
    curve_st_sri = np.mean([r['curve_st_sri'] for r in results], axis=0)
    curve_shap = np.mean([r['curve_shap'] for r in results], axis=0)
    curve_random = np.mean([r['curve_random'] for r in results], axis=0)
    curve_reverse = np.mean([r['curve_reverse'] for r in results], axis=0)

    x_steps = np.arange(len(curve_st_sri))

    ax2.plot(x_steps, curve_st_sri, color=colors[0], linewidth=2.5, label='ST-SRI', marker='o', markersize=4)
    ax2.plot(x_steps, curve_shap, color=colors[1], linewidth=2.5, label='SHAP', marker='s', markersize=4)
    ax2.plot(x_steps, curve_random, color=colors[2], linewidth=2.5, label='Random', marker='^', markersize=4)
    ax2.plot(x_steps, curve_reverse, color=colors[3], linewidth=2.5, label='Reverse', marker='v', markersize=4)

    ax2.set_xlabel('Perturbation Steps', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Perturbation Curves', fontsize=13, weight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/aopc_comparison.png", dpi=200)
    print(f"   Plot saved: {RESULT_DIR}/aopc_comparison.png")


if __name__ == '__main__':
    run_e15()
