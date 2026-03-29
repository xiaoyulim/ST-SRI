"""从已保存的 CSV 重新生成 e5_enhanced 图像"""
import os, json, csv, numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RESULT_DIR = "./results/e5_faithfulness"
CSV_PATH = os.path.join(RESULT_DIR, "e5_enhanced_results.csv")
PLOT_PATH = os.path.join(RESULT_DIR, "e5_enhanced_plot.png")

rows = []
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        sub = int(r["subject"])
        row = {k: float(v) for k, v in r.items() if k != "subject"}
        row["subject"] = sub
        rows.append(row)

drops_rec = np.array([r["drop_recent"] for r in rows])
drops_rnd = np.array([r["drop_random"] for r in rows])
drops_emd = np.array([r["drop_emd"] for r in rows])
N = len(rows)

t_rand, p_rand = stats.ttest_rel(drops_emd, drops_rnd)
rf_recent = float(np.mean(drops_emd)) / (float(np.mean(drops_rec)) + 1e-9)
rf_random = float(np.mean(drops_emd)) / (float(np.mean(drops_rnd)) + 1e-9)

print(f"N={N}")
print(f"Recent: {np.mean(drops_rec):.3f}±{np.std(drops_rec):.3f}%")
print(f"Random: {np.mean(drops_rnd):.3f}±{np.std(drops_rnd):.3f}%")
print(f"EMD:    {np.mean(drops_emd):.3f}±{np.std(drops_emd):.3f}%")
print(f"Rf_recent={rf_recent:.3f}  Rf_random={rf_random:.3f}")
print(f"EMD vs Random: t={t_rand:.3f}, p={p_rand:.4e}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# — 左：柱状图 + 散点 —
ax = axes[0]
groups = [drops_rec, drops_rnd, drops_emd]
labels = ["Mask Recent\n(0–20ms)", "Mask Random\n(avg N=10)", "Mask EMD\n(Peak±20ms)"]
colors_bar = ["#9e9e9e", "#64b5f6", "#e53935"]
colors_dot = ["#424242", "#1565c0", "#7f0000"]

bars = ax.bar([0, 1, 2],
              [np.mean(g) for g in groups],
              yerr=[np.std(g) / np.sqrt(N) for g in groups],
              align="center", alpha=0.65, ecolor="black",
              capsize=8, width=0.5, color=colors_bar)

rng = np.random.RandomState(7)
for xi, (g, col) in enumerate(zip(groups, colors_dot)):
    jitter = rng.normal(0, 0.06, len(g))
    ax.scatter(np.full(len(g), xi) + jitter, g, color=col, alpha=0.5, s=20, zorder=3)
for i in range(N):
    ax.plot([1, 2], [drops_rnd[i], drops_emd[i]], color="gray", alpha=0.08, linewidth=0.7)

y_top = max(np.max(drops_emd), np.max(drops_rnd))
y_sig = y_top * 1.08
ax.plot([1, 1, 2, 2], [y_sig, y_sig + 0.3, y_sig + 0.3, y_sig], lw=1.5, c="k")
sig_sym = "**" if p_rand < 0.01 else ("*" if p_rand < 0.05 else "n.s.")
ax.text(1.5, y_sig + 0.4, f"{sig_sym}\n(p={p_rand:.2e})", ha="center", fontsize=9)

for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., h / 2,
            f"{h:.2f}%", ha="center", va="center",
            color="white", fontweight="bold", fontsize=11)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Accuracy Drop (%)", fontsize=12)
ax.set_title(f"Faithfulness Evaluation (N={N})\n"
             f"Rf_recent={rf_recent:.2f}×  |  Rf_random={rf_random:.2f}×", fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# — 右：每受试者差异条形图 —
ax2 = axes[1]
diff = drops_emd - drops_rnd
colors_sub = ["#e53935" if d > 0 else "#1565c0" for d in diff]
sub_labels = [f"S{r['subject']}" for r in rows]
ax2.bar(range(N), diff, color=colors_sub, alpha=0.7, width=0.7)
ax2.axhline(0, color="black", linewidth=1)
ax2.axhline(float(np.mean(diff)), color="red", linewidth=1.5, linestyle="--",
            label=f"Mean = {np.mean(diff):+.2f}%")
ax2.set_xticks(range(N))
ax2.set_xticklabels(sub_labels, rotation=45, ha="right", fontsize=7)
ax2.set_ylabel("Δ(drop_EMD − drop_Random) (%)", fontsize=11)
ax2.set_title("Per-Subject: EMD Drop − Random Drop\n(+Red = EMD more critical)", fontsize=10)
ax2.legend(fontsize=10)
ax2.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
print(f"图像已保存: {PLOT_PATH}")
plt.close()
