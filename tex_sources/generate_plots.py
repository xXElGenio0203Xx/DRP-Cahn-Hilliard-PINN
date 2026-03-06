#!/usr/bin/env python3
"""
Generate comparative plots for the Cahn-Hilliard PINN experiment report.
Reads summary.json and metrics.npz from each predictions/ subfolder.
Saves plots into tex_sources/figures/.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation
import matplotlib.ticker as ticker

# ── Paths ────────────────────────────────────────────────────────────────
PRED_DIR = os.path.join(os.path.dirname(__file__), "..", "predictions")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load all summaries ───────────────────────────────────────────────────
experiments = {}
for name in sorted(os.listdir(PRED_DIR)):
    sfile = os.path.join(PRED_DIR, name, "summary.json")
    mfile = os.path.join(PRED_DIR, name, "metrics.npz")
    if os.path.isfile(sfile):
        with open(sfile) as f:
            s = json.load(f)
        s["_dir"] = name
        if os.path.isfile(mfile):
            m = np.load(mfile)
            s["_adam_losses"] = m.get("adam_losses", np.array([]))
            s["_bfgs_losses"] = m.get("bfgs_losses", np.array([]))
        else:
            s["_adam_losses"] = np.array([])
            s["_bfgs_losses"] = np.array([])
        experiments[name] = s

print(f"Loaded {len(experiments)} experiments: {list(experiments.keys())}")

# ── Color scheme ─────────────────────────────────────────────────────────
COLORS = {
    "cahn_hilliard_canonical": "#1f77b4",
    "adam_lr_high":   "#ff7f0e",
    "adam_lr_low":    "#2ca02c",
    "adam_only":      "#d62728",
    "bfgs_batch_200": "#9467bd",
    "bfgs_batch_1000":"#8c564b",
    "bfgs_vanilla":   "#e377c2",
    "hessian_scaled": "#7f7f7f",
    "lbfgs":          "#bcbd22",
    "power_2":        "#17becf",
    "power_4":        "#aec7e8",
    "ssbfgs_ab":      "#ff9896",
    "ssbfgs_ol":      "#98df8a",
    "ssbroyden1":     "#c5b0d5",
    "warmup_0":       "#c49c94",
    "warmup_500":     "#f7b6d2",
    "warmup_5000":    "#dbdb8d",
}

NICE_NAMES = {
    "cahn_hilliard_canonical": "Canonical (SSBroyden2)",
    "adam_lr_high":   "Adam LR=0.01",
    "adam_lr_low":    "Adam LR=0.0001",
    "adam_only":      "Adam Only",
    "bfgs_batch_200": "Batch=200",
    "bfgs_batch_1000":"Batch=1000",
    "bfgs_vanilla":   "Vanilla BFGS",
    "hessian_scaled": "Hessian Scaled",
    "lbfgs":          "L-BFGS-B",
    "power_2":        "Power p=2",
    "power_4":        "Power p=4",
    "ssbfgs_ab":      "SSBFGS-AB",
    "ssbfgs_ol":      "SSBFGS-OL",
    "ssbroyden1":     "SSBroyden1 (DFP)",
    "warmup_0":       "No Warmup (0)",
    "warmup_500":     "Warmup 500",
    "warmup_5000":    "Warmup 5000",
}

# ===========================================================================
# PLOT 1: Bar chart — Final BFGS loss by experiment (log scale)
# ===========================================================================
print("Plot 1: Final loss bar chart...")

names_sorted = sorted(
    [k for k, v in experiments.items() if v.get("final_bfgs_loss") is not None],
    key=lambda k: experiments[k]["final_bfgs_loss"]
)
# Add adam_only at end
if "adam_only" in experiments:
    names_sorted.append("adam_only")

labels = [NICE_NAMES.get(n, n) for n in names_sorted]
losses = []
for n in names_sorted:
    s = experiments[n]
    val = s.get("final_bfgs_loss") or s.get("final_adam_loss") or 1.0
    losses.append(val)
colors = [COLORS.get(n, "#333333") for n in names_sorted]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(labels)), losses, color=colors, edgecolor="k", linewidth=0.5)
ax.set_xscale("log")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Final Loss (log scale)", fontsize=11)
ax.set_title("Final Loss by Experiment Configuration", fontsize=13, fontweight="bold")
ax.invert_yaxis()
# Add value labels
for i, (v, bar) in enumerate(zip(losses, bars)):
    ax.text(v * 1.3, i, f"{v:.2e}", va="center", fontsize=7.5)
ax.axvline(x=experiments["cahn_hilliard_canonical"]["final_bfgs_loss"],
           color="blue", ls="--", lw=1, alpha=0.5, label="Canonical baseline")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "final_loss_bar.pdf"), dpi=300)
fig.savefig(os.path.join(FIG_DIR, "final_loss_bar.png"), dpi=150)
plt.close(fig)

# ===========================================================================
# PLOT 2: Grouped bar — Optimizer variant comparison
# ===========================================================================
print("Plot 2: Optimizer variant comparison...")

variant_keys = ["cahn_hilliard_canonical", "ssbfgs_ab", "ssbfgs_ol",
                "bfgs_vanilla", "lbfgs", "ssbroyden1"]
variant_labels = [NICE_NAMES[k] for k in variant_keys]
variant_losses = [experiments[k]["final_bfgs_loss"] for k in variant_keys]
variant_times  = [experiments[k]["total_time_s"] / 3600 for k in variant_keys]
variant_colors = [COLORS[k] for k in variant_keys]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Loss
bars1 = ax1.bar(range(len(variant_keys)), variant_losses, color=variant_colors,
                edgecolor="k", linewidth=0.5)
ax1.set_yscale("log")
ax1.set_xticks(range(len(variant_keys)))
ax1.set_xticklabels(variant_labels, rotation=30, ha="right", fontsize=8)
ax1.set_ylabel("Final BFGS Loss (log)", fontsize=10)
ax1.set_title("Optimizer Variant: Final Loss", fontsize=12, fontweight="bold")
for i, v in enumerate(variant_losses):
    ax1.text(i, v * 1.5, f"{v:.2e}", ha="center", fontsize=7, rotation=0)

# Time
bars2 = ax2.bar(range(len(variant_keys)), variant_times, color=variant_colors,
                edgecolor="k", linewidth=0.5)
ax2.set_xticks(range(len(variant_keys)))
ax2.set_xticklabels(variant_labels, rotation=30, ha="right", fontsize=8)
ax2.set_ylabel("Total Wall Time (hours)", fontsize=10)
ax2.set_title("Optimizer Variant: Wall Time", fontsize=12, fontweight="bold")
for i, v in enumerate(variant_times):
    ax2.text(i, v + 0.05, f"{v:.1f}h", ha="center", fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "optimizer_variant_comparison.pdf"), dpi=300)
fig.savefig(os.path.join(FIG_DIR, "optimizer_variant_comparison.png"), dpi=150)
plt.close(fig)

# ===========================================================================
# PLOT 3: Warmup duration sweep
# ===========================================================================
print("Plot 3: Warmup sweep...")

warmup_keys = ["warmup_0", "warmup_500", "cahn_hilliard_canonical", "warmup_5000"]
warmup_adam  = [0, 500, 2000, 5000]
warmup_loss  = [experiments[k]["final_bfgs_loss"] for k in warmup_keys]
warmup_drift = [experiments[k]["mass_drift"] for k in warmup_keys]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(warmup_adam, warmup_loss, "o-", markersize=8, color="#1f77b4", linewidth=2)
for i, (x, y) in enumerate(zip(warmup_adam, warmup_loss)):
    ax1.annotate(f"{y:.2e}", (x, y), textcoords="offset points",
                 xytext=(8, 8), fontsize=8)
ax1.set_xlabel("Adam Warmup Epochs", fontsize=11)
ax1.set_ylabel("Final Loss (log)", fontsize=11)
ax1.set_yscale("log")
ax1.set_title("Effect of Warmup Duration on Loss", fontsize=12, fontweight="bold")
ax1.set_xticks(warmup_adam)

ax2.plot(warmup_adam, warmup_drift, "s-", markersize=8, color="#d62728", linewidth=2)
for i, (x, y) in enumerate(zip(warmup_adam, warmup_drift)):
    ax2.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                 xytext=(8, 8), fontsize=8)
ax2.set_xlabel("Adam Warmup Epochs", fontsize=11)
ax2.set_ylabel("Mass Drift", fontsize=11)
ax2.set_title("Effect of Warmup Duration on Mass Conservation", fontsize=12, fontweight="bold")
ax2.set_xticks(warmup_adam)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "warmup_sweep.pdf"), dpi=300)
fig.savefig(os.path.join(FIG_DIR, "warmup_sweep.png"), dpi=150)
plt.close(fig)

# ===========================================================================
# PLOT 4: Adam LR sweep
# ===========================================================================
print("Plot 4: Adam LR sweep...")

lr_keys   = ["adam_lr_low", "cahn_hilliard_canonical", "adam_lr_high"]
lr_values = [0.0001, 0.001, 0.01]
lr_losses = [experiments[k]["final_bfgs_loss"] for k in lr_keys]

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(lr_values, lr_losses, "D-", markersize=10, color="#ff7f0e", linewidth=2)
for x, y in zip(lr_values, lr_losses):
    ax.annotate(f"{y:.2e}", (x, y), textcoords="offset points",
                xytext=(10, 10), fontsize=9)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Adam Learning Rate", fontsize=11)
ax.set_ylabel("Final BFGS Loss", fontsize=11)
ax.set_title("Effect of Adam Learning Rate", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "adam_lr_sweep.pdf"), dpi=300)
fig.savefig(os.path.join(FIG_DIR, "adam_lr_sweep.png"), dpi=150)
plt.close(fig)

# ===========================================================================
# PLOT 5: BFGS batch size (resampling frequency)
# ===========================================================================
print("Plot 5: Batch size sweep...")

batch_keys   = ["bfgs_batch_200", "cahn_hilliard_canonical", "bfgs_batch_1000"]
batch_values = [200, 500, 1000]
batch_losses = [experiments[k]["final_bfgs_loss"] for k in batch_keys]

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(batch_values, batch_losses, "^-", markersize=10, color="#9467bd", linewidth=2)
for x, y in zip(batch_values, batch_losses):
    ax.annotate(f"{y:.2e}", (x, y), textcoords="offset points",
                xytext=(10, 10), fontsize=9)
ax.set_yscale("log")
ax.set_xlabel("BFGS Batch Size (N_change)", fontsize=11)
ax.set_ylabel("Final BFGS Loss", fontsize=11)
ax.set_title("Effect of Collocation Resampling Frequency", fontsize=12, fontweight="bold")
ax.set_xticks(batch_values)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "batch_size_sweep.pdf"), dpi=300)
fig.savefig(os.path.join(FIG_DIR, "batch_size_sweep.png"), dpi=150)
plt.close(fig)

# ===========================================================================
# PLOT 6: Power transform comparison
# ===========================================================================
print("Plot 6: Power transform...")

power_keys   = ["cahn_hilliard_canonical", "power_2", "power_4"]
power_values = [1, 2, 4]
power_losses = [experiments[k]["final_bfgs_loss"] for k in power_keys]

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.bar(range(3), power_losses, color=["#1f77b4", "#17becf", "#aec7e8"],
       edgecolor="k", linewidth=0.5, width=0.5)
ax.set_xticks(range(3))
ax.set_xticklabels([f"p = {p}" for p in power_values], fontsize=11)
ax.set_yscale("log")
ax.set_ylabel("Final BFGS Loss", fontsize=11)
ax.set_title("Power Transform: Minimize $L^{1/p}$", fontsize=12, fontweight="bold")
for i, v in enumerate(power_losses):
    ax.text(i, v * 1.3, f"{v:.2e}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "power_transform.pdf"), dpi=300)
fig.savefig(os.path.join(FIG_DIR, "power_transform.png"), dpi=150)
plt.close(fig)

# ===========================================================================
# PLOT 7: Mass drift comparison (all experiments)
# ===========================================================================
print("Plot 7: Mass drift comparison...")

drift_data = []
for k, v in experiments.items():
    if "mass_drift" in v:
        drift_data.append((v["mass_drift"], k))
drift_data.sort()

d_names = [NICE_NAMES.get(d[1], d[1]) for d in drift_data]
d_vals  = [d[0] for d in drift_data]
d_colors = [COLORS.get(d[1], "#333") for d in drift_data]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(d_names)), d_vals, color=d_colors, edgecolor="k", linewidth=0.5)
ax.set_yticks(range(len(d_names)))
ax.set_yticklabels(d_names, fontsize=9)
ax.set_xlabel("Mass Drift (|max <U> - min <U>|)", fontsize=11)
ax.set_title("Mass Conservation by Experiment", fontsize=13, fontweight="bold")
ax.invert_yaxis()
for i, v in enumerate(d_vals):
    ax.text(v + 0.0005, i, f"{v:.4f}", va="center", fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "mass_drift_bar.pdf"), dpi=300)
fig.savefig(os.path.join(FIG_DIR, "mass_drift_bar.png"), dpi=150)
plt.close(fig)

# ===========================================================================
# PLOT 8: Loss curves overlay — Optimizer Variants
# ===========================================================================
print("Plot 8: Loss curves overlay (variants)...")

fig, ax = plt.subplots(figsize=(10, 5))
for k in variant_keys:
    s = experiments[k]
    adam_l = s["_adam_losses"]
    bfgs_l = s["_bfgs_losses"]
    
    # Build full loss curve
    full = list(adam_l)
    adam_n = s["adam_epochs"]
    if len(bfgs_l) > 0:
        bfgs_valid = bfgs_l[bfgs_l > 0]
        # BFGS losses are logged every Nprint=100 iters
        bfgs_x = adam_n + (np.arange(1, len(bfgs_valid) + 1)) * 100
        ax.semilogy(bfgs_x, bfgs_valid, color=COLORS[k], linewidth=1.2,
                    label=NICE_NAMES[k], alpha=0.85)
    if len(adam_l) > 0:
        ax.semilogy(range(1, len(adam_l) + 1), adam_l, color=COLORS[k],
                    linewidth=0.6, alpha=0.4)

ax.axvline(2000, color="gray", ls="--", lw=0.8, alpha=0.5, label="Adam→BFGS")
ax.set_xlabel("Iteration", fontsize=11)
ax.set_ylabel("Loss (log)", fontsize=11)
ax.set_title("Training Curves: Optimizer Variants", fontsize=13, fontweight="bold")
ax.legend(fontsize=8, ncol=2, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "loss_curves_variants.pdf"), dpi=300)
fig.savefig(os.path.join(FIG_DIR, "loss_curves_variants.png"), dpi=150)
plt.close(fig)

# ===========================================================================
# PLOT 9: Loss curves overlay — Warmup sweep
# ===========================================================================
print("Plot 9: Loss curves overlay (warmup)...")

fig, ax = plt.subplots(figsize=(10, 5))
warmup_colors = {"warmup_0": "#c49c94", "warmup_500": "#f7b6d2",
                 "cahn_hilliard_canonical": "#1f77b4", "warmup_5000": "#dbdb8d"}
for k in warmup_keys:
    s = experiments[k]
    adam_l = s["_adam_losses"]
    bfgs_l = s["_bfgs_losses"]
    adam_n = s["adam_epochs"]
    
    if len(adam_l) > 0:
        ax.semilogy(range(1, len(adam_l) + 1), adam_l, color=warmup_colors[k],
                    linewidth=0.6, alpha=0.5)
    if len(bfgs_l) > 0:
        bfgs_valid = bfgs_l[bfgs_l > 0]
        bfgs_x = adam_n + (np.arange(1, len(bfgs_valid) + 1)) * 100
        ax.semilogy(bfgs_x, bfgs_valid, color=warmup_colors[k], linewidth=1.5,
                    label=NICE_NAMES[k], alpha=0.9)
    if adam_n > 0:
        ax.axvline(adam_n, color=warmup_colors[k], ls=":", lw=0.6, alpha=0.4)

ax.set_xlabel("Iteration", fontsize=11)
ax.set_ylabel("Loss (log)", fontsize=11)
ax.set_title("Training Curves: Warmup Duration Sweep", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "loss_curves_warmup.pdf"), dpi=300)
fig.savefig(os.path.join(FIG_DIR, "loss_curves_warmup.png"), dpi=150)
plt.close(fig)

# ===========================================================================
# PLOT 10: Scatter — Loss vs Time (Pareto front)
# ===========================================================================
print("Plot 10: Loss vs time scatter (Pareto)...")

fig, ax = plt.subplots(figsize=(8, 6))
for k, s in experiments.items():
    loss = s.get("final_bfgs_loss") or s.get("final_adam_loss") or 1.0
    time_h = s["total_time_s"] / 3600
    ax.scatter(time_h, loss, c=COLORS.get(k, "#333"), s=80, edgecolors="k",
               linewidth=0.5, zorder=5)
    ax.annotate(NICE_NAMES.get(k, k), (time_h, loss),
                textcoords="offset points", xytext=(6, 6), fontsize=6.5)

ax.set_yscale("log")
ax.set_xlabel("Total Wall Time (hours)", fontsize=11)
ax.set_ylabel("Final Loss (log)", fontsize=11)
ax.set_title("Loss vs. Compute Cost", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "pareto_loss_time.pdf"), dpi=300)
fig.savefig(os.path.join(FIG_DIR, "pareto_loss_time.png"), dpi=150)
plt.close(fig)

# ===========================================================================
# PLOT 11: Hessian scaling comparison
# ===========================================================================
print("Plot 11: Hessian scaling...")

h_keys = ["cahn_hilliard_canonical", "hessian_scaled"]
h_labels = ["No H₀ Scaling", "Nocedal-Wright H₀"]
h_losses = [experiments[k]["final_bfgs_loss"] for k in h_keys]
h_times  = [experiments[k]["total_time_s"] / 3600 for k in h_keys]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.bar(h_labels, h_losses, color=["#1f77b4", "#7f7f7f"], edgecolor="k", width=0.4)
ax1.set_ylabel("Final Loss", fontsize=10)
ax1.set_title("Loss", fontsize=11, fontweight="bold")
for i, v in enumerate(h_losses):
    ax1.text(i, v + 0.0003, f"{v:.4f}", ha="center", fontsize=9)

ax2.bar(h_labels, h_times, color=["#1f77b4", "#7f7f7f"], edgecolor="k", width=0.4)
ax2.set_ylabel("Time (hours)", fontsize=10)
ax2.set_title("Wall Time", fontsize=11, fontweight="bold")
for i, v in enumerate(h_times):
    ax2.text(i, v + 0.05, f"{v:.1f}h", ha="center", fontsize=9)

fig.suptitle("Nocedal-Wright Initial Hessian Scaling", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "hessian_scaling.pdf"), dpi=300)
fig.savefig(os.path.join(FIG_DIR, "hessian_scaling.png"), dpi=150)
plt.close(fig)

# ===========================================================================
# PLOT 12: Summary radar / heatmap
# ===========================================================================
print("Plot 12: Summary heatmap...")

# Create a heatmap: experiments x metrics (normalized to [0,1])
heatmap_keys = [
    "cahn_hilliard_canonical", "adam_lr_high", "adam_lr_low", "adam_only",
    "bfgs_batch_200", "bfgs_batch_1000", "bfgs_vanilla", "hessian_scaled",
    "lbfgs", "power_2", "power_4", "ssbfgs_ab", "ssbfgs_ol", "ssbroyden1",
    "warmup_0", "warmup_500", "warmup_5000"
]

metrics_names = ["Final Loss", "Total Time (s)", "Mass Drift"]
raw = np.zeros((len(heatmap_keys), 3))
for i, k in enumerate(heatmap_keys):
    s = experiments[k]
    raw[i, 0] = s.get("final_bfgs_loss") or s.get("final_adam_loss") or 1.0
    raw[i, 1] = s["total_time_s"]
    raw[i, 2] = s.get("mass_drift", 0.0)

# Log-normalize final loss
log_loss = np.log10(raw[:, 0] + 1e-10)
normed = np.zeros_like(raw)
for j in range(3):
    col = raw[:, j] if j > 0 else log_loss
    mn, mx = col.min(), col.max()
    if mx > mn:
        normed[:, j] = (col - mn) / (mx - mn)
    else:
        normed[:, j] = 0.5

fig, ax = plt.subplots(figsize=(7, 9))
im = ax.imshow(normed, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(3))
ax.set_xticklabels(metrics_names, fontsize=10)
ax.set_yticks(range(len(heatmap_keys)))
ax.set_yticklabels([NICE_NAMES.get(k, k) for k in heatmap_keys], fontsize=8)

# Annotate with raw values
for i in range(len(heatmap_keys)):
    ax.text(0, i, f"{raw[i,0]:.2e}", ha="center", va="center", fontsize=6.5)
    ax.text(1, i, f"{raw[i,1]:.0f}", ha="center", va="center", fontsize=6.5)
    ax.text(2, i, f"{raw[i,2]:.4f}", ha="center", va="center", fontsize=6.5)

ax.set_title("Experiment Overview (Green = Better)", fontsize=12, fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Normalized (0=best, 1=worst)")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "summary_heatmap.pdf"), dpi=300)
fig.savefig(os.path.join(FIG_DIR, "summary_heatmap.png"), dpi=150)
plt.close(fig)

print(f"\nAll plots saved to {FIG_DIR}/")
print("Files:")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"  {f}")
