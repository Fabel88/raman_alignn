"""
05_clean_plots.py — Clean diagnostic plots
Loads from eval dir (predictions_full.npz + predictions_summary.csv)

Usage:
    python 05_clean_plots.py \
        --eval_dir runs/full_local_001/eval \
        --wn_min 50 --wn_max 1000
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--eval_dir", default="runs/full_local_001/eval")
parser.add_argument("--wn_min",   type=float, default=50.0)
parser.add_argument("--wn_max",   type=float, default=1000.0)
parser.add_argument("--n_bins",   type=int,   default=100)
args = parser.parse_args()

EVAL     = Path(args.eval_dir)
WAVENUMS = np.linspace(args.wn_min, args.wn_max, args.n_bins)

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelsize":    13,
    "axes.titlesize":    14,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   11,
    "figure.facecolor":  "white",
})

BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"
PURPLE = "#8172B2"

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading from {EVAL}/")
npz      = np.load(EVAL / "predictions_full.npz", allow_pickle=True)
y_true   = npz["y_true"]
y_pred   = npz["y_pred"]
jids     = npz["jids"].tolist()
summary  = pd.read_csv(EVAL / "predictions_summary.csv")
cos_sims = summary["cosine_sim"].values
mae_vals = summary["per_sample_mae"].values
print(f"  {len(y_true)} structures loaded")


# ── Plot 1: Cosine Similarity Histogram ───────────────────────────────────────
def plot_cosine_histogram(cos_sims, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(cos_sims, bins=40, color=BLUE, edgecolor="white", linewidth=0.6, alpha=0.88)

    mean   = cos_sims.mean()
    median = np.median(cos_sims)
    ax.axvline(mean,   color="black", lw=1.8, linestyle="--", label=f"Mean = {mean:.3f}")
    ax.axvline(median, color=PURPLE,  lw=1.8, linestyle=":",  label=f"Median = {median:.3f}")

    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Number of Structures")
    ax.set_title("Cosine Similarity Distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot 2: MAE Histogram ─────────────────────────────────────────────────────
def plot_mae_histogram(mae_vals, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(mae_vals, bins=40, color=BLUE, edgecolor="white", linewidth=0.6, alpha=0.88)

    mean   = mae_vals.mean()
    median = np.median(mae_vals)
    ax.axvline(mean,   color="black", lw=1.8, linestyle="--", label=f"Mean = {mean:.4f}")
    ax.axvline(median, color=PURPLE,  lw=1.8, linestyle=":",  label=f"Median = {median:.4f}")

    ax.set_xlabel("Per-Sample MAE")
    ax.set_ylabel("Number of Structures")
    ax.set_title("MAE Distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot 3: Spectrum Gallery ──────────────────────────────────────────────────
def plot_spectrum_gallery(y_true, y_pred, jids, cos_sims, wavenums, out_path):
    sorted_idx = np.argsort(cos_sims)
    best_idx   = sorted_idx[-2:][::-1]

    mid_pool = np.where((cos_sims >= 0.70) & (cos_sims <= 0.80))[0]
    if len(mid_pool) >= 2:
        mid_idx = mid_pool[np.argsort(np.abs(cos_sims[mid_pool] - 0.75))[:2]]
    else:
        mid_idx = np.argsort(np.abs(cos_sims - 0.75))[:2]

    worst_idx = sorted_idx[:2]

    groups = [
        (best_idx,  BLUE,   GREEN),
        (mid_idx,   BLUE,   ORANGE),
        (worst_idx, BLUE,   RED),
    ]

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.30)

    for col, (idx_pair, actual_color, pred_color) in enumerate(groups):
        for row, idx in enumerate(idx_pair):
            ax      = fig.add_subplot(gs[row, col])
            actual  = y_true[idx]
            pred    = y_pred[idx]
            cos_val = cos_sims[idx]
            jid     = jids[idx] if isinstance(jids[idx], str) else jids[idx].decode()

            ax.fill_between(wavenums, actual, pred,
                            where=(actual >= pred),
                            alpha=0.15, color=actual_color, interpolate=True)
            ax.fill_between(wavenums, actual, pred,
                            where=(actual < pred),
                            alpha=0.15, color=pred_color, interpolate=True)

            ax.plot(wavenums, actual, color=actual_color, lw=2.0,
                    label="Actual", zorder=3)
            ax.plot(wavenums, pred,   color=pred_color,   lw=2.0,
                    linestyle="--", label="Predicted", zorder=4)

            ax.set_xlim(wavenums[0], wavenums[-1])
            ax.set_ylim(-0.05, 1.12)
            ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=10)
            ax.set_ylabel("Norm. Intensity",   fontsize=10)
            ax.set_title(f"{jid}\nCos = {cos_val:.4f}", fontsize=10)
            ax.tick_params(labelsize=9)

            if row == 0 and col == 0:
                ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("Raman Spectrum Predictions", fontsize=15, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Run ───────────────────────────────────────────────────────────────────────
print("\nGenerating plots...")
plot_cosine_histogram(cos_sims, EVAL / "clean_01_cosine_histogram.png")
plot_mae_histogram   (mae_vals, EVAL / "clean_02_mae_histogram.png")
plot_spectrum_gallery(y_true, y_pred, jids, cos_sims, WAVENUMS,
                      EVAL / "clean_03_spectrum_gallery.png")
print(f"\nDone. Saved to: {EVAL}")
