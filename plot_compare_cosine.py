"""
plot_compare_cosine.py — Overlay cosine similarity histograms from multiple runs

Usage:
    python plot_compare_cosine.py \
        --runs runs/local_bins50_002/eval runs/local_bins500_001/eval \
        --labels "50 bins" "500 bins" \
        --output_dir runs/run_compare_plots
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cosine as cosine_distance

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
})

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def load_cosine_sims(eval_dir: str):
    path = Path(eval_dir) / "predictions_full.npz"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")
    data    = np.load(str(path), allow_pickle=True)
    y_true  = data["y_true"]
    y_pred  = data["y_pred"]
    cos_sims = np.array([1.0 - cosine_distance(y_true[i], y_pred[i])
                         for i in range(len(y_true))])
    return cos_sims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",       nargs="+", required=True,
                        help="Paths to eval directories")
    parser.add_argument("--labels",     nargs="+", required=True,
                        help="Label for each run (same order as --runs)")
    parser.add_argument("--output_dir", default="runs/run_compare_plots")
    parser.add_argument("--bins",       type=int, default=40,
                        help="Number of histogram bins")
    args = parser.parse_args()

    if len(args.runs) != len(args.labels):
        raise ValueError("--runs and --labels must have the same number of entries")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load cosine sims for each run
    all_sims = []
    for run_dir in args.runs:
        print(f"Loading {run_dir} ...")
        sims = load_cosine_sims(run_dir)
        all_sims.append(sims)
        print(f"  N={len(sims)}  mean={sims.mean():.4f}  median={np.median(sims):.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    for sims, label, color in zip(all_sims, args.labels, COLORS):
        ax.hist(sims, bins=args.bins, alpha=0.5, color=color,
                edgecolor="white", linewidth=0.4, label=label)
        ax.axvline(sims.mean(), color=color, lw=2, linestyle="--",
                   label=f"{label} mean={sims.mean():.3f}")

    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Cosine Similarity Distribution — Run Comparison")
    ax.legend()

    out_path = out / "compare_cosine_histogram.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
