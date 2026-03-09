"""
04_evaluate.py — Spectrum-to-spectrum evaluation of ALIGNN Raman model
Model predicts a 100-bin normalised Raman spectrum per crystal structure.

Usage:
    python 04_evaluate.py \
        --data_dir   alignn_dataset \
        --model_path runs/full_local_001/best_model.pt \
        --output_dir runs/full_local_001/eval
"""

import os
import json
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")

import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from scipy.spatial.distance import cosine as cosine_distance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────────────────────
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
ACCENT  = "#4C72B0"
ACCENT2 = "#DD8452"
GOOD    = "#55A868"
BAD     = "#C44E52"
GRAY    = "#8C8C8C"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_test_data(data_dir: str):
    data_dir = Path(data_dir)
    test_ids = (data_dir / "test.txt").read_text().strip().splitlines()
    test_ids = [t.strip() for t in test_ids if t.strip()]
    log.info(f"Test IDs: {len(test_ids)}")

    id_to_target = {}
    with open(data_dir / "id_prop.csv") as f:
        for row in csv.DictReader(f):
            jid = row["jid"].strip()
            arr = np.array(json.loads(row["target"]), dtype=np.float32)
            id_to_target[jid] = arr

    structures_dir = data_dir / "structures"
    records, missing = [], []
    for jid in test_ids:
        vasp_path = structures_dir / f"{jid}.vasp"
        if not vasp_path.exists() or jid not in id_to_target:
            missing.append(jid)
            continue
        records.append({"jid": jid, "target": id_to_target[jid],
                        "atoms_path": str(vasp_path)})
    if missing:
        log.warning(f"  Skipped {len(missing)} entries (missing file or target)")
    log.info(f"Loaded {len(records)} test records")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Graph building
# ─────────────────────────────────────────────────────────────────────────────

def build_graphs(records, cutoff: float = 8.0):
    from jarvis.core.atoms import Atoms
    from alignn.graphs import Graph
    graphs, failed = [], []
    for i, rec in enumerate(tqdm(records, desc="Building graphs", unit="struct")):
        try:
            atoms   = Atoms.from_poscar(rec["atoms_path"])
            g, lg   = Graph.atom_dgl_multigraph(
                atoms, cutoff=cutoff, atom_features="cgcnn",
                max_neighbors=12, use_canonize=True,
            )
            lattice = torch.tensor(atoms.lattice_mat,
                                   dtype=torch.float32).unsqueeze(0)
            graphs.append((g, lg, lattice, rec["target"], rec["jid"]))
        except Exception as e:
            failed.append((rec.get("jid", i), str(e)))
    if failed:
        log.warning(f"  {len(failed)} structures failed graph build")
    log.info(f"Built {len(graphs)} graphs successfully.")
    return graphs


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str, device: torch.device, n_bins: int = 100):
    from alignn.models.alignn import ALIGNN, ALIGNNConfig
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict  = checkpoint["model_state"]
        config_dict = checkpoint.get("model_config", {})
    else:
        state_dict  = checkpoint
        config_dict = {}

    if config_dict:
        config = ALIGNNConfig(**config_dict)
    else:
        config = ALIGNNConfig(name="alignn", output_features=n_bins)
    model = ALIGNN(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    log.info(f"Loaded model  ({sum(p.numel() for p in model.parameters()):,} params)")
    log.info(f"  Best val MAE from checkpoint: {checkpoint.get('best_val_mae', 'N/A')}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, graph_data, device, batch_size: int = 32):
    import dgl
    preds, targets, jids = [], [], []
    for start in tqdm(range(0, len(graph_data), batch_size),
                      desc="Inference", unit="batch"):
        chunk = graph_data[start: start + batch_size]
        gs    = dgl.batch([c[0] for c in chunk]).to(device)
        lgs   = dgl.batch([c[1] for c in chunk]).to(device)
        lat   = torch.cat([c[2] for c in chunk], dim=0).to(device)
        tgt   = torch.stack([torch.tensor(c[3]) for c in chunk])

        out = model([gs, lgs, lat])
        preds.append(out.cpu())
        targets.append(tgt)
        jids.extend([c[4] for c in chunk])

    preds   = torch.cat(preds).numpy()    # (N, 100)
    targets = torch.cat(targets).numpy()  # (N, 100)
    return preds, targets, jids


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(a, b):
    return np.array([1.0 - cosine_distance(a[i], b[i]) for i in range(len(a))])

def peak_position_error(y_true, y_pred, wavenums):
    true_peaks = wavenums[np.argmax(y_true, axis=1)]
    pred_peaks = wavenums[np.argmax(y_pred, axis=1)]
    return np.abs(true_peaks - pred_peaks), true_peaks, pred_peaks

def spectral_overlap(y_true, y_pred):
    """Bhattacharyya coefficient — clips negatives before computing."""
    overlaps = []
    for i in range(len(y_true)):
        t = np.clip(y_true[i], 0, None)
        p = np.clip(y_pred[i], 0, None)
        t = t / (t.sum() + 1e-10)
        p = p / (p.sum() + 1e-10)
        overlaps.append(float(np.sum(np.sqrt(t * p))))
    return np.array(overlaps)

def compute_all_metrics(y_true, y_pred, wavenums):
    yt_flat = y_true.flatten()
    yp_flat = y_pred.flatten()

    mae_global  = mean_absolute_error(yt_flat, yp_flat)
    rmse_global = np.sqrt(mean_squared_error(yt_flat, yp_flat))
    r2_global   = r2_score(yt_flat, yp_flat)

    per_sample_mae = np.mean(np.abs(y_true - y_pred), axis=1)
    per_sample_r2  = np.array([r2_score(y_true[i], y_pred[i]) for i in range(len(y_true))])
    cos_sims       = cosine_similarity(y_true, y_pred)
    peak_errs, true_peaks, pred_peaks = peak_position_error(y_true, y_pred, wavenums)
    overlaps       = spectral_overlap(y_true, y_pred)
    per_bin_mae    = np.mean(np.abs(y_true - y_pred), axis=0)
    per_bin_r2     = np.array([r2_score(y_true[:, b], y_pred[:, b])
                                for b in range(y_true.shape[1])])

    pearson_r,  _ = stats.pearsonr(yt_flat, yp_flat)
    spearman_r, _ = stats.spearmanr(yt_flat, yp_flat)

    return {
        "N":                     len(y_true),
        "Global MAE":            mae_global,
        "Global RMSE":           rmse_global,
        "Global R²":             r2_global,
        "Pearson r":             pearson_r,
        "Spearman r":            spearman_r,
        "per_sample_mae":        per_sample_mae,
        "per_sample_r2":         per_sample_r2,
        "cosine_similarity":     cos_sims,
        "peak_error_cm":         peak_errs,
        "true_peaks":            true_peaks,
        "pred_peaks":            pred_peaks,
        "spectral_overlap":      overlaps,
        "per_bin_mae":           per_bin_mae,
        "per_bin_r2":            per_bin_r2,
        "Mean cosine sim":       cos_sims.mean(),
        "Median cosine sim":     np.median(cos_sims),
        "Mean peak error cm":    peak_errs.mean(),
        "Median peak error cm":  np.median(peak_errs),
        "Mean spectral overlap": overlaps.mean(),
        "Frac cosine > 0.90":    (cos_sims > 0.90).mean() * 100,
        "Frac cosine > 0.95":    (cos_sims > 0.95).mean() * 100,
        "Frac cosine > 0.99":    (cos_sims > 0.99).mean() * 100,
        "Mean per-sample MAE":   per_sample_mae.mean(),
        "Median per-sample MAE": np.median(per_sample_mae),
        "Mean per-sample R²":    per_sample_r2.mean(),
    }

def print_metrics(m):
    log.info("=" * 56)
    log.info("  TEST SET METRICS  —  Spectrum Prediction (100 bins)")
    log.info("=" * 56)
    log.info(f"  N structures         : {m['N']}")
    log.info("  — Global (all bins flattened) —")
    log.info(f"  Global MAE           : {m['Global MAE']:.5f}")
    log.info(f"  Global RMSE          : {m['Global RMSE']:.5f}")
    log.info(f"  Global R²            : {m['Global R²']:.5f}")
    log.info(f"  Pearson r            : {m['Pearson r']:.5f}")
    log.info(f"  Spearman r           : {m['Spearman r']:.5f}")
    log.info("  — Per-sample —")
    log.info(f"  Mean MAE/spectrum    : {m['Mean per-sample MAE']:.5f}")
    log.info(f"  Median MAE/spectrum  : {m['Median per-sample MAE']:.5f}")
    log.info(f"  Mean R²/spectrum     : {m['Mean per-sample R²']:.5f}")
    log.info("  — Spectral similarity —")
    log.info(f"  Mean cosine sim      : {m['Mean cosine sim']:.5f}")
    log.info(f"  Median cosine sim    : {m['Median cosine sim']:.5f}")
    log.info(f"  Cosine sim > 0.90    : {m['Frac cosine > 0.90']:.1f}%")
    log.info(f"  Cosine sim > 0.95    : {m['Frac cosine > 0.95']:.1f}%")
    log.info(f"  Cosine sim > 0.99    : {m['Frac cosine > 0.99']:.1f}%")
    log.info(f"  Mean spectral overlap: {m['Mean spectral overlap']:.5f}")
    log.info("  — Peak position —")
    log.info(f"  Mean peak error      : {m['Mean peak error cm']:.2f} cm⁻¹")
    log.info(f"  Median peak error    : {m['Median peak error cm']:.2f} cm⁻¹")
    log.info("=" * 56)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_dashboard(y_true, y_pred, metrics, wavenums, out_path):
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    # 1. Mean predicted vs mean actual spectrum
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(wavenums, y_true.mean(axis=0), color=ACCENT,  lw=2, label="Actual (mean)")
    ax1.plot(wavenums, y_pred.mean(axis=0), color=ACCENT2, lw=2, linestyle="--", label="Predicted (mean)")
    ax1.fill_between(wavenums,
                     y_true.mean(0) - y_true.std(0),
                     y_true.mean(0) + y_true.std(0),
                     alpha=0.15, color=ACCENT, label="±1σ actual")
    ax1.set_xlabel("Wavenumber (cm⁻¹)"); ax1.set_ylabel("Normalised Intensity")
    ax1.set_title("Mean Spectrum: Actual vs Predicted"); ax1.legend()

    # 2. Cosine similarity distribution
    ax2 = fig.add_subplot(gs[0, 2])
    cos = metrics["cosine_similarity"]
    ax2.hist(cos, bins=30, color=ACCENT, edgecolor="white", alpha=0.85)
    ax2.axvline(cos.mean(), color=BAD, lw=2, linestyle="--",
                label=f"Mean={cos.mean():.3f}")
    ax2.set_xlabel("Cosine Similarity"); ax2.set_ylabel("Count")
    ax2.set_title("Cosine Similarity Distribution"); ax2.legend()

    # 3. Per-bin MAE
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(wavenums, metrics["per_bin_mae"], color=BAD, lw=1.5)
    ax3.fill_between(wavenums, 0, metrics["per_bin_mae"], alpha=0.25, color=BAD)
    ax3.set_xlabel("Wavenumber (cm⁻¹)"); ax3.set_ylabel("MAE")
    ax3.set_title("Per-Bin MAE Across Test Set")

    # 4. Peak position error histogram
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(metrics["peak_error_cm"], bins=30, color=ACCENT2, edgecolor="white", alpha=0.85)
    ax4.axvline(metrics["Mean peak error cm"], color=BAD, lw=2, linestyle="--",
                label=f"Mean={metrics['Mean peak error cm']:.1f} cm⁻¹")
    ax4.set_xlabel("Peak Position Error (cm⁻¹)"); ax4.set_ylabel("Count")
    ax4.set_title("Peak Position Error"); ax4.legend()

    # 5. Per-bin R²
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.plot(wavenums, metrics["per_bin_r2"], color=GOOD, lw=1.5)
    ax5.fill_between(wavenums, 0, np.clip(metrics["per_bin_r2"], 0, 1),
                     alpha=0.2, color=GOOD)
    ax5.axhline(0, color=GRAY, lw=1, linestyle=":")
    ax5.set_xlabel("Wavenumber (cm⁻¹)"); ax5.set_ylabel("R²")
    ax5.set_title("Per-Bin R² Across Test Set")
    ax5.set_ylim(-0.1, 1.05)

    # 6. Metrics table
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis("off")
    table_data = [
        ["Metric", "Value"],
        ["N",                 f"{metrics['N']}"],
        ["Global MAE",        f"{metrics['Global MAE']:.5f}"],
        ["Global R²",         f"{metrics['Global R²']:.5f}"],
        ["Mean cosine sim",   f"{metrics['Mean cosine sim']:.4f}"],
        ["Cosine > 0.95",     f"{metrics['Frac cosine > 0.95']:.1f}%"],
        ["Mean R²/spectrum",  f"{metrics['Mean per-sample R²']:.4f}"],
        ["Mean peak err",     f"{metrics['Mean peak error cm']:.1f} cm⁻¹"],
        ["Spectral overlap",  f"{metrics['Mean spectral overlap']:.4f}"],
    ]
    tbl = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EEF2F8")
        cell.set_edgecolor("white")

    fig.suptitle("ALIGNN Raman Spectrum Prediction — Test Set Evaluation",
                 fontsize=15, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


def plot_random_spectrum_overlays(y_true, y_pred, jids, wavenums, out_path, n: int = 12):
    idx   = np.random.choice(len(y_true), size=min(n, len(y_true)), replace=False)
    ncols = 4
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2))
    axes = axes.flatten()
    for i, si in enumerate(idx):
        ax = axes[i]
        cos_sim = 1.0 - cosine_distance(y_true[si], y_pred[si])
        ax.plot(wavenums, y_true[si],  color=ACCENT,  lw=1.8, label="Actual")
        ax.plot(wavenums, y_pred[si],  color=ACCENT2, lw=1.8, linestyle="--", label="Predicted")
        ax.fill_between(wavenums, y_true[si], y_pred[si], alpha=0.12, color=GRAY)
        ax.set_title(f"{jids[si]}\ncos={cos_sim:.3f}", fontsize=8)
        ax.set_xlabel("cm⁻¹", fontsize=7)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Random Test Spectra: Actual vs Predicted", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


def plot_best_worst_spectra(y_true, y_pred, jids, metrics, wavenums, out_path, n: int = 4):
    cos       = metrics["cosine_similarity"]
    best_idx  = np.argsort(cos)[-n:][::-1]
    worst_idx = np.argsort(cos)[:n]
    fig, axes = plt.subplots(2, n, figsize=(16, 8))
    for col, (bi, wi) in enumerate(zip(best_idx, worst_idx)):
        for row, (idx, label, color) in enumerate([
            (bi, "BEST",  GOOD),
            (wi, "WORST", BAD),
        ]):
            ax = axes[row, col]
            ax.plot(wavenums, y_true[idx], color=ACCENT, lw=2,  label="Actual")
            ax.plot(wavenums, y_pred[idx], color=color,  lw=2,  linestyle="--", label="Predicted")
            ax.fill_between(wavenums, y_true[idx], y_pred[idx], alpha=0.15, color=color)
            ax.set_title(f"{label}: {jids[idx]}\nCos={cos[idx]:.4f}", fontsize=8)
            ax.set_xlabel("cm⁻¹", fontsize=8)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.legend(fontsize=7)
    fig.suptitle(f"Top-{n} Best and Worst Predictions by Cosine Similarity",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


def plot_cosine_vs_peak_error(metrics, out_path):
    cos  = metrics["cosine_similarity"]
    perr = metrics["peak_error_cm"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(cos, perr, alpha=0.35, s=14,
                    c=metrics["per_sample_mae"], cmap="viridis", edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Per-sample MAE")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Peak Position Error (cm⁻¹)")
    ax.set_title("Cosine Similarity vs Peak Position Error\n(color = per-sample MAE)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


def plot_per_sample_distributions(metrics, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cos = metrics["cosine_similarity"]
    axes[0].hist(cos, bins=35, color=ACCENT, edgecolor="white", alpha=0.85)
    axes[0].axvline(cos.mean(),      color=BAD,  lw=2, linestyle="--", label=f"Mean={cos.mean():.4f}")
    axes[0].axvline(np.median(cos),  color=GOOD, lw=2, linestyle=":",  label=f"Median={np.median(cos):.4f}")
    axes[0].set_xlabel("Cosine Similarity"); axes[0].set_ylabel("Count")
    axes[0].set_title("Per-Sample Cosine Similarity"); axes[0].legend()

    r2s = metrics["per_sample_r2"]
    axes[1].hist(r2s, bins=35, color=ACCENT2, edgecolor="white", alpha=0.85)
    axes[1].axvline(r2s.mean(), color=BAD, lw=2, linestyle="--", label=f"Mean={r2s.mean():.4f}")
    axes[1].set_xlabel("R² (per spectrum)"); axes[1].set_ylabel("Count")
    axes[1].set_title("Per-Sample R²"); axes[1].legend()

    ov = metrics["spectral_overlap"]
    axes[2].hist(ov, bins=35, color=GOOD, edgecolor="white", alpha=0.85)
    axes[2].axvline(ov.mean(), color=BAD, lw=2, linestyle="--", label=f"Mean={ov.mean():.4f}")
    axes[2].set_xlabel("Spectral Overlap (Bhattacharyya)"); axes[2].set_ylabel("Count")
    axes[2].set_title("Per-Sample Spectral Overlap"); axes[2].legend()

    fig.suptitle("Per-Sample Metric Distributions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


def plot_mean_spectrum_with_std(y_true, y_pred, wavenums, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, arr, label, color in [
        (axes[0], y_true, "Actual",    ACCENT),
        (axes[1], y_pred, "Predicted", ACCENT2),
    ]:
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        ax.plot(wavenums, mean, color=color, lw=2, label=f"Mean {label}")
        ax.fill_between(wavenums, mean - std, mean + std,
                        alpha=0.25, color=color, label="±1σ")
        ax.fill_between(wavenums,
                        np.percentile(arr, 5,  axis=0),
                        np.percentile(arr, 95, axis=0),
                        alpha=0.12, color=color, label="5–95 pctile")
        ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Normalised Intensity")
        ax.set_title(f"{label} Spectrum Distribution"); ax.legend()
    fig.suptitle("Spectrum Distributions Across Test Set", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


def plot_error_heatmap(y_true, y_pred, jids, wavenums, out_path, max_samples: int = 100):
    n   = min(len(y_true), max_samples)
    cos = np.array([1.0 - cosine_distance(y_true[i], y_pred[i]) for i in range(len(y_true))])
    idx = np.argsort(cos)[::-1][:n]
    err_map = np.abs(y_true[idx] - y_pred[idx])
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(err_map, aspect="auto", cmap="hot_r", origin="upper",
                   extent=[wavenums[0], wavenums[-1], n, 0])
    plt.colorbar(im, ax=ax, label="|Error|")
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel(f"Structure (sorted best→worst cosine, n={n})")
    ax.set_title("Absolute Error Heatmap: Structures × Wavenumber")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


def plot_global_parity(y_true, y_pred, out_path):
    yt = y_true.flatten()
    yp = y_pred.flatten()
    if len(yt) > 50000:
        idx = np.random.choice(len(yt), 50000, replace=False)
        yt, yp = yt[idx], yp[idx]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hexbin(yt, yp, gridsize=50, cmap="Blues", mincnt=1, linewidths=0.2)
    lo = min(yt.min(), yp.min()) * 0.97
    hi = max(yt.max(), yp.max()) * 1.03
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Perfect")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual Intensity (per bin)")
    ax.set_ylabel("Predicted Intensity (per bin)")
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    ax.set_title(f"Global Bin-Level Parity  R²={r2:.4f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_predictions(y_true, y_pred, jids, metrics, out_dir):
    cos  = metrics["cosine_similarity"]
    r2s  = metrics["per_sample_r2"]
    perr = metrics["peak_error_cm"]
    mae  = metrics["per_sample_mae"]
    ov   = metrics["spectral_overlap"]

    csv_path = out_dir / "predictions_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["jid", "cosine_sim", "per_sample_r2", "per_sample_mae",
                    "peak_error_cm", "spectral_overlap",
                    "true_peak_cm", "pred_peak_cm"])
        for i, jid in enumerate(jids):
            w.writerow([jid, f"{cos[i]:.6f}", f"{r2s[i]:.6f}", f"{mae[i]:.6f}",
                        f"{perr[i]:.4f}", f"{ov[i]:.6f}",
                        f"{metrics['true_peaks'][i]:.2f}",
                        f"{metrics['pred_peaks'][i]:.2f}"])
    log.info(f"  Saved: {csv_path}")

    npz_path = out_dir / "predictions_full.npz"
    np.savez_compressed(str(npz_path), y_true=y_true, y_pred=y_pred,
                        jids=np.array(jids))
    log.info(f"  Saved: {npz_path}")


def save_metrics_json(metrics, out_path):
    serialisable = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            serialisable[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            serialisable[k] = float(v)
        else:
            serialisable[k] = v
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    log.info(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="alignn_dataset")
    parser.add_argument("--model_path", default="runs/full_local_001/best_model.pt")
    parser.add_argument("--output_dir", default="runs/full_local_001/eval")
    parser.add_argument("--cutoff",     type=float, default=8.0)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--n_bins",     type=int,   default=100)
    parser.add_argument("--wn_min",     type=float, default=0.0)
    parser.add_argument("--wn_max",     type=float, default=1000.0)
    args = parser.parse_args()

    out      = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    wavenums = np.linspace(args.wn_min, args.wn_max, args.n_bins)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    records    = load_test_data(args.data_dir)
    graph_data = build_graphs(records, cutoff=args.cutoff)
    model      = load_model(args.model_path, device, n_bins=args.n_bins)
    y_pred, y_true, jids = run_inference(model, graph_data, device, args.batch_size)

    # ── Diagnostic: inspect raw prediction values ────────────────────────────
    log.info("── Prediction diagnostics ──")
    log.info(f"  y_pred shape : {y_pred.shape}")
    log.info(f"  y_pred range : [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    log.info(f"  y_pred mean  : {y_pred.mean():.4f}  std={y_pred.std():.4f}")
    log.info(f"  y_true range : [{y_true.min():.4f}, {y_true.max():.4f}]")
    log.info(f"  y_true mean  : {y_true.mean():.4f}  std={y_true.std():.4f}")
    log.info(f"  Sample pred[0][:10] : {y_pred[0][:10].round(4).tolist()}")
    log.info(f"  Sample true[0][:10] : {y_true[0][:10].round(4).tolist()}")
    log.info("────────────────────────────")

    metrics = compute_all_metrics(y_true, y_pred, wavenums)
    print_metrics(metrics)
    save_metrics_json(metrics, out / "metrics.json")

    log.info("Generating plots...")
    plot_summary_dashboard       (y_true, y_pred, metrics, wavenums, out / "00_summary_dashboard.png")
    plot_mean_spectrum_with_std  (y_true, y_pred, wavenums,           out / "01_mean_spectrum_std.png")
    plot_random_spectrum_overlays(y_true, y_pred, jids, wavenums,     out / "02_random_spectra.png")
    plot_best_worst_spectra      (y_true, y_pred, jids, metrics, wavenums, out / "03_best_worst_spectra.png")
    plot_per_sample_distributions(metrics,                             out / "04_per_sample_distributions.png")
    plot_cosine_vs_peak_error    (metrics,                             out / "05_cosine_vs_peak_error.png")
    plot_error_heatmap           (y_true, y_pred, jids, wavenums,     out / "06_error_heatmap.png")
    plot_global_parity           (y_true, y_pred,                     out / "07_global_parity.png")

    save_predictions(y_true, y_pred, jids, metrics, out)

    log.info(f"\nAll outputs saved to: {out}/")
    log.info("Done.")


if __name__ == "__main__":
    main()
