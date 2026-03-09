"""
plot_learning_curve.py — Plot training history from a saved checkpoint
Usage:
    python plot_learning_curve.py \
        --checkpoint runs/full_local_001/best_model.pt \
        --output_dir runs/full_local_001
"""

import argparse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",  default="runs/full_local_001/best_model.pt")
parser.add_argument("--output_dir",  default="runs/full_local_001")
args = parser.parse_args()

OUT  = Path(args.output_dir)
ckpt = torch.load(args.checkpoint, map_location="cpu")
history = ckpt.get("history", [])

if not history:
    print("No history found in checkpoint.")
    exit()

epochs     = [h["epoch"]     for h in history]
train_mae  = [h["train_mae"] for h in history]
val_mae    = [h["val_mae"]   for h in history]
lr         = [h.get("lr", None) for h in history]

best_epoch = epochs[val_mae.index(min(val_mae))]
best_mae   = min(val_mae)

print(f"Epochs in history : {len(history)}")
print(f"Best val MAE      : {best_mae:.5f} at epoch {best_epoch}")

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.labelsize":   12,
    "axes.titlesize":   13,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.facecolor": "white",
})

BLUE   = "#4C72B0"
ORANGE = "#DD8452"
PURPLE = "#8172B2"

has_lr = any(v is not None for v in lr)
n_cols = 2 if has_lr else 1
fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4.5))
if n_cols == 1:
    axes = [axes]

# ── MAE panel ─────────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(epochs, train_mae, color=BLUE,   lw=2.0, label="Train MAE")
ax.plot(epochs, val_mae,   color=ORANGE, lw=2.0, label="Val MAE")
ax.axvline(best_epoch, color="black", lw=1.2, linestyle="--", alpha=0.5)
ax.axhline(best_mae,   color="black", lw=1.2, linestyle=":",  alpha=0.5)
ax.text(best_epoch + 0.5, ax.get_ylim()[0],
        f"epoch {best_epoch}", fontsize=8, va="bottom", color="black", alpha=0.7)
ax.set_xlabel("Epoch")
ax.set_ylabel("MAE")
ax.set_title(f"Learning Curve  (best val MAE = {best_mae:.5f})")
ax.legend()

# ── LR panel ──────────────────────────────────────────────────────────────
if has_lr:
    ax2 = axes[1]
    ax2.semilogy(epochs, lr, color=PURPLE, lw=2.0)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate (log scale)")
    ax2.set_title("Learning Rate Schedule")

fig.tight_layout()
out_path = OUT / "learning_curve.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
