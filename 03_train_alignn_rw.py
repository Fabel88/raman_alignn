"""
03_train_alignn_rw.py  — Redwing training script
Follows Gurunathan et al. 2022 architecture for spectrum prediction with ALIGNN.

Key changes from 03_train_alignn.py:
  - 6 ALIGNN + 6 GCN layers  (paper architecture)
  - max_neighbors 20          (paper: 20 nearest neighbors)
  - MSE + cosine combo loss   (our addition — fixes blurry predictions)
  - CosineAnnealingWarmRestarts LR schedule (prevents dead LR at epoch 80)
  - Gradient clipping
  - Mixed precision (torch.cuda.amp) — ~1.5x speedup
  - Frequency-weighted sampler — upsamples hard high-freq structures
  - Cosine similarity logged on val every epoch alongside MAE
  - Checkpoint every 50 epochs for cluster preemption recovery

Usage:
    python 03_train_alignn_rw.py \
        --data_dir  alignn_dataset \
        --config    alignn_raman_config_rw.json \
        --output_dir runs/redwing_001
"""

import os, json, time, math, argparse, logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

import pandas as pd
from tqdm import tqdm

# ── ALIGNN imports ────────────────────────────────────────────────────────────
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from alignn.graphs import Graph
from jarvis.core.atoms import Atoms

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="""
    03_train_alignn_rw.py  — Redwing ALIGNN Raman training script
    Architecture follows Gurunathan et al. 2022 (phonon DOS with ALIGNN).
    Adds cosine loss, warm-restart LR, mixed precision, and weighted sampling.
    Example:
        python 03_train_alignn_rw.py \\
            --data_dir  alignn_dataset \\
            --config    alignn_raman_config_rw.json \\
            --output_dir runs/redwing_001
""")
parser.add_argument("--data_dir",   type=str, required=True)
parser.add_argument("--config",     type=str, required=True)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--resume",     type=str, default=None,
                    help="Path to checkpoint to resume from")
parser.add_argument("--epochs",     type=int, default=None,
                    help="Override epochs from config")
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--lr",         type=float, default=None)
parser.add_argument("--n_bins",     type=int, default=100)
parser.add_argument("--wn_min",     type=float, default=50.0)
parser.add_argument("--wn_max",     type=float, default=1000.0)
parser.add_argument("--test_mode",  action="store_true",
                    help="Run 2 batches per epoch for quick smoke test")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Output directory
# ─────────────────────────────────────────────────────────────────────────────
if args.output_dir is None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = f"./runs/alignn_raman_rw_{stamp}"
OUT = Path(args.output_dir)
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUT / "training.log"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
with open(args.config) as f:
    cfg = json.load(f)

model_cfg  = cfg["model"]
train_cfg  = cfg["training"]
graph_cfg  = cfg["graph"]

EPOCHS      = args.epochs     or train_cfg["epochs"]
BATCH_SIZE  = args.batch_size or train_cfg["batch_size"]
LR          = args.lr         or train_cfg["learning_rate"]
CUTOFF      = graph_cfg["cutoff"]
MAX_NBRS    = graph_cfg["max_neighbors"]
GRAD_CLIP   = train_cfg.get("grad_clip_norm", 1.0)
USE_AMP     = train_cfg.get("mixed_precision", True) and torch.cuda.is_available()
COS_WEIGHT  = train_cfg.get("cosine_loss_weight", 0.2)
CKPT_EVERY  = train_cfg.get("checkpoint_every_n_epochs", 50)
USE_WSAMPLER= train_cfg.get("frequency_weighted_sampling", True)

log.info("=" * 60)
log.info("  ALIGNN Raman — Redwing Training Script")
log.info("  Architecture: Gurunathan et al. 2022")
log.info("=" * 60)
log.info(f"  Output dir    : {OUT}")
log.info(f"  Epochs        : {EPOCHS}")
log.info(f"  Batch size    : {BATCH_SIZE}")
log.info(f"  LR            : {LR}")
log.info(f"  max_neighbors : {MAX_NBRS}  (paper: 20)")
log.info(f"  ALIGNN layers : {model_cfg['alignn_layers']}  (paper: 6)")
log.info(f"  GCN layers    : {model_cfg['gcn_layers']}     (paper: 6)")
log.info(f"  Cosine weight : {COS_WEIGHT}")
log.info(f"  Mixed prec.   : {USE_AMP}")
log.info(f"  Grad clip     : {GRAD_CLIP}")
log.info(f"  Freq. sampling: {USE_WSAMPLER}")
log.info("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
WAVENUMS = np.linspace(args.wn_min, args.wn_max, args.n_bins)

class RamanSpectrumDataset(torch.utils.data.Dataset):
    """
    Loads crystal structures from VASP files and Raman spectra from id_prop.csv.

    Each sample returns:
        graph      : DGL atom graph
        line_graph : DGL bond line graph
        lattice    : FloatTensor (3, 3) — unit cell lattice matrix
        target     : FloatTensor (n_bins,) — normalised Raman spectrum
    """
    def __init__(self, data_dir, split_file, n_bins=100,
                 cutoff=8.0, max_neighbors=20, test_mode=False):
        import csv
        data_dir   = Path(data_dir)
        self.structures_dir = data_dir / "structures"
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.n_bins = n_bins
        self.test_mode = test_mode

        # Load split IDs
        ids = (data_dir / split_file).read_text().strip().splitlines()
        ids = [i.strip() for i in ids if i.strip()]

        # Load targets from id_prop.csv
        id_to_target = {}
        with open(data_dir / "id_prop.csv") as f:
            for row in csv.DictReader(f):
                jid = row["jid"].strip()
                arr = np.array(json.loads(row["target"]), dtype=np.float32)
                id_to_target[jid] = arr

        # Build record list
        self.records = []
        skipped = 0
        for jid in ids:
            vasp = self.structures_dir / f"{jid}.vasp"
            if not vasp.exists() or jid not in id_to_target:
                skipped += 1
                continue
            self.records.append((jid, str(vasp), id_to_target[jid]))
        if skipped:
            log.warning(f"  [{split_file}] Skipped {skipped} missing entries")

        # Pre-compute target tensors
        self.targets = [
            torch.tensor(r[2], dtype=torch.float32)
            for r in self.records
        ]

        log.info(f"  [{split_file}] {len(self.records)} structures loaded")

    def __len__(self):
        return min(len(self.records), 2 * self.max_neighbors) \
               if self.test_mode else len(self.records)

    def __getitem__(self, idx):
        jid, vasp_path, _ = self.records[idx]
        target = self.targets[idx]
        atoms  = Atoms.from_poscar(vasp_path)
        graph, line_graph = Graph.atom_dgl_multigraph(
            atoms,
            cutoff=self.cutoff,
            atom_features="cgcnn",
            max_neighbors=self.max_neighbors,
            use_canonize=True,
        )
        lattice = torch.tensor(atoms.lattice_mat, dtype=torch.float32)
        return graph, line_graph, lattice, target


def collate_fn(batch):
    import dgl
    graphs, line_graphs, lattices, targets = zip(*batch)
    return (
        dgl.batch(graphs),
        dgl.batch(line_graphs),
        torch.stack(lattices),   # (batch, 3, 3)
        torch.stack(targets),    # (batch, n_bins)
    )


def make_frequency_weighted_sampler(dataset, wavenums):
    """
    Upsample structures whose dominant peak is above 400 cm⁻¹ (the hard cases).
    Weight = 2.0 for peak > 400 cm⁻¹, 1.0 otherwise.
    """
    threshold_bin = np.searchsorted(wavenums, 400.0)
    weights = []
    for tgt in dataset.targets:
        peak_bin = tgt.argmax().item()
        weights.append(2.0 if peak_bin >= threshold_bin else 1.0)
    weights = torch.tensor(weights, dtype=torch.float64)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def combo_loss(pred, target, cosine_weight=0.2):
    """
    MSE loss + cosine similarity loss.
    MSE: penalises pointwise magnitude errors
    Cosine: penalises wrong spectral shape — key fix for blurry predictions
    """
    mse = F.mse_loss(pred, target)
    cos = (1.0 - F.cosine_similarity(pred, target, dim=1)).mean()
    return mse + cosine_weight * cos, mse.item(), cos.item()


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

model_config = ALIGNNConfig(
    name             = model_cfg["name"],
    alignn_layers    = model_cfg["alignn_layers"],
    gcn_layers       = model_cfg["gcn_layers"],
    atom_input_features   = model_cfg.get("atom_input_features", 92),
    edge_input_features   = model_cfg.get("edge_input_features", 80),
    triplet_input_features= model_cfg.get("triplet_input_features", 40),
    embedding_features    = model_cfg.get("embedding_features", 64),
    hidden_features  = model_cfg["hidden_features"],
    output_features  = model_cfg["output_features"],
    link             = model_cfg.get("link", "identity"),
    classification   = False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = ALIGNN(model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
log.info(f"Model: ALIGNNConfig(alignn_layers={model_config.alignn_layers}, "
         f"gcn_layers={model_config.gcn_layers}, "
         f"hidden={model_config.hidden_features}, "
         f"output={model_config.output_features})")
log.info(f"Parameters: {n_params:,}")
log.info(f"Device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────
train_dataset = RamanSpectrumDataset(
    args.data_dir, "train.txt", n_bins=args.n_bins,
    cutoff=CUTOFF, max_neighbors=MAX_NBRS, test_mode=args.test_mode,
)
val_dataset = RamanSpectrumDataset(
    args.data_dir, "val.txt", n_bins=args.n_bins,
    cutoff=CUTOFF, max_neighbors=MAX_NBRS, test_mode=args.test_mode,
)

if USE_WSAMPLER:
    sampler = make_frequency_weighted_sampler(train_dataset, WAVENUMS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler, collate_fn=collate_fn,
                              num_workers=4, pin_memory=True)
    log.info("Using frequency-weighted sampler (2x weight for peak > 400 cm⁻¹)")
else:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=4, pin_memory=True)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn,
                        num_workers=4, pin_memory=True)

log.info(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

# ─────────────────────────────────────────────────────────────────────────────
# Optimiser & scheduler
# ─────────────────────────────────────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR,
                  weight_decay=train_cfg.get("weight_decay", 1e-5))

# CosineAnnealingWarmRestarts: restarts at epoch 100, 300, 700, ...
# This prevents the LR dying at epoch 80 as happened in the 50-epoch run
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0    = train_cfg.get("scheduler_T0", 100),
    T_mult = train_cfg.get("scheduler_T_mult", 2),
    eta_min= train_cfg.get("scheduler_eta_min", 1e-7),
)

scaler = GradScaler(enabled=USE_AMP)

# ─────────────────────────────────────────────────────────────────────────────
# Resume from checkpoint
# ─────────────────────────────────────────────────────────────────────────────
start_epoch   = 0
best_val_mae  = float("inf")
best_val_cos  = -float("inf")
history       = []

if args.resume:
    log.info(f"Resuming from {args.resume}")
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    if "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    start_epoch  = ckpt["epoch"]
    best_val_mae = ckpt.get("best_val_mae", float("inf"))
    best_val_cos = ckpt.get("best_val_cos", -float("inf"))
    history      = ckpt.get("history", [])
    log.info(f"  Resumed at epoch {start_epoch}  "
             f"best_val_mae={best_val_mae:.5f}  best_val_cos={best_val_cos:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helper
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(epoch, val_mae, val_cos, is_best=False):
    ckpt = {
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state":    scaler.state_dict(),
        "best_val_mae":    best_val_mae,
        "best_val_cos":    best_val_cos,
        "history":         history,
        "model_config":    model_config.model_dump(),
    }
    # Periodic checkpoint every CKPT_EVERY epochs
    if epoch % CKPT_EVERY == 0:
        path = OUT / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(ckpt, path)
        log.info(f"  Periodic checkpoint saved: {path}")
    # Best model
    if is_best:
        torch.save(ckpt, OUT / "best_model.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Cosine similarity metric (vectorised, no scipy dependency)
# ─────────────────────────────────────────────────────────────────────────────

def batch_cosine_sim(pred, target):
    """Mean cosine similarity over a batch. Both (N, D) tensors."""
    num = (pred * target).sum(dim=1)
    den = pred.norm(dim=1) * target.norm(dim=1) + 1e-8
    return (num / den).mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = total_mae = total_cos_sim = n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, (graph, line_graph, lattice, target) in enumerate(loader):
            if args.test_mode and i >= 2:
                break

            graph      = graph.to(device)
            line_graph = line_graph.to(device)
            lattice    = lattice.to(device)
            target     = target.to(device)

            with autocast(enabled=USE_AMP):
                pred = model([graph, line_graph, lattice])
                loss, mse_val, cos_val = combo_loss(pred, target, COS_WEIGHT)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()

            bs = target.size(0)
            total_loss    += loss.item() * bs
            total_mae     += F.l1_loss(pred, target).item() * bs
            total_cos_sim += batch_cosine_sim(pred.detach(), target) * bs
            n             += bs

    return total_loss / n, total_mae / n, total_cos_sim / n


log.info(f"Starting training from epoch {start_epoch + 1}/{EPOCHS}")
log.info(f"{'Epoch':>6}  {'train_loss':>10}  {'train_mae':>9}  "
         f"{'val_mae':>8}  {'val_cos':>8}  {'lr':>10}  {'flag':>6}")

for epoch in range(start_epoch, EPOCHS):
    t0 = time.time()

    train_loss, train_mae, train_cos = run_epoch(train_loader, train=True)
    scheduler.step()
    val_loss,   val_mae,   val_cos   = run_epoch(val_loader,   train=False)

    lr_now = optimizer.param_groups[0]["lr"]
    is_best_mae = val_mae  < best_val_mae
    is_best_cos = val_cos  > best_val_cos
    is_best     = is_best_mae  # primary criterion

    if is_best_mae:
        best_val_mae = val_mae
    if is_best_cos:
        best_val_cos = val_cos

    flag = "★" if is_best else ""

    log.info(
        f"Epoch {epoch+1:>4}/{EPOCHS}  "
        f"train_loss={train_loss:.5f}  train_mae={train_mae:.5f}  "
        f"val_mae={val_mae:.5f}  val_cos={val_cos:.4f}  "
        f"lr={lr_now:.2e}  {flag}"
    )
    if is_best:
        log.info(f"  ★ New best  val_mae={best_val_mae:.5f}  val_cos={val_cos:.4f}")

    # Save history
    history.append({
        "epoch":      epoch + 1,
        "train_loss": train_loss,
        "train_mae":  train_mae,
        "val_loss":   val_loss,
        "val_mae":    val_mae,
        "val_cos":    val_cos,
        "lr":         lr_now,
    })

    save_checkpoint(epoch + 1, val_mae, val_cos, is_best=is_best)

# ─────────────────────────────────────────────────────────────────────────────
# Save training history CSV
# ─────────────────────────────────────────────────────────────────────────────
history_df = pd.DataFrame(history)
history_df.to_csv(OUT / "training_history.csv", index=False)
log.info(f"Training history saved to {OUT / 'training_history.csv'}")

# ─────────────────────────────────────────────────────────────────────────────
# Training curves plot
# ─────────────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history_df["epoch"], history_df["train_mae"],  label="Train MAE")
    axes[0].plot(history_df["epoch"], history_df["val_mae"],    label="Val MAE")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MAE")
    axes[0].set_title("MAE"); axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["val_cos"], color="green", label="Val Cosine Sim")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_title("Spectral Shape (Cosine Sim)"); axes[1].legend()

    axes[2].semilogy(history_df["epoch"], history_df["lr"], color="purple")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("LR (log scale)")
    axes[2].set_title("Learning Rate Schedule")

    fig.suptitle(f"ALIGNN Raman — Redwing Training  (best val MAE={best_val_mae:.5f})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=150)
    plt.close(fig)
    log.info(f"Training curves saved to {OUT / 'training_curves.png'}")
except Exception as e:
    log.warning(f"Could not save training curves: {e}")

log.info("")
log.info(f"Done. Best val MAE : {best_val_mae:.5f}")
log.info(f"      Best val cos : {best_val_cos:.4f}")
log.info(f"Best model         : {OUT / 'best_model.pt'}")
