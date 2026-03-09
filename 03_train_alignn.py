"""
03_train_alignn.py  (updated — lattice matrix fix)
===================================================
Train ALIGNN to predict Raman spectra from crystal structure.

Key fixes confirmed from smoke test:
  - Use ALIGNNConfig directly (not TrainingConfig().model)
  - model name must be 'alignn' not 'alignn_atomwise'
  - Graph builder returns (crystal_graph, line_graph) — 2 DGL graphs
  - Lattice matrix passed as third element: model([g, lg, lattice])
  - collate_fn stacks lattice tensors separately from DGL graphs

Usage:
    # Sanity check first (50 samples, 5 epochs)
    python 03_train_alignn.py --data_dir ./alignn_dataset --test_mode

    # Full local run (RTX 4060, batch_size=32)
    python 03_train_alignn.py \
        --data_dir   ./alignn_dataset \
        --config     ./alignn_raman_config_local.json \
        --output_dir ./runs/run_001

    # Resume from checkpoint
    python 03_train_alignn.py \
        --data_dir  ./alignn_dataset \
        --resume    ./runs/run_001/best_model.pt
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# CLI
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",   type=str, required=True)
parser.add_argument("--config",     type=str, default="./alignn_raman_config_local.json")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--epochs",     type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--lr",         type=float, default=None)
parser.add_argument("--resume",     type=str, default=None)
parser.add_argument("--test_mode",  action="store_true",
                    help="50 samples, 5 epochs — pipeline check only")
parser.add_argument("--n_bins",     type=int, default=100)
args = parser.parse_args()

# =============================================================================
# Output directory + logging
# =============================================================================
if args.output_dir is None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = f"./runs/alignn_raman_{stamp}"

OUT = Path(args.output_dir)
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(OUT / "training.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)
log.info(f"Output dir:  {OUT}")
log.info(f"CUDA:        {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log.info(f"GPU:         {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM:        {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# =============================================================================
# Load config
# =============================================================================
from alignn.config import ALIGNNConfig

# Defaults
model_cfg = dict(
    name="alignn",
    alignn_layers=6,
    gcn_layers=6,
    hidden_features=256,
    output_features=args.n_bins,
    link="identity",
    classification=False,
    atom_input_features=92,
    edge_input_features=80,
    triplet_input_features=40,
)
train_cfg = dict(
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    weight_decay=1e-5,
    num_workers=2,
    save_interval=10,
)

config_path = Path(args.config)
if config_path.exists():
    with open(config_path) as f:
        raw = json.load(f)
    def strip_comments(d):
        return {k: (strip_comments(v) if isinstance(v, dict) else v)
                for k, v in d.items() if not k.startswith("__")}
    raw = strip_comments(raw)
    model_cfg.update(raw.get("model", {}))
    train_cfg.update(raw.get("training", {}))
    log.info(f"Loaded config: {config_path}")
else:
    log.warning(f"Config not found at {config_path} — using defaults")

# CLI overrides
if args.epochs:     train_cfg["epochs"]       = args.epochs
if args.batch_size: train_cfg["batch_size"]   = args.batch_size
if args.lr:         train_cfg["learning_rate"]= args.lr
if args.test_mode:
    train_cfg["epochs"]     = 5
    train_cfg["batch_size"] = 8
    log.info("TEST MODE — 5 epochs, batch_size=8, 50 samples")

# Only pass fields that ALIGNNConfig actually accepts
valid_fields = ALIGNNConfig.model_fields.keys()
model_config = ALIGNNConfig(**{k: v for k, v in model_cfg.items() if k in valid_fields})

log.info(f"Model:  alignn_layers={model_config.alignn_layers}  "
         f"gcn_layers={model_config.gcn_layers}  "
         f"hidden={model_config.hidden_features}  "
         f"output={model_config.output_features}")
log.info(f"Train:  epochs={train_cfg['epochs']}  "
         f"batch={train_cfg['batch_size']}  "
         f"lr={train_cfg['learning_rate']}")

# =============================================================================
# Dataset
# =============================================================================
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from jarvis.io.vasp.inputs import Poscar
from alignn.graphs import Graph


class RamanDataset(Dataset):
    """
    Each __getitem__ returns:
        graph      : DGL crystal graph
        line_graph : DGL line graph (bond angles)
        lattice    : FloatTensor (3,3) — unit cell lattice matrix
        target     : FloatTensor (n_bins,) — normalised Raman spectrum
    """

    def __init__(self, data_dir, split_file, n_bins=100,
                 cutoff=8.0, max_neighbors=12, test_mode=False):
        self.data_dir      = Path(data_dir)
        self.struct_dir    = self.data_dir / "structures"
        self.n_bins        = n_bins
        self.cutoff        = cutoff
        self.max_neighbors = max_neighbors

        split_ids = set(
            (self.data_dir / split_file).read_text().strip().split("\n")
        )
        df = pd.read_csv(self.data_dir / "id_prop.csv")
        df = df[df["jid"].isin(split_ids)].reset_index(drop=True)
        if test_mode:
            df = df.head(50)

        self.jids    = df["jid"].tolist()
        self.targets = [
            torch.tensor(json.loads(t), dtype=torch.float32)
            for t in df["target"]
        ]
        log.info(f"  [{split_file}] {len(self.jids)} entries")

    def __len__(self):
        return len(self.jids)

    def __getitem__(self, idx):
        jid    = self.jids[idx]
        target = self.targets[idx]

        atoms = Poscar.from_file(
            str(self.struct_dir / f"{jid}.vasp")
        ).atoms

        graph, line_graph = Graph.atom_dgl_multigraph(
            atoms,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            atom_features="cgcnn",
        )

        # Lattice matrix — passed as third element to model
        lattice = torch.tensor(atoms.lattice_mat, dtype=torch.float32)

        return graph, line_graph, lattice, target


def collate_fn(batch):
    import dgl
    graphs, line_graphs, lattices, targets = zip(*batch)
    return (
        dgl.batch(graphs),
        dgl.batch(line_graphs),
        torch.stack(lattices),    # (batch, 3, 3)
        torch.stack(targets),     # (batch, n_bins)
    )


# =============================================================================
# Loaders
# =============================================================================
data_dir = Path(args.data_dir)

train_set = RamanDataset(data_dir, "train.txt",
                         n_bins=args.n_bins, test_mode=args.test_mode)
val_set   = RamanDataset(data_dir, "val.txt",
                         n_bins=args.n_bins, test_mode=args.test_mode)

train_loader = DataLoader(train_set, batch_size=train_cfg["batch_size"],
                          shuffle=True,  collate_fn=collate_fn,
                          num_workers=train_cfg["num_workers"],
                          pin_memory=torch.cuda.is_available())
val_loader   = DataLoader(val_set,   batch_size=train_cfg["batch_size"],
                          shuffle=False, collate_fn=collate_fn,
                          num_workers=train_cfg["num_workers"],
                          pin_memory=torch.cuda.is_available())

log.info(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

# =============================================================================
# Model
# =============================================================================
from alignn.models.alignn import ALIGNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = ALIGNN(model_config).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"Parameters: {n_params:,}")

# =============================================================================
# Loss / optimiser / scheduler
# =============================================================================
criterion = torch.nn.L1Loss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_cfg["learning_rate"],
    weight_decay=train_cfg["weight_decay"],
)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=train_cfg["learning_rate"],
    steps_per_epoch=len(train_loader),
    epochs=train_cfg["epochs"],
    pct_start=0.05,
)

# =============================================================================
# Resume
# =============================================================================
start_epoch  = 0
best_val_mae = float("inf")
history      = {"train_mae": [], "val_mae": [], "lr": []}

if args.resume:
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch  = ckpt["epoch"] + 1
    best_val_mae = ckpt.get("best_val_mae", float("inf"))
    history      = ckpt.get("history", history)
    log.info(f"Resumed from epoch {start_epoch}")

# =============================================================================
# Train / eval
# =============================================================================
def run_epoch(loader, train: bool):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for graph, line_graph, lattice, target in loader:
            graph      = graph.to(device)
            line_graph = line_graph.to(device)
            lattice    = lattice.to(device)
            target     = target.to(device)

            pred = model([graph, line_graph, lattice])
            loss = criterion(pred, target)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item() * len(target)
            n          += len(target)

    return total_loss / n


def save_checkpoint(epoch, val_mae, is_best=False):
    ckpt = {
        "epoch": epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_mae":    best_val_mae,
        "history":         history,
        "model_config":    model_config.model_dump(),
    }
    torch.save(ckpt, OUT / f"checkpoint_epoch_{epoch:04d}.pt")
    if is_best:
        torch.save(ckpt, OUT / "best_model.pt")
        log.info(f"  ★ New best  val_mae={val_mae:.5f}")


def plot_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    e = range(1, len(history["train_mae"]) + 1)
    axes[0].plot(e, history["train_mae"], label="train", color="#00d4ff")
    axes[0].plot(e, history["val_mae"],   label="val",   color="#f59e0b")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MAE")
    axes[0].set_title("Spectrum MAE"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(e, history["lr"], color="#8b5cf6")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("LR")
    axes[1].set_title("Learning Rate"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "training_curves.png", dpi=150)
    plt.close()

# =============================================================================
# Main loop
# =============================================================================
log.info(f"\n{'='*55}\nStarting — {train_cfg['epochs']} epochs\n{'='*55}")

for epoch in range(start_epoch, train_cfg["epochs"]):
    train_mae = run_epoch(train_loader, train=True)
    val_mae   = run_epoch(val_loader,   train=False)
    lr        = scheduler.get_last_lr()[0]

    history["train_mae"].append(train_mae)
    history["val_mae"].append(val_mae)
    history["lr"].append(lr)

    is_best = val_mae < best_val_mae
    if is_best:
        best_val_mae = val_mae

    log.info(
        f"Epoch {epoch+1:>4}/{train_cfg['epochs']}  "
        f"train={train_mae:.5f}  val={val_mae:.5f}  "
        f"lr={lr:.2e}" + (" ★" if is_best else "")
    )

    if (epoch + 1) % train_cfg.get("save_interval", 10) == 0 or is_best:
        save_checkpoint(epoch + 1, val_mae, is_best)
    if (epoch + 1) % 10 == 0:
        plot_curves()

plot_curves()
with open(OUT / "history.json", "w") as f:
    json.dump(history, f, indent=2)

log.info(f"\nDone. Best val MAE: {best_val_mae:.5f}")
log.info(f"Best model: {OUT / 'best_model.pt'}")
