# ALIGNN Raman Spectrum Prediction

> **RamanGPT paper data and models.** The corrected dataset and trained models behind *RamanGPT maps bidirectionally between crystal structures and Raman spectra of inorganic materials* (Scientific Reports, 2026) are in [`ramangpt_release/`](ramangpt_release/). The older data and runs in this repository were built from a database export that lost the mode labels of the Raman tensors for 1,210 materials, so those tensors were paired with the wrong phonon modes and the older results do not reproduce the published ones. See [`ramangpt_release/README.md`](ramangpt_release/README.md) for details.

Predicting Raman spectra of crystal structures using the **ALIGNN** (Atomistic Line Graph Neural Network) graph neural network. The model takes a crystal structure as input and outputs a normalized Raman spectrum as a multi-bin vector.

---

## Overview

- **Dataset:** 5,092 crystal structures from the Raman database (`ramandb.json`)
- **Target:** Normalized Raman spectrum binned over 50–1000 cm⁻¹
- **Model:** ALIGNN (rotationally invariant GNN), multi-output regression
- **Split:** 4,073 train / 509 val / 510 test

---

## Project Structure

```
raman_alignn/
├── 02_preprocess_crd.py               # Preprocess raw JSON → alignn_dataset/
├── 03_train_alignn.py                 # Local training script
├── 03_train_alignn_rw.py              # Redwing cluster training script
├── 04_evaluate.py                     # Evaluation on test set
├── 05_clean_plots.py                  # simple plots
├── plot_learning_curve.py             # Learning curve from checkpoint
├── plot_compare_cosine.py             # Compare cosine similarity across runs
│
├── alignn_raman_config_local.json          # Local config: 4+4 layers, 100 bins
├── alignn_raman_config_local_50bins.json   # Local config: 4+4 layers, 50 bins
├── alignn_raman_config_rw.json             # Redwing config: 6+6 layers, 100 bins
│
├── alignn_dataset/                    # 100-bin dataset
│   ├── id_prop.csv                    # JID → target spectrum
│   ├── train.txt / val.txt / test.txt # Dataset splits
│   └── structures/                    # VASP structure files (not tracked)
│
├── alignn_dataset/bins_50/            # 50-bin dataset
├── alignn_dataset/bins_500/           # 500-bin dataset
│
└── runs/                              # Training outputs
    └── {run_name}/
        ├── best_model.pt
        └── eval/
```

---

## Environment Setup

```bash
conda activate alignn_raman
cd ~/projects/raman_alignn
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

---

## Usage

### 1. Preprocess

```bash
python 02_preprocess_crd.py \
    --json   data/ramandb.json \
    --out    alignn_dataset \
    --n_bins 100 \
    --wn_min 50.0 \
    --wn_max 1000.0
```

### 2. Train (local)

```bash
python 03_train_alignn.py \
    --data_dir   alignn_dataset \
    --config     alignn_raman_config_local.json \
    --output_dir runs/full_local_001
```

### 3. Evaluate

```bash
python 04_evaluate.py \
    --data_dir   alignn_dataset \
    --model_path runs/full_local_001/best_model.pt \
    --output_dir runs/full_local_001/eval \
    --wn_min 50 --wn_max 1000
```

### 4. Plot

```bash
# Clean plots for a single run
python 05_clean_plots.py \
    --eval_dir runs/full_local_001/eval \
    --wn_min 50 --wn_max 1000 --n_bins 100

# Compare cosine similarity across runs
python plot_compare_cosine.py \
    --runs   runs/local_bins50_002/eval runs/local_bins500_001/eval \
    --labels "50 bins" "500 bins" \
    --output_dir runs/run_compare_plots
```

---

## Configs

| Config | Layers | Bins | Neighbors | Epochs | Notes |
|---|---|---|---|---|---|
| `alignn_raman_config_local.json` | 4+4 | 100 | 12 | 50 | Local GPU |
| `alignn_raman_config_local_50bins.json` | 4+4 | 50 | 12 | 50 | Local GPU |
| `alignn_raman_config_rw.json` | 6+6 | 100 | 20 | 600 | Redwing cluster |

---

## Key Results

| Run | Bins | Val MAE | Mean Cosine Sim |
|---|---|---|---|
| `full_local_001` | 100 | 0.03808 | 0.463 |

---

## Notes

- ALIGNN is rotationally **invariant**, suitable for averaged spectra
- `max_neighbors=12` is required for local GPU (RTX 4060 8.6GB) to avoid OOM
- `max_neighbors=20` used on Redwing for the paper architecture
- Loss: MSE during training; cosine similarity is the primary evaluation metric
- Wavenumber range: 50–1000 cm⁻¹
