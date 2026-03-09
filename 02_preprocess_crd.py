"""
02_preprocess_crd.py  (v2 — compute activity from raman_tensor)
===============================================================
Convert ramandb.json into ALIGNN training format.

Key fix over v1:
  - Compute raman_activity from raman_tensor for ALL 5099 entries
  - Use tensor index to align with freq_cm (handles length mismatches)
  - Null tensor entries = Raman-inactive modes → skipped
  - Powder-averaged activity: I = 45a² + 7b²

ramandb.json structure:
  {
    "id":             "mp-3252",
    "atoms":          { lattice_mat, coords, elements, cartesian, ... },
    "freq_cm":        [0.0006, 21.04, 82.9, ...],   # ALL 3N phonon modes
    "raman_activity": [0.0, 0.002, ...],             # may be shorter than freq_cm
    "raman_tensor":   [null, null, [[...]], ...]     # 3x3 per mode, null=inactive
  }

Usage:
    python 02_preprocess_crd.py \
        --json  ~/projects/raman_alignn/data/ramandb.json \
        --out   ~/projects/raman_alignn/alignn_dataset
"""

import json
import argparse
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from jarvis.core.atoms import Atoms
from jarvis.io.vasp.inputs import Poscar

try:
    from jarvis.analysis.structure.spacegroup import Spacegroup3D
    HAS_SPG = True
except ImportError:
    HAS_SPG = False

# =============================================================================
# CLI
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--json",       type=str,   required=True)
parser.add_argument("--out",        type=str,   default="./alignn_dataset")
parser.add_argument("--n_bins",     type=int,   default=100)
parser.add_argument("--wn_min",     type=float, default=50.0)
parser.add_argument("--wn_max",     type=float, default=1000.0)
parser.add_argument("--fwhm",       type=float, default=10.0,
                    help="Lorentzian FWHM in cm⁻¹")
parser.add_argument("--train_frac", type=float, default=0.8)
parser.add_argument("--val_frac",   type=float, default=0.1)
args = parser.parse_args()

OUT        = Path(args.out)
STRUCT_DIR = OUT / "structures"
OUT.mkdir(parents=True, exist_ok=True)
STRUCT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_WN = np.linspace(args.wn_min, args.wn_max, args.n_bins)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(OUT / "preprocessing_log.txt"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)
log.info(f"Input:   {args.json}")
log.info(f"Output:  {OUT}")
log.info(f"Bins:    {args.n_bins}  ({args.wn_min:.0f}–{args.wn_max:.0f} cm⁻¹)")
log.info(f"FWHM:    {args.fwhm} cm⁻¹")


# =============================================================================
# averaged Raman activity from 3x3 tensor
# =============================================================================
def tensor_to_activity(tensor_3x3):
    """
    Compute orientation-averaged Raman scattering activity
    from the Raman tensor α (3×3 matrix).

    Formula: I = 45a² + 7b²
      a  = Tr(α) / 3                        (isotropic invariant)
      b² = ½[(α₁₁-α₂₂)² + (α₂₂-α₃₃)² + (α₃₃-α₁₁)²]
           + 3(α₁₂² + α₁₃² + α₂₃²)         (anisotropy — symmetrised)

    Uses symmetrised tensor: α_sym = (α + αᵀ) / 2
    """
    t  = np.array(tensor_3x3, dtype=float)
    ts = (t + t.T) / 2.0  # symmetrise

    a  = np.trace(ts) / 3.0

    b2 = 0.5 * (
        (ts[0,0] - ts[1,1])**2 +
        (ts[1,1] - ts[2,2])**2 +
        (ts[2,2] - ts[0,0])**2
    ) + 3.0 * (ts[0,1]**2 + ts[0,2]**2 + ts[1,2]**2)

    return 45.0 * a**2 + 7.0 * b2


# =============================================================================
# Extract (freq, activity) pairs from one entry using tensor
# =============================================================================
def extract_modes(entry):
    """
    Returns list of (freq_cm, activity) for Raman-active modes only.

    Strategy:
      - Iterate over raman_tensor by index
      - Skip null entries (Raman-inactive modes)
      - Use freq_cm[i] for the frequency of tensor[i]
      - If len(tensor) > len(freq): truncate tensor to len(freq)
    """
    freq_cm = entry.get("freq_cm", [])
    tensors = entry.get("raman_tensor", [])

    if not freq_cm or not tensors:
        return []

    n = min(len(freq_cm), len(tensors))
    modes = []

    for i in range(n):
        t = tensors[i]
        if t is None:
            continue  # Raman-inactive mode
        f = freq_cm[i]
        if f < 5.0:
            continue  # acoustic mode
        try:
            act = tensor_to_activity(t)
        except Exception:
            continue
        if act < 1e-12:
            continue
        modes.append((f, act))

    return modes


# =============================================================================
# Discretize modes onto fixed wavenumber grid
# =============================================================================
def discretize_modes(modes, target_wn=TARGET_WN, fwhm=args.fwhm):
    """
    Place Lorentzian peaks at each active mode frequency,
    sum into a continuous spectrum, normalise to [0, 1].
    """
    if not modes:
        return None

    gamma    = fwhm / 2.0
    spectrum = np.zeros(len(target_wn))

    for freq, act in modes:
        lorentz   = (gamma**2) / ((target_wn - freq)**2 + gamma**2)
        spectrum += act * lorentz

    max_val = spectrum.max()
    if max_val < 1e-10:
        return None

    return (spectrum / max_val).astype(np.float32)


# =============================================================================
# Main loop
# =============================================================================
log.info(f"Loading {args.json} ...")
with open(args.json) as f:
    data = json.load(f)
log.info(f"Loaded {len(data)} entries")

rows_idprop = []
rows_meta   = []
skipped     = []

for entry in tqdm(data, desc="Processing"):
    eid = entry.get("id", "unknown")

    # ── Compute spectrum from tensor ──────────────────────────────────────────
    modes = extract_modes(entry)
    if not modes:
        skipped.append((eid, "no_active_modes"))
        continue

    spectrum = discretize_modes(modes)
    if spectrum is None:
        skipped.append((eid, "zero_spectrum"))
        continue

    # Filter to entries that have modes inside the target window
    in_window = [(f, a) for f, a in modes if args.wn_min <= f <= args.wn_max]
    if not in_window:
        skipped.append((eid, "no_modes_in_window"))
        continue

    # ── Structure ─────────────────────────────────────────────────────────────
    atoms_dict = entry.get("atoms")
    if atoms_dict is None:
        skipped.append((eid, "no_atoms"))
        continue

    try:
        atoms = Atoms(
            lattice_mat=atoms_dict["lattice_mat"],
            elements=atoms_dict["elements"],
            coords=atoms_dict["coords"],
            cartesian=atoms_dict.get("cartesian", True),
        )
    except Exception as e:
        skipped.append((eid, f"atoms_error: {e}"))
        continue

    if not (2 <= len(atoms.elements) <= 200):
        skipped.append((eid, "atoms_out_of_range"))
        continue

    # ── Space group ───────────────────────────────────────────────────────────
    spacegroup     = None
    crystal_system = None
    if HAS_SPG:
        try:
            spg            = Spacegroup3D(atoms)
            spacegroup     = spg.space_group_symbol
            crystal_system = spg.crystal_system
        except Exception:
            pass

    # ── Write POSCAR ──────────────────────────────────────────────────────────
    try:
        Poscar(atoms).write_file(str(STRUCT_DIR / f"{eid}.vasp"))
    except Exception as e:
        skipped.append((eid, f"poscar_error: {e}"))
        continue

    # ── Collect ───────────────────────────────────────────────────────────────
    rows_idprop.append({
        "jid":    eid,
        "target": json.dumps(spectrum.tolist()),
    })

    rows_meta.append({
        "jid":             eid,
        "formula":         atoms.composition.reduced_formula,
        "n_atoms":         len(atoms.elements),
        "spacegroup":      spacegroup,
        "crystal_system":  crystal_system,
        "n_active_modes":  len(modes),
        "n_in_window":     len(in_window),
        "freq_min":        min(f for f, _ in modes),
        "freq_max":        max(f for f, _ in modes),
    })

# =============================================================================
# Write outputs
# =============================================================================
log.info(f"\nProcessed: {len(rows_idprop)}  |  Skipped: {len(skipped)}")

df_idprop = pd.DataFrame(rows_idprop)
df_idprop.to_csv(OUT / "id_prop.csv", index=False)
log.info(f"Wrote id_prop.csv  ({len(df_idprop)} entries)")

df_meta = pd.DataFrame(rows_meta)
df_meta.to_csv(OUT / "metadata.csv", index=False)
log.info("Wrote metadata.csv")

if skipped:
    df_skip = pd.DataFrame(skipped, columns=["id", "reason"])
    df_skip.to_csv(OUT / "skipped.csv", index=False)
    log.info("Skip reasons:")
    for reason, count in df_skip.reason.value_counts().items():
        log.info(f"  {reason:<40} {count}")

# =============================================================================
# Train / val / test split
# =============================================================================
all_ids  = df_idprop["jid"].tolist()

stratify = None
if "crystal_system" in df_meta.columns and df_meta["crystal_system"].notna().any():
    cs_map  = df_meta.set_index("jid")["crystal_system"].to_dict()
    labels  = [cs_map.get(i, "unknown") for i in all_ids]
    if min(Counter(labels).values()) >= 2:
        stratify = labels
        log.info("Stratifying split by crystal system")

test_frac = round(1.0 - args.train_frac - args.val_frac, 6)

train_ids, temp_ids = train_test_split(
    all_ids,
    test_size=round(1.0 - args.train_frac, 6),
    random_state=42,
    stratify=stratify,
)
stratify_temp = None
if stratify:
    cs_map        = df_meta.set_index("jid")["crystal_system"].to_dict()
    temp_labels   = [cs_map.get(i, "unknown") for i in temp_ids]
    if min(Counter(temp_labels).values()) >= 2:
        stratify_temp = temp_labels

val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=round(test_frac / (test_frac + args.val_frac), 6),
    random_state=42,
    stratify=stratify_temp,
)

for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
    (OUT / f"{name}.txt").write_text("\n".join(ids))
    log.info(f"  {name:>6}: {len(ids):>5,} entries")

# =============================================================================
# Summary
# =============================================================================
log.info(f"""
{'='*55}
DONE
  id_prop.csv   : {len(df_idprop)} entries
  structures/   : {len(list(STRUCT_DIR.glob('*.vasp')))} POSCAR files
  train         : {len(train_ids)}
  val           : {len(val_ids)}
  test          : {len(test_ids)}

Spectrum config:
  n_bins        = {args.n_bins}
  wn_min        = {args.wn_min} cm⁻¹
  wn_max        = {args.wn_max} cm⁻¹
  lineshape     = Lorentzian  FWHM={args.fwhm} cm⁻¹
  activity      = computed from raman_tensor  (I = 45a² + 7b²)

Next:
  python 03_train_alignn.py \\
      --data_dir {OUT} \\
      --config   alignn_raman_config_local.json \\
      --test_mode
{'='*55}
""")
