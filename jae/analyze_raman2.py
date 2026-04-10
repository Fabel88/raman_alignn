"""
RamanGPT Inverse Model Analysis (JARVIS-native)
================================================
Uses jarvis.core.atoms.Atoms for proper structure handling:
  - Lattice parameters, angles, volume from Lattice object
  - Composition, formula, density from Atoms
  - Spacegroup from jarvis.analysis.structure.spacegroup

Usage:
  python analyze_raman_jarvis.py raman_res.json output_prefix
"""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import mean_absolute_error
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from jarvis.core.atoms import Atoms
from jarvis.core.composition import Composition

# ─── Style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

C_TARGET = '#5B8DBE'
C_PRED = '#D4726A'
C_ACCENT = '#6EAE6E'


# ─── Helpers ─────────────────────────────────────────────────────────

def dict_to_atoms(d):
    """Convert a target/pred dict to jarvis Atoms object."""
    return Atoms.from_dict(d)


def safe_atoms(d):
    """Safely build Atoms, return None on failure."""
    try:
        return dict_to_atoms(d)
    except Exception:
        return None


def kld_hist(tv, pv, n_bins=50):
    """KL divergence between two value arrays via histograms."""
    vals = np.concatenate([tv, pv])
    bins = np.linspace(vals.min(), vals.max(), n_bins + 1)
    p, _ = np.histogram(tv, bins=bins, density=True)
    q, _ = np.histogram(pv, bins=bins, density=True)
    p = p + 1e-10; q = q + 1e-10
    p /= p.sum(); q /= q.sum()
    return entropy(p, q)


def rmse_coords(tc, pc):
    """RMSE between fractional coords with periodic wrapping + greedy assignment."""
    tc = np.array(tc) % 1.0
    pc = np.array(pc) % 1.0
    if len(tc) != len(pc) or len(tc) == 0:
        return np.nan
    diffs = tc[:, None, :] - pc[None, :, :]
    diffs = diffs - np.round(diffs)
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    used = set()
    total_sq = 0.0
    for i in range(len(tc)):
        row = dists[i].copy()
        for j in used:
            row[j] = 1e10
        best_j = np.argmin(row)
        used.add(best_j)
        total_sq += row[best_j]**2
    return np.sqrt(total_sq / len(tc))


def element_accuracy(t_elems, p_elems):
    """Composition overlap fraction."""
    tc = Counter(t_elems)
    pc = Counter(p_elems)
    if tc == pc:
        return 1.0
    all_e = set(tc.keys()) | set(pc.keys())
    match = sum(min(tc.get(e, 0), pc.get(e, 0)) for e in all_e)
    total = max(sum(tc.values()), sum(pc.values()))
    return match / total if total > 0 else 0.0


def get_spacegroup_safe(atoms):
    """Get spacegroup number, return None on failure."""
    try:
        return atoms.get_spacegroup[0]
    except Exception:
        return None


# ─── Main ────────────────────────────────────────────────────────────

def main(json_path='raman_res.json', prefix='ramangpt'):

    with open(json_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} structures from {json_path}")

    records = []
    skipped = 0
    for entry in data:
        mid = entry['id']
        t_atoms = safe_atoms(entry['target'])
        p_atoms = safe_atoms(entry['pred'])
        if t_atoms is None or p_atoms is None:
            skipped += 1
            continue

        # ── Lattice parameters via jarvis Lattice ──
        try:
            t_abc = t_atoms.lattice.lat_lengths()
            t_ang = t_atoms.lattice.lat_angles()
            p_abc = p_atoms.lattice.lat_lengths()
            p_ang = p_atoms.lattice.lat_angles()
        except Exception:
            skipped += 1
            continue

        # ── Volume ──
        t_vol = t_atoms.volume
        p_vol = p_atoms.volume

        # ── Density (guard against zero volume) ──
        try:
            t_den = t_atoms.density if t_vol > 1e-6 else np.nan
        except Exception:
            t_den = np.nan
        try:
            p_den = p_atoms.density if p_vol > 1e-6 else np.nan
        except Exception:
            p_den = np.nan

        # ── Composition ──
        t_formula = t_atoms.composition.reduced_formula
        p_formula = p_atoms.composition.reduced_formula
        t_weight = t_atoms.composition.weight
        p_weight = p_atoms.composition.weight

        # ── Coordinate RMSE ──
        rmse = rmse_coords(t_atoms.frac_coords, p_atoms.frac_coords)

        # ── Element accuracy ──
        elem_acc = element_accuracy(t_atoms.elements, p_atoms.elements)

        # ── Spacegroup ──
        t_spg = get_spacegroup_safe(t_atoms)
        p_spg = get_spacegroup_safe(p_atoms)

        # ── Num atoms ──
        t_nat = t_atoms.num_atoms
        p_nat = p_atoms.num_atoms

        records.append({
            'id': mid,
            'target_a': t_abc[0], 'target_b': t_abc[1], 'target_c': t_abc[2],
            'target_alpha': t_ang[0], 'target_beta': t_ang[1], 'target_gamma': t_ang[2],
            'pred_a': p_abc[0], 'pred_b': p_abc[1], 'pred_c': p_abc[2],
            'pred_alpha': p_ang[0], 'pred_beta': p_ang[1], 'pred_gamma': p_ang[2],
            'target_vol': t_vol, 'pred_vol': p_vol,
            'target_density': t_den, 'pred_density': p_den,
            'target_weight': t_weight, 'pred_weight': p_weight,
            'target_formula': t_formula, 'pred_formula': p_formula,
            'target_spg': t_spg, 'pred_spg': p_spg,
            'target_natoms': t_nat, 'pred_natoms': p_nat,
            'rmse': rmse,
            'elem_acc': elem_acc,
        })

    if skipped:
        print(f"Skipped {skipped} entries due to parsing errors")

    df = pd.DataFrame(records)

    # Derived columns
    df['n_atoms_match'] = (df['target_natoms'] == df['pred_natoms']).astype(int)
    df['formula_match'] = (df['target_formula'] == df['pred_formula']).astype(int)
    df['spg_match'] = (df['target_spg'] == df['pred_spg']).astype(int)

    # ══════════════════════════════════════════════════════════════════
    # Summary table
    # ══════════════════════════════════════════════════════════════════
    props = {
        'a (Å)':       ('target_a', 'pred_a'),
        'b (Å)':       ('target_b', 'pred_b'),
        'c (Å)':       ('target_c', 'pred_c'),
        'α (deg)':     ('target_alpha', 'pred_alpha'),
        'β (deg)':     ('target_beta', 'pred_beta'),
        'γ (deg)':     ('target_gamma', 'pred_gamma'),
        'Volume (ų)': ('target_vol', 'pred_vol'),
        'Density (g/cm³)': ('target_density', 'pred_density'),
        'Weight (AMU)': ('target_weight', 'pred_weight'),
    }

    summary_rows = []
    for name, (tcol, pcol) in props.items():
        tv = df[tcol].values; pv = df[pcol].values
        mask = np.isfinite(tv) & np.isfinite(pv)
        tv, pv = tv[mask], pv[mask]
        mae = mean_absolute_error(tv, pv)
        kld = kld_hist(tv, pv)
        emd = wasserstein_distance(tv, pv)
        summary_rows.append({
            'Property': name,
            'Min': f'{tv.min():.2f}', 'Max': f'{tv.max():.2f}',
            'MAE': f'{mae:.3f}',
            'KLD': f'{kld:.3f}', 'EMD': f'{emd:.3f}',
        })

    valid_rmse = df['rmse'].dropna()
    summary_rows.append({'Property': 'Coord RMSE', 'Min': f'{valid_rmse.min():.4f}',
        'Max': f'{valid_rmse.max():.4f}', 'MAE': f'{valid_rmse.mean():.4f} (mean)',
        'KLD': '-', 'EMD': '-'})
    summary_rows.append({'Property': 'Elem. accuracy', 'Min': f'{df["elem_acc"].min():.3f}',
        'Max': f'{df["elem_acc"].max():.3f}', 'MAE': f'{df["elem_acc"].mean():.3f} (mean)',
        'KLD': '-', 'EMD': '-'})
    summary_rows.append({'Property': 'Formula match', 'Min': '-', 'Max': '-',
        'MAE': f'{df["formula_match"].mean()*100:.1f}%', 'KLD': '-', 'EMD': '-'})
    summary_rows.append({'Property': 'N_atoms match', 'Min': '-', 'Max': '-',
        'MAE': f'{df["n_atoms_match"].mean()*100:.1f}%', 'KLD': '-', 'EMD': '-'})
    summary_rows.append({'Property': 'Spacegroup match', 'Min': '-', 'Max': '-',
        'MAE': f'{df["spg_match"].mean()*100:.1f}%', 'KLD': '-', 'EMD': '-'})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f'{prefix}_summary.csv', index=False)
    print("\n═══ Summary Table ═══")
    print(summary_df.to_string(index=False))

    df.to_csv(f'{prefix}_per_structure.csv', index=False)

    # ══════════════════════════════════════════════════════════════════
    # Figure 1: Distribution comparison
    # ══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('RamanGPT: Predicted vs Target Property Distributions',
                 fontsize=15, fontweight='bold', y=1.02)

    dist_props = [
        ('target_a', 'pred_a', 'a (Å)'),
        ('target_b', 'pred_b', 'b (Å)'),
        ('target_c', 'pred_c', 'c (Å)'),
        ('target_gamma', 'pred_gamma', 'γ (°)'),
        ('target_alpha', 'pred_alpha', 'α (°)'),
        ('target_beta', 'pred_beta', 'β (°)'),
        ('target_vol', 'pred_vol', 'Volume (ų)'),
        ('target_weight', 'pred_weight', 'Weight (AMU)'),
    ]

    for ax, (tcol, pcol, label) in zip(axes.flat, dist_props):
        tv = df[tcol].dropna().values
        pv = df[pcol].dropna().values
        lo = min(tv.min(), pv.min())
        hi = max(tv.max(), pv.max())
        # Cap lattice length histograms at 20 Å
        if 'Å' in label:
            hi = min(hi, 20.0)
        bins = np.linspace(lo, hi, 30)
        ax.hist(tv, bins=bins, alpha=0.6, color=C_TARGET, label='Target', edgecolor='white', linewidth=0.5)
        ax.hist(pv, bins=bins, alpha=0.6, color=C_PRED, label='Predicted', edgecolor='white', linewidth=0.5)
        ax.set_xlabel(label); ax.set_ylabel('Count')
        if 'Å' in label:
            ax.set_xlim(lo, 20.0)
        kld = kld_hist(tv, pv); emd = wasserstein_distance(tv, pv)
        ax.text(0.97, 0.95, f'KLD={kld:.3f}\nEMD={emd:.2f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    axes[0, 0].legend(framealpha=0.9)
    plt.tight_layout()
    fig.savefig(f'{prefix}_fig1_distributions.pdf')
    fig.savefig(f'{prefix}_fig1_distributions.png')
    print(f"\nSaved: {prefix}_fig1_distributions")

    # ══════════════════════════════════════════════════════════════════
    # Figure 2: Parity plots for lattice params
    # ══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('RamanGPT: Parity Plots (Predicted vs Target)',
                 fontsize=15, fontweight='bold', y=1.02)

    parity_props = [
        ('target_a', 'pred_a', 'a (Å)'),
        ('target_b', 'pred_b', 'b (Å)'),
        ('target_c', 'pred_c', 'c (Å)'),
        ('target_alpha', 'pred_alpha', 'α (°)'),
        ('target_beta', 'pred_beta', 'β (°)'),
        ('target_gamma', 'pred_gamma', 'γ (°)'),
    ]

    for ax, (tcol, pcol, label) in zip(axes.flat, parity_props):
        tv = df[tcol].values; pv = df[pcol].values
        mask = np.isfinite(tv) & np.isfinite(pv)
        tv, pv = tv[mask], pv[mask]
        # Cap lattice lengths at 20 Å for cleaner plots
        if 'Å' in label:
            cap = 20.0
            keep = (tv <= cap) & (pv <= cap)
            tv, pv = tv[keep], pv[keep]
            lims = [0, cap]
        else:
            lims = [min(tv.min(), pv.min()) * 0.9, max(tv.max(), pv.max()) * 1.1]
        ax.scatter(tv, pv, alpha=0.5, s=20, c=C_TARGET, edgecolors='none')
        ax.plot(lims, lims, '--', color=C_ACCENT, linewidth=2)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(f'Target {label}'); ax.set_ylabel(f'Predicted {label}')
        mae = mean_absolute_error(tv, pv)
        ax.text(0.05, 0.95, f'MAE={mae:.3f}', transform=ax.transAxes,
                ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    plt.tight_layout()
    fig.savefig(f'{prefix}_fig2_parity.pdf')
    fig.savefig(f'{prefix}_fig2_parity.png')
    print(f"Saved: {prefix}_fig2_parity")

    # ══════════════════════════════════════════════════════════════════
    # Figure 3: Volume, RMSE, Element accuracy
    # ══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('RamanGPT: Volume, Coordinate RMSE, and Element Accuracy',
                 fontsize=14, fontweight='bold', y=1.03)

    # Volume parity
    ax = axes[0]
    tv = df['target_vol'].values; pv = df['pred_vol'].values
    mask = np.isfinite(tv) & np.isfinite(pv)
    ax.scatter(tv[mask], pv[mask], alpha=0.5, s=20, c=C_TARGET, edgecolors='none')
    lims = [0, max(tv[mask].max(), pv[mask].max()) * 1.1]
    ax.plot(lims, lims, '--', color=C_ACCENT, linewidth=2)
    ax.set_xlabel('Target Volume (ų)'); ax.set_ylabel('Predicted Volume (ų)')
    mae = mean_absolute_error(tv[mask], pv[mask])
    ax.text(0.05, 0.95, f'MAE={mae:.2f}', transform=ax.transAxes,
            ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.set_title('(a) Volume comparison')

    # RMSE distribution
    ax = axes[1]
    vr = df['rmse'].dropna().values
    ax.hist(vr, bins=30, alpha=0.7, color=C_PRED, edgecolor='white')
    ax.axvline(np.median(vr), color='k', ls='--', label=f'Median={np.median(vr):.4f}')
    ax.set_xlabel('Coordinate RMSE (fractional)'); ax.set_ylabel('Count')
    ax.legend(); ax.set_title('(b) RMSE distribution')

    # Element accuracy
    ax = axes[2]
    ax.hist(df['elem_acc'].values, bins=np.linspace(0, 1, 21), alpha=0.7,
            color=C_ACCENT, edgecolor='white')
    ax.axvline(df['elem_acc'].mean(), color='k', ls='--', label=f'Mean={df["elem_acc"].mean():.3f}')
    ax.set_xlabel('Element Accuracy'); ax.set_ylabel('Count')
    ax.legend(); ax.set_title('(c) Composition accuracy')

    plt.tight_layout()
    fig.savefig(f'{prefix}_fig3_vol_rmse_elem.pdf')
    fig.savefig(f'{prefix}_fig3_vol_rmse_elem.png')
    print(f"Saved: {prefix}_fig3_vol_rmse_elem")

    # ══════════════════════════════════════════════════════════════════
    # Figure 4: MAE bar chart + RMSE vs natoms
    # ══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('RamanGPT: Error Summary', fontsize=14, fontweight='bold', y=1.03)

    ax = axes[0]
    prop_names = []; mae_vals = []
    for name, (tcol, pcol) in props.items():
        tv = df[tcol].values; pv = df[pcol].values
        mask = np.isfinite(tv) & np.isfinite(pv)
        mae_vals.append(mean_absolute_error(tv[mask], pv[mask]))
        prop_names.append(name)
    bars = ax.barh(prop_names, mae_vals, color=C_TARGET, edgecolor='white')
    ax.set_xlabel('MAE'); ax.set_title('(a) MAE by property')
    for bar, val in zip(bars, mae_vals):
        ax.text(bar.get_width() + max(mae_vals)*0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    ax = axes[1]
    mask = df['rmse'].notna()
    ax.scatter(df.loc[mask, 'target_natoms'], df.loc[mask, 'rmse'],
               alpha=0.5, s=20, c=C_PRED, edgecolors='none')
    ax.set_xlabel('Number of atoms'); ax.set_ylabel('Coordinate RMSE (frac)')
    ax.set_title('(b) RMSE vs. structure complexity')

    plt.tight_layout()
    fig.savefig(f'{prefix}_fig4_error_summary.pdf')
    fig.savefig(f'{prefix}_fig4_error_summary.png')
    print(f"Saved: {prefix}_fig4_error_summary")

    # ══════════════════════════════════════════════════════════════════
    # Print key results
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("KEY RESULTS FOR RAMANGPT PAPER")
    print("="*70)
    print(f"Number of test structures: {len(df)}")
    print(f"Lattice a MAE: {mean_absolute_error(df['target_a'], df['pred_a']):.3f} Å")
    print(f"Lattice b MAE: {mean_absolute_error(df['target_b'], df['pred_b']):.3f} Å")
    print(f"Lattice c MAE: {mean_absolute_error(df['target_c'], df['pred_c']):.3f} Å")
    print(f"Volume MAE: {mean_absolute_error(df['target_vol'], df['pred_vol']):.2f} ų")
    den_mask = df['target_density'].notna() & df['pred_density'].notna()
    print(f"Density MAE: {mean_absolute_error(df.loc[den_mask,'target_density'], df.loc[den_mask,'pred_density']):.3f} g/cm³ ({den_mask.sum()}/{len(df)} valid)")
    print(f"Coordinate RMSE (mean/median): {df['rmse'].mean():.4f} / {df['rmse'].median():.4f}")
    print(f"Element accuracy (mean): {df['elem_acc'].mean():.3f}")
    print(f"Formula exact match: {df['formula_match'].mean()*100:.1f}%")
    print(f"N_atoms exact match: {df['n_atoms_match'].mean()*100:.1f}%")
    print(f"Spacegroup match: {df['spg_match'].mean()*100:.1f}%")

    print("\n── KLD / EMD Table ──")
    print(f"{'Property':<18} {'Min':>8} {'Max':>8} {'MAE':>8} {'KLD':>8} {'EMD':>8}")
    print("-"*60)
    for name, (tcol, pcol) in props.items():
        tv = df[tcol].values; pv = df[pcol].values
        mask = np.isfinite(tv) & np.isfinite(pv)
        tv, pv = tv[mask], pv[mask]
        mae = mean_absolute_error(tv, pv)
        kld = kld_hist(tv, pv); emd = wasserstein_distance(tv, pv)
        print(f"{name:<18} {tv.min():>8.2f} {tv.max():>8.2f} {mae:>8.3f} {kld:>8.3f} {emd:>8.3f}")

    print(f"\nAll outputs saved with prefix: {prefix}_*")
    return df, summary_df


if __name__ == '__main__':
    json_path = sys.argv[1] if len(sys.argv) > 1 else 'raman_res.json'
    prefix = sys.argv[2] if len(sys.argv) > 2 else 'ramangpt'
    main(json_path, prefix)
