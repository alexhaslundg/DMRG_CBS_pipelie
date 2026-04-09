"""
Complete Basis Set (CBS) Extrapolation Analyzer
Supports both diatomic molecules (N2) and hydrogen chains with appropriate handling.

Key improvements:
- System-type aware analysis (diatomic vs chain)
- Handles missing basis set ladders gracefully
- Separate extrapolation strategies for different system types
- Clear warnings when CBS is not possible

Shared plotting & export utilities (moved from notebooks):
- plot_cbs_molecule        — generic CBS convergence plot for any molecule
- plot_cbs_group           — H-chain group CBS convergence plot
- to_minimal_schema        — convert cbs_df to standardised 11-column CSV schema
- load_mol_bd_energies     — scan *_BD*.json files for bond-dimension sweep data
- extrapolate_bd           — fit exp-log² model to BD sweep data
- plot_mol_bd_convergence  — plot BD convergence, return extrapolated DataFrame
- plot_mol_cbs_from_bd_extrap — CBS from BD-extrapolated energies
- parse_hchain_filename    — parse H-chain JSON filename into metadata dict
"""
import re
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=UserWarning)

sns.set(style="whitegrid", context="talk")

# ── Colour / style constants (imported from central style module) ─────────────
# NOTE: plotting_style is a local module not included in this repo.
# The CBS extrapolation functions (hf_model, corr_model, cbs_3pt_algebraic, etc.)
# work without it. Only the plotting helpers (plot_cbs_molecule etc.) require it.
# To use the plotting helpers, either provide your own plotting_style.py or
# replace these imports with plain matplotlib defaults.
try:
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', 'plotting'))
    from plotting_style import (
        METHOD_COLORS, METHOD_MARKERS, BASIS_COLORS, BASIS_CARDINAL_MAP,
        ETYPE_COLORS, ETYPE_STYLES, EXTRAP_PRIORITY,
        GEO_LABEL, GEO_ORDER, GEO_COLORS, GEO_MARKERS,
        NREPS_LINESTYLES, MARKER_SIZE, LINE_WIDTH, LIT_REF, LIT_COLORS,
        STYLE_CONFIG,
    )
except ImportError:
    # Fallback: plotting helpers will not work, but extrapolation functions will.
    METHOD_COLORS = METHOD_MARKERS = BASIS_COLORS = BASIS_CARDINAL_MAP = {}
    ETYPE_COLORS = ETYPE_STYLES = EXTRAP_PRIORITY = {}
    GEO_LABEL = GEO_ORDER = GEO_COLORS = GEO_MARKERS = {}
    NREPS_LINESTYLES = {}
    MARKER_SIZE = 8; LINE_WIDTH = 2; LIT_REF = {}; LIT_COLORS = {}
    STYLE_CONFIG = {}

# Standardised output column schema
SCHEMA_COLS = [
    'system_name', 'bond_length_bohr', 'bond_length_ang', 'method', 'basis_scheme',
    'extrapolation_type', 'CBS_energy', 'CBS_energy_per_atom',
    'n_atoms', 'source', 'units', 'note',
]


# ── Public CBS model functions ────────────────────────────────────────────────

def hf_model(X, Ecbs, a, b):
    """E_HF(X) = E_HF_CBS + a·exp(-b·X)   [Feller 1992]"""
    return Ecbs + a * np.exp(-b * X)


def corr_model(X, Ec, a):
    """E_c(X) = E_c_CBS + A·X^{-3}        [Helgaker et al. 1997]"""
    return Ec + a * X**(-3)


def cbs_3pt_algebraic(E1, E2, E3):
    """
    CBS limit via 3-point algebraic exponential extrapolation.
    Assumes E(X) = E_CBS + A·exp(-B·X), solved analytically for 3 cardinal numbers.
    Returns None when energies are non-convex or denominator is near zero.
    """
    if not (E1 - E2 > E2 - E3):
        return None
    denom = E1 - 2 * E2 + E3
    if abs(denom) < 1e-12:
        return None
    return (E1 * E3 - E2**2) / denom


# Fixed exponent B for the Halkier HF two-point formula (Halkier et al. 1999).
_HALKIER_B = 1.637

# Unit conversion used when matching CSV reference energies (bond_length_bohr) to
# Angstrom-based bond lengths used in plot_cbs_molecule.
_BOHR_TO_ANG = 0.529177210903

# Maps geo_label strings to stretch_factor values used in reference_energies_spline.csv.
_GEO_LABEL_TO_STRETCH = {'eq': 1.0, '1.5x': 1.5, '2.0x': 2.0}


def hf_halkier_two_point(E_n, E_m, n, m, B=_HALKIER_B):
    """
    Two-point HF/CBS extrapolation with a *fixed* exponential decay constant B.

    Reference: Halkier et al., Chem. Phys. Lett. 302 (1999) 437-446, Eq. (4).
    As cited in ByteDance Supplementary Note 6.2, Eq. (4):

        E_HF_CBS = E_HF_n - (E_HF_n - E_HF_{n+1}) / (1 - exp(-B))

    where B = 1.637 is a universal constant and n, n+1 are the ζ cardinalities.
    This implementation generalises to any two cardinal numbers n < m:

        E_HF_CBS = E_n - (E_n - E_m) / (1 - exp(-B * (m - n)))

    which reduces to the published formula when m = n + 1.

    Args:
        E_n: HF energy at the smaller cardinal number n.
        E_m: HF energy at the larger cardinal number m (m > n).
        n:   Smaller ζ cardinal number.
        m:   Larger ζ cardinal number.
        B:   Exponential decay constant (default: 1.637).

    Returns:
        Extrapolated CBS HF energy.
    """
    denom = 1.0 - np.exp(-B * (m - n))
    if abs(denom) < 1e-12:
        raise ValueError(f"Halkier denominator too small for n={n}, m={m}, B={B}")
    return E_n - (E_n - E_m) / denom


# ── Shared helper ─────────────────────────────────────────────────────────────

def _safe(row, col):
    """Return float value of row[col], or None if missing / NaN."""
    if col not in row.index:
        return None
    v = row[col]
    return None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)


# ── Export utility ────────────────────────────────────────────────────────────

def to_minimal_schema(cbs_df: pd.DataFrame, system_name: str,
                      n_atoms: int, source: str) -> pd.DataFrame:
    """
    Convert a CBSAnalyzer cbs_df to the standardised 11-column minimal schema.

    Keeps one row per (bond_length, method, basis_scheme) using the highest-priority
    extrapolation type with a valid CBS_energy (order defined by EXTRAP_PRIORITY).

    Args:
        cbs_df:      DataFrame produced by CBSAnalyzer.extract_all_cbs_methods()
        system_name: Label for the system (e.g. 'LiH_singlet', 'H4')
        n_atoms:     Number of atoms (used to compute CBS_energy_per_atom)
        source:      Provenance string (e.g. 'mar_25', 'local_CBS_DMRG')

    Returns:
        DataFrame with columns defined by SCHEMA_COLS.
    """
    rows = []
    for (method, basis_scheme, bond_length_bohr, bond_length_ang), grp in cbs_df.groupby(
            ['method', 'basis_scheme', 'bond_length_bohr', 'bond_length_ang'], sort=False):
        for etype in EXTRAP_PRIORITY:
            subset = grp #[grp['extrapolation_type'] == etype]
            if not subset.empty and pd.notna(subset.iloc[0]['CBS_energy']):
                row  = subset.iloc[0]
                cbs_e = float(row['CBS_energy'])
                rows.append({
                    'system_name':         system_name,
                    'bond_length_bohr':         round(float(bond_length_bohr), 6),
                    'bond_length_ang':         round(float(bond_length_ang), 6),
                    'method':              method,
                    'basis_scheme':        basis_scheme,
                    'extrapolation_type':  etype,
                    'CBS_energy':          cbs_e,
                    'CBS_energy_per_atom': cbs_e / n_atoms,
                    'n_atoms':             n_atoms,
                    'source':              source,
                    'units':               'Ha',
                    'note':                str(row.get('note', '') or ''),
                })
                break
    return pd.DataFrame(rows, columns=SCHEMA_COLS)


# ── H-chain file parsing ──────────────────────────────────────────────────────

_HCHAIN_FN_RE = re.compile(
    r'H_single_chain_(\d+)rep_singlet_bohr_H(\d+)_dA([\d\.]+)_(\w+)_(cc-[^_]+)\.json'
)


def parse_hchain_filename(path) -> Optional[dict]:
    """
    Parse an H-chain JSON filename and return a metadata dict.

    Expected filename pattern::

        H_single_chain_{n}rep_singlet_bohr_H{N}_dA{dA}_{method}_{basis}.json

    Returns a dict with keys ``n_reps``, ``n_atoms``, ``system_name``,
    ``dA_str``, ``method``, ``basis``, or *None* if the pattern does not match.
    """
    m = _HCHAIN_FN_RE.match(Path(path).name)
    if m:
        return {
            'n_reps':      int(m.group(1)),
            'n_atoms':     int(m.group(2)),
            'system_name': f'H{m.group(2)}',
            'dA_str':      m.group(3),
            'method':      m.group(4),
            'basis':       m.group(5),
        }
    return None


# ── CBS convergence plotting ──────────────────────────────────────────────────

_ENERGY_SOURCE_MAP = {
    'cbs':    None,              # CBS reference lines only (default)
    'final':  'energy',          # DMRG energy at the individual basis sets
    'min_bd': 'extrap_energy_bds',  # bond-dim extrapolated minimum energy
}


def plot_cbs_molecule(individual_df: pd.DataFrame, cbs_df: pd.DataFrame,
                      system_name: str, basis_scheme: str = 'DZ/TZ/QZ',
                      title_suffix: str = '',
                      methods: Optional[List[str]] = None,
                      energy_source: str = 'cbs',
                      ref_df: Optional[pd.DataFrame] = None) -> None:
    """
    Plot CBS convergence for any molecule (LiH, BeH2, N2, …).

    Rows    = bond lengths, columns = methods (HF, FCI, DMRG, CCSDT).
    Individual basis energies are shown as scatter points; CBS fit curves
    and reference lines are overlaid where available.

    Args:
        individual_df: DataFrame from CBSAnalyzer.extract_all_cbs_methods()[1]
        cbs_df:        DataFrame from CBSAnalyzer.extract_all_cbs_methods()[0]
        system_name:   Used for subplot titles and to filter individual_df rows
        basis_scheme:  Basis scheme string to filter cbs_df (e.g. 'DZ/TZ/QZ')
        title_suffix:  Extra text appended to the figure title
        methods:       List of methods to plot, e.g. ['DMRG', 'FCI'].
                       None (default) plots all methods present in the data.
        energy_source: Which energy values to show as scatter points:
                       'cbs'    — individual basis-set energies (default)
                       'final'  — same as 'cbs' (energy column)
                       'min_bd' — bond-dimension extrapolated energy column
                       CBS reference lines are always shown when cbs_df is
                       provided, regardless of energy_source.
        ref_df:        Optional DataFrame from reference_energies_spline.csv
                       (columns: system, stretch_factor, bond_length_bohr,
                       ref_energy_spline).  When provided, a black dash-dot
                       horizontal line is added to every subplot showing the
                       spline CBS reference energy for the matching bond length.
    """
    if energy_source not in _ENERGY_SOURCE_MAP:
        raise ValueError(
            f"energy_source={energy_source!r} not recognised. "
            f"Choose from {list(_ENERGY_SOURCE_MAP)}"
        )
    if 'system_name' in individual_df.columns:
        indiv = individual_df[individual_df['system_name'] == system_name].copy()
    else:
        indiv = individual_df.copy()

    if indiv.empty:
        print(f"No individual data for {system_name}")
        return

    d_A_vals = sorted(indiv['bond_length_bohr'].unique())
    _available_methods = [m for m in ['HF', 'FCI', 'DMRG', 'CCSDT'] if m in indiv['method'].unique()]
    methods_to_plot = (
        [m for m in methods if m in _available_methods]
        if methods is not None else _available_methods
    )

    # Build bond_length ((borh)) → ref_energy lookup from optional spline CSV
    _ref_lookup: Dict[float, float] = {}
    if ref_df is not None:
        _ref_sub = ref_df[ref_df['system'].str.lower() == system_name.lower()]
        for _, _rr in _ref_sub.iterrows():
            _bl_ang = float(_rr['bond_length_bohr']) * _BOHR_TO_ANG
            if pd.notna(_rr.get('ref_energy_spline')):
                _ref_lookup[_bl_ang] = float(_rr['ref_energy_spline'])

    n_rows, n_cols = len(d_A_vals), len(methods_to_plot)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5.5 * n_cols, 4.5 * n_rows),
                              squeeze=False)
    fig.suptitle(
        f"{system_name} CBS Convergence: Energy vs Cardinal Number X  {title_suffix}",
        fontsize=13, fontweight='bold', y=1.01,
    )

    cbs_available = (cbs_df is not None
                     and not cbs_df.empty
                     and 'extrapolation_type' in cbs_df.columns)

    for ri, d_A in enumerate(d_A_vals):
        mdata = indiv[abs(indiv['bond_length_bohr'] - d_A) < 1e-4]

        for ci, method in enumerate(methods_to_plot):
            ax     = axes[ri][ci]
            ax.ticklabel_format(useOffset=False, style='plain')
            color  = METHOD_COLORS.get(method, 'gray')
            marker = METHOD_MARKERS.get(method, 'o')

            mmdata = (mdata[mdata['method'] == method].copy()
                          .assign(cardinal=lambda d: d['basis'].map(BASIS_CARDINAL_MAP))
                          .dropna(subset=['cardinal', 'energy'])
                          .sort_values('cardinal'))

            if mmdata.empty:
                ax.set_visible(False)
                continue

            X_vals = mmdata['cardinal'].values.astype(float)

            # Select the energy column based on energy_source
            energy_col = _ENERGY_SOURCE_MAP.get(energy_source)
            if energy_col is not None and energy_col in mmdata.columns:
                _ecol_vals = mmdata[energy_col].values.astype(float)
                if not np.all(np.isnan(_ecol_vals)):
                    E_vals = _ecol_vals
                    scatter_label = f'Basis energies ({energy_source})'
                else:
                    E_vals = mmdata['energy'].values.astype(float)
                    scatter_label = 'Basis energies'
            else:
                E_vals = mmdata['energy'].values.astype(float)
                scatter_label = 'Basis energies'

            ax.scatter(X_vals, E_vals, color=color, marker=marker, s=90, zorder=5,
                       label=scatter_label, edgecolors='k', linewidths=0.8)
            ax.plot(X_vals, E_vals, color=color, linewidth=0.7, alpha=0.35, zorder=4)

            for xi, ei in zip(X_vals, E_vals):
                ax.annotate(f"{ei:.5f}",
                            xy=(xi, ei),
                            xytext=(5, 3), textcoords='offset points',
                            fontsize=7, color=color)

            if cbs_available:
                mask = (
                    (abs(cbs_df['bond_length_bohr'] - d_A) < 1e-3) &
                    (cbs_df['method']       == method) &
                    (cbs_df['basis_scheme'] == basis_scheme)
                )
                if 'system_name' in cbs_df.columns:
                    mask &= (cbs_df['system_name'] == system_name)
                cbs_sub = cbs_df[mask].drop_duplicates(subset=['extrapolation_type'])

                X_fine = np.linspace(X_vals[0] - 0.1, X_vals[-1] + 1.8, 300)

                if method == 'HF':
                    row_hf = cbs_sub[cbs_sub['extrapolation_type'] == 'hf_exponential']
                    if not row_hf.empty:
                        r = row_hf.iloc[0]
                        Ecbs, a, b = _safe(r, 'CBS_energy'), _safe(r, 'HF_fit_a'), _safe(r, 'HF_fit_b')
                        if all(v is not None for v in [Ecbs, a, b]):
                            ax.plot(X_fine, hf_model(X_fine, Ecbs, a, b),
                                    color=color, linestyle=':', linewidth=1.8, alpha=0.75,
                                    label=r'Fit: $E_{HF}+ae^{-bX}$', zorder=3)
                else:
                    # two_part smooth fit curve
                    row_tp = cbs_sub[cbs_sub['extrapolation_type'] == 'two_part']
                    if not row_tp.empty:
                        r = row_tp.iloc[0]
                        E_hf_cbs = _safe(r, 'E_HF_CBS')
                        E_c_cbs  = _safe(r, 'E_corr_CBS')
                        hf_a     = _safe(r, 'HF_fit_a')
                        hf_b     = _safe(r, 'HF_fit_b')
                        corr_a   = _safe(r, 'Corr_fit_a')
                        if all(v is not None for v in [E_hf_cbs, E_c_cbs, hf_a, hf_b, corr_a]):
                            E_fit = (hf_model(X_fine, E_hf_cbs, hf_a, hf_b)
                                     + corr_model(X_fine, E_c_cbs, corr_a))
                            ax.plot(X_fine, E_fit,
                                    color=color, linestyle=':', linewidth=1.8, alpha=0.75,
                                    label=r'Fit: $E_{HF}(X)+E_c(X)$', zorder=3)

                    # halkier_two_part smooth fit curve (highest priority type)
                    row_hk = cbs_sub[cbs_sub['extrapolation_type'] == 'halkier_two_part']
                    if not row_hk.empty:
                        r = row_hk.iloc[0]
                        E_hf_cbs = _safe(r, 'E_HF_CBS')
                        E_c_cbs  = _safe(r, 'E_corr_CBS')
                        corr_a   = _safe(r, 'Corr_fit_a')
                        # Recover HF exponential amplitude from stored HF_{key} columns
                        hf_amp = None
                        for bk, xk in [('QZ', 4), ('TZ', 3), ('aQZ', 4), ('aTZ', 3)]:
                            e_hf_k = _safe(r, f'HF_{bk}')
                            if e_hf_k is not None and E_hf_cbs is not None:
                                hf_amp = (e_hf_k - E_hf_cbs) / np.exp(-_HALKIER_B * xk)
                                break
                        if all(v is not None for v in [E_hf_cbs, E_c_cbs, corr_a, hf_amp]):
                            E_fit_hk = (hf_model(X_fine, E_hf_cbs, hf_amp, _HALKIER_B)
                                        + corr_model(X_fine, E_c_cbs, corr_a))
                            ax.plot(X_fine, E_fit_hk,
                                    color=color, linestyle='--', linewidth=1.8, alpha=0.75,
                                    label=r'Halkier fit: $E_{HF}(X)+E_c(X)$', zorder=3)

                # CBS reference lines: show best available extrapolation
                row_hk = cbs_sub[cbs_sub['extrapolation_type'] == 'halkier_two_part']
                if not row_hk.empty and method != 'HF':
                    v   = row_hk.iloc[0]['CBS_energy']
                    ax.axhline(v, color=color, linestyle='-', linewidth=2.0, alpha=0.90,
                               label=f'CBS (Halkier) = {v:.5f} Ha', zorder=6)
                    ax.annotate(f'{v:.5f}',
                                xy=(1.0, v), xycoords=('axes fraction', 'data'),
                                xytext=(4, 0), textcoords='offset points',
                                fontsize=7.5, color=color, va='center')
                else:
                    etype_primary = 'hf_exponential' if method == 'HF' else 'two_part'
                    row_p = cbs_sub[cbs_sub['extrapolation_type'] == etype_primary]
                    if not row_p.empty:
                        v   = row_p.iloc[0]['CBS_energy']
                        lbl = 'CBS (exp fit)' if method == 'HF' else 'CBS (two-part)'
                        ax.axhline(v, color=color, linestyle='--', linewidth=2.0, alpha=0.90,
                                   label=f'{lbl} = {v:.5f} Ha', zorder=6)
                        ax.annotate(f'{v:.5f}',
                                    xy=(1.0, v), xycoords=('axes fraction', 'data'),
                                    xytext=(4, 0), textcoords='offset points',
                                    fontsize=7.5, color=color, va='center')

                row_3 = cbs_sub[cbs_sub['extrapolation_type'] == '3pt_exp_total']
                if not row_3.empty:
                    v3 = row_3.iloc[0]['CBS_energy']
                    ax.axhline(v3, color=color, linestyle='-.', linewidth=1.5, alpha=0.55,
                               label=f'CBS (3-pt) = {v3:.5f} Ha', zorder=5)

            # Spline CBS reference (same value for all methods at this bond length)
            for _ref_bl, _ref_e in _ref_lookup.items():
                if abs(_ref_bl - d_A) < 0.01:
                    ax.axhline(_ref_e, color='black', linestyle='--', linewidth=1.8,
                               alpha=0.65, label=f'CBS ref (spline) = {_ref_e:.5f} Ha',
                               zorder=7)
                    break

            ax.set_xlabel('Cardinal number X', fontsize=10)
            ax.set_ylabel('Energy (Ha)', fontsize=10)
            ax.set_xticks([2, 3, 4, 5])
            ax.set_xticklabels(['DZ', 'TZ', 'QZ', '5Z'])
            ax.set_title(f'{system_name}  d = {d_A:.4g} (borh)  –  {method}',
                         fontsize=10, fontweight='bold')
            ax.legend(fontsize=7, loc='lower right')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_cbs_group(sys_name: str, dA_str: str,
                   indiv_df: pd.DataFrame, cbs_df: pd.DataFrame,
                   methods: Optional[List[str]] = None,
                   energy_source: str = 'cbs',
                   ref_df: Optional[pd.DataFrame] = None) -> None:
    """
    Plot CBS convergence for one H-chain (system, geometry) group.

    A simpler, single-row version of plot_cbs_molecule tailored to H-chain data
    where only one bond length exists per group.

    Args:
        sys_name:      System label, e.g. 'H4'
        dA_str:        Geometry string from filename, e.g. '1.800'
                       (bond length in Bohr, as encoded in H-chain filenames)
        indiv_df:      Individual-basis DataFrame for this group
        cbs_df:        CBS DataFrame for this group
        methods:       List of methods to plot, e.g. ['DMRG', 'FCI'].
                       None (default) plots all methods present in the data.
        energy_source: Which energy column to use for scatter points:
                       'cbs' or 'final' → 'energy' column (default)
                       'min_bd' → 'extrap_energy_bds' column
        ref_df:        Optional DataFrame from reference_energies_spline.csv
                       (columns: system, stretch_factor, bond_length_bohr,
                       ref_energy_spline).  When provided, a black dashed
                       horizontal line is added to every subplot showing the
                       spline CBS reference energy for this geometry.
    """
    if energy_source not in _ENERGY_SOURCE_MAP:
        raise ValueError(
            f"energy_source={energy_source!r} not recognised. "
            f"Choose from {list(_ENERGY_SOURCE_MAP)}"
        )
    _available = [m for m in ['HF', 'FCI', 'DMRG'] if m in indiv_df['method'].unique()]
    methods_to_plot = (
        [m for m in methods if m in _available]
        if methods is not None else _available
    )
    if not methods_to_plot:
        return

    # Lookup spline CBS reference energy for this geometry (dA_str is in Bohr)
    _ref_e_group = None
    if ref_df is not None:
        _ref_sub = ref_df[ref_df['system'].str.lower() == sys_name.lower()]
        for _, _rr in _ref_sub.iterrows():
            if (abs(float(_rr['bond_length_bohr']) - float(dA_str)) < 0.01
                    and pd.notna(_rr.get('ref_energy_spline'))):
                _ref_e_group = float(_rr['ref_energy_spline'])
                break

    fig, axes = plt.subplots(1, len(methods_to_plot),
                             figsize=(5 * len(methods_to_plot), 4), squeeze=False)

    for ci, method in enumerate(methods_to_plot):
        ax = axes[0][ci]
        ax.ticklabel_format(useOffset=False, style='plain')

        pts = indiv_df[indiv_df['method'] == method].dropna(subset=['energy'])
        if pts.empty:
            ax.set_visible(False)
            continue

        Xs = pts['basis'].map(BASIS_CARDINAL_MAP).values.astype(float)
        _ecol = _ENERGY_SOURCE_MAP.get(energy_source)
        if _ecol is not None and _ecol in pts.columns:
            _ev = pts[_ecol].values.astype(float)
            Es = _ev if not np.all(np.isnan(_ev)) else pts['energy'].values
        else:
            Es = pts['energy'].values
        ok = ~np.isnan(Xs)

        color  = METHOD_COLORS.get(method, 'gray')
        marker = METHOD_MARKERS.get(method, 'o')

        ax.scatter(Xs[ok], Es[ok], color=color, marker=marker,
                   s=70, zorder=5, label='data')
        for x, e in zip(Xs[ok], Es[ok]):
            ax.annotate(str(int(x)), (x, e),
                        textcoords='offset points', xytext=(4, 4), fontsize=8)

        sub_cbs = cbs_df[cbs_df['method'] == method]
        for _, row in sub_cbs.iterrows():
            etype = row.get('extrapolation_type', '')
            ecbs  = _safe(row, 'CBS_energy')
            if ecbs is None:
                continue
            style  = ETYPE_STYLES.get(etype, {'ls': ':', 'lw': 1.0, 'label_prefix': etype})
            scheme = row.get('basis_scheme', '')
            ax.axhline(ecbs, ls=style['ls'], lw=style['lw'],
                       color=color, alpha=0.75,
                       label=f"{style['label_prefix']}({scheme}): {ecbs:.6f}")

        if _ref_e_group is not None:
            ax.axhline(_ref_e_group, color='black', linestyle='--', linewidth=1.8,
                       alpha=0.65, label=f'CBS ref (spline) = {_ref_e_group:.6f}',
                       zorder=7)

        ax.set_xlabel('Cardinal number X')
        ax.set_ylabel('E (Ha)')
        ax.set_title(f'{sys_name} dA={dA_str} | {method}', fontsize=9)
        ax.legend(fontsize=6)

    fig.suptitle(f'{sys_name} dA={dA_str}: CBS convergence', fontsize=11)
    fig.tight_layout()
    plt.show()


# ── FCI / method PES fitting ─────────────────────────────────────────────────

def fit_fci_pes(
    individual_df: pd.DataFrame,
    system_name: str,
    method: str = 'FCI',
    basis_scheme: str = 'DZ/TZ/QZ',
    cbs_df: Optional[pd.DataFrame] = None,
    fit_type: str = 'spline',
    n_points: int = 300,
    ax=None,
    show_plot: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit and optionally plot a potential-energy surface (PES) for the chosen
    method as a function of bond length.

    A cubic spline (fit_type='spline') is recommended for H2 with only 3
    data points; a Morse potential (fit_type='morse') works for ≥4 points.

    When *cbs_df* is provided, CBS-extrapolated energies are used (highest
    EXTRAP_PRIORITY type per bond length).  Otherwise, the highest-cardinal-
    number individual basis energy for each bond length is used.

    Args:
        individual_df:  DataFrame from CBSAnalyzer.extract_all_cbs_methods()[1]
        system_name:    System label used to filter rows (e.g. 'H2_singlet')
        method:         Quantum-chemistry method label (e.g. 'FCI', 'DMRG')
        basis_scheme:   Basis-scheme string used to filter cbs_df rows
        cbs_df:         Optional CBS DataFrame; preferred energy source
        fit_type:       'spline' for natural cubic spline,
                        'morse' for Morse potential (falls back to spline on failure)
        n_points:       Number of points in the returned smooth curve
        ax:             Matplotlib Axes to plot on; None creates a new figure
        show_plot:      Whether to display the figure (calls plt.tight_layout +
                        plt.show when True and ax was None)

    Returns:
        (r_smooth, E_smooth): 1-D arrays of bond lengths and fitted energies
    """
    from scipy.interpolate import CubicSpline

    # ── 1. Collect (bond_length, energy) data points ──────────────────────────
    pts: List[Tuple[float, float]] = []

    if cbs_df is not None and not cbs_df.empty:
        mask = (cbs_df['method'] == method) & (cbs_df['basis_scheme'] == basis_scheme)
        if 'system_name' in cbs_df.columns:
            mask &= (cbs_df['system_name'] == system_name)
        src = cbs_df[mask].copy()
        for bl, grp in src.groupby('bond_length_bohr'):
            for etype in EXTRAP_PRIORITY:
                row = grp[grp['extrapolation_type'] == etype]
                if not row.empty:
                    e = _safe(row.iloc[0], 'CBS_energy')
                    if e is not None:
                        pts.append((float(bl), float(e)))
                        break

    if len(pts) < 2:
        # Fallback: highest-cardinal individual basis energy per bond length
        sub = individual_df[individual_df['method'] == method].copy()
        if 'system_name' in sub.columns:
            sub = sub[sub['system_name'] == system_name]
        sub = sub.dropna(subset=['energy'])
        sub['_cardinal'] = sub['basis'].map(BASIS_CARDINAL_MAP)
        sub = sub.dropna(subset=['_cardinal'])
        for bl, grp in sub.groupby('bond_length_bohr'):
            best = grp.loc[grp['_cardinal'].idxmax()]
            pts.append((float(bl), float(best['energy'])))

    if len(pts) < 2:
        raise ValueError(
            f"fit_fci_pes: need ≥2 data points for {method} in {system_name}; "
            f"got {len(pts)}"
        )

    r_pts = np.array([p[0] for p in pts])
    E_pts = np.array([p[1] for p in pts])
    order = np.argsort(r_pts)
    r_pts, E_pts = r_pts[order], E_pts[order]

    r_smooth = np.linspace(r_pts[0], r_pts[-1], n_points)

    # ── 2. Fit ────────────────────────────────────────────────────────────────
    if fit_type == 'spline':
        cs = CubicSpline(r_pts, E_pts, bc_type='natural')
        E_smooth = cs(r_smooth)

    elif fit_type == 'morse':
        def _morse(r, De, re, a, E0):
            return E0 + De * (1 - np.exp(-a * (r - re))) ** 2 - De

        re0 = r_pts[np.argmin(E_pts)]
        p0  = [0.2, re0, 1.5, E_pts.min()]
        try:
            popt, _ = curve_fit(_morse, r_pts, E_pts, p0=p0, maxfev=20_000)
            E_smooth = _morse(r_smooth, *popt)
        except Exception as exc:
            warnings.warn(
                f"fit_fci_pes: Morse fit failed ({exc}); falling back to cubic spline."
            )
            cs = CubicSpline(r_pts, E_pts, bc_type='natural')
            E_smooth = cs(r_smooth)
    else:
        raise ValueError(
            f"fit_fci_pes: fit_type={fit_type!r} not supported. "
            "Use 'spline' or 'morse'."
        )

    # ── 3. Plot ───────────────────────────────────────────────────────────────
    if show_plot:
        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=(7, 4.5))

        color  = METHOD_COLORS.get(method, 'gray')
        marker = METHOD_MARKERS.get(method, 'o')

        ax.scatter(r_pts, E_pts,
                   color=color, marker=marker, s=80, zorder=5,
                   edgecolors='k', linewidths=0.8,
                   label=f'{method} data ({len(r_pts)} pts)')
        ax.plot(r_smooth, E_smooth,
                color=color, linewidth=2.0, linestyle='--', alpha=0.85,
                label=f'{method} {fit_type} fit')

        ax.set_xlabel('Bond length ((borh))',
                      fontsize=STYLE_CONFIG['label_fontsize'])
        ax.set_ylabel('Energy (Ha)',
                      fontsize=STYLE_CONFIG['label_fontsize'])
        ax.set_title(f'{system_name}: {method} PES ({fit_type} fit)',
                     fontsize=STYLE_CONFIG['title_fontsize'])
        ax.legend(fontsize=STYLE_CONFIG['legend_fontsize'])
        ax.grid(True, alpha=STYLE_CONFIG['grid_alpha'])
        ax.ticklabel_format(useOffset=False, style='plain')

        if created_fig:
            plt.tight_layout()
            plt.show()

    return r_smooth, E_smooth


# ── Bond-dimension extrapolation utilities ────────────────────────────────────

def load_mol_bd_energies(data_dir) -> pd.DataFrame:
    """
    Scan ``*_BD*.json`` files in *data_dir* and return a DataFrame of DMRG
    bond-dimension sweep energies.

    Columns: ``system_name``, ``geo_label``, ``basis``, ``bond_dim``,
    ``energy``, ``hf_energy``, ``converged``.

    The geometry label is extracted from ``system_name`` via a trailing ``_N.Nx``
    suffix (e.g. ``LiH_1.5x`` → ``'1.5x'``); systems without such a suffix are
    labelled ``'eq'``.
    """
    rows = []
    for path in sorted(Path(data_dir).glob('*_BD*.json')):
        m = re.search(r'_BD(\d+)\.json$', path.name)
        if not m:
            continue
        bond_dim = int(m.group(1))
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue

        sys_info = d.get('system_info', {})
        sys_name = sys_info.get('system_name', path.stem)
        basis    = (sys_info.get('basis_sets') or [None])[0]
        geo_keys = [k for k in d if k != 'system_info']
        if not geo_keys:
            continue
        geo_data = d[geo_keys[0]]
        if basis not in geo_data:
            continue

        dmrg = geo_data[basis].get('DMRG', {})
        hf   = geo_data[basis].get('HF', {})

        m_geo = re.search(r'_(\d+\.?\d*x)', sys_name)
        geo_label = m_geo.group(1) if m_geo else 'eq'

        rows.append({
            'system_name': sys_name,
            'geo_label':   geo_label,
            'basis':       basis,
            'bond_dim':    bond_dim,
            'energy':      dmrg.get('energy'),
            'hf_energy':   hf.get('energy') or dmrg.get('hf_energy'),
            'converged':   dmrg.get('converged', False),
        })
    return pd.DataFrame(rows)


def _bd_explog2(D, E_inf, A, k):
    """exp-log² bond-dimension model: E(D) = E_inf + A·exp(-k·log(D)²)"""
    return E_inf + A * np.exp(-k * np.log(D)**2)


def _monotonicity_filter(bds: np.ndarray, ens: np.ndarray):
    """Return (bds, ens) keeping only strictly decreasing energies (sorted by D)."""
    order = np.argsort(bds)
    bds, ens = bds[order], ens[order]
    mask, running_min = np.zeros(len(bds), bool), np.inf
    for i in range(len(bds)):
        if ens[i] < running_min:
            running_min = ens[i]
            mask[i] = True
    return bds[mask & (bds > 1)], ens[mask & (bds > 1)]


def extrapolate_bd(bond_dims, energies, min_points: int = 3) -> dict:
    """
    Fit the exp-log² model to bond-dimension sweep data after monotonicity filtering.

    Args:
        bond_dims:   Array-like of bond dimensions
        energies:    Corresponding energies
        min_points:  Minimum number of monotone points required for a fit

    Returns:
        dict with keys ``n_mono``, ``M_mono``, ``E_mono``, and optionally
        ``E_inf_explog2``, ``A_explog2``, ``k_explog2`` if fit succeeded.
    """
    M, E = np.array(bond_dims, float), np.array(energies, float)
    M_m, E_m = _monotonicity_filter(M, E)
    res = {'n_mono': len(M_m), 'M_mono': M_m, 'E_mono': E_m}
    if len(M_m) >= min_points:
        try:
            p0 = [E_m.min() - 1e-4, E_m.max() - E_m.min(), 0.5]
            popt, _ = curve_fit(_bd_explog2, M_m, E_m, p0=p0,
                                bounds=([-np.inf, 0, 0], [np.inf, np.inf, np.inf]),
                                maxfev=10000)
            res.update({
                'E_inf_explog2': float(popt[0]),
                'A_explog2':     float(popt[1]),
                'k_explog2':     float(popt[2]),
            })
        except Exception:
            pass
    return res


def plot_mol_bd_convergence(df: pd.DataFrame, molecule: str,
                             geo_order=None, basis_sets=None,
                             converged_only: bool = True,
                             figsize_per=(4.5, 3.5),
                             ref_df: Optional[pd.DataFrame] = None,
                             ref_system: Optional[str] = None) -> pd.DataFrame:
    """
    Plot DMRG bond-dimension convergence and return extrapolated energies.

    Layout: rows = geometry variants (eq/1.5x/2.0x), columns = basis sets.

    Args:
        df:             DataFrame from load_mol_bd_energies()
        molecule:       System label for the figure title
        geo_order:      List of geo_labels to show (default: GEO_ORDER)
        basis_sets:     List of basis sets to show (default: sorted by cardinal number)
        converged_only: If True, skip rows where converged=False
        figsize_per:    (width, height) per subplot panel
        ref_df:         Optional DataFrame from reference_energies_spline.csv
                        (columns: system, stretch_factor, bond_length_bohr,
                        ref_energy_spline).  When provided, a black dashed
                        horizontal line is added to every subplot showing the
                        spline CBS reference energy for the matching geometry.
        ref_system:     System name to look up in ref_df (defaults to molecule).

    Returns:
        DataFrame with columns ``system_name``, ``geo_label``, ``basis``,
        ``E_inf_explog2``, ``n_mono``.
    """
    geos  = geo_order  or [g for g in GEO_ORDER  if g in df['geo_label'].unique()]
    bases = basis_sets or sorted(df['basis'].unique(),
                                  key=lambda b: BASIS_CARDINAL_MAP.get(b, 99))
    nrows, ncols = len(geos), len(bases)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                              squeeze=False)
    extrap_rows = []

    for ri, geo in enumerate(geos):
        # Look up reference energy for this geometry (same for all basis subplots)
        _ref_e_geo = None
        if ref_df is not None:
            _mol = (ref_system or molecule)
            _sf  = _GEO_LABEL_TO_STRETCH.get(geo)
            if _sf is not None:
                _rr = ref_df[
                    (ref_df['system'].str.lower() == _mol.lower()) &
                    (abs(ref_df['stretch_factor'] - _sf) < 0.01)
                ]
                if not _rr.empty and pd.notna(_rr.iloc[0]['ref_energy_spline']):
                    _ref_e_geo = float(_rr.iloc[0]['ref_energy_spline'])

        for ci, basis in enumerate(bases):
            ax  = axes[ri][ci]
            ax.ticklabel_format(useOffset=False, style='plain')
            sub = df[(df['geo_label'] == geo) & (df['basis'] == basis)].copy()
            if converged_only:
                sub = sub[sub['converged']]
            sub = sub.dropna(subset=['energy']).sort_values('bond_dim')
            if sub.empty:
                ax.set_visible(False)
                continue

            M, E   = sub['bond_dim'].values.astype(float), sub['energy'].values
            color  = GEO_COLORS.get(geo, 'gray')
            marker = GEO_MARKERS.get(geo, 'o')

            ax.scatter(M, E, color='lightgrey', s=40, zorder=3, label='all')

            res = extrapolate_bd(M, E)
            if 'M_mono' in res and len(res['M_mono']) > 0:
                ax.scatter(res['M_mono'], res['E_mono'], color=color,
                           marker=marker, s=60, zorder=5,
                           label=f"mono ({res['n_mono']} pts)")
            if 'E_inf_explog2' in res:
                M_fine = np.geomspace(M.min(), M.max() * 1.5, 300)
                ax.plot(M_fine,
                        _bd_explog2(M_fine, res['E_inf_explog2'],
                                    res['A_explog2'], res['k_explog2']),
                        '-', color=color, lw=1.8, label='exp-log² fit')
                ax.axhline(res['E_inf_explog2'], color=color, ls='--', lw=1,
                           label=f"E∞={res['E_inf_explog2']:.6f}")
                sys_name = sub['system_name'].iloc[0]
                extrap_rows.append({
                    'system_name':   sys_name,
                    'geo_label':     geo,
                    'basis':         basis,
                    'E_inf_explog2': res['E_inf_explog2'],
                    'n_mono':        res['n_mono'],
                })

            if _ref_e_geo is not None:
                ax.axhline(_ref_e_geo, color='black', linestyle='--', linewidth=1.8,
                           alpha=0.65, label=f'CBS ref (spline) = {_ref_e_geo:.5f}',
                           zorder=7)

            ax.set_xlabel('Bond dimension M')
            ax.set_ylabel('E (Ha)')
            ax.set_title(f'{molecule} {geo} | {basis}', fontsize=9)
            ax.legend(fontsize=6, loc='lower right')

    fig.suptitle(f'{molecule}: DMRG BD Convergence (exp-log² model)', fontsize=11, y=1.01)
    fig.tight_layout()
    plt.show()

    cols = ['system_name', 'geo_label', 'basis', 'E_inf_explog2', 'n_mono']
    return pd.DataFrame(extrap_rows, columns=cols)


def plot_mol_cbs_from_bd_extrap(df_extrap: pd.DataFrame, df_raw: pd.DataFrame,
                                 molecule: str, basis_scheme: str = 'TZ/QZ/5Z',
                                 figsize_per=(6, 4),
                                 ref_df: Optional[pd.DataFrame] = None,
                                 ref_system: Optional[str] = None) -> pd.DataFrame:
    """
    Plot CBS extrapolation from BD-extrapolated energies and return CBS values.

    For each geometry variant, plots E_inf_explog2 vs cardinal number X and
    overlays the two-step CBS fit (HF exponential + correlation X^-3).

    Args:
        df_extrap:     DataFrame from plot_mol_bd_convergence()
        df_raw:        Raw BD DataFrame from load_mol_bd_energies() (for HF energies)
        molecule:      System label
        basis_scheme:  Which basis cardinal numbers to use for the fit
                       ('TZ/QZ/5Z', 'DZ/TZ/QZ/5Z', or 'QZ/5Z')
        figsize_per:   (width, height) per subplot panel
        ref_df:        Optional DataFrame from reference_energies_spline.csv
                       (columns: system, stretch_factor, bond_length_bohr,
                       ref_energy_spline).  When provided, a black dashed
                       horizontal line is added to each subplot showing the
                       spline CBS reference energy for the matching geometry.
        ref_system:    System name to look up in ref_df (defaults to molecule).

    Returns:
        DataFrame with columns ``geo_label``, ``basis_scheme``,
        ``E_HF_CBS``, ``E_corr_CBS``, ``E_total_CBS``.
    """
    scheme_cardinals = {
        'DZ/TZ/QZ':   [2, 3, 4],
        'TZ/QZ/5Z':    [3, 4, 5],
        'DZ/TZ/QZ/5Z': [2, 3, 4, 5],
        'QZ/5Z':       [4, 5],
        'DZ/TZ':       [2, 3],
    }
    fit_Xs = scheme_cardinals.get(basis_scheme, [3, 4, 5])
    geos   = [g for g in GEO_ORDER if g in df_extrap['geo_label'].unique()]
    fig, axes = plt.subplots(1, len(geos),
                              figsize=(figsize_per[0] * len(geos), figsize_per[1]),
                              squeeze=False)
    cbs_rows = []

    for ci, geo in enumerate(geos):
        ax  = axes[0][ci]
        ax.ticklabel_format(useOffset=False, style='plain')

        # Look up spline CBS reference energy for this geometry
        _ref_e_geo = None
        if ref_df is not None:
            _mol = (ref_system or molecule)
            _sf  = _GEO_LABEL_TO_STRETCH.get(geo)
            if _sf is not None:
                _rr = ref_df[
                    (ref_df['system'].str.lower() == _mol.lower()) &
                    (abs(ref_df['stretch_factor'] - _sf) < 0.01)
                ]
                if not _rr.empty and pd.notna(_rr.iloc[0]['ref_energy_spline']):
                    _ref_e_geo = float(_rr.iloc[0]['ref_energy_spline'])

        sub = df_extrap[df_extrap['geo_label'] == geo].copy()
        sub['X'] = sub['basis'].map(BASIS_CARDINAL_MAP)
        sub = sub.dropna(subset=['E_inf_explog2', 'X']).sort_values('X')

        # HF energies from raw data (BD-independent — take highest converged BD)
        hf_raw = (df_raw[(df_raw['geo_label'] == geo) & df_raw['converged']]
                  .dropna(subset=['hf_energy']))
        hf_map = (hf_raw.groupby('basis')
                  .apply(lambda g: g.loc[g['bond_dim'].idxmax(), 'hf_energy'])
                  if not hf_raw.empty else pd.Series(dtype=float))
        sub['hf_energy']   = sub['basis'].map(hf_map)
        sub['corr_energy'] = sub['E_inf_explog2'] - sub['hf_energy']
        sub = sub.dropna(subset=['hf_energy'])

        if sub.empty:
            ax.set_visible(False)
            continue

        Xs    = sub['X'].values.astype(float)
        E_tot = sub['E_inf_explog2'].values
        E_hf  = sub['hf_energy'].values
        E_c   = sub['corr_energy'].values
        color  = GEO_COLORS.get(geo, 'steelblue')
        marker = GEO_MARKERS.get(geo, 'o')

        ax.scatter(Xs, E_tot, color=color, s=70, marker=marker,
                   zorder=5, label='BD-extrap energy')
        for x, e, b in zip(Xs, E_tot, sub['basis'].values):
            ax.annotate(b.replace('cc-p', '').replace('aug-', 'a'), (x, e),
                        textcoords='offset points', xytext=(5, 3), fontsize=7)

        fit_mask = np.isin(Xs.astype(int), fit_Xs)
        Xf, Ehf_f, Ec_f = Xs[fit_mask], E_hf[fit_mask], E_c[fit_mask]
        popt_hf = popt_c = None

        if len(Xf) >= 3:
            try:
                popt_hf, _ = curve_fit(hf_model, Xf, Ehf_f,
                                        p0=[Ehf_f.min() - 1e-4, 1., 1.],
                                        maxfev=10000)
            except Exception as e:
                print(f'  HF fit failed ({geo}): {e}')
        if len(Xf) >= 2:
            try:
                popt_c, _ = curve_fit(corr_model, Xf, Ec_f,
                                       p0=[Ec_f.min() - 1e-4, -0.1],
                                       maxfev=10000)
            except Exception as e:
                print(f'  Corr fit failed ({geo}): {e}')

        if popt_hf is not None and popt_c is not None:
            E_hf_cbs  = float(popt_hf[0])
            E_c_cbs   = float(popt_c[0])
            E_tot_cbs = E_hf_cbs + E_c_cbs
            cbs_rows.append({
                'geo_label':    geo,
                'basis_scheme': basis_scheme,
                'E_HF_CBS':     E_hf_cbs,
                'E_corr_CBS':   E_c_cbs,
                'E_total_CBS':  E_tot_cbs,
            })
            Xf2 = np.linspace(Xs.min() - 0.2, Xs.max() + 0.3, 200)
            ax.plot(Xf2,
                    hf_model(Xf2, *popt_hf) + corr_model(Xf2, *popt_c),
                    '--', color=color, lw=1.8, label='CBS fit')
            ax.axhline(E_tot_cbs, color=color, ls=':', lw=1.2,
                       label=f'E_CBS={E_tot_cbs:.6f} Ha')
            print(f'{molecule} {geo:5s}  E_CBS={E_tot_cbs:.6f}  '
                  f'(HF={E_hf_cbs:.6f}, corr={E_c_cbs:.6f})')

        for j, (ref_name, ref_val) in enumerate(LIT_REF.get(molecule, {}).items()):
            if geo == 'eq':
                ax.axhline(ref_val, color=LIT_COLORS[j % len(LIT_COLORS)],
                           ls='-.', lw=1.3, alpha=0.8,
                           label=f'{ref_name}={ref_val:.4f}')

        if _ref_e_geo is not None:
            ax.axhline(_ref_e_geo, color='black', linestyle='--', linewidth=1.8,
                       alpha=0.65, label=f'CBS ref (spline) = {_ref_e_geo:.5f}',
                       zorder=7)

        ax.set_xlabel('Cardinal number X')
        ax.set_ylabel('E (Ha)')
        ax.set_xticks(sorted(set(Xs.astype(int))))
        ax.set_title(f'{molecule} {geo}\nBD→∞ + CBS ({basis_scheme})', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')

    fig.suptitle(f'{molecule}: CBS from BD-extrapolated energies', fontsize=11)
    fig.tight_layout()
    plt.show()
    return pd.DataFrame(cbs_rows)


# ── CBSAnalyzer class (unchanged from original) ───────────────────────────────

class CBSAnalyzer:
    """
    CBS extrapolation analyzer supporting multiple methods and system types.

    System Types:
    - Diatomic (e.g., N2): Full CBS extrapolation when basis ladder exists
    - Hydrogen chains: Single-basis analysis or DMRG bond-dimension extrapolation
    """

    def __init__(self, source_file: Optional[str] = None,
                 source_dataframe: Optional[pd.DataFrame] = None):
        self.source_file = Path(source_file) if source_file else None
        self.data = self._load_data() if self.source_file else source_dataframe
        self.exact_bond_length = self.data.get('system_info', {}).get('exact_bond_length')
        self.system_info = self.data.get('system_info', {})
        self.global_system_name = self.system_info.get('system_name', 'Unknown')
        self.bond_lengths = [k for k in self.data.keys() if k != 'system_info']
        self.date_added = datetime.now().strftime('%Y-%m-%d')
        self._analyze_system_types()

    @classmethod
    def _merge_json_files(cls, file_paths) -> dict:
        """
        Merge an iterable of per-result JSON files into a single unified data dict.

        Each file should contain one (system, basis, method) result:
          {
            "system_info": {...},
            "<geometry_key>": { "<basis>": { "<method>": { "energy": ..., ... } } }
          }

        The merged dict collects all basis sets / methods for the same geometry
        key under one top-level entry.  Also:
          - Normalises "converged" -> "success" and
            "von_neumann_entropy" -> "von_neumann_entropy_mo" for compatibility.
          - Synthesises a standalone HF entry from hf_energy embedded in DMRG
            data (only when the DMRG calculation itself succeeded).

        Returns a dict with geometry keys plus "system_info" and "_system_spins".
        """
        merged: dict = {}
        system_info: Optional[dict] = None
        system_spins: dict = {}

        for json_file in sorted(Path(fp) for fp in file_paths):
            try:
                with open(json_file, 'r') as f:
                    file_data = json.load(f)
            except Exception as e:
                warnings.warn(f"Could not load {json_file.name}: {e}")
                continue

            file_si = file_data.get("system_info", {})
            if system_info is None:
                system_info = file_si

            for key, value in file_data.items():
                if key == "system_info":
                    continue
                if not isinstance(value, dict):
                    continue

                if key not in system_spins:
                    system_spins[key] = {
                        "spin":       file_si.get("spin"),
                        "spin_state": file_si.get("spin_state"),
                        "atom":       file_si.get("atom"),
                        "charge":     file_si.get("charge", 0),
                    }

                if key not in merged:
                    merged[key] = {}

                for basis, method_data in value.items():
                    if not isinstance(method_data, dict):
                        continue
                    if basis not in merged[key]:
                        merged[key][basis] = {}

                    for method, calc_data in method_data.items():
                        if not isinstance(calc_data, dict):
                            continue

                        if "converged" in calc_data and "success" not in calc_data:
                            calc_data["success"] = calc_data["converged"]
                        if "von_neumann_entropy" in calc_data and "von_neumann_entropy_mo" not in calc_data:
                            calc_data["von_neumann_entropy_mo"] = calc_data["von_neumann_entropy"]

                        merged[key][basis][method] = calc_data

                        dmrg_ok = (calc_data.get("success", calc_data.get("converged", False))
                                   and calc_data.get("energy") is not None)
                        if (method == "DMRG" and "hf_energy" in calc_data
                                and "HF" not in merged[key][basis] and dmrg_ok):
                            hf_calc_time = calc_data.get("orbital_info", {}).get("calculation_time")
                            merged[key][basis]["HF"] = {
                                "energy": calc_data["hf_energy"],
                                "success": True,
                                "converged": True,
                                "calculation_time": hf_calc_time,
                                "peak_memory_gb": None,
                                "error": None,
                            }

        merged["system_info"] = system_info if system_info is not None else {}
        merged["_system_spins"] = system_spins
        return merged

    @classmethod
    def from_directory(cls, directory: str, glob_pattern: str = "*.json") -> 'CBSAnalyzer':
        """
        Create a CBSAnalyzer from a directory of per-file JSON results.

        New output format stores one (system, basis, method) per JSON file:
          {
            "system_info": {..., "system_name": "C_triplet"},
            "C_triplet": { "aug-cc-pVDZ": { "DMRG": { "energy": ..., "hf_energy": ..., ... } } }
          }

        All matching files in the directory are merged into the unified format
        that the rest of CBSAnalyzer expects.  See _merge_json_files for details.
        To merge a hand-picked subset of files use from_files() instead.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        file_paths = [f for f in sorted(directory.glob(glob_pattern))
                      if f.parent == directory]
        merged = cls._merge_json_files(file_paths)
        merged["system_info"].setdefault("system_name", directory.name)

        instance = cls.__new__(cls)
        instance.source_file = directory
        instance.data = merged
        instance.system_info = merged.get("system_info", {})
        instance.global_system_name = instance.system_info.get("system_name", directory.name)
        instance.bond_lengths = [k for k in merged.keys()
                                  if k not in ("system_info", "_system_spins")]
        instance.date_added = datetime.now().strftime('%Y-%m-%d')
        instance._analyze_system_types()
        return instance

    @classmethod
    def from_files(cls, file_paths) -> 'CBSAnalyzer':
        """
        Create a CBSAnalyzer by merging a specific list of JSON files.

        Useful when you want to select a subset of files rather than loading an
        entire directory.
        """
        merged = cls._merge_json_files(list(file_paths))
        merged["system_info"].setdefault("system_name", "merged_files")

        instance = cls.__new__(cls)
        instance.source_file = None
        instance.data = merged
        instance.system_info = merged.get("system_info", {})
        instance.global_system_name = instance.system_info.get("system_name", "merged_files")
        instance.bond_lengths = [k for k in merged.keys()
                                  if k not in ("system_info", "_system_spins")]
        instance.date_added = datetime.now().strftime('%Y-%m-%d')
        instance._analyze_system_types()
        return instance

    def _load_data(self) -> dict:
        try:
            with open(self.source_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            warnings.warn(f"File not found at {self.source_file}. Using empty data.")
            return {"system_info": {"system_name": "Unknown"}}

    def _analyze_system_types(self):
        """Analyze what types of systems are present in the data."""
        self.system_types = {}
        self.diatomic_keys = []
        self.chain_keys = []

        for key in self.bond_lengths:
            geom = self.parse_geometry(key)
            system_type = geom.get('system_type', 'unknown')
            self.system_types[key] = system_type

            if system_type == 'diatomic':
                self.diatomic_keys.append(key)
            elif system_type == 'chain':
                self.chain_keys.append(key)

        print(f"\n{'='*60}")
        print(f"System Analysis Summary")
        print(f"{'='*60}")
        print(f"Total geometries: {len(self.bond_lengths)}")
        print(f"  - Diatomic systems: {len(self.diatomic_keys)}")
        print(f"  - Chain systems: {len(self.chain_keys)}")
        print(f"{'='*60}\n")

    def parse_geometry(self, key: str, bond_data: dict = None) -> Dict:
        """
        Parse geometry string for diatomic, single-chain, or double-chain formats.

        Examples:
        - Diatomic: 'R_1.09' (N2)
        - Single-chain: 'dA_0.80_n4' (H chain)
        - Double-chain: 'dA_0.74_dB_0.80_n4' (H2 chain)

        If bond_data is provided, checks for exact geometry metadata stored by
        DMRG_run_calculations.py ('_bond_length_ang', '_d_A_bohr') before
        falling back to regex key parsing.
        """
        # Prefer exact geometry metadata written by DMRG_run_calculations.py
        if bond_data is not None:
            if '_bond_length_ang' in bond_data:
                R = float(bond_data['_bond_length_ang'])
                return {
                    "dA": R, "dB": np.nan, "nreps": np.nan,
                    "system_type": "diatomic",
                    "system_name": f"R_{R:.6g}",
                    "bond_length_ang": R,
                    "bond_length_bohr": R / _BOHR_TO_ANG,
                    "units_orig": bond_data.get('_units', 'angstrom'),
                }
            if '_d_A_bohr' in bond_data:
                dA = float(bond_data['_d_A_bohr'])
                dB = float(bond_data['_d_B_bohr']) if '_d_B_bohr' in bond_data else np.nan
                m_n = re.search(r'_n(\d+)', key)
                nreps = int(m_n.group(1)) if m_n else np.nan
                _units_orig = bond_data.get('_units', 'bohr')
                if not np.isnan(dB):
                    bond_length = (dA + dB) / 2
                    return {
                        "dA": dA, "dB": dB, "nreps": nreps,
                        "system_type": "chain",
                        "system_name": (f"H{nreps}_dA{dA:.6g}_dB{dB:.6g}"
                                        if not np.isnan(nreps) else f"dA{dA:.6g}_dB{dB:.6g}"),
                        "bond_length_bohr": bond_length,
                        "bond_length_ang": bond_length * _BOHR_TO_ANG,
                        "units_orig": _units_orig,
                    }
                else:
                    return {
                        "dA": dA, "dB": np.nan, "nreps": nreps,
                        "system_type": "chain",
                        "system_name": (f"H{nreps}" if not np.isnan(nreps) else f"dA{dA:.6g}"),
                        "bond_length_bohr": dA,
                        "bond_length_ang": dA * _BOHR_TO_ANG,
                        "units_orig": _units_orig,
                    }

        # Regex fallbacks — infer units from system_info when bond_data metadata is absent
        _si_units = self.system_info.get('units', 'angstrom') if hasattr(self, 'system_info') else 'angstrom'

        # Diatomic pattern (e.g., R_1.09 for N2)
        m_diatomic = re.match(r"R_(?P<R>[\d\.]+)", key)
        if m_diatomic:
            R = float(m_diatomic.group("R"))
            if _si_units == 'bohr':
                bl_ang, bl_bohr = R * _BOHR_TO_ANG, R
            else:
                bl_ang, bl_bohr = R, R / _BOHR_TO_ANG
            return {
                "dA": R, "dB": np.nan, "nreps": np.nan,
                "system_type": "diatomic",
                "system_name": f"R_{R:.2f}",
                "bond_length_ang": bl_ang,
                "bond_length_bohr": bl_bohr,
                "units_orig": _si_units,
            }

        # Double-chain pattern (keys use bohr by convention)
        m_double = re.match(r"dA_(?P<dA>[\d\.]+)_dB_(?P<dB>[\d\.]+)_n(?P<nreps>\d+)", key)
        if m_double:
            dA = float(m_double.group("dA"))
            dB = float(m_double.group("dB"))
            nreps = int(m_double.group("nreps"))
            bl = (dA + dB) / 2
            return {
                "dA": dA, "dB": dB, "nreps": nreps,
                "system_type": "chain",
                "system_name": f"H{nreps}_dA{dA:.2f}_dB{dB:.2f}",
                "bond_length_bohr": bl,
                "bond_length_ang": bl * _BOHR_TO_ANG,
                "units_orig": "bohr",
            }

        # Single-chain pattern (keys use bohr by convention)
        m_single = re.match(r"dA_(?P<dA>[\d\.]+)_n(?P<nreps>\d+)", key)
        if m_single:
            dA = float(m_single.group("dA"))
            nreps = int(m_single.group("nreps"))
            return {
                "dA": dA, "dB": np.nan, "nreps": nreps,
                "system_type": "chain",
                "system_name": f"H{nreps}",
                "bond_length_bohr": dA,
                "bond_length_ang": dA * _BOHR_TO_ANG,
                "units_orig": "bohr",
            }

        # Atomic pattern (e.g., "C_triplet", "O_triplet", "N_quartet")
        m_atomic = re.match(r"^(?P<atom>[A-Z][a-z]?)_(?P<state>\w+)$", key)
        if m_atomic:
            return {
                "dA": np.nan, "dB": np.nan, "nreps": np.nan,
                "system_type": "atomic",
                "system_name": key,
                "bond_length_ang": 0.0,
                "bond_length_bohr": 0.0,
                "units_orig": "N/A",
            }

        # Fallback
        try:
            bond_length = self.parse_bond_length(key)
            if _si_units == 'bohr':
                bl_ang, bl_bohr = bond_length * _BOHR_TO_ANG, bond_length
            else:
                bl_ang, bl_bohr = bond_length, bond_length / _BOHR_TO_ANG
            return {
                "dA": bond_length, "dB": np.nan, "nreps": np.nan,
                "system_type": "unknown",
                "system_name": f"R_{bond_length:.2f}",
                "bond_length_ang": bl_ang,
                "bond_length_bohr": bl_bohr,
                "units_orig": _si_units,
            }
        except Exception:
            raise ValueError(f"Could not parse geometry from '{key}'")

    def parse_bond_length(self, key: str) -> float:
        """Extract the first numeric value from a bond-length key."""
        try:
            return float(key)
        except Exception:
            pass
        m = re.search(r"[-+]?\d*\.\d+|\d+", key)
        if m:
            return float(m.group())
        raise ValueError(f"Could not extract numeric bond length from key: {key!r}")

    def _detect_available_basis_sets(self, bond_data: dict) -> dict:
        """Detect which basis sets are available in the data."""
        basis_map = {}
        prefixes = ["", "aug-"]
        for prefix in prefixes:
            for key, short in zip(["DZ", "TZ", "QZ", "5Z"], ["DZ", "TZ", "QZ", "5Z"]):
                name = f"{prefix}cc-pV{short}"
                if name in bond_data:
                    basis_map[f"{'a' if prefix == 'aug-' else ''}{short}"] = name
        return basis_map

    def _extrapolate_hf_exponential(self, energies: list, X_values: list) -> Tuple[float, dict]:
        """Extrapolate HF energy using: E_HF(X) = E_HF(CBS) + a*exp(-b*X)"""
        energies = np.array(energies)
        X_values = np.array(X_values)
        try:
            p0 = [energies[-1], energies[0] - energies[-1], 1.0]
            popt, _ = curve_fit(hf_model, X_values, energies, p0=p0, maxfev=10000)
            E_cbs, a, b = popt
            residuals = energies - hf_model(X_values, *popt)
            rmse = np.sqrt(np.mean(residuals**2))
            return E_cbs, {'a': a, 'b': b, 'rmse': rmse, 'fit_method': 'exponential', 'converged': True}
        except Exception as e:
            warnings.warn(f"HF exponential fit failed: {e}. Using largest basis value.")
            return energies[-1], {'fit_method': 'fallback', 'converged': False, 'error': str(e)}

    def _extrapolate_hf_halkier(self, energies: list, X_values: list) -> Tuple[float, dict]:
        """
        Two-point HF/CBS extrapolation using the Halkier et al. 1999 formula with
        fixed exponential constant B = 1.637 (ByteDance Supplementary Note 6.2, Eq. 4).

        Uses the last two points in X_values / energies so the formula is always
        applied to the two largest (most accurate) basis sets available.
        """
        energies = np.array(energies)
        X_values = np.array(X_values)
        # Use last two points (largest basis sets)
        E_n, E_m = float(energies[-2]), float(energies[-1])
        n,   m   = float(X_values[-2]), float(X_values[-1])
        try:
            E_cbs = hf_halkier_two_point(E_n, E_m, n, m)
            return E_cbs, {'fit_method': 'halkier', 'B': _HALKIER_B,
                           'n': n, 'm': m, 'converged': True}
        except Exception as e:
            warnings.warn(f"Halkier HF fit failed: {e}. Using largest basis value.")
            return float(energies[-1]), {'fit_method': 'fallback', 'converged': False, 'error': str(e)}

    def _extrapolate_correlation_power(self, hf_energies: list, total_energies: list,
                                       X_values: list) -> Tuple[float, dict]:
        """
        Extrapolate correlation energy using E_c(X) = E_c(CBS) + a·X^{-3}  [Helgaker 1997].
        Only the two-point analytical formula is implemented (validated for TZ/QZ pair).
        """
        hf_energies    = np.array(hf_energies)
        total_energies = np.array(total_energies)
        X_values       = np.array(X_values)
        corr_energies  = total_energies - hf_energies
        try:
            if len(X_values) == 2:
                X1, X2 = X_values
                E1, E2 = corr_energies
                a = (E1 - E2) / (X1**(-3) - X2**(-3))
                E_c_cbs = E2 - a * X2**(-3)
                rmse = np.nan
            else:
                E_c_cbs = a = rmse = None

            return E_c_cbs, {
                'a': a, 'rmse': rmse,
                'fit_method': f'X^-3 ({len(X_values)}-point)',
                'converged': True,
                'corr_energies': corr_energies.tolist(),
            }
        except Exception as e:
            warnings.warn(f"Correlation power fit failed: {e}. Using largest basis correlation.")
            return corr_energies[-1], {
                'fit_method': 'fallback', 'converged': False,
                'error': str(e), 'corr_energies': corr_energies.tolist(),
            }

    def _compute_filtered_bd_extrapolation(self, mdata: dict) -> dict | None:
        """
        Fit E(D) = E_inf + A·exp(-k·(log D)²) to the bond-dimension sweep data after
        applying a monotonicity filter.  Returns a dict or None if fit cannot be performed.
        """
        extrap_result = mdata.get('extrap_result') or {}
        ei   = extrap_result.get('extrapolation_info') or {}
        bds  = np.array(ei.get('extrap_bond_dims', []), dtype=float)
        eners = np.array(ei.get('extrap_eners',    []), dtype=float)
        if bds.size < 3:
            return None

        order = np.argsort(bds)
        bds, eners = bds[order], eners[order]
        mono_mask = np.zeros(len(bds), dtype=bool)
        running_min = np.inf
        for i in range(len(bds)):
            if eners[i] < running_min:
                running_min = eners[i]
                mono_mask[i] = True

        valid_bd = bds[mono_mask & (bds > 1)]
        valid_en = eners[mono_mask & (bds > 1)]
        if valid_bd.size < 3:
            return None

        def _model(D, E_inf, A, k):
            return E_inf + A * np.exp(-k * (np.log(D))**2)

        e_min = float(valid_en.min())
        try:
            p0 = [valid_en.min() - 1e-4, valid_en.max() - valid_en.min(), 0.5]
            bounds = ([-np.inf, 0, 0], [np.inf, np.inf, np.inf])
            popt, _ = curve_fit(_model, valid_bd, valid_en, p0=p0, bounds=bounds, maxfev=10000)
            return {'E_inf': float(popt[0]), 'A': float(popt[1]), 'k': float(popt[2]),
                    'E_min': e_min, 'n_mono': int(valid_bd.size)}
        except Exception:
            return {'E_inf': None, 'E_min': e_min, 'n_mono': int(valid_bd.size)}

    def extract_cbs_energies(self, use_two_part_extrapolation: bool = True,
                              include_chains: bool = True,
                              warn_on_missing_cbs: bool = True,
                              extrapolate_hf_independently: bool = True,
                              include_3pt_exponential: bool = True,
                              bd_energy_mode: str = "raw") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract CBS energies with system-type awareness.

        Args:
            use_two_part_extrapolation:  Use HF(exp) + Corr(X^-3) for correlated methods
            include_chains:              If False, skip hydrogen chains entirely
            warn_on_missing_cbs:         Print warnings when CBS cannot be performed
            extrapolate_hf_independently: Also extrapolate HF using exponential fit
            include_3pt_exponential:     Also include 3-point exponential extrapolation
            bd_energy_mode:              Which energy to use for DMRG in CBS:
                                           "raw"         – use the stored DMRG energy (default)
                                           "extrapolated" – use E_inf from exp-log² BD fit
                                           "min_energy"  – use the lowest observed energy in
                                                           the monotonic BD sweep

        Returns:
            (cbs_results_df, individual_basis_df)
        """
        cbs_results       = []
        individual_results = []
        systems_without_cbs = []

        _system_spins = self.data.get("_system_spins", {})

        for bond_length_key, bond_data in self.data.items():
            if bond_length_key in ("system_info", "_system_spins"):
                continue

            geom        = self.parse_geometry(bond_length_key, bond_data=bond_data)
            bl_bohr     = geom["bond_length_bohr"]
            bl_ang      = geom.get("bond_length_ang", bl_bohr * _BOHR_TO_ANG)
            units_orig  = geom.get("units_orig", "unknown")
            system_type = geom.get("system_type", "unknown")
            system_name = geom.get("system_name", bond_length_key)
            n_reps      = geom.get("nreps")

            _spin_meta = _system_spins.get(bond_length_key, {})
            spin       = _spin_meta.get("spin")
            spin_state = _spin_meta.get("spin_state")

            if not include_chains and system_type == "chain":
                continue

            available_basis = self._detect_available_basis_sets(bond_data)
            cbs_blockers = []
            if len(available_basis) < 2:
                cbs_blockers.append(
                    f"Only {len(available_basis)} basis set(s) available — need at least 2 for CBS")

            methods   = ["HF", "FCI", "DMRG", "CCSDT", "MP2"]
            extracted = {m: {} for m in methods}
            calc_times = {m: {} for m in methods}
            peak_memory = {m: {} for m in methods}
            entropies   = {m: {} for m in methods}

            for basis_key, basis_name in available_basis.items():
                for method in ["HF", "FCI", "DMRG", "CCSDT"]:
                    if method not in bond_data.get(basis_name, {}):
                        continue
                    mdata = bond_data[basis_name][method]

                    _filtered_bd_extrap = None
                    if method == "DMRG":
                        _filtered_bd_extrap = self._compute_filtered_bd_extrapolation(mdata)

                    energy_val = mdata.get("energy")
                    if method == "DMRG" and bd_energy_mode != "raw" and _filtered_bd_extrap is not None:
                        if bd_energy_mode == "extrapolated" and _filtered_bd_extrap.get('E_inf') is not None:
                            energy_val = _filtered_bd_extrap['E_inf']
                        elif bd_energy_mode == "min_energy" and _filtered_bd_extrap.get('E_min') is not None:
                            energy_val = _filtered_bd_extrap['E_min']
                    extracted[method][basis_key] = energy_val

                    calc_times[method][basis_key]  = mdata.get("calculation_time")
                    peak_memory[method][basis_key] = mdata.get("peak_memory_gb")

                    try:
                        entropy_dict = mdata.get("von_neumann_entropy_mo") or mdata.get("von_neumann_entropy") or {}
                        entropies[method][basis_key] = (entropy_dict.get("entropy")
                                                         if isinstance(entropy_dict, dict) else None)
                    except Exception:
                        entropies[method][basis_key] = None

                    _dmrg_sel   = mdata.get("dmrg_selection_info") or {}
                    _extrap     = mdata.get("extrap_result") or {}
                    _extrap_info = _extrap.get("extrapolation_info") or {}
                    _extrap_bds = _extrap_info.get("extrap_bond_dims")

                    individual_results.append({
                        "source_file": str(self.source_file.name) if self.source_file else "N/A",
                        "system_name": system_name, "system_type": system_type,
                        "spin": spin, "spin_state": spin_state,
                        "bond_length_bohr": bl_bohr,
                        "bond_length_ang": bl_ang,
                        "units_orig": units_orig,
                        "n_reps": n_reps if pd.notna(n_reps) else None,
                        "dA": geom.get("dA"),
                        "dB": geom.get("dB") if pd.notna(geom.get("dB")) else None,
                        "method": method, "basis": basis_name, "basis_key": basis_key,
                        "energy": mdata.get("energy"),
                        "calculation_time": mdata.get("calculation_time"),
                        "peak_memory_gb": mdata.get("peak_memory_gb"),
                        "entropy": entropies[method][basis_key],
                        "success": mdata.get("success", mdata.get("converged", False)),
                        "error": mdata.get("error"),
                        "unrestricted": self.system_info.get("unrestricted"),
                        "max_bond_dim": self.system_info.get("max_bond_dim"),
                        "energy_tol": self.system_info.get("energy_tol"),
                        "discard_tol": self.system_info.get("discard_tol"),
                        "active_space_method": mdata.get("active_space_method"),
                        "final_bond_dim": mdata.get("final_bond_dim"),
                        "sweeps_run": mdata.get("sweeps_run"),
                        "final_discard_weight": mdata.get("final_discard_weight"),
                        "initial_active_space_size": _dmrg_sel.get("initial_active_space_size"),
                        "final_active_space_size": _dmrg_sel.get("n_selected"),
                        "active_space_entropy_threshold": _dmrg_sel.get("entropy_threshold"),
                        "as_selection_bond_dim": _dmrg_sel.get("as_bond_dim"),
                        "as_selection_n_sweeps": _dmrg_sel.get("as_n_sweeps"),
                        "orbital_reorder_method": _dmrg_sel.get("reorder_method"),
                        "extrap_energy_dws": _extrap.get("extrapolated_energy_dws"),
                        "extrap_error_dws": _extrap.get("error_dws"),
                        "extrap_energy_bds": _extrap.get("extrapolated_energy_bds"),
                        "extrap_error_bds": _extrap.get("error_bds"),
                        "extrap_bond_dims": str(_extrap_bds) if _extrap_bds is not None else None,
                        "extrap_svd_cutoff": _extrap_info.get("svd_cutoff"),
                        "filtered_bond_dim_extrapolation": (_filtered_bd_extrap['E_inf']
                                                             if _filtered_bd_extrap is not None else None),
                        "filtered_bd_min_energy": (_filtered_bd_extrap['E_min']
                                                    if _filtered_bd_extrap is not None else None),
                    })

                    # MP2
                    if "mp2_total_energy" in mdata:
                        extracted["MP2"][basis_key] = mdata["mp2_total_energy"]
                        calc_times["MP2"][basis_key]  = mdata.get("calculation_time")
                        peak_memory["MP2"][basis_key] = mdata.get("peak_memory_gb")
                        individual_results.append({
                            "source_file": str(self.source_file.name) if self.source_file else "N/A",
                            "system_name": system_name, "system_type": system_type,
                            "spin": spin, "spin_state": spin_state,
                            "bond_length_bohr": bl_bohr,
                            "bond_length_ang": bl_ang,
                            "units_orig": units_orig,
                            "n_reps": n_reps if pd.notna(n_reps) else None,
                            "dA": geom.get("dA"),
                            "dB": geom.get("dB") if pd.notna(geom.get("dB")) else None,
                            "method": "MP2", "basis": basis_name, "basis_key": basis_key,
                            "energy": mdata["mp2_total_energy"],
                            "calculation_time": mdata.get("calculation_time"),
                            "peak_memory_gb": mdata.get("peak_memory_gb"),
                            "entropy": None,
                            "success": mdata.get("success", mdata.get("converged", False)),
                            "error": mdata.get("error"),
                            **{k: None for k in [
                                "unrestricted", "max_bond_dim", "energy_tol", "discard_tol",
                                "active_space_method", "final_bond_dim", "sweeps_run",
                                "final_discard_weight", "initial_active_space_size",
                                "final_active_space_size", "active_space_entropy_threshold",
                                "as_selection_bond_dim", "as_selection_n_sweeps",
                                "orbital_reorder_method", "extrap_energy_dws", "extrap_error_dws",
                                "extrap_energy_bds", "extrap_error_bds", "extrap_bond_dims",
                                "extrap_svd_cutoff", "filtered_bond_dim_extrapolation",
                                "filtered_bd_min_energy",
                            ]},
                        })

            # Determine extrapolation schemes
            schemes = []
            for keys, card_nums, name in [
                (['DZ', 'TZ'], [2, 3], 'DZ/TZ'),
                (['DZ', 'TZ', 'QZ'], [2, 3, 4], 'DZ/TZ/QZ'),
                (['TZ', 'QZ'],       [3, 4],    'TZ/QZ'),
                (['TZ', 'QZ', '5Z'], [3, 4, 5], 'TZ/QZ/5Z'),
            ]:
                if all(k in available_basis for k in keys):
                    schemes.append({'basis_keys': keys, 'cardinal_nums': card_nums, 'name': name})
                a_keys = [f'a{k}' for k in keys]
                if all(k in available_basis for k in a_keys):
                    schemes.append({'basis_keys': a_keys, 'cardinal_nums': card_nums, 'name': 'aug-' + name})

            if len(schemes) == 0 and len(available_basis) >= 2:
                cbs_blockers.append(
                    f"Available basis sets {list(available_basis.values())} do not form a valid "
                    f"CBS ladder (need DZ/TZ/QZ, TZ/QZ, or TZ/QZ/5Z)")

            if cbs_blockers:
                systems_without_cbs.append({
                    'key': bond_length_key, 'system': system_name, 'type': system_type,
                    'n_basis': len(available_basis), 'basis': list(available_basis.values()),
                    'reason': '; '.join(cbs_blockers),
                })

            for scheme in schemes:
                basis_keys = scheme['basis_keys']
                X_vals     = scheme['cardinal_nums']
                scheme_name = scheme['name']

                for method in methods:
                    energies     = extracted[method]
                    times        = calc_times[method]
                    memory       = peak_memory[method]
                    entropy_vals = entropies[method]

                    if not all(k in energies and energies[k] is not None for k in basis_keys):
                        continue

                    E_vals     = [energies[k] for k in basis_keys]
                    time_vals  = [times.get(k) for k in basis_keys]
                    memory_vals = [memory.get(k) for k in basis_keys]

                    hf_energies_data = extracted["HF"]
                    hf_available = all(k in hf_energies_data and hf_energies_data[k] is not None
                                       for k in basis_keys)
                    HF_vals = [hf_energies_data[k] for k in basis_keys] if hf_available else None

                    note = ""
                    _base_row = {
                        "source_file": str(self.source_file.name) if self.source_file else "N/A",
                        "system_name": system_name, "system_type": system_type,
                        "spin": spin, "spin_state": spin_state,
                        "bond_length_bohr": bl_bohr,
                        "bond_length_ang": bl_ang,
                        "units_orig": units_orig,
                        "n_reps": n_reps if pd.notna(n_reps) else None,
                        "dA": geom.get("dA"),
                        "dB": geom.get("dB") if pd.notna(geom.get("dB")) else None,
                        "method": method, "basis_scheme": scheme_name,
                        **{f"E_{k}": energies.get(k) for k in basis_keys},
                        **{f"calc_time_{k}": times.get(k) for k in basis_keys},
                        **{f"peak_memory_{k}": memory.get(k) for k in basis_keys},
                        **{f"entropy_{k}": entropy_vals.get(k) for k in basis_keys},
                        "total_calc_time": sum(t for t in time_vals if t is not None),
                        "max_memory": max((m for m in memory_vals if m is not None), default=None),
                    }

                    if method == "HF":
                        if extrapolate_hf_independently:
                            E_CBS, hf_fit_info = self._extrapolate_hf_exponential(E_vals, X_vals)
                            note = "" if hf_fit_info['converged'] else "HF exponential fit failed"
                            cbs_results.append({
                                **_base_row,
                                "extrapolation_type": "hf_exponential",
                                "CBS_energy": E_CBS,
                                "CBS_source": f'hf_exp_{scheme_name}',
                                "HF_fit_a": hf_fit_info.get('a'),
                                "HF_fit_b": hf_fit_info.get('b'),
                                "HF_fit_rmse": hf_fit_info.get('rmse'),
                                "note": note,
                            })

                    # Halkier two-point scheme (ByteDance Supplementary Note 6.2, Eq. 4 & 5).
                    # HF component: fixed-B exponential (B=1.637, Halkier et al. 1999).
                    # Correlation component: X^{-3} analytical two-point (Helgaker 1997,
                    #   equivalent to Halkier Eq. 5).
                    # Applied whenever ≥ 2 basis sets are available and HF energies exist.
                    if method != "HF" and hf_available and len(X_vals) >= 2:
                        E_HF_CBS_hk, hf_hk_info = self._extrapolate_hf_halkier(HF_vals, X_vals)
                        # Use last two points for the correlation two-point formula
                        X2 = X_vals[-2:]
                        E_c_CBS_hk, corr_hk_info = self._extrapolate_correlation_power(
                            HF_vals[-2:], E_vals[-2:], X2)
                        E_CBS_hk = E_HF_CBS_hk + E_c_CBS_hk
                        note_hk = ("" if hf_hk_info['converged'] and corr_hk_info['converged']
                                   else "Extrapolation issues")
                        cbs_results.append({
                            **_base_row,
                            "extrapolation_type": "halkier_two_part",
                            **{f"HF_{k}": HF_vals[i] for i, k in enumerate(basis_keys)},
                            **{f"Corr_{k}": E_vals[i] - HF_vals[i] for i, k in enumerate(basis_keys)},
                            "E_HF_CBS": E_HF_CBS_hk, "E_corr_CBS": E_c_CBS_hk,
                            "CBS_energy": E_CBS_hk,
                            "CBS_source": f'halkier_two_part_{scheme_name}',
                            "HF_fit_a": None, "HF_fit_b": _HALKIER_B,
                            "HF_fit_rmse": None,
                            "Corr_fit_a": corr_hk_info.get('a'),
                            "Corr_fit_rmse": corr_hk_info.get('rmse'),
                            "note": note_hk,
                        })

                    elif use_two_part_extrapolation and hf_available and X_vals == [3, 4]:
                        E_HF_CBS, hf_fit_info  = self._extrapolate_hf_exponential(HF_vals, X_vals)
                        E_c_CBS, corr_fit_info = self._extrapolate_correlation_power(HF_vals, E_vals, X_vals)
                        E_CBS = E_HF_CBS + E_c_CBS
                        note_tp = ("" if hf_fit_info['converged'] and corr_fit_info['converged']
                                   else "Extrapolation fit issues")
                        cbs_results.append({
                            **_base_row,
                            "extrapolation_type": "two_part",
                            **{f"HF_{k}": HF_vals[i] for i, k in enumerate(basis_keys)},
                            **{f"Corr_{k}": E_vals[i] - HF_vals[i] for i, k in enumerate(basis_keys)},
                            "E_HF_CBS": E_HF_CBS, "E_corr_CBS": E_c_CBS,
                            "CBS_energy": E_CBS, "CBS_source": f'two_part_{scheme_name}',
                            "HF_fit_a": hf_fit_info.get('a'), "HF_fit_b": hf_fit_info.get('b'),
                            "HF_fit_rmse": hf_fit_info.get('rmse'),
                            "Corr_fit_a": corr_fit_info.get('a'),
                            "Corr_fit_rmse": corr_fit_info.get('rmse'),
                            "note": note_tp,
                        })

                    if include_3pt_exponential and len(E_vals) == 3:
                        if not all(E_vals[i] > E_vals[i + 1] for i in range(len(E_vals) - 1)):
                            note_3pt = "Energies not decreasing with basis size"
                            E_CBS_3pt = E_vals[-1]
                            src_3pt = f"{basis_keys[-1]}_fallback_3pt"
                        else:
                            _cbs_3pt = cbs_3pt_algebraic(E_vals[0], E_vals[1], E_vals[2])
                            if _cbs_3pt is None:
                                E_CBS_3pt = E_vals[2]
                                src_3pt   = f'X{X_vals[2]}_fallback_exp'
                            else:
                                E_CBS_3pt = _cbs_3pt
                                src_3pt   = f'3pt_exp_{scheme_name}'
                            note_3pt = ""

                        result_3pt = {
                            **_base_row,
                            "extrapolation_type": "3pt_exp_total" if "3pt_exp" in src_3pt else "fallback",
                            "CBS_energy": E_CBS_3pt, "CBS_source": src_3pt,
                            "note": note_3pt,
                        }
                        if hf_available:
                            result_3pt.update({
                                **{f"HF_{k}": HF_vals[i] for i, k in enumerate(basis_keys)},
                                **{f"Corr_{k}": E_vals[i] - HF_vals[i] for i, k in enumerate(basis_keys)},
                            })
                        cbs_results.append(result_3pt)

                    if len(E_vals) >= 2:
                        E_CBS_exp, exp_fit_info = self._extrapolate_hf_exponential(E_vals, X_vals)
                        note_exp = "" if exp_fit_info['converged'] else "Exponential fit failed"
                        result_exp = {
                            **_base_row,
                            "extrapolation_type": "exp_total_fit" if exp_fit_info['converged'] else "fallback",
                            "CBS_energy": E_CBS_exp,
                            "CBS_source": (f'exp_fit_{scheme_name}' if exp_fit_info['converged']
                                           else f'{basis_keys[-1]}_fallback_exp_fit'),
                            "exp_fit_a": exp_fit_info.get('a'),
                            "exp_fit_b": exp_fit_info.get('b'),
                            "exp_fit_rmse": exp_fit_info.get('rmse'),
                            "note": note_exp,
                        }
                        if hf_available:
                            result_exp.update({
                                **{f"HF_{k}": HF_vals[i] for i, k in enumerate(basis_keys)},
                                **{f"Corr_{k}": E_vals[i] - HF_vals[i] for i, k in enumerate(basis_keys)},
                            })
                        cbs_results.append(result_exp)

        if warn_on_missing_cbs and systems_without_cbs:
            print(f"\n{'='*70}")
            print(f"⚠️  Systems WITHOUT CBS Extrapolation ({len(systems_without_cbs)} total)")
            print(f"{'='*70}")
            for s in systems_without_cbs:
                print(f"\nSystem: {s['key']}")
                print(f"  Type: {s['type']}")
                print(f"  Name: {s['system']}")
                print(f"  Available basis sets: {s['basis']}")
                print(f"  ❌ CBS blocked: {s['reason']}")
            print(f"{'='*70}\n")

        return pd.DataFrame(cbs_results), pd.DataFrame(individual_results)

    def extract_all_cbs_methods(self, bd_energy_mode: str = "raw") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract CBS energies using ALL methods for comparison.

        Args:
            bd_energy_mode: Which energy to use for DMRG in CBS — "raw", "extrapolated",
                            or "min_energy". See extract_cbs_energies() for details.
        """
        df_two_part_cbs, df_two_part_indiv = self.extract_cbs_energies(
            use_two_part_extrapolation=True,
            bd_energy_mode=bd_energy_mode)
        df_original_cbs, _ = self.extract_cbs_energies(
            use_two_part_extrapolation=False,
            bd_energy_mode=bd_energy_mode)

        combined_cbs_df = pd.concat([df_two_part_cbs, df_original_cbs], ignore_index=True)
        return combined_cbs_df, df_two_part_indiv

    def extract_dmrg_convergence(self) -> pd.DataFrame:
        """Extract DMRG convergence data (bond dimension extrapolation)."""
        results = []

        for bond_length_key, bond_data in self.data.items():
            if bond_length_key in ("system_info", "_system_spins"):
                continue

            geom        = self.parse_geometry(bond_length_key, bond_data=bond_data)
            bl_float    = geom["bond_length_bohr"]
            system_name = geom["system_name"]
            system_type = geom["system_type"]

            for basis, method_data in bond_data.items():
                if basis == "system_info" or "DMRG" not in method_data:
                    continue

                dmrg_data    = method_data["DMRG"]
                extrap_result = dmrg_data.get('extrap_result')
                dmrg_info    = dmrg_data.get('DMRG_extrapolation_info') or {}

                def _row(bond_dim, energy_val, dws_val, e_dws, e_bds):
                    if isinstance(energy_val, (list, np.ndarray)):
                        energy_val = energy_val[0]
                    return {
                        'source_file': str(self.source_file.name) if self.source_file else "N/A",
                        'system_name': system_name, 'system_type': system_type,
                        'bond_length_bohr': bl_float, 'basis': basis,
                        'bond_dim': bond_dim, 'energy': float(energy_val),
                        'dws': float(dws_val) if dws_val is not None else float('nan'),
                        'extrap_energy_dws': e_dws, 'extrap_energy_bds': e_bds,
                    }

                if extrap_result:
                    try:
                        ei = extrap_result.get('extrapolation_info', {})
                        e_dws = extrap_result.get('extrapolated_energy_dws')
                        e_bds = extrap_result.get('extrapolated_energy_bds')
                        for bd, dw, en in zip(ei.get('extrap_bond_dims', []),
                                               ei.get('extrap_dws', []),
                                               ei.get('extrap_eners', [])):
                            results.append(_row(bd, en, dw, e_dws, e_bds))
                    except Exception as e:
                        warnings.warn(f"Error processing extrap_result at {bl_float}, {basis}: {e}")

                elif dmrg_info and 'bond_dims_used' in dmrg_info:
                    try:
                        extrapolated = dmrg_data.get('extrapolated_energy')
                        for i, bond_dim in enumerate(dmrg_info['bond_dims_used']):
                            results.append(_row(
                                bond_dim, dmrg_info['eners'][i],
                                dmrg_info['dws'][i], extrapolated, None))
                    except Exception as e:
                        warnings.warn(f"Error processing DMRG_extrapolation_info at {bl_float}, {basis}: {e}")

                elif dmrg_data.get('energy_history') and dmrg_data.get('bond_dim_history'):
                    try:
                        for en, bd, dw in zip(
                                dmrg_data['energy_history'],
                                dmrg_data['bond_dim_history'],
                                dmrg_data.get('discard_history', [None] * len(dmrg_data['energy_history']))):
                            results.append(_row(bd, en, dw, None, None))
                    except Exception as e:
                        warnings.warn(f"Error processing DMRG sweep history at {bl_float}, {basis}: {e}")

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('dws').groupby(
                ['system_name', 'basis', 'bond_dim', 'bond_length_bohr'], as_index=False
            ).last()
        return df

    # ── Plotting methods (kept from original CBSAnalyzer) ─────────────────────

    def plot_cbs_curves(self, cbs_df: pd.DataFrame = None, xlims=None, ylims=None,
                        methods: list = None, basis_sets: list = None, cbs_sources=None):
        """CBS energy vs bond length. Color = Method, separate subplots per CBS_source."""
        if cbs_df is None:
            cbs_df, _ = self.extract_cbs_energies()
        methods = methods or ['HF', 'FCI', 'DMRG', 'CCSDT', 'MP2']
        if cbs_sources is None:
            cbs_sources = sorted(cbs_df['CBS_source'].unique())
        else:
            cbs_sources = [src for src in cbs_sources if src in cbs_df['CBS_source'].unique()]

        n_plots = len(cbs_sources)
        if n_plots == 0:
            print("No data to plot")
            return

        _x_col = 'bond_length_bohr'
        _x_label = "Bond Length (Bohr)"
        fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6), squeeze=False)
        for idx, source in enumerate(cbs_sources):
            ax = axes.flatten()[idx]
            data_source = cbs_df[cbs_df['CBS_source'] == source].sort_values(_x_col)
            for method in methods:
                data_method = data_source[data_source['method'] == method]
                if data_method.empty:
                    continue
                linestyle = '--' if any(data_method['basis_scheme'].str.startswith('aug-')) else '-'
                ax.plot(data_method[_x_col], data_method['CBS_energy'],
                        marker=METHOD_MARKERS[method], linestyle=linestyle,
                        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                        color=METHOD_COLORS[method], label=method, alpha=0.8)
            if xlims: ax.set_xlim(xlims)
            if ylims: ax.set_ylim(ylims)
            ax.set_xlabel(_x_label, fontsize=12)
            ax.set_ylabel("Energy (Hartree)", fontsize=12)
            ax.set_title(f"CBS Source: {source}", fontsize=14, fontweight='bold')
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)
        plt.suptitle(f"CBS Extrapolated Energies by Source - {self.global_system_name}",
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    def plot_dmrg_convergence_subplots(self, dmrg_df: pd.DataFrame = None,
                                        ncols: int = 4, x_axis: str = 'bond_dim',
                                        basis_sets: list = None, ylims: tuple = None):
        """DMRG energy vs bond dimension. One figure per (system, bond_length), one subplot per basis."""
        if dmrg_df is None:
            dmrg_df = self.extract_dmrg_convergence()
        if dmrg_df.empty:
            print("No DMRG convergence data available to plot.")
            return

        _basis_order = ['cc-pVDZ', 'cc-pVTZ', 'cc-pVQZ', 'cc-pV5Z',
                        'aug-cc-pVDZ', 'aug-cc-pVTZ', 'aug-cc-pVQZ', 'aug-cc-pV5Z']
        if basis_sets is None:
            available = dmrg_df['basis'].unique()
            basis_sets = ([b for b in _basis_order if b in available] +
                          sorted(b for b in available if b not in _basis_order))

        sys_bl_pairs = (dmrg_df[['system_name', 'bond_length_bohr']].drop_duplicates()
                         .sort_values(['system_name', 'bond_length_bohr']))

        for _, (sys_name, bond_length) in sys_bl_pairs.iterrows():
            sys_data = dmrg_df[
                (dmrg_df['system_name'] == sys_name) &
                (dmrg_df['bond_length_bohr'] == bond_length) &
                (dmrg_df['energy'] <= 0) &
                (dmrg_df['energy'] >= -1000)
            ]
            if sys_data.empty:
                continue

            present_bases = [b for b in basis_sets if b in sys_data['basis'].unique()]
            if not present_bases:
                continue

            nrows = (len(present_bases) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(5 * ncols, 4 * nrows), squeeze=False)
            axes_flat = axes.flatten()

            for i, basis in enumerate(present_bases):
                ax = axes_flat[i]
                basis_data = sys_data[sys_data['basis'] == basis].sort_values(x_axis)
                if basis_data.empty:
                    ax.set_title(f'{basis}\n(no data)', fontsize=11)
                    ax.axis('off')
                    continue

                ls = '--' if basis.startswith('aug-') else '-'
                ax.plot(basis_data[x_axis], basis_data['energy'],
                        marker='o', linestyle=ls, color='steelblue',
                        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                        label='DMRG energy', alpha=0.9)

                if 'extrap_energy_bds' in basis_data.columns:
                    e_bds = basis_data['extrap_energy_bds'].dropna()
                    if not e_bds.empty:
                        ax.axhline(y=e_bds.iloc[0], linestyle='--',
                                   color='darkorange', linewidth=1.5,
                                   label=f'extrap (BDS) = {e_bds.iloc[0]:.6f}')

                if x_axis == 'bond_dim':
                    bd_vals = basis_data['bond_dim'].values.astype(float)
                    en_vals = basis_data['energy'].values.astype(float)
                    mono_mask = np.zeros(len(bd_vals), dtype=bool)
                    running_min = np.inf
                    for _i in range(len(bd_vals)):
                        if en_vals[_i] < running_min:
                            running_min = en_vals[_i]
                            mono_mask[_i] = True
                    excluded_bd = bd_vals[~mono_mask]
                    excluded_en = en_vals[~mono_mask]
                    if excluded_bd.size > 0:
                        ax.scatter(excluded_bd, excluded_en, marker='x', color='firebrick',
                                   s=60, zorder=5, linewidths=1.8, label='excluded (non-monotone)')
                    _ei = {'extrap_bond_dims': bd_vals.tolist(), 'extrap_eners': en_vals.tolist()}
                    _bd_fit = self._compute_filtered_bd_extrapolation(
                        {'extrap_result': {'extrapolation_info': _ei}})
                    if _bd_fit is not None:
                        E_inf_fit = _bd_fit['E_inf']
                        def _model_exp_logD(D, E_inf, A, k):
                            return E_inf + A * np.exp(-k * (np.log(D))**2)
                        valid_bd = bd_vals[mono_mask & (bd_vals > 1)]
                        try:
                            bd_fine = np.linspace(valid_bd.min(), valid_bd.max() * 1.05, 200)
                            ax.plot(bd_fine,
                                    _model_exp_logD(bd_fine, E_inf_fit, _bd_fit['A'], _bd_fit['k']),
                                    linestyle='-', color='mediumseagreen', linewidth=1.5,
                                    label='exp-logD fit (filtered)', alpha=0.85)
                        except Exception:
                            pass
                        ax.axhline(y=E_inf_fit, linestyle=':',
                                   color='mediumseagreen', linewidth=1.5,
                                   label=f'E_inf = {E_inf_fit:.6f}')
                        if dmrg_df is not None:
                            dmrg_df.loc[basis_data.index, 'filtered_bond_dim_extrapolation'] = E_inf_fit

                ax.set_title(basis, fontsize=11, fontweight='bold')
                ax.set_xlabel(x_axis.replace('_', ' ').title(), fontsize=9)
                ax.set_ylabel('Energy (Hartree)', fontsize=9)
                ax.legend(fontsize=8, loc='best')
                ax.grid(True, alpha=0.3)
                if ylims is not None:
                    ax.set_ylim(ylims)

            for j in range(len(present_bases), len(axes_flat)):
                axes_flat[j].axis('off')

            fig.suptitle(f'{sys_name}  —  bond length = {bond_length} (borh)',
                         fontsize=14, fontweight='bold', y=1.01)
            plt.tight_layout()
            plt.show()


# ── Standalone sweep-convergence utility ──────────────────────────────────────

def plot_sweep_convergence_by_bd(data_dir, system_names=None, figsize=None):
    """
    Plot DMRG energy history during sweeps, colored by bond dimension.

    One figure per system. Rows = spin states (singlet/triplet),
    cols = basis sets (cc-pVDZ … cc-pV5Z).

    Args:
        data_dir:     path to directory containing *_DMRG_*_BD{N}.json files
        system_names: list of system names to include (e.g. ['C']); None = all
        figsize:      (width, height) override; None -> (5*ncols, 4*nrows)

    Returns:
        dict mapping system_name -> matplotlib Figure
    """
    import matplotlib.cm as cm

    CANONICAL_SPIN  = ["singlet", "triplet"]
    CANONICAL_BASIS = ["cc-pVDZ", "cc-pVTZ", "cc-pVQZ", "cc-pV5Z"]

    data_dir = Path(data_dir)
    records = {}

    for f in sorted(data_dir.glob("*_DMRG_*_BD*.json")):
        m = re.search(r'_BD(\d+)\.json$', f.name)
        if m is None:
            continue
        bond_dim = int(m.group(1))
        try:
            with open(f) as fh:
                data = json.load(fh)
        except Exception as e:
            warnings.warn(f"Could not load {f.name}: {e}")
            continue

        sys_info   = data.get("system_info", {})
        sys_name   = sys_info.get("atom", sys_info.get("system_name", "unknown"))
        spin_state = sys_info.get("spin_state", "unknown")

        if system_names is not None and sys_name not in system_names:
            continue

        data_keys = [k for k in data if k != "system_info"]
        if not data_keys:
            continue
        sys_key = data_keys[0]

        for basis, method_data in data[sys_key].items():
            if not isinstance(method_data, dict):
                continue
            dmrg           = method_data.get("DMRG", {})
            energy_history = dmrg.get("energy_history")
            if not energy_history:
                continue
            (records
             .setdefault(sys_name, {})
             .setdefault(spin_state, {})
             .setdefault(basis, {}))[bond_dim] = energy_history

    if not records:
        print("No sweep data found.")
        return {}

    figs = {}
    for sys_name, spin_records in sorted(records.items()):
        spin_states = [s for s in CANONICAL_SPIN if s in spin_records]
        all_bases   = {b for sr in spin_records.values() for b in sr}
        bases       = [b for b in CANONICAL_BASIS if b in all_bases]
        sys_bds     = sorted({bd for sr in spin_records.values()
                               for br in sr.values() for bd in br})
        n_bds = len(sys_bds)
        if not spin_states or not bases:
            continue

        cmap_v      = cm.viridis
        bd_to_color = {bd: cmap_v(i / max(n_bds - 1, 1)) for i, bd in enumerate(sys_bds)}

        nrows, ncols = len(spin_states), len(bases)
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=figsize or (5 * ncols, 4 * nrows),
                                  squeeze=False)
        for row_idx, spin in enumerate(spin_states):
            for col_idx, basis in enumerate(bases):
                ax      = axes[row_idx][col_idx]
                bd_data = spin_records.get(spin, {}).get(basis, {})
                if row_idx == 0:
                    ax.set_title(basis, fontsize=11, fontweight="bold")
                ax.set_xlabel("Sweep number", fontsize=9)
                if col_idx == 0:
                    ax.set_ylabel(f"{spin}\nEnergy (Ha)", fontsize=9)
                if not bd_data:
                    ax.axis("off")
                    continue
                for bd in sorted(bd_data):
                    eh     = bd_data[bd]
                    sweeps = list(range(1, len(eh) + 1))
                    ax.plot(sweeps, eh, color=bd_to_color[bd],
                            linewidth=1.5, marker="o", markersize=4, alpha=0.85)
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        from matplotlib.lines import Line2D
        legend_handles = [Line2D([0], [0], color=bd_to_color[bd], lw=2, label=str(bd))
                          for bd in sys_bds]
        fig.legend(handles=legend_handles, title='Bond dim',
                   loc='center left', bbox_to_anchor=(1.0, 0.5),
                   fontsize=8, title_fontsize=9, frameon=True)
        fig.suptitle(f"{sys_name} — DMRG sweep convergence by bond dimension",
                     fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        plt.show()
        figs[sys_name] = fig

    return figs

def plot_energy_bar_chart(plot_data, bond_length, system_name):
    """
    Plot a bar chart comparing CBS energies for different methods and basis schemes at a specific bond length.
    Args:        plot_data: DataFrame containing columns 'method', 'basis_scheme', and 'CBS_energy'.
                 bond_length: The bond length (in bohr) to include in the plot title.
                 system_name: The name of the system to include in the plot title.
    """
    
    plot_data = plot_data.drop_duplicates(subset=['method', 'basis_scheme', 'extrapolation_type'])
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = plot_data['method'] + '/' + plot_data['basis_scheme']
    energies = plot_data['CBS_energy']

    bars = ax.bar(methods, energies, color=['#4C9BE8', '#E8784C', '#4CE87A'], edgecolor='black', linewidth=0.8)

    padding = (energies.max() - energies.min()) * 0.5
    ax.set_ylim(energies.min() - padding, energies.max() + padding)

    for bar, val in zip(bars, energies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - (padding * 0.1),
            f'{val:.6f}',
            ha='center', va='top', fontsize=10, fontweight='bold', color='grey'
        )

    ax.set_ylabel('CBS Energy (Hartree)')
    ax.set_title(f'{system_name} CBS Energies for bond length {bond_length} bohr')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()