"""
Minimal example: H4 square molecule with DMRG.

Runs HF + DMRG on a small H4 system using a minimal basis.
Should complete in a few minutes on a laptop.

Usage:
    python examples/h4/run_h4.py
"""

import sys
from pathlib import Path

# Allow running directly without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dmrg_pipeline.workflows.create_mol import create_H4
from dmrg_pipeline.workflows.calculate_energy import run_hf, run_dmrg

# --- System parameters ---
R = 1.738        # H-H distance (Angstrom)
ANGLE = 90.0     # Square geometry
BASIS = "sto-3g"

print("=" * 50)
print(f"H4 square  R={R} Å  basis={BASIS}")
print("=" * 50)

mol = create_H4(radius=R, angle=ANGLE, basis=BASIS)

# --- Hartree-Fock ---
hf_result = run_hf(mol)
print(f"HF energy:   {hf_result['energy']:.8f} Ha")

# --- DMRG ---
dmrg_result = run_dmrg(
    mol=mol,
    orbital_method="MP2",
    max_bond_dim=200,
    max_sweeps=20,
    energy_tol=1e-6,
    verbose=1,
)
print(f"DMRG energy: {dmrg_result['energy']:.8f} Ha")
print(f"Converged:   {dmrg_result['converged']}")
