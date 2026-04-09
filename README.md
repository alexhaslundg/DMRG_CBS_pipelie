# dmrg_pipeline

Automated DMRG workflows built on [Block2](https://github.com/block-hczhai/block2-preview) and [PySCF](https://pyscf.org/). Handles the full pipeline from molecule setup through orbital preparation, active space selection, production DMRG, and CBS extrapolation — including cluster job management.

---

## What this does

- **Orbital construction** — HF or MP2 natural orbitals, with optional Pipek-Mezey / IBO localization
- **Active space selection** — occupation-based pre-selection followed by DMRG entropy-based refinement
- **Fiedler reordering** — optimizes orbital ordering for DMRG convergence via the interaction matrix
- **Production DMRG** — SU(2)-symmetric DMRG with configurable bond dimension schedule, noise, and SVD schedules
- **Backward extrapolation** — truncation-error extrapolation to the exact bond-dimension limit
- **CBS extrapolation** — complete basis set limit from cc-pVXZ or aug-cc-pCVXZ ladder calculations
- **Parallelization** — geometry/basis/method tasks distributed across cores via `multiprocessing.Pool`
- **Fault tolerance** — pre-extrapolation checkpointing so SLURM-killed jobs don't lose results

---

## Supported systems

| Type | Flag | Description |
|---|---|---|
| Atom | `--atomic` | Single atom (any spin) |
| Diatomic | `--diatomic` | A₂ or AB bond dissociation |
| Triatomic | `--triatomic` | Linear/bent triatomic |
| H4 | `--H4_mol` | Square/rectangular H4 |
| H chain | `--single_chain` | Uniform hydrogen chain |
| H double chain | `--double_chain` | Dimerized hydrogen chain |
| Custom XYZ | `--tme` | Arbitrary geometry from multi-frame XYZ file |

---

## Requirements

```
Python >= 3.9
pyscf
pyblock2       # see install note below
numpy
scipy
psutil
```

Optional (for analysis and plotting):
```
pandas
matplotlib
seaborn
```

### Installing Block2

Block2 must be installed from source or via pip with MPI support. See the [Block2 documentation](https://block2.readthedocs.io/en/latest/user/installation.html):

```bash
pip install pyblock2
# or for MPI support:
pip install pyblock2-mpi
```

---

## Installation

```bash
git clone https://github.com/your-org/dmrg_pipeline.git
cd dmrg_pipeline
pip install -e .
```

Or without installing (scripts work from the repo root):

```bash
pip install -r requirements.txt
```

---

## Quick start

### Minimal example (H4, runs in minutes)

```bash
python examples/h4/run_h4.py
```

Or in Python:

```python
from dmrg_pipeline.workflows.create_mol import create_H4
from dmrg_pipeline.workflows.calculate_energy import run_hf, run_dmrg

mol = create_H4(radius=1.738, angle=90.0, basis="sto-3g")

hf = run_hf(mol)
print(f"HF:   {hf['energy']:.8f} Ha")

dmrg = run_dmrg(mol, orbital_method="MP2", max_bond_dim=200)
print(f"DMRG: {dmrg['energy']:.8f} Ha")
```

### Active space selection with DMRG entropy

```python
from dmrg_pipeline.orbitals.orbital_prep import (
    construct_orbitals,
    select_active_space_with_DMRG,
    apply_selection_and_reordering,
)

orbitals, occupations, info, mf = construct_orbitals(mol, method="MP2")

selected_orbs, selected_idx, entropies, energy, dmrg_info = select_active_space_with_DMRG(
    mol=mol,
    mf=mf,
    orbitals=orbitals,
    occupations=occupations,
    initial_active_space_size=20,
    entropy_threshold=1e-3,
)

# Get final orbitals ready for production DMRG (selected + Fiedler-reordered)
final_orbs, final_occs, final_map = apply_selection_and_reordering(
    orbitals=orbitals,
    occupations=occupations,
    dmrg_info=dmrg_info,
)
```

### Running from the command line (diatomic, multiple basis sets)

```bash
python scripts/run_calculations.py \
    --diatomic \
    --atom1 N --atom2 N \
    --bond_lengths 1.0 1.1 1.2 1.3 \
    --list_of_basis_sets cc-pVDZ cc-pVTZ cc-pVQZ \
    --methods_to_run HF DMRG \
    --max_bond_dim 500 \
    --output_file_path ./results \
    --n_parallel_jobs 4
```

Results are saved as JSON files under `./results/`.

### SLURM cluster (example submission)

```bash
#!/bin/bash
#SBATCH --job-name=dmrg_n2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

python scripts/run_calculations.py \
    --diatomic --atom1 N --atom2 N \
    --bond_lengths 1.0 1.1 1.2 \
    --list_of_basis_sets cc-pVDZ cc-pVTZ \
    --methods_to_run DMRG \
    --max_bond_dim 1000 \
    --n_threads_per_process 4 \
    --n_parallel_jobs 4 \
    --output_file_path ./results
```

---

## Key DMRG parameters

| Parameter | Default | Description |
|---|---|---|
| `--orbital_method` | `HF` | Orbital basis: `HF` or `MP2` natural orbitals |
| `--max_bond_dim` | 400 | Maximum bond dimension |
| `--energy_tol` | 1e-6 | Convergence threshold (Ha) |
| `--use_dmrg_active_space_selection` | True | Use entropy-based active space selection |
| `--as_entropy_threshold` | 1e-3 | Entropy cutoff for orbital selection |
| `--perform_reordering` | True | Fiedler orbital reordering |
| `--perform_extrapolation` | True | Backward truncation-error extrapolation |

See `python scripts/run_calculations.py --help` for the full list.

---

## Project structure

```
dmrg_pipeline/
├── dmrg_pipeline/
│   ├── workflows/
│   │   ├── create_mol.py          # Molecule builders (diatomic, H-chains, TME, ...)
│   │   └── calculate_energy.py    # HF, FCI, CCSD(T), DMRG runners
│   ├── orbitals/
│   │   └── orbital_prep.py        # Orbital construction, active space selection, Fiedler
│   ├── analysis/
│   │   ├── cbs_extrapolate.py     # CBS extrapolation (HF + correlation)
│   │   └── combine_and_process.py # Post-processing and result aggregation
│   └── utils/
│       ├── utils.py               # Memory monitoring, JSON serialization helpers
│       └── basis_sets.py          # Hardcoded aug-cc-pCVXZ basis sets for N
├── examples/
│   └── h4/run_h4.py               # Minimal working example
├── scripts/
│   └── run_calculations.py        # Main CLI entry point
├── tests/
│   └── test_index_tracking.py     # Index tracking demonstration/test
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

---

## Notes on `cbs_extrapolate.py`

This module imports a `plotting_style` module from your local analysis environment. To use it standalone, either:

1. Copy your `plotting_style.py` into the repo, or
2. Remove the style imports and replace the `METHOD_COLORS` etc. references with plain matplotlib defaults.

The CBS extrapolation *functions themselves* (`hf_model`, `corr_model`, `cbs_3pt_algebraic`, etc.) have no dependency on `plotting_style` and work independently.

---

## Citation

If you use this pipeline in published work, please cite Block2 and PySCF:

- Zhai, H. & Chan, G. K.-L. *J. Chem. Phys.* **154**, 224116 (2021) — Block2
- Sun, Q. et al. *WIREs Comput. Mol. Sci.* **8**, e1340 (2018) — PySCF

---

## License

MIT — see [LICENSE](LICENSE).
