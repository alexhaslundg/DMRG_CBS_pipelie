import itertools
import re

import numpy as np
from pyscf import gto

from dmrg_pipeline.utils.basis_sets import CUSTOM_BASIS_MAP

# -------------------------
# Molecule creation from config
# -------------------------

def parse_atom_string(atom_str):
    """Parse atom string "Atom(symbol='X', coords=(x,y,z), ...)"."""
    symbol_match = re.search(r"symbol='(\w+)'", atom_str)
    if not symbol_match:
        raise ValueError(f"Cannot parse symbol from: {atom_str}")
    symbol = symbol_match.group(1)

    coords_match = re.search(r"coords=\(([-\d.e]+),\s*([-\d.e]+),\s*([-\d.e]+)\)", atom_str)
    if not coords_match:
        raise ValueError(f"Cannot parse coordinates from: {atom_str}")
    x, y, z = map(float, coords_match.groups())

    units_match = re.search(r"units='(\w+)'", atom_str)
    units = units_match.group(1) if units_match else 'bohr'
    return symbol, x, y, z, units

def create_molecule_from_config(config, basis='cc-pVDZ', symmetry=True):
    """Create PySCF molecule from configuration dictionary from the psiformer output."""
    if 'molecule' not in config:
        raise ValueError("Config must contain 'molecule' key")
    
    mol = gto.Mole()
    atom_list = []
    units = 'bohr'

    for atom_str in config['molecule']:
        symbol, x, y, z, atom_units = parse_atom_string(atom_str)
        atom_list.append([symbol, (x, y, z)])
        units = atom_units

    mol.atom = atom_list
    mol.unit = units
    mol.basis = config.get('basis', basis)
    mol.charge = config.get('charge', 0)
    mol.spin = config.get('spin', 0)
    mol.symmetry = symmetry
    mol.build()
    return mol

# -------------------------
# Generate Molecule Config
# -------------------------
def get_atom_charge(symbol):
    """Get nuclear charge for common atoms."""
    charges = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18
    }
    return charges.get(symbol, 0)

def get_default_electrons(atom1, atom2, atom3=None):
    """Get default electron configuration for neutral diatomic molecule."""
    total_electrons = get_atom_charge(atom1) + get_atom_charge(atom2)
    if atom3 is not None:
        total_electrons += get_atom_charge(atom3)
    # For neutral molecule, equal alpha and beta if even electrons, alpha+1 if odd
    if total_electrons % 2 == 0:
        return [total_electrons // 2, total_electrons // 2]
    else:
        return [total_electrons // 2 + 1, total_electrons // 2]

def create_diatomic_mol(atom1, atom2, bond_length, basis="sto-3g", charge=0, spin=None, units="angstrom"):
    """Create a PySCF molecule object for a diatomic molecule."""
    # Create geometry - place first atom at origin, second along z-axis
    geometry = [
        (atom1, (0, 0, 0)),
        (atom2, (0, 0, bond_length))
    ]
    
    # Calculate electrons if not specified by charge
    if charge == 0:
        electrons = get_default_electrons(atom1, atom2)
        total_electrons = sum(electrons)
    else:
        total_electrons = get_atom_charge(atom1) + get_atom_charge(atom2) - charge
        if total_electrons % 2 == 0:
            electrons = [total_electrons // 2, total_electrons // 2]
        else:
            electrons = [total_electrons // 2 + 1, total_electrons // 2]
    
    # Calculate spin if not specified
    if spin is None:
        spin = electrons[0] - electrons[1]  # alpha - beta
    
    print(f"Creating {atom1}-{atom2} at {bond_length} {units}: charge={charge}, spin={spin}, electrons={electrons}")
    mol = gto.Mole()

    # If using one of the custom CV-nZ sets, parse text block
    if basis in CUSTOM_BASIS_MAP:
        raw_basis = CUSTOM_BASIS_MAP[basis]
        parsed_basis = gto.parse(raw_basis)
        mol.build(
            atom=geometry,
            unit=units,
            charge=charge,
            spin=spin,
            basis=parsed_basis,
            symmetry=True,
            verbose=0,
        )
    else:
        # For built-in PySCF basis sets
        mol.build(
            atom=geometry,
            unit=units,
            charge=charge,
            spin=spin,
            basis=basis,
            symmetry=True,
            verbose=0,
        )

    
    return mol

def create_atomic_mol(atom, basis="cc-pVDZ", charge=0, spin=0, units="angstrom"):
    """Create a PySCF molecule object for a single atom.
    
    Parameters
    ----------
    atom : str
        Atom symbol (e.g., 'C', 'O', 'N')
    basis : str
        Basis set to use
    charge : int
        Charge of the atom
    spin : int
        Spin multiplicity - 1 (number of unpaired electrons).
        For singlet: spin=0, for triplet: spin=2
    units : str
        Units for coordinates (angstrom or bohr)
    
    Returns
    -------
    mol : pyscf.gto.Mole
        PySCF molecule object
    """
    # Single atom at origin
    geometry = [(atom, (0, 0, 0))]
    
    # Get number of electrons
    total_electrons = get_atom_charge(atom) - charge
    
    # Spin label for printing
    spin_label = "singlet" if spin == 0 else ("triplet" if spin == 2 else f"spin{spin}")
    
    print(f"Creating {atom} atom ({spin_label}): charge={charge}, spin={spin}, electrons={total_electrons}")
    
    mol = gto.Mole()
    
    # If using one of the custom CV-nZ sets, parse text block
    if basis in CUSTOM_BASIS_MAP:
        raw_basis = CUSTOM_BASIS_MAP[basis]
        parsed_basis = gto.parse(raw_basis)
        mol.build(
            atom=geometry,
            unit=units,
            charge=charge,
            spin=spin,
            basis=parsed_basis,
            symmetry=True,
            verbose=0,
        )
    else:
        # For built-in PySCF basis sets
        mol.build(
            atom=geometry,
            unit=units,
            charge=charge,
            spin=spin,
            basis=basis,
            symmetry=True,
            verbose=0,
        )
    
    return mol

def create_system_info_atomic(sample_task, args, base_system_name):
    """Create system_info dictionary for atomic systems."""
    spin_label = "singlet" if args.spin == 0 else ("triplet" if args.spin == 2 else f"spin{args.spin}")
    sample_mol = create_atomic_mol(
        atom=args.atom,
        basis=args.list_of_basis_sets[0],
        charge=args.charge,
        spin=args.spin
    )
    return {
        "geometry_type": "atomic",
        "atom": args.atom,
        "spin_state": spin_label,
        "charge": int(sample_mol.charge),
        "spin": int(sample_mol.spin),
        "basis_sets": args.list_of_basis_sets,
        "methods": args.methods_to_run,
        "system_name": base_system_name,
        "unrestricted": args.unrestricted,
        "mf_conv_tol": args.mf_conv_tol,
        "max_bond_dim": args.max_bond_dim,
        "active_space_orbitals": args.n_orbitals_for_initial_active_space,
        "noise_schedule": args.noise_schedule,
        "svd_schedule": args.svd_schedule,
        "davidson_schedule": args.davidson_schedule,
        "energy_tol": args.energy_tol,
        "discard_tol": args.discard_tol,
        "parallelization": {
            "n_parallel_jobs": args.n_parallel_jobs,
            "n_threads_per_process": args.n_threads_per_process
        }
    }

def create_system_info_diatomic(sample_task, args, base_system_name):
    """Create system_info dictionary for diatomic molecules."""
    sample_mol = create_diatomic_mol(
        atom1=args.atom1,
        atom2=args.atom2,
        bond_length=sample_task['geometry_params']['bond_length'],
        basis=args.list_of_basis_sets[0],
        charge=args.charge,
        spin=args.spin,
        units=args.units
    )
    return {
        "geometry_type": "diatomic",
        "units": args.units,
        "atom1": args.atom1,
        "atom2": args.atom2,
        "charge": int(sample_mol.charge),
        "spin": int(sample_mol.spin),
        "basis_sets": args.list_of_basis_sets,
        "methods": args.methods_to_run,
        "system_name": base_system_name,
        "unrestricted": args.unrestricted,
        "mf_conv_tol": args.mf_conv_tol,
        "max_bond_dim": args.max_bond_dim,
        "active_space_orbitals": args.n_orbitals_for_initial_active_space,
        "noise_schedule": args.noise_schedule,
        "svd_schedule": args.svd_schedule,
        "davidson_schedule": args.davidson_schedule,
        "energy_tol": args.energy_tol,
        "discard_tol": args.discard_tol,
        "parallelization": {
            "n_parallel_jobs": args.n_parallel_jobs,
            "n_threads_per_process": args.n_threads_per_process
        }
    }


# TRIATOMIC MOLECULES

def create_triatomic_mol(center_atom, atom1, atom2, bond_length, basis="sto-3g", charge=0, spin=None, units="angstrom"):
    """Create a PySCF molecule object for a linear, triatomic molecule."""
    # Create geometry - place first atom at origin, second along z-axis, third along y-axis
    geometry = [
        (center_atom, (0, 0, 0)),
        (atom1, (0, 0, bond_length)),
        (atom2, (0, 0, -bond_length))
    ]
    
    # Calculate electrons if not specified by charge
    if charge == 0:
        electrons = get_default_electrons(atom1, atom2, center_atom)
        total_electrons = sum(electrons)
    else:
        total_electrons = get_atom_charge(atom1) + get_atom_charge(atom2) - charge
        if total_electrons % 2 == 0:
            electrons = [total_electrons // 2, total_electrons // 2]
        else:
            electrons = [total_electrons // 2 + 1, total_electrons // 2]
    
    # Calculate spin if not specified
    if spin is None:
        spin = electrons[0] - electrons[1]  # alpha - beta

    
    # Build molecule
    mol = gto.Mole()
    mol.build(
        atom=geometry,
        unit=units,
        charge=charge,
        spin=spin,
        basis=basis
    )
    
    return mol


def create_system_info_triatomic(sample_task, args, base_system_name):
    """Create system_info dictionary for triatomic molecules."""
    sample_mol = create_triatomic_mol(
        center_atom=args.center_atom,
        atom1=args.atom1,
        atom2=args.atom2,
        bond_length=sample_task['geometry_params']['bond_length'],
        basis=args.list_of_basis_sets[0],
        units=args.units,
    )
    return {
        "geometry_type": "triatomic",
        "units": args.units,
        "center_atom": args.center_atom,
        "atom1": args.atom1,
        "atom2": args.atom2,
        "charge": int(sample_mol.charge),
        "spin": int(sample_mol.spin),
        "basis_sets": args.list_of_basis_sets,
        "methods": args.methods_to_run,
        "system_name": base_system_name,
        "mf_conv_tol": args.mf_conv_tol,
        "max_bond_dim": args.max_bond_dim,
        "active_space_orbitals": args.n_orbitals_for_initial_active_space,
        "noise_schedule": args.noise_schedule,
        "svd_schedule": args.svd_schedule,
        "davidson_schedule": args.davidson_schedule,
        "energy_tol": args.energy_tol,
        "discard_tol": args.discard_tol,
        "unrestricted": args.unrestricted,
        "parallelization": {
            "n_parallel_jobs": args.n_parallel_jobs,
            "n_threads_per_process": args.n_threads_per_process
        }
    }


# CREATE H4 MOLECULE


def create_H4(radius=1.738, angle=90, basis="sto-3g", charge=0, spin=None, units="angstrom"):

    t = np.radians(angle / 2)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    
    quadrants = itertools.product((1, -1), (1, -1))
    coords = [(i * x, j * y, 0.0) for i, j in quadrants]

    # Attach the hydrogen symbol to each coordinate
    geometry = [("H", c) for c in coords]

    print("Geometry (angstrom):")
    for g in geometry:
        print(g)

    mol = gto.Mole()
    mol.build(
        atom=geometry,
        unit=units,
        charge=charge,
        spin=0,
        basis=basis
    )
    return mol


def create_system_info_H4(sample_task, args, base_system_name):
    """Create system_info dictionary for H4 molecules."""
    sample_mol = create_H4(
        radius=sample_task['geometry_params']['radius'],
        angle=sample_task['geometry_params']['angle'],
        basis=args.list_of_basis_sets[0],
        units=args.units
    )
    return {
        "geometry_type": "H4",
        "units": args.units,
        "charge": int(sample_mol.charge),
        "spin": int(sample_mol.spin),
        "basis_sets": args.list_of_basis_sets,
        "methods": args.methods_to_run,
        "system_name": base_system_name,
        "radii_values": args.radii,
        "angle_values": args.angles,
        "mf_conv_tol": args.mf_conv_tol,
        "max_bond_dim": args.max_bond_dim,
        "active_space_orbitals": args.n_orbitals_for_initial_active_space,
        "noise_schedule": args.noise_schedule,
        "svd_schedule": args.svd_schedule,
        "davidson_schedule": args.davidson_schedule,
        "energy_tol": args.energy_tol,
        "discard_tol": args.discard_tol,
        "unrestricted": args.unrestricted,
        "parallelization": {
            "n_parallel_jobs": args.n_parallel_jobs,
            "n_threads_per_process": args.n_threads_per_process
        }
    }

# CREATE H SINGLE CHAIN MOLECULE

def create_H_single_chain(d_A=1.0, n_rep=4, basis="sto-3g", charge=0, spin=None, units="angstrom"):
    """
    Create a single chain of hydrogen atoms.
    
    Parameters:
    -----------
    d_A : float
        Distance between consecutive H atoms along each chain
    d_B : float
        Distance between the two parallel chains
    n_rep : int
        Number of H atoms in each chain
    basis : str
        Basis set for quantum chemistry calculation
    charge : int
        Total charge of the system
    spin : int or None
        Spin multiplicity (None = auto-detect)
    units : str
        Unit system ('angstrom' or 'bohr')
    
    Returns:
    --------
    mol : gto.Mole
        PySCF molecule object
    """
    
    geometry = []
    
    # First chain (y = +d_B/2)
    for i in range(n_rep):
        x = i * d_A
        y = 0.0
        geometry.append(("H", (x, y, 0.0)))
    
    print(f"Geometry ({units}):")
    for g in geometry:
        print(f"  {g[0]:2s}  {g[1][0]:8.4f}  {g[1][1]:8.4f}  {g[1][2]:8.4f}")
    
    mol = gto.Mole()
    mol.build(
        atom=geometry,
        unit=units,
        charge=charge,
        spin=spin if spin is not None else 0,
        basis=basis
    )
    return mol

def create_system_info_single_chain(sample_task, args, base_system_name):
    """Create system_info dictionary for single chain."""
    sample_mol = create_H_single_chain(
        d_A=sample_task['geometry_params']['d_A'],
        n_rep=sample_task['geometry_params']['n_rep'],
        basis=args.list_of_basis_sets[0],
        units=args.units
    )
    return {
        "geometry_type": "H_single_chain",
        "units": args.units,
        "charge": int(sample_mol.charge),
        "spin": int(sample_mol.spin),
        "basis_sets": args.list_of_basis_sets,
        "methods": args.methods_to_run,
        "system_name": base_system_name,
        "d_A_values": args.d_A,
        "n_reps_values": args.n_rep,
        "unrestricted": args.unrestricted,
        "mf_conv_tol": args.mf_conv_tol,
        "max_bond_dim": args.max_bond_dim,
        "active_space_orbitals": args.n_orbitals_for_initial_active_space,
        "noise_schedule": args.noise_schedule,
        "svd_schedule": args.svd_schedule,
        "davidson_schedule": args.davidson_schedule,
        "energy_tol": args.energy_tol,
        "discard_tol": args.discard_tol,
        "parallelization": {
            "n_parallel_jobs": args.n_parallel_jobs,
            "n_threads_per_process": args.n_threads_per_process
        }
    }



# CREATE H DOUBLE CHAIN MOLECULE
def create_H_double_chain(d_A=1.0, d_B=2.0, n_rep=4, basis="sto-3g", charge=0, spin=None, units="angstrom"):
    """
    Create a double chain of hydrogen atoms.
    
    Parameters:
    -----------
    d_A : float
        Distance between consecutive H atoms along each chain
    d_B : float
        Distance between the two parallel chains
    n_rep : int
        Number of H atoms in each chain
    basis : str
        Basis set for quantum chemistry calculation
    charge : int
        Total charge of the system
    spin : int or None
        Spin multiplicity (None = auto-detect)
    units : str
        Unit system ('angstrom' or 'bohr')
    
    Returns:
    --------
    mol : gto.Mole
        PySCF molecule object
    """
    
    geometry = []
    
    # First chain (y = +d_B/2)
    for i in range(n_rep):
        x = i * d_A
        y = d_B / 2
        geometry.append(("H", (x, y, 0.0)))
    
    # Second chain (y = -d_B/2)
    for i in range(n_rep):
        x = i * d_A
        y = -d_B / 2
        geometry.append(("H", (x, y, 0.0)))
    
    print(f"Geometry ({units}):")
    for g in geometry:
        print(f"  {g[0]:2s}  {g[1][0]:8.4f}  {g[1][1]:8.4f}  {g[1][2]:8.4f}")
    
    mol = gto.Mole()
    mol.build(
        atom=geometry,
        unit=units,
        charge=charge,
        spin=spin if spin is not None else 0,
        basis=basis
    )
    return mol
# def create_H_double_chain(radius=1.738, angle=90, n_rep=2, basis="sto-3g", charge=0, spin=None, units="angstrom"):
#     """This set up mirrors the Ferminet paper where the geometry is parameteriszed by the radius and the angle"""

#     t = np.radians(angle / 2)
#     x = radius * np.cos(t)
#     y = radius * np.sin(t)
    
#     quadrants = itertools.product((1, -1), (1, -1))
#     coords = [(i * x, j * y, 0.0) for i, j in quadrants]

#     # Attach the hydrogen symbol to each coordinate
#     geometry = [("H", c) for c in coords]

#     print("Geometry (angstrom):")
#     for g in geometry:
#         print(g)

#     mol = gto.Mole()
#     mol.build(
#         atom=geometry,
#         unit=units,
#         charge=charge,
#         spin=0,
#         basis=basis
#     )
#     return mol

def create_system_info_double_chain(sample_task, args, base_system_name):
    """Create system_info dictionary for double chain."""
    sample_mol = create_H_double_chain(
        d_A=sample_task['geometry_params']['d_A'],
        d_B=sample_task['geometry_params']['d_B'],
        n_rep=sample_task['geometry_params']['n_rep'],
        basis=args.list_of_basis_sets[0],
        units=args.units
    )
    return {
        "geometry_type": "H_double_chain",
        "units": args.units,
        "charge": int(sample_mol.charge),
        "spin": int(sample_mol.spin),
        "basis_sets": args.list_of_basis_sets,
        "methods": args.methods_to_run,
        "system_name": base_system_name,
        "d_A_values": args.d_A,
        "d_B_values": args.d_B,
        "n_reps_values": args.n_rep,
        "unrestricted": args.unrestricted,
        "mf_conv_tol": args.mf_conv_tol,
        "max_bond_dim": args.max_bond_dim,
        "active_space_orbitals": args.n_orbitals_for_initial_active_space,
        "noise_schedule": args.noise_schedule,
        "svd_schedule": args.svd_schedule,
        "davidson_schedule": args.davidson_schedule,
        "energy_tol": args.energy_tol,
        "discard_tol": args.discard_tol,
        "parallelization": {
            "n_parallel_jobs": args.n_parallel_jobs,
            "n_threads_per_process": args.n_threads_per_process
        }
    }


# CREATE TME MOLECULE FROM XYZ FILE

def parse_xyz_frames(xyz_file_path):
    """
    Parse a multi-frame xyz file into a dict keyed by dihedral angle.

    Returns
    -------
    dict : {angle_deg (float) -> atom_list}
        atom_list is a list of (symbol, (x, y, z)) tuples with coordinates
        as floats in the units of the original file (Bohr for TME).

    Notes
    -----
    Handles OCR-artifact spaces inside coordinate numbers (e.g. '1.410 834 003'
    which appears as three separate tokens but is really one float).  The
    strategy: strip the atom symbol from each line, join the rest into one
    string, then use a regex to extract exactly three signed-float patterns.
    Angle is extracted from the comment line as the first float preceding '°' or '◦'.
    """
    frames = {}
    with open(xyz_file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Try to read atom count
        try:
            n_atoms = int(line)
        except ValueError:
            i += 1
            continue

        i += 1
        if i >= len(lines):
            break

        # Comment line — extract angle
        comment = lines[i].strip()
        i += 1

        angle_match = re.search(r'([\d.]+)\s*[°◦]', comment)
        if angle_match:
            angle = float(angle_match.group(1))
        else:
            angle = float(len(frames))  # fallback: sequential index

        # Read atom lines
        atom_list = []
        for _ in range(n_atoms):
            if i >= len(lines):
                break
            atom_line = lines[i].strip()
            i += 1
            if not atom_line:
                continue
            tokens = atom_line.split()
            if len(tokens) < 2:
                continue
            symbol = tokens[0]
            # Merge coordinate tokens, handling OCR-artifact spaces within numbers.
            # A token is a "fragment" (continuation of the previous number) if it
            # contains only digits (no '.', no leading sign).  Such fragments
            # are concatenated onto the preceding token rather than starting a
            # new coordinate.
            coord_tokens = tokens[1:]
            merged = []
            buf = ''
            for tok in coord_tokens:
                is_fragment = tok.lstrip('0123456789') == '' and '.' not in tok
                if buf == '' or not is_fragment:
                    if buf:
                        merged.append(buf)
                    buf = tok
                else:
                    buf += tok  # glue fragment onto current number
            if buf:
                merged.append(buf)

            if len(merged) == 3:
                atom_list.append((symbol, tuple(float(v) for v in merged)))
            else:
                print(f"Warning: expected 3 coords on line '{atom_line}', "
                      f"got {len(merged)}: {merged}")

        frames[angle] = atom_list

    return frames


def create_TME_mol(xyz_file, angle_deg, basis, charge=0, spin=0, units="bohr"):
    """
    Create a PySCF molecule object for TME (C6H8) from an xyz file.

    Parameters
    ----------
    xyz_file : str
        Path to the multi-frame xyz file (TME_1a1.xyz or TME_3b1.xyz).
    angle_deg : float
        Dihedral angle of the endgroups in degrees. The closest available
        frame in the xyz file is selected.
    basis : str
        Basis set name.
    charge : int
        Molecular charge (default 0).
    spin : int
        2S value: 0 for singlet, 2 for triplet.
    units : str
        Coordinate units of the xyz file (default 'bohr').

    Returns
    -------
    mol : pyscf.gto.Mole
    """
    frames = parse_xyz_frames(xyz_file)
    if not frames:
        raise ValueError(f"No frames parsed from {xyz_file}")

    closest_angle = min(frames.keys(), key=lambda a: abs(a - angle_deg))
    if abs(closest_angle - angle_deg) > 1.0:
        print(f"Warning: requested angle {angle_deg}° not found; using closest {closest_angle}°")
    atom_list = frames[closest_angle]

    spin_label = "singlet" if spin == 0 else ("triplet" if spin == 2 else f"spin{spin}")
    print(f"Creating TME at {closest_angle}° ({spin_label}): charge={charge}, spin={spin}, "
          f"n_atoms={len(atom_list)}, units={units}")
    print(f"Geometry ({units}):")
    for sym, (x, y, z) in atom_list:
        print(f"  {sym:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")

    mol = gto.Mole()
    mol.build(
        atom=atom_list,
        unit=units,
        charge=charge,
        spin=spin,
        basis=basis,
        symmetry=False,
        verbose=0,
    )
    return mol


def create_system_info_TME(sample_task, args, base_system_name):
    """Create system_info dictionary for TME calculations."""
    sample_mol = create_TME_mol(
        xyz_file=sample_task['geometry_params']['xyz_file'],
        angle_deg=sample_task['geometry_params']['angle'],
        basis=args.list_of_basis_sets[0],
        charge=args.charge,
        spin=args.spin,
        units='bohr',
    )
    return {
        "geometry_type": "TME",
        "units": "bohr",
        "xyz_file": args.xyz_file,
        "spin_state": args.spin_label,
        "charge": int(sample_mol.charge),
        "spin": int(sample_mol.spin),
        "basis_sets": args.list_of_basis_sets,
        "methods": args.methods_to_run,
        "system_name": base_system_name,
        "tme_angles": args.tme_angles,
        "unrestricted": args.unrestricted,
        "mf_conv_tol": args.mf_conv_tol,
        "max_bond_dim": args.max_bond_dim,
        "active_space_orbitals": args.n_orbitals_for_initial_active_space,
        "noise_schedule": args.noise_schedule,
        "svd_schedule": args.svd_schedule,
        "davidson_schedule": args.davidson_schedule,
        "energy_tol": args.energy_tol,
        "discard_tol": args.discard_tol,
        "parallelization": {
            "n_parallel_jobs": args.n_parallel_jobs,
            "n_threads_per_process": args.n_threads_per_process
        }
    }

