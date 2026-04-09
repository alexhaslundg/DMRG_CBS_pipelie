#!/usr/bin/env python3
"""
Block2 DMRG calculation for arbitrary molecules with fixed convergence logic


"""
import os
import json
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import psutil

import dmrg_pipeline.workflows.create_mol as create_mol
import dmrg_pipeline.utils.utils as dmrg_utils
import dmrg_pipeline.workflows.calculate_energy as calculate


def str_to_bool(v):
    """Convert string representation of boolean to actual boolean.
    
    Handles: 'true', 'false', 'True', 'False', '1', '0', 1, 0, True, False
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)
    if isinstance(v, str):
        if v.lower() in ('true', '1', 'yes', 'on'):
            return True
        elif v.lower() in ('false', '0', 'no', 'off'):
            return False
    raise argparse.ArgumentTypeError(f"Cannot convert '{v}' to boolean")


def save_results(output_data, output_path, system_name, bond_length, methods, basis, suffix="", units="angstrom"):
    """Save results to JSON file with proper error handling.
    # extended file name to match the compute tracking: 
    # run_id="${system}_R${R}_${method}_${basis}_${timestamp}_${SLURM_JOB_ID}"
    # """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    system_tag = system_name or "system"
    units_tag = units or "angstrom"
    if suffix:
        filename = output_path / f"{system_tag}_{units_tag}_{suffix}.json"
    else:
        filename = output_path / f"{system_tag}_{units_tag}.json"
    
    # Convert all numpy types before saving
    output_data_converted = dmrg_utils.convert_numpy(output_data)
    
    try:
        with open(filename, 'w') as f:
            json.dump(output_data_converted, f, indent=4)
        print(f"✅ Results saved to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return False


def load_existing_results(output_path, system_name, suffix="", units="angstrom"):
    """Load existing results from JSON file if it exists."""
    output_path = Path(output_path)
    system_tag = system_name or "system"
    units_tag = units or "angstrom"
    
    if suffix:
        filename = output_path / f"{system_tag}_{units_tag}_{suffix}.json"
    else:
        filename = output_path / f"{system_tag}_{units_tag}.json"
    
    if filename.exists():
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            print(f"✅ Loaded existing results from {filename}")
            return data
        except Exception as e:
            print(f"⚠️ Error loading existing results: {e}")
            return {}
    else:
        print(f"📝 No existing results found at {filename}")
        return {}


def _atomic_json_update(json_path, geo_key, basis, method, result):
    """
    Atomically merge one result entry into an on-disk JSON file.

    Reads the existing file (if any), inserts json_data[geo_key][basis][method],
    then writes back via a temp-file rename so the file is never half-written.
    Called from the pre-extrapolation callback in process_single_calculation().
    """
    json_path = Path(json_path)
    # Load existing data
    data = {}
    if json_path.exists():
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception:
            data = {}

    data.setdefault(geo_key, {}).setdefault(basis, {})[method] = result

    tmp_path = json_path.with_suffix(".tmp")
    try:
        data_converted = dmrg_utils.convert_numpy(data)
        with open(tmp_path, "w") as f:
            json.dump(data_converted, f, indent=4)
        os.replace(tmp_path, json_path)
    except Exception as e:
        print(f"⚠️  _atomic_json_update failed for {json_path}: {e}")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def is_calculation_complete(result_dict, method_name):
    """Check if a calculation has been completed successfully."""
    if method_name not in result_dict:
        return False
    
    method_result = result_dict[method_name]
    
    # Check if it's a valid result dictionary
    if not isinstance(method_result, dict):
        return False
    
    # For DMRG, check if converged.
    # A result with _pre_extrap_save=True has the DMRG sweep done but the
    # backward extrapolation not yet run — treat it as incomplete so it gets
    # re-queued and the extrapolation is eventually completed.
    if method_name == "DMRG":
        converged = method_result.get("converged", False)
        pre_extrap = method_result.get("_pre_extrap_save", False)
        return converged and not pre_extrap
    
    # For HF and FCI, check if successful and has energy
    return method_result.get("success", False) and method_result.get("energy") is not None



def generate_run_name(system_name, geometry_params, units, basis, unrestricted, args):
    """
    Generate a standardized run name for file organization.

    Format: H#_bond_length_unit(b/a)_basis_unrestricted/restricted

    Examples:
        H4_R1.50_a_cc-pVDZ_restricted
        H10_dA1.75_b_aug-cc-pVTZ_unrestricted
        C_triplet_a_cc-pVDZ_unrestricted
    """
    # System identifier (e.g., H4, H10, N2, C)
    sys_tag = system_name

    # Geometry identifier
    if 'bond_length' in geometry_params:
        geom_tag = f"R{geometry_params['bond_length']:.6g}"
    elif 'radius' in geometry_params and 'angle' in geometry_params:
        geom_tag = f"R{geometry_params['radius']:.6g}_theta{geometry_params['angle']:.1f}"
    elif 'd_A' in geometry_params:
        if 'd_B' in geometry_params:
            geom_tag = f"dA{geometry_params['d_A']:.6g}_dB{geometry_params['d_B']:.6g}"
        else:
            geom_tag = f"dA{geometry_params['d_A']:.6g}"
    elif 'atom' in geometry_params and 'spin' in geometry_params:
        # For atomic systems: use spin state label
        spin = geometry_params['spin']
        spin_labels = {0: 'singlet', 1: 'doublet', 2: 'triplet', 3: 'quartet', 4: 'quintet', 5: 'sextet'}
        geom_tag = spin_labels.get(spin, f'spin{spin}')
    else:
        geom_tag = "unknown"

    # Units tag: 'a' for angstrom, 'b' for bohr
    units_tag = 'a' if units.lower().startswith('ang') else 'b'

    # Basis set (clean up any special characters)
    basis_tag = basis.replace('-', '').replace('*', 's')

    # Restricted/unrestricted tag
    spin_tag = 'unrestricted' if unrestricted else 'restricted'

    return f"{sys_tag}_{geom_tag}_{units_tag}_{basis_tag}_{spin_tag}"


def process_single_calculation(task_args):
    """
    Worker function to process a single geometry/basis/method combination.
    This runs in a separate process with isolated memory.
    
    Parameters:
    -----------
    task_args : dict
        Contains all parameters needed for one calculation
        
    Returns:
    --------
    dict with results and metadata
    """
    # Unpack arguments
    geometry_params = task_args['geometry_params']
    basis = task_args['basis']
    method_name = task_args['method_name']
    args = task_args['args']
    method_dict = task_args['method_dict']
    
    geometry_key = None  # Initialize to avoid UnboundLocalError
    run_name = None  # Initialize run_name
    
    try:
        print(f"🔄 Process {os.getpid()}: Starting {method_name} for {geometry_params} with {basis}")
        
        # Get spin label for geometry key
        spin_label = "singlet" if args.spin == 0 else ("triplet" if args.spin == 2 else f"spin{args.spin}")
        
        # Determine system name for run_name generation
        if args.atomic:
            system_name = f"{args.atom}"
        elif args.diatomic:
            system_name = f"{args.atom1}2" if args.atom1 == args.atom2 else f"{args.atom1}{args.atom2}"
        elif args.triatomic:
            system_name = f"{args.atom1}{args.center_atom}{args.atom2}"
        elif args.H4_mol:
            system_name = "H4"
        elif args.single_chain:
            system_name = f"H{geometry_params['n_rep']}"  # e.g., H10 for n_rep=5
        elif args.double_chain:
            system_name = f"H{geometry_params['n_rep'] * 2}"  # double chain has 2x atoms
        else:
            system_name = "molecule"
        
        # Generate run_name for this calculation
        run_name = generate_run_name(
            system_name=system_name,
            geometry_params=geometry_params,
            units=args.units,
            basis=basis,
            unrestricted=args.unrestricted,
            args=args
        )
        
        # Create active space output directory
        active_space_output_dir = os.path.join(
            args.output_file_path, "active_space", run_name
        )
        
        print(f"🏷️  Run name: {run_name}")
        print(f"📂 Active space output: {active_space_output_dir}")
        
        # Create molecule based on geometry type
        if args.atomic: 
            mol = create_mol.create_atomic_mol(
                atom=args.atom,
                basis=basis,
                charge=args.charge,
                spin=args.spin
            )
            geometry_key = f"{args.atom}_{spin_label}"

        elif args.diatomic:
            mol = create_mol.create_diatomic_mol(
                atom1=args.atom1,
                atom2=args.atom2,
                bond_length=geometry_params['bond_length'],
                basis=basis,
                charge=args.charge,
                spin=args.spin,
                units=args.units
            )
            geometry_key = f"R_{geometry_params['bond_length']:.6g}_{spin_label}"

        elif args.triatomic:
            mol = create_mol.create_triatomic_mol(
                center_atom=args.center_atom,
                atom1=args.atom1,
                atom2=args.atom2,
                bond_length=geometry_params['bond_length'],
                basis=basis,
                units=args.units
            )
            geometry_key = f"R_{geometry_params['bond_length']:.6g}_{spin_label}"

        elif args.H4_mol:
            mol = create_mol.create_H4(
                radius=geometry_params['radius'],
                angle=geometry_params['angle'],
                basis=basis,
                units=args.units
            )
            geometry_key = f"R_{geometry_params['radius']:.6g}_theta_{geometry_params['angle']:.6g}_{spin_label}"

        elif args.single_chain:
            mol = create_mol.create_H_single_chain(
                d_A=geometry_params['d_A'],
                n_rep=geometry_params['n_rep'],
                basis=basis,
                units=args.units
            )
            geometry_key = f"dA_{geometry_params['d_A']:.6g}_n{geometry_params['n_rep']}_{spin_label}"

        elif args.double_chain:
            mol = create_mol.create_H_double_chain(
                d_A=geometry_params['d_A'],
                d_B=geometry_params['d_B'],
                n_rep=geometry_params['n_rep'],
                basis=basis,
                units=args.units
            )
            geometry_key = f"dA_{geometry_params['d_A']:.6g}_dB_{geometry_params['d_B']:.6g}_n{geometry_params['n_rep']}_{spin_label}"

        elif args.tme:
            mol = create_mol.create_TME_mol(
                xyz_file=geometry_params['xyz_file'],
                angle_deg=geometry_params['angle'],
                basis=basis,
                charge=args.charge,
                spin=args.spin,
                units='bohr',
            )
            geometry_key = f"angle_{geometry_params['angle']:.1f}_{spin_label}"
            system_name = "TME"

        # set the max memory for the molecule based on available system memory
        num_workers = args.n_parallel_jobs
        avail_mb = psutil.virtual_memory().available / 1e6

        # Use 80% of memory divided by #workers
        mol.max_memory = int((avail_mb * 0.8) / num_workers)


        # Get method function
        method_func = method_dict.get(method_name)
        if method_func is None:
            return {
                'geometry_key': geometry_key,
                'basis': basis,
                'method_name': method_name,
                'result': None,
                'error': f"Method {method_name} not found"
            }
        
        # Run calculation
        if method_name == "DMRG":
            # Use fewer threads per DMRG process when running in parallel
            n_threads = max(1, args.n_threads_per_process)

            # Build occupation thresholds tuple from separate arguments
            occupation_thresholds = (
                getattr(args, 'occupation_threshold_lower', 0.01),
                getattr(args, 'occupation_threshold_upper', 1.99)
            )

            # ----------------------------------------------------------
            # Build pre-extrapolation fault-tolerance callback.
            # Saves the DMRG result (sweep converged, no extrap yet) to
            # the main output JSON before backward extrapolation starts,
            # so a SLURM-killed job doesn't lose the result entirely.
            # ----------------------------------------------------------
            _pre_extrap_base_system = task_args.get("base_system_name", "")
            _pre_extrap_units = getattr(args, "units", "angstrom") or "angstrom"
            _pre_extrap_suffix = getattr(args, "file_name_suffix", "") or ""
            if _pre_extrap_suffix:
                _pre_extrap_json_path = (
                    Path(args.output_file_path)
                    / f"{_pre_extrap_base_system}_{_pre_extrap_units}_{_pre_extrap_suffix}.json"
                )
            else:
                _pre_extrap_json_path = (
                    Path(args.output_file_path)
                    / f"{_pre_extrap_base_system}_{_pre_extrap_units}.json"
                )
            _pre_extrap_geo_key = geometry_key
            _pre_extrap_basis = basis

            def _pre_extrap_callback(dmrg_partial, _path=_pre_extrap_json_path,
                                     _geo=_pre_extrap_geo_key, _bas=_pre_extrap_basis):
                """Persist pre-extrapolation DMRG result to the main JSON."""
                dmrg_partial["_pre_extrap_save"] = True
                _atomic_json_update(_path, _geo, _bas, "DMRG", dmrg_partial)
                print(f"💾 Pre-extrapolation save → {_path.name}")

            result = method_func(
                mol=mol,
                # Orbital construction options
                orbital_method=getattr(args, 'orbital_method', 'MP2'),
                localize_orbitals=getattr(args, 'localize_orbitals', False),
                localization_method=getattr(args, 'localization_method', 'PM'),
                # Active space selection
                n_orbitals_for_initial_active_space=args.n_orbitals_for_initial_active_space,
                occupation_thresholds=occupation_thresholds,
                use_dmrg_active_space_selection=getattr(args, 'use_dmrg_active_space_selection', False),
                as_entropy_threshold=getattr(args, 'as_entropy_threshold', 1e-3),
                as_bond_dim=getattr(args, 'as_bond_dim', 80),
                as_n_sweeps=getattr(args, 'as_n_sweeps', 4),
                as_energy_window_occ=getattr(args, 'as_energy_window_occ', None),
                as_energy_window_virt=getattr(args, 'as_energy_window_virt', None),
                # Fiedler reordering
                perform_reordering=getattr(args, 'perform_reordering', True),
                reorder_method=getattr(args, 'reorder_method', 'fiedler'),
                reorder_bond_dim=getattr(args, 'reorder_bond_dim', 50),
                reorder_sweeps=getattr(args, 'reorder_sweeps', 10),
                # Main DMRG parameters
                initial_bond_dim=args.initial_bond_dim,
                max_bond_dim=args.max_bond_dim,
                max_sweeps=args.max_sweeps,
                energy_tol=args.energy_tol,
                discard_tol=args.discard_tol,
                intra_bd_energy_tol=args.intra_bd_energy_tol,
                max_sweeps_per_bd=args.max_sweeps_per_bd,
                noise_schedule=args.noise_schedule,
                svd_schedule=args.svd_schedule,
                davidson_schedule=args.davidson_schedule,
                # Backward extrapolation
                perform_extrapolation=getattr(args, 'perform_extrapolation', True),
                extrap_sweeps_per_bd=getattr(args, 'extrap_sweeps_per_bd', 4),
                extrap_bd_reductions=getattr(args, 'extrap_bd_reductions', None),
                # Orbital visualization options
                generate_cube_files=getattr(args, 'generate_cube_files', False),
                cube_resolution=getattr(args, 'cube_resolution', 80),
                cube_margin=getattr(args, 'cube_margin', 3.0),
                generate_py3dmol_viz=getattr(args, 'generate_py3dmol_viz', False),
                py3dmol_n_orbitals=getattr(args, 'py3dmol_n_orbitals', 4),
                py3dmol_isoval=getattr(args, 'py3dmol_isoval', 0.02),
                # General options
                mf=None,
                mf_conv_tol=args.mf_conv_tol,
                unrestricted=args.unrestricted,
                n_threads=n_threads,
                scratch_dir=None,  # auto-creates under $TMPDIR, cleaned up by run_dmrg()
                verbose=1,
                # Output options
                output_dir=active_space_output_dir,
                run_name=run_name,
                # Fault-tolerance: save main result before backward extrapolation
                pre_extrap_callback=_pre_extrap_callback,
                # MPS persistence: save converged KET after forward pass
                save_mps_dir=getattr(args, 'save_mps_dir', None),
            )
        else:
            result = method_func(
                mol=mol,
                mf_conv_tol=args.mf_conv_tol,
                unrestricted=args.unrestricted,
            )
        
        # Clean up mol object to free memory
        del mol
        dmrg_utils.cleanup_memory()
        
        print(f"✅ Process {os.getpid()}: Completed {method_name} for {geometry_params} with {basis}")
        
        return {
            'geometry_key': geometry_key,
            'geometry_params': geometry_params,
            'basis': basis,
            'method_name': method_name,
            'units': args.units,
            'run_name': run_name,
            'result': result,
            'error': None
        }

    except Exception as e:
        print(f"❌ Process {os.getpid()}: Error in {method_name} for {geometry_params} with {basis}: {e}")
        import traceback
        return {
            'geometry_key': geometry_key,
            'geometry_params': geometry_params,
            'basis': basis,
            'method_name': method_name,
            'units': args.units,
            'run_name': run_name,
            'result': None,
            'error': str(e) + "\n" + traceback.format_exc()
        }


def create_task_list_atomic(args, output_data, method_dict):
    """Create list of tasks for atomic calculations."""
    tasks = []
    
    # Create geometry key that includes spin state
    spin_label = "singlet" if args.spin == 0 else ("triplet" if args.spin == 2 else f"spin{args.spin}")
    geometry_params = {'atom': args.atom, 'spin': args.spin}
    geometry_key = f"{args.atom}_{spin_label}"

    for basis in args.list_of_basis_sets:
        for method_name in args.methods_to_run:
            # Check if already completed
            if (geometry_key in output_data and 
                basis in output_data[geometry_key] and
                is_calculation_complete(output_data[geometry_key][basis], method_name)):
                print(f"⏭️  Skipping {method_name} for {geometry_key} with {basis} - already completed")
                continue
            
            # Add to task list
            tasks.append({
                'geometry_params': geometry_params,
                'basis': basis,
                'method_name': method_name,
                'args': args,
                'unrestricted': args.unrestricted,
                'method_dict': method_dict,
            })
    
    return tasks

def create_task_list_diatomic(args, output_data, method_dict):
    """Create list of tasks for diatomic molecule calculations."""
    tasks = []

    for bond_length in args.bond_lengths:
        geometry_params = {'bond_length': bond_length}
        geometry_key = f"R_{bond_length:.6g}"
        
        for basis in args.list_of_basis_sets:
            for method_name in args.methods_to_run:
                # Check if already completed
                if (geometry_key in output_data and 
                    basis in output_data[geometry_key] and
                    is_calculation_complete(output_data[geometry_key][basis], method_name)):
                    print(f"⏭️  Skipping {method_name} for {geometry_key} with {basis} - already completed")
                    continue
                
                # Add to task list
                tasks.append({
                    'geometry_params': geometry_params,
                    'basis': basis,
                    'method_name': method_name,
                    'args': args,
                    'unrestricted': args.unrestricted,
                    'method_dict': method_dict,
                })
    
    return tasks


def create_task_list_triatomic(args, output_data, method_dict):
    """Create list of tasks for triatomic molecules."""
    tasks = []
    
    for bond_length in args.bond_lengths:
        geometry_params = {'bond_length': bond_length}
        geometry_key = f"R_{bond_length:.6g}"

        for basis in args.list_of_basis_sets:
            for method_name in args.methods_to_run:
                # Skip already computed
                if (geometry_key in output_data and
                    basis in output_data[geometry_key] and
                    is_calculation_complete(output_data[geometry_key][basis], method_name)):
                    print(f"⏭️ Skipping {method_name} for {geometry_key}/{basis}")
                    continue

                tasks.append({
                    'geometry_params': geometry_params,
                    'basis': basis,
                    'method_name': method_name,
                    'args': args,
                    'method_dict': method_dict,
                })

    return tasks


def create_task_list_H4(args, output_data, method_dict):
    """Create list of tasks for H4 (square, rectangular, general angle)."""
    tasks = []
    
    for R in args.radii:
        for angle in args.angles:
            geometry_params = {'radius': R, 'angle': angle}
            geometry_key = f"R_{R:.6g}_theta_{angle:.6g}"
            
            for basis in args.list_of_basis_sets:
                for method_name in args.methods_to_run:
                    # Skip already computed
                    if (geometry_key in output_data and 
                        basis in output_data[geometry_key] and
                        is_calculation_complete(output_data[geometry_key][basis], method_name)):
                        print(f"⏭️ Skipping {method_name} for {geometry_key}/{basis}")
                        continue

                    tasks.append({
                        'geometry_params': geometry_params,
                        'basis': basis,
                        'method_name': method_name,
                        'args': args,
                        'method_dict': method_dict,

                    })
    
    return tasks


def create_task_list_single_chain(args, output_data, method_dict):
    """Create list of tasks for single chain calculations."""
    tasks = []
    
    for n_rep in args.n_rep:
        for d_A in args.d_A:
            geometry_params = {'d_A': d_A, 'n_rep': n_rep}
            geometry_key = f"dA_{d_A:.6g}_n{n_rep}_{args.units}"
            
            for basis in args.list_of_basis_sets:
                for method_name in args.methods_to_run:
                    # Check if already completed
                    if (geometry_key in output_data and 
                        basis in output_data[geometry_key] and
                        is_calculation_complete(output_data[geometry_key][basis], method_name)):
                        print(f"⏭️  Skipping {method_name} for {geometry_key} with {basis} - already completed")
                        continue
                    
                    # Add to task list
                    tasks.append({
                        'geometry_params': geometry_params,
                        'basis': basis,
                        'method_name': method_name,
                        'args': args,
                        'method_dict': method_dict,

                    })
    
    return tasks


def create_task_list_double_chain(args, output_data, method_dict):
    """Create list of tasks for double chain calculations."""
    tasks = []
    
    for n_rep in args.n_rep:
        for d_A in args.d_A:
            for d_B in args.d_B:
                geometry_params = {'d_A': d_A, 'd_B': d_B, 'n_rep': n_rep}
                geometry_key = f"dA_{d_A:.6g}_dB_{d_B:.6g}_n{n_rep}_{args.units}"
                
                for basis in args.list_of_basis_sets:
                    for method_name in args.methods_to_run:
                        # Check if already completed
                        if (geometry_key in output_data and 
                            basis in output_data[geometry_key] and
                            is_calculation_complete(output_data[geometry_key][basis], method_name)):
                            print(f"⏭️  Skipping {method_name} for {geometry_key} with {basis} - already completed")
                            continue
                        
                        # Add to task list
                        tasks.append({
                            'geometry_params': geometry_params,
                            'basis': basis,
                            'method_name': method_name,
                            'args': args,
                            'method_dict': method_dict,
    
                        })
    
    return tasks


def create_task_list_tme(args, output_data, method_dict):
    """Create list of tasks for TME calculations across dihedral angles."""
    tasks = []
    spin_label = "singlet" if args.spin == 0 else ("triplet" if args.spin == 2 else f"spin{args.spin}")

    for angle in args.tme_angles:
        geometry_params = {'angle': angle, 'xyz_file': args.xyz_file}
        geometry_key = f"angle_{angle:.1f}_{spin_label}"

        for basis in args.list_of_basis_sets:
            for method_name in args.methods_to_run:
                if (geometry_key in output_data and
                        basis in output_data[geometry_key] and
                        is_calculation_complete(output_data[geometry_key][basis], method_name)):
                    print(f"⏭️  Skipping {method_name} for {geometry_key} with {basis} - already completed")
                    continue

                tasks.append({
                    'geometry_params': geometry_params,
                    'basis': basis,
                    'method_name': method_name,
                    'args': args,
                    'method_dict': method_dict,
                })

    return tasks


def merge_and_save_results(results, output_data, args, base_system_name, geometry_key, args_methods, args_basis):
    """
    Merge results into output_data and save intermediate results for a specific geometry.
    
    Parameters:
    -----------
    results : list
        List of result dictionaries from parallel processes
    output_data : dict
        Main data structure to merge results into
    args : argparse.Namespace
        Command line arguments
    base_system_name : str
        Base name for the system
    geometry_key : str
        Current geometry key being processed
    """
    # Merge results into output_data
    for result_dict in results:
        if result_dict['error'] is None:
            geo_key = result_dict['geometry_key']
            geo_params = result_dict.get('geometry_params', {})
            basis = result_dict['basis']
            method_name = result_dict['method_name']
            units = result_dict.get('units', 'angstrom')

            # Initialize nested structure if needed
            if geo_key not in output_data:
                output_data[geo_key] = {}
            if basis not in output_data[geo_key]:
                output_data[geo_key][basis] = {}

            # Store exact geometry metadata at the geometry level (used by CBSAnalyzer)
            if 'bond_length' in geo_params:
                output_data[geo_key]['_bond_length_ang'] = geo_params['bond_length']
            elif 'd_A' in geo_params:
                output_data[geo_key]['_d_A_bohr'] = geo_params['d_A']
                if 'd_B' in geo_params:
                    output_data[geo_key]['_d_B_bohr'] = geo_params['d_B']
            elif 'angle' in geo_params:
                output_data[geo_key]['_tme_angle_deg'] = geo_params['angle']
                # Store xyz geometry string so notebooks can load and visualize it
                if '_xyz_geometry_bohr' not in output_data[geo_key]:
                    frames = create_mol.parse_xyz_frames(geo_params['xyz_file'])
                    closest = min(frames.keys(), key=lambda a: abs(a - geo_params['angle']))
                    atom_list = frames[closest]
                    xyz_lines = [f"{sym} {x:.9f} {y:.9f} {z:.9f}" for sym, (x, y, z) in atom_list]
                    output_data[geo_key]['_xyz_geometry_bohr'] = "\n".join(xyz_lines)
            if units:
                output_data[geo_key]['_units'] = units

            # Store result
            output_data[geo_key][basis][method_name] = result_dict['result']

            print(f"✅ Merged result: {geo_key}/{basis}/{method_name}")
        else:
            print(f"❌ Failed task: {result_dict['geometry_key']}/{result_dict['basis']}/{result_dict['method_name']}")
            print(f"   Error: {result_dict['error']}")
    
    # Save intermediate results for this geometry
    print(f"\n💾 Saving intermediate results for {geometry_key}...")

    if len(args.bond_lengths) == 1:
        bond_lengths = f"{args.bond_lengths[0]}"
    else:
        bond_lengths = str(args.bond_lengths).replace(' ', '').replace(',', '_')
    if len(args.list_of_basis_sets) == 1:
        basis_sets_ran = args.list_of_basis_sets[0]
    else:
        basis_sets_ran = args.list_of_basis_sets

    if len(args.methods_to_run) == 1:
        methods_ran = args.methods_to_run[0]
    else:
        methods_ran = args.methods_to_run
    

    save_results(output_data, args.output_file_path, base_system_name, bond_lengths, methods_ran, basis_sets_ran, suffix=f"{args.file_name_suffix}", units=args.units)
    # save_results(output_data, args.output_file_path, base_system_name, geometry_key, args.methods_to_run, args.list_of_basis_sets,
    #             suffix=f"_intermediate_{geometry_key}")

def run_parallel_calculations(
        args, output_data, method_dict,
        base_system_name, task_creator_func, system_info_creator_func):
    """
    Generic function to run parallel calculations for any molecular system.
    """
    # --------------------------------------------------
    # Create task list
    # --------------------------------------------------
    all_tasks = task_creator_func(args, output_data, method_dict)

    # Propagate base_system_name so process_single_calculation can build
    # the correct JSON path for the pre-extrapolation fault-tolerance save.
    for task in all_tasks:
        task["base_system_name"] = base_system_name

    print(f"\n📋 Total tasks to run: {len(all_tasks)}")

    if len(all_tasks) == 0:
        print("✅ All calculations already completed!")
        return

    # --------------------------------------------------
    # Create system_info *immediately* so intermediate
    # saves include it (but avoid duplicates)
    # --------------------------------------------------
    if "system_info" not in output_data:
        output_data["system_info"] = system_info_creator_func(
            all_tasks[0], args, base_system_name
        )

    # --------------------------------------------------
    # Group tasks by geometry for intermediate saving
    # --------------------------------------------------
    tasks_by_geometry = {}
    for task in all_tasks:
        # Get spin label for geometry key
        spin = task['args'].spin
        spin_label = "singlet" if spin == 0 else ("triplet" if spin == 2 else f"spin{spin}")
        
        if args.atomic:
            geo_key = f"{task['geometry_params']['atom']}_{spin_label}"
        elif args.diatomic or args.triatomic:
            geo_key = f"R_{task['geometry_params']['bond_length']:.2f}_{spin_label}"
        elif args.H4_mol:
            geo_key = (
                f"R_{task['geometry_params']['radius']:.2f}_"
                f"theta_{task['geometry_params']['angle']:.2f}_{spin_label}"
            )
        elif args.single_chain:
            geo_key = (
                f"dA_{task['geometry_params']['d_A']:.2f}_"
                f"n{task['geometry_params']['n_rep']}_{spin_label}"
            )
        elif args.double_chain:
            geo_key = (
                f"dA_{task['geometry_params']['d_A']:.2f}_"
                f"dB_{task['geometry_params']['d_B']:.2f}_"
                f"n{task['geometry_params']['n_rep']}_{spin_label}"
            )
        elif args.tme:
            geo_key = f"angle_{task['geometry_params']['angle']:.2f}_{spin_label}"

        tasks_by_geometry.setdefault(geo_key, []).append(task)

    # --------------------------------------------------
    # Process each geometry's tasks and save results
    # --------------------------------------------------
    for geo_key, tasks in tasks_by_geometry.items():
        print(f"\n{'='*60}")
        print(f"Processing geometry: {geo_key}")
        print(f"Number of tasks for this geometry: {len(tasks)}")
        print(f"{'='*60}")

        with Pool(processes=args.n_parallel_jobs) as pool:
            results = pool.map(process_single_calculation, tasks)

        # Save partial results (now includes system_info)
        merge_and_save_results(results, output_data, args, base_system_name, geo_key, args.methods_to_run, args.list_of_basis_sets)


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Block2 DMRG calculation with clear tolerances")
    parser.add_argument("--base_config", type=str, help="Base configuration file path")
    
    # molecule-specific arguments
    parser.add_argument("--bond_lengths", type=float, nargs='+', default=[0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
    parser.add_argument("--output_file_path", type=str)
    parser.add_argument("--file_name_suffix", type=str, default="", help="Suffix for output file name")
    parser.add_argument("--atom1", type=str, default="H", help="First atom symbol for diatomic molecule")
    parser.add_argument("--atom2", type=str, default="H", help="Second atom symbol for diatomic molecule")
    parser.add_argument("--atom", type=str, default="C", help="Atom symbol for atomic calculations")
    parser.add_argument("--atomic", action='store_true', help="Flag to indicate if the system is an atom")
    parser.add_argument("--diatomic", action='store_true', help="Flag to indicate if the system is a diatomic molecule")
    parser.add_argument("--triatomic", action='store_true', help="Flag to indicate if the system is a triatomic molecule")
    parser.add_argument("--center_atom", type=str, default="O", help="Center atom symbol for triatomic molecule")
    parser.add_argument("--angles", type=float, nargs='+', default=[90.0], help="Angles for H4 molecule")
    parser.add_argument("--radii", type=float, nargs='+', default=[1.738], help="Radii for H4 molecule")
    parser.add_argument("--H4_mol", action='store_true', help="Flag to indicate if the system is H4 molecule")
    parser.add_argument("--double_chain", action='store_true', help="Flag to indicate if the system is H double chain")
    parser.add_argument("--single_chain", action='store_true', help="Flag to indicate if the system is H single chain")
    parser.add_argument("--tme", action='store_true', help="Flag to indicate if the system is TME (C6H8) from xyz file")
    parser.add_argument("--xyz_file", type=str, default=None, help="Path to multi-frame xyz file for TME geometry")
    parser.add_argument("--tme_angles", type=float, nargs='+', default=[0.01, 45.0], help="Dihedral angles (degrees) to compute for TME")
    parser.add_argument("--spin_label", type=str, default="singlet", help="Spin state label for naming (singlet/triplet)")
    parser.add_argument("--d_A", type=float, nargs='+', default=[1.0], help="Distance d_A for H chains (units of angstroms)")
    parser.add_argument("--d_B", type=float, nargs='+', default=[2.0], help="Distance d_B for H double chain (units of angstroms)")
    parser.add_argument("--units", type=str, default="angstrom", help="Units for distances (angstrom or bohr)")
    parser.add_argument("--n_rep", type=int, nargs='+', default=[4], help="Number of repetitions for H chains")
    parser.add_argument("--charge", type=int, default=0, help="Molecular charge")
    parser.add_argument("--spin", type=int, default=0, help="Molecular spin")

    # shared HF, FCI, DMRG arguments
    parser.add_argument("--mf_conv_tol", type=float, default=1e-14)
    parser.add_argument("--unrestricted", action='store_true', help="Use unrestricted calculations")
    parser.add_argument("--list_of_basis_sets", type=str, nargs='+', default=['sto-3g', 'cc-pVDZ', 'cc-pVTZ', 'cc-pVQZ'], help="List of basis sets to use")
    parser.add_argument("--methods_to_run", type=str, nargs='+', default=['HF', 'DMRG', "FCI", "CCSDT"], help="Methods to run: HF, FCI, DMRG, CCSDT")


    # DMRG-specific arguments
    parser.add_argument("--initial_bond_dim", type=int, default=100, help="Initial bond dimension for DMRG")
    parser.add_argument("--max_bond_dim", type=int, default=400, help="Maximum bond dimension for DMRG")
    parser.add_argument("--max_sweeps", type=int, default=35, help="Maximum number of sweeps for DMRG, for the fine tuning")
    parser.add_argument("--noise_schedule", type=float, nargs='+', default=[1e-3, 5e-4, 1e-4, 1e-5, 1e-6, 0], help="Noise schedule for DMRG")
    parser.add_argument("--svd_schedule", type=float, nargs='+', default=[1e-5, 5e-6, 1e-6, 1e-7, 1e-8, 1e-12], help="SVD cutoff schedule for DMRG")
    parser.add_argument("--davidson_schedule", type=float, nargs='+', default=[1e-6, 5e-7, 1e-7, 1e-8, 1e-9, 1e-10], help="Davidson tolerance schedule for DMRG")
    parser.add_argument("--energy_tol", type=float, default=1e-6, help="Energy convergence tolerance for DMRG")
    parser.add_argument("--discard_tol", type=float, default=2e-7, help="Discarded weight tolerance for DMRG")
    parser.add_argument("--intra_bd_energy_tol", type=float, default=1e-5, help="Energy convergence threshold per BD block during backward extrapolation")
    parser.add_argument("--max_sweeps_per_bd", type=int, default=10, help="Maximum sweeps per BD block during backward extrapolation")

    # DMRG orbital construction arguments
    parser.add_argument("--orbital_method", type=str, default='HF', choices=['HF', 'MP2'], 
                       help="Method for orbital construction: 'HF' or 'MP2'")
    parser.add_argument("--localize_orbitals", type=str_to_bool, default=False, help="Localize orbitals before DMRG (true/false or 1/0)")
    parser.add_argument("--localization_method", type=str, default='IBO', choices=['PM', 'IBO'],
                       help="Localization method: 'PM' (Pipek-Mezey) or 'IBO'")
    

    # DMRG active space selection arguments
    parser.add_argument("--n_orbitals_for_initial_active_space", type=int, default=None, help="Number of orbitals in the active space for DMRG")
    parser.add_argument("--occupation_threshold_lower", type=float, default=0.01, 
                       help="Lower occupation threshold for active space selection")
    parser.add_argument("--occupation_threshold_upper", type=float, default=1.99, 
                       help="Upper occupation threshold for active space selection")
    parser.add_argument("--use_dmrg_active_space_selection", type=bool, default=True,
                       help="Use DMRG orbital entropies for active space selection")
    parser.add_argument("--as_entropy_threshold", type=float, default=1e-3,
                       help="Entropy threshold for DMRG-based active space selection")
    parser.add_argument("--as_bond_dim", type=int, default=100,
                       help="Bond dimension for active space selection DMRG")
    parser.add_argument("--as_n_sweeps", type=int, default=6,
                       help="Number of sweeps for active space selection DMRG")
    parser.add_argument("--as_energy_window_occ", type=float, default=None,
                       help="Energy pre-selection: Hartree below HOMO to include. "
                            "Orbitals with ε < HOMO - value become frozen core. "
                            "None = keep all occupied orbitals.")
    parser.add_argument("--as_energy_window_virt", type=float, default=None,
                       help="Energy pre-selection: Hartree above LUMO to include. "
                            "Orbitals with ε > LUMO + value are dropped from the "
                            "entropy DMRG pool. None = keep all virtuals.")

    # DMRG Fiedler reordering arguments
    parser.add_argument("--perform_reordering", type=bool, default=True,
                       help="Perform Fiedler orbital reordering (default: True)")
    parser.add_argument("--no_reordering", action='store_false', dest='perform_reordering',
                       help="Disable Fiedler orbital reordering")
    parser.add_argument("--reorder_method", type=str, default='fiedler', 
                       choices=['fiedler', 'gaopt', 'genetic'],
                       help="Reordering strategy: 'fiedler', 'gaopt', or 'genetic'")
    parser.add_argument("--reorder_bond_dim", type=int, default=50, 
                       help="Bond dimension for reordering DMRG")
    parser.add_argument("--reorder_sweeps", type=int, default=10, 
                       help="Sweeps for reordering DMRG")

    # DMRG backward extrapolation arguments
    parser.add_argument("--perform_extrapolation", type=bool, default=True,
                       help="Perform backward extrapolation (default: True)")
    parser.add_argument("--no_extrapolation", action='store_false', dest='perform_extrapolation',
                       help="Disable backward extrapolation")
    parser.add_argument("--extrap_sweeps_per_bd", type=int, default=4, 
                       help="Sweeps per bond dimension in backward extrapolation")
    parser.add_argument("--extrap_bd_reductions", type=float, nargs='+', default=None,
                       help="Bond dimension reduction factors for extrapolation (default: [0.9, 0.75, 0.6, 0.45, 0.25, 0.1])")

    # Orbital visualization arguments
    parser.add_argument("--generate_cube_files", action='store_true', 
                       help="Generate cube files for selected active space orbitals")
    parser.add_argument("--cube_resolution", type=int, default=80,
                       help="Grid resolution for cube files (points per dimension)")
    parser.add_argument("--cube_margin", type=float, default=3.0,
                       help="Margin around molecule for cube files (Bohr)")
    parser.add_argument("--generate_py3dmol_viz", action='store_true',
                       help="Generate interactive py3Dmol HTML visualization")
    parser.add_argument("--py3dmol_n_orbitals", type=int, default=4,
                       help="Number of orbitals to include in py3Dmol visualization")
    parser.add_argument("--py3dmol_isoval", type=float, default=0.02,
                       help="Isosurface value for py3Dmol visualization")

    # MPS persistence
    parser.add_argument("--save_mps_dir", type=str, default=None,
                       help="If set, copy the converged MPS (KET) to this directory after "
                            "the forward pass completes, before entropy calculation / "
                            "backward extrapolation.  Use this to checkpoint the wavefunction "
                            "so the calculation can be resumed with a larger bond dimension.")

    # parallelization arguments
    parser.add_argument("--n_parallel_jobs", type=int, default=None,
                       help="Number of parallel jobs (default: auto-detect based on CPUs)")
    parser.add_argument("--n_threads_per_process", type=int, default=2,
                       help="Number of threads per DMRG process (default: 2)")

    args = parser.parse_args()

    base_system_name = None

    print("Block2 DMRG Calculation for Arbitrary Molecules")
    print("="*60)

    method_dict = {
        "HF": calculate.run_hf,
        "FCI": calculate.run_fci,
        "DMRG": calculate.run_dmrg,
        "CCSDT": calculate.run_ccsdt,
        "MP2": calculate.run_mp2
    }
    
    # # Max bond dim based on basis set (currently all set to 1)
    # max_bond_dim_dict = {
    #     "sto-3g": 1,
    #     "cc-pVDZ": 1,
    #     "cc-pVTZ": 1,
    #     "cc-pVQZ": 1,
    #     "def2-TZVPD": 1,
    #     "aug-cc-pVDZ": 1,
    #     "aug-cc-pVTZ": 1,
    #     "aug-cc-pVQZ": 1
    # }
    
    # Determine system name upfront
    # Get spin label
    spin_label = "singlet" if args.spin == 0 else ("triplet" if args.spin == 2 else f"spin{args.spin}")
    
    if args.base_config is None:
        if args.atomic: 
            base_system_name = f"{args.atom}_{spin_label}"
        elif args.diatomic:
            base_system_name = f"{args.atom1}2_{spin_label}" if args.atom1 == args.atom2 else f"{args.atom1}{args.atom2}_{spin_label}"
        elif args.triatomic:
            base_system_name = f"{args.atom1}{args.center_atom}{args.atom2}_{spin_label}"
        elif args.H4_mol:
            base_system_name = f"H4_{spin_label}"
        elif args.double_chain:
            base_system_name = f"H_double_chain_{args.n_rep[0]}rep_{spin_label}"
        elif args.single_chain:
            base_system_name = f"H_single_chain_{args.n_rep[0]}rep_{spin_label}"
        elif args.tme:
            base_system_name = f"TME_{args.spin_label}"
    
    # Load existing results
    output_data = load_existing_results(args.output_file_path, base_system_name, suffix=args.file_name_suffix, units=args.units)

    # Auto-detect number of parallel jobs
    if args.n_parallel_jobs is None:
        total_cpus = cpu_count()
        args.n_parallel_jobs = max(1, total_cpus // args.n_threads_per_process - 1)
    
    print(f"🚀 Parallelization settings:")
    print(f"   Total CPUs available: {cpu_count()}")
    print(f"   Parallel jobs: {args.n_parallel_jobs}")
    print(f"   Threads per process: {args.n_threads_per_process}")
    print(f"   Total threads used: {args.n_parallel_jobs * args.n_threads_per_process}")
    
    # Run calculations based on system type
    if args.atomic:
        print("\n" + "="*60)
        print("PARALLELIZED ATOMIC CALCULATIONS")
        print("="*60)
        run_parallel_calculations(
            args=args,
            output_data=output_data,
            method_dict=method_dict,
            base_system_name=base_system_name,
            task_creator_func=create_task_list_atomic,
            system_info_creator_func=create_mol.create_system_info_atomic,
        )
    if args.diatomic:
        print("\n" + "="*60)
        print("PARALLELIZED DIATOMIC CALCULATIONS")
        print("="*60)
        run_parallel_calculations(
            args=args,
            output_data=output_data,
            method_dict=method_dict,
            base_system_name=base_system_name,
            task_creator_func=create_task_list_diatomic,
            system_info_creator_func=create_mol.create_system_info_diatomic,
        )
    
    elif args.triatomic:
        print("\n" + "="*60)
        print("PARALLELIZED TRIATOMIC CALCULATIONS")
        print("="*60)
        run_parallel_calculations(
            args=args,
            output_data=output_data,
            method_dict=method_dict,
            base_system_name=base_system_name,
            task_creator_func=create_task_list_triatomic,
            system_info_creator_func=create_mol.create_system_info_triatomic,
        )
    
    elif args.H4_mol:
        print("\n" + "="*60)
        print("PARALLELIZED H4 CALCULATIONS")
        print("="*60)
        run_parallel_calculations(
            args=args,
            output_data=output_data,
            method_dict=method_dict,
            base_system_name=base_system_name,
            task_creator_func=create_task_list_H4,
            system_info_creator_func=create_mol.create_system_info_H4,
        )
    
    elif args.single_chain:
        print("\n" + "="*60)
        print("PARALLELIZED SINGLE CHAIN CALCULATIONS")
        print("="*60)
        run_parallel_calculations(
            args=args,
            output_data=output_data,
            method_dict=method_dict,
            base_system_name=base_system_name,
            task_creator_func=create_task_list_single_chain,
            system_info_creator_func=create_mol.create_system_info_single_chain,
        )
    
    elif args.double_chain:
        print("\n" + "="*60)
        print("PARALLELIZED DOUBLE CHAIN CALCULATIONS")
        print("="*60)
        run_parallel_calculations(
            args=args,
            output_data=output_data,
            method_dict=method_dict,
            base_system_name=base_system_name,
            task_creator_func=create_task_list_double_chain,
            system_info_creator_func=create_mol.create_system_info_double_chain,
        )

    elif args.tme:
        print("\n" + "="*60)
        print("PARALLELIZED TME CALCULATIONS")
        print("="*60)
        if args.xyz_file is None:
            raise ValueError("--xyz_file must be provided when using --tme")
        run_parallel_calculations(
            args=args,
            output_data=output_data,
            method_dict=method_dict,
            base_system_name=base_system_name,
            task_creator_func=create_task_list_tme,
            system_info_creator_func=create_mol.create_system_info_TME,
        )

    # Final save after all calculations
    if base_system_name is None: 
        base_system_name = "calculation-pick-a-name"
    
    print(f"\n{'='*60}")
    print("All calculations complete. Saving final results...")
    print(f"{'='*60}")

    save_results(
        output_data,
        args.output_file_path,
        base_system_name,
        bond_length=None,
        methods=None,
        basis=None,
        suffix=args.file_name_suffix,
        units=args.units,
    )
