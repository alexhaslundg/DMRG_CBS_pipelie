import os
import time
import tempfile

import numpy as np
import scipy.stats
import scipy.optimize
from pyscf import cc, mp, scf
from pyscf import fci as pyscf_fci
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

# imports from DMRG code
import dmrg_pipeline.utils.utils as dmrg_utils
import dmrg_pipeline.orbitals.orbital_prep as dmrg_orb_prep

import json



def _checkpoint_save(result, output_dir, run_name):
    """Save current result dict to a checkpoint JSON file."""
    if output_dir is None or run_name is None:
        return
    checkpoint_path = os.path.join(output_dir, f"{run_name}_checkpoint.json")
    try:
        with open(checkpoint_path, 'w') as f:
            json.dump(dmrg_utils.convert_numpy(result), f, indent=2)
    except Exception as e:
        print(f"⚠️  Could not save checkpoint to {checkpoint_path}: {e}")


def calculate_von_neumann_entropy(rdm1, occupation_threshold=1e-10):
    """
    Calculate von Neumann entropy from a 1-particle reduced density matrix (1-RDM).
    
    S = -Tr(ρ log ρ) = -∑_i n_i log(n_i)
    
    where n_i are the natural orbital occupation numbers (eigenvalues of the 1-RDM).
    
    Parameters:
    -----------
    rdm1 : np.ndarray
        One-particle reduced density matrix (shape: [n_orbitals, n_orbitals])
    occupation_threshold : float
        Threshold below which occupation numbers are treated as zero
        (to avoid log(0) issues)
    
    Returns:
    --------
    dict : Contains entropy value and natural orbital occupations
    """
    # enforce Hermiticity
    rdm1 = (rdm1 + rdm1.T.conj()) / 2
    # enforce trace normalization
    rdm1 /= np.trace(rdm1)

    # Diagonalize the 1-RDM to get natural orbital occupation numbers
    eigenvalues, eigenvectors = np.linalg.eigh(rdm1)
    
    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    occupations = eigenvalues[idx]
    natural_orbitals = eigenvectors[:, idx]
    
    # Calculate von Neumann entropy
    # Filter out very small occupation numbers to avoid numerical issues
    valid_occupations = occupations[occupations > occupation_threshold]
    
    # S = -sum(n_i * log(n_i)) for each occupation number n_i
    entropy = -np.sum(valid_occupations * np.log(valid_occupations))
    
    return {
        "entropy": float(entropy),
        "occupations": occupations.tolist(),
        "num_significant_occupations": int(np.sum(occupations > occupation_threshold)),
        "occupation_threshold": occupation_threshold,
        "trace_rdm1": float(np.trace(rdm1))  # Should equal number of electrons
    }
# -------------------------
# HF calculation
# -------------------------
def run_hf(mol, mf_conv_tol=1e-10, unrestricted=False):
    start = time.time()
    result = {"method": "HF", "energy": None, "success": False, "error": None,
              "mf_conv_tol": mf_conv_tol, "calculation_time": None, "details": {}}

    try:
        with dmrg_utils.peak_memory_monitor() as mon:
            if unrestricted:
                mf = scf.UHF(mol)
            elif mol.spin != 0:
                # Use ROHF for open-shell restricted calculations
                mf = scf.ROHF(mol)
            else:
                mf = scf.RHF(mol)
            mf.conv_tol = mf_conv_tol
            mf.kernel()
        end = time.time()

        result.update({
            "energy": float(mf.e_tot),
            "success": bool(getattr(mf, "converged", True)),
            "calculation_time": end - start,
            "peak_memory_gb": mon.get("peak_gb"),
            "details": {"mf_converged": getattr(mf, "converged", None)}
        })
    except Exception as e:
        result.update({"error": str(e), "calculation_time": time.time()-start})

    return result

# -------------------------
# FCI calculation
# -------------------------

def run_fci(mol, mf_conv_tol=1e-10, occupation_threshold=1e-10, mf=None, unrestricted=False):
    """
    Run Full CI calculation and compute von Neumann entropy.
    
    For FCI, the 1-RDM will have fractional occupations reflecting correlation,
    resulting in non-zero entropy.
    """
    
    result = {
        "method": "FCI",
        "energy": None,
        "success": False,
        "error": None,
        "mf_conv_tol": mf_conv_tol,
        "calculation_time": None,
        "von_neumann_entropy_mo": None,
        "von_neumann_entropy_ao": None,
        "1-RDM_ao": None,
        "1-RDM_mo": None,
        "details": {}
    }
    
    try:
        with dmrg_utils.peak_memory_monitor() as mon:
            start = time.time()
            if mf is None:
                if unrestricted:
                    mf = scf.UHF(mol)
                elif mol.spin != 0:
                    # Use ROHF for open-shell restricted calculations
                    mf = scf.ROHF(mol)
                else:
                    mf = scf.RHF(mol)
                mf.conv_tol = mf_conv_tol
                mf.kernel()
            
            cis = pyscf_fci.FCI(mf)
            # Compute FCI energy and wavefunction
            fci_e, fci_civec = cis.kernel()
            
            # Get the 1-RDM from FCI
            # For spin-restricted FCI
            norb = mf.mo_coeff.shape[1]
            nelec = mol.nelectron
            
            # Calculate 1-RDM in MO basis
            rdm1_mo = cis.make_rdm1(fci_civec, norb, nelec)
            
            # Transform to AO basis if needed: P_AO = C @ P_MO @ C.T
            # But for entropy calculation, MO basis is more natural
            rdm1_ao = mf.mo_coeff @ rdm1_mo @ mf.mo_coeff.T
            
            # Calculate von Neumann entropy
            entropy_info_mo = calculate_von_neumann_entropy(rdm1_mo, occupation_threshold)
            entropy_info_ao = calculate_von_neumann_entropy(rdm1_ao, occupation_threshold)

            end = time.time()
            result.update({
                "energy": float(fci_e),
                "success": True,
                "calculation_time": end - start,
                "peak_memory_gb": mon.get("peak_gb"),
                "von_neumann_entropy_mo": entropy_info_mo,
                "von_neumann_entropy_ao": entropy_info_ao,
                "details": {
                    "mf_energy": float(mf.e_tot),
                    "correlation_energy": float(fci_e - mf.e_tot),
                    "num_electrons": nelec,
                    "num_orbitals": norb
                },
                #'1-RDM_ao': rdm1_ao.tolist(),
                #'1-RDM_mo': rdm1_mo.tolist()
            })
    except Exception as e:
        result.update({
            "error": str(e),
            "calculation_time": time.time() - start
        })
    
    return result






def backward_extrapolation(driver, mpo, ket, max_bond_dim, sweeps_per_bd=4,
                          bd_reductions=[0.9, 0.75, 0.6, 0.45, 0.25, 0.1],
                          intra_bd_energy_tol=1e-5, max_sweeps_per_bd=10,
                          verbose=1):
    """
    Perform backward extrapolation to zero discarded weight.

    For each reduced bond dimension, sweeps are run until the intra-sweep
    energy change falls below intra_bd_energy_tol (or max_sweeps_per_bd is
    reached). The last sweep of each block is used as the data point for
    extrapolation.

    Parameters:
    -----------
    driver : DMRGDriver
        Block2 DMRG driver instance
    mpo : MPO
        Matrix Product Operator
    ket : MPS
        Converged MPS state
    max_bond_dim : int
        Maximum bond dimension used in convergence
    sweeps_per_bd : int
        Minimum number of sweeps per bond dimension level (kept for
        compatibility; actual sweeps may exceed this up to max_sweeps_per_bd)
    bd_reductions : list
        Relative bond dimension multipliers for backward sweep
    intra_bd_energy_tol : float
        Energy convergence threshold within each BD block
    max_sweeps_per_bd : int
        Maximum sweeps allowed per BD block
    verbose : int
        Verbosity level
    """
    if verbose:
        print(f"\n=== Starting Backward Extrapolation ===")

    start_time = time.time()
    svd_cutoff = 1e-10
    noise_val = 1e-7
    thr_val = 1e-10

    bd_blocks = [max(8, int(max_bond_dim * r)) for r in bd_reductions]

    if verbose:
        print(f"Bond dimension blocks: {bd_blocks}")
        print(f"Convergence tol per block: {intra_bd_energy_tol:.1e}, max sweeps: {max_sweeps_per_bd}")

    # Accumulators for all individual sweeps (for extrapolation_info)
    all_ds = []
    all_dws = []
    all_eners = []
    all_bond_dims_used = []
    all_noises_used = []

    # One representative data point per BD block (last converged sweep)
    extrap_dws_list = []
    extrap_eners_list = []
    extrap_bd_list = []

    for bd in bd_blocks:
        if verbose:
            print(f"\n  BD block = {bd}:")
        block_energies = []
        for sw_idx in range(max_sweeps_per_bd):
            noise = noise_val if sw_idx == 0 else 0.0
            energy = driver.dmrg(mpo, ket, n_sweeps=1,
                                 bond_dims=[bd], noises=[noise],
                                 thrds=[thr_val], cutoff=svd_cutoff,
                                 iprint=max(0, verbose - 1))
            ds, dws, eners = driver.get_dmrg_results()
            last_dw = float(dws[-1]) if (hasattr(dws, '__len__') and len(dws) > 0) else 0.0
            ener_val = float(energy)  # driver.dmrg() always returns a scalar

            all_ds.append(int(ds[-1]) if (hasattr(ds, '__len__') and len(ds) > 0) else bd)
            all_dws.append(last_dw)
            all_eners.append(ener_val)
            all_bond_dims_used.append(bd)
            all_noises_used.append(noise)
            block_energies.append(ener_val)

            if len(block_energies) >= 2:
                delta_e = abs(block_energies[-1] - block_energies[-2])
                if verbose:
                    print(f"    Sweep {sw_idx+1}/{max_sweeps_per_bd}: E={ener_val:.10f}, dE={delta_e:.2e}, dw={last_dw:.2e}")
                if delta_e < intra_bd_energy_tol:
                    if verbose:
                        print(f"    -> Converged at BD={bd}")
                    break
            else:
                if verbose:
                    print(f"    Sweep {sw_idx+1}/{max_sweeps_per_bd}: E={ener_val:.10f}, dw={last_dw:.2e}")

        extrap_eners_list.append(block_energies[-1])
        extrap_dws_list.append(all_dws[-1])
        extrap_bd_list.append(bd)

    if len(extrap_eners_list) < 1:
        if verbose:
            print("Warning: Not enough data for extrapolation")
        return {
            'extrapolated_energy': None,
            'error': None,
            'extrapolation_info': {},
            'extrapolation_time': time.time() - start_time
        }

    extrap_dws = np.array(extrap_dws_list)
    extrap_bd = np.array(extrap_bd_list)
    extrap_eners = np.array(extrap_eners_list)
    
    if verbose:
        print('\n--- Extrapolation Data ---')
        print('Bond Dims (blocks):', extrap_bd.tolist())
        print('Discarded Weights: ', extrap_dws)
        print('Energies:          ', extrap_eners)
    
    # Linear regression E vs discarded weight
    extrapolated_energy_dws = None
    error_dws = None
    extrapolated_energy_bds = None
    error_bds = None
    
    try:
        if extrap_dws.size >= 2:
            mask = extrap_dws > 1e-15
            valid_dws = extrap_dws[mask]
            valid_eners = extrap_eners[mask]
            
            if len(valid_dws) >= 2:
                reg = scipy.stats.linregress(valid_dws, valid_eners)
                extrapolated_energy_dws = float(reg.intercept)
                error_estimate = abs(reg.intercept - np.min(valid_eners)) / max(1.0, len(valid_eners))
                error_dws = float(error_estimate)
                
                if verbose:
                    print(f'\nExtrapolated energy = {extrapolated_energy_dws:.15f} +/- {error_dws:.10f}')
                    print(f'R-squared = {reg.rvalue**2:.6f}')
            else:
                if verbose:
                    print("Not enough valid points; using highest-accuracy energy")
                extrapolated_energy_dws = float(extrap_eners[-1]) if extrap_eners.size > 0 else None
        else:
            if verbose:
                print("Insufficient data for extrapolation")
    except Exception as e:
        if verbose:
            print(f"Extrapolation failed: {e}")

    # Extrapolation info for exponential decay based on the bond dimenion 
    try: 
        def model_exp_logD(D, E_inf, A, k):
            # Formula: E(D) = E_inf + A * exp(-const * (log D)^2) from https://arxiv.org/pdf/2601.04621#page=37.10 p.39
            # A is the prefactor (energy units)
            # k is the "const" in your formula (dimensionless decay rate)
            return E_inf + A * np.exp(-k * (np.log(D))**2)

        try:
            if extrap_dws.size >= 2:
                mask = extrap_dws > 1e-15
                valid_bd = extrap_bd[mask]
                valid_eners = extrap_eners[mask]
            # Initial guesses:
            # E_inf: Slightly below min energy
            # A:     Positive scale of error
            # k:     Start with 1.0 (standard assumption)
            p0 = [valid_eners.min() - 1e-4, valid_eners.max() - valid_eners.min(), 0.5]
            
            # Bounds: k must be positive (decay), A must be positive
            bounds = ([-np.inf, 0, 0], [np.inf, np.inf, np.inf])
            
            popt, pcov = scipy.optimize.curve_fit(model_exp_logD, valid_bd, valid_eners, p0=p0, bounds=bounds, maxfev=10000)
            
            Einf_exp, A_fit, k_fit = popt
            Einf_exp_err = np.sqrt(pcov[0, 0])
            extrapolated_energy_bds = Einf_exp
            error_bds = Einf_exp_err
            
            print(f"\n[Fit 1] Exponential-in-log(D)")
            print(f"Model: E = E_inf + A * exp(-const * (ln D)^2)")
            print(f"{'-'*40}")
            print(f"  E_inf = {Einf_exp:20.15f} +/- {Einf_exp_err:10.2e}")
            print(f"  A     = {A_fit:.3e}")
            print(f"  const = {k_fit:.3f}   (fitted decay rate)")
        except Exception as e:
            if verbose:
                print(f"Exponential-in-log(D) fit failed: {e}")
    except Exception as e:
        if verbose:
            print(f"Exponential fit failed: {e}")
            extrapolated_energy_bds = None
            error_bds = None
    
    extrapolation_info = {
        "ds": all_ds,
        "dws": all_dws,
        "eners": all_eners,
        "bond_dims_used": all_bond_dims_used,
        "noises_used": all_noises_used,
        "thrds_used": [thr_val] * len(all_ds),
        "svd_cutoff": svd_cutoff,
        "extrap_dws": extrap_dws.tolist(),
        "extrap_eners": extrap_eners.tolist(),
        "extrap_bond_dims": extrap_bd.tolist()
    }
    
    if verbose:
        print(f"=== Backward Extrapolation Complete ===\n")
    
    return {
        'extrapolated_energy_dws': extrapolated_energy_dws,
        'error_dws': error_dws,
        'extrapolated_energy_bds': extrapolated_energy_bds,
        'error_bds': error_bds,
        'extrapolation_info': extrapolation_info,
        'extrapolation_time': time.time() - start_time
    }


# ============================================================================
# MAIN DMRG CALCULATION (Refactored Version)
# ============================================================================

def run_dmrg(
    mol,
    # Orbital construction options
    orbital_method='MP2',
    localize_orbitals=False,
    localization_method='PM',
    # Active space selection
    n_orbitals_for_initial_active_space=None,
    occupation_thresholds=(0.01, 1.99),
    use_dmrg_active_space_selection=True,
    as_entropy_threshold=1e-3,
    as_bond_dim=80,
    as_n_sweeps=4,
    # Energy-window pre-selection (applied before entropy DMRG)
    as_energy_window_occ=None,
    as_energy_window_virt=None,
    # Fiedler reordering
    perform_reordering=True,
    reorder_method='fiedler',
    reorder_bond_dim=50,
    reorder_sweeps=10,
    # Main DMRG parameters
    main_dmrg_symmetry='SU2',
    initial_bond_dim=100,
    max_bond_dim=1000,
    max_sweeps=30,
    energy_tol=1e-6,
    discard_tol=1e-10,
    intra_bd_energy_tol=1e-5,
    max_sweeps_per_bd=10,
    noise_schedule=None,
    svd_schedule=None,
    davidson_schedule=None,
    # Backward extrapolation
    perform_extrapolation=True,
    extrap_sweeps_per_bd=4,
    extrap_bd_reductions=None,
    # Orbital visualization options
    generate_cube_files=False,
    cube_resolution=80,
    cube_margin=3.0,
    generate_py3dmol_viz=False,
    py3dmol_n_orbitals=4,
    py3dmol_isoval=0.02,
    # General options
    mf=None,
    mf_conv_tol=1e-10,
    unrestricted=False,
    n_threads=4,
    scratch_dir=None,
    verbose=1,
    # Output options
    output_dir=None,
    run_name=None,
    # Fault-tolerance: called with a copy of result after DMRG sweep but before
    # backward extrapolation.  Survives SLURM kills that happen during extrap.
    pre_extrap_callback=None,
    # MPS persistence: if set, the converged KET is copied here after the forward
    # pass (before entropy calculation / backward extrapolation).
    save_mps_dir=None,
):
    """
    Run DMRG calculation with modular workflow:
    
    1. Construct orbitals (HF or MP2 natural orbitals)
    2. Select active space (occupation-based or DMRG entropy-based)
    3. Prepare integrals for DMRG
    4. Perform Fiedler orbital reordering
    5. Run main DMRG optimization
    6. Perform backward extrapolation
    7. (Optional) Generate orbital visualizations (cube files and py3Dmol HTML)
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object
        
    Orbital Construction:
    ---------------------
    orbital_method : str, default='MP2'
        Method for orbital construction: 'HF' or 'MP2'
    localize_orbitals : bool, default=False
        Whether to localize orbitals
    localization_method : str, default='PM'
        Localization method: 'PM' (Pipek-Mezey) or 'IBO'
        
    Active Space Selection:
    -----------------------
    n_orbitals_for_initial_active_space : int, optional
        Number of orbitals in active space (None = all)
    occupation_thresholds : tuple, default=(0.01, 1.99)
        (lower, upper) thresholds for occupation-based selection
    use_dmrg_active_space_selection : bool, default=False
        Use DMRG orbital entropies for active space selection
    as_entropy_threshold : float, default=1e-3
        Entropy threshold for DMRG-based selection
    as_bond_dim : int, default=80
        Bond dimension for active space selection DMRG
    as_n_sweeps : int, default=4
        Sweeps for active space selection DMRG
        
    Fiedler Reordering:
    -------------------
    perform_reordering : bool, default=True
        Whether to perform Fiedler orbital reordering
    reorder_method : str, default='fiedler'
        Reordering strategy: 'fiedler', 'gaopt', 'genetic'
    reorder_bond_dim : int, default=50
        Bond dimension for reordering DMRG
    reorder_sweeps : int, default=10
        Sweeps for reordering DMRG
        
    Main DMRG Parameters:
    ---------------------
    main_dmrg_symmetry : str, default='SU2'
        Symmetry type for main DMRG: 'SU2' or 'SZ'
        - SU2: Spin-adapted (faster, proper spin states, ~30-50% smaller bond dim)
        - SZ: Sz-conserving only (useful for debugging or when SU2 has issues)
    initial_bond_dim : int, default=100
        Initial bond dimension
    max_bond_dim : int, default=1000
        Maximum bond dimension
    max_sweeps : int, default=30
        Maximum sweeps for convergence
    energy_tol : float, default=1e-6
        Energy convergence tolerance (fine-tuning phase at max bond dim)
    discard_tol : float, default=1e-10
        Discarded weight tolerance (fine-tuning phase at max bond dim)
    intra_bd_energy_tol : float, default=1e-5
        Energy convergence tolerance within each bond dimension stage during
        the pretraining ramp. Sweeps at a given BD continue until |dE| < tol.
    max_sweeps_per_bd : int, default=10
        Maximum number of sweeps per bond dimension stage in the pretraining ramp.
    noise_schedule : list, optional
        Noise values per stage
    svd_schedule : list, optional
        SVD cutoffs per stage
    davidson_schedule : list, optional
        Davidson thresholds per stage
        
    Backward Extrapolation:
    -----------------------
    perform_extrapolation : bool, default=True
        Whether to perform backward extrapolation
    extrap_sweeps_per_bd : int, default=4
        Sweeps per bond dimension in extrapolation
    extrap_bd_reductions : list, optional
        Bond dimension reduction factors
        
    Orbital Visualization:
    ----------------------
    generate_cube_files : bool, default=False
        Generate cube files for selected active space orbitals
    cube_resolution : int, default=80
        Grid resolution for cube files (points per dimension)
    cube_margin : float, default=3.0
        Margin around molecule for cube files (Bohr)
    generate_py3dmol_viz : bool, default=False
        Generate interactive py3Dmol HTML visualization
    py3dmol_n_orbitals : int, default=4
        Number of orbitals to include in py3Dmol visualization
    py3dmol_isoval : float, default=0.02
        Isosurface value for py3Dmol visualization
        
    General Options:
    ----------------
    mf : pyscf.scf object, optional
        Pre-computed mean-field object
    mf_conv_tol : float, default=1e-10
        SCF convergence tolerance
    unrestricted : bool, default=False
        Use unrestricted (UHF) calculation
    n_threads : int, default=4
        Number of threads for DMRG
    scratch_dir : str, optional
        Scratch directory for DMRG
    verbose : int, default=1
        Verbosity level
        
    Output Options:
    ---------------
    output_dir : str, optional
        Directory to save output files
    run_name : str, optional
        Identifier for output files
        
    Returns
    -------
    dict : Results dictionary containing energies, timings, and metadata
    """
    print(f"Molecule Geometry is in units of {mol.unit}")
    start_time = time.time()

    # Set default schedules
    if noise_schedule is None:
        noise_schedule = [1e-3, 5e-4, 1e-4, 0.0, 0.0]
    if svd_schedule is None:
        svd_schedule = [1e-5, 5e-6, 1e-6, 1e-7, 1e-8]
    if davidson_schedule is None:
        davidson_schedule = [1e-6, 5e-7, 1e-7, 1e-8, 1e-9]
    if extrap_bd_reductions is None:
        extrap_bd_reductions = [0.9, 0.75, 0.6, 0.45, 0.25, 0.1]
    
    # Create scratch directory - ensures DMRG uses fast scratch filesystem
    if scratch_dir is None:
        scratch_dir = tempfile.mkdtemp(prefix="dmrg_main_")
    os.makedirs(scratch_dir, exist_ok=True)
    if verbose:
        print(f"📁 DMRG scratch directory: {scratch_dir}")
    
    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize result containers
    result = {
        'method': 'DMRG',
        'energy': None,
        'converged': False,
        'error': None,
        'run_name': run_name,
        'output_dir': output_dir,
    }
    
    mem_history = []
    
    if verbose:
        print("\n" + "="*70)
        print("DMRG CALCULATION (Modular Workflow)")
        print("="*70)
        print(f"Run name: {run_name}")
        print(f"Molecule: {mol.nelectron} electrons, spin={mol.spin}")
        print(f"Orbital method: {orbital_method}")
        print(f"Max bond dimension: {max_bond_dim}")
        if localize_orbitals == False:
            print("Localization: None")
        else:
            print(f"Localization: {localize_orbitals} ({localization_method})")
    
    try:
        # =====================================================================
        # STEP 1: Construct Orbitals
        # =====================================================================
        if verbose:
            print("\n" + "-"*60)
            print("STEP 1: ORBITAL CONSTRUCTION")
            print("-"*60)
        
        orbitals, occupations, orb_info, mf = dmrg_orb_prep.construct_orbitals(
            mol,
            method=orbital_method,
            localize=localize_orbitals,
            localization_method=localization_method,
            mf=mf,
            mf_conv_tol=mf_conv_tol,
            unrestricted=unrestricted,
            verbose=verbose
        )
        
        # orb_info is JSON serializable; mf is returned separately
        result['orbital_info'] = orb_info
        result['hf_energy'] = orb_info.get('energy', None)
        
        mem_history.append(dmrg_utils.get_mem_gb())
        
        # # =====================================================================
        # # STEP 1b: Energy-window pre-selection (optional)
        # # =====================================================================
        # # Reduce the orbital pool using canonical HF energies before the entropy
        # # DMRG.  This is the key fix for TZ/QZ: with ~140 MOs the entropy DMRG
        # # itself times out; cutting diffuse virtuals and deep core orbitals first
        # # brings the pool down to a manageable size (~40-60 orbs).
        # presel_info = {}
        # occupations_full_for_plot = occupations.copy()   # save for diagnostic plot
        # if as_energy_window_occ is not None or as_energy_window_virt is not None:
        #     if verbose:
        #         print("\n" + "-"*60)
        #         print("STEP 1b: ENERGY-WINDOW PRE-SELECTION")
        #         print("-"*60)
        #     presel_indices, n_presel_frozen, presel_info = \
        #         dmrg_orb_prep.preselect_by_energy_window(
        #             mol=mol,
        #             mf=mf,
        #             orbitals=orbitals,
        #             occupations=occupations,
        #             window_occ_ha=as_energy_window_occ,
        #             window_virt_ha=as_energy_window_virt,
        #             verbose=verbose,
        #         )
        #     orbitals    = orbitals[:, presel_indices]
        #     occupations = occupations[presel_indices]
        #     # Cap initial AS size to the new pool
        #     if n_orbitals_for_initial_active_space is not None:
        #         n_orbitals_for_initial_active_space = min(
        #             n_orbitals_for_initial_active_space, len(presel_indices))
        #     else:
        #         n_orbitals_for_initial_active_space = len(presel_indices)
        #     result['energy_presel_info'] = presel_info

        # =====================================================================
        # STEP 2: Active Space Selection
        # =====================================================================
        if verbose:
            print("\n" + "-"*60)
            print("STEP 2: ACTIVE SPACE SELECTION")
            print("-"*60)

        if use_dmrg_active_space_selection:

            # Run DMRG-based active space selection
            h1e_reordered, eri_reordered, ecore, n_elec, active_orbitals, selected_ordered_indices, orbital_entropies, as_energy, dmrg_info = \
                dmrg_orb_prep.select_active_space_with_DMRG(
                    mol=mol,
                    mf=mf,
                    orbitals=orbitals,
                    localized_orbitals=localize_orbitals,
                    unrestricted=unrestricted,
                    occupations=occupations,
                    initial_active_space_size=n_orbitals_for_initial_active_space,
                    n_threads=n_threads,
                    scratch_dir=os.path.join(scratch_dir, "as_select"),
                    as_bond_dim=as_bond_dim,
                    as_n_sweeps=as_n_sweeps,
                    entropy_threshold=as_entropy_threshold,
                    verbose=verbose,
                    output_dir=output_dir,
                    run_name=run_name,
                    generate_cubes=generate_cube_files,
                    cube_resolution=cube_resolution,
                    cube_margin=cube_margin,
                    generate_py3dmol=generate_py3dmol_viz,
                    py3dmol_n_orbitals=py3dmol_n_orbitals,
                    py3dmol_isoval=py3dmol_isoval
                )
            
            n_orb = active_orbitals.shape[1]
            active_occupations = occupations[selected_ordered_indices]

            result['active_space_method'] = 'DMRG_entropy'
            result['orbital_entropies'] = orbital_entropies.tolist()
            result['as_selection_energy'] = float(as_energy)
            result['dmrg_selection_info'] = dmrg_info

            # # ── Diagnostic plot ──────────────────────────────────────────────
            # if output_dir is not None and presel_info:
            #     # Build entropy-selected mask for the pre-selected pool.
            #     # dmrg_info['selected_indices_in_initial_active'] are positions
            #     # within the initial active space (= the pre-selected pool when
            #     # n_orbitals_for_initial_active_space == len(presel_pool)).
            #     n_presel = len(presel_info.get('selected_indices', []))
            #     entropy_mask = np.zeros(n_presel, dtype=bool)
            #     sel_in_init = dmrg_info.get('selected_indices_in_initial_active', [])
            #     entropy_mask[np.array(sel_in_init, dtype=int)] = True
            #     plot_path = os.path.join(
            #         output_dir,
            #         f"{run_name}_orbital_selection.png" if run_name else "orbital_selection.png"
            #     )
            #     dmrg_orb_prep.plot_orbital_selection_summary(
            #         mol=mol,
            #         mf=mf,
            #         occupations_full=occupations_full_for_plot,
            #         presel_info=presel_info,
            #         orbital_entropies=orbital_entropies,
            #         entropy_selected_mask=entropy_mask,
            #         run_name=run_name,
            #         save_path=plot_path,
            #     )
            
        else:
            # Use occupation-based selection
            active_orbitals, n_orb, n_elec, active_info = dmrg_orb_prep.select_active_space(
                orbitals=orbitals,
                occupations=occupations,
                mol=mol,
                n_active_orbitals=n_orbitals_for_initial_active_space,
                occupation_thresholds=occupation_thresholds,
                output_dir=output_dir,
                run_name=run_name,
                generate_cubes=generate_cube_files,
                cube_resolution=cube_resolution,
                cube_margin=cube_margin,
                generate_py3dmol=generate_py3dmol_viz,
                py3dmol_n_orbitals=py3dmol_n_orbitals,
                py3dmol_isoval=py3dmol_isoval,
                verbose=verbose
            )
            
            active_occupations = active_info['selected_occupations']
            selected_indices = active_info['selected_indices']
            
            # Convert numpy arrays to lists for JSON serialization
            active_info_serializable = {
                'n_orbitals': int(active_info['n_orbitals']),
                'n_electrons': int(active_info['n_electrons']),
                'selected_indices': active_info['selected_indices'].tolist() if hasattr(active_info['selected_indices'], 'tolist') else list(active_info['selected_indices']),
                'selected_occupations': active_info['selected_occupations'].tolist() if hasattr(active_info['selected_occupations'], 'tolist') else list(active_info['selected_occupations']),
                'occupation_thresholds': list(active_info['occupation_thresholds']),
                'entropy': float(active_info['entropy']),
                'n_core': int(active_info['n_core']),
                'n_active': int(active_info['n_active']),
                'n_virtual': int(active_info['n_virtual'])
            }
            
            result['active_space_method'] = 'occupation_threshold'
            result['active_space_info'] = active_info_serializable
        
        n_orb = active_orbitals.shape[1]
        # n_elec = mol.nelectron # removed because n_elec is determined by the active space selection method and may differ from total electrons
        
        if verbose:
            print(f"\nActive space: {n_elec} electrons in {n_orb} orbitals")
        
        mem_history.append(dmrg_utils.get_mem_gb())
        
        # # =====================================================================
        # # STEP 3: Prepare Integrals
        # # =====================================================================
        # if verbose:
        #     print("\n" + "-"*60)
        #     print("STEP 3: PREPARE INTEGRALS")
        #     print("-"*60)
        
        # # FIXED: Pass localized_orbitals flag to ensure correct symmetry handling
        # ncas, n_elec, spin, ecore, h1e, eri, orb_sym = dmrg_orb_prep.prepare_integrals_for_dmrg(
        #         mf, ncas=n_orb, ncore=0, verbose=verbose,
        #         localized_orbitals=localize_orbitals, unrestricted=unrestricted
        #     )

        # # FIXED: Convert orbital symmetries to integers if they are strings
        # # Block2's C++ interface cannot handle numpy string types
        # if orb_sym is not None and len(orb_sym) > 0:
        #     if isinstance(orb_sym[0], (str, np.str_)):
        #         if verbose:
        #             print(f"Warning: orb_sym contains strings, converting to integer labels")
        #         # Convert string symmetry labels to integer indices
        #         # Create a mapping from unique symmetry labels to integers
        #         unique_syms = {}
        #         orb_sym_int = []
        #         for sym in orb_sym:
        #             if sym not in unique_syms:
        #                 unique_syms[sym] = len(unique_syms)
        #             orb_sym_int.append(unique_syms[sym])
        #         orb_sym = orb_sym_int
        #         if verbose:
        #             print(f"  Symmetry mapping: {unique_syms}")
        #     else:
        #         # Already integers, ensure they are Python ints not numpy types
        #         orb_sym = [int(s) for s in orb_sym]

        # mem_history.append(dmrg_utils.get_mem_gb())

        # # =====================================================================
        # # STEP 4: Fiedler Orbital Reordering
        # # =====================================================================
        # reorder_info = {}
        # reorder_indices = None

        # # CRITICAL FIX: Store original integrals BEFORE reordering
        # # The SU2 driver will handle reordering internally with proper symmetrization
        # h1e_original = h1e.copy()
        # eri_original = eri.copy()

        # if perform_reordering:
        #     if verbose:
        #         print("\n" + "-"*60)
        #         print("STEP 4: FIEDLER ORBITAL REORDERING")
        #         print("-"*60)

        #     reorder_indices, h1e_reordered, eri_reordered, reorder_info = dmrg_orb_prep.perform_fiedler_reordering(
        #         mol=mol,
        #         h1e=h1e,
        #         eri=eri,
        #         ecore=ecore,
        #         n_elec=n_elec,
        #         spin=spin,
        #         n_threads=n_threads,
        #         scratch_dir=os.path.join(scratch_dir, "reorder"),
        #         approx_bond_dim=reorder_bond_dim,
        #         approx_sweeps=reorder_sweeps,
        #         reorder_method=reorder_method,
        #         verbose=verbose
        #     )

        #     result['reorder_info'] = reorder_info

        #     if verbose:
        #         print(f"\n✓ Orbital reordering complete. Indices saved for main DMRG.")
        #         print(f"  Reordering will be applied by DMRG driver with proper symmetrization.")

        # mem_history.append(dmrg_utils.get_mem_gb())

        # =====================================================================
        # STEP 5: Main DMRG Optimization
        # =====================================================================

        # Determine symmetry type for main DMRG
        if main_dmrg_symmetry.upper() == 'SU2':
            symm_type = SymmetryTypes.SU2
        elif main_dmrg_symmetry.upper() == 'SZ':
            symm_type = SymmetryTypes.SZ
        else:
            raise ValueError(f"Invalid main_dmrg_symmetry: {main_dmrg_symmetry}. Must be 'SU2' or 'SZ'")

        if verbose:
            print("\n" + "-"*60)
            print(f"STEP 5: MAIN DMRG OPTIMIZATION ({main_dmrg_symmetry.upper()})")
            print("-"*60)

        # Create scratch for main DMRG
        main_scratch = os.path.join(scratch_dir, "main_dmrg") if scratch_dir else None
        if main_scratch is not None:
            os.makedirs(main_scratch, exist_ok=True)
            if verbose:
                print(f"Using scratch directory: {main_scratch}")

        # Chdir to main_scratch so block2's ./nodex/ is isolated per run (avoids stale-file conflicts)
        _original_dir_main = os.getcwd()
        if main_scratch is not None:
            os.chdir(main_scratch)

        # Initialize driver with scratch directory for fast I/O
        stack_mem = int(os.environ.get('BLOCK2_MAX_MEMORY', 2 * 1024**3))
        driver = DMRGDriver(symm_type=symm_type, n_threads=n_threads, stack_mem=stack_mem)

        # # CRITICAL FIX: Disable point group symmetry when reordering is performed
        # # Fiedler reordering breaks point group symmetry structure, so we must use C1 (no symmetry)
        # # Only use orb_sym if no reordering will be applied
        # if reorder_indices is not None:
        #     if verbose:
        #         print("⚠️  Disabling point group symmetry due to orbital reordering")
        #         print("   (Reordering breaks point group structure, using C1 symmetry)")
        #     orb_sym_to_use = None  # C1 symmetry (no point group)
        # else:
        #     orb_sym_to_use = orb_sym

        driver.initialize_system(n_sites=n_orb, n_elec=n_elec, spin=mol.spin, orb_sym=None)


        # # CRITICAL FIX: Use ORIGINAL integrals + pass reorder indices to driver
        # # The driver will handle reordering internally with proper symmetrization
        # # This prevents the massive "integral symmetrize error" we were seeing
        # if verbose and reorder_indices is not None:
        #     print(f"Using original (non-reordered) integrals with reorder indices from Step 4")
        #     print(f"Driver will apply reordering internally with proper symmetrization")

        # IMPORTANT: h1e_reordered and eri_reordered are already in Fiedler-reordered form
        # from select_active_space_with_DMRG, so we don't need to pass reorder parameter
        # ecore contains nuclear repulsion energy (and frozen core contribution if ncore > 0)
        mpo = driver.get_qc_mpo(h1e=h1e_reordered, g2e=eri_reordered, ecore=ecore,
                                iprint=verbose, reorder=None) 
        
        # Helper for schedule indexing
        def sched_val(sched, idx):
            return float(sched[idx]) if idx < len(sched) else float(sched[-1])
        
        # Build bond dimension schedule
        n_stages = max(1, len(noise_schedule))
        bd_schedule = np.unique(
            np.round(np.linspace(initial_bond_dim, max_bond_dim, n_stages)).astype(int)
        ).tolist()
        if bd_schedule[0] != initial_bond_dim:
            bd_schedule.insert(0, int(initial_bond_dim))
        if bd_schedule[-1] != int(max_bond_dim):
            bd_schedule.append(int(max_bond_dim))
        
        # Initialize MPS
        ket = driver.get_random_mps(tag="KET", bond_dim=int(bd_schedule[0]), nroots=1)
        
        energies = []
        discards = []
        bond_dims = []
        effective_bond_dims = []
        noises_used = []
        svd_cuts_used = []
        davidson_used = []
        delta_es = []
        converged = False
        converged_energy = None
        converged_discard = None
        last_used_bond_dim = None

        # Pretraining phase (bond dimension ramp, one sweep per stage)
        if verbose:
            print("\nPretraining phase (bond dimension ramp):")

        for stage_idx, bd in enumerate(bd_schedule):
            bd = int(bd)
            last_used_bond_dim = bd
            noise = float(sched_val(noise_schedule, stage_idx))
            svd_cut = float(sched_val(svd_schedule, stage_idx))
            dav_thr = float(sched_val(davidson_schedule, stage_idx))

            if verbose:
                print(f"  Stage {stage_idx+1}/{len(bd_schedule)}: BD={bd}, Davidson={dav_thr:.1e}, Noise={noise:.1e}, SVD={svd_cut:.1e}")

            energy = driver.dmrg(mpo, ket, n_sweeps=1,
                                bond_dims=[bd], noises=[noise],
                                thrds=[dav_thr], cutoff=svd_cut,
                                iprint=max(0, verbose-1))

            ds, dws, eners = driver.get_dmrg_results()
            last_dw = float(dws[-1]) if (hasattr(dws, '__len__') and len(dws) > 0) else 0.0
            eff_bd = int(ds[-1]) if (hasattr(ds, '__len__') and len(ds) > 0) else bd

            energies.append(energy)
            discards.append(last_dw)
            bond_dims.append(bd)
            effective_bond_dims.append(eff_bd)
            noises_used.append(noise)
            svd_cuts_used.append(svd_cut)
            davidson_used.append(dav_thr)
            delta_es.append(None)
            mem_history.append(dmrg_utils.get_mem_gb())

            if verbose:
                print(f"    -> E={energy:.10f}, dw={last_dw:.2e}, BD_eff={eff_bd}")
        
        # Fine-tuning phase
        if verbose:
            print("\nFine-tuning phase (convergence at max bond dimension):")
        
        final_bd = int(bd_schedule[-1])
        last_used_bond_dim = final_bd
        
        for ft_idx in range(max_sweeps):
            noise = 0.0 if ft_idx > 0 else float(sched_val(noise_schedule, -1))
            svd_cut = float(sched_val(svd_schedule, -1))
            dav_thr = float(sched_val(davidson_schedule, -1))
            
            if verbose:
                print(f"  Sweep {ft_idx+1}/{max_sweeps}: BD={final_bd} SVD={svd_cut:.1e} Davidson={dav_thr:.1e} Noise={noise:.1e}")
            
            energy = driver.dmrg(mpo, ket, n_sweeps=1,
                                bond_dims=[final_bd], noises=[noise],
                                thrds=[dav_thr], cutoff=svd_cut,
                                iprint=max(0, verbose-1))

            ds, dws, eners = driver.get_dmrg_results()
            energies.append(energy)
            last_dw = float(dws[-1]) if (hasattr(dws, '__len__') and len(dws) > 0) else 0.0
            eff_bd = int(ds[-1]) if (hasattr(ds, '__len__') and len(ds) > 0) else final_bd
            discards.append(last_dw)
            bond_dims.append(final_bd)
            effective_bond_dims.append(eff_bd)
            noises_used.append(noise)
            svd_cuts_used.append(svd_cut)
            davidson_used.append(dav_thr)
            mem_history.append(dmrg_utils.get_mem_gb())

            # Check convergence
            if len(energies) >= 2:
                delta_e = abs(energies[-1] - energies[-2])
                delta_es.append(delta_e)

                if verbose:
                    print(f"    -> E={energy:.10f}, dE={delta_e:.2e}, dw={last_dw:.2e}, BD_eff={eff_bd}")

                if delta_e < energy_tol and last_dw < discard_tol:
                    converged = True
                    converged_energy = energies[-1]
                    converged_discard = discards[-1]
                    if verbose:
                        print(f"  ✓ Converged at sweep {ft_idx+1}")
                    break
            else:
                delta_es.append(None)
        
        if not converged and energies:
            converged_energy = energies[-1]
            converged_discard = discards[-1] if discards else None
            if verbose:
                print(f"  ⚠ Did not converge after {max_sweeps} sweeps")
        
        main_dmrg_time = time.time() - start_time
        
        result.update({
            'energy': converged_energy,
            'converged': converged,
            'final_bond_dim': last_used_bond_dim,
            'final_effective_bond_dim': effective_bond_dims[-1] if effective_bond_dims else None,
            'bd_saturated': (effective_bond_dims[-1] >= last_used_bond_dim) if effective_bond_dims else None,
            'sweeps_run': len(energies),
            'final_discard_weight': converged_discard,
            'energy_history': energies,
            'discard_history': discards,
            'bond_dim_history': bond_dims,
            'effective_bond_dim_history': effective_bond_dims,
            'noise_history': noises_used,
            'svd_cut_history': svd_cuts_used,
            'davidson_thr_history': davidson_used,
            'delta_e_history': delta_es,
            'memory_history_gb': mem_history,
        })

        # Checkpoint: save main DMRG result before potentially slow/failing steps
        _checkpoint_save(result, output_dir, run_name)

        # Persist the converged MPS to a permanent directory so it can be
        # reloaded later (e.g. to continue with a larger bond dimension).
        # We copy the entire main_scratch tree, which contains block2's KET
        # files, before entropy conversion / backward extrapolation can modify
        # or delete the wavefunction data.
        if save_mps_dir is not None:
            import shutil
            try:
                os.makedirs(save_mps_dir, exist_ok=True)
                for item in os.listdir(main_scratch):
                    src = os.path.join(main_scratch, item)
                    dst = os.path.join(save_mps_dir, item)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)
                print(f"💾 MPS saved to {save_mps_dir}")
                result['mps_save_dir'] = save_mps_dir
            except Exception as _mps_exc:
                print(f"⚠️  MPS save failed: {_mps_exc}")

        # =====================================================================
        # STEP 6: Orbital Entropy Calculation
        # =====================================================================
        if verbose:
            print("\n" + "-"*60)
            print("STEP 6: ORBITAL ENTROPY CALCULATION")
            print("-"*60)
        
        # Convert to SZ for entropy calculation
        ket = driver.load_mps("KET")

        # # Align MPS center before conversion (recommended practice)
        # driver.align_mps_center(ket, ref=0)

        # # FIXED: For non-singlet states, specify Sz component to extract
        # # For singlet (spin=0): Only one component, no sz argument needed
        # # For non-singlet (spin>0): Extract Sz=0 component explicitly
        # if mol.spin > 0:
        #     if verbose:
        #         print(f"\nNon-singlet state detected (spin={mol.spin}). Extracting Sz=0 component.")
        #     zket = driver.mps_change_to_sz(ket, "ZKET", sz=0)
        # else:
        #     zket = driver.mps_change_to_sz(ket, "ZKET")

        # # FIXED: Switch driver to SZ symmetry
        # # Use same orb_sym_to_use as main calculation (disabled if reordering was applied)
        # driver.symm_type = SymmetryTypes.SZ
        # driver.initialize_system(n_sites=n_orb, n_elec=n_elec, spin=mol.spin, orb_sym=orb_sym_to_use)

        # # Rebuild MPO in SZ symmetry with same reordering as main calculation
        # # CRITICAL: Must use h1e_original/eri_original with reorder parameter to match the MPS basis
        # mpo_sz = driver.get_qc_mpo(h1e=h1e_original, g2e=eri_original, ecore=ecore,
        #                            iprint=0, reorder=reorder_indices)

        # Convert to SZ symmetry for entropy calculation
        # The integrals are already reordered, so we pass them as-is
        zket, zdriver = dmrg_orb_prep.convert_su2_to_sz_symmetry(
            eri=eri_reordered,
            h1e=h1e_reordered,
            ket=ket,
            ecore=0,
            driver=driver,
            verbose=verbose,
            n_elec=n_elec,
            spin=mol.spin,
            orb_sym_int=None
        )


        # Get orbital entropies
        orb_entropies = zdriver.get_orbital_entropies(zket, orb_type=1)
        total_orbital_entropy = float(np.sum(orb_entropies))

        # Get 1-RDM for von Neumann entropy
        try:
            RDM1_a_b = zdriver.get_conventional_npdm(zket, 1)
            RDM1 = RDM1_a_b[0] + RDM1_a_b[1]
            von_neumann_info = calculate_von_neumann_entropy(RDM1)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not compute 1-RDM: {e}")
            von_neumann_info = None
        
        if verbose:
            print(f"Total orbital entropy: {total_orbital_entropy:.6f}")
        
        result.update({
            'orbital_entropies': orb_entropies.tolist(),
            'total_orbital_entropy': total_orbital_entropy,
            'von_neumann_entropy': von_neumann_info,
        })

        # Checkpoint: save with orbital entropy before backward extrapolation
        _checkpoint_save(result, output_dir, run_name)

        # Fault-tolerance: persist main result to the top-level JSON before
        # starting backward extrapolation (which can be killed by SLURM).
        if pre_extrap_callback is not None:
            try:
                pre_extrap_callback(dict(result))
            except Exception as _cb_exc:
                if verbose:
                    print(f"⚠️  Pre-extrapolation save failed: {_cb_exc}")

        # delete the zdriver and the zket
        del zdriver, zket
        
        # # =====================================================================
        # # STEP 7: Backward Extrapolation
        # # =====================================================================
        extrap_result = None
        
        if perform_extrapolation and converged:
            if verbose:
                print("\n" + "-"*60)
                print("STEP 7: BACKWARD EXTRAPOLATION")
                print("-"*60)
            
            # Switch back to SU2
            driver.symm_type = SymmetryTypes.SU2
            driver.initialize_system(n_sites=n_orb, n_elec=n_elec, spin=mol.spin)
            ket = driver.load_mps("KET")
            
            extrap_result = backward_extrapolation(
                driver, mpo, ket, last_used_bond_dim,
                sweeps_per_bd=extrap_sweeps_per_bd,
                bd_reductions=extrap_bd_reductions,
                intra_bd_energy_tol=intra_bd_energy_tol,
                max_sweeps_per_bd=max_sweeps_per_bd,
                verbose=verbose
            )
        
        if extrap_result is not None:
            result.update({
                'extrap_result': extrap_result,
            })
            # Checkpoint: save with extrapolation result
            _checkpoint_save(result, output_dir, run_name)

        # Cleanup DMRG objects and large arrays to free memory
        del driver, mpo, ket
        
        # Delete large integral and orbital arrays
        try:
            del h1e_reordered, eri_reordered, orbitals, active_orbitals, occupations, active_occupations
        except NameError:
            pass
        
        dmrg_utils.cleanup_memory()
        
    except Exception as e:
        import traceback
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
        if verbose:
            print(f"\n❌ DMRG calculation failed: {e}")
    
    # Final timing and memory
    total_time = time.time() - start_time
    result.update({
        'calculation_time': total_time,
        'memory_history_gb': mem_history,
        'peak_memory_gb': max(mem_history) if mem_history else None,
    })
    
    # Restore original directory before scratch cleanup
    try:
        os.chdir(_original_dir_main)
    except NameError:
        pass

    # Cleanup scratch
    try:
        dmrg_utils.clean_scratch_dir(scratch_dir)
    except Exception:
        pass
    
    if verbose:
        print("\n" + "="*70)
        print("DMRG CALCULATION COMPLETE")
        print("="*70)
        print(f"Final energy: {result.get('energy', 'N/A')}")
        print(f"Extrapolated: {result.get('extrapolated_energy', 'N/A')}")
        print(f"Converged: {result.get('converged', False)}")
        _eff = result.get('final_effective_bond_dim')
        _req = result.get('final_bond_dim')
        _sat = result.get('bd_saturated')
        print(f"Bond dim (requested/effective): {_req}/{_eff}  {'⚠ SATURATED — consider increasing max_bond_dim' if _sat else '✓ not saturated'}")
        print(f"Total time: {total_time:.2f}s")
        print("="*70 + "\n")
    
    return result

def run_ccsdt(mol, unrestricted=False, mf_conv_tol=1e-10, occupation_threshold=1e-10):
    """
    Run CCSD(T) calculation and compute von Neumann entropy of the 1-RDM.

    For CCSD(T), the 1-RDM will have fractional occupations reflecting correlation.
    """

    start = time.time()
    result = {
        "method": "CCSDT",
        "energy": None,
        "success": False,
        "error": None,
        "mf_conv_tol": mf_conv_tol,
        "calculation_time": None,
        "1-RDM_ao": None,
        "1-RDM_mo": None,
        "details": {}
    }

    try:
        with dmrg_utils.peak_memory_monitor() as mon:
            # === SCF step ===
            if unrestricted:
                mf = scf.UHF(mol)
            elif mol.spin != 0:
                # Use ROHF for open-shell restricted calculations
                mf = scf.ROHF(mol)
            else:
                mf = scf.RHF(mol)

            # Set convergence tolerance before running SCF
            mf.conv_tol = mf_conv_tol
            mf.kernel()

            # === CCSD(T) step ===
            if unrestricted:
                mycc = cc.UCCSD(mf)
            else:
                mycc = cc.CCSD(mf)
            mycc.kernel()
            et = mycc.ccsd_t()

            e_hf = mf.e_tot
            e_ccsd = mycc.e_tot
            e_ccsdt = e_ccsd + et

            # === 1-RDM ===
            dm1 = mycc.make_rdm1()

            # If unrestricted, make a block-diagonal matrix
            if unrestricted:
                dm1_a, dm1_b = dm1
                # dm1 = np.block([[dm1_a, np.zeros_like(dm1_a)],
                #                 [np.zeros_like(dm1_b), dm1_b]])
                dm1 = dm1_a + dm1_b  # Sum for total occupation

            # ALWAYS convert tuple to array before passing to entropy function
            if isinstance(dm1, tuple):
                dm1_a, dm1_b = dm1
                dm1_total = dm1_a + dm1_b
            else:
                dm1_total = dm1

            # Calculate von Neumann entropy (assume function exists)
            entropy_info_mo = calculate_von_neumann_entropy(dm1_total, occupation_threshold)

            # Get number of orbitals (handle unrestricted case)
            if unrestricted:
                # For unrestricted, mo_coeff is a tuple (mo_coeff_alpha, mo_coeff_beta)
                num_orbitals = mf.mo_coeff[0].shape[1]  # Both alpha and beta have same number of orbitals
            else:
                # For restricted, mo_coeff is a single array
                num_orbitals = mf.mo_coeff.shape[1]

            end = time.time()
            result.update({
                "energy": float(e_ccsdt),
                "e_ccsdt": float(e_ccsdt),
                "e_ccsd": float(e_ccsd),
                "e_hf": float(e_hf),
                "success": True,
                "calculation_time": end - start,
                "peak_memory_gb": mon.get("peak_gb"),
                "von_neumann_entropy_mo": entropy_info_mo,
                "details": {
                    "mf_energy": float(e_hf),
                    "correlation_energy": float(e_ccsdt - e_hf),
                    "num_electrons": mol.nelectron,
                    "num_orbitals": num_orbitals  # Now correctly handles both cases
                },
                #"1-RDM_mo": dm1.tolist()
            })

    except Exception as e:
        import traceback
        result.update({
            "error": str(e),
            "traceback": traceback.format_exc(),  # Add full traceback for debugging
            "calculation_time": time.time() - start
        })

    return result

def run_mp2(
    mol,
    mf_conv_tol=1e-10,
    mf=None,
    unrestricted=False,
    verbose=1
):
    """
    Run MP2 calculation with support for both restricted and unrestricted cases.
    
    Parameters:
    -----------
    mol : pyscf.gto.Mole
        Molecule object
    mf_conv_tol : float
        SCF convergence tolerance
    mf : pyscf.scf object or None
        Pre-computed mean-field object. If None, will compute HF/UHF
    unrestricted : bool
        Whether to use unrestricted formalism
    verbose : int
        Verbosity level
        
    Returns:
    --------
    dict : Results dictionary containing energies and calculation details
    """
    import time
    import numpy as np
    from pyscf import scf, mp
    
    start_time = time.time()
    
    # -------------------------
    # Mean-field calculation
    # -------------------------
    if mf is None:
        if unrestricted:
            mf = scf.UHF(mol)
        else:
            mf = scf.RHF(mol)
        mf.conv_tol = mf_conv_tol
        mf.kernel()
        if not getattr(mf, 'converged', False):
            print(f"Warning: HF did not converge to mf_conv_tol={mf_conv_tol}")
    
    # Detect calculation type
    is_uhf = isinstance(unrestricted, bool) and unrestricted
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"MP2 CALCULATION")
        print(f"{'='*60}")
        print(f"Calculation type: {'Unrestricted' if is_uhf else 'Restricted'}")
        print(f"Total electrons: {mol.nelectron}")
        print(f"HF energy: {mf.e_tot:.8f} Ha")
    
    # -------------------------
    # MP2 calculation
    # -------------------------
    try:
        mp2 = mp.MP2(mf)
        mp2.kernel()
        mp2_calc_time = time.time() - start_time
        
        if verbose:
            print(f"\nMP2 Results:")
            print(f"  Correlation energy: {mp2.e_corr:.8f} Ha")
            print(f"  Total energy: {mp2.e_tot:.8f} Ha")
            print(f"  Calculation time: {mp2_calc_time:.2f} s")
        
        # Get natural orbitals if available
        matorbs = None
        natorbs = None
        rdm1_avg = None
        
        if hasattr(mp2, "make_rdm1"):
            rdm1 = mp2.make_rdm1(ao_repr=False)
            
            if is_uhf:
                # For UHF, average alpha and beta densities
                if isinstance(rdm1, (tuple, list)):
                    rdm1_alpha, rdm1_beta = rdm1
                    rdm1_avg = (rdm1_alpha + rdm1_beta) / 2.0
                else:
                    rdm1_avg = rdm1
            

                matorbs, natorbs_trans = np.linalg.eigh(rdm1_avg)
                matorbs = matorbs[::-1]
                natorbs_trans = natorbs_trans[:, ::-1]
                natorbs = mf.mo_coeff[0] @ natorbs_trans
            else:
                # Restricted case
                if np.ndim(rdm1) == 2:
                    matorbs, natorbs_trans = np.linalg.eigh(rdm1)
                    matorbs = matorbs[::-1]
                    natorbs_trans = natorbs_trans[:, ::-1]
                    natorbs = mf.mo_coeff @ natorbs_trans
                    rdm1_avg = rdm1
        
        # calculate the von Neumann entropy when density is available
        vn_entropy = calculate_von_neumann_entropy(rdm1_avg) if rdm1_avg is not None else None
        
        if verbose and matorbs is not None:
            print(f"\nNatural Orbital Occupations:")
            print(f"  Largest: {matorbs[:5]}")
            print(f"  Smallest: {matorbs[-5:]}")
        
        return {
            'converged': True,
            'hf_energy': mf.e_tot,
            'mp2_corr_energy': mp2.e_corr,
            'energy': mp2.e_tot,
            "von_neumann_entropy_mo": vn_entropy,
            'calculation_time': mp2_calc_time,
            'matorbs': matorbs,
            'is_uhf': is_uhf,
            'error': None
        }
        
    except Exception as e:
        if verbose:
            print(f"\nMP2 calculation failed: {e}")
        
        return {
            'converged': False,
            'hf_energy': mf.e_tot,
            'mp2_corr_energy': None,
            'energy': None,
            "von_neumann_entropy_mo": None,
            'calculation_time': time.time() - start_time,
            'matorbs': None,
            'is_uhf': is_uhf,
            'error': str(e)
        }