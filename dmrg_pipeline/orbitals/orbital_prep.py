"""
Orbital Construction and Active Space Selection Module (FIXED VERSION)

This module provides functionality for:
1. Constructing molecular orbitals (HF, MP2 natural orbitals)
2. Orbital transformations (localization, natural orbitals)
3. Active space selection based on occupation thresholds
4. Visualization of orbital occupations
5. Cube file generation for orbital visualization

FIXES APPLIED:
- Added missing occupations assignment in HF case
- Removed circular imports for visualization functions
- Fixed mf.mo_coeff update to match transformed orbitals
- Fixed return value consistency
- Added proper handling of custom orbital subsets in DMRG functions
"""

import tempfile
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from pyscf import scf, mp, lo
from pyscf.tools import cubegen
import os
import time
import copy  # ADDED: needed for mf copying
import warnings
import json  # ADDED: for consolidated orbital data output
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock2._pyscf.ao2mo import integrals as itg


def construct_orbitals(mol, method='MP2', localize=False, localization_method='PM',
                       mf=None, mf_conv_tol=1e-10, unrestricted=False, verbose=1):
    """
    Construct molecular orbitals using specified method and optional transformations.
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object
    method : str, default='MP2'
        Orbital construction method: 'HF' or 'MP2'
    localize : bool, default=False
        Whether to localize orbitals (Pipek-Mezey or IBO)
    localization_method : str, default='PM'
        Localization scheme: 'PM' (Pipek-Mezey) or 'IBO' (Intrinsic Bond Orbitals)
    mf : pyscf.scf object, optional
        Pre-computed mean-field object (if None, will be computed)
    mf_conv_tol : float, default=1e-10
        SCF convergence tolerance
    unrestricted : bool, default=False
        Use unrestricted (UHF/UMP2) calculation
    verbose : int, default=1
        Verbosity level
        
    Returns
    -------
    orbitals : np.ndarray
        Orbital coefficient matrix (AO basis → MO basis)
    occupations : np.ndarray
        Orbital occupation numbers
    info : dict
        Dictionary containing calculation metadata
    mf : pyscf.scf object
        Mean-field object with mo_coeff updated to match final orbitals
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"ORBITAL CONSTRUCTION")
        print(f"{'='*70}")
        print(f"Method: {method}")
        print(f"Restricted: {not unrestricted}")
        if localize:
            print(f"Localize: {localize}")
            print(f"Localization method: {localization_method}")
        else: 
            print("Localize: False")
    
    info = {
        'method': method,
        'unrestricted': unrestricted,
        'localized': localize,
        'use_natural_orbitals': False,
        'calculation_time': 0.0
    }
    
    start_time = time.time()
    
    # -------------------------
    # 1. Mean-field calculation
    # -------------------------
    if mf is None:
        if verbose:
            if unrestricted:
                scf_type = "UHF"
            elif mol.spin != 0:
                scf_type = "ROHF"
            else:
                scf_type = "RHF"
            print(f"\nRunning {scf_type} calculation...")

        # Choose appropriate SCF method
        if unrestricted:
            mf = scf.UHF(mol)
        elif mol.spin != 0:
            # Use ROHF for open-shell restricted calculations
            mf = scf.ROHF(mol)
        else:
            # Use RHF for closed-shell
            mf = scf.RHF(mol)
        # Remove near-linearly-dependent AOs via canonical orthogonalization.
        # scf.addons.remove_linear_dep_ patches mf._eigh; it fires when cond(S) > 1/lindep_trigger (default 1e10).
        # threshold=1e-6 discards overlap eigenvalues < 1e-6 (more aggressive than the 1e-8 default,
        # needed for compressed geometries like d=1.0 bohr cc-pVQZ where cond ~ 1e10).
        scf.addons.remove_linear_dep_(mf, threshold=1e-6)
        mf.conv_tol = mf_conv_tol
        mf.kernel()

        if not mf.converged:
            print(f"Warning: HF did not converge (tol={mf_conv_tol})")
            _attempt_scf_recovery(mol, mf, mf_conv_tol, verbose)
    
    # Check overlap matrix condition
    _check_overlap_condition(mol, verbose)
    
    n_orbitals = mf.mo_coeff[0].shape[1] if unrestricted else mf.mo_coeff.shape[1]
    
    if verbose:
        print(f"HF Energy: {mf.e_tot:.8f} Ha")
        print(f"Total orbitals: {n_orbitals}")
    
    # -------------------------
    # 2. Orbital construction (HF or MP2)
    # -------------------------
    if method.upper() == 'HF':
        if unrestricted:
            orbitals = (mf.mo_coeff[0] + mf.mo_coeff[1]) / 2.0
            occupations = (mf.mo_occ[0] + mf.mo_occ[1])
            if verbose:
                print("Warning: Averaging alpha/beta orbitals for UHF")
        else:
            orbitals = mf.mo_coeff
            occupations = mf.mo_occ  # FIXED: Added this line
        
        info['energy'] = mf.e_tot

    elif method.upper() == 'MP2':
        # Compute MP2 natural orbitals
        if verbose:
            print(f"\nComputing {'U' if unrestricted else 'R'}MP2 natural orbitals...")
        
        try:
            mp2 = mp.MP2(mf)
            mp2.kernel()
            
            info['mp2_correlation_energy'] = mp2.e_corr
            info['energy'] = mp2.e_tot
            info['use_natural_orbitals'] = True
            
            if verbose:
                print(f"MP2 correlation energy: {mp2.e_corr:.8f} Ha")
                print(f"MP2 total energy: {mp2.e_tot:.8f} Ha")
            
            # Construct natural orbitals from MP2 density
            orbitals, occupations = _construct_natural_orbitals(
                mp2, mf, unrestricted, verbose
            )
            
        except Exception as e:
            print(f"Warning: MP2 failed ({e}). Falling back to HF orbitals.")
            if unrestricted:
                orbitals = mf.mo_coeff[0]
                occupations = (mf.mo_occ[0] + mf.mo_occ[1]) / 2.0
            else:
                orbitals = mf.mo_coeff
                occupations = mf.mo_occ
            info['energy'] = mf.e_tot
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'HF' or 'MP2'.")
    
    # -------------------------
    # 3. Orbital localization (optional)
    # -------------------------
    if localize:
        if verbose:
            print(f"\nLocalizing orbitals using {localization_method}...")
        
        orbitals = _localize_orbitals(
            mol, orbitals, occupations, 
            localization_method, verbose
        )
        info['localized'] = True
    
    # FIXED: Update mf.mo_coeff with final transformed orbitals
    # This ensures downstream DMRG functions use the correct orbitals
    if method.upper() == 'MP2' or localize:
        if verbose:
            print(f"\nUpdating mf.mo_coeff with transformed orbitals...")
        mf_updated = copy.copy(mf)
        mf_updated.mo_coeff = orbitals
        mf_updated.mo_occ = occupations
        mf = mf_updated
    
    info['calculation_time'] = time.time() - start_time
    
    if verbose:
        print(f"\nOrbital construction completed in {info['calculation_time']:.2f}s")
        print(f"Final orbital shape: {orbitals.shape}")
        print(f"{'='*70}\n")
    
    return orbitals, occupations, info, mf  # FIXED: Consistently return 4 values


def _attempt_scf_recovery(mol, mf, conv_tol, verbose):
    """Attempt SCF recovery with level shift if initial convergence fails."""
    try:
        if verbose:
            print("Attempting SCF recovery with level shift...")
        
        mf_recover = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
        scf.addons.remove_linear_dep_(mf_recover, threshold=1e-6)
        mf_recover.conv_tol = conv_tol
        mf_recover.level_shift = 0.5
        mf_recover.max_cycle = max(50, getattr(mf, 'max_cycle', 50))
        mf_recover.kernel()
        
        if mf_recover.converged:
            mf = mf_recover
            if verbose:
                print("SCF recovery successful")
    except Exception as e:
        if verbose:
            print(f"SCF recovery failed: {e}")


def _check_overlap_condition(mol, verbose):
    """Check overlap matrix condition number for numerical stability."""
    try:
        S = mol.intor('int1e_ovlp')
        cond_S = np.linalg.cond(S)
        
        if cond_S > 1e3:
            print(f"WARNING: Overlap matrix condition number = {cond_S:.3e}")
            print("         Numerical instability may occur")
        elif verbose:
            print(f"Overlap condition number: {cond_S:.3e}")
    except Exception as e:
        print(f"Could not compute overlap condition: {e}")


def _construct_natural_orbitals(mp2, mf, unrestricted, verbose):
    """Construct natural orbitals from MP2 density matrix."""
    
    if not hasattr(mp2, 'make_rdm1'):
        raise AttributeError("MP2 object lacks make_rdm1 method")
    
    rdm1 = mp2.make_rdm1(ao_repr=False)
    
    if unrestricted:
        # Average alpha and beta densities
        if isinstance(rdm1, (tuple, list)):
            rdm1_avg = (rdm1[0] + rdm1[1]) / 2.0
        else:
            rdm1_avg = rdm1
        
        # Diagonalize to get natural orbitals
        occupations, natorbs_mo = np.linalg.eigh(rdm1_avg)
        
        # Sort by descending occupation
        idx = np.argsort(occupations)[::-1]
        occupations = occupations[idx]
        natorbs_mo = natorbs_mo[:, idx]
        
        # Transform to AO basis using alpha MOs
        orbitals = mf.mo_coeff[0] @ natorbs_mo
        
    else:
        # Restricted case
        occupations, natorbs_mo = np.linalg.eigh(rdm1)
        
        # Sort by descending occupation
        idx = np.argsort(occupations)[::-1]
        occupations = occupations[idx]
        natorbs_mo = natorbs_mo[:, idx]
        
        # Transform to AO basis
        orbitals = mf.mo_coeff @ natorbs_mo
    
    if verbose:
        print(f"Natural orbital occupation range: [{occupations.min():.6f}, {occupations.max():.6f}]")
    
    return orbitals, occupations


def _localize_orbitals(mol, orbitals, occupations, method, verbose):
    """
    Localize molecular orbitals.
    
    Currently implements localization for occupied orbitals only.
    """
    
    # Determine number of occupied orbitals
    n_occ = int(np.sum(occupations > 0.5))
    
    if n_occ == 0:
        print("Warning: No occupied orbitals found for localization")
        return orbitals
    
    C_occ = orbitals[:, :n_occ]
    
    if method.upper() == 'PM':
        # Pipek-Mezey localization
        loc_orbs = lo.PM(mol, C_occ).kernel()
    elif method.upper() == 'IBO':
        # Intrinsic Bond Orbitals
        loc_orbs = lo.ibo.ibo(mol, C_occ)
    else:
        raise ValueError(f"Unknown localization method: {method}")
    
    # Combine localized occupied and original virtual orbitals
    orbitals_localized = orbitals.copy()
    orbitals_localized[:, :n_occ] = loc_orbs
    
    if verbose:
        print(f"Localized {n_occ} occupied orbitals")
    
    return orbitals_localized


def prepare_integrals_for_dmrg(mf, ncas, ncore=0, verbose=1, localized_orbitals=False, unrestricted=False):
    """
    Prepare integrals for DMRG calculations using Block2's built-in functions.
    
    This is the recommended approach from Block2 documentation.
    
    Parameters
    ----------
    mf : pyscf.scf object
        Mean-field object (RHF or UHF)
    ncas : int
        Number of active space orbitals
    ncore : int, default=0
        Number of core orbitals
    verbose : int, default=1
        Verbosity level
    localized_orbitals : bool, default=False
        Whether the orbitals are localized (IBO, Boys, PM, etc.).
        If True, uses C1 symmetry (no point group symmetry).
        If False, uses the molecule's point group symmetry.
    unrestricted : bool, default=False
        Whether to use unrestricted (UHF) or restricted (RHF) integrals.
        If True, uses get_uhf_integrals.
        If False, uses get_rhf_integrals.
        
    Returns
    -------
    ncas : int
        Number of active orbitals
    n_elec : int or tuple
        Number of electrons (int for RHF, tuple for UHF)
    spin : int
        Spin quantum number
    ecore : float
        Core energy
    h1e : np.ndarray or tuple
        One-electron integrals
    g2e : np.ndarray or tuple
        Two-electron integrals (with 8-fold permutational symmetry)
    orb_sym : list
        Orbital symmetry labels
        
    Notes
    -----
    g2e_symm=8 specifies 8-fold permutational symmetry for 2-electron integrals,
    which is the maximum symmetry for real orbitals and provides optimal efficiency.
    
    pg_symm controls point group symmetry:
    - pg_symm=True: Use molecule's point group (C2v, D2h, etc.) - for canonical/natural orbitals
    - pg_symm=False: Use C1 symmetry (no symmetry) - for localized orbitals
    
    Localized orbitals (IBO, Boys, PM, ER, FB) don't respect point group symmetry,
    so we must use pg_symm=False (C1 symmetry) to avoid "orbitals not strictly 
    symmetrized" errors.
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PREPARING INTEGRALS FOR DMRG")
        print(f"{'='*60}")
        print(f"Number of core orbitals: {ncore}")
        print(f"Number of active orbitals: {ncas}")
    
    # Determine point group symmetry setting
    # Localized orbitals don't respect point group symmetry -> use C1
    use_point_group_symmetry = not localized_orbitals
    
    if verbose:
        if unrestricted:
            print("Orbital type: Unrestricted (UHF)")
        else:
            print("Orbital type: Restricted (RHF)")
            
        if localized_orbitals:
            print("Orbital localization: Yes (using C1 symmetry)")
        else:
            print("Orbital localization: No (using point group symmetry)")
            
        print("2-electron integral symmetry: 8-fold permutational symmetry")
    
    # Select appropriate integral preparation function
    if unrestricted:
        # Unrestricted (UHF) case
        if verbose:
            print("\nCalling get_uhf_integrals...")
            
        result = itg.get_uhf_integrals(
            mf, 
            ncore, 
            ncas, 
            g2e_symm=8,                      # 8-fold permutational symmetry
            pg_symm=use_point_group_symmetry  # Point group symmetry (False for localized)
        )
    else:
        # Restricted (RHF) case
        if verbose:
            print("\nCalling get_rhf_integrals...")
            
        result = itg.get_rhf_integrals(
            mf, 
            ncore, 
            ncas, 
            g2e_symm=8,                      # 8-fold permutational symmetry
            pg_symm=use_point_group_symmetry  # Point group symmetry (False for localized)
        )
    
    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = result
    
    if verbose:
        print(f"\nIntegral preparation results:")
        print(f"  Core energy: {ecore:.8f} Ha")
        print(f"  Number of electrons in active space: {n_elec}")
        print(f"  Spin: {spin}")
        
        # Determine actual symmetry used
        if not use_point_group_symmetry or all(s == 0 for s in orb_sym):
            symm_name = "C1 (no point group symmetry)"
        else:
            # Try to get actual point group name
            if hasattr(mf, 'mol') and hasattr(mf.mol, 'groupname'):
                symm_name = mf.mol.groupname
            else:
                symm_name = "Point group symmetry"
        
        print(f"  Point group: {symm_name}")
        print(f"  Orbital symmetries: {orb_sym[:min(10, len(orb_sym))]}{'...' if len(orb_sym) > 10 else ''}")
        print(f"{'='*60}\n")
    
    return ncas, n_elec, spin, ecore, h1e, g2e, orb_sym


def convert_su2_to_sz_symmetry(eri, h1e, ket, ecore, driver, verbose, n_elec, spin, orb_sym_int=None):
    """
    Convert MPS from SU(2) symmetry to SZ symmetry for entropy and ordering calculations.

    This function performs the following steps:
    1. Aligns the MPS center for stable conversion
    2. Extracts CSF coefficients (for diagnostics)
    3. Converts the MPS from SU(2) to SZ symmetry
    4. Reconfigures the driver to use SZ symmetry
    5. Rebuilds the MPO in SZ symmetry
    6. Properly normalizes the converted MPS

    IMPORTANT: All DMRG calculations should be performed in SU(2) symmetry for
    computational efficiency. Only convert to SZ when needed for:
    - Computing orbital entropies (get_orbital_entropies)
    - Computing orbital interaction matrices for Fiedler ordering
    - Other SZ-specific analyses

    Parameters
    ----------
    eri : np.ndarray
        Two-electron integrals (g2e) in chemist notation
    h1e : np.ndarray
        One-electron integrals (core Hamiltonian)
    ket : MPS
        The MPS wavefunction in SU(2) symmetry (from DMRG optimization)
    ecore : float
        Core energy (nuclear repulsion + core electron energy)
    driver : DMRGDriver
        Block2 DMRG driver object (will be modified to use SZ symmetry)
    verbose : int
        Verbosity level (0=silent, 1=normal, 2+=detailed)
    n_elec : int or tuple
        Number of electrons in the active space
    spin : int
        Total spin quantum number (2S, where S is total spin)
    orb_sym_int : list of int, optional
        Orbital symmetry labels (irrep indices). If None, uses trivial symmetry (all 0).

    Returns
    -------
    zket : MPS
        The MPS in SZ symmetry, properly normalized and ready for:
        - Entropy calculations via driver.get_orbital_entropies(zket)
        - Interaction matrix via driver.get_orbital_interaction_matrix(zket)
        - Other SZ-based analyses

    Notes
    -----
    - This function modifies the driver object in-place (changes to SZ symmetry)
    - After calling this function, the driver will be configured for SZ symmetry
    - The returned zket is normalized such that <zket|zket> = 1
    - For non-singlet states (spin > 0), extracts the Sz=0 component

    For Fiedler Ordering:
    - You only need the returned zket to compute orbital interaction matrix
    - No need to return h1e, eri in SZ - these are orbital-basis integrals, not symmetry-specific
    - The driver is already reconfigured to SZ, so you can immediately call:
        minfo = driver.get_orbital_interaction_matrix(zket)
        reorder_idx = driver.orbital_reordering_interaction_matrix(minfo, strategy='fiedler')
    """

    # Determine number of orbitals from h1e shape
    if h1e.ndim == 1:
        n_orb = int(np.sqrt(len(h1e)))
    elif h1e.ndim == 2:
        n_orb = h1e.shape[0]
    else:
        raise ValueError(f"Unexpected h1e dimensions: {h1e.ndim}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"CONVERTING SU(2) → SZ SYMMETRY")
        print(f"{'='*60}")
        print(f"Number of orbitals: {n_orb}")
        print(f"Number of electrons: {n_elec}")
        print(f"Spin (2S): {spin}")

    # Step 1: Align MPS center before conversion (recommended by Block2 developers)
    # This ensures the MPS is in canonical form for stable conversion
    driver.align_mps_center(ket, ref=0)

    # Step 2: Extract CSF coefficients for diagnostic purposes
    # This shows the dominant configuration state functions
    if verbose >= 2:
        print("\nDominant CSF configurations (cutoff=0.05):")
    csfs, coeffs = driver.get_csf_coefficients(ket, cutoff=0.05, iprint=verbose)

    # Step 3: Convert MPS from SU(2) to SZ symmetry
    # For non-singlet states (spin > 0), we extract the Sz=0 component
    # For singlet states (spin = 0), only one Sz component exists
    if spin > 0:
        if verbose:
            print(f"\nNon-singlet state detected (spin={spin}).")
            print(f"Extracting Sz=0 component for entropy/interaction calculations.")
        zket = driver.mps_change_to_sz(ket, "TEMP_SZ_MPS", sz=0)
    else:
        if verbose:
            print(f"\nSinglet state detected (spin=0). Converting to SZ.")
        zket = driver.mps_change_to_sz(ket, "TEMP_SZ_MPS")

    # Step 4: Reconfigure driver to use SZ symmetry
    # This modifies the driver in-place for subsequent SZ operations
    if verbose:
        print(f"\nReconfiguring driver to SZ symmetry...")

    driver.symm_type = SymmetryTypes.SZ

    # Use provided orbital symmetries, or default to trivial (all irrep 0)
    if orb_sym_int is None:
        orb_sym_int = [0] * n_orb
        if verbose >= 2:
            print(f"Using trivial orbital symmetries (all irrep 0)")

    driver.initialize_system(n_sites=n_orb, n_elec=n_elec, spin=spin, orb_sym=orb_sym_int)

    # Step 5: Rebuild MPO in SZ symmetry
    # The Hamiltonian must be reconstructed in the new symmetry basis
    if verbose:
        print(f"Rebuilding MPO in SZ symmetry...")

    mpo_sz = driver.get_qc_mpo(h1e=h1e, g2e=eri, ecore=ecore, iprint=0)

    # Step 6: Normalize the converted MPS
    # After conversion, the MPS norm may not be exactly 1
    if verbose:
        print(f"\nNormalizing converted MPS...")

    impo = driver.get_identity_mpo()

    # Check energy before normalization (for diagnostics)
    energy_before_norm = driver.expectation(zket, mpo_sz, zket)
    norm2_before = driver.expectation(zket, impo, zket)

    if verbose >= 2:
        print(f"  Norm² before normalization: {norm2_before:.10f}")
        print(f"  Energy before normalization: {energy_before_norm:.10f} Ha")

    # Normalize: <zket|zket> should equal 1
    norm = np.sqrt(norm2_before)
    if abs(norm - 1.0) > 1e-8:
        zket.iscale(1.0 / norm)
        if verbose >= 2:
            print(f"  Applied normalization factor: {1.0/norm:.10f}")
    else:
        if verbose >= 2:
            print(f"  MPS already normalized (norm ≈ 1)")

    # Verify normalization
    norm2_after = driver.expectation(zket, impo, zket)
    energy_after_norm = driver.expectation(zket, mpo_sz, zket) / norm2_after

    if verbose:
        print(f"\n  Norm² after normalization: {norm2_after:.10f}")
        print(f"  Energy after normalization: {energy_after_norm:.10f} Ha")

        # Energy should be preserved (just rescaled by norm)
        expected_energy = energy_before_norm / norm2_before
        energy_diff = abs(energy_after_norm - expected_energy)
        if energy_diff > 1e-6:
            print(f"  WARNING: Energy changed by {energy_diff:.2e} Ha during normalization!")
        else:
            print(f"  ✓ Energy preserved within tolerance ({energy_diff:.2e} Ha)")

        print(f"{'='*60}\n")

    # delete unused variables to free memory
    del csfs, coeffs, mpo_sz, impo
    
    return zket, driver

def preform_fielder_ordering(h1e, eri, driver, zket, reorder_method = "fiedler", verbose=1):
    """Function to perform Fiedler ordering using the orbital interaction matrix from the driver."""
    # Compute optimal reordering
    minfo = driver.get_orbital_interaction_matrix(zket)
    reorder_indices = driver.orbital_reordering_interaction_matrix(minfo, strategy=reorder_method)

    if verbose:
        print(f"\nReordering indices ({reorder_method}):")
        print(f"  {reorder_indices.tolist()}")

    # Reorder Hamiltonian tensors
    if verbose:
        print(f"\nReordering one-body and two-body tensors...")

    # Ensure h1e and eri are numpy arrays with proper dimensions
    h1e = np.asarray(h1e)
    eri = np.asarray(eri)

    if verbose:
        print(f"h1e shape: {h1e.shape}, eri shape: {eri.shape}")

    # Handle h1e: reshape to 2D if needed
    if h1e.ndim == 1:
        # Reshape 1D array to 2D square matrix
        n_orb_h1e = int(np.sqrt(len(h1e)))
        if n_orb_h1e * n_orb_h1e != len(h1e):
            raise ValueError(f"Cannot reshape h1e of length {len(h1e)} to square matrix")
        h1e = h1e.reshape(n_orb_h1e, n_orb_h1e)

    # Get number of orbitals from h1e shape (now guaranteed to be 2D)
    n_orb = h1e.shape[0]

    # Reorder h1e (always 2D after reshape above)
    h1e_reordered = h1e[np.ix_(reorder_indices, reorder_indices)]

    # Handle eri reordering: unpack if compressed, then reorder
    if eri.ndim == 1:
        # 1D compressed format - unpack it first using Block2's built-in function
        if verbose:
            print(f"  Unpacking compressed ERI format...")
        eri = driver.unpack_g2e(eri, n_sites=n_orb)
        if verbose:
            print(f"  Unpacked ERI shape: {eri.shape}")
    
    # Now eri should be 4D (after unpacking) or already 4D/2D
    if eri.ndim == 4:
        # Full 4D tensor format - apply 4D permutation
        eri_reordered = eri[np.ix_(reorder_indices, reorder_indices, reorder_indices, reorder_indices)]
    elif eri.ndim == 2:
        # 2D packed format (n_orbs^2 x n_orbs^2) - reorder as 2D then reshape
        eri_reordered = eri[np.ix_(reorder_indices, reorder_indices)]
        eri_reordered = eri_reordered[:, np.ix_(reorder_indices, reorder_indices)].reshape(eri.shape)
    else:
        raise ValueError(f"Unexpected eri shape after unpacking: {eri.shape}")
    
    # Prepare info dictionary
    reorder_info = {
        'reorder_indices': reorder_indices.tolist(),
        'reorder_method': reorder_method,
    }
    
    if verbose:
        print(f"{'='*60}\n")
    
    # Cleanup
    del driver, zket
    
    return reorder_indices,reorder_info, eri_reordered, h1e_reordered




def apply_selection_and_reordering(orbitals, occupations, dmrg_info, verbose=1):
    """
    Helper function to apply both entropy selection and Fiedler reordering to orbitals.

    This function takes the output from select_active_space_with_DMRG and applies
    the selection and reordering to get the final orbital set ready for production DMRG.

    Parameters
    ----------
    orbitals : np.ndarray
        Full orbital coefficient matrix (AO × MO)
    occupations : np.ndarray
        Full occupation number array
    dmrg_info : dict
        Output dictionary from select_active_space_with_DMRG containing index mappings
    verbose : int, default=1
        Verbosity level

    Returns
    -------
    selected_reordered_orbitals : np.ndarray
        Orbital coefficients after selection and reordering (AO × n_selected)
    selected_reordered_occs : np.ndarray
        Occupation numbers after selection and reordering
    final_to_original_map : np.ndarray
        Maps final orbital index to original orbital index
    """

    # Extract index mappings
    idx_map = dmrg_info['index_mappings']

    # Get selected orbitals in original space
    selected_orig_indices = np.array(idx_map['selected_in_original'])

    # Get reordering permutation (tells us how to reorder the initial active space)
    reorder_perm = np.array(idx_map['reorder_permutation'])

    # Get which positions in reordered space correspond to selected orbitals
    selected_reordered_positions = np.array(idx_map['selected_in_reordered'])

    # Apply selection: get the selected orbital coefficients
    selected_orbitals = orbitals[:, selected_orig_indices]
    selected_occs = occupations[selected_orig_indices]

    # Apply reordering: reorder the selected orbitals according to Fiedler
    # We need to figure out the correct permutation for the selected subset
    # Since selected_reordered_positions tells us where each selected orbital
    # ends up in the reordered space, we sort by these positions
    reorder_for_selected = np.argsort(selected_reordered_positions)

    selected_reordered_orbitals = selected_orbitals[:, reorder_for_selected]
    selected_reordered_occs = selected_occs[reorder_for_selected]
    final_to_original_map = selected_orig_indices[reorder_for_selected]


    if verbose:
        print(f"\n{'='*60}")
        print(f"APPLYING SELECTION & REORDERING")
        print(f"{'='*60}")
        print(f"Input: {orbitals.shape[1]} total orbitals")
        print(f"After selection: {len(selected_orig_indices)} orbitals")
        print(f"After reordering: {len(selected_reordered_orbitals[0])} orbitals (same count)")
        print(f"\nFinal orbital order (original indices):")
        print(f"  {final_to_original_map.tolist()}")
        print(f"\nFinal occupations (reordered):")
        for i, (orig_idx, occ) in enumerate(zip(final_to_original_map, selected_reordered_occs)):
            print(f"  Position {i:2d}: Orig[{orig_idx:3d}] = {occ:.6f}")
        print(f"{'='*60}\n")

    return selected_reordered_orbitals, selected_reordered_occs, final_to_original_map


def preselect_by_energy_window(mol, mf, orbitals, occupations,
                               window_occ_ha=None, window_virt_ha=None,
                               verbose=1):
    """
    Pre-filter the orbital pool using canonical HF orbital energies before the
    entropy-selection DMRG run.  This is the recommended first step for large
    basis sets (TZ/QZ) where the entropy DMRG itself would be too expensive
    if given all N_MO orbitals.

    Strategy
    --------
    * Uses ``mf.mo_energy`` which always stores the **canonical HF eigenvalues**
      (unchanged by MP2 natural-orbital construction or orbital localization).
    * HOMO/LUMO boundary determined from ``mol.nelectron`` (unambiguous for
      closed-shell RHF; does not depend on an occupation threshold).
    * Occupied block  (canonical indices 0 … n_occ-1):
      drop orbitals with  ε < ε_HOMO − window_occ_ha  (frozen-core-like).
    * Virtual block   (canonical indices n_occ … N_MO-1):
      drop orbitals with  ε > ε_LUMO + window_virt_ha  (diffuse/Rydberg-like).

    Notes on localized orbitals
    ---------------------------
    ``_localize_orbitals`` rotates **only the occupied block** (columns 0…n_occ-1
    of ``orbitals``).  The virtual columns are unchanged canonical HF orbitals, so
    ``mf.mo_energy[n_occ:]`` remains a valid energy label for each virtual column.
    For the occupied block after localization the canonical energy labels are no
    longer orbital-specific, but the block boundaries (which orbitals are occupied
    vs. virtual) are still correct — and for H atoms there are no deep core
    orbitals to freeze anyway.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    mf  : pyscf scf object — ``mf.mo_energy`` must be the canonical HF eigenvalues.
    orbitals    : np.ndarray, shape (n_ao, n_mo) — current working orbitals.
    occupations : np.ndarray, shape (n_mo,)       — current occupation numbers.
    window_occ_ha  : float or None
        Hartree window *below* HOMO to include. ``None`` → keep all occupied.
    window_virt_ha : float or None
        Hartree window *above* LUMO to include. ``None`` → keep all virtual.
    verbose : int

    Returns
    -------
    selected_indices : np.ndarray
        Integer indices (into ``orbitals``/``occupations``) to keep.
    n_frozen_core : int
        Number of occupied orbitals excluded below the lower energy bound.
    presel_info : dict
        Diagnostic information (energies, thresholds, counts) for logging/plotting.
    """
    n_mo  = orbitals.shape[1]
    # Canonical HF energies — always in canonical MO order
    mo_energy_full = np.asarray(mf.mo_energy)
    # Trim in case mo_energy has more entries than the working orbital set
    mo_energy = mo_energy_full[:n_mo]

    # HOMO/LUMO from electron count (closed-shell RHF)
    n_occ_canonical = mol.nelectron // 2
    if n_occ_canonical >= n_mo or n_occ_canonical == 0:
        # Edge cases: degenerate system or no electrons — return everything
        return np.arange(n_mo), 0, {}

    e_homo = mo_energy[n_occ_canonical - 1]
    e_lumo = mo_energy[n_occ_canonical]

    e_low  = (e_homo - window_occ_ha)  if window_occ_ha  is not None else -np.inf
    e_high = (e_lumo + window_virt_ha) if window_virt_ha is not None else  np.inf

    selected_mask = (mo_energy >= e_low) & (mo_energy <= e_high)
    selected_indices = np.where(selected_mask)[0]

    # Count occupied orbitals that fell below the lower bound (become frozen core)
    n_frozen_core = int(np.sum(~selected_mask[:n_occ_canonical]))
    n_virt_cut    = int(np.sum(~selected_mask[n_occ_canonical:]))

    presel_info = {
        'e_homo': float(e_homo),
        'e_lumo': float(e_lumo),
        'e_low':  float(e_low),
        'e_high': float(e_high),
        'window_occ_ha':  window_occ_ha,
        'window_virt_ha': window_virt_ha,
        'n_mo_before': n_mo,
        'n_mo_after':  int(len(selected_indices)),
        'n_frozen_core': n_frozen_core,
        'n_virt_cut':    n_virt_cut,
        'mo_energy_all': mo_energy.tolist(),
        'occupations_all': occupations.tolist(),
        'selected_indices': selected_indices.tolist(),
    }

    if verbose:
        print(f"\n  ── Energy pre-selection ──────────────────────────────────")
        print(f"  Canonical HF:  ε_HOMO = {e_homo:+.4f} Ha,  ε_LUMO = {e_lumo:+.4f} Ha")
        print(f"  Window:  [{e_low:+.4f},  {e_high:+.4f}] Ha")
        print(f"  Kept {len(selected_indices)} / {n_mo} orbitals  "
              f"({n_frozen_core} occ frozen,  {n_virt_cut} virt removed)")
        print(f"  ─────────────────────────────────────────────────────────\n")

    return selected_indices, n_frozen_core, presel_info


def plot_orbital_selection_summary(
        mol, mf, occupations_full, presel_info,
        orbital_entropies=None,
        entropy_selected_mask=None,
        run_name=None,
        save_path=None,
        show=False):
    """
    Multi-panel diagnostic figure showing how the orbital pool is reduced at each
    selection stage.

    Panels
    ------
    Top  : Scatter — canonical HF orbital energy vs. occupation number.
           Colour/marker encodes selection stage:
             • grey  — removed by energy pre-selection
             • blue  — kept by energy window, removed by entropy threshold
             • gold  — final active space (passed entropy threshold)
           Dashed lines show the energy window boundaries (ε_low, ε_high).
    Bottom (optional, only if ``orbital_entropies`` is provided):
           Bar chart of single-orbital entropies for the pre-selected pool,
           coloured gold for entropy-selected, blue otherwise.

    Parameters
    ----------
    mol         : pyscf.gto.Mole
    mf          : pyscf scf object — ``mf.mo_energy`` canonical HF energies.
    occupations_full : np.ndarray — occupations for ALL orbitals before pre-selection.
    presel_info : dict — returned by ``preselect_by_energy_window``.
    orbital_entropies    : np.ndarray or None — entropies for the *pre-selected* pool.
    entropy_selected_mask: boolean array or None — True where entropy threshold passed,
                           indexed into the pre-selected pool.
    run_name    : str or None — used in the figure title and saved filename.
    save_path   : str or None — if given, figure is saved here (PNG).
    show        : bool — call plt.show() (useful in interactive sessions).
    """
    n_mo  = len(occupations_full)
    mo_energy = np.asarray(mf.mo_energy)[:n_mo]

    sel_idx  = np.array(presel_info.get('selected_indices', np.arange(n_mo)))
    e_low    = presel_info.get('e_low',  -np.inf)
    e_high   = presel_info.get('e_high',  np.inf)
    e_homo   = presel_info.get('e_homo', None)
    e_lumo   = presel_info.get('e_lumo', None)

    # Build selection-stage labels for every orbital
    # 0 = removed by energy pre-selection
    # 1 = in pre-selected pool, but below entropy threshold (or entropy not run yet)
    # 2 = final active space
    stage = np.zeros(n_mo, dtype=int)  # default: removed
    stage[sel_idx] = 1                 # kept by energy window
    if entropy_selected_mask is not None and len(entropy_selected_mask) == len(sel_idx):
        final_idx = sel_idx[entropy_selected_mask]
        stage[final_idx] = 2

    colours = {0: '#aaaaaa', 1: '#4c7bb5', 2: '#e8a020'}
    labels  = {0: 'removed (energy cut)', 1: 'energy window (pre-selected)', 2: 'final active space'}
    markers = {0: 'x', 1: 'o', 2: '*'}

    n_panels = 2 if (orbital_entropies is not None) else 1
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(10, 4 * n_panels),
                             constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    # ── Panel 1: energy vs occupation ────────────────────────────────────────
    ax = axes[0]
    for s in [0, 1, 2]:
        mask = stage == s
        if not np.any(mask):
            continue
        ax.scatter(mo_energy[mask], occupations_full[mask],
                   c=colours[s], marker=markers[s],
                   s=60 if s == 2 else 40,
                   alpha=0.85, label=f"{labels[s]} ({mask.sum()})",
                   zorder=3 if s == 2 else 2)

    # Energy window boundaries
    if np.isfinite(e_low):
        ax.axvline(e_low,  color='steelblue', ls='--', lw=1.2, label=f'ε_low = {e_low:.3f} Ha')
    if np.isfinite(e_high):
        ax.axvline(e_high, color='tomato',    ls='--', lw=1.2, label=f'ε_high = {e_high:.3f} Ha')
    if e_homo is not None:
        ax.axvline(e_homo, color='black', ls=':', lw=0.8, alpha=0.5, label=f'ε_HOMO = {e_homo:.3f} Ha')
    if e_lumo is not None:
        ax.axvline(e_lumo, color='black', ls=':', lw=0.8, alpha=0.5, label=f'ε_LUMO = {e_lumo:.3f} Ha')

    ax.set_xlabel('Canonical HF orbital energy (Ha)')
    ax.set_ylabel('Occupation number')
    ax.set_title(f'Orbital selection stages{" — " + run_name if run_name else ""}')
    ax.legend(fontsize=8, loc='center right')
    ax.grid(True, alpha=0.3)

    # ── Panel 2: entropy bar chart ────────────────────────────────────────────
    if orbital_entropies is not None:
        ax2 = axes[1]
        n_presel = len(sel_idx)
        x = np.arange(n_presel)

        if entropy_selected_mask is not None and len(entropy_selected_mask) == n_presel:
            bar_colors = [colours[2] if s else colours[1] for s in entropy_selected_mask]
        else:
            bar_colors = [colours[1]] * n_presel

        ax2.bar(x, orbital_entropies, color=bar_colors, alpha=0.85, width=0.8)
        ax2.set_xlabel('Orbital index in pre-selected pool')
        ax2.set_ylabel('Single-orbital entropy')
        ax2.set_title('Single-orbital entropies (gold = final active space)')
        ax2.grid(True, axis='y', alpha=0.3)

        # Annotate original MO indices on x-axis for the selected ones
        if n_presel <= 80:
            ax2.set_xticks(x)
            ax2.set_xticklabels([str(i) for i in sel_idx], rotation=90, fontsize=6)

    title_str = run_name or 'orbital_selection'
    fig.suptitle(f'Orbital selection summary — {title_str}', fontsize=11, y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Orbital selection plot saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def select_active_space_with_DMRG(
    mol,
    mf,
    orbitals,
    occupations,
    localized_orbitals,
    unrestricted,
    initial_active_space_size=None,
    n_threads=4,
    scratch_dir=None,
    as_bond_dim=80,
    as_n_sweeps=4,
    as_noise=1e-5,
    entropy_threshold=1e-3,
    verbose=1,
    output_dir=None,
    run_name=None,
    occupation_thresholds=(0.01, 1.99),
    generate_cubes=False,
    cube_resolution=80,
    cube_margin=3.0,
    generate_py3dmol=False,
    py3dmol_n_orbitals=4,
    py3dmol_isoval=0.02
):
    """
    Run DMRG with a large active space and small bond dimension for a few sweeps,
    then select the active space based on single-orbital entropies.
    
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object
    mf : pyscf.scf object
        Mean-field object (RHF or UHF) with mo_coeff matching orbitals
    orbitals : np.ndarray
        Orbital coefficients in AO basis (AO × MO)
    localized_orbitals : bool 
        Whether the input orbitals are localized -- this impact the symmetry labels for integral preperation; and may affect DMRG performance
    occupations : np.ndarray
        Occupation numbers for each orbital
    initial_active_space_size : int, optional
        Number of orbitals to include in initial DMRG calculation.
        If None, uses all orbitals.
    n_threads : int, default=4
        Number of threads for DMRG
    scratch_dir : str, optional
        Scratch directory for DMRG
    as_bond_dim : int, default=80
        Bond dimension for active space selection DMRG
    as_n_sweeps : int, default=4
        Number of sweeps for active space selection
    as_noise : float, default=1e-5
        Noise level for active space selection
    entropy_threshold : float, default=1e-3
        Entropy threshold for selecting orbitals
    verbose : int, default=1
        Verbosity level
    output_dir : str, optional
        Directory to save output files
    run_name : str, optional
        Identifier for output files
    occupation_thresholds : tuple, default=(0.01, 1.99)
        Occupation thresholds for visualization
    generate_cubes : bool, default=False
        Generate cube files for orbital visualization
    cube_resolution : int, default=80
        Grid resolution for cube files (points per dimension)
    cube_margin : float, default=3.0
        Margin around molecule for cube files (Bohr)
    generate_py3dmol : bool, default=False
        Generate interactive py3Dmol visualization (Jupyter-friendly)
    py3dmol_n_orbitals : int, default=4
        Number of orbitals to include in py3Dmol visualization
    py3dmol_isoval : float, default=0.02
        Isosurface value for py3Dmol visualization
        
    Returns
    -------
    selected_orbitals : np.ndarray
        Selected active space orbital coefficients
    selected_indices : np.ndarray
        Indices of selected orbitals
    orbital_entropies : np.ndarray
        Single-orbital entropies from DMRG
    as_energy : float
        Energy from the approximate DMRG calculation
    dmrg_info : dict
        Dictionary with DMRG calculation metadata
    """
    # Select an initial large active space using the occupation numbers
    if initial_active_space_size is not None:
        # Select orbitals with occupation numbers closest to 1
        occ_deviation = np.abs(occupations - 1.0)
        sorted_indices = np.argsort(occ_deviation)
        selected_indices = sorted_indices[:initial_active_space_size]
        mo_coeff_active = orbitals[:, selected_indices]
        occ_active = occupations[selected_indices]

        if verbose:
            print(f"Initial active space size set to {initial_active_space_size}")
            print(f"Selected orbital indices for initial DMRG: {selected_indices.tolist()}")

        # Occupied orbitals (occ > 0.5) not included in the initial active space must be
        # treated as frozen core so that ecore and n_elec are correct.
        all_occupied_indices = np.where(occupations > 0.5)[0]
        selected_set = set(selected_indices.tolist())
        frozen_core_indices = np.array([i for i in all_occupied_indices if i not in selected_set])
        initial_ncore = len(frozen_core_indices)

        if initial_ncore > 0:
            if verbose:
                print(f"Frozen core for initial DMRG: {initial_ncore} orbital(s) "
                      f"(original indices: {frozen_core_indices.tolist()})")
            # get_rhf_integrals treats the first ncore columns of mo_coeff as core.
            combined = np.concatenate([frozen_core_indices, selected_indices])
            mf_temp = copy.copy(mf)
            mf_temp.mo_coeff = orbitals[:, combined]
            mf_temp.mo_occ = occupations[combined]
        else:
            mf_temp = copy.copy(mf)
            mf_temp.mo_coeff = mo_coeff_active
            mf_temp.mo_occ = occ_active
    else:
        mo_coeff_active = orbitals
        selected_indices = np.arange(orbitals.shape[1])
        initial_ncore = 0
        mf_temp = mf  # Use as-is

        if verbose:
            print(f"Using all {orbitals.shape[1]} orbitals for initial DMRG")

    # Prepare integrals using Block2's built-in functions
    ncas, n_elec, spin, ecore, h1e, eri, orb_sym = prepare_integrals_for_dmrg(
        mf_temp, ncas=mo_coeff_active.shape[1], ncore=initial_ncore,
        verbose=verbose, localized_orbitals=localized_orbitals, unrestricted=unrestricted
    )

    n_orb = ncas
    
    if verbose:
        print(f"\n{'='*60}")
        print("ACTIVE SPACE SELECTION VIA DMRG ORBITAL ENTROPIES")
        print(f"{'='*60}")
        print(f"Number of orbitals: {n_orb}")
        print(f"Number of electrons: {n_elec}")
        print(f"Spin: {spin}")
        print(f"Bond dimension: {as_bond_dim}")
        print(f"Sweeps: {as_n_sweeps}")
        print(f"Entropy threshold: {entropy_threshold}")
    
    # Create scratch directory for this DMRG calculation
    if scratch_dir is not None:
        os.makedirs(scratch_dir, exist_ok=True)
        if verbose:
            print(f"Using scratch directory: {scratch_dir}")

    # Chdir to scratch_dir so block2's ./nodex/ is isolated per run (avoids stale-file conflicts)
    _original_dir = os.getcwd()
    if scratch_dir is not None:
        os.chdir(scratch_dir)

    # Initialize DMRG driver (SU2 symmetry for efficiency)
    stack_mem = int(os.environ.get('BLOCK2_MAX_MEMORY', 2 * 1024**3))
    driver = DMRGDriver(
        symm_type=SymmetryTypes.SU2,
        n_threads=n_threads,
        stack_mem=stack_mem
    )

    # TODO: test if this is even relevant

    # FIXED: Handle orbital symmetries - convert string types to integers
    # Block2's C++ interface cannot handle numpy string types
    if orb_sym is not None and len(orb_sym) > 0:
        # Check if orb_sym contains strings (happens with point group symmetry)
        if isinstance(orb_sym[0], (str, np.str_)):
            # For SU2 symmetry, we can't use point group symmetry labels
            # Use trivial symmetry (all orbitals in same irrep)
            orb_sym_int = [0] * n_orb
            if verbose:
                print(f"Warning: Converting string orbital symmetries to trivial (all irrep 0)")
                print(f"         Point group symmetry disabled for DMRG active space selection")
        else:
            # Already integers
            orb_sym_int = orb_sym
    else:
        orb_sym_int = [0] * n_orb

    driver.initialize_system(n_sites=n_orb, n_elec=n_elec, spin=spin, orb_sym=orb_sym_int)
    
    # Build MPO
    mpo = driver.get_qc_mpo(h1e=h1e, g2e=eri, ecore=ecore, iprint=verbose)
    
    # Initial random MPS
    ket = driver.get_random_mps(tag="AS_SELECT", bond_dim=as_bond_dim, nroots=1)
    
    # DMRG schedules
    bond_dims = [as_bond_dim] * as_n_sweeps
    noises = [as_noise] * (as_n_sweeps // 2) + [as_noise * 0.1] * (as_n_sweeps - as_n_sweeps // 2)
    thrds = [1e-6] * as_n_sweeps
    
    if verbose:
        print(f"\nRunning approximate DMRG for active space selection...")
        print("Using SU2 symmetry")
    
    # Run DMRG (this is in SU2 symm)
    as_energy = driver.dmrg(
        mpo, ket,
        n_sweeps=as_n_sweeps,
        bond_dims=bond_dims,
        noises=noises,
        thrds=thrds,
        cutoff=1e-10,
        iprint=verbose
    )
    
    if verbose:
        print(f"\nApproximate DMRG energy: {as_energy:.10f} Ha")
    


    # Convert SU2 MPS to SZ symmetry for entropy calculations and Fiedler ordering
    zket, zdriver = convert_su2_to_sz_symmetry(eri, h1e, ket, ecore, driver, verbose, n_elec, spin, orb_sym_int)

    # Compute entropies in the initial active space
    orbital_entropies = zdriver.get_orbital_entropies(zket, orb_type=1)


    # Compute Fiedler reordering in the initial active space
    reorder_indices, reorder_info, eri_reordered, h1e_reordered = preform_fielder_ordering(
        h1e, eri, zdriver, zket, reorder_method="fiedler", verbose=verbose
    )

    # === Index Tracking Strategy ===
    # We have three index spaces to manage:
    # 1. Original orbital indices (0 to n_total_orbitals-1)
    # 2. Initial active space indices (0 to len(selected_indices)-1)
    # 3. Reordered indices (after Fiedler)
    # 4. Final selected indices (subset after entropy selection)

    # Create index mapping for clarity
    index_map = {
        'initial_active_to_original': selected_indices,  # Maps initial active space → original orbitals
        'reorder_permutation': reorder_indices,          # Fiedler permutation within initial active space
        'entropies_in_initial_active': orbital_entropies # Entropies indexed by initial active space
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"INDEX TRACKING SUMMARY")
        print(f"{'='*60}")
        print(f"Initial active space: {len(selected_indices)} orbitals")
        print(f"  Original indices: {selected_indices.tolist()}")
        print(f"Fiedler reordering: {reorder_indices.tolist()}")
        print(f"\nOrbital entropies (in initial active space order):")
        for i in range(len(orbital_entropies)):
            orig_idx = selected_indices[i]
            entropy = orbital_entropies[i]
            marker = " *" if entropy > entropy_threshold else ""
            print(f"  Init[{i:2d}] = Orig[{orig_idx:3d}]: entropy = {entropy:.6f}{marker}")
        print(f"\n  (* = selected, threshold = {entropy_threshold})")

    # Select orbitals based on entropy threshold (in initial active space)
    entropy_selected_mask = orbital_entropies > entropy_threshold
    entropy_selected_in_initial = np.where(entropy_selected_mask)[0]

    # Map selected indices back to original orbital space
    final_selected_indices = selected_indices[entropy_selected_in_initial]

    # Also track the Fiedler-reordered version of the selected orbitals
    # This tells us: "what are the selected orbital indices in Fiedler-reordered space?"
    selected_in_reordered_space = []
    for init_idx in entropy_selected_in_initial:
        # Find where init_idx appears in the reorder permutation
        reordered_position = np.where(reorder_indices == init_idx)[0]
        if len(reordered_position) > 0:
            selected_in_reordered_space.append(reordered_position[0])
    selected_in_reordered_space = np.array(selected_in_reordered_space)

    # Store comprehensive index mapping
    index_map['entropy_selected_in_initial_active'] = entropy_selected_in_initial
    index_map['entropy_selected_in_original'] = final_selected_indices
    index_map['entropy_selected_in_reordered'] = selected_in_reordered_space

    # Always include at least the occupied orbitals if nothing passes threshold
    if len(final_selected_indices) == 0:
        n_occ = mol.nelectron // 2
        n_keep = min(max(n_occ, 4), n_orb)  # cannot exceed initial active space size
        entropy_selected_in_initial = np.arange(n_keep)
        final_selected_indices = selected_indices[entropy_selected_in_initial]
        if verbose:
            print(f"\nWarning: No orbitals passed entropy threshold. Using first {len(final_selected_indices)} orbitals from initial active space.")
        # Update index map and reordered-space positions
        index_map['entropy_selected_in_initial_active'] = entropy_selected_in_initial
        index_map['entropy_selected_in_original'] = final_selected_indices
        selected_in_reordered_space = np.array([np.where(reorder_indices == i)[0][0] for i in entropy_selected_in_initial])
        index_map['entropy_selected_in_reordered'] = selected_in_reordered_space

    # Extract final selected orbital coefficients and occupations from ORIGINAL space
    selected_orbitals = orbitals[:, final_selected_indices]
    selected_occs = occupations[final_selected_indices]

    # Determine how many occupied orbitals from the initial AS were dropped by entropy selection.
    # Dropped occupied orbitals must be treated as frozen core: we need to re-run
    # prepare_integrals_for_dmrg with ncore=k_dropped so that ecore and h1e_eff are correct.
    all_occ_in_initial = np.where(occupations[selected_indices] > 0.5)[0]  # positions in initial AS
    dropped_occ_in_initial = np.array([i for i in all_occ_in_initial
                                        if i not in set(entropy_selected_in_initial.tolist())])
    k_dropped = len(dropped_occ_in_initial)

    if k_dropped == 0:
        # ncore=0: all electrons are already accounted for in sub-indexed integrals.
        n_elec_final = n_elec  # = mol.nelectron
        # idx used below for sub-indexing the Fiedler-reordered integrals
        idx = np.sort(selected_in_reordered_space)
        if verbose:
            print(f"n_elec (initial AS):      {n_elec}")
            print(f"n_elec (final AS):        {n_elec_final}")
            print(f"Frozen core:              0 orbital(s)")
    else:
        # k_dropped occupied orbitals must become frozen core.
        # Re-run prepare_integrals_for_dmrg with ncore=k_dropped so that:
        #   - ecore includes core orbital kinetic/nuclear + Coulomb/exchange energies
        #   - h1e_eff is Fock-dressed with contributions from the k_dropped core orbitals
        #   - n_elec = mol.nelectron - 2*k_dropped
        dropped_occ_original = selected_indices[dropped_occ_in_initial]  # original orbital indices
        active_original = final_selected_indices                           # original orbital indices

        # Build mf_frozen: core orbitals first, then active (get_rhf_integrals uses [:ncore] as core)
        combined = np.concatenate([dropped_occ_original, active_original])
        mf_frozen = copy.copy(mf)
        mf_frozen.mo_coeff = orbitals[:, combined]
        mf_frozen.mo_occ = occupations[combined]

        _, n_elec_final, _, ecore, h1e_cas, eri_cas, _ = prepare_integrals_for_dmrg(
            mf_frozen, ncore=k_dropped, ncas=len(active_original),
            localized_orbitals=localized_orbitals, unrestricted=unrestricted, verbose=0
        )

        # Apply Fiedler permutation to the newly computed integrals.
        # selected_in_reordered_space[i] = position of i-th selected orbital in Fiedler ordering.
        # argsort gives the permutation from natural order → Fiedler order.
        fiedler_perm = np.argsort(selected_in_reordered_space)
        h1e_reordered = h1e_cas[np.ix_(fiedler_perm, fiedler_perm)]
        eri_cas = np.asarray(eri_cas)
        if eri_cas.ndim == 1:
            eri_cas = zdriver.unpack_g2e(eri_cas, n_sites=len(active_original))
        eri_reordered = eri_cas[np.ix_(fiedler_perm, fiedler_perm, fiedler_perm, fiedler_perm)]
        # Set idx to identity so the sub-indexing below is a no-op
        idx = np.arange(len(active_original))

        if verbose:
            print(f"n_elec (initial AS):      {n_elec}")
            print(f"n_elec (final AS):        {n_elec_final}")
            print(f"Frozen core:              {k_dropped} orbital(s) → ncore={k_dropped}")
            print(f"  Dropped original indices: {dropped_occ_original.tolist()}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"FINAL SELECTION RESULTS")
        print(f"{'='*60}")
        print(f"Selected {len(final_selected_indices)} orbitals based on entropy threshold")
        print(f"  Original orbital indices: {final_selected_indices.tolist()}")
        print(f"  Initial active space indices: {entropy_selected_in_initial.tolist()}")
        print(f"  Reordered space positions: {selected_in_reordered_space.tolist()}")
        print(f"\nSelected occupations:")
        print(f"  Min: {selected_occs.min():.4f}")
        print(f"  Max: {selected_occs.max():.4f}")
        print(f"  Mean: {selected_occs.mean():.4f}")
        print(f"{'='*60}\n")
    
    # Save outputs if requested
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        prefix = f"{run_name}_" if run_name else ""

        # ===== NEW: Save consolidated orbital data (single JSON file) =====
        # Build selection mask for all orbitals in the initial active space (indexed by initial active index)
        selection_mask = np.zeros(len(orbital_entropies), dtype=bool)
        selection_mask[entropy_selected_in_initial] = True

        # Prepare metadata
        consolidation_metadata = {
            'selection_method': 'DMRG_entropy',
            'entropy_threshold': entropy_threshold,
            'n_selected': len(final_selected_indices),
            'as_bond_dim': as_bond_dim,
            'as_n_sweeps': as_n_sweeps,
            'occupation_thresholds': occupation_thresholds,
            'initial_active_space_indices': selected_indices.tolist()
        }

        # Save consolidated file with all orbital information
        save_consolidated_orbital_data(
            output_dir=output_dir,
            run_name=run_name,
            occupations=occupations[selected_indices],  # Occupations for orbitals in initial active space
            indices=selected_indices,  # Original indices from full orbital set
            entropies=orbital_entropies,
            energy=as_energy,
            selected_mask=selection_mask,
            orbital_coefficients=None,  # Don't save coefficients by default (file size)
            metadata=consolidation_metadata,
            verbose=verbose
        )

        # ===== LEGACY: Also save individual text files for backward compatibility =====
        np.savetxt(
            os.path.join(output_dir, f"{prefix}orbital_entropies.txt"),
            orbital_entropies,
            header=f"Run: {run_name}\nSingle-orbital von Neumann entropies"
        )

        np.savetxt(
            os.path.join(output_dir, f"{prefix}entropy_selected_indices.txt"),
            final_selected_indices,
            fmt='%d',
            header=f"Run: {run_name}\nOrbitals selected by entropy > {entropy_threshold}"
        )

        # Visualize selected occupations
        try:
            visualize_occupations(
                occupations={f'{run_name}\nEntropy-Selected Active Space': selected_occs},
                thresholds=occupation_thresholds,
                save_path=os.path.join(output_dir, f"{prefix}entropy_selected.png")
            )
        except Exception as e:
            if verbose:
                print(f"Warning: Could not save visualization: {e}")
        
        # Generate cube files if requested
        if generate_cubes:
            try:
                cube_dir = os.path.join(output_dir, 'cube_files')
                
                cube_files = generate_cube_files(
                    mol=mol,
                    orbitals=selected_orbitals,
                    occupations=selected_occs,
                    output_dir=cube_dir,
                    orbital_indices=None,
                    cube_resolution=cube_resolution,
                    margin=cube_margin,
                    run_name=run_name,
                    verbose=verbose
                )
                
                density_cube = generate_density_cube(
                    mol=mol,
                    orbitals=selected_orbitals,
                    occupations=selected_occs,
                    output_dir=cube_dir,
                    cube_resolution=cube_resolution,
                    margin=cube_margin,
                    run_name=run_name,
                    verbose=verbose
                )
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not generate cube files: {e}")
        
        # Generate py3Dmol visualization if requested
        if generate_py3dmol:
            try:
                html_file = os.path.join(
                    output_dir,
                    f"{prefix}visualization.html" if run_name else "visualization.html"
                )
                
                view = visualize_orbitals_py3dmol(
                    mol=mol,
                    orbitals=selected_orbitals,
                    occupations=selected_occs,
                    orbital_indices=np.arange(min(py3dmol_n_orbitals, len(final_selected_indices))),
                    n_orbitals=py3dmol_n_orbitals,
                    isoval=py3dmol_isoval,
                    output_html=html_file,
                    cube_resolution=cube_resolution,
                    cube_margin=cube_margin,
                    run_name=run_name,
                    verbose=verbose
                )
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not generate py3Dmol visualization: {e}")
    
    # Prepare comprehensive info dictionary with clear index tracking
    dmrg_info = {
        'n_elec_final': n_elec_final,   # ADD THIS
        'n_elec_initial': n_elec,   
        'n_selected': len(final_selected_indices),
        'selected_indices_original': final_selected_indices.tolist(),
        'selected_indices_in_initial_active': entropy_selected_in_initial.tolist(),
        'selected_indices_in_reordered': selected_in_reordered_space.tolist(),
        'orbital_entropies': orbital_entropies.tolist(),
        'entropy_threshold': entropy_threshold,
        'as_energy': float(as_energy),
        'as_bond_dim': as_bond_dim,
        'as_n_sweeps': as_n_sweeps,
        'initial_active_space_indices': selected_indices.tolist(),
        'initial_active_space_size': len(selected_indices),
        'reorder_method': reorder_info['reorder_method'],
        'reorder_permutation': reorder_indices.tolist(),
        'index_mappings': {
            'description': 'Index mappings between different orbital spaces',
            'spaces': {
                'original': 'Full orbital set (0 to n_orbitals-1)',
                'initial_active': 'Initial DMRG active space (subset of original)',
                'reordered': 'Fiedler-reordered initial active space',
                'final_selected': 'Entropy-selected orbitals (subset of initial active)'
            },
            'initial_active_to_original': index_map['initial_active_to_original'].tolist(),
            'reorder_permutation': index_map['reorder_permutation'].tolist(),
            'selected_in_initial_active': index_map['entropy_selected_in_initial_active'].tolist(),
            'selected_in_original': index_map['entropy_selected_in_original'].tolist(),
            'selected_in_reordered': index_map['entropy_selected_in_reordered'].tolist()
        }
    }

    # Extract sub-integrals for only the selected orbitals from the Fiedler-reordered integrals.
    # idx is set in the k_dropped==0 branch (sorted reordered-space positions) or
    # in the k_dropped>0 branch (identity permutation, integrals already fully regenerated).
    h1e_selected = h1e_reordered[np.ix_(idx, idx)]

    eri_reordered = np.asarray(eri_reordered)
    if eri_reordered.ndim == 4:
        eri_selected = eri_reordered[np.ix_(idx, idx, idx, idx)]
    elif eri_reordered.ndim == 1:
        # Compressed 1D format – unpack, sub-select, keep as 4D
        n_full = h1e_reordered.shape[0]
        eri_4d = eri_reordered.reshape(n_full, n_full, n_full, n_full)
        eri_selected = eri_4d[np.ix_(idx, idx, idx, idx)]
    else:
        raise ValueError(f"Unexpected eri_reordered shape: {eri_reordered.shape}")

    if verbose:
        print(f"\nExtracted sub-integrals for {len(idx)} selected orbitals "
              f"(from {h1e_reordered.shape[0]}-orbital reordered space)")
        print(f"  h1e: {h1e_reordered.shape} -> {h1e_selected.shape}")
        print(f"  eri: {eri_reordered.shape} -> {eri_selected.shape}")
        print(f"  ecore (nuclear repulsion + frozen core): {ecore:.8f} Ha")

    # Cleanup
    del driver, mpo, ket, zket, zdriver
    os.chdir(_original_dir)

    return h1e_selected, eri_selected, ecore, n_elec_final, selected_orbitals, final_selected_indices, orbital_entropies, as_energy, dmrg_info


def perform_fiedler_reordering(
    mol,
    mf=None,
    mo_coeff_active=None,
    h1e=None,
    eri=None,
    ecore=None,
    n_elec=None,
    spin=None,
    n_threads=4,
    scratch_dir=None,
    approx_bond_dim=50,
    approx_sweeps=10,
    reorder_method='fiedler',
    verbose=1
):
    """
    Perform Fiedler orbital reordering using approximate DMRG.
    
    SUPPORTS TWO USAGE MODES:
    1. Pass pre-computed integrals (h1e, eri, ecore, n_elec, spin) - avoids recomputation
    2. Pass mo_coeff_active and mf - computes integrals internally
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object
    mf : pyscf.scf object, optional
        Mean-field object (required only if h1e/eri not provided)
    mo_coeff_active : np.ndarray, optional
        Active space orbital coefficients (required only if h1e/eri not provided)
    h1e : np.ndarray, optional
        One-electron integrals (if None, computed from mo_coeff_active and mf)
    eri : np.ndarray, optional
        Two-electron integrals (if None, computed from mo_coeff_active and mf)
    ecore : float, optional
        Core energy (if None, computed from mo_coeff_active and mf)
    n_elec : int, optional
        Number of electrons (if None, extracted from integrals computation)
    spin : int, optional
        Total spin (if None, extracted from integrals computation)
    n_threads : int, default=4
        Number of threads for DMRG
    scratch_dir : str, optional
        Scratch directory for DMRG
    approx_bond_dim : int, default=50
        Bond dimension for approximate DMRG
    approx_sweeps : int, default=10
        Number of sweeps for approximate DMRG
    reorder_method : str, default='fiedler'
        Reordering strategy ('fiedler', 'gaopt', or 'genetic')
    verbose : int, default=1
        Verbosity level
        
    Returns
    -------
    reorder_indices : np.ndarray
        Indices for reordering orbitals
    h1e_reordered : np.ndarray
        Reordered one-electron integrals
    eri_reordered : np.ndarray
        Reordered two-electron integrals
    reorder_info : dict
        Dictionary with reordering metadata
    """
    
    # If integrals not provided, compute them from mo_coeff_active
    if h1e is None or eri is None:
        raise ValueError("Must provide (h1e, eri, ecore)")
        # if mo_coeff_active is None or mf is None:
        #     raise ValueError("Must provide either (h1e, eri, ecore) or (mo_coeff_active, mf)")
        
        # # FIXED: Create temporary mf with custom orbitals
        # mf_temp = copy.copy(mf)
        # mf_temp.mo_coeff = mo_coeff_active
        # # Note: mo_occ doesn't matter for integral transformation
        
        # # Compute integrals
        # ncas, n_elec_computed, spin_computed, ecore, h1e, eri, orb_sym = prepare_integrals_for_dmrg(
        #     mf_temp, ncas=mo_coeff_active.shape[1], ncore=0, verbose=verbose
        # )
        
        # if n_elec is None:
        #     n_elec = n_elec_computed
        # if spin is None:
        #     spin = spin_computed
    else:
        # Integrals provided; ensure n_elec and spin are also provided
        if n_elec is None or spin is None:
            raise ValueError("Must provide n_elec and spin when providing pre-computed integrals")
        
        ncas = h1e.shape[0]
    
    n_orb = ncas
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"FIEDLER ORBITAL REORDERING (recalcualted in SU2 and converted to SZ symmetry for fiedler ordering)")
        print(f"{'='*60}")
        print(f"Number of orbitals: {n_orb}")
        print(f"Number of electrons: {n_elec}")
        print(f"Spin: {spin}")
        print(f"Bond dimension: {approx_bond_dim}")
        print(f"Sweeps: {approx_sweeps}")
        print(f"Reordering method: {reorder_method}")
    
    # Create scratch directory for this DMRG calculation
    if scratch_dir is not None:
        os.makedirs(scratch_dir, exist_ok=True)
        if verbose:
            print(f"Using scratch directory: {scratch_dir}")
    
    # Use SZ symmetry for reordering (needed for orbital interaction matrix)
    stack_mem = int(os.environ.get('BLOCK2_MAX_MEMORY', 2 * 1024**3))
    # driver = DMRGDriver(
    #     scratch=scratch_dir,
    #     symm_type=SymmetryTypes.SU2,
    #     n_threads=n_threads,
    #     stack_mem=stack_mem
    # )
    driver = DMRGDriver(
        symm_type=SymmetryTypes.SU2,
        n_threads=n_threads,
        stack_mem=stack_mem
    )


    # For SU2 symmetry, we don't use orb_sym (point group symmetry)
    # SU2 only uses particle number and Sz quantum numbers
    driver.initialize_system(n_sites=n_orb, n_elec=n_elec, spin=spin, )
    
    # Build MPO
    mpo = driver.get_qc_mpo(h1e=h1e, g2e=eri, ecore=ecore, iprint=verbose, reorder=None)
    
    # Initial random MPS
    ket = driver.get_random_mps(tag="REORDER", bond_dim=approx_bond_dim, nroots=1)
    
    # DMRG schedules
    bond_dims = [approx_bond_dim] * approx_sweeps
    noises = [1e-5] * (approx_sweeps // 2) + [1e-6] * (approx_sweeps - approx_sweeps // 2)
    thrds = [1e-6] * (approx_sweeps // 2) + [1e-7] * (approx_sweeps - approx_sweeps // 2)
    
    if verbose:
        print(f"\nRunning approximate DMRG for orbital reordering...")
    
    # Run DMRG
    reorder_energy = driver.dmrg(
        mpo, ket,
        n_sweeps=approx_sweeps,
        bond_dims=bond_dims,
        noises=noises,
        thrds=thrds,
        cutoff=1e-10,
        iprint=verbose
    )
    
    if verbose:
        print(f"\nApproximate DMRG energy: {reorder_energy:.10f} Ha")
    
    # Get orbital interaction matrix (requires SZ symmetry)
    ket = driver.load_mps("REORDER")

    # Convert SU2 MPS to SZ symmetry for orbital interaction matrix calculation
    zket = convert_su2_to_sz_symmetry(eri, h1e, ket, ecore, driver, verbose, n_elec, spin)

    # Compute orbital interaction matrix (only needs zket in SZ)
    minfo = driver.get_orbital_interaction_matrix(zket)
    
    # Compute optimal reordering
    reorder_indices = driver.orbital_reordering_interaction_matrix(minfo, strategy=reorder_method)

    if verbose:
        print(f"\nReordering indices ({reorder_method}):")
        print(f"  {reorder_indices.tolist()}")

    # Reorder Hamiltonian tensors
    if verbose:
        print(f"\nReordering one-body and two-body tensors...")

    # Ensure h1e and eri are numpy arrays with proper dimensions
    h1e = np.asarray(h1e)
    eri = np.asarray(eri)

    if verbose:
        print(f"h1e shape: {h1e.shape}, eri shape: {eri.shape}")

    # Handle h1e: reshape to 2D if needed
    if h1e.ndim == 1:
        # Reshape 1D array to 2D square matrix
        n_orb_h1e = int(np.sqrt(len(h1e)))
        if n_orb_h1e * n_orb_h1e != len(h1e):
            raise ValueError(f"Cannot reshape h1e of length {len(h1e)} to square matrix")
        h1e = h1e.reshape(n_orb_h1e, n_orb_h1e)

    # Get number of orbitals from h1e shape (now guaranteed to be 2D)
    n_orb = h1e.shape[0]

    # Reorder h1e (always 2D after reshape above)
    h1e_reordered = h1e[np.ix_(reorder_indices, reorder_indices)]

    # Handle eri reordering: unpack if compressed, then reorder
    if eri.ndim == 1:
        # 1D compressed format - unpack it first using Block2's built-in function
        if verbose:
            print(f"  Unpacking compressed ERI format...")
        eri = driver.unpack_g2e(eri, n_sites=n_orb)
        if verbose:
            print(f"  Unpacked ERI shape: {eri.shape}")
    
    # Now eri should be 4D (after unpacking) or already 4D/2D
    if eri.ndim == 4:
        # Full 4D tensor format - apply 4D permutation
        eri_reordered = eri[np.ix_(reorder_indices, reorder_indices, reorder_indices, reorder_indices)]
    elif eri.ndim == 2:
        # 2D packed format (n_orbs^2 x n_orbs^2) - reorder as 2D then reshape
        eri_reordered = eri[np.ix_(reorder_indices, reorder_indices)]
        eri_reordered = eri_reordered[:, np.ix_(reorder_indices, reorder_indices)].reshape(eri.shape)
    else:
        raise ValueError(f"Unexpected eri shape after unpacking: {eri.shape}")
    
    # Prepare info dictionary
    reorder_info = {
        'reorder_indices': reorder_indices.tolist(),
        'reorder_method': reorder_method,
        'reorder_energy': float(reorder_energy),
        'approx_bond_dim': approx_bond_dim,
        'approx_sweeps': approx_sweeps
    }
    
    if verbose:
        print(f"{'='*60}\n")
    
    # Cleanup
    del driver, mpo, ket, minfo
    
    return reorder_indices, h1e_reordered, eri_reordered, reorder_info



# CODE FOR SAVING ACTIVE SPACE 

def select_active_space(orbitals, occupations, mol, 
                       n_active_orbitals=None,
                       occupation_thresholds=(0.01, 1.99),
                       occupation_cutoff=1e-10,
                       output_dir=None,
                       run_name=None,
                       generate_cubes=False,
                       cube_resolution=80,
                       cube_margin=3.0,
                       cube_orbitals='selected',
                       generate_py3dmol=False,
                       py3dmol_n_orbitals=4,
                       py3dmol_isoval=0.02,
                       verbose=1):
    """
    Select active space orbitals based on occupation number thresholds.
    
    Selection strategy:
    1. Core orbitals: occupation > upper_threshold (mandatory)
    2. Active orbitals: lower_threshold < occupation < upper_threshold
    3. Virtual orbitals: occupation < lower_threshold
    
    If n_active_orbitals is specified, orbitals are selected to fill this number,
    prioritizing: core → active → virtual (in order of importance).
    
    Can generate cube files and/or interactive py3Dmol visualizations for 
    orbital analysis.
    
    Parameters
    ----------
    orbitals : np.ndarray
        Orbital coefficient matrix (AO × MO)
    occupations : np.ndarray
        Orbital occupation numbers
    mol : pyscf.gto.Mole
        Molecule object
    n_active_orbitals : int, optional
        Number of orbitals in active space (if None, use all)
    occupation_thresholds : tuple, default=(0.01, 1.99)
        (lower, upper) thresholds for active space
    occupation_cutoff : float, default=1e-10
        Threshold for treating occupations as zero
    output_dir : str, optional
        Directory to save active space analysis files
    run_name : str, optional
        Identifier for output files
    generate_cubes : bool, default=False
        Generate cube files for orbital visualization
    cube_resolution : int, default=80
        Grid resolution for cube files (points per dimension)
    cube_margin : float, default=3.0
        Margin around molecule for cube files (Bohr)
    cube_orbitals : str, default='selected'
        Which orbitals to generate cubes for:
        - 'selected': Only selected active space orbitals
        - 'all': All orbitals
        - 'active': Only correlation-active orbitals (excluding core/virtual)
        - list/array: Specific orbital indices
    generate_py3dmol : bool, default=False
        Generate interactive py3Dmol visualization (Jupyter-friendly)
    py3dmol_n_orbitals : int, default=4
        Number of orbitals to include in py3Dmol visualization
    py3dmol_isoval : float, default=0.02
        Isosurface value for py3Dmol visualization
    verbose : int, default=1
        Verbosity level
        
    Returns
    -------
    active_orbitals : np.ndarray
        Selected active space orbital coefficients
    n_active : int
        Number of active orbitals
    n_electrons : int
        Number of active electrons (total electrons in molecule)
    active_info : dict
        Dictionary with active space metadata
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"ACTIVE SPACE SELECTION")
        print(f"{'='*70}")
    
    lower_thresh, upper_thresh = occupation_thresholds
    
    # Sort orbitals by occupation (descending)
    sort_idx = np.argsort(occupations)[::-1]
    occs_sorted = occupations[sort_idx]
    
    # Compute entropy of occupied orbitals
    occs_nonzero = occs_sorted[occs_sorted > occupation_cutoff]
    entropy = scipy.stats.entropy(occs_nonzero) if len(occs_nonzero) > 0 else 0.0
    
    if verbose:
        print(f"\nOrbital occupation statistics:")
        print(f"  Range: [{occs_sorted.min():.6f}, {occs_sorted.max():.6f}]")
        print(f"  Entropy: {entropy:.6f}")
        print(f"  Thresholds: [{lower_thresh}, {upper_thresh}]")
    
    # Categorize orbitals
    core_mask = occs_sorted > upper_thresh
    active_mask = (occs_sorted > lower_thresh) & (occs_sorted <= upper_thresh)
    virtual_mask = occs_sorted <= lower_thresh
    
    core_idx = sort_idx[core_mask]
    active_idx = sort_idx[active_mask]
    virtual_idx = sort_idx[virtual_mask]
    
    if verbose:
        print(f"\nOrbital categorization:")
        print(f"  Core (occ > {upper_thresh}): {len(core_idx)}")
        print(f"  Active ({lower_thresh} < occ ≤ {upper_thresh}): {len(active_idx)}")
        print(f"  Virtual (occ ≤ {lower_thresh}): {len(virtual_idx)}")
    
    # Select orbitals
    if n_active_orbitals is None:
        # Use all orbitals
        selected_idx = np.arange(len(occupations))
        n_active = len(occupations)
        
        if verbose:
            print(f"\nUsing FULL active space: {mol.nelectron} electrons in {n_active} orbitals")
    
    else:
        n_active = int(n_active_orbitals)
        
        if n_active > len(occupations):
            print(f"Warning: Requested {n_active} orbitals exceeds available {len(occupations)}")
            n_active = len(occupations)
        
        # Build selection: core → active → virtual
        selected_idx = list(core_idx)
        
        if len(selected_idx) > n_active:
            raise ValueError(
                f"Requested {n_active} orbitals < {len(core_idx)} mandatory core orbitals"
            )
        
        # Add correlation-active orbitals
        n_remaining = n_active - len(selected_idx)
        selected_idx.extend(active_idx[:n_remaining])
        n_remaining = n_active - len(selected_idx)
        
        # Fill remaining with virtual orbitals
        if n_remaining > 0:
            selected_idx.extend(virtual_idx[:n_remaining])
        
        selected_idx = np.sort(np.array(selected_idx))
        
        if verbose:
            print(f"\nSelected active space: {mol.nelectron} electrons in {n_active} orbitals")
    
    # Extract selected orbitals and occupations
    active_orbitals = orbitals[:, selected_idx]
    selected_occs = occupations[selected_idx]
    
    # Prepare output info
    active_info = {
        'n_orbitals': n_active,
        'n_electrons': mol.nelectron,
        'selected_indices': selected_idx,
        'selected_occupations': selected_occs,
        'occupation_thresholds': occupation_thresholds,
        'entropy': entropy,
        'n_core': len(core_idx),
        'n_active': len(active_idx),
        'n_virtual': len(virtual_idx)
    }
    
    if verbose:
        print(f"\nSelected occupation statistics:")
        print(f"  Min: {selected_occs.min():.6f}")
        print(f"  Max: {selected_occs.max():.6f}")
        print(f"  Mean: {selected_occs.mean():.6f}")
        print(f"  Sum: {selected_occs.sum():.2f}")
        print(f"{'='*70}\n")
    
    # Save outputs if requested
    if output_dir is not None:
        _save_active_space_data(
            output_dir, run_name, selected_occs, selected_idx,
            occs_sorted, mol, n_active, occupation_thresholds,
            active_info, verbose
        )
        
        # Visualize
        visualize_occupations(
            occupations={f'{run_name}\nActive Space Selection': selected_occs},
            thresholds=occupation_thresholds,
            save_path=os.path.join(
                output_dir,
                f"{run_name}_active_space.png" if run_name else "active_space.png"
            )
        )
        
        # Generate cube files if requested
        if generate_cubes:
            cube_dir = os.path.join(output_dir, 'cube_files')
            
            # Determine which orbitals to generate cubes for
            if cube_orbitals == 'selected':
                cube_indices = selected_idx
            elif cube_orbitals == 'all':
                cube_indices = None  # All orbitals
            elif cube_orbitals == 'active':
                cube_indices = active_idx
            elif isinstance(cube_orbitals, (list, np.ndarray)):
                cube_indices = cube_orbitals
            else:
                raise ValueError(f"Invalid cube_orbitals: {cube_orbitals}")
            
            # Generate orbital cube files
            cube_files = generate_cube_files(
                mol=mol,
                orbitals=orbitals,
                occupations=occupations,
                output_dir=cube_dir,
                orbital_indices=cube_indices,
                cube_resolution=cube_resolution,
                margin=cube_margin,
                run_name=run_name,
                verbose=verbose
            )
            
            active_info['cube_files'] = cube_files
            
            # Also generate total density cube
            density_cube = generate_density_cube(
                mol=mol,
                orbitals=active_orbitals,
                occupations=selected_occs,
                output_dir=cube_dir,
                cube_resolution=cube_resolution,
                margin=cube_margin,
                run_name=run_name,
                verbose=verbose
            )
            
            active_info['density_cube'] = density_cube
        
        # Generate py3Dmol visualization if requested
        if generate_py3dmol:
            try:
                # In generate_py3dmol section
                html_file = os.path.join(
                    output_dir,
                    f"{run_name}_visualization.html" if run_name else "visualization.html"  # Direct use
                )
                
                view = visualize_orbitals_py3dmol(
                    mol=mol,
                    orbitals=active_orbitals,
                    occupations=selected_occs,
                    orbital_indices=np.arange(min(py3dmol_n_orbitals, len(selected_idx))),
                    n_orbitals=py3dmol_n_orbitals,
                    isoval=py3dmol_isoval,
                    output_html=html_file,
                    cube_resolution=cube_resolution,
                    cube_margin=cube_margin,
                    run_name=run_name,
                    verbose=verbose
                )
                
                if view is not None:
                    active_info['py3dmol_view'] = view
                    active_info['py3dmol_html'] = html_file
            
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not generate py3Dmol visualization: {e}")
    
    return active_orbitals, n_active, mol.nelectron, active_info


def _save_active_space_data(output_dir, run_name, selected_occs, selected_idx,
                            all_occs, mol, n_active, thresholds, info, verbose):
    """Save active space selection data to files."""

    os.makedirs(output_dir, exist_ok=True)

    prefix = f"{run_name}_" if run_name else ""

    # ensure that the occupations are sorted descending for all_occs
    all_occs = np.sort(all_occs)[::-1]

    # ===== NEW: Save consolidated orbital data (single JSON file) =====
    # Build selection mask for all orbitals
    n_total_orbitals = len(all_occs)
    selection_mask = np.zeros(n_total_orbitals, dtype=bool)

    # Get sorted indices to align with all_occs
    sorted_indices = np.argsort(all_occs)[::-1]

    # Mark selected orbitals in the sorted array
    for idx in selected_idx:
        # Find position in sorted array
        pos = np.where(sorted_indices == idx)[0]
        if len(pos) > 0:
            selection_mask[pos[0]] = True

    # Prepare metadata
    consolidation_metadata = {
        'selection_method': 'occupation_threshold',
        'occupation_thresholds': list(thresholds),
        'n_selected': len(selected_idx),
        'n_core': info['n_core'],
        'n_active': info['n_active'],
        'n_virtual': info['n_virtual'],
        'entropy': info['entropy'],
        'molecule_electrons': mol.nelectron,
        'molecule_basis': str(mol.basis)
    }

    # Save consolidated file with all orbital information
    save_consolidated_orbital_data(
        output_dir=output_dir,
        run_name=run_name,
        occupations=all_occs,  # All occupations (sorted)
        indices=sorted_indices,  # Indices corresponding to sorted occupations
        entropies=None,  # Not available in occupation-based selection
        energy=None,  # Not available at this stage
        selected_mask=selection_mask,
        orbital_coefficients=None,  # Don't save coefficients by default (file size)
        metadata=consolidation_metadata,
        verbose=verbose
    )

    # ===== LEGACY: Also save individual text files for backward compatibility =====
    np.savetxt(
        os.path.join(output_dir, f"{prefix}selected_occupations.txt"),
        selected_occs,
        header=f"Run: {run_name}\nSelected orbital occupation numbers"
    )

    np.savetxt(
        os.path.join(output_dir, f"{prefix}selected_indices.txt"),
        selected_idx,
        fmt='%d',
        header=f"Run: {run_name}\nSelected orbital indices (0-indexed)"
    )

    np.savetxt(
        os.path.join(output_dir, f"{prefix}all_occupations_sorted.txt"),
        all_occs,
        header=f"Run: {run_name}\nAll occupation numbers (sorted descending)"
    )
    
    # Save summary
    summary_path = os.path.join(output_dir, f"{prefix}summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Active Space Selection Summary\n")
        f.write(f"{'='*70}\n")
        f.write(f"Run: {run_name}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Molecule:\n")
        f.write(f"  Formula: {mol.atom}\n")
        f.write(f"  Electrons: {mol.nelectron}\n")
        f.write(f"  Basis: {mol.basis}\n\n")
        
        f.write(f"Active Space:\n")
        f.write(f"  Orbitals: {n_active}\n")
        f.write(f"  Electrons: {mol.nelectron}\n")
        f.write(f"  Thresholds: {thresholds}\n\n")
        
        f.write(f"Orbital Distribution:\n")
        f.write(f"  Core: {info['n_core']}\n")
        f.write(f"  Active: {info['n_active']}\n")
        f.write(f"  Virtual: {info['n_virtual']}\n\n")
        
        f.write(f"Statistics:\n")
        f.write(f"  Entropy: {info['entropy']:.6f}\n")
        f.write(f"  Occupation min: {selected_occs.min():.6f}\n")
        f.write(f"  Occupation max: {selected_occs.max():.6f}\n")
        f.write(f"  Occupation mean: {selected_occs.mean():.6f}\n")
        f.write(f"  Occupation sum: {selected_occs.sum():.2f}\n")
    
    if verbose:
        print(f"Active space data saved to: {output_dir}")
        print(f"  File prefix: {prefix}")


# Example usage
if __name__ == "__main__":
    from pyscf import gto
    
    # Build molecule
    mol = gto.M(
        atom='H 0 0 0; H 0 0 0.74',
        basis='cc-pvdz',
        verbose=0
    )
    
    # Construct orbitals
    orbitals, occupations, info = construct_orbitals(
        mol, 
        method='MP2',
        localize=False,
        unrestricted=False,
        verbose=1
    )
    
    # Select active space
    active_orbs, n_orb, n_elec, active_info = select_active_space(
        orbitals,
        occupations,
        mol,
        n_active_orbitals=6,
        occupation_thresholds=(0.02, 1.98),
        output_dir='./active_space_output',
        run_name='H2_0.74A_ccpvdz_rhf',
        verbose=1
    )
    
    print(f"\nFinal active space: {n_elec} electrons in {n_orb} orbitals")
    print(f"Active orbital matrix shape: {active_orbs.shape}")


def visualize_index_transformations(dmrg_info, save_path=None):
    """
    Create a visual diagram showing how orbital indices transform through the workflow.

    Parameters
    ----------
    dmrg_info : dict
        Output dictionary from select_active_space_with_DMRG
    save_path : str, optional
        If provided, save the visualization to this file

    Returns
    -------
    None (prints to console and optionally saves to file)
    """
    idx_map = dmrg_info['index_mappings']

    # Extract key arrays
    init_to_orig = np.array(idx_map['initial_active_to_original'])
    reorder_perm = np.array(idx_map['reorder_permutation'])
    sel_in_init = np.array(idx_map['selected_in_initial_active'])
    sel_in_orig = np.array(idx_map['selected_in_original'])
    sel_in_reord = np.array(idx_map['selected_in_reordered'])

    n_init = len(init_to_orig)
    n_sel = len(sel_in_orig)

    # Create visualization
    lines = []
    lines.append("\n" + "="*80)
    lines.append("INDEX TRANSFORMATION DIAGRAM")
    lines.append("="*80)

    lines.append("\n📍 STAGE 1: Original → Initial Active Space")
    lines.append("   (Select orbitals with occupation near 1.0)")
    lines.append("   " + "-"*70)
    lines.append(f"   Original indices:      {init_to_orig.tolist()}")
    lines.append(f"   Initial active:        [0, 1, 2, ..., {n_init-1}]")
    lines.append(f"   Size: {n_init} orbitals")

    lines.append("\n⚡ STAGE 2: Run DMRG (SU2 → SZ conversion)")
    lines.append("   (Compute orbital entropies and interaction matrix)")
    lines.append("   " + "-"*70)

    # Show entropies
    entropies = dmrg_info['orbital_entropies']
    threshold = dmrg_info['entropy_threshold']
    lines.append(f"   Entropies (threshold = {threshold}):")
    for i in range(min(10, len(entropies))):
        marker = " *SELECTED*" if i in sel_in_init else ""
        lines.append(f"     Init[{i:2d}] = Orig[{init_to_orig[i]:3d}]: {entropies[i]:.6f}{marker}")
    if len(entropies) > 10:
        lines.append(f"     ... ({len(entropies) - 10} more orbitals)")

    lines.append("\n🔍 STAGE 3: Entropy Selection")
    lines.append("   (Keep only orbitals with entropy > threshold)")
    lines.append("   " + "-"*70)
    lines.append(f"   Selected in initial active:  {sel_in_init.tolist()}")
    lines.append(f"   Selected in original:        {sel_in_orig.tolist()}")
    lines.append(f"   Size: {n_sel} orbitals selected")

    lines.append("\n🔄 STAGE 4: Fiedler Reordering")
    lines.append("   (Reorder initial active space for optimal DMRG)")
    lines.append("   " + "-"*70)
    lines.append(f"   Reorder permutation: {reorder_perm.tolist()}")
    lines.append(f"   Selected positions in reordered space: {sel_in_reord.tolist()}")

    lines.append("\n📊 STAGE 5: Final Result")
    lines.append("   (Selected orbitals in Fiedler-reordered sequence)")
    lines.append("   " + "-"*70)

    # Show final mapping
    lines.append("   Final ordering (position → original index):")
    # Sort by reordered positions to get final order
    final_order_indices = sel_in_init[np.argsort(sel_in_reord)]
    final_orig_indices = init_to_orig[final_order_indices]

    for i, orig_idx in enumerate(final_orig_indices):
        lines.append(f"     Position {i:2d} ← Original orbital {orig_idx:3d}")

    lines.append("\n" + "="*80)
    lines.append("SUMMARY")
    lines.append("="*80)
    lines.append(f"  Started with:     {n_init} orbitals (initial active space)")
    lines.append(f"  Selected:         {n_sel} orbitals (entropy > {threshold})")
    lines.append(f"  Final ordering:   Optimized via {dmrg_info['reorder_method']} reordering")
    lines.append(f"  DMRG energy:      {dmrg_info['as_energy']:.10f} Ha")
    lines.append("="*80 + "\n")

    # Print to console
    output = "\n".join(lines)
    print(output)

    # Save to file if requested
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(output)
        print(f"📄 Index transformation diagram saved to: {save_path}\n")


def save_consolidated_orbital_data(
    output_dir,
    run_name,
    occupations,
    indices=None,
    entropies=None,
    energy=None,
    selected_mask=None,
    orbital_coefficients=None,
    metadata=None,
    verbose=1
):
    """
    Save all orbital information in a single comprehensive JSON file.

    This consolidates orbital indices, occupations, entropies (if available),
    energy (if available), and selection status into one file for easier analysis.

    Parameters
    ----------
    output_dir : str
        Directory to save the consolidated file
    run_name : str
        Identifier for the run
    occupations : np.ndarray
        Orbital occupation numbers
    indices : np.ndarray, optional
        Orbital indices (default: 0, 1, 2, ..., n_orbitals-1)
    entropies : np.ndarray, optional
        Single-orbital von Neumann entropies
    energy : float, optional
        Total energy from calculation
    selected_mask : np.ndarray of bool, optional
        Boolean array indicating which orbitals are selected
    orbital_coefficients : np.ndarray, optional
        Orbital coefficient matrix (AO × MO) - saved if provided
    metadata : dict, optional
        Additional metadata to include in the file
    verbose : int, default=1
        Verbosity level

    Returns
    -------
    filepath : str
        Path to the saved JSON file
    """
    os.makedirs(output_dir, exist_ok=True)

    n_orbitals = len(occupations)

    # Default indices if not provided
    if indices is None:
        indices = np.arange(n_orbitals)

    # Build orbital data list
    orbital_data = []
    for i in range(n_orbitals):
        orb_info = {
            'index': int(indices[i]) if i < len(indices) else i,
            'occupation': float(occupations[i])
        }

        # Add entropy if available
        if entropies is not None and i < len(entropies):
            orb_info['entropy'] = float(entropies[i])

        # Add selection status if available
        if selected_mask is not None and i < len(selected_mask):
            orb_info['is_selected'] = bool(selected_mask[i])

        # Optionally add orbital coefficients (may make file large)
        if orbital_coefficients is not None and i < orbital_coefficients.shape[1]:
            orb_info['coefficients'] = orbital_coefficients[:, i].tolist()

        orbital_data.append(orb_info)

    # Build consolidated output structure
    output_data = {
        'run_name': run_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_orbitals': n_orbitals,
        'energy': float(energy) if energy is not None else None,
        'orbitals': orbital_data
    }

    # Add metadata if provided
    if metadata is not None:
        output_data['metadata'] = metadata

    # Save to JSON file
    prefix = f"{run_name}_" if run_name else ""
    filepath = os.path.join(output_dir, f"{prefix}orbital_data_consolidated.json")

    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)

    if verbose:
        print(f"\n📄 Consolidated orbital data saved to: {filepath}")
        print(f"   Contains {n_orbitals} orbitals with:")
        print(f"   - Indices: ✓")
        print(f"   - Occupations: ✓")
        print(f"   - Entropies: {'✓' if entropies is not None else '✗'}")
        print(f"   - Energy: {'✓' if energy is not None else '✗'}")
        print(f"   - Selection status: {'✓' if selected_mask is not None else '✗'}")
        print(f"   - Orbital coefficients: {'✓' if orbital_coefficients is not None else '✗'}")

    return filepath


def visualize_occupations(occupations, thresholds, methods=None, save_path=None):
    """
    Visualize natural orbital occupation numbers.
    
    Parameters
    ----------
    occupations : dict
        Dictionary with method names as keys and occupation arrays as values
    thresholds : list or tuple
        [lower, upper] occupation thresholds for active space
    methods : list, optional
        Subset of methods to plot (default: all in occupations dict)
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    if methods is None:
        methods = list(occupations.keys())
    
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    
    for ax, method in zip(axes, methods):
        occs = occupations[method]
        
        # Bar plot of occupations
        ax.bar(range(len(occs)), occs, color='steelblue', 
               edgecolor='black', alpha=0.7)
        
        # Threshold lines
        ax.axhline(y=thresholds[0], color='r', linestyle='--', 
                   alpha=0.5, label=f'Thresholds: [{thresholds[0]}, {thresholds[1]}]')
        ax.axhline(y=thresholds[1], color='r', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Orbital Index', fontsize=11)
        ax.set_ylabel('Occupation Number', fontsize=11)
        ax.set_title(f'{method}', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 2.1])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def generate_cube_files(mol, orbitals, occupations=None, 
                       output_dir='cube_files',
                       orbital_indices=None,
                       cube_resolution=80,
                       margin=3.0,
                       run_name=None,
                       verbose=1):
    """
    Generate cube files for molecular orbitals.
    
    Cube files can be visualized with software like VMD, PyMOL, Avogadro, or Jmol.
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object
    orbitals : np.ndarray
        Orbital coefficient matrix (AO × MO)
    occupations : np.ndarray, optional
        Orbital occupation numbers (used for naming files)
    output_dir : str, default='cube_files'
        Directory to save cube files
    orbital_indices : list or np.ndarray, optional
        Specific orbital indices to generate cubes for (default: all)
    cube_resolution : int, default=80
        Grid points per dimension (higher = better quality, slower)
    margin : float, default=3.0
        Margin around molecule in Bohr (larger = bigger box)
    run_name : str, optional
        Prefix for cube file names
    verbose : int, default=1
        Verbosity level
        
    Returns
    -------
    cube_files : list
        List of generated cube file paths
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"GENERATING CUBE FILES")
        print(f"{'='*70}")
        print(f"Output directory: {output_dir}")
        print(f"Resolution: {cube_resolution} points per dimension")
        print(f"Margin: {margin} Bohr")
    
    os.makedirs(output_dir, exist_ok=True)
    
    n_orbitals = orbitals.shape[1]
    
    # Determine which orbitals to generate
    if orbital_indices is None:
        orbital_indices = range(n_orbitals)
    else:
        orbital_indices = np.asarray(orbital_indices)
        # Check validity
        if np.any(orbital_indices >= n_orbitals) or np.any(orbital_indices < 0):
            raise ValueError(f"Invalid orbital indices. Must be in range [0, {n_orbitals-1}]")
    
    if verbose:
        print(f"Generating cube files for {len(orbital_indices)} orbitals")
    
    prefix = f"{run_name}_" if run_name else ""
    cube_files = []
    
    start_time = time.time()
    
    for i, orb_idx in enumerate(orbital_indices):
        # Get orbital coefficient vector
        mo_coeff = orbitals[:, orb_idx]
        
        # Generate occupation info for filename
        if occupations is not None:
            occ = occupations[orb_idx]
            occ_str = f"_occ{occ:.4f}".replace('.', 'p')
        else:
            occ_str = ""
        
        # Construct filename
        cube_filename = os.path.join(
            output_dir,
            f"{prefix}orbital_{orb_idx:03d}{occ_str}.cube"
        )
        
        # Generate cube file
        cubegen.orbital(mol, cube_filename, mo_coeff, nx=cube_resolution, 
                       ny=cube_resolution, nz=cube_resolution, margin=margin)
        
        cube_files.append(cube_filename)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(orbital_indices)} cube files...")
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\nGenerated {len(cube_files)} cube files in {elapsed:.2f}s")
        print(f"Average time per orbital: {elapsed/len(cube_files):.3f}s")
        print(f"\nCube files saved to: {output_dir}")
        print(f"Visualization tips:")
        print(f"  - VMD: vmd {cube_files[0]}")
        print(f"  - PyMOL: load {cube_files[0]}")
        print(f"  - Avogadro: File → Open → {cube_files[0]}")
        print(f"{'='*70}\n")
    
    return cube_files


def generate_density_cube(mol, orbitals, occupations, 
                         output_dir='cube_files',
                         cube_resolution=80,
                         margin=3.0,
                         run_name=None,
                         density_type='total',
                         verbose=1):
    """
    Generate cube file for electron density.
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object
    orbitals : np.ndarray
        Orbital coefficient matrix (AO × MO)
    occupations : np.ndarray
        Orbital occupation numbers
    output_dir : str, default='cube_files'
        Directory to save cube file
    cube_resolution : int, default=80
        Grid points per dimension
    margin : float, default=3.0
        Margin around molecule in Bohr
    run_name : str, optional
        Prefix for cube file name
    density_type : str, default='total'
        Type of density: 'total', 'spin' (for unrestricted)
    verbose : int, default=1
        Verbosity level
        
    Returns
    -------
    cube_file : str
        Path to generated cube file
    """
    
    if verbose:
        print(f"\nGenerating {density_type} density cube file...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute density matrix in AO basis
    # D = C * occ * C^T
    dm = orbitals @ np.diag(occupations) @ orbitals.T
    
    prefix = f"{run_name}_" if run_name else ""
    cube_filename = os.path.join(
        output_dir,
        f"{prefix}density_{density_type}.cube"
    )
    
    # Generate cube file for density
    cubegen.density(mol, cube_filename, dm, nx=cube_resolution,
                   ny=cube_resolution, nz=cube_resolution, margin=margin)
    
    if verbose:
        print(f"Density cube file saved to: {cube_filename}")
    
    return cube_filename


def visualize_orbitals_py3dmol(mol, orbitals, occupations, orbital_indices=None, 
                               n_orbitals=None, isoval=0.02, output_html=None,
                               cube_resolution=80, cube_margin=3.0, 
                               run_name=None, verbose=1):
    """
    Visualize molecular orbitals using py3Dmol (Jupyter-friendly).
    
    Generates cube files on-the-fly and creates an interactive visualization.
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object
    orbitals : np.ndarray
        Orbital coefficient matrix (AO × MO)
    occupations : np.ndarray
        Orbital occupation numbers
    orbital_indices : list or np.ndarray, optional
        Specific orbital indices to visualize (default: first n_orbitals)
    n_orbitals : int, optional
        Number of orbitals to visualize (default: 4 or all if orbital_indices given)
    isoval : float, default=0.02
        Isosurface value for visualization
    output_html : str, optional
        If provided, save visualization as HTML file
    cube_resolution : int, default=80
        Grid resolution for cube files
    cube_margin : float, default=3.0
        Margin around molecule in Bohr
    run_name : str, optional
        Name prefix for output files
    verbose : int, default=1
        Verbosity level
    
    Returns
    -------
    view : py3Dmol view object or None
        Interactive py3Dmol viewer, or None if py3Dmol not available
    """
    try:
        import py3Dmol
    except ImportError:
        print("py3Dmol not installed. Install with: pip install py3Dmol")
        return None
    
    # Determine which orbitals to visualize
    if orbital_indices is not None:
        orbital_indices = np.asarray(orbital_indices)
        vis_occs = occupations[orbital_indices]
    else:
        if n_orbitals is None:
            n_orbitals = min(4, len(occupations))
        orbital_indices = np.arange(n_orbitals)
        vis_occs = occupations[:n_orbitals]
    
    n_vis = len(orbital_indices)
    
    if verbose:
        print(f"\nGenerating py3Dmol visualization for {n_vis} orbitals...")
    
    # Generate cube files in a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        cube_files = generate_cube_files(
            mol=mol,
            orbitals=orbitals,
            occupations=occupations,
            output_dir=tmpdir,
            orbital_indices=orbital_indices,
            cube_resolution=cube_resolution,
            margin=cube_margin,
            run_name=run_name,
            verbose=0  # Suppress output from generate_cube_files
        )
        
        # Create grid view
        n_cols = 2
        n_rows = (n_vis + 1) // 2
        
        view = py3Dmol.view(width=800, height=400*n_rows, viewergrid=(n_rows, n_cols))
        
        for idx, (orb_idx, cube_file) in enumerate(zip(orbital_indices, cube_files)):
            row = idx // n_cols
            col = idx % n_cols
            
            # Read cube file
            with open(cube_file, 'r') as f:
                cube_data = f.read()
            
            # Add molecule to grid position
            view.addModel(cube_data, 'cube', viewer=(row, col))
            view.setStyle({'stick': {}}, viewer=(row, col))
            
            # Add isosurfaces
            view.addVolumetricData(cube_data, 'cube', {
                'isoval': isoval, 
                'color': 'blue', 
                'opacity': 0.75
            }, viewer=(row, col))
            
            view.addVolumetricData(cube_data, 'cube', {
                'isoval': -isoval, 
                'color': 'red', 
                'opacity': 0.75
            }, viewer=(row, col))
            
            # Add label with orbital index and occupation
            occ = occupations[orb_idx]
            view.addLabel(
                f'Orbital {orb_idx}: occ={occ:.4f}',
                {'position': {'x': 0, 'y': 3, 'z': 0}, 'fontSize': 14},
                viewer=(row, col)
            )
            
            view.zoomTo(viewer=(row, col))
        
        if output_html:
            html = view._make_html()
            with open(output_html, 'w') as f:
                f.write(html)
            if verbose:
                print(f"Visualization saved to {output_html}")
        
        if verbose:
            print(f"py3Dmol visualization ready with {n_vis} orbitals")
        
        return view
