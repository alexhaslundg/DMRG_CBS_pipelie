"""
Test script demonstrating clear index tracking for DMRG active space selection.

This script shows how to use the improved index tracking system to manage
orbital transformations through:
1. Initial active space selection (by occupation)
2. Entropy-based selection (via DMRG)
3. Fiedler reordering (for optimal DMRG)
"""

import numpy as np
from pyscf import gto, scf
from dmrg_pipeline.orbitals.orbital_prep import (
    construct_orbitals,
    select_active_space_with_DMRG,
    apply_selection_and_reordering
)


def print_index_tracking_summary(dmrg_info):
    """
    Pretty-print the index tracking information.
    """
    idx_map = dmrg_info['index_mappings']

    print("\n" + "="*70)
    print("INDEX TRACKING SUMMARY")
    print("="*70)

    print("\n📊 Orbital Space Sizes:")
    print(f"  Initial active space: {dmrg_info['initial_active_space_size']} orbitals")
    print(f"  Final selected space: {dmrg_info['n_selected']} orbitals")

    print("\n🔢 Index Mappings:")
    print(f"\n  1. Initial Active → Original:")
    print(f"     {idx_map['initial_active_to_original']}")

    print(f"\n  2. Fiedler Reordering Permutation:")
    print(f"     {idx_map['reorder_permutation']}")

    print(f"\n  3. Selected Orbitals (different representations):")
    print(f"     - In initial active indices: {idx_map['selected_in_initial_active']}")
    print(f"     - In original indices:       {idx_map['selected_in_original']}")
    print(f"     - In reordered positions:    {idx_map['selected_in_reordered']}")

    print("\n📝 Space Descriptions:")
    for space, desc in idx_map['spaces'].items():
        print(f"  {space:20s}: {desc}")

    print("\n⚡ Entropy Selection:")
    print(f"  Threshold: {dmrg_info['entropy_threshold']}")
    print(f"  DMRG energy: {dmrg_info['as_energy']:.8f} Ha")

    print("\n🔄 Reordering:")
    print(f"  Method: {dmrg_info['reorder_method']}")

    print("="*70 + "\n")


def demonstrate_index_tracking():
    """
    Complete demonstration of index tracking workflow.
    """

    print("\n" + "="*70)
    print("DEMONSTRATION: Index Tracking for DMRG Active Space Selection")
    print("="*70 + "\n")

    # Step 1: Create a simple molecule
    print("Step 1: Creating H4 square molecule...")
    mol = gto.M(
        atom='''
            H  0.0  0.0  0.0
            H  1.5  0.0  0.0
            H  1.5  1.5  0.0
            H  0.0  1.5  0.0
        ''',
        basis='sto-3g',
        symmetry=False,
        verbose=0
    )
    print(f"  Molecule: {mol.nelectron} electrons, {mol.nao} orbitals")

    # Step 2: Construct orbitals (MP2 natural orbitals)
    print("\nStep 2: Constructing MP2 natural orbitals...")
    orbitals, occupations, info, mf = construct_orbitals(
        mol,
        method='MP2',
        localize=False,
        unrestricted=False,
        verbose=0
    )
    print(f"  Generated {orbitals.shape[1]} orbitals")
    print(f"  Occupation range: [{occupations.min():.4f}, {occupations.max():.4f}]")

    # Step 3: Select active space with DMRG
    print("\nStep 3: Running DMRG active space selection...")
    print("  (This performs: initial selection → DMRG → entropy calculation → Fiedler ordering)")

    selected_orbs, selected_indices, entropies, energy, dmrg_info = select_active_space_with_DMRG(
        mol=mol,
        mf=mf,
        orbitals=orbitals,
        occupations=occupations,
        localized_orbitals=False,
        unrestricted=False,
        initial_active_space_size=6,  # Start with 6 orbitals for initial DMRG
        entropy_threshold=1e-4,        # Select orbitals with entropy > 1e-4
        as_bond_dim=50,                # Small bond dim for quick test
        as_n_sweeps=4,                 # Few sweeps for quick test
        verbose=0,                     # Suppress detailed DMRG output
        output_dir=None                # Don't save outputs for this test
    )

    print(f"  ✓ DMRG completed: E = {energy:.8f} Ha")
    print(f"  ✓ Selected {len(selected_indices)} orbitals based on entropy")

    # Step 4: Display index tracking information
    print("\nStep 4: Index tracking summary:")
    print_index_tracking_summary(dmrg_info)

    # Step 5: Apply selection and reordering to get final orbital set
    print("Step 5: Applying selection + reordering to get final orbital coefficients...")

    final_orbs, final_occs, final_to_original = apply_selection_and_reordering(
        orbitals=orbitals,
        occupations=occupations,
        dmrg_info=dmrg_info,
        verbose=1
    )

    print(f"\n  ✓ Final orbital matrix shape: {final_orbs.shape}")
    print(f"  ✓ Final orbital order (original → final):")
    for i, orig_idx in enumerate(final_to_original):
        print(f"      Position {i} ← Original orbital {orig_idx} (occ = {final_occs[i]:.4f})")

    # Step 6: Verification
    print("\nStep 6: Verification...")
    print("  Checking that selected orbitals match expected indices...")

    idx_map = dmrg_info['index_mappings']
    selected_orig = np.array(idx_map['selected_in_original'])

    # The final_to_original should be a reordered version of selected_orig
    assert set(final_to_original) == set(selected_orig), "Index mismatch!"
    print("  ✓ Index consistency verified!")

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\n💡 Key Takeaways:")
    print("  1. Index mappings are stored in dmrg_info['index_mappings']")
    print("  2. Use apply_selection_and_reordering() to get final orbitals")
    print("  3. All transformations are explicitly tracked and verifiable")
    print("  4. The final orbital set is ready for production DMRG calculations")
    print("="*70 + "\n")

    return final_orbs, final_occs, final_to_original, dmrg_info


if __name__ == "__main__":
    # Run the demonstration
    final_orbs, final_occs, final_map, info = demonstrate_index_tracking()

    # Optional: You can now use final_orbs for production DMRG
    print("\n✅ You can now use 'final_orbs' for production DMRG calculations.")
    print("   These orbitals are:")
    print("   - Selected based on DMRG orbital entropies")
    print("   - Reordered via Fiedler for optimal DMRG convergence")
    print("   - Fully tracked with explicit index mappings")
