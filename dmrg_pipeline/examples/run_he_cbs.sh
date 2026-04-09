#!/bin/bash
#SBATCH --job-name=DMRG_He_DTQ
#SBATCH --output=logs/dmrg_he_dtq_%j.out
#SBATCH --error=logs/dmrg_he_dtq_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --qos=normal
#SBATCH --partition=qist-fast
#SBATCH --account=qist

# He atom: HF + DMRG across cc-pVDZ, cc-pVTZ, cc-pVQZ.
# Small system — runs in minutes. Used for CBS extrapolation benchmarks.

set -e

echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "=============================================================="

# ============================
# ENVIRONMENT
# ============================
source /groups/qist/ahaslund/miniforge3_new/etc/profile.d/conda.sh
conda activate psi

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ============================
# PATHS
# ============================
PROJECT_DIR="/groups/qist/ahaslund/pheno_ahaslund/workspace/DMRG_CBS_pipelie"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

SCRIPT="/groups/qist/ahaslund/pheno_ahaslund/workspace/DMRG_CBS_pipelie/scripts/run_calculations.py"

SAVE_DIR="$PROJECT_DIR/benchmarks/He_CBS"
LOG_DIR="$SAVE_DIR/logs"
BENCHMARKS_DIR="$SAVE_DIR/data"

mkdir -p "$LOG_DIR" "$BENCHMARKS_DIR"
cd "$PROJECT_DIR"

# ============================
# PARAMETERS
# ============================
BASIS_SETS=(cc-pVDZ cc-pVTZ cc-pVQZ)
METHODS=(HF DMRG)

# He is a 2-electron singlet — small bond dim is exact
MAX_BOND_DIM=50
ENERGY_TOL=1e-10
DISCARD_TOL=1e-10
MF_CONV_TOL=1e-14
MAX_SWEEPS=20

# ============================
# RUN
# ============================
echo "System: He atom (singlet, charge=0, spin=0)"
echo "Basis sets: ${BASIS_SETS[*]}"
echo "Methods: ${METHODS[*]}"
echo "=============================================================="

/usr/bin/time -v python "$SCRIPT" \
  --atomic \
  --atom He \
  --charge 0 \
  --spin 0 \
  --methods_to_run "${METHODS[@]}" \
  --list_of_basis_sets "${BASIS_SETS[@]}" \
  --orbital_method HF \
  --use_dmrg_active_space_selection false \
  --n_orbitals_for_initial_active_space 9999 \
  --perform_reordering false \
  --max_bond_dim "${MAX_BOND_DIM}" \
  --energy_tol "${ENERGY_TOL}" \
  --discard_tol "${DISCARD_TOL}" \
  --mf_conv_tol "${MF_CONV_TOL}" \
  --max_sweeps "${MAX_SWEEPS}" \
  --file_name_suffix "He_DTQ" \
  --output_file_path "$BENCHMARKS_DIR" \
  --n_parallel_jobs 3 \
  --n_threads_per_process 1 \
  > "$LOG_DIR/He_DTQ.log" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] ✓ He CBS calculation complete"
    echo "Results saved to: $BENCHMARKS_DIR"
else
    echo "[$(date)] ✗ FAILED (exit ${EXIT_CODE}) — see $LOG_DIR/He_DTQ.log"
fi
