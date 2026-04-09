#!/bin/bash
#SBATCH --job-name=DMRG_H8_dA3.6_TZ
#SBATCH --output=/groups/qist/ahaslund/pheno_ahaslund/workspace/NQCP-clean/logs/dmrg_h8_dA3.6_%j.out
#SBATCH --error=/groups/qist/ahaslund/pheno_ahaslund/workspace/NQCP-clean/logs/dmrg_h8_dA3.6_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=950G
#SBATCH --time=30-00:00:00
#SBATCH --qos=normal
#SBATCH --partition=qist-fat
#SBATCH --account=qist

# H8 single chain, d_A = 3.6 bohr, full active space (no entropy pre-selection).
# MPS saved to persistent storage after the forward pass converges.
#
# Uses dmrg_pipeline package (scripts/run_calculations.py).

set -e

echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "=============================================================="

# ============================
# MEMORY LIMIT FOR BLOCK2
# ============================
export BLOCK2_MAX_MEMORY=$((950 * 1024 * 1024 * 1024))
echo "BLOCK2_MAX_MEMORY = $BLOCK2_MAX_MEMORY bytes"

# ============================
# ENVIRONMENT
# ============================
source /groups/qist/ahaslund/miniforge3_new/etc/profile.d/conda.sh
conda activate psi

export LD_PRELOAD=/groups/qist/ahaslund/pheno_ahaslund/libfixcpu.so
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ============================
# FAST SCRATCH DIRECTORY
# ============================
export SCRATCH_DIR="/lustre/hpc/qist/ahaslund/pheno_ahaslund/scratch/dmrg_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_DIR"
export TMPDIR="$SCRATCH_DIR"
echo "Using scratch: $SCRATCH_DIR"

cleanup() {
    echo "Cleaning up scratch: $SCRATCH_DIR"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

# ============================
# PATHS
# ============================
PROJECT_DIR="/groups/qist/ahaslund/pheno_ahaslund/workspace/NQCP-clean"

# Point PYTHONPATH at the repo root so `import dmrg_pipeline` resolves
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

SCRIPT="$PROJECT_DIR/dmrg_pipeline/scripts/run_calculations.py"

SAVE_DIR="$PROJECT_DIR/benchmarks/Phase0a-DMRG_Reference_Energy"
LOG_DIR="$SAVE_DIR/logs"
COMPUTE_STATS_DIR="$SAVE_DIR/compute_stats"
BENCHMARKS_DIR="$SAVE_DIR/data"
MPS_BASE_DIR="$SAVE_DIR/mps"

mkdir -p "$LOG_DIR" "$COMPUTE_STATS_DIR" "$BENCHMARKS_DIR" "$MPS_BASE_DIR"

cd "$PROJECT_DIR"
timestamp=$(date +"%Y%m%d_%H%M%S")

# ============================
# SWEEP PARAMETERS
# ============================
SYSTEM="H_chain"

D_A_VALUES=(3.6)
N_REP_VALUES=(8)         # n_rep=8 → H8 (8 atoms in chain, 8 electrons)
METHODS=(HF DMRG)
BASIS_SETS=(cc-pVTZ)

MAX_BOND_DIM=2000
NOISE_SCHEDULE="1e-3 5e-4 1e-4 1e-5 1e-6 0"
SVD_SCHEDULE="1e-5 5e-6 1e-6 1e-7 1e-8 1e-12"
DAVIDSON_SCHEDULE="1e-6 5e-7 1e-7 1e-8 1e-9 1e-10"
MF_CONV_TOL=1e-14
ENERGY_TOL=5e-5
DISCARD_TOL=5e-5
MAX_SWEEPS=50

# ============================
# COUNT TOTAL RUNS
# ============================
total_runs=0
for basis in "${BASIS_SETS[@]}"; do
  for d_a in "${D_A_VALUES[@]}"; do
    for n_rep in "${N_REP_VALUES[@]}"; do
      for method in "${METHODS[@]}"; do
        total_runs=$((total_runs + 1))
      done
    done
  done
done

echo "Total jobs to run: $total_runs"
echo "=============================================================="

# ============================
# MAIN SWEEP
# ============================
current_run=0
for basis in "${BASIS_SETS[@]}"; do
  for d_a in "${D_A_VALUES[@]}"; do
    for n_rep in "${N_REP_VALUES[@]}"; do
      for method in "${METHODS[@]}"; do
        current_run=$((current_run + 1))
        run_id="H${n_rep}_dA${d_a}_${method}_${basis}"

        LOG_FILE="${LOG_DIR}/${run_id}.log"
        TIME_FILE="${COMPUTE_STATS_DIR}/${run_id}_time.txt"
        STATS_JSON="${COMPUTE_STATS_DIR}/${run_id}_stats.json"
        MPS_SAVE_DIR="${MPS_BASE_DIR}/${run_id}"

        echo "--------------------------------------------------------------"
        echo "Running [$current_run/$total_runs]: ${run_id}"
        echo "  Chain: H${n_rep},  d_A=${d_a} Bohr,  Method=${method},  Basis=${basis}"
        echo "  Full orbital space (no entropy pre-selection)"
        echo "  MPS will be saved to: ${MPS_SAVE_DIR}"
        echo "--------------------------------------------------------------"

        /usr/bin/time -v python "$SCRIPT" \
          --single_chain \
          --d_A "${d_a}" \
          --units bohr \
          --localize_orbitals True \
          --n_rep "${n_rep}" \
          --methods_to_run "${method}" \
          --list_of_basis_sets "${basis}" \
          --use_dmrg_active_space_selection false \
          --n_orbitals_for_initial_active_space 9999 \
          --file_name_suffix "${run_id}" \
          --max_bond_dim "${MAX_BOND_DIM}" \
          --noise_schedule ${NOISE_SCHEDULE} \
          --svd_schedule ${SVD_SCHEDULE} \
          --davidson_schedule ${DAVIDSON_SCHEDULE} \
          --mf_conv_tol "${MF_CONV_TOL}" \
          --energy_tol "${ENERGY_TOL}" \
          --discard_tol "${DISCARD_TOL}" \
          --max_sweeps "${MAX_SWEEPS}" \
          --output_file_path "$BENCHMARKS_DIR" \
          --save_mps_dir "${MPS_SAVE_DIR}" \
          --n_parallel_jobs 1 \
          --n_threads_per_process 16 \
          > "$LOG_FILE" 2> "$TIME_FILE"

        EXIT_CODE=$?

        # ============================
        # PARSE /usr/bin/time -v OUTPUT
        # ============================
        STATS_SYSTEM="${SYSTEM}"       \
        STATS_N_REP="${n_rep}"         \
        STATS_D_A="${d_a}"             \
        STATS_METHOD="${method}"       \
        STATS_BASIS="${basis}"         \
        STATS_RUN_ID="${run_id}"       \
        STATS_TIMESTAMP="${timestamp}" \
        STATS_EXIT_CODE="${EXIT_CODE}" \
        STATS_HOSTNAME="$(hostname)"   \
        TIME_FILE="${TIME_FILE}"       \
        STATS_JSON="${STATS_JSON}"     \
        python3 - <<'EOF'
import re, json, os

stats = {
    "system":               os.environ["STATS_SYSTEM"],
    "n_hydrogen":           int(os.environ["STATS_N_REP"]),
    "interatomic_distance": float(os.environ["STATS_D_A"]),
    "method":               os.environ["STATS_METHOD"],
    "basis":                os.environ["STATS_BASIS"],
    "job_id":               os.environ.get("SLURM_JOB_ID", ""),
    "run_id":               os.environ["STATS_RUN_ID"],
    "timestamp":            os.environ["STATS_TIMESTAMP"],
    "exit_code":            int(os.environ["STATS_EXIT_CODE"]),
    "hostname":             os.environ["STATS_HOSTNAME"],
}

time_file  = os.environ["TIME_FILE"]
stats_json = os.environ["STATS_JSON"]

try:
    with open(time_file) as f:
        text = f.read()

    patterns = {
        "user_time_sec":        r"User time \(seconds\): ([\d.]+)",
        "system_time_sec":      r"System time \(seconds\): ([\d.]+)",
        "max_resident_set_kb":  r"Maximum resident set size \(kbytes\): (\d+)",
        "cpu_percent":          r"Percent of CPU this job got: (\d+)%",
    }
    for k, p in patterns.items():
        m = re.search(p, text)
        if m:
            stats[k] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))

    if "max_resident_set_kb" in stats:
        stats["max_memory_gb"] = round(stats["max_resident_set_kb"] / 1024**2, 3)

except Exception as e:
    print(f"Warning: could not parse time file: {e}")

with open(stats_json, "w") as f:
    json.dump(stats, f, indent=2)

print(f"Saved stats → {stats_json}")
if "max_memory_gb" in stats:
    print(f"  Memory: {stats['max_memory_gb']:.2f} GB")
EOF

        if [ $EXIT_CODE -eq 0 ]; then
            echo "[$(date)] ✓ Completed: ${run_id}"
        else
            echo "[$(date)] ✗ FAILED (exit ${EXIT_CODE}): ${run_id}"
        fi
        echo ""

      done  # method
    done    # n_rep
  done      # d_a
done        # basis

echo "=============================================================="
echo "All H8 jobs completed at: $(date)"
echo "=============================================================="
