#!/bin/bash
# =============================================================================
# submit_v2.sh — SLURM Array Job for Cahn-Hilliard PINN Experiment Suite v2
# =============================================================================
#
# Launches 27 configs × 3 seeds = 81 jobs as a SLURM array.
#
# Usage:
#     sbatch oscar/submit_v2.sh              # submit all 81 jobs
#     sbatch --array=0-5 oscar/submit_v2.sh  # submit only first 2 configs
#
# Task ID mapping:   task_id = config_index * 3 + seed_index
#   - config_index: 0..26  (alphabetical order of configs)
#   - seed_index:   0..2   → seeds [1, 2, 3]
#
# Resource estimate (per job, CPU-only float64 PINN):
#   - 51K iterations × ~0.5s each ≈ ~7 hours (small network)
#   - Larger networks (C1, C2) may take ~12-18 hours
#   - 4GB memory is sufficient for all configs (largest Hessian ~120 MB)
#
# =============================================================================

#SBATCH --job-name=ch_pinn_v2
#SBATCH --array=0-80
#SBATCH --time=48:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch
#SBATCH --output=slurm_logs/v2_%A_%a.out
#SBATCH --error=slurm_logs/v2_%A_%a.err

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
module load python/3.11.0
module load gcc/12.3

# Activate the project virtual environment
source /oscar/home/emaciaso/DRP/.venv/bin/activate

# Change to the project directory
cd /oscar/home/emaciaso/DRP/Optimizing_the_Optimizer_PINNs

# Ensure output directories exist
mkdir -p slurm_logs

# ---------------------------------------------------------------------------
# Config array — sorted by config filename across both v2 subfolders
# ---------------------------------------------------------------------------
CONFIGS=($(printf '%s\n' configs/v2_core/*.yaml configs/v2_fh_ablation/*.yaml | sort -t/ -k3,3))
EXPECTED_CONFIGS=27
REFERENCE_FILE="reference_solution_t10_dt0p01.npz"

SEEDS=(1 2 3)

# ---------------------------------------------------------------------------
# Map SLURM_ARRAY_TASK_ID → (config, seed)
# ---------------------------------------------------------------------------
N_SEEDS=${#SEEDS[@]}
CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
CONFIG_COUNT=${#CONFIGS[@]}

CONFIG_FILE="${CONFIGS[$CONFIG_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

if [ "$CONFIG_COUNT" -ne "$EXPECTED_CONFIGS" ]; then
    echo "ERROR: Expected $EXPECTED_CONFIGS configs, found $CONFIG_COUNT"
    exit 1
fi

if [ ! -f "$REFERENCE_FILE" ]; then
    echo "ERROR: Reference solution not found: $REFERENCE_FILE"
    exit 1
fi

echo "========================================================================"
echo "SLURM Job ID:       $SLURM_JOB_ID"
echo "Array Task ID:      $SLURM_ARRAY_TASK_ID"
echo "Config index:       $CONFIG_IDX"
echo "Config file:        $CONFIG_FILE"
echo "Seed:               $SEED"
echo "Reference file:     $REFERENCE_FILE"
echo "Hostname:           $(hostname)"
echo "Date:               $(date)"
echo "Working directory:  $(pwd)"
echo "Python:             $(which python)"
echo "========================================================================"

# ---------------------------------------------------------------------------
# Run the experiment
# ---------------------------------------------------------------------------
python -u run_experiment.py "$CONFIG_FILE" --seed "$SEED"

echo ""
echo "========================================================================"
echo "Job finished at $(date)"
echo "Exit code: $?"
echo "========================================================================"
