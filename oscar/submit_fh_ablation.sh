#!/bin/bash
# =============================================================================
# submit_fh_ablation.sh — SLURM Array Job for the v2 F/H ablation sweep
# =============================================================================
#
# Launches 7 configs × 3 seeds = 21 jobs as a SLURM array.
#
# Usage:
#     sbatch oscar/submit_fh_ablation.sh
#     sbatch --array=0-5 oscar/submit_fh_ablation.sh
#
# Task ID mapping:   task_id = config_index * 3 + seed_index
#   - config_index: 0..6   (sorted by config filename)
#   - seed_index:   0..2   → seeds [1, 2, 3]
#
# Config set:
#   F4_points_5k
#   F5_adam_resample_200
#   F6_adam_resample_1000
#   H1_rad_k1_0p5
#   H2_rad_k1_2p0
#   H3_rad_k2_0p5
#   H4_rad_k2_2p0
# =============================================================================

#SBATCH --job-name=ch_pinn_fh
#SBATCH --array=0-20
#SBATCH --time=48:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch
#SBATCH --output=slurm_logs/fh_%A_%a.out
#SBATCH --error=slurm_logs/fh_%A_%a.err

module load python/3.11.11-5e66
module load gcc/11.5.0-wfyo

source /oscar/home/dwong33/DRP/.venv/bin/activate
cp /oscar/home/dwong33/DRP/Optimizing_the_Optimizer_PINNs/_optimize.py $(python -c "import scipy.optimize; import os; print(os.path.dirname(scipy.optimize.__file__))")/_optimize.py

cd /oscar/home/dwong33/DRP/Optimizing_the_Optimizer_PINNs
mkdir -p slurm_logs

CONFIGS=($(printf '%s\n' configs/v2_fh_ablation/*.yaml | sort -t/ -k3,3))
EXPECTED_CONFIGS=7
REFERENCE_FILE="reference_solution_t10_dt0p01.npz"
SEEDS=(1 2 3)

N_SEEDS=${#SEEDS[@]}
CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
CONFIG_COUNT=${#CONFIGS[@]}

CONFIG_FILE="${CONFIGS[$CONFIG_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

if [ "$CONFIG_COUNT" -ne "$EXPECTED_CONFIGS" ]; then
    echo "ERROR: Expected $EXPECTED_CONFIGS FH ablation configs, found $CONFIG_COUNT"
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

python -u run_experiment.py "$CONFIG_FILE" --seed "$SEED"

echo ""
echo "========================================================================"
echo "Job finished at $(date)"
echo "Exit code: $?"
echo "========================================================================"
