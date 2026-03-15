#!/bin/bash
# =============================================================================
# submit_baselines.sh — SLURM Array Job for BFGS/L-BFGS Baselines
# =============================================================================
#
# Launches 2 baseline configs × 3 seeds = 6 jobs as a SLURM array.
# These are the control experiments for the SSBroyden ablation.
#
# Usage:
#     sbatch oscar/submit_baselines.sh                 # submit all 6 jobs
#     sbatch --array=0-2 oscar/submit_baselines.sh     # vanilla BFGS only
#     sbatch --array=3-5 oscar/submit_baselines.sh     # L-BFGS only
#
# Task ID mapping:   task_id = config_index * 3 + seed_index
#   - config_index: 0..1  (bfgs_vanilla_64, lbfgs_64)
#   - seed_index:   0..2  → seeds [1, 2, 3]
#
# =============================================================================

#SBATCH --job-name=ch_baselines
#SBATCH --array=0-5
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch
#SBATCH --output=slurm_logs/baselines_%A_%a.out
#SBATCH --error=slurm_logs/baselines_%A_%a.err

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
source /oscar/home/emaciaso/pinn_env/bin/activate
cd /oscar/home/emaciaso/DRP-Cahn-Hilliard-PINN
mkdir -p slurm_logs

# ---------------------------------------------------------------------------
# Config array — 2 baseline configs
# ---------------------------------------------------------------------------
CONFIGS=(
    configs/ssbroyden/bfgs_vanilla_64.yaml    # 0 — vanilla BFGS (control)
    configs/ssbroyden/lbfgs_64.yaml           # 1 — L-BFGS-B
)

SEEDS=(1 2 3)

# ---------------------------------------------------------------------------
# Map SLURM_ARRAY_TASK_ID → (config, seed)
# ---------------------------------------------------------------------------
N_SEEDS=${#SEEDS[@]}
CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))

CONFIG_FILE="${CONFIGS[$CONFIG_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [ -z "$CONFIG_FILE" ]; then
    echo "ERROR: No config for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID (config_idx=$CONFIG_IDX)"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "========================================================================"
echo "SLURM Job ID:       $SLURM_JOB_ID"
echo "Array Task ID:      $SLURM_ARRAY_TASK_ID"
echo "Config index:       $CONFIG_IDX / 1"
echo "Config file:        $CONFIG_FILE"
echo "Seed:               $SEED"
echo "Hostname:           $(hostname)"
echo "Date:               $(date)"
echo "Working directory:  $(pwd)"
echo "Python:             $(which python)"
echo "========================================================================"

# ---------------------------------------------------------------------------
# Run the experiment
# ---------------------------------------------------------------------------
python -u run_experiment.py "$CONFIG_FILE" --seed "$SEED"
RET=$?

echo ""
echo "========================================================================"
echo "Job finished at $(date)"
echo "Exit code: $RET"
echo "========================================================================"

exit $RET
