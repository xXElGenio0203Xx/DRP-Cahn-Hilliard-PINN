#!/bin/bash
# =============================================================================
# submit_experiments.sh — SLURM job array for all 17 Cahn-Hilliard experiments
# =============================================================================
#
# Submits all 17 configs as a SLURM job array on OSCAR.
# Each array task runs one experiment independently.
#
# Usage:
#   sbatch submit_experiments.sh          # submit all 17
#   sbatch --array=0 submit_experiments.sh   # submit only canonical
#   sbatch --array=0-4 submit_experiments.sh # submit first 5
#
# Monitor:
#   squeue -u $USER                       # check status
#   cat slurm_logs/ch_pinn_<jobid>_<taskid>.out  # check output
#
# =============================================================================

#SBATCH --job-name=ch_pinn
#SBATCH --array=0-16
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/ch_pinn_%A_%a.out
#SBATCH --error=slurm_logs/ch_pinn_%A_%a.err
#SBATCH --mail-type=END,FAIL

# --- Config file lookup table (indexed by SLURM_ARRAY_TASK_ID) ---------------
# Alphabetically sorted to match `ls configs/`
CONFIGS=(
    "configs/adam_lr_high.yaml"          # 0
    "configs/adam_lr_low.yaml"           # 1
    "configs/adam_only.yaml"             # 2
    "configs/bfgs_batch_1000.yaml"       # 3
    "configs/bfgs_batch_200.yaml"        # 4
    "configs/bfgs_vanilla.yaml"          # 5
    "configs/cahn_hilliard_canonical.yaml" # 6
    "configs/hessian_scaled.yaml"        # 7
    "configs/lbfgs.yaml"                 # 8
    "configs/power_2.yaml"              # 9
    "configs/power_4.yaml"              # 10
    "configs/ssbfgs_ab.yaml"            # 11
    "configs/ssbfgs_ol.yaml"            # 12
    "configs/ssbroyden1.yaml"           # 13
    "configs/warmup_0.yaml"             # 14
    "configs/warmup_500.yaml"           # 15
    "configs/warmup_5000.yaml"          # 16
)

# --- Validate task ID --------------------------------------------------------
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: Not running inside a SLURM job array."
    echo "Usage: sbatch submit_experiments.sh"
    exit 1
fi

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$CONFIG" ]; then
    echo "ERROR: No config for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

# --- Setup -------------------------------------------------------------------
echo "========================================"
echo "SLURM Job Array Task: $SLURM_ARRAY_TASK_ID / 16"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Config: $CONFIG"
echo "Date: $(date)"
echo "========================================"

# Create log directory
mkdir -p slurm_logs

# Activate virtual environment
source "$HOME/pinn_env/bin/activate"

# Move to the project directory (adjust if you cloned elsewhere)
cd "$SLURM_SUBMIT_DIR"

# --- Run experiment ----------------------------------------------------------
echo ""
echo "Starting experiment..."
python -u run_experiment.py "$CONFIG"
EXIT_CODE=$?

echo ""
echo "========================================"
echo "Exit code: $EXIT_CODE"
echo "Finished: $(date)"
echo "========================================"

exit $EXIT_CODE
