#!/bin/bash
# =============================================================================
# submit_adam_only.sh — SLURM Array Job for Adam-Only Sweep (Burgers-matched)
# =============================================================================
#
# Launches 12 configs × 3 seeds = 36 jobs as a SLURM array.
#
# Usage:
#     sbatch oscar/submit_adam_only.sh                  # submit all 36 jobs
#     sbatch --array=0-2 oscar/submit_adam_only.sh      # submit only first config (3 seeds)
#     sbatch --array=33-35 oscar/submit_adam_only.sh    # submit only 51k_lr5e-3 (3 seeds)
#
# Task ID mapping:   task_id = config_index * 3 + seed_index
#   - config_index: 0..11  (alphabetical order of adam_only configs)
#   - seed_index:   0..2   → seeds [1, 2, 3]
#
# Resource estimate (per job, CPU-only float64 PINN, Adam-only):
#   - 1K  epochs:  ~10 min
#   - 5K  epochs:  ~50 min
#   - 10K epochs:  ~1.5 hours
#   - 51K epochs:  ~7 hours
#   No Hessian storage needed (Adam-only), so memory is light.
#
# =============================================================================

#SBATCH --job-name=ch_adam
#SBATCH --array=0-35
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch
#SBATCH --output=slurm_logs/adam_%A_%a.out
#SBATCH --error=slurm_logs/adam_%A_%a.err

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
# Config array — 12 Adam-only configs (4 epoch counts × 3 learning rates)
# Alphabetical order matches `ls configs/adam_only/`
# ---------------------------------------------------------------------------
CONFIGS=(
    configs/adam_only/adam_10k_lr1e-2.yaml    #  0
    configs/adam_only/adam_10k_lr1e-3.yaml    #  1
    configs/adam_only/adam_10k_lr5e-3.yaml    #  2
    configs/adam_only/adam_1k_lr1e-2.yaml     #  3
    configs/adam_only/adam_1k_lr1e-3.yaml     #  4
    configs/adam_only/adam_1k_lr5e-3.yaml     #  5
    configs/adam_only/adam_51k_lr1e-2.yaml    #  6
    configs/adam_only/adam_51k_lr1e-3.yaml    #  7
    configs/adam_only/adam_51k_lr5e-3.yaml    #  8  ★ primary Burgers-analog
    configs/adam_only/adam_5k_lr1e-2.yaml     #  9
    configs/adam_only/adam_5k_lr1e-3.yaml     # 10
    configs/adam_only/adam_5k_lr5e-3.yaml     # 11
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
echo "Config index:       $CONFIG_IDX / 11"
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

echo ""
echo "========================================================================"
echo "Job finished at $(date)"
echo "Exit code: $?"
echo "========================================================================"
