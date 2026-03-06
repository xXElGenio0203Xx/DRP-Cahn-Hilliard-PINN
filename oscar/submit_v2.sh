#!/bin/bash
# =============================================================================
# submit_v2.sh — SLURM Array Job for Cahn-Hilliard PINN Experiment Suite v2
# =============================================================================
#
# Launches 20 configs × 3 seeds = 60 jobs as a SLURM array.
#
# Usage:
#     sbatch oscar/submit_v2.sh              # submit all 60 jobs
#     sbatch --array=0-5 oscar/submit_v2.sh  # submit only first 2 configs
#
# Task ID mapping:   task_id = config_index * 3 + seed_index
#   - config_index: 0..19  (alphabetical order of configs)
#   - seed_index:   0..2   → seeds [1, 2, 3]
#
# Resource estimate (per job, CPU-only float64 PINN):
#   - 51K iterations × ~0.5s each ≈ ~7 hours (small network)
#   - Larger networks (C1, C2) may take ~12-18 hours
#   - 4GB memory is sufficient for all configs (largest Hessian ~120 MB)
#
# =============================================================================

#SBATCH --job-name=ch_pinn_v2
#SBATCH --array=0-59
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
# Config array — alphabetical order matches generate_configs.py
# ---------------------------------------------------------------------------
CONFIGS=(
    configs/A1_bfgs.yaml
    configs/A2_ssbfgs_ol.yaml
    configs/A3_ssbfgs_ab.yaml
    configs/A4_ssbroyden1.yaml
    configs/A5_ssbroyden2.yaml
    configs/A6_lbfgs.yaml
    configs/B1_warmup_0.yaml
    configs/B2_warmup_5000.yaml
    configs/B3_warmup_10000.yaml
    configs/C1_deep_8x20.yaml
    configs/C2_wide_5x30.yaml
    configs/D1_adam_only.yaml
    configs/E1_radam_1000.yaml
    configs/E2_radam_5000.yaml
    configs/F1_nchange_200.yaml
    configs/F2_nchange_1000.yaml
    configs/F3_points_20k.yaml
    configs/G1_hessian_scaled.yaml
    configs/G2_power_2.yaml
    configs/G3_power_4.yaml
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

echo "========================================================================"
echo "SLURM Job ID:       $SLURM_JOB_ID"
echo "Array Task ID:      $SLURM_ARRAY_TASK_ID"
echo "Config index:       $CONFIG_IDX"
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
