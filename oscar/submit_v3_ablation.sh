#!/bin/bash
# =============================================================================
# submit_v3_ablation.sh — SLURM Array Job for V3 64x64 Ablations
# =============================================================================
#
# Launches 11 configs × 3 seeds = 33 jobs as a SLURM array.
#
# All configs use SSBroyden2 on the 64×64 domain (our best setup, L2≈0.317)
# and vary one or more hyperparameters to push accuracy further.
#
# Usage:
#     sbatch oscar/submit_v3_ablation.sh                    # submit all 33 jobs
#     sbatch --array=0-2 oscar/submit_v3_ablation.sh        # V3_A only (3 seeds)
#     sbatch --array=24-32 oscar/submit_v3_ablation.sh      # power configs only
#
# Task ID mapping:   task_id = config_index * 3 + seed_index
#   - config_index: 0..10  (V3_A through V3_K)
#   - seed_index:   0..2   → seeds [1, 2, 3]
#
# Config index reference:
#   0  V3_A  n_interior=16K          |  6  V3_G  combined + batch=200
#   1  V3_B  n_initial=1K            |  7  V3_H  combined + warmup=2K
#   2  V3_C  RAD k2=0.5              |  8  V3_I  power=2
#   3  V3_D  batch_size=200          |  9  V3_J  power=4
#   4  V3_E  batch_size=1000         | 10  V3_K  power=4 + combined
#   5  V3_F  combined sampling       |
#
# Resource estimate (per job, CPU-only float64 PINN, 1K Adam + 50K BFGS):
#   - 6-12 hours wall time
#   - 16GB memory (autograd + 50K RAD candidates + Hessian approx)
#   - V3_A/F/G/H/K use 16K interior points → may need slightly more memory
#
# =============================================================================

#SBATCH --job-name=ch_v3_abl
#SBATCH --array=0-32
#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch
#SBATCH --output=slurm_logs/v3_abl_%A_%a.out
#SBATCH --error=slurm_logs/v3_abl_%A_%a.err

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
source /oscar/home/emaciaso/pinn_env/bin/activate
cd /oscar/home/emaciaso/DRP-Cahn-Hilliard-PINN
mkdir -p slurm_logs

# ---------------------------------------------------------------------------
# Config array — 11 V3 ablation configs (all 64×64 SSBroyden2)
# ---------------------------------------------------------------------------
CONFIGS=(
    configs/v3_64_ablation/V3_A_nint_16k.yaml           #  0 — n_interior=16K
    configs/v3_64_ablation/V3_B_nic_1k.yaml              #  1 — n_initial=1K
    configs/v3_64_ablation/V3_C_rad_k2_0p5.yaml          #  2 — RAD k2=0.5
    configs/v3_64_ablation/V3_D_batch_200.yaml            #  3 — batch_size=200
    configs/v3_64_ablation/V3_E_batch_1000.yaml           #  4 — batch_size=1000
    configs/v3_64_ablation/V3_F_combined_sampling.yaml    #  5 — combined sampling
    configs/v3_64_ablation/V3_G_combined_batch200.yaml    #  6 — combined + batch=200
    configs/v3_64_ablation/V3_H_combined_warmup2k.yaml    #  7 — combined + warmup=2K
    configs/v3_64_ablation/V3_I_power_2.yaml              #  8 — power=2
    configs/v3_64_ablation/V3_J_power_4.yaml              #  9 — power=4
    configs/v3_64_ablation/V3_K_power4_combined.yaml      # 10 — power=4 + combined
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
echo "Config index:       $CONFIG_IDX / 10"
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
