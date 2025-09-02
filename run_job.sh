#!/bin/bash
# SLURM SBATCH HEADER ----------------------------------------------------------
#SBATCH --account= <YOUR ACCOUNT NAME>
#SBATCH --job-name=sbi_pipeline
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out

# FAIL FAST + NICE LOGGING -----------------------------------------------------
set -Eeuo pipefail
echo "[$(date)] Job starting on $(hostname) with SLURM_JOBID=${SLURM_JOB_ID:-N/A}"
echo "CWD: $(pwd)"

# MODULES / ENV ----------------------------------------------------------------
module load scipy-stack
module load httpproxy

# Activate your venv (adjust if different)
source venv/bin/activate

# Threading + non-interactive plotting
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

# KNOBS (override via: sbatch --export=ALL,NUM_SIMS=100000,MAX_EPOCHS=200,LR=5e-4 run_job.sh)
PY=${PY:-python}

# sims
NUM_SIMS=${NUM_SIMS:-100000}
SHARD_SIZE=${SHARD_SIZE:-10000}
SEED=${SEED:-0}

echo "[$(date)] Python: $(which ${PY})"
echo "[$(date)] NV: $(python -c 'import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())')"

# PATHS (optional sanity print)
echo "[$(date)] BASE_SIMS_PATH from config.py should be used by scripts."

# STEP 1 — Base simulations -----------------------------------------------------
echo "[$(date)] Step 1: Generating base simulations (NUM_SIMS=${NUM_SIMS}, SHARD_SIZE=${SHARD_SIZE})"
srun --ntasks=1 ${PY} 1_generate_base_simulations.py \
  --num-sims "${NUM_SIMS}" \
  --shard-size "${SHARD_SIZE}" \
  --seed "${SEED}"

# STEP 2 — Train ---------------------------------------------------------------
echo "[$(date)] Step 2: Training"
srun --ntasks=1 ${PY} 2_train_model.py \
  --seed "${SEED}"

echo "[$(date)] Job finished OK."
