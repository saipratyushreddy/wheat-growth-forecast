# Sourced by every job script: go to the project, activate its .venv, use the CPUs SLURM gave us.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
mkdir -p outputs/week2/slurm outputs/week3/slurm outputs/week4/slurm outputs/week5/slurm
test -f data/derived/near_samples.parquet || { echo "data/derived missing: the data job (scripts/swan/data.slurm) has not run"; exit 1; }
