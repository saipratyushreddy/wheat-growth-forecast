# Sourced by every job script: go to the project, activate its .venv, use the CPUs SLURM gave us.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
test -f data/derived/near_samples.parquet || { echo "data/derived missing: run scripts/swan/push_to_swan.sh on the laptop"; exit 1; }
