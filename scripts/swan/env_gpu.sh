# Sourced by every week-6 job script: go to the project, activate the GPU environment (.venv-gpu, see setup_gpu.sh).
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"
source .venv-gpu/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf-cache}"   # model weights (a few GB) go next to the project, not into $HOME
mkdir -p outputs/week6/slurm data/gen
test -f data/photos/thumbs_2016.parquet || { echo "thumbnails missing (week-3 photo job has not run)"; exit 1; }
