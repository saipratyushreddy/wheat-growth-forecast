# Sourced by every job script: activate the env, go to the repo root, use the CPUs SLURM gave us.
module load anaconda
source activate "$WORK/conda/wheat" 2>/dev/null || conda activate "$WORK/conda/wheat"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=src
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
mkdir -p outputs/week2/slurm
