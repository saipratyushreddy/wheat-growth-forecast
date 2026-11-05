#!/bin/bash
# Official test splits, scored once (the runner refuses a second run).
#SBATCH --job-name=wheat-test
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/week2/slurm/%x-%j.out
source scripts/swan/env.sh
python -u src/run_week2.py --protocol official --eval test
