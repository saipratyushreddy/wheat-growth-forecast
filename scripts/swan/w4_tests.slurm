#!/bin/bash
#SBATCH --job-name=wheat-w4-tests
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/week4/slurm/%x-%j.out
source scripts/swan/env.sh
python -u -m pytest tests -q -p no:warnings
