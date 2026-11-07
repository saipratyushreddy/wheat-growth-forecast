#!/bin/bash
#SBATCH --job-name=wheat-w5-tests
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/week5/slurm/%x-%j.out
source scripts/swan/env.sh
python -u -m pytest tests -q -p no:warnings
