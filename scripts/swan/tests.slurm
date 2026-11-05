#!/bin/bash
#SBATCH --job-name=wheat-tests
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/week2/slurm/%x-%j.out
source scripts/swan/env.sh
python -m pytest tests -q -p no:warnings
