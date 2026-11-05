#!/bin/bash
# Combine the six held-out years and write the leave-one-year-out summaries.
#SBATCH --job-name=wheat-loyo-agg
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/week2/slurm/%x-%j.out
source scripts/swan/env.sh
python -u src/run_week2.py --protocol loyo
