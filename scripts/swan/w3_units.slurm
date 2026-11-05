#!/bin/bash
# Week 3: 42 fit units (outer fold x inner held-out year), week-2 settings frozen. A finished unit is skipped.
#SBATCH --job-name=wheat-w3-unit
#SBATCH --partition=batch
#SBATCH --array=0-41
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/week3/slurm/%x-%A_%a.out
source scripts/swan/env.sh
python -u src/run_week3.py unit $SLURM_ARRAY_TASK_ID
