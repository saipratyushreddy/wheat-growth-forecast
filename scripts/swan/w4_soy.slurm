#!/bin/bash
# Week 4 part A: soybean leave-one-season-out, one task per held-out season (2016-2022), wheat settings frozen.
#SBATCH --job-name=wheat-w4-soy
#SBATCH --partition=batch
#SBATCH --array=0-6
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/week4/slurm/%x-%A_%a.out
source scripts/swan/env.sh
python -u src/run_week4.py soy_unit $SLURM_ARRAY_TASK_ID
