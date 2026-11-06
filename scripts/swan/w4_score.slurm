#!/bin/bash
# Week 4: WHAT=soy_score (once) or retrieval (once; wheat photos, week-3 predictions).
#SBATCH --job-name=wheat-w4-score
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/week4/slurm/%x-%j.out
source scripts/swan/env.sh
python -u src/run_week4.py "$WHAT"
