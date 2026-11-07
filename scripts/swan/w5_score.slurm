#!/bin/bash
# Week 5: scoring, once (docs/07).
#SBATCH --job-name=wheat-w5-score
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/week5/slurm/%x-%j.out
source scripts/swan/env.sh
python -u src/run_week5.py score
