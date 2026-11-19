#!/bin/bash
# Week 6: scoring, once (docs/08).
#SBATCH --job-name=wheat-w6-score
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/week6/slurm/%x-%j.out
source scripts/swan/env_gpu.sh
python -u src/run_week6.py score
