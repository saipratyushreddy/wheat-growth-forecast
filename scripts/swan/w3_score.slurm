#!/bin/bash
# Week 3 scoring: pass validation, test (once) or loyo:  sbatch --export=ALL,WHAT=validation scripts/swan/w3_score.slurm
#SBATCH --job-name=wheat-w3-score
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/week3/slurm/%x-%j.out
source scripts/swan/env.sh
python -u src/run_week3.py score "$WHAT"
