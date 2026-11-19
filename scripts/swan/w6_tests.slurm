#!/bin/bash
# Week 6: the full test suite (the new generation tests run tiny random networks on the CPU).
#SBATCH --job-name=wheat-w6-tests
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/week6/slurm/%x-%j.out
source scripts/swan/env_gpu.sh
python -u -m pytest tests -q -p no:warnings
