#!/bin/bash
# Week 4: download FIP1SOY (pinned revision), build data/derived_soy, check it matches the laptop's tables.
#SBATCH --job-name=wheat-w4-data
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/week4/slurm/%x-%j.out
source scripts/swan/env.sh
python -u scripts/swan/build_soy.py
