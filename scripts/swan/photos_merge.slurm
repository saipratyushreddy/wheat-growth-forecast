#!/bin/bash
# Merge the 48 shards into one features table and one thumbnail table per year, then delete the shards.
#SBATCH --job-name=wheat-photos-merge
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/week3/slurm/%x-%j.out
source scripts/swan/env.sh
python -u src/photos.py merge
ls -la data/photos
