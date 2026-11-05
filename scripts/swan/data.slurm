#!/bin/bash
# Download FIP1 (pinned revision), rebuild data/derived, and check it matches the laptop's tables. Skips finished steps.
#SBATCH --job-name=wheat-data
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/week2/slurm/%x-%j.out
set -euo pipefail
cd "${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
source .venv/bin/activate
python -u scripts/swan/build_data.py
