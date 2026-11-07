#!/bin/bash
# Week 5 (docs/07): one wheat fold per task (held-out year), switch off and on; task 6 = soybean N3 / N3-true.
#SBATCH --job-name=wheat-w5-unit
#SBATCH --partition=batch
#SBATCH --array=0-6
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/week5/slurm/%x-%A_%a.out
source scripts/swan/env.sh
YEARS=(2016 2017 2018 2019 2021 2022)
if [ "$SLURM_ARRAY_TASK_ID" -lt 6 ]; then
    python -u src/run_week5.py unit ${YEARS[$SLURM_ARRAY_TASK_ID]}
else
    python -u src/run_week5.py soy
fi
