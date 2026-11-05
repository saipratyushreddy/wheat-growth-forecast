#!/bin/bash
# Inner leave-one-year-out tuning: one array task per held-out training year.
#SBATCH --job-name=wheat-tune
#SBATCH --array=0-4
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/week2/slurm/%x-%A_%a.out
source scripts/swan/env.sh
YEARS=(2016 2017 2018 2021 2022)
python -u src/run_week2.py --stage tune --year ${YEARS[$SLURM_ARRAY_TASK_ID]}
