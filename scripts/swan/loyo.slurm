#!/bin/bash
# Leave-one-year-out (headline): one array task per held-out year.
#SBATCH --job-name=wheat-loyo
#SBATCH --array=0-5
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/week2/slurm/%x-%A_%a.out
source scripts/swan/env.sh
YEARS=(2016 2017 2018 2019 2021 2022)
python -u src/run_week2.py --protocol loyo --year ${YEARS[$SLURM_ARRAY_TASK_ID]}
