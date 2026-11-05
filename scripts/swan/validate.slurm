#!/bin/bash
# Aggregate tuning into settings.json, then the validation sanity check (fit on train).
#SBATCH --job-name=wheat-validate
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/week2/slurm/%x-%j.out
source scripts/swan/env.sh
python src/run_week2.py --stage tune
test -f outputs/week2/settings.json
python src/run_week2.py --protocol official --eval validation
