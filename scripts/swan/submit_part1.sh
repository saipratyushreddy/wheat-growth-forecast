#!/bin/bash
# From the repo root on Swan (after git pull):
#   data (download + rebuild + verify) -> tests -> tuning (5 years in parallel) -> settings + validation sanity check.
set -euo pipefail
mkdir -p outputs/week2/slurm
d=$(sbatch --parsable scripts/swan/data.slurm)
t=$(sbatch --parsable --dependency=afterok:$d scripts/swan/tests.slurm)
u=$(sbatch --parsable --dependency=afterok:$t scripts/swan/tune.slurm)
v=$(sbatch --parsable --dependency=afterok:$u scripts/swan/validate.slurm)
echo "data $d, tests $t, tune $u, validate $v"
