#!/bin/bash
# From the repo root on Swan: tests -> tuning (5 years in parallel) -> settings + validation sanity check.
set -euo pipefail
mkdir -p outputs/week2/slurm
t=$(sbatch --parsable scripts/swan/tests.slurm)
u=$(sbatch --parsable --dependency=afterok:$t scripts/swan/tune.slurm)
v=$(sbatch --parsable --dependency=afterok:$u scripts/swan/validate.slurm)
echo "tests $t, tune $u, validate $v"
