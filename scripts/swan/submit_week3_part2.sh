#!/bin/bash
# Only after the validation check has been reviewed: official test once -> leave-one-year-out.
set -euo pipefail
t=$(sbatch --parsable --export=ALL,WHAT=test scripts/swan/w3_score.slurm)
l=$(sbatch --parsable --dependency=afterok:$t --export=ALL,WHAT=loyo scripts/swan/w3_score.slurm)
echo "test $t, loyo $l"
