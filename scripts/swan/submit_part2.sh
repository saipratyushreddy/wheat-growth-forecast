#!/bin/bash
# Only after the validation sanity check has been reviewed: official test once -> leave-one-year-out -> summaries.
set -euo pipefail
t=$(sbatch --parsable scripts/swan/test.slurm)
l=$(sbatch --parsable --dependency=afterok:$t scripts/swan/loyo.slurm)
a=$(sbatch --parsable --dependency=afterok:$l scripts/swan/loyo_aggregate.slurm)
echo "test $t, loyo $l, aggregate $a"
