#!/bin/bash
# After the photo merge has finished (or pass its job id to wait for it):  bash scripts/swan/submit_week3_part1.sh [JOBID]
# tests (including the photo recomputation check) -> 42 fit units -> validation sanity check.
set -euo pipefail
mkdir -p outputs/week3/slurm
dep=${1:+--dependency=afterok:$1}
t=$(sbatch --parsable $dep scripts/swan/tests.slurm)
u=$(sbatch --parsable --dependency=afterok:$t scripts/swan/w3_units.slurm)
v=$(sbatch --parsable --dependency=afterok:$u --export=ALL,WHAT=validation scripts/swan/w3_score.slurm)
echo "tests $t, units $u (42), validation $v"
