#!/bin/bash
# Week 6, part 1 (docs/08-generation-design.md), from the repo root on Swan after git pull and setup_gpu.sh:
# tests -> thumbnails to latents and features (6 years) -> cover estimators with their gates (6 folds) -> smoke test.
# Nothing here is a result. Look at the smoke test's speed and memory, then fix the number of steps (a clarification).
set -euo pipefail
mkdir -p outputs/week6/slurm
t=$(sbatch --parsable scripts/swan/w6_tests.slurm)
p=$(sbatch --parsable --dependency=afterok:$t scripts/swan/w6_prep.slurm)
e=$(sbatch --parsable --dependency=afterok:$p scripts/swan/w6_estimator.slurm)
s=$(sbatch --parsable --dependency=afterok:$e scripts/swan/w6_smoke.slurm)
echo "tests $t, prep $p (6), estimators $e (6), smoke $s"
