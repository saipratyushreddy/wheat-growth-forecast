#!/bin/bash
# Week 5 (docs/07-weather-upper-bound-design.md), from the repo root on Swan after git pull:
# tests -> 6 wheat folds + soybean -> scoring (once).
set -euo pipefail
mkdir -p outputs/week5/slurm
t=$(sbatch --parsable scripts/swan/w5_tests.slurm)
u=$(sbatch --parsable --dependency=afterok:$t scripts/swan/w5_units.slurm)
s=$(sbatch --parsable --dependency=afterok:$u scripts/swan/w5_score.slurm)
echo "tests $t, units $u (7), score $s"
