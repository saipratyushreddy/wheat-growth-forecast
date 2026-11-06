#!/bin/bash
# Week 4 (docs/05-week4-design.md), from the repo root on Swan after git pull:
# soybean data -> tests -> 7 soybean folds -> soybean scoring (once) -> retrieval (once).
set -euo pipefail
mkdir -p outputs/week4/slurm
d=$(sbatch --parsable scripts/swan/w4_data.slurm)
t=$(sbatch --parsable --dependency=afterok:$d scripts/swan/w4_tests.slurm)
u=$(sbatch --parsable --dependency=afterok:$t scripts/swan/w4_soy.slurm)
s=$(sbatch --parsable --dependency=afterok:$u --export=ALL,WHAT=soy_score scripts/swan/w4_score.slurm)
r=$(sbatch --parsable --dependency=afterok:$t --export=ALL,WHAT=retrieval scripts/swan/w4_score.slurm)
echo "data $d, tests $t, soybean folds $u (7), soybean score $s, retrieval $r"
