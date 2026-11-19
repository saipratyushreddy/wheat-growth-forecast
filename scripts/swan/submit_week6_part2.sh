#!/bin/bash
# Week 6, part 2: train the six generators, generate, score once. Usage: bash scripts/swan/submit_week6_part2.sh STEPS [TIME]
# STEPS is the number fixed in docs/08 (a clarification made after the smoke test); TIME the time limit per training task.
set -euo pipefail
STEPS="${1:?give the number of training steps fixed in docs/08}"
TIME="${2:-12:00:00}"
mkdir -p outputs/week6/slurm
tr=$(sbatch --parsable --time="$TIME" --export=ALL,STEPS="$STEPS" scripts/swan/w6_train.slurm)
g=$(sbatch --parsable --dependency=afterok:$tr scripts/swan/w6_generate.slurm)
s=$(sbatch --parsable --dependency=afterok:$g scripts/swan/w6_score.slurm)
echo "train $tr (6, $STEPS steps, $TIME), generate $g (6), score $s"
