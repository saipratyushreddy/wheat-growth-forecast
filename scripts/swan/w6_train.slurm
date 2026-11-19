#!/bin/bash
# Week 6: fine-tune the generator of each fold. STEPS (and the time limit) are fixed in a clarification of docs/08 first.
# Resumable: resubmitting continues from the last checkpoint.
#SBATCH --job-name=wheat-w6-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_24gb|gpu_32gb|gpu_48gb|gpu_140gb
#SBATCH --array=0-5
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/week6/slurm/%x-%A_%a.out
source scripts/swan/env_gpu.sh
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU:', torch.cuda.get_device_name(0))"
FOLDS=(2016 2017 2018 2019 2021 2022)
python -u src/run_week6.py train ${FOLDS[$SLURM_ARRAY_TASK_ID]} --steps "${STEPS:?give STEPS}"
