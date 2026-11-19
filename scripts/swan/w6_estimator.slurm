#!/bin/bash
# Week 6: the cover read-out estimator of each fold, its gate on real held-out photos, and its read-out of every photo.
#SBATCH --job-name=wheat-w6-estimator
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_24gb|gpu_32gb|gpu_48gb|gpu_140gb
#SBATCH --array=0-5
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/week6/slurm/%x-%A_%a.out
source scripts/swan/env_gpu.sh
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU:', torch.cuda.get_device_name(0))"
FOLDS=(2016 2017 2018 2019 2021 2022)
python -u src/run_week6.py estimator ${FOLDS[$SLURM_ARRAY_TASK_ID]}
