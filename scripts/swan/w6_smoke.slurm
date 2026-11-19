#!/bin/bash
# Week 6 smoke test (code check, never a result): 300 training steps on fold 2019 and 16 generated queries, in the
# separate smoke directories. Only runtime, memory and "the loss is finite" are looked at.
#SBATCH --job-name=wheat-w6-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_l40s|gpu_a30
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=outputs/week6/slurm/%x-%j.out
source scripts/swan/env_gpu.sh
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU:', torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,memory.total --format=csv
export WEEK6_TAG=smoke
python -u src/run_week6.py train 2019 --steps 300
python -u src/run_week6.py generate 2019 --limit 16
python -c "import torch; print('peak GPU memory (GB):', round(torch.cuda.max_memory_allocated()/1e9, 1))" || true
