#!/bin/bash
# From the repo root on Swan (after git pull and bash scripts/swan/setup.sh): photo index -> 48 shards -> merge.
set -euo pipefail
mkdir -p outputs/week3/slurm
source .venv/bin/activate
PYTHONPATH=src python src/photos.py index
python -c "from huggingface_hub import get_token; print('Hugging Face token:', 'found' if get_token() else 'none (slower, rate-limited)')"
p=$(sbatch --parsable scripts/swan/photos.slurm)
m=$(sbatch --parsable --dependency=afterok:$p scripts/swan/photos_merge.slurm)
echo "photos $p (48 shards, 8 at a time), merge $m"
