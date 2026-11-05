#!/bin/bash
# Week 3: stream the FIP1 photos in 48 shards (download at the pinned revision -> verify SHA-256 -> DINOv2 features +
# thumbnail -> delete). At most 8 shards at once, to stay under the Hugging Face rate limit. PNGs only ever sit in the
# node's local $TMPDIR, never on /lustre. A finished shard is skipped on resubmission.
#SBATCH --job-name=wheat-photos
#SBATCH --partition=batch
#SBATCH --array=0-47%8
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/week3/slurm/%x-%A_%a.out
source scripts/swan/env.sh
export TMPDIR=${TMPDIR:-/tmp}
python -u src/photos.py shard $SLURM_ARRAY_TASK_ID
