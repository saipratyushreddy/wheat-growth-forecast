#!/bin/bash
# One-time, on a Swan login node, from the project directory: a second environment for the week-6 GPU work (CUDA build of
# PyTorch from PyPI, diffusers). The CPU .venv of weeks 1-5 is left as it is.
#   cd /lustre/work/cseguo/skasara2/wheat-growth-forecast && git pull && bash scripts/swan/setup_gpu.sh
set -euo pipefail
BASE=$(.venv/bin/python -c "import sys; print(sys.base_prefix)")
[ -d .venv-gpu ] || "$BASE/bin/python3.14" -m venv .venv-gpu
source .venv-gpu/bin/activate
pip install --upgrade pip
pip install numpy==2.5.4 pandas==3.1.1 pyarrow==26.0.1 scipy==1.18.1 scikit-learn==1.9.1 pytest==9.2.0 huggingface_hub==2.4.0 transformers==5.22.1 pillow==13.0.0 matplotlib==3.11.2
pip install torch==2.15.0 diffusers==0.43.0
python -c "import torch, diffusers, transformers; print('ok torch', torch.__version__, 'cuda build', torch.version.cuda, 'diffusers', diffusers.__version__)"
echo "On a GPU node, check: srun -p gpu --gres=gpu:1 --constraint='gpu_l40s|gpu_a30' --time=00:10:00 .venv-gpu/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'"
