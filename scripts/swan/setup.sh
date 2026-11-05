#!/bin/bash
# One-time setup on a Swan login node, from the repo root in $WORK:  bash scripts/swan/setup.sh
# Creates a conda env in $WORK with the same package versions as the local .venv.
set -euo pipefail
module load anaconda
ENV=$WORK/conda/wheat
if [ ! -d "$ENV" ]; then
    conda create -y -p "$ENV" -c conda-forge python=3.14
fi
source activate "$ENV" 2>/dev/null || conda activate "$ENV"
pip install numpy==2.5.4 pandas==3.1.1 pyarrow==26.0.1 scipy==1.18.1 scikit-learn==1.9.1 pytest==9.2.0
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -c "import numpy, pandas, sklearn, torch; print('ok', numpy.__version__, pandas.__version__, sklearn.__version__, torch.__version__)"
