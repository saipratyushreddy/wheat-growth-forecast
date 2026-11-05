#!/bin/bash
# One-time setup on a Swan login node, from the project directory (as in the cauliflower project: a .venv in the repo):
#   cd /lustre/work/cseguo/skasara2 && git clone https://github.com/saipratyushreddy/wheat-growth-forecast.git
#   cd wheat-growth-forecast && bash scripts/swan/setup.sh
# Same package versions as the laptop's .venv (Python 3.14). If no python3.14 is on the PATH, a Python 3.14
# interpreter is made once with conda in $WORK/conda/py314 and the .venv is built from it.
set -euo pipefail
if command -v python3.14 >/dev/null; then
    PY=python3.14
else
    if [ ! -x "$WORK/conda/py314/bin/python3.14" ]; then
        module load anaconda
        conda create -y -p "$WORK/conda/py314" -c conda-forge python=3.14
    fi
    PY="$WORK/conda/py314/bin/python3.14"
fi
[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy==2.5.4 pandas==3.1.1 pyarrow==26.0.1 scipy==1.18.1 scikit-learn==1.9.1 pytest==9.2.0 huggingface_hub==2.4.0
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu   # CPU build: week 2 needs no GPU
python -c "import numpy, pandas, sklearn, torch; print('ok', numpy.__version__, pandas.__version__, sklearn.__version__, torch.__version__)"
