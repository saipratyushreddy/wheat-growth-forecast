# Wheat growth forecasting (FIP 1.0)

Forecasting a wheat plot's future growth (canopy cover, height, head count) and end-of-season outcomes (heading, final height, yield)
from what has been observed so far (images, measured traits, weather, genetics), tested on unseen plots, varieties and years, with
calibrated prediction ranges.

Data: FIP 1.0 (Roth et al., *GigaScience* 2025, doi:10.1093/gigascience/giaf051; Hugging Face `mikeboss/FIP1`). Data are not committed.
Builds on the evaluation design of the earlier cauliflower project (GrowliFlower).

Status: [plan](docs/01-plan.md) (includes the data check); week 1 baselines: [design and results](docs/02-baselines-design.md); week 2 learned models: [design](docs/03-week2-design.md).

Setup: `python3 -m venv .venv && .venv/bin/pip install pandas pyarrow numpy scikit-learn scipy huggingface_hub matplotlib pytest`,
then download the tables: `python -c "from huggingface_hub import snapshot_download as s; s('mikeboss/FIP1', repo_type='dataset', local_dir='data/fip1', allow_patterns=['data/*.parquet','miappe/*','README.md'])"`
and run `python src/audit.py`. Week 1: `python src/series.py`, `pytest tests`, then `PYTHONPATH=src python src/run_week1.py --protocol official --eval validation` (then `--eval test` once, then `--protocol loyo`).
