# Wheat growth forecasting (FIP 1.0)

Forecasting a wheat plot's future growth (canopy cover, height, head count) and end-of-season outcomes (heading, final height, yield)
from what has been observed so far (images, measured traits, weather, genetics), tested on unseen plots, varieties and years, with
calibrated prediction ranges.

Data: FIP 1.0 (Roth et al., *GigaScience* 2025, doi:10.1093/gigascience/giaf051; Hugging Face `mikeboss/FIP1`). Data are not committed.
Builds on the evaluation design of the earlier cauliflower project (GrowliFlower).

Status: [plan](docs/01-plan.md) (includes the data check); week 1 baselines: [design and results](docs/02-baselines-design.md); week 2 learned models: [design and results](docs/03-week2-design.md); week 3 photos and ranges: [design and results](docs/04-week3-design.md); week 4 soybean replication and future-photo retrieval: [design and results](docs/05-week4-design.md). **Report: [docs/06-report.md](docs/06-report.md).**

Setup: `python3 -m venv .venv && .venv/bin/pip install pandas pyarrow numpy scikit-learn scipy huggingface_hub matplotlib pytest`,
then download the tables: `python -c "from huggingface_hub import snapshot_download as s; s('mikeboss/FIP1', repo_type='dataset', local_dir='data/fip1', allow_patterns=['data/*.parquet','miappe/*','README.md'])"`
and run `python src/audit.py`. Week 1: `python src/series.py`, `pytest tests`, then `PYTHONPATH=src python src/run_week1.py --protocol official --eval validation` (then `--eval test` once, then `--protocol loyo`).

Swan (UNL HCC): week-2 runs go there. Clone this repo in `/lustre/work/cseguo/skasara2`, run `bash scripts/swan/setup.sh` once, then
`bash scripts/swan/submit_part1.sh` (downloads FIP1 at a pinned revision, rebuilds `data/derived` and checks it matches the laptop's
tables, runs the tests, tunes, then the validation check). After review, `bash scripts/swan/submit_part2.sh`. Results come back to the
laptop with `scripts/swan/pull_from_swan.sh`.
