"""
Week 4: download FIP1SOY (pinned revision), build data/derived_soy, and check it matches the laptop's tables exactly.

  python scripts/swan/build_soy.py            # on Swan
  python scripts/swan/build_soy.py --record   # laptop only: write scripts/swan/derived_soy_hashes.json
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from build_data import content_hash  # noqa: E402

TABLES = ["plots", "series", "weather", "site_weather", "near_samples"]
HASHES = os.path.join(os.path.dirname(__file__), "derived_soy_hashes.json")


def hashes():
    return {t: content_hash(f"data/derived_soy/{t}.parquet") for t in TABLES}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    if a.record:
        json.dump(hashes(), open(HASHES, "w"), indent=1)
        sys.exit(0)
    env = dict(os.environ, PYTHONPATH="src")
    if not os.path.exists("data/fip1soy/data-00000-of-00003.arrow"):
        subprocess.run([sys.executable, "src/soy.py", "download"], check=True, env=env)
    subprocess.run([sys.executable, "src/soy.py", "build"], check=True, env=env)
    want, got = json.load(open(HASHES)), hashes()
    bad = [t for t in TABLES if want[t] != got[t]]
    if bad:
        sys.exit(f"soybean tables differ from the laptop's: {bad}")
    print("soybean tables match the laptop's:", TABLES)
