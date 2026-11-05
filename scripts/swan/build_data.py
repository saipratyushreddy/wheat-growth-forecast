"""
Rebuild data/ on Swan from scratch and check it matches the laptop's tables exactly.

  python scripts/swan/build_data.py            # download FIP1 (pinned revision), build data/derived, verify hashes
  python scripts/swan/build_data.py --record   # laptop only: write scripts/swan/derived_hashes.json from data/derived

Content hashes are taken over the table values (pandas row hashes), not file bytes.
"""
import argparse
import json
import os
import subprocess
import sys

import pandas as pd

REVISION = "11cdd84bc0ba8d5c6641ea7b53550b1658527e1c"  # mikeboss/FIP1 revision used on the laptop
TABLES = ["plots", "series", "weather", "weather2", "near_samples"]
HASHES = os.path.join(os.path.dirname(__file__), "derived_hashes.json")


def content_hash(path):
    d = pd.read_parquet(path)
    d = d[sorted(d.columns)]
    return {"rows": len(d), "cols": list(d.columns),
            "hash": format(int(pd.util.hash_pandas_object(d, index=False).sum()) & (2 ** 64 - 1), "016x")}


def hashes():
    return {t: content_hash(f"data/derived/{t}.parquet") for t in TABLES}


def build():
    if not os.path.exists("data/fip1/data"):
        from huggingface_hub import snapshot_download
        snapshot_download("mikeboss/FIP1", repo_type="dataset", revision=REVISION, local_dir="data/fip1",
                          allow_patterns=["data/*.parquet", "miappe/*", "README.md"])
    env = dict(os.environ, PYTHONPATH="src")
    if not os.path.exists("data/derived/series.parquet"):
        subprocess.run([sys.executable, "src/series.py"], check=True, env=env)
    if not os.path.exists("data/derived/weather2.parquet"):
        subprocess.run([sys.executable, "src/series.py", "--weather-extra"], check=True, env=env)
    if not os.path.exists("data/derived/near_samples.parquet"):
        sys.path.insert(0, "src")
        import near as N
        from common import load_derived
        plots, series, _ = load_derived()
        series = series.merge(plots[["plot_uid", "genotype_id"]], on="plot_uid")
        N.build_samples(plots, series).to_parquet("data/derived/near_samples.parquet", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    if a.record:
        json.dump(hashes(), open(HASHES, "w"), indent=1)
        print("recorded", HASHES)
    else:
        build()
        want, got = json.load(open(HASHES)), hashes()
        bad = [t for t in TABLES if want[t] != got[t]]
        for t in TABLES:
            print(t, got[t]["rows"], got[t]["hash"], "OK" if t not in bad else f"MISMATCH (laptop {want[t]['hash']})")
        if bad:
            raise SystemExit(f"derived tables differ from the laptop's: {bad}")
        print("data/derived matches the laptop")
