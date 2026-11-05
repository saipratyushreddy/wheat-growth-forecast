"""
Load the FIP 1.0 tables (Hugging Face mikeboss/FIP1, downloaded to data/fip1) without the heavy alignment columns.

One row = one plot in one year. Split names follow the dataset: train, validation, test_plot, test_genotype, test_environment,
test_genotype_environment.
"""
import glob
import os

import pandas as pd
import pyarrow.parquet as pq

SPLITS = ["train", "validation", "test_plot", "test_genotype", "test_environment", "test_genotype_environment"]
SERIES = ["canopy_cover", "height", "spike_count", "senescence"]
FINALS = ["heading", "height_final", "yield", "yield_adjusted", "protein"]
WEATHER = ["temperature_air_10cm", "temperature_air_200cm", "temperature_soil_5cm", "humidity_air_10cm", "humidity_air_200cm",
           "precipitation_200cm", "irradiance_solar_200cm"]
HEAVY = ("alignment_", "image_inner_plot_transforms")


def split_files(root, split):
    return sorted(glob.glob(os.path.join(root, "data", f"{split}-*.parquet")))


def columns(root, heavy=False, markers=False, weather=False):
    cols = pq.read_schema(split_files(root, "train")[0]).names
    keep = []
    for c in cols:
        if not heavy and c.startswith(HEAVY):
            continue
        if not markers and c.startswith("marker_"):
            continue
        if not weather and any(c.startswith(w) for w in WEATHER):
            continue
        keep.append(c)
    return keep


def load(root="data/fip1", splits=SPLITS, **kw):
    cols = columns(root, **kw)
    out = []
    for s in splits:
        for f in split_files(root, s):
            d = pq.read_table(f, columns=cols).to_pandas()
            d["split"] = s
            out.append(d)
    return pd.concat(out, ignore_index=True)
