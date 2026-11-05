"""
Tidy tables built once from the FIP 1.0 parquets and cached in data/derived/.

plots.parquet   one row per plot-year: ids, split, sowing date, final outcomes (float32), heading in days after sowing.
series.parquet  one row per plot-year x trait x date: das (days after sowing), gdd (our degree days), value (float32).
                Same-date duplicates are averaged. Canopy cover stays a 0-1 fraction.
weather.parquet one row per year x day: daily mean air temperature at 2 m (float32).

Degree days (our definition, used everywhere): base 0 degC, daily mean of hourly air temperature at 2 m, summed from the
sowing day. It does not reproduce the dataset's heading values in degC d (see docs/01-plan.md).

Usage:  python src/series.py --root data/fip1 --out data/derived
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from load_fip import FINALS, SERIES, load

NEAR = ["canopy_cover", "height"]


def year_weather(root):
    """Daily mean 2 m air temperature per harvest year, read from one row per year (weather is identical within a year)."""
    cols = ["harvest_year", "temperature_air_200cm_values", "temperature_air_200cm_dates"]
    got = {}
    for f in sorted(glob.glob(os.path.join(root, "data", "*.parquet"))):
        pf = pq.ParquetFile(f)
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i, columns=cols).to_pandas()
            for y, g in t.groupby("harvest_year"):
                got.setdefault(int(y), g.iloc[0])
            del t
    rows = []
    for y, r in sorted(got.items()):
        t = pd.DataFrame({"time": pd.to_datetime(pd.Series(r.temperature_air_200cm_dates)),
                          "t": np.asarray(r.temperature_air_200cm_values, np.float32)})
        dm = t.groupby(t.time.dt.normalize()).t.mean()
        rows.append(pd.DataFrame({"year": y, "date": dm.index, "tmean": dm.values.astype(np.float32)}))
    return pd.concat(rows, ignore_index=True)


def gdd_table(weather, sow_by_year):
    """Cumulative degree days from sowing, per year and date (0 on the sowing day's start; includes the sowing day)."""
    out = []
    for y, w in weather.groupby("year"):
        w = w.sort_values("date")
        w = w[w.date >= sow_by_year[y]]
        out.append(pd.DataFrame({"year": y, "date": w.date.values, "gdd": np.cumsum(np.clip(w.tmean.values, 0, None))}))
    return pd.concat(out, ignore_index=True)


def build(root, out):
    os.makedirs(out, exist_ok=True)
    d = load(root)
    d["sowing_date"] = pd.to_datetime(d.sowing_date).dt.normalize()
    sow = d.groupby("harvest_year").sowing_date.first().to_dict()
    assert d.groupby("harvest_year").sowing_date.nunique().max() == 1, "one sowing date per year expected"

    plots = d[["plot_uid", "harvest_year", "genotype_id", "split", "sowing_date", "range", "row"]].rename(
        columns={"harvest_year": "year"}).copy()
    for t in FINALS:
        plots[t] = d[f"{t}_value"].astype(np.float32)
    hd = pd.to_datetime(d.heading_date).dt.normalize()
    plots["heading_das"] = (hd - d.sowing_date).dt.days.astype("float32")
    assert plots.plot_uid.is_unique

    rows = []
    for t in SERIES:
        for uid, sd, dates, vals in zip(d.plot_uid, d.sowing_date, d[f"{t}_dates"], d[f"{t}_values"]):
            if dates is None or len(dates) == 0:
                continue
            dt = pd.to_datetime(pd.Series(dates)).dt.normalize()
            rows.append(pd.DataFrame({"plot_uid": uid, "trait": t, "date": dt.values,
                                      "das": (dt - sd).dt.days.values, "value": np.asarray(vals, np.float32)}))
    s = pd.concat(rows, ignore_index=True)
    n0 = len(s)
    s = s.groupby(["plot_uid", "trait", "date", "das"], as_index=False).value.mean()
    s = s.merge(plots[["plot_uid", "year"]], on="plot_uid")

    w = year_weather(root)
    g = gdd_table(w, sow)
    s = s.merge(g, on=["year", "date"], how="left")
    print(f"series rows {n0} -> {len(s)} after averaging same-date duplicates; missing gdd {s.gdd.isna().sum()}")

    plots.to_parquet(os.path.join(out, "plots.parquet"), index=False)
    s.to_parquet(os.path.join(out, "series.parquet"), index=False)
    w.to_parquet(os.path.join(out, "weather.parquet"), index=False)
    return plots, s, w


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/fip1")
    ap.add_argument("--out", default="data/derived")
    a = ap.parse_args()
    build(a.root, a.out)
