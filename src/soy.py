"""
Week-4 soybean data (docs/05-week4-design.md, part A): FIP1SOY at a pinned revision -> data/derived_soy/.

plots.parquet    one row per plot-season (the tables list each plot once per weather station; kept once). `year` is the
                 season, named by its sowing year (the table's harvest_year is the sowing year + 1). Seasons 2016-2022.
series.parquet   canopy cover per plot x date: das, gdd (our degree days), value (float32, kept as measured, also > 1).
weather.parquet  one site calendar of daily mean 2 m air temperature, matched by calendar date: the soybean tables
                 where they cover a day, the wheat tables (data/derived/weather.parquet, same station) otherwise.
near_samples.parquet  the wheat rule (src/near.py), canopy cover only.

Season days count from 1 January of the sowing year (common.season_start(1, 0)); every soybean computation runs inside
`with season():`.

  python src/soy.py download    # data/fip1soy at the pinned revision
  python src/soy.py build       # data/derived_soy/*.parquet
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

import common
import near as N
from series import gdd_table

REPO = "mikeboss/FIP1SOY"
REVISION = "47c1f16a973cc1d12a5164b4cf7f20d337fe2e53"
ROOT = "data/fip1soy"
OUT = "data/derived_soy"
SEASONS = list(range(2016, 2023))  # 2015 dropped: no weather covers its growing period
WHEAT_WEATHER = "data/derived/weather.parquet"


def season():
    return common.season_start(1, 0)


def download():
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=ROOT)


def _read(cols):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    ts = []
    for f in sorted(glob.glob(f"{ROOT}/data-*.arrow")):
        with pa.memory_map(f) as src:
            ts.append(ipc.open_stream(src).read_all().select(cols))
    return pa.concat_tables(ts).to_pandas()


def site_weather(wheat_weather):
    """Daily mean temperature by calendar date: soybean tables (hourly, one station), wheat tables where absent."""
    cols = ["weather_location", "temperature_200cm_values", "temperature_200cm_dates", "temperature_200cm_times"]
    d = _read(cols)
    d = d[d.weather_location == "Eschikon-FIP"]
    parts = []
    for v, dt, tm in zip(d.temperature_200cm_values, d.temperature_200cm_dates, d.temperature_200cm_times):
        if v is None or len(v) == 0:
            continue
        parts.append(pd.DataFrame({"time": pd.to_datetime(pd.Series(dt).astype(str) + " " + pd.Series(tm).astype(str)),
                                   "t": np.asarray(v, float)}))
    h = pd.concat(parts).drop_duplicates("time")
    soy = h.groupby(h.time.dt.normalize()).t.mean()  # NaN only if every hour of the day is missing
    wheat = wheat_weather.drop_duplicates("date").set_index(pd.to_datetime(wheat_weather.drop_duplicates("date").date))
    cal = soy.combine_first(wheat.tmean.astype(float)).sort_index()
    return pd.DataFrame({"date": cal.index, "tmean": cal.values.astype(np.float32),
                         "source": np.where(cal.index.isin(soy.dropna().index), "soybean", "wheat")})


def season_weather(cal, seasons):
    """weather table (year = season, date, tmean) over 1 January - 31 December of each season; no gaps allowed."""
    out = []
    for y in seasons:
        days = pd.date_range(f"{y}-01-01", f"{y}-12-31")
        w = cal.set_index("date").reindex(days)
        gaps = int(w.tmean.isna().sum())
        w["tmean"] = w.tmean.interpolate(limit=2, limit_area="inside")  # gaps of up to 2 days (one day, 2020-05-06)
        assert w.tmean.notna().all(), f"weather gap longer than 2 days in season {y}"
        if gaps:
            print(f"season {y}: {gaps} day(s) without temperature, interpolated")
        out.append(pd.DataFrame({"year": y, "date": days, "tmean": w.tmean.values.astype(np.float32)}))
    return pd.concat(out, ignore_index=True)


def build():
    os.makedirs(OUT, exist_ok=True)
    cols = ["plot_uid", "genotype_id", "harvest_year", "sowing_date", "canopy_cover_values", "canopy_cover_dates",
            "yield_values"]
    d = _read(cols)
    d = d[d.harvest_year.notna()].copy()
    d["cc"] = d.canopy_cover_values.apply(lambda v: tuple(np.asarray(v, float)) if v is not None else ())
    # each plot is listed once per weather station; the rows must agree before keeping one
    assert d.groupby("plot_uid").cc.nunique().max() == 1 and d.groupby("plot_uid").genotype_id.nunique().max() == 1
    d = d.drop_duplicates("plot_uid").reset_index(drop=True)
    d["sowing_date"] = pd.to_datetime(d.sowing_date)
    d["year"] = d.sowing_date.dt.year
    assert (d.harvest_year.astype(int) == d.year + 1).all(), "harvest_year expected to be the sowing year + 1"
    assert d.groupby("year").sowing_date.nunique().max() == 1
    d = d[d.year.isin(SEASONS)].reset_index(drop=True)

    plots = pd.DataFrame({"plot_uid": d.plot_uid, "year": d.year, "genotype_id": d.genotype_id, "split": "train",
                          "sowing_date": d.sowing_date,
                          "yield": d.yield_values.apply(lambda v: float(v[0]) if v is not None and len(v) else np.nan
                                                        ).astype(np.float32)})
    rows = []
    for uid, sd, dates, vals in zip(d.plot_uid, d.sowing_date, d.canopy_cover_dates, d.canopy_cover_values):
        if dates is None or len(dates) == 0:
            continue
        dt = pd.to_datetime(pd.Series(dates)).dt.normalize()
        rows.append(pd.DataFrame({"plot_uid": uid, "trait": "canopy_cover", "date": dt.values,
                                  "das": (dt - sd).dt.days.values, "value": np.asarray(vals, np.float32)}))
    s = pd.concat(rows, ignore_index=True)
    s = s.groupby(["plot_uid", "trait", "date", "das"], as_index=False).value.mean()
    s = s.merge(plots[["plot_uid", "year", "genotype_id"]], on="plot_uid")

    cal = site_weather(pd.read_parquet(WHEAT_WEATHER))
    w = season_weather(cal, SEASONS)
    g = gdd_table(w, plots.groupby("year").sowing_date.first().to_dict())
    s = s.merge(g, on=["year", "date"], how="left")
    assert s.gdd.notna().all()
    with season():
        smp = N.build_samples(plots, s)
    plots.to_parquet(f"{OUT}/plots.parquet", index=False)
    s.to_parquet(f"{OUT}/series.parquet", index=False)
    w.to_parquet(f"{OUT}/weather.parquet", index=False)
    cal.to_parquet(f"{OUT}/site_weather.parquet", index=False)
    smp.to_parquet(f"{OUT}/near_samples.parquet", index=False)
    print(len(plots), "plot-seasons,", plots.genotype_id.nunique(), "varieties,", len(s), "measurements,", len(smp),
          "samples; per season:", plots.groupby("year").size().to_dict())
    print("weather days by source in the seasons:",
          cal[cal.date.dt.year.isin(SEASONS)].source.value_counts().to_dict())


def load():
    return (pd.read_parquet(f"{OUT}/plots.parquet"), pd.read_parquet(f"{OUT}/series.parquet"),
            pd.read_parquet(f"{OUT}/weather.parquet"), pd.read_parquet(f"{OUT}/near_samples.parquet"))


if __name__ == "__main__":
    {"download": download, "build": build}[sys.argv[1]]()
