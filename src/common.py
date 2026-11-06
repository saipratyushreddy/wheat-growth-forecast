"""
Shared pieces for the week-1 baselines (docs/02-baselines-design.md): derived tables, climatological future degree days,
equal-weight-per-year average curves, markers, and the variety-resampling bootstrap.
"""
import contextlib
import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DERIVED = "data/derived"
OFFICIAL_TEST = ["test_plot", "test_genotype", "test_environment", "test_genotype_environment"]


def load_derived(root=DERIVED):
    plots = pd.read_parquet(os.path.join(root, "plots.parquet"))
    series = pd.read_parquet(os.path.join(root, "series.parquet"))
    weather = pd.read_parquet(os.path.join(root, "weather.parquet"))
    return plots, series, weather


# Start of the season-day count: wheat (default) counts from 1 September before harvest; soybean (week 4) from 1 January
# of the sowing year, set only inside `with season_start(1, 0):`.
_SEASON = {"month": 9, "offset": -1}


@contextlib.contextmanager
def season_start(month, offset):
    old = dict(_SEASON)
    _SEASON.update(month=month, offset=offset)
    try:
        yield
    finally:
        _SEASON.update(old)


def season_day(dates, years):
    """Day index within the weather year (wheat: 0 = 1 September before harvest), comparable across years."""
    start = pd.to_datetime([f"{int(y) + _SEASON['offset']}-{_SEASON['month']:02d}-01" for y in np.asarray(years)])
    return np.asarray((pd.DatetimeIndex(np.asarray(dates)) - start).days)


class ClimTemp:
    """Training years' average daily mean temperature per season day; gives climatological degree-day increments."""

    def __init__(self, weather, years):
        w = weather[weather.year.isin(list(years))].copy()
        w["sday"] = season_day(w.date, w.year)
        t = w.groupby("sday").tmean.mean().reindex(range(366)).interpolate(limit_direction="both")
        self.cum = np.concatenate([[0.0], np.cumsum(np.clip(t.values, 0, None))])  # cum[k] = sum over days < k

    def increment(self, sday_from, sday_to):
        """Degree days accumulated after day sday_from up to and including day sday_to."""
        a = np.clip(np.asarray(sday_from) + 1, 0, 366)
        b = np.clip(np.asarray(sday_to) + 1, 0, 366)
        return self.cum[b] - self.cum[a]


class Curve:
    """Average curve of one trait on one axis ('das' or 'gdd'): per-year bin means, interpolated, equal weight per year."""

    def __init__(self, s, axis, width, lo=None, hi=None):
        self.axis, self.width = axis, width
        lo = s[axis].min() if lo is None else lo
        hi = s[axis].max() if hi is None else hi
        self.grid = np.arange(np.floor(lo / width) * width, hi + width, width) + width / 2
        per_year = []
        for _, g in s.groupby("year"):
            b = g.groupby(np.floor(g[axis].values / width)).value.mean()
            x, y = b.index.values * width + width / 2, b.values
            v = np.interp(self.grid, x, y, left=np.nan, right=np.nan)
            per_year.append(v)
        self.values = np.nanmean(np.vstack(per_year), axis=0) if per_year else np.full(len(self.grid), np.nan)
        ok = ~np.isnan(self.values)
        self.grid, self.values = self.grid[ok], self.values[ok]

    def __call__(self, x):
        x = np.asarray(x, float)
        if len(self.grid) == 0:
            return np.full(x.shape, np.nan)
        return np.interp(x, self.grid, self.values)  # flat beyond the ends


def load_markers(root="data/fip1"):
    """Genotype id -> marker vector (0/1/2, float32), read one row per genotype."""
    got = {}
    for f in sorted(glob.glob(os.path.join(root, "data", "*.parquet"))):
        pf = pq.ParquetFile(f)
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i, columns=["genotype_id", "marker_biallelic_codes"]).to_pandas()
            for gid, m in zip(t.genotype_id, t.marker_biallelic_codes):
                if m is not None and len(m) and gid not in got:
                    got[gid] = np.asarray(m, np.float32)
            del t
    ids = sorted(got)
    return ids, np.vstack([got[i] for i in ids])


def bootstrap_mae(err, groups, n=2000, seed=0, other=None):
    """MAE and 95% interval, resampling groups (varieties) with replacement. With `other` (errors of a second method on
    the same samples), returns the paired difference MAE(err) - MAE(other) instead."""
    err = np.abs(np.asarray(err, float))
    codes, g = np.unique(np.asarray(groups), return_inverse=True)
    G = len(codes)
    s1 = np.bincount(g, err, G)
    cnt = np.bincount(g, minlength=G).astype(float)
    s2 = np.bincount(g, np.abs(np.asarray(other, float)), G) if other is not None else None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, G, size=(n, G))
    w = np.stack([np.bincount(r, minlength=G) for r in draws]).astype(float)
    m1 = (w @ s1) / (w @ cnt)
    point = s1.sum() / cnt.sum()
    if s2 is not None:
        m1 = m1 - (w @ s2) / (w @ cnt)
        point = point - s2.sum() / cnt.sum()
    lo, hi = np.percentile(m1, [2.5, 97.5])
    return float(point), float(lo), float(hi)
