"""
Week-6 data (docs/08-generation-design.md): photo metadata, thumbnails, training pairs, conditions, queries.

No torch here, so the leakage tests of the data rules run anywhere. The big arrays live in data/gen (not in git):
  latents_<year>.npy   (n, 4, 20, 30) float16  frozen-VAE latents of the thumbnails of that year, rows in sorted path order
  dino_<year>.npy      (n, 1536) float32       DINOv2 features of the same thumbnails (CLS | mean patch), same rows
  paths_<year>.parquet the path of each row
"""
import glob
import io
import os
import zlib

import numpy as np
import pandas as pd

TAG = os.environ.get("WEEK6_TAG", "")  # "smoke": separate directories, never used for results
GEN = "data/gen"
OUT = "outputs/week6" + (f"/{TAG}" if TAG else "")
RUN = GEN + (f"/{TAG}" if TAG else "")  # checkpoints and generations
THUMBS = "data/photos/thumbs_{year}.parquet"
INDEX = "data/derived/photo_index.parquet"
W3_FEATS = "data/photos/features_*.parquet"
QUERIES = "outputs/week4/retrieval_queries.parquet"
YEARS = [2016, 2017, 2018, 2019, 2021, 2022]
W, H = 240, 160
PAIR_MIN, PAIR_MAX = 7, 35  # days between the input photo and the target photo of a training pair
COND_DIM = 5
MEAS_TOL = 1  # a photo's measured cover / height: the nearest measurement within this many days
HEIGHT_FC_TOL = 3  # a height forecast made at most this many days before the query's issue day
VARIANTS = ["Gen", "Gen-persist", "Gen-oracle"]


# ---------------------------------------------------------------- photos

def photo_meta(plots, index_path=INDEX):
    """Every inner-plot photo (151,150): plot, year, path, days after sowing; sorted by (year, path)."""
    idx = pd.read_parquet(index_path)
    sow = plots.set_index("plot_uid").sowing_date
    idx = idx.assign(das=(pd.to_datetime(idx.date) - idx.plot_uid.map(sow)).dt.days.astype(np.int64))
    return idx.sort_values(["year", "path"]).reset_index(drop=True)[["plot_uid", "year", "path", "das"]]


def week3_paths():
    """Photo path of each row of week 3's feature tables (the row numbers stored in week 4's queries)."""
    fs = sorted(glob.glob(W3_FEATS))
    return pd.concat([pd.read_parquet(f, columns=["path"]) for f in fs], ignore_index=True).path.values


def load_thumbs(year, paths=None):
    """{path: jpeg bytes} for a year (or only the given paths)."""
    t = pd.read_parquet(THUMBS.format(year=year))
    if paths is not None:
        t = t[t.path.isin(set(paths))]
    return dict(zip(t.path, t.jpeg))


def decode(jpeg):
    from PIL import Image
    return np.array(Image.open(io.BytesIO(jpeg)).convert("RGB"), np.uint8)  # (160, 240, 3)


# ---------------------------------------------------------------- measurements

def trait_lookup(series, trait):
    s = series[series.trait == trait].sort_values(["plot_uid", "das"])
    return {u: (g.das.values.astype(np.int64), g.value.values.astype(np.float32)) for u, g in s.groupby("plot_uid")}


def nearest(lk, uid, t, tol):
    """The plot's measurement nearest day t (ties: earlier) within tol days, else NaN."""
    if uid not in lk:
        return np.nan
    d, v = lk[uid]
    i = int(np.argmin(np.abs(d - t) * 10 + (d > t)))
    return float(v[i]) if abs(int(d[i]) - t) <= tol else np.nan


def latest(lk, uid, t):
    """The plot's latest measurement on or before day t, else NaN."""
    if uid not in lk:
        return np.nan
    d, v = lk[uid]
    k = int(np.searchsorted(d, t, side="right")) - 1
    return float(v[k]) if k >= 0 else np.nan


def cond_matrix(cover, height, days_ahead, das_target):
    """Condition vectors [cover, height, height missing, days ahead / 28, days after sowing / 300]. Cover is required."""
    cover, height = np.asarray(cover, float), np.asarray(height, float)
    assert not np.isnan(cover).any(), "cover is required for every condition"
    miss = np.isnan(height)
    return np.stack([np.clip(cover, 0, 1), np.where(miss, 0.0, height), miss.astype(float),
                     np.asarray(days_ahead, float) / 28.0, np.asarray(das_target, float) / 300.0], 1).astype(np.float32)


# ---------------------------------------------------------------- folds and training pairs

def is_monitor(uid):
    """10% of plots (fixed by a hash of the id) are used only to watch the loss, never to train or choose anything."""
    return zlib.crc32(uid.encode()) % 10 == 0


def fold_split(meta, fold):
    """Boolean masks over the rows of meta: train, monitor (held-out plots of the training years), held-out year."""
    held = (meta.year == fold).values
    mon = (~held) & meta.plot_uid.map(is_monitor).values
    return ~held & ~mon, mon, held


def estimator_rows(meta, cover, fold):
    """Photos the fold's cover estimator is trained on: the training years' photos with a measured cover."""
    return (meta.year != fold).values & ~np.isnan(cover)


def enumerate_pairs(plot_ids, das, lo=PAIR_MIN, hi=PAIR_MAX):
    """All (a, b): the same plot, das[b] - das[a] between lo and hi days. Row indices into the given arrays."""
    plot_ids, das = np.asarray(plot_ids), np.asarray(das)
    order = np.lexsort((das, plot_ids))
    p, d = plot_ids[order], das[order]
    cut = np.flatnonzero(np.r_[True, p[1:] != p[:-1], True])
    A, B = [], []
    for s, e in zip(cut[:-1], cut[1:]):
        dd = d[s:e]
        j0, j1 = np.searchsorted(dd, dd + lo, "left"), np.searchsorted(dd, dd + hi, "right")
        for i in np.flatnonzero(j1 > j0):
            j = np.arange(j0[i], j1[i])
            A.append(np.full(len(j), s + i))
            B.append(s + j)
    if not A:
        return np.zeros(0, int), np.zeros(0, int)
    return order[np.concatenate(A)], order[np.concatenate(B)]


class PairData:
    """Training pairs of one fold: latents of the photos, and for each photo b its measured cover and height."""

    def __init__(self, lat, das, cover, height, a, b):
        keep = ~np.isnan(cover[b])  # a pair needs a measured cover at its target photo
        self.lat, self.das, self.cover, self.height = lat, das, cover, height
        self.a, self.b = a[keep], b[keep]

    def __len__(self):
        return len(self.a)

    def batch(self, rng, n):
        i = rng.integers(0, len(self.a), n)
        a, b = self.a[i], self.b[i]
        c = cond_matrix(self.cover[b], self.height[b], self.das[b] - self.das[a], self.das[b])
        return self.lat[a].astype(np.float32), self.lat[b].astype(np.float32), c


def photo_measurements(meta, cover_lk, height_lk):
    """Measured cover and height at each photo (nearest within MEAS_TOL days; NaN if none)."""
    cov = np.array([nearest(cover_lk, u, t, MEAS_TOL) for u, t in zip(meta.plot_uid, meta.das)])
    hei = np.array([nearest(height_lk, u, t, MEAS_TOL) for u, t in zip(meta.plot_uid, meta.das)])
    return cov, hei


# ---------------------------------------------------------------- queries (week 4's, unchanged)

def load_queries(series):
    """Week 4's 8,113 queries with the photo paths and the condition vectors of each variant.

    Gen          cover = the E+P forecast, height = the E forecast of the latest height sample issued on or before the
                 query's issue day (within HEIGHT_FC_TOL days); missing if none
    Gen-persist  the latest measured cover and height on or before the issue day
    Gen-oracle   the true cover at the target and the height measurement nearest the target day (uses the target)
    """
    q = pd.read_parquet(QUERIES)
    paths = week3_paths()
    for col, k in [("now_path", "R-now"), ("future_path", "future"), ("ret_path", "R-E+P")]:
        q[col] = np.where(q[k].notna(), paths[q[k].fillna(0).astype(int).values], None)
    q = q.sort_values("qid").reset_index(drop=True)
    cover_lk, height_lk = trait_lookup(series, "canopy_cover"), trait_lookup(series, "height")
    q["height_fc"] = height_forecast(q)
    days = q.target_das - q.now_das
    q["days_ahead"] = days
    c_now = np.array([latest(cover_lk, u, t) for u, t in zip(q.plot_uid, q.issue_das)])
    h_now = np.array([latest(height_lk, u, t) for u, t in zip(q.plot_uid, q.issue_das)])
    h_true = np.array([nearest(height_lk, u, t, 3) for u, t in zip(q.plot_uid, q.target_das)])
    cond = {"Gen": cond_matrix(q["E+P"], q.height_fc, days, q.target_das),
            "Gen-persist": cond_matrix(c_now, h_now, days, q.target_das),
            "Gen-oracle": cond_matrix(q.target, h_true, days, q.target_das)}
    return q, cond


def height_forecast(q, root="outputs/week3/units", with_issue=False):
    """Week 3's E forecast of height for the query's horizon, from the latest height sample issued on or before the
    query's issue day and at most HEIGHT_FC_TOL days before it. Never a sample issued later."""
    parts = []
    for y in sorted(q.year.unique()):
        u = pd.read_parquet(f"{root}/{y}_none_near.parquet")
        parts.append(u[u.trait == "height"][["plot_uid", "h", "issue_das", "E"]])
    hf = pd.concat(parts).rename(columns={"issue_das": "hf_issue"}).sort_values("hf_issue")
    left = q[["qid", "plot_uid", "h", "issue_das"]].sort_values("issue_das")
    m = pd.merge_asof(left, hf, left_on="issue_das", right_on="hf_issue", by=["plot_uid", "h"], direction="backward",
                      tolerance=HEIGHT_FC_TOL)
    m = q[["qid"]].merge(m[["qid", "E", "hf_issue"]], on="qid", how="left")
    return m if with_issue else m.E.values


# ---------------------------------------------------------------- distance

def standardiser(F, years, fold):
    """Mean and std of the features of the photos of the fold's training years (never the held-out year)."""
    m = np.asarray(years) != fold
    return F[m].mean(0), F[m].std(0) + 1e-6


def cosine_distance(A, B):
    A, B = A / np.linalg.norm(A, axis=1, keepdims=True), B / np.linalg.norm(B, axis=1, keepdims=True)
    return 1 - np.sum(A * B, axis=1)
