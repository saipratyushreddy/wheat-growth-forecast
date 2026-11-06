"""
Week-4 runner (docs/05-week4-design.md).

Part A, soybean replication of the within-field correction (leave-one-season-out, settings frozen from wheat):
  python src/run_week4.py soy_unit K       # fit the fold holding out season SEASONS[K]; N3, G, S, E for that season
  python src/run_week4.py soy_score        # once: H1 (E-w vs N3) and the other comparisons

Part B, future-photo retrieval on wheat (leave-one-year-out, week-3 predictions reused, nothing refit):
  python src/run_week4.py retrieval        # once: queries, retrieved photos, distances, comparisons, example figure
  python src/run_week4.py retrieval_examples   # redraw the example figure from the saved queries
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

import features2 as F2
import near2 as N2
import seq as S
import soy as SOY
from common import bootstrap_mae

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = "outputs/week4"
SOY_UNITS = f"{OUT}/soy_units"
FEATS_SOY = [c for c in N2.HISTORY if c not in ("o_a", "o_age")] + N2.TIMING + ["var_res", "var_res_n", "var_dev_y"]
SOY_STATIC = ["var_dev_y"]
TRAIT = "canopy_cover"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- part A: soybean

def soy_features(samples, series, weather, plots, train_plots, train_years, alphas_near):
    """Wheat's history and timing inputs (canopy cover only) + soybean genetics from other seasons' training plots."""
    with SOY.season():
        pred = N2.history_features(samples, series, weather, plots, train_plots, train_years, alphas_near,
                                   traits=[TRAIT], other=False)
    train_plots = set(train_plots)
    tr = pred[pred.plot_uid.isin(train_plots)]
    per_plot = tr.groupby(["plot_uid", "trait", "h"], as_index=False).agg(val=("res", "mean"), genotype_id=(
        "genotype_id", "first"), year=("year", "first"))
    vr = F2.other_year_mean(per_plot, pred, by=("trait", "h"))
    pred["var_res"], pred["var_res_n"] = vr["mean"].values, vr["n"].values
    P = plots.set_index("plot_uid", drop=False)
    dev = F2.year_deviation(plots, "yield", train_plots)
    pred["var_dev_y"] = pred.plot_uid.map(F2.other_year_mean(dev, P)["mean"]).values
    pred["genotyped"] = 0.0  # no markers for soybean
    return pred


def within(df, m, ref="N3"):
    """The within-field version of model m (docs/04 part A, docs/05): subtract, on each issue date, the mean over the
    field's plots of m's correction to N3. Uses predictions only, no targets. Clipped to [0, 1]."""
    corr = df[m] - df[ref]
    mean = corr.groupby([df.year, df.h, df.issue_das]).transform("mean")
    return np.clip(df[m] - mean, 0.0, 1.0)


def soy_unit(k):
    os.makedirs(SOY_UNITS, exist_ok=True)
    y = SOY.SEASONS[k]
    f = f"{SOY_UNITS}/{y}.parquet"
    if os.path.exists(f):
        log(y, "already done")
        return
    settings = json.load(open("outputs/week2/settings.json"))
    alphas = json.load(open("outputs/week1/official_alphas.json"))
    plots, series, weather, samples = SOY.load()
    train = list(plots.plot_uid[plots.year != y])
    years = sorted(plots.year[plots.year != y].unique())
    feat = soy_features(samples, series, weather, plots, train, years, alphas["near"])
    log(y, "features", len(feat))
    feat["G"] = N2.fit_g(feat, train, settings["G"][TRAIT], FEATS_SOY)
    sd = S.SeqData(feat, series, train, static=SOY_STATIC, traits=[TRAIT], missing_col="var_dev_y")
    e = settings["S"][TRAIT]
    feat["S"] = S.to_forecast(feat, S.train_predict(sd, train, epochs=[e])[e])
    feat["E"] = (feat.G + feat.S) / 2
    out = feat[feat.year == y][["plot_uid", "genotype_id", "year", "h", "issue_das", "target", "N3", "G", "S", "E"]]
    out.to_parquet(f, index=False)
    log(y, "saved", len(out))


SOY_PAIRS = [("E-w", "N3"), ("E", "N3"), ("G", "N3"), ("G-w", "N3"), ("E-w", "E")]


def compare(p, pairs, years_rule):
    """Paired MAE differences (points of cover) per held-out season and pooled; years better; the rule for pooled rows."""
    rows = []
    for h, g in p.groupby("h"):
        for m, ref in pairs:
            per_year = {}
            for name, gg in [(f"season_{y}", x) for y, x in g.groupby("year")] + [("pooled", g)]:
                e, e0 = (gg[m] - gg.target).values * 100, (gg[ref] - gg.target).values * 100
                d, lo, hi = bootstrap_mae(e, gg.genotype_id.values, other=e0)
                if name != "pooled":
                    per_year[name] = d
                rows.append(dict(set=name, h=h, method=m, ref=ref, n=len(gg), mae=np.mean(np.abs(e)),
                                 mae_ref=np.mean(np.abs(e0)), d=d, d_lo=lo, d_hi=hi))
            r = rows[-1]
            r["years_better"] = int(sum(v < 0 for v in per_year.values()))
            r["years"] = len(per_year)
            r["holds"] = bool(r["d_hi"] < 0 and r["years_better"] >= years_rule)
    return pd.DataFrame(rows)


def soy_score():
    if os.path.exists(f"{OUT}/soy_near.csv"):
        raise SystemExit("soybean already scored; it is run once (docs/05-week4-design.md)")
    missing = [y for y in SOY.SEASONS if not os.path.exists(f"{SOY_UNITS}/{y}.parquet")]
    if missing:
        raise SystemExit(f"soybean units missing: {missing}")
    p = pd.concat([pd.read_parquet(f"{SOY_UNITS}/{y}.parquet") for y in SOY.SEASONS], ignore_index=True)
    p["E-w"], p["G-w"] = within(p, "E"), within(p, "G")
    p.to_parquet(f"{OUT}/soy_pred.parquet", index=False)
    r = compare(p, SOY_PAIRS, years_rule=5)
    r.to_csv(f"{OUT}/soy_near.csv", index=False)
    h1 = r[(r.set == "pooled") & (r.method == "E-w") & (r.ref == "N3")]
    n = int(h1.holds.sum())
    verdict = "replicates" if n == 3 else ("partly" if n else "does not replicate")
    log("H1 holds at", n, "of 3 horizons:", verdict)
    print(r[r.set == "pooled"].round(3).to_string())


# ---------------------------------------------------------------- part B: retrieval

R_METHODS = ["R-now", "R-var", "R-N3", "R-E+P", "R-oracle"]
R_PAIRS = [("R-E+P", "R-now"), ("R-E+P", "R-N3"), ("R-N3", "R-now"), ("R-var", "R-now"), ("R-oracle", "R-now")]
WIN_DAS, MEAS_TOL, FUT_TOL = 7, 1, 3


def candidates(meta, series):
    """Photos paired with the same plot's canopy-cover measurement within MEAS_TOL days (nearest; ties -> earlier)."""
    cc = series[series.trait == TRAIT][["plot_uid", "das", "value"]].rename(columns={"das": "m_das"})
    ph = meta[["plot_uid", "year", "genotype_id", "das", "path", "row"]].sort_values("das")
    m = pd.merge_asof(ph, cc.sort_values("m_das").assign(m_das=lambda x: x.m_das.astype(np.int64)),
                      left_on="das", right_on="m_das", by="plot_uid", direction="nearest", tolerance=MEAS_TOL)
    return m.dropna(subset=["value"]).rename(columns={"value": "cover"}).reset_index(drop=True)


def _pick(c, key, value, target_das):
    """Row of c minimising |key - value|, then |das - target_das|, then path."""
    o = np.lexsort((c.path.values, np.abs(c.das.values - target_das), np.abs(c[key].values - value)))
    return c.iloc[o[0]]


def retrieve(queries, cand, own):
    """For each query: the retrieved photo row index per method. `own` = the query plots' own photos (for R-now and the
    true future photo). Candidates from the query's own year are never used."""
    out = []
    own_by = {u: g.sort_values(["das", "path"]) for u, g in own.groupby("plot_uid")}
    for (y, tdas), qs in queries.groupby(["year", "target_das"]):
        c = cand[(cand.year != y) & (np.abs(cand.das - tdas) <= WIN_DAS)]
        for _, q in qs.iterrows():
            o = own_by[q.plot_uid]
            now = o[o.das <= q.issue_das].iloc[-1]
            fut = o.iloc[np.lexsort((o.path.values, (o.das.values > tdas), np.abs(o.das.values - tdas)))[0]]
            r = {"qid": q.qid, "future": fut.row, "R-now": now.row, "now_das": now.das, "fut_das": fut.das}
            if len(c) == 0:
                out.append(r)
                continue
            r["R-N3"] = _pick(c, "cover", q.N3, tdas).row
            r["R-E+P"] = _pick(c, "cover", q["E+P"], tdas).row
            r["R-oracle"] = _pick(c, "cover", q.target, tdas).row
            cv = c[c.genotype_id == q.genotype_id]
            r["var_fallback"] = len(cv) == 0
            r["R-var"] = _pick(c, "cover", q.N3, tdas).row if len(cv) == 0 else _pick(
                cv.assign(z=0.0), "z", 0.0, tdas).row
            out.append(r)
    return pd.DataFrame(out)


def retrieval():
    import photos as PH
    import run_week2 as R
    if os.path.exists(f"{OUT}/retrieval.csv"):
        raise SystemExit("retrieval already scored; it is run once (docs/05-week4-design.md)")
    D = R.Data()
    plots, series = D.plots, D.series
    meta, X = PH.load_features()
    sow = plots.set_index("plot_uid").sowing_date
    meta = meta.assign(das=(pd.to_datetime(meta.date) - meta.plot_uid.map(sow)).dt.days.astype(np.int64),
                       genotype_id=meta.plot_uid.map(plots.set_index("plot_uid").genotype_id).values,
                       row=np.arange(len(meta)))
    cand = candidates(meta, series)
    smp = D.samples[["plot_uid", "trait", "h", "issue_das", "target_das"]]
    rng = np.random.default_rng(0)
    rows = []
    for y in sorted(plots.year.unique()):
        p = pd.read_parquet(f"outputs/week3/units/{y}_none_near.parquet")
        p = p[(p.trait == TRAIT) & p.h.isin([14, 28])].merge(smp, on=["plot_uid", "trait", "h", "issue_das"])
        own = meta[meta.year == y]
        # eligible: the plot has a photo within FUT_TOL days of the target day, and one on or before the issue day
        od = own.groupby("plot_uid").das.apply(np.array)
        ok = [bool(np.any(np.abs(od[u] - t) <= FUT_TOL) and np.any(od[u] <= i)) if u in od.index else False
              for u, t, i in zip(p.plot_uid, p.target_das, p.issue_das)]
        p = p[ok]
        # one query per plot and horizon, issue date drawn at random (seed 0)
        p = p.sort_values(["plot_uid", "h", "issue_das"]).reset_index(drop=True)
        p = p.loc[p.assign(u=rng.random(len(p))).groupby(["plot_uid", "h"]).u.idxmin().values]
        rows.append(p)
        log(y, "queries", len(p))
    q = pd.concat(rows, ignore_index=True)
    q["qid"] = np.arange(len(q))
    own = meta[meta.plot_uid.isin(set(q.plot_uid))]
    r = retrieve(q, cand, own)
    q = q.merge(r, on="qid")
    # distances on features standardised with the training years' photos of each fold
    for y in sorted(q.year.unique()):
        tr = (meta.year != y).values
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        m = q.year == y
        F = (X[q.loc[m, "future"].values.astype(int)] - mu) / sd
        Fn = F / np.linalg.norm(F, axis=1, keepdims=True)
        for k in R_METHODS:
            v = q.loc[m, k]
            ok = v.notna().values
            Z = (X[v[ok].values.astype(int)] - mu) / sd
            dist = np.full(m.sum(), np.nan)
            dist[ok] = 1 - np.sum(Fn[ok] * Z / np.linalg.norm(Z, axis=1, keepdims=True), axis=1)
            q.loc[m, f"d_{k}"] = dist
    q.to_parquet(f"{OUT}/retrieval_queries.parquet", index=False)
    res = []
    for h, g in q.groupby("h"):
        for m, ref in R_PAIRS:
            gg = g.dropna(subset=[f"d_{m}", f"d_{ref}"])
            per = {}
            for name, x in [(f"year_{y}", v) for y, v in gg.groupby("year")] + [("pooled", gg)]:
                dm = x[f"d_{m}"].values - x[f"d_{ref}"].values
                d, lo, hi = _boot_mean(dm, x.genotype_id.values)
                if name != "pooled":
                    per[name] = d
                res.append(dict(set=name, h=h, method=m, ref=ref, n=len(x), dist=x[f"d_{m}"].mean(),
                                dist_ref=x[f"d_{ref}"].mean(), d=d, d_lo=lo, d_hi=hi))
            res[-1].update(years_better=int(sum(v < 0 for v in per.values())), years=len(per),
                           better=bool(res[-1]["d_hi"] < 0 and sum(v < 0 for v in per.values()) >= 4))
    res = pd.DataFrame(res)
    res.to_csv(f"{OUT}/retrieval.csv", index=False)
    log("var fallback rate", float(q.var_fallback.mean()))
    print(res[res.set == "pooled"].round(4).to_string())
    examples(q, meta)


def retrieval_examples():
    """Redraw the example figure from saved queries (no scoring)."""
    import photos as PH
    meta, _ = PH.load_features()
    examples(pd.read_parquet(f"{OUT}/retrieval_queries.parquet"), meta)


def _boot_mean(d, groups, n=2000, seed=0):
    """Mean of paired differences d, 95% interval resampling varieties."""
    codes, g = np.unique(groups, return_inverse=True)
    G = len(codes)
    s, c = np.bincount(g, d, G), np.bincount(g, minlength=G).astype(float)
    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(r, minlength=G) for r in rng.integers(0, G, size=(n, G))]).astype(float)
    lo, hi = np.percentile((w @ s) / (w @ c), [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def examples(q, meta, year=2019, h=28, n=8):
    """Eight random queries (seed 0) from 2019 at 28 days: today's photo, the R-E+P photo, the true future photo."""
    import io

    from PIL import Image, ImageDraw
    sel = q[(q.year == year) & (q.h == h)].sample(n, random_state=0)
    paths = set()
    for k in ["R-now", "R-E+P", "future"]:
        paths |= set(meta.path.values[sel[k].values.astype(int)])
    th = pd.concat([pd.read_parquet(f, filters=[("path", "in", list(paths))]) for f in
                    sorted(__import__("glob").glob("data/photos/thumbs_*.parquet"))]).set_index("path").jpeg
    W, H, top = 240, 160, 18
    img = Image.new("RGB", (3 * W, n * (H + top)), "white")
    dr = ImageDraw.Draw(img)
    for i, (_, r) in enumerate(sel.iterrows()):
        for j, (k, label) in enumerate([("R-now", f"today, day {r.issue_das}"),
                                        ("R-E+P", f"retrieved, E+P {r['E+P'] * 100:.0f}%"),
                                        ("future", f"actual day {r.target_das}, {r.target * 100:.0f}%")]):
            p = meta.path.values[int(r[k])]
            img.paste(Image.open(io.BytesIO(th[p])).convert("RGB"), (j * W, i * (H + top) + top))
            dr.text((j * W + 4, i * (H + top) + 3), label, fill="black")
    img.save(f"{OUT}/retrieval_examples.png")
    sel.to_csv(f"{OUT}/retrieval_examples.csv", index=False)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "soy_unit":
        soy_unit(int(sys.argv[2]))
    elif cmd == "soy_score":
        soy_score()
    elif cmd == "retrieval":
        retrieval()
    elif cmd == "retrieval_examples":
        retrieval_examples()
    else:
        sys.exit(__doc__)
