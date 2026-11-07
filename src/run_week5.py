"""
Week-5 runner (docs/07-weather-upper-bound-design.md): how much would knowing the coming temperature help?

Every "-true" model uses the ACTUAL degree days between the issue date (or cutoff) and the target. It breaks the input
rule on purpose: an upper bound, never a forecast. Leave-one-year-out only; the official test splits are not used.

  python src/run_week5.py unit YEAR      # wheat fold holding out YEAR: N3, N3-true, G, G-true; L1, L1-true, F0 heading
  python src/run_week5.py soy            # soybean leave-one-season-out: N3, N3-true (no learned model)
  python src/run_week5.py score          # once: comparisons, the rule, the field-wide split
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

import final2 as FL
import near as N
import near2 as N2
import run_week2 as R
import soy as SOY
from common import bootstrap_mae

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = "outputs/week5"
UNITS = f"{OUT}/units"
YEARS = [2016, 2017, 2018, 2019, 2021, 2022]
CUTOFFS = [120, 150, 180]
PAIRS = [("N3-true", "N3"), ("G-true", "G"), ("G-true", "N3")]
RULE_GAIN, RULE_YEARS = 0.10, 5


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def near_fold(D, settings, y, future):
    train = list(D.plots.plot_uid[D.plots.year != y])
    years = sorted(D.plots.year[D.plots.year != y].unique())
    feat = N2.near_features(D.samples, D.series, D.weather, D.w2, D.plots, D.markers, train, years, D.alphas,
                            future=future)
    feat["G"] = np.nan
    for t, cfg in settings["G"].items():
        m = feat.trait == t
        feat.loc[m, "G"] = N2.fit_g(feat[m], train, cfg, N2.FEATS_G)
    return feat[feat.year == y]


def heading_fold(D, y):
    train = list(D.plots.plot_uid[D.plots.year != y])
    years = sorted(D.plots.year[D.plots.year != y].unique())
    P = D.plots.set_index("plot_uid")
    trm = P.index.isin(set(train)) & P.heading_das.notna().values
    f0 = P[trm].groupby("year").heading_das.mean().mean()
    held = P[(P.year == y) & P.heading_das.notna()]
    rows = []
    for c in CUTOFFS:
        l1 = FL.heading_gdd_level(D.plots, D.gdd, D.weather, train, years, c)[0]
        l1t = FL.heading_gdd_level(D.plots, D.gdd, D.weather, train, years, c, future="true")[0]
        rows.append(pd.DataFrame({"plot_uid": held.index, "genotype_id": held.genotype_id.values, "year": y,
                                  "cutoff": c, "target": held.heading_das.values.astype(float), "F0": f0,
                                  "L1": l1.reindex(held.index).values, "L1-true": l1t.reindex(held.index).values}))
    return pd.concat(rows, ignore_index=True)


def unit(y):
    os.makedirs(UNITS, exist_ok=True)
    f = f"{UNITS}/{y}_near.parquet"
    if os.path.exists(f):
        log(y, "already done")
        return
    D = R.Data()
    settings = json.load(open("outputs/week2/settings.json"))
    keys = ["plot_uid", "genotype_id", "year", "trait", "h", "phase", "issue_das", "target"]
    a = near_fold(D, settings, y, "clim")
    log(y, "clim done", len(a))
    # check (docs/07): with the switch off, N3 and G are week 3's for this fold
    w3 = pd.read_parquet(f"outputs/week3/units/{y}_none_near.parquet")
    k = ["plot_uid", "trait", "h", "issue_das"]
    m = w3.merge(a[k + ["N3", "G"]], on=k, suffixes=("", "_w5"))
    assert len(m) == len(w3) == len(a), (len(m), len(w3), len(a))
    for c in ["N3", "G"]:
        diff = float(np.max(np.abs(m[c] - m[f"{c}_w5"])))
        log(y, c, "max difference from week 3:", diff)
        assert diff < 1e-6, f"{c} does not reproduce week 3"
    b = near_fold(D, settings, y, "true")
    log(y, "true done", len(b))
    assert (a.index == b.index).all()
    out = a[keys + ["N3", "G"]].copy()
    out["N3-true"], out["G-true"] = b.N3.values, b.G.values
    heading_fold(D, y).to_parquet(f"{UNITS}/{y}_heading.parquet", index=False)
    out.to_parquet(f, index=False)
    log(y, "saved", len(out))


def soy():
    f = f"{OUT}/soy_pred.parquet"
    if os.path.exists(f):
        log("soybean already done")
        return
    alphas = json.load(open("outputs/week1/official_alphas.json"))
    plots, series, weather, samples = SOY.load()
    parts = []
    for y in SOY.SEASONS:
        train = list(plots.plot_uid[plots.year != y])
        years = sorted(plots.year[plots.year != y].unique())
        r = {}
        with SOY.season():
            for fut in ["clim", "true"]:
                p, _ = N.fit_predict(samples, series, weather, train, [], years, alphas["near"], future=fut)
                r[fut] = p[p.year == y]
        o = r["clim"][["plot_uid", "genotype_id", "year", "trait", "h", "issue_das", "target", "N3"]].copy()
        o["N3-true"] = r["true"].N3.values
        parts.append(o)
        log("soybean", y, len(o))
    pd.concat(parts, ignore_index=True).to_parquet(f, index=False)


# ---------------------------------------------------------------- scoring

def compare(p, pairs, by, scale, years_rule=None, gain_rule=None):
    """Paired MAE differences per held-out year and pooled. Rule (pooled rows): interval below 0, at least `years_rule`
    years better and, if gain_rule, a relative gain of at least gain_rule."""
    rows = []
    for key, g in p.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        for m, ref in pairs:
            per_year = {}
            for name, gg in [(f"year_{y}", x) for y, x in g.groupby("year")] + [("pooled", g)]:
                e, e0 = (gg[m] - gg.target).values * scale, (gg[ref] - gg.target).values * scale
                d, lo, hi = bootstrap_mae(e, gg.genotype_id.values, other=e0)
                if name != "pooled":
                    per_year[name] = d
                rows.append(dict(zip(by, key), set=name, method=m, ref=ref, n=len(gg), mae=np.mean(np.abs(e)),
                                 mae_ref=np.mean(np.abs(e0)), d=d, d_lo=lo, d_hi=hi, rel=d / np.mean(np.abs(e0))))
            r = rows[-1]
            r["years_better"] = int(sum(v < 0 for v in per_year.values()))
            r["years"] = len(per_year)
            if years_rule is not None:
                r["holds"] = bool(r["d_hi"] < 0 and r["years_better"] >= years_rule and
                                  (gain_rule is None or -r["rel"] >= gain_rule))
    return pd.DataFrame(rows)


def field_split(p, models):
    """docs/04 A2: each error = its mean over the field on that issue date (field-wide) + the rest (plot-specific)."""
    key = ["year", "trait", "h", "issue_das"]
    rows = []
    for m in models:
        e = (p[m] - p.target) * 100.0  # cover points and cm
        c = e.groupby([p[k] for k in key]).transform("mean")
        for (y, t, h), idx in p.groupby(["year", "trait", "h"]).groups.items():
            rows.append(dict(year=y, trait=t, h=h, method=m, total=np.mean(np.abs(e[idx])),
                             field=np.mean(np.abs(c[idx])), plot=np.mean(np.abs((e - c)[idx]))))
    return pd.DataFrame(rows)


def score():
    if os.path.exists(f"{OUT}/near.csv"):
        raise SystemExit("week 5 already scored; it is run once (docs/07)")
    missing = [y for y in YEARS if not os.path.exists(f"{UNITS}/{y}_near.parquet")]
    if missing or not os.path.exists(f"{OUT}/soy_pred.parquet"):
        raise SystemExit(f"missing units {missing} or soybean predictions")
    p = pd.concat([pd.read_parquet(f"{UNITS}/{y}_near.parquet") for y in YEARS], ignore_index=True)
    r = compare(p, PAIRS, ["trait", "h"], 100.0, RULE_YEARS, RULE_GAIN)
    r.loc[~((r.method == "N3-true") & r.h.isin([14, 28])), "holds"] = np.nan  # the rule is for N3-true at 14-28 d
    r.to_csv(f"{OUT}/near.csv", index=False)
    compare(p, PAIRS, ["trait", "h", "phase"], 100.0).query("set == 'pooled'").to_csv(f"{OUT}/near_phase.csv",
                                                                                         index=False)
    field_split(p, ["N3", "N3-true", "G", "G-true"]).to_csv(f"{OUT}/field_split.csv", index=False)

    hd = pd.concat([pd.read_parquet(f"{UNITS}/{y}_heading.parquet") for y in YEARS], ignore_index=True)
    compare(hd, [("L1-true", "F0"), ("L1", "F0"), ("L1-true", "L1")], ["cutoff"], 1.0).to_csv(
        f"{OUT}/heading.csv", index=False)
    s = pd.read_parquet(f"{OUT}/soy_pred.parquet")
    compare(s, [("N3-true", "N3")], ["h"], 100.0, 5).to_csv(f"{OUT}/soy.csv", index=False)

    rule = r[(r.set == "pooled") & r.holds.notna()]
    print(rule[["trait", "h", "mae", "mae_ref", "d", "d_lo", "d_hi", "rel", "years_better", "holds"]].round(3)
          .to_string())
    go = bool(rule[rule.h == 28].holds.astype(bool).any())
    log("decision:", "fetch archived weather forecasts" if go else "temperature does not explain the shift")
    print(r[r.set == "pooled"].round(3).to_string())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "unit":
        unit(int(sys.argv[2]))
    else:
        {"soy": soy, "score": score}[cmd]()
