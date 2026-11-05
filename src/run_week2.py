"""
Week-2 runner (docs/03-week2-design.md).

  python src/run_week2.py --stage tune                          # inner leave-one-year-out on train -> settings.json
  python src/run_week2.py --protocol official --eval validation # sanity check, fit on train
  python src/run_week2.py --protocol official --eval test       # official test splits; refuses to run twice
  python src/run_week2.py --protocol loyo                       # headline; after the official test, settings frozen
  python src/run_week2.py --summarize loyo                      # (re)write summaries from saved predictions

Writes outputs/week2/. Week-1 baselines are recomputed in the same fits, with week-1 ridge strengths frozen.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import final as F
import final2 as F2
import near2 as N2
import seq as S
from common import OFFICIAL_TEST, bootstrap_mae, load_derived, load_markers
from series import gdd_table

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = "outputs/week2"
NEAR_METHODS = ["N0", "N2", "N3", "N4", "N5", "G", "GW", "S", "E"]
FINAL_METHODS = ["F0", "F1", "F2", "F3", "L0+D", "L1", "L1+D"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Data:
    def __init__(self):
        self.plots, self.series, self.weather = load_derived()
        self.series = self.series.merge(self.plots[["plot_uid", "genotype_id"]], on="plot_uid")
        self.samples = pd.read_parquet("data/derived/near_samples.parquet")
        self.w2 = pd.read_parquet("data/derived/weather2.parquet")
        self.markers = load_markers()
        self.alphas = json.load(open("outputs/week1/official_alphas.json"))
        self.gdd = gdd_table(self.weather, self.plots.groupby("year").sowing_date.first().to_dict())


# ---------------------------------------------------------------- one fit (any protocol)

def fit_all(D, train, years, settings, samples=None):
    """Fit every week-2 model and the week-1 baselines on `train`; return (near predictions, final predictions)."""
    samples = D.samples if samples is None else samples
    feat = N2.near_features(samples, D.series, D.weather, D.w2, D.plots, D.markers, train, years, D.alphas)
    log("near features", len(feat))
    cfg = {t: settings["G"][t] for t in ["canopy_cover", "height"]}
    feat["G"] = _g_per_trait(feat, train, cfg, N2.FEATS_G)
    log("G done")
    feat["GW"] = _g_per_trait(feat, train, cfg, N2.FEATS_W)
    log("GW done")
    sd = S.SeqData(feat, D.series, train)
    eps = sorted(set(settings["S"].values()))
    res = S.train_predict(sd, train, epochs=eps, log=None)
    feat["S"] = np.nan
    for t, e in settings["S"].items():
        m = feat.trait == t
        feat.loc[m, "S"] = S.to_forecast(feat[m], res[e][m])
    feat["E"] = (feat.G + feat.S) / 2
    log("S done")

    ff = F2.features(D.plots, D.series, D.weather, D.gdd, D.markers, train, D.alphas)
    choice = {tuple(json.loads(k)): [tuple(v)] for k, v in settings["D"].items()}
    pf2 = F2.fit_predict(D.plots, ff, D.weather, D.gdd, train, years, choice)
    feats_w1 = {c: F.cutoff_features(D.plots, D.series, D.gdd, c) for c in F.CUTOFFS}
    pf1, _ = F.fit_predict(D.plots, feats_w1, train, [], D.markers, D.alphas["final"])
    pf = pd.concat([pf1, pf2], ignore_index=True)
    log("final done")
    keep = ["plot_uid", "genotype_id", "year", "trait", "h", "phase", "issue_das", "target"] + NEAR_METHODS
    return feat[keep], pf


def _g_per_trait(feat, train, cfg, cols):
    out = pd.Series(np.nan, index=feat.index)
    for t, c in cfg.items():
        m = feat.trait == t
        out.loc[m] = N2.fit_g(feat[m], train, c, cols)
    return out


# ---------------------------------------------------------------- tuning (inner leave-one-year-out on train)

def tune(D):
    os.makedirs(OUT, exist_ok=True)
    tr_all = D.plots[D.plots.split == "train"]
    years = sorted(tr_all.year.unique())
    for y in years:
        path = f"{OUT}/tune_scores_{y}.json"
        if os.path.exists(path):
            log("tuning: held-out year", y, "already scored")
            continue
        inner = tr_all.plot_uid[tr_all.year != y]
        held = set(tr_all.plot_uid[tr_all.year == y])
        iy = [v for v in years if v != y]
        samples = D.samples[D.samples.plot_uid.isin(set(tr_all.plot_uid))]
        feat = N2.near_features(samples, D.series, D.weather, D.w2, D.plots, D.markers, inner, iy, D.alphas)
        h = feat.plot_uid.isin(held)
        sc = {"G": {}, "S": {}, "D": {}}
        for t in ["canopy_cover", "height"]:
            m = feat.trait == t
            for i, cfg in enumerate(N2.G_GRID):
                p = N2.fit_g(feat[m], inner, cfg)
                sc["G"][f"{t}|{i}"] = float(np.mean(np.abs(p[h[m]] - feat.target[m & h])))
                log(y, "G", t, cfg, round(sc["G"][f"{t}|{i}"] * 100, 3))
        sd = S.SeqData(feat, D.series, inner)
        res = S.train_predict(sd, inner, epochs=S.EPOCH_GRID)
        for e, r in res.items():
            p = S.to_forecast(feat, r)
            for t in ["canopy_cover", "height"]:
                m = (feat.trait == t) & h
                sc["S"][f"{t}|{e}"] = float(np.mean(np.abs(p[m] - feat.target[m])))
        log(y, "S", {k: round(v * 100, 3) for k, v in sc["S"].items()})

        ff = F2.features(D.plots, D.series, D.weather, D.gdd, D.markers, inner, D.alphas)
        choice = {k: F2.CANDIDATES for k in ff}
        pf = F2.fit_predict(D.plots, ff, D.weather, D.gdd, inner, iy, choice)
        pf = pf[pf.plot_uid.isin(held) & pf.method.str.startswith("L0+D_")]
        for (t, c, meth), g in pf.groupby(["target", "cutoff", "method"]):
            e = (g.pred - g.truth).values
            sc["D"][f"{t}|{c}|{meth.split('_')[1]}"] = float(np.mean(np.abs(e - e.mean())))  # within-year MAE
        json.dump(sc, open(path, "w"), indent=1)
        log("tuning: held-out year", y, "done")

    scores = [json.load(open(f"{OUT}/tune_scores_{y}.json")) for y in years]
    mean = {part: pd.DataFrame([s[part] for s in scores]).mean() for part in ["G", "S", "D"]}
    settings = {"G": {}, "S": {}, "D": {}}
    for t in ["canopy_cover", "height"]:
        g = mean["G"][[k for k in mean["G"].index if k.startswith(t + "|")]]
        settings["G"][t] = N2.G_GRID[int(g.idxmin().split("|")[1])]
        s = mean["S"][[k for k in mean["S"].index if k.startswith(t + "|")]]
        settings["S"][t] = int(s.idxmin().split("|")[1])
    for key in sorted({k.rsplit("|", 1)[0] for k in mean["D"].index}):
        d = mean["D"][[k for k in mean["D"].index if k.rsplit("|", 1)[0] == key]]
        t, c = key.split("|")
        settings["D"][json.dumps([t, int(c)])] = list(F2.CANDIDATES[int(d.idxmin().rsplit("|", 1)[1])])
    json.dump(settings, open(f"{OUT}/settings.json", "w"), indent=1)
    for part in mean:
        mean[part].rename("mean_over_held_out_years").to_csv(f"{OUT}/tune_{part}.csv")
    log("settings", json.dumps(settings))


# ---------------------------------------------------------------- summaries

def summarize_near(pred, sets):
    rows = []
    for name, uids in sets.items():
        d = pred[pred.plot_uid.isin(uids)]
        for (trait, h), g in d.groupby(["trait", "h"]):
            for phase, gp in [("all", g)] + [(str(k), v) for k, v in g.groupby("phase", observed=True)]:
                e3 = (gp.N3 - gp.target).values * 100
                e0 = (gp.N0 - gp.target).values * 100
                for m in NEAR_METHODS:
                    e = (gp[m] - gp.target).values * 100
                    mae, lo, hi = bootstrap_mae(e, gp.genotype_id)
                    dm, dlo, dhi = bootstrap_mae(e, gp.genotype_id, other=e3)
                    rows.append({"set": name, "trait": trait, "h": h, "phase": phase, "method": m, "n": len(gp),
                                 "mae": mae, "lo": lo, "hi": hi, "rmse": float(np.sqrt(np.mean(e ** 2))),
                                 "skill_vs_N0": 1 - mae / np.mean(np.abs(e0)), "d_vs_N3": dm, "d_lo": dlo, "d_hi": dhi})
    return pd.DataFrame(rows)


def _wranks(x, W):
    """Midranks of x in each weighted resample (rows of W are multiplicities); ties share their midrank."""
    order = np.argsort(x, kind="stable")
    xs, Ws = x[order], W[:, order]
    cum = np.cumsum(Ws, axis=1)
    grp = np.concatenate([[0], np.cumsum(xs[1:] != xs[:-1])])
    last = np.r_[np.nonzero(np.diff(grp))[0], len(xs) - 1]
    end = cum[:, last]
    prev = np.concatenate([np.zeros((W.shape[0], 1)), end[:, :-1]], axis=1)
    mid = (prev + 1 + end) / 2
    r = np.empty_like(cum)
    r[:, order] = mid[:, grp]
    return r


def _wpearson(a, b, W):
    s = W.sum(1, keepdims=True)
    ma, mb = (W * a).sum(1, keepdims=True) / s, (W * b).sum(1, keepdims=True) / s
    cov = (W * (a - ma) * (b - mb)).sum(1)
    return cov / np.sqrt((W * (a - ma) ** 2).sum(1) * (W * (b - mb) ** 2).sum(1))


def bootstrap_spearman(df, methods, n=2000, seed=0):
    """Within-year Spearman averaged over years, for each method, with 95% intervals from resampling varieties within
    each year; also paired differences between every pair. df: year, genotype_id, truth, and one column per method."""
    per_year_pt, per_year_bs = {m: [] for m in methods}, {m: [] for m in methods}
    for i, (_, g) in enumerate(df.groupby("year")):
        if len(g) <= 5:
            continue
        codes, gi = np.unique(g.genotype_id.values, return_inverse=True)
        rng = np.random.default_rng(seed + i)
        draws = rng.integers(0, len(codes), size=(n, len(codes)))
        Wg = np.stack([np.bincount(r, minlength=len(codes)) for r in draws]).astype(float)
        W = Wg[:, gi]
        rt = _wranks(g.truth.values, W)
        for m in methods:
            x = g[m].values
            if np.unique(x).size < 2:
                per_year_pt[m].append(np.nan)
                per_year_bs[m].append(np.full(n, np.nan))
                continue
            per_year_pt[m].append(spearmanr(x, g.truth.values).statistic)
            per_year_bs[m].append(_wpearson(_wranks(x, W), rt, W))
    pt = {m: float(np.mean(v)) if v else np.nan for m, v in per_year_pt.items()}
    bs = {m: np.mean(np.vstack(v), 0) if v else np.full(n, np.nan) for m, v in per_year_bs.items()}
    return pt, bs


def summarize_final(pred, plots, sets, geno_ids):
    P = plots.set_index("plot_uid")
    dmg = F.damaged(plots)
    rows = []
    for name, uids in sets.items():
        d = pred[pred.plot_uid.isin(uids)].copy()
        d["genotype_id"] = d.plot_uid.map(P.genotype_id)
        d["year"] = d.plot_uid.map(P.year)
        for subset in ["all", "genotyped", "no_damaged"]:
            ds = d
            if subset == "genotyped":
                ds = d[d.genotype_id.isin(geno_ids)]
            elif subset == "no_damaged":
                ds = d[~d.plot_uid.isin(dmg) & d.target.isin(["yield", "yield_adjusted"])]
            for (target, cut), g in ds.groupby(["target", "cutoff"]):
                w = g.pivot_table(index=["plot_uid", "year", "genotype_id", "truth"], columns="method",
                                  values="pred").reset_index()
                ms = [m for m in FINAL_METHODS if m in w and (m != "F2" or subset == "genotyped")]
                w = w.dropna(subset=ms)
                pt, bs = bootstrap_spearman(w, ms)
                refs = [r for r in ["F1", "F2"] if r in ms]
                for m in ms:
                    e = (w[m] - w.truth).values
                    mae, lo, hi = bootstrap_mae(e, w.genotype_id)
                    r = {"set": name, "subset": subset, "target": target, "cutoff": cut, "method": m, "n": len(w),
                         "mae": mae, "lo": lo, "hi": hi, "rmse": float(np.sqrt(np.mean(e ** 2))),
                         "medae": float(np.median(np.abs(e))), "spearman": pt[m],
                         "sp_lo": np.nanpercentile(bs[m], 2.5), "sp_hi": np.nanpercentile(bs[m], 97.5)}
                    e0 = (w["F0"] - w.truth).values
                    r["d_mae_vs_F0"], r["dm_lo"], r["dm_hi"] = bootstrap_mae(e, w.genotype_id, other=e0)
                    for ref in refs:
                        diff = bs[m] - bs[ref]
                        r[f"sp_d_{ref}"] = pt[m] - pt[ref]
                        r[f"sp_d_{ref}_lo"], r[f"sp_d_{ref}_hi"] = np.nanpercentile(diff, [2.5, 97.5])
                    rows.append(r)
    return pd.DataFrame(rows)


def per_year_wins(pn, pf, plots):
    """LOYO: per held-out year MAE of each method, and how many years each method beats the bar."""
    pn = pn.copy()
    rows = []
    for (trait, h, y), g in pn.groupby(["trait", "h", "year"]):
        for m in NEAR_METHODS:
            rows.append({"task": "near", "what": f"{trait}_h{h}", "year": y, "method": m,
                         "mae": float(np.mean(np.abs(g[m] - g.target)) * 100)})
    P = plots.set_index("plot_uid")
    pf = pf.assign(year=pf.plot_uid.map(P.year))
    for (t, c, m, y), g in pf.groupby(["target", "cutoff", "method", "year"]):
        rows.append({"task": "final", "what": f"{t}_c{c}", "year": y, "method": m,
                     "mae": float(np.mean(np.abs(g.pred - g.truth)))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tune", "run"], default="run")
    ap.add_argument("--protocol", choices=["official", "loyo"], default="official")
    ap.add_argument("--eval", choices=["validation", "test"], default="validation")
    ap.add_argument("--summarize", choices=["official_validation", "official_test", "loyo"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    D = Data()
    if a.stage == "tune":
        tune(D)
        return
    if a.summarize:
        return summarize(D, a.summarize)
    settings = json.load(open(f"{OUT}/settings.json"))
    split = D.plots.set_index("plot_uid").split
    if a.protocol == "official":
        tag = f"official_{a.eval}"
        if a.eval == "test" and os.path.exists(f"{OUT}/{tag}_near_pred.parquet"):
            raise SystemExit("official test already scored; it is run once (docs/03-week2-design.md)")
        if a.eval == "test" and not os.path.exists(f"{OUT}/official_validation_near_pred.parquet"):
            raise SystemExit("run the validation sanity check first (docs/03-week2-design.md)")
        train = split.index[split == "train"]
        years = sorted(D.plots.year[D.plots.split == "train"].unique())
        pn, pf = fit_all(D, train, years, settings)
        names = ["validation"] if a.eval == "validation" else OFFICIAL_TEST
        keep = set(split.index[split.isin(names)])
        pn[pn.plot_uid.isin(keep)].to_parquet(f"{OUT}/{tag}_near_pred.parquet", index=False)
        pf[pf.plot_uid.isin(keep)].to_parquet(f"{OUT}/{tag}_final_pred.parquet", index=False)
    else:
        tag = "loyo"
        if not os.path.exists(f"{OUT}/official_test_near_pred.parquet"):
            raise SystemExit("run the official test first (docs/03-week2-design.md)")
        if os.path.exists(f"{OUT}/loyo_near_pred.parquet"):
            raise SystemExit("leave-one-year-out already run; it is run once")
        for y in sorted(D.plots.year.unique()):
            p = f"{OUT}/loyo_{y}_near_pred.parquet"
            if os.path.exists(p):  # resume after an interruption; settings are frozen so a fold is deterministic
                log("year", y, "already done")
                continue
            train = D.plots.plot_uid[D.plots.year != y]
            years = sorted(D.plots.year[D.plots.year != y].unique())
            pn, pf = fit_all(D, train, years, settings)
            test = set(D.plots.plot_uid[D.plots.year == y])
            pf[pf.plot_uid.isin(test)].to_parquet(f"{OUT}/loyo_{y}_final_pred.parquet", index=False)
            pn[pn.plot_uid.isin(test)].to_parquet(p, index=False)
            log("year", y, "done")
        yrs = sorted(D.plots.year.unique())
        pd.concat([pd.read_parquet(f"{OUT}/loyo_{y}_final_pred.parquet") for y in yrs]).to_parquet(
            f"{OUT}/loyo_final_pred.parquet", index=False)
        pd.concat([pd.read_parquet(f"{OUT}/loyo_{y}_near_pred.parquet") for y in yrs]).to_parquet(
            f"{OUT}/loyo_near_pred.parquet", index=False)
    summarize(D, tag)


def summarize(D, tag):
    pn = pd.read_parquet(f"{OUT}/{tag}_near_pred.parquet")
    pf = pd.read_parquet(f"{OUT}/{tag}_final_pred.parquet")
    split = D.plots.set_index("plot_uid").split
    if tag == "loyo":
        sets = {f"year_{y}": set(D.plots.plot_uid[D.plots.year == y]) for y in sorted(D.plots.year.unique())}
        sets["pooled"] = set(D.plots.plot_uid)
        per_year_wins(pn, pf, D.plots).to_csv(f"{OUT}/loyo_per_year.csv", index=False)
    else:
        names = ["validation"] if tag.endswith("validation") else OFFICIAL_TEST
        sets = {n: set(split.index[split == n]) for n in names}
    sn = summarize_near(pn, sets)
    sf = summarize_final(pf, D.plots, sets, set(D.markers[0]))
    sn.to_csv(f"{OUT}/{tag}_near.csv", index=False)
    sf.to_csv(f"{OUT}/{tag}_final.csv", index=False)
    show = sn[sn.phase == "all"].pivot_table(index=["set", "trait", "h"], columns="method", values="mae").round(2)
    print(show[NEAR_METHODS].to_string())
    showf = sf[sf.subset == "genotyped"].pivot_table(index=["set", "target", "cutoff"], columns="method",
                                                     values="spearman").round(3)
    print(showf.to_string())


if __name__ == "__main__":
    main()
