"""
Week-3 runner (docs/04-week3-design.md): photo models (part B) and year-held-out conformal ranges (part C).

  python src/run_week3.py unit K              # fit unit K (see UNITS): one outer fold x one inner held-out year (or none)
  python src/run_week3.py score validation    # sanity check (bugs only); needs unit official/none
  python src/run_week3.py score test          # official test splits, once; needs the official inner units
  python src/run_week3.py score loyo          # headline; after the test; needs all leave-one-year-out units

Every unit fits on its training plots only, with week-2 settings frozen, and saves predictions for the plots it holds out.
Inner units (an inner held-out year) give the residuals that calibrate the ranges of their outer fold.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

import final2 as F2
import near as N
import near2 as N2
import photo_feats as PF
import photos as PH
import ranges as RG
import run_week2 as R
import seq as S
from common import OFFICIAL_TEST, bootstrap_mae

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = "outputs/week3"
UNITS_DIR = f"{OUT}/units"
NEAR_M = ["N3", "G", "G+P", "S", "E", "E+P"]
FINAL_M = ["L0+D", "L1+D", "L0+D+P", "L1+D+P", "L1"]
NEAR_GROUPS = ["trait", "h", "phase"]
FINAL_GROUPS = ["target", "cutoff"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def units(plots):
    off = sorted(plots.year[plots.split == "train"].unique())
    yrs = sorted(plots.year.unique())
    return [("official", None)] + [("official", y) for y in off] + [(Y, None) for Y in yrs] + \
        [(Y, y) for Y in yrs for y in yrs if y != Y]


def unit_name(outer, inner):
    return f"{outer}_{'none' if inner is None else inner}"


def unit_plots(plots, outer, inner):
    """(training plots, training years, plots to predict) for a unit."""
    base = plots[plots.split == "train"] if outer == "official" else plots[plots.year != outer]
    if inner is None:
        pred = plots[~plots.plot_uid.isin(set(base.plot_uid))]
        train = base
    else:
        pred, train = base[base.year == inner], base[base.year != inner]
    return list(train.plot_uid), sorted(train.year.unique()), set(pred.plot_uid)


# ---------------------------------------------------------------- one unit

def fit_unit(D, settings, outer, inner, photos=None):
    train, years, pred_plots = unit_plots(D.plots, outer, inner)
    keep = set(train) | pred_plots
    samples = D.samples[D.samples.plot_uid.isin(keep)]
    feat = N2.near_features(samples, D.series, D.weather, D.w2, D.plots, D.markers, train, years, D.alphas)
    log(unit_name(outer, inner), "near features", len(feat))
    meta, X = photos if photos is not None else PH.load_features()
    ph = PF.photo_pcs(meta, X, D.plots, train)
    feat = feat.join(PF.near_inputs(feat, ph))
    for name, cols in [("G", N2.FEATS_G), ("G+P", N2.FEATS_G + PF.NEAR_COLS)]:
        feat[name] = np.nan
        for t, cfg in settings["G"].items():
            m = feat.trait == t
            feat.loc[m, name] = N2.fit_g(feat[m], train, cfg, cols)
    log("G, G+P done")
    sd = S.SeqData(feat, D.series, train)
    res = S.train_predict(sd, train, epochs=sorted(set(settings["S"].values())))
    feat["S"] = np.nan
    for t, e in settings["S"].items():
        m = feat.trait == t
        feat.loc[m, "S"] = S.to_forecast(feat[m], res[e][m])
    feat["E"], feat["E+P"] = (feat.G + feat.S) / 2, (feat["G+P"] + feat.S) / 2
    log("S done")

    ff = F2.features(D.plots, D.series, D.weather, D.gdd, D.markers, train, D.alphas)
    choice = {tuple(json.loads(k)): [tuple(v)] for k, v in settings["D"].items()}
    pf = F2.fit_predict(D.plots, ff, D.weather, D.gdd, train, years, choice)
    ffp = {k: v.join(PF.final_inputs(D.plots, ph, k[1])) for k, v in ff.items()}
    pfp = F2.fit_predict(D.plots, ffp, D.weather, D.gdd, train, years, choice, cols=F2.FEATS + PF.FINAL_COLS,
                         suffix="+P")
    log("final done")
    cols = ["plot_uid", "genotype_id", "year", "trait", "h", "phase", "issue_das", "target"] + NEAR_M
    near = feat.loc[feat.plot_uid.isin(pred_plots), cols]
    fin = pd.concat([pf, pfp], ignore_index=True)
    fin = fin[fin.plot_uid.isin(pred_plots)]
    return near, fin


# ---------------------------------------------------------------- scoring helpers

def compare_near(pred, sets, pairs):
    rows = []
    for name, uids in sets.items():
        d = pred[pred.plot_uid.isin(uids)]
        for (trait, h), g in d.groupby(["trait", "h"]):
            for m, ref in pairs:
                e, e0 = (g[m] - g.target).values * 100, (g[ref] - g.target).values * 100
                dm, lo, hi = bootstrap_mae(e, g.genotype_id, other=e0)
                yrs = g.groupby("year").apply(lambda x: np.mean(np.abs(x[m] - x.target)) < np.mean(
                    np.abs(x[ref] - x.target)), include_groups=False)
                rows.append(dict(set=name, trait=trait, h=h, method=m, ref=ref, mae=np.mean(np.abs(e)),
                                 mae_ref=np.mean(np.abs(e0)), d=dm, d_lo=lo, d_hi=hi, years_better=int(yrs.sum()),
                                 years=len(yrs), adds=bool(hi < 0 and (len(yrs) < 6 or yrs.sum() >= 4))))
    return pd.DataFrame(rows)


def compare_final(pred, plots, sets, geno_ids, pairs):
    P = plots.set_index("plot_uid")
    rows = []
    for name, uids in sets.items():
        d = pred[pred.plot_uid.isin(uids)].assign(genotype_id=lambda x: x.plot_uid.map(P.genotype_id),
                                                  year=lambda x: x.plot_uid.map(P.year))
        for subset in ["all", "genotyped"]:
            ds = d if subset == "all" else d[d.genotype_id.isin(geno_ids)]
            for (t, c), g in ds.groupby(["target", "cutoff"]):
                w = g.pivot_table(index=["plot_uid", "year", "genotype_id", "truth"], columns="method",
                                  values="pred").reset_index()
                for m, ref in pairs:
                    if m not in w or ref not in w:
                        continue
                    ww = w.dropna(subset=[m, ref])
                    pt, bs = R.bootstrap_spearman(ww, [m, ref])
                    diff = bs[m] - bs[ref]
                    e, e0 = (ww[m] - ww.truth).values, (ww[ref] - ww.truth).values
                    dm, lo, hi = bootstrap_mae(e, ww.genotype_id, other=e0)
                    rows.append(dict(set=name, subset=subset, target=t, cutoff=c, method=m, ref=ref, n=len(ww),
                                     spearman=pt[m], spearman_ref=pt[ref], sp_d=pt[m] - pt[ref],
                                     sp_lo=np.nanpercentile(diff, 2.5), sp_hi=np.nanpercentile(diff, 97.5),
                                     mae=np.mean(np.abs(e)), mae_ref=np.mean(np.abs(e0)), d_mae=dm, dm_lo=lo,
                                     dm_hi=hi))
    return pd.DataFrame(rows)


def near_long(near, methods):
    rows = []
    for m in methods:
        rows.append(near[["plot_uid", "year", "trait", "h", "phase", "target"]].assign(
            method=m, pred=near[m].values, truth=near.target.values))
    return pd.concat(rows, ignore_index=True)


def ranges_for(outer_units, inner_units, kind):
    """Calibrate on the inner units' residuals, apply to the outer unit's predictions."""
    if kind == "near":
        cal = near_long(pd.concat(inner_units), ["E", "E+P"])
        tgt = near_long(outer_units, ["E", "E+P"])
        groups = ["method"] + NEAR_GROUPS
        lo_lim = lambda d: 0.0  # noqa: E731
        hi_lim = lambda d: np.where(d.trait == "canopy_cover", 1.0, np.inf)  # noqa: E731
    else:
        keep = ["L0+D", "L1+D", "L0+D+P", "L1+D+P"]
        cal = pd.concat(inner_units)
        cal = cal[cal.method.isin(keep)].merge(_years, on="plot_uid")
        tgt = outer_units[outer_units.method.isin(keep)].merge(_years, on="plot_uid")
        groups = ["method"] + FINAL_GROUPS
        lo_lim, hi_lim = (lambda d: 0.0), None
    cal = cal.assign(res=cal.truth - cal.pred)
    q = pd.concat([RG.half_widths(cal, groups, lv) for lv in RG.LEVELS], ignore_index=True)
    return RG.apply(tgt, q, groups, lo_lim, hi_lim), q


_years = None


# ---------------------------------------------------------------- main

def main():
    global _years
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["unit", "score"])
    ap.add_argument("arg")
    a = ap.parse_args()
    os.makedirs(UNITS_DIR, exist_ok=True)
    D = R.Data()
    _years = D.plots[["plot_uid", "year"]]
    settings = json.load(open("outputs/week2/settings.json"))
    U = units(D.plots)
    if a.step == "unit":
        outer, inner = U[int(a.arg)]
        name = unit_name(outer, inner)
        if os.path.exists(f"{UNITS_DIR}/{name}_final.parquet"):
            log(name, "already done")
            return
        near, fin = fit_unit(D, settings, outer, inner)
        near.to_parquet(f"{UNITS_DIR}/{name}_near.parquet", index=False)
        fin.to_parquet(f"{UNITS_DIR}/{name}_final.parquet", index=False)
        log(name, "saved", len(near), len(fin))
        return
    score(D, a.arg, U)


def _load(name):
    return (pd.read_parquet(f"{UNITS_DIR}/{name}_near.parquet"), pd.read_parquet(f"{UNITS_DIR}/{name}_final.parquet"))


def _need(names):
    miss = [n for n in names if not os.path.exists(f"{UNITS_DIR}/{n}_final.parquet")]
    if miss:
        raise SystemExit(f"units not done: {miss}")


def score(D, what, U):
    split = D.plots.set_index("plot_uid").split
    geno = set(D.markers[0])
    near_pairs = [("G+P", "G"), ("E+P", "E"), ("E", "N3"), ("E+P", "N3")]
    final_pairs = [("L0+D+P", "L0+D"), ("L1+D+P", "L1+D")]
    if what == "validation":
        _need(["official_none"])
        near, fin = _load("official_none")
        sets = {"validation": set(split.index[split == "validation"])}
        compare_near(near, sets, near_pairs).to_csv(f"{OUT}/validation_near.csv", index=False)
        compare_final(fin, D.plots, sets, geno, final_pairs).to_csv(f"{OUT}/validation_final.csv", index=False)
        photo_sanity(D).to_csv(f"{OUT}/photo_sanity.csv", index=False)
        log("validation scored")
        return
    if what == "test":
        if os.path.exists(f"{OUT}/test_near.csv"):
            raise SystemExit("official test already scored; it is run once (docs/04-week3-design.md)")
        if not os.path.exists(f"{OUT}/validation_near.csv"):
            raise SystemExit("score validation first")
        inner = [unit_name(o, i) for o, i in U if o == "official" and i is not None]
        _need(["official_none"] + inner)
        near, fin = _load("official_none")
        sets = {n: set(split.index[split == n]) for n in OFFICIAL_TEST}
        compare_near(near, sets, near_pairs).to_csv(f"{OUT}/test_near.csv", index=False)
        compare_final(fin, D.plots, sets, geno, final_pairs).to_csv(f"{OUT}/test_final.csv", index=False)
        inn = [_load(n) for n in inner]
        rn, _ = ranges_for(near[near.plot_uid.isin(set().union(*sets.values()))], [x[0] for x in inn], "near")
        rf, _ = ranges_for(fin[fin.plot_uid.isin(set().union(*sets.values()))], [x[1] for x in inn], "final")
        rn["set"], rf["set"] = rn.plot_uid.map(split), rf.plot_uid.map(split)
        RG.summarize(rn, ["set", "method", "trait", "h"]).to_csv(f"{OUT}/test_ranges_near.csv", index=False)
        RG.summarize(rf, ["set", "method", "target", "cutoff"]).to_csv(f"{OUT}/test_ranges_final.csv", index=False)
        log("official test scored")
        return
    # leave-one-year-out
    if not os.path.exists(f"{OUT}/test_near.csv"):
        raise SystemExit("score the official test first (docs/04-week3-design.md)")
    if os.path.exists(f"{OUT}/loyo_near.csv"):
        raise SystemExit("leave-one-year-out already scored; it is run once")
    yrs = sorted(D.plots.year.unique())
    _need([unit_name(o, i) for o, i in U if o != "official"])
    near = pd.concat([_load(unit_name(y, None))[0] for y in yrs], ignore_index=True)
    fin = pd.concat([_load(unit_name(y, None))[1] for y in yrs], ignore_index=True)
    sets = {f"year_{y}": set(D.plots.plot_uid[D.plots.year == y]) for y in yrs}
    sets["pooled"] = set(D.plots.plot_uid)
    compare_near(near, sets, near_pairs).to_csv(f"{OUT}/loyo_near.csv", index=False)
    compare_final(fin, D.plots, sets, geno, final_pairs).to_csv(f"{OUT}/loyo_final.csv", index=False)
    rn_all, rf_all, qs = [], [], []
    for y in yrs:
        inn = [_load(unit_name(y, i)) for i in yrs if i != y]
        rn, qn = ranges_for(_load(unit_name(y, None))[0], [x[0] for x in inn], "near")
        rf, qf = ranges_for(_load(unit_name(y, None))[1], [x[1] for x in inn], "final")
        rn_all.append(rn)
        rf_all.append(rf)
        qs.append(pd.concat([qn.assign(kind="near"), qf.assign(kind="final")]).assign(held_out=y))
    rn, rf = pd.concat(rn_all), pd.concat(rf_all)
    RG.summarize(rn, ["year", "method", "trait", "h"]).to_csv(f"{OUT}/loyo_ranges_near.csv", index=False)
    RG.summarize(rn, ["year", "method", "trait", "h", "phase"]).to_csv(f"{OUT}/loyo_ranges_near_phase.csv",
                                                                        index=False)
    RG.summarize(rf, ["year", "method", "target", "cutoff"]).to_csv(f"{OUT}/loyo_ranges_final.csv", index=False)
    pd.concat(qs).to_csv(f"{OUT}/loyo_ranges_q.csv", index=False)
    log("leave-one-year-out scored")


def photo_sanity(D):
    """Part B sanity check: photo PCs (fit on official train) predict the same day's canopy cover in 2019 (a year not in
    official train)."""
    from sklearn.linear_model import Ridge
    meta, X = PH.load_features()
    train = list(D.plots.plot_uid[D.plots.split == "train"])
    ph = PF.photo_pcs(meta, X, D.plots, train)
    cc = D.series[D.series.trait == "canopy_cover"][["plot_uid", "das", "value", "year"]]
    m = ph.merge(cc, on=["plot_uid", "das"])
    tr, te = m[m.plot_uid.isin(set(train))], m[m.year == 2019]
    r = Ridge(alpha=1.0).fit(tr[PF.PC], tr.value).predict(te[PF.PC])
    ss = ((te.value - r) ** 2).sum() / ((te.value - te.value.mean()) ** 2).sum()
    return pd.DataFrame([{"n_train": len(tr), "n_2019": len(te), "r2_2019": 1 - ss,
                          "mae_points": float(np.mean(np.abs(te.value - r)) * 100)}])


if __name__ == "__main__":
    main()
