"""
Week-1 baselines runner (docs/02-baselines-design.md).

  python src/run_week1.py --protocol official --eval validation   # settings chosen on validation; can be rerun
  python src/run_week1.py --protocol official --eval test         # official test splits; refuses to run twice
  python src/run_week1.py --protocol loyo                         # after the official test; reuses its settings

Writes outputs/week1/<protocol>_<eval>_{near,final}.csv and <protocol>_alphas.json.
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import final as F
import near as N
from common import OFFICIAL_TEST, bootstrap_mae, load_derived, load_markers
from series import gdd_table

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = "outputs/week1"


def near_samples(plots, series):
    p = "data/derived/near_samples.parquet"
    if os.path.exists(p):
        return pd.read_parquet(p)
    s = N.build_samples(plots, series)
    s.to_parquet(p, index=False)
    return s


def summarize_near(pred, sets, ref="N0"):
    rows = []
    for name, uids in sets.items():
        d = pred[pred.plot_uid.isin(uids)]
        for (trait, h), g in d.groupby(["trait", "h"]):
            for phase, gp in [("all", g)] + [(str(k), v) for k, v in g.groupby("phase", observed=True)]:
                for m in ["N0", "N1", "N2", "N3", "N4", "N5"]:
                    e, e0 = (gp[m] - gp.target).values * 100, (gp[ref] - gp.target).values * 100
                    mae, lo, hi = bootstrap_mae(e, gp.genotype_id)
                    dm, dlo, dhi = bootstrap_mae(e, gp.genotype_id, other=e0)
                    rows.append({"set": name, "trait": trait, "h": h, "phase": phase, "method": m, "n": len(gp),
                                 "plots": gp.plot_uid.nunique(), "mae": mae, "lo": lo, "hi": hi,
                                 "rmse": float(np.sqrt(np.mean(e ** 2))), "skill": 1 - mae / np.mean(np.abs(e0)),
                                 "d_vs_N0": dm, "d_lo": dlo, "d_hi": dhi})
    return pd.DataFrame(rows)


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
            for (target, cut, m), g in ds.groupby(["target", "cutoff", "method"]):
                if m == "F2" and subset != "genotyped":
                    continue
                e = (g.pred - g.truth).values
                mae, lo, hi = bootstrap_mae(e, g.genotype_id)
                rho = [spearmanr(x.pred, x.truth).statistic for _, x in g.groupby("year") if len(x) > 5 and x.pred.nunique() > 1]
                rows.append({"set": name, "subset": subset, "target": target, "cutoff": cut, "method": m, "n": len(g),
                             "mae": mae, "lo": lo, "hi": hi, "rmse": float(np.sqrt(np.mean(e ** 2))),
                             "medae": float(np.median(np.abs(e))),
                             "spearman_within_year": float(np.mean(rho)) if rho else np.nan})
    return pd.DataFrame(rows)


def run(plots, series, weather, samples, markers, train, val, years, eval_sets, alphas=None):
    pn, an = N.fit_predict(samples, series, weather, train, val, years, None if alphas is None else alphas["near"])
    gdd = gdd_table(weather, plots.groupby("year").sowing_date.first().to_dict())
    feats = {c: F.cutoff_features(plots, series, gdd, c) for c in F.CUTOFFS}
    pf, af = F.fit_predict(plots, feats, train, val, markers, None if alphas is None else alphas["final"])
    return summarize_near(pn, eval_sets), summarize_final(pf, plots, eval_sets, set(markers[0])), {"near": an, "final": af}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["official", "loyo"], default="official")
    ap.add_argument("--eval", choices=["validation", "test"], default="validation")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    plots, series, weather = load_derived()
    series = series.merge(plots[["plot_uid", "genotype_id"]], on="plot_uid")
    samples = near_samples(plots, series)
    markers = load_markers()
    print(f"plots {len(plots)}, near samples {len(samples)}, genotyped varieties {len(markers[0])}")

    if a.protocol == "official":
        tag = f"official_{a.eval}"
        if a.eval == "test" and os.path.exists(f"{OUT}/{tag}_near.csv"):
            raise SystemExit("official test already scored; it is run once (docs/02-baselines-design.md)")
        split = plots.set_index("plot_uid").split
        train, val = split.index[split == "train"], split.index[split == "validation"]
        years = sorted(plots.year[plots.split == "train"].unique())
        names = ["validation"] if a.eval == "validation" else OFFICIAL_TEST
        sets = {n: set(split.index[split == n]) for n in names}
        alphas = None
        if a.eval == "test":
            alphas = json.load(open(f"{OUT}/official_alphas.json"))  # settings frozen on validation
        sn, sf, ch = run(plots, series, weather, samples, markers, train, val, years, sets, alphas)
        if a.eval == "validation":
            json.dump(ch, open(f"{OUT}/official_alphas.json", "w"), indent=1)
    else:
        tag = "loyo"
        if not os.path.exists(f"{OUT}/official_test_near.csv"):
            raise SystemExit("run the official test first (docs/02-baselines-design.md, protocol 2)")
        alphas = json.load(open(f"{OUT}/official_alphas.json"))
        parts_n, parts_f = [], []
        for y in sorted(plots.year.unique()):
            train = plots.plot_uid[plots.year != y]
            test = set(plots.plot_uid[plots.year == y])
            years = sorted(plots.year[plots.year != y].unique())
            sn, sf, _ = run(plots, series, weather, samples, markers, train, [], years, {f"year_{y}": test}, alphas)
            parts_n.append(sn)
            parts_f.append(sf)
            print("done", y)
        sn, sf = pd.concat(parts_n), pd.concat(parts_f)
    sn.to_csv(f"{OUT}/{tag}_near.csv", index=False)
    sf.to_csv(f"{OUT}/{tag}_final.csv", index=False)
    show = sn[(sn.phase == "all")].pivot_table(index=["set", "trait", "h"], columns="method", values="mae").round(2)
    print(show.to_string())
    showf = sf[sf.subset == "all"].pivot_table(index=["set", "target", "cutoff"], columns="method", values="mae").round(3)
    print(showf.to_string())


if __name__ == "__main__":
    main()
