"""
Week-3 part A (docs/04-week3-design.md): where and why the week-2 learned models lose to N3, from the saved
leave-one-year-out predictions only (no new fit, no test reuse).

A1  error difference vs N3 by held-out year x issue day (14-day bins) and phase
A2  error split into the field-wide part (mean error over all plots of a year x trait x horizon x issue date) and the
    plot-specific remainder
A3  share of a held-out year's samples outside the training years' 1st-99th percentile range of degree days so far,
    degree-day increment, or anomaly; error inside vs outside

  python src/analysis_w2.py      # writes outputs/week3/partA_*.csv
"""
import os

import numpy as np
import pandas as pd

from common import ClimTemp, Curve, load_derived

OUT = "outputs/week3"
M = ["N3", "G", "S", "E"]


def main():
    os.makedirs(OUT, exist_ok=True)
    plots, series, weather = load_derived()
    p = pd.read_parquet("outputs/week2/loyo_near_pred.parquet")
    smp = pd.read_parquet("data/derived/near_samples.parquet",
                          columns=["plot_uid", "trait", "h", "issue_das", "issue_gdd", "issue_sday", "target_sday",
                                   "last"])
    p = p.merge(smp, on=["plot_uid", "trait", "h", "issue_das"], how="left", validate="one_to_one")
    for m in M:
        p[f"e_{m}"] = (p[m] - p.target) * 100  # points / cm

    # A1
    p["bin"] = (p.issue_das // 14) * 14
    a1 = p.groupby(["year", "trait", "h", "bin"]).agg(
        n=("target", "size"), **{f"mae_{m}": (f"e_{m}", lambda x: np.mean(np.abs(x))) for m in M}).reset_index()
    for m in ["G", "S", "E"]:
        a1[f"d_{m}"] = a1[f"mae_{m}"] - a1.mae_N3
    a1.to_csv(f"{OUT}/partA1_by_issue_day.csv", index=False)

    # A2
    key = ["year", "trait", "h", "issue_das"]
    rows = []
    for m in M:
        common = p.groupby(key)[f"e_{m}"].transform("mean")
        p[f"c_{m}"], p[f"r_{m}"] = common, p[f"e_{m}"] - common
    for (y, t, h), g in p.groupby(["year", "trait", "h"]):
        r = {"year": y, "trait": t, "h": h, "n": len(g)}
        for m in M:
            r[f"total_{m}"] = np.mean(np.abs(g[f"e_{m}"]))
            r[f"field_{m}"] = np.mean(np.abs(g[f"c_{m}"]))
            r[f"plot_{m}"] = np.mean(np.abs(g[f"r_{m}"]))
        rows.append(r)
    pd.DataFrame(rows).to_csv(f"{OUT}/partA2_field_vs_plot.csv", index=False)

    # A3: per held-out year, ranges from the other years' samples (the fold's training years)
    rows = []
    for y in sorted(p.year.unique()):
        tr_years = [v for v in sorted(plots.year.unique()) if v != y]
        clim = ClimTemp(weather, tr_years)
        tr_series = series[series.year.isin(tr_years)]
        allp = p.copy()
        allp["gdd_inc"] = clim.increment(allp.issue_sday.values, allp.target_sday.values)
        allp["anom"] = np.nan
        for t in ["canopy_cover", "height"]:
            c = Curve(tr_series[tr_series.trait == t], "gdd", 50)
            mt = allp.trait == t
            allp.loc[mt, "anom"] = allp.loc[mt, "last"].values - c(allp.loc[mt, "issue_gdd"].values.astype(float))
        for (t, h), g in allp.groupby(["trait", "h"]):
            tr, te = g[g.year != y], g[g.year == y]
            out = np.zeros(len(te), bool)
            for v in ["issue_gdd", "gdd_inc", "anom"]:
                lo, hi = np.nanpercentile(tr[v], [1, 99])
                out |= ((te[v] < lo) | (te[v] > hi)).values
            r = {"year": y, "trait": t, "h": h, "n": len(te), "share_outside": out.mean()}
            for m in M:
                e = np.abs(te[f"e_{m}"].values)
                r[f"mae_in_{m}"] = e[~out].mean() if (~out).any() else np.nan
                r[f"mae_out_{m}"] = e[out].mean() if out.any() else np.nan
            rows.append(r)
    pd.DataFrame(rows).to_csv(f"{OUT}/partA3_extrapolation.csv", index=False)
    print("part A written to", OUT)


if __name__ == "__main__":
    main()
