"""
End-of-season task (docs/02-baselines-design.md): forecast heading (days after sowing), final height (cm), yield_adjusted
and yield (t/ha) from what is known at a cutoff (days 120, 150, 180, 210 after sowing; heading only up to 180).
Baselines F0-F3; F2 (markers) only for genotyped varieties.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

TARGETS = {"heading_das": 1.0, "height_final": 100.0, "yield_adjusted": 1.0, "yield": 1.0}  # scale to report units
CUTOFFS = [120, 150, 180, 210]
HEADING_MAX_CUTOFF = 180
ALPHAS = [0.1, 1, 10, 100, 1000]
MARKER_ALPHAS = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
F3_BASE = ["cc_last", "cc_max", "cc_slope14", "cc_missing", "h_last", "h_missing", "gdd_cut"]


def cutoff_features(plots, series, gdd, cutoff):
    """Features from measurements dated on or before day `cutoff` after sowing (one row per plot-year). `gdd` is the
    per-year degree-day table from series.gdd_table (weather up to the cutoff only)."""
    s = series[series.das <= cutoff]
    f = pd.DataFrame(index=plots.plot_uid.values)
    cc = s[s.trait == "canopy_cover"].sort_values("das")
    g = cc.groupby("plot_uid")
    f["cc_last"] = g.value.last()
    f["cc_max"] = g.value.max()
    last_das = g.das.last()

    def v_at(gr):
        return np.interp(gr.das.values[-1] - 14, gr.das.values, gr.value.values)
    f["cc_slope14"] = (f.cc_last - g.apply(v_at, include_groups=False)) / 14.0
    f["cc_missing"] = f.cc_last.isna().astype(float)
    hh = s[s.trait == "height"].sort_values("das").groupby("plot_uid")
    f["h_last"] = hh.value.last()
    f["h_missing"] = f.h_last.isna().astype(float)
    cut_date = plots.sowing_date + pd.Timedelta(days=cutoff)
    key = pd.DataFrame({"year": plots.year.values, "date": cut_date.values})
    f["gdd_cut"] = key.merge(gdd, on=["year", "date"], how="left").gdd.values  # degree days from sowing to the cutoff
    f.index.name = "plot_uid"
    return f


def year_dev(plots, target, train_plots):
    """Deviation of each training plot from its year's training mean."""
    t = plots[plots.plot_uid.isin(train_plots)].dropna(subset=[target])
    return t.assign(dev=t[target] - t.groupby("year")[target].transform("mean"))


def fit_predict(plots, feats_by_cut, train_plots, val_plots, markers=None, alphas=None):
    """Returns a long table: plot_uid, target, cutoff, method, pred (report units) plus the chosen ridge strengths."""
    train_plots, val_plots = set(train_plots), set(val_plots)
    chosen = {} if alphas is None else dict(alphas)
    P = plots.set_index("plot_uid")
    rows = []
    for target, scale in TARGETS.items():
        dev = year_dev(plots, target, train_plots)
        f0 = dev.groupby("year")[target].mean().mean()
        vdev = dev.groupby("genotype_id").dev.mean()
        f1_dev = P.genotype_id.map(vdev)
        base = pd.DataFrame({"F0": f0, "F1": f0 + f1_dev.fillna(0.0)}, index=P.index)

        if markers is not None:
            ids, M = markers
            pos = {g: i for i, g in enumerate(ids)}
            has = P.genotype_id.isin(pos)
            tr = dev[dev.genotype_id.isin(pos)]
            Xtr = M[[pos[g] for g in tr.genotype_id]]
            mu = Xtr.mean(0)
            key = f"{target}_markers"
            if key not in chosen:
                va = P[has & P.index.isin(val_plots) & P[target].notna()]
                ymean = dev.groupby("year")[target].mean()
                va_dev = va[target] - va.year.map(ymean)  # validation years are training years in protocol 1
                Xva = M[[pos[g] for g in va.genotype_id]] - mu
                maes = {a: np.mean(np.abs(Ridge(alpha=a).fit(Xtr - mu, tr.dev.values).predict(Xva) - va_dev.values))
                        for a in MARKER_ALPHAS}
                chosen[key] = min(maes, key=maes.get)
            m = Ridge(alpha=chosen[key]).fit(Xtr - mu, tr.dev.values)
            base["F2"] = np.nan
            idx = P.index[has]
            base.loc[idx, "F2"] = f0 + m.predict(M[[pos[g] for g in P.loc[idx, "genotype_id"]]] - mu)

        for cut, F in feats_by_cut.items():
            if target == "heading_das" and cut > HEADING_MAX_CUTOFF:
                continue
            X = F.reindex(P.index).copy()
            X["var_dev"] = f1_dev.fillna(0.0)
            X["var_missing"] = f1_dev.isna().astype(float)
            cols = F3_BASE + ["var_dev", "var_missing"]
            trm = X.index.isin(train_plots) & P[target].notna().values
            fill = X.loc[trm, cols].mean()
            X[cols] = X[cols].fillna(fill)  # missing measurements -> training mean, flagged by the *_missing columns
            sc = StandardScaler().fit(X.loc[trm, cols])
            Xtr, ytr = sc.transform(X.loc[trm, cols]), P.loc[trm, target].values
            key = f"{target}_c{cut}"
            if key not in chosen:
                vam = X.index.isin(val_plots) & P[target].notna().values
                Xva = sc.transform(X.loc[vam, cols])
                vy = P.loc[vam, "year"].values

                def within_year_mae(err):  # remove each validation year's mean error: rewards plot skill, not year lookup
                    return np.mean(np.abs(err - pd.Series(err).groupby(vy).transform("mean").values))
                maes = {a: within_year_mae(Ridge(alpha=a).fit(Xtr, ytr).predict(Xva) - P.loc[vam, target].values)
                        for a in ALPHAS}
                chosen[key] = min(maes, key=maes.get)
            pred3 = Ridge(alpha=chosen[key]).fit(Xtr, ytr).predict(sc.transform(X[cols]))
            for meth in base.columns:
                rows.append(pd.DataFrame({"plot_uid": P.index, "target": target, "cutoff": cut, "method": meth,
                                          "pred": base[meth].values * scale}))
            rows.append(pd.DataFrame({"plot_uid": P.index, "target": target, "cutoff": cut, "method": "F3",
                                      "pred": pred3 * scale}))
    out = pd.concat(rows, ignore_index=True)
    truth = plots.melt(id_vars=["plot_uid"], value_vars=list(TARGETS), var_name="target", value_name="truth")
    truth["truth"] = truth.truth * truth.target.map(TARGETS)
    return out.merge(truth, on=["plot_uid", "target"]).dropna(subset=["pred", "truth"]), chosen


def damaged(plots):
    """Plot-years whose raw yield is more than 4 robust SD below their year's median (fixed in the data check)."""
    g = plots.groupby("year")["yield"]
    med = g.transform("median")
    mad = g.transform(lambda x: 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x))))
    return set(plots.plot_uid[(plots["yield"] < med - 4 * mad)])
