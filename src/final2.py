"""
Week-2 end-of-season models (docs/03-week2-design.md): forecast = level + deviation.

Level: L0 (= F0, training years' mean) for every target; L1 for heading (day on which degree days, extrapolated with
climatology after the cutoff, reach the training years' mean degree days at heading).
Deviation D: the plot's value minus its year's training mean, from within-year-centred in-season anomalies at the cutoff
and genetics (other years only). D-lin (ridge) or D-gbm (trees), chosen per target and cutoff in tuning.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import features2 as F2
import final as F
from common import ClimTemp, Curve, season_day

INSEASON = ["cc_anom", "cc_anom_max", "cc_slope14", "h_anom", "h_slope14"]
FLAGS = ["cc_missing", "h_missing"]
GEN = ["var_dev", "var_n", "mk"] + [f"pc{i}" for i in range(F2.N_PCS)] + ["genotyped"]
FEATS = INSEASON + FLAGS + GEN
CANDIDATES = [("lin", dict(alpha=a)) for a in [0.1, 1, 10, 100, 1000]] + \
    [("gbm", dict(max_leaf_nodes=l, max_iter=it)) for l in (7, 15) for it in (100, 300)]
GBM_FIXED = dict(loss="squared_error", learning_rate=0.05, min_samples_leaf=50, early_stopping=False, random_state=0)


def _slope14(g):
    return (g.value.values[-1] - np.interp(g.das.values[-1] - 14, g.das.values, g.value.values)) / 14.0


def inseason(plots, series, cutoff, curves):
    """In-season anomalies at the cutoff (values dated on or before it only), centred within each year over all plots
    of that year that have the feature."""
    s = series[series.das <= cutoff]
    f = pd.DataFrame(index=pd.Index(plots.plot_uid.values, name="plot_uid"))
    for trait, pre in [("canopy_cover", "cc"), ("height", "h")]:
        d = s[s.trait == trait].sort_values("das")
        d = d.assign(anom=d.value.astype(float).values - curves[trait](d.gdd.values.astype(float)))
        g = d.groupby("plot_uid")
        f[f"{pre}_anom"] = g.anom.last()
        if pre == "cc":
            f["cc_anom_max"] = g.anom.max()
        f[f"{pre}_slope14"] = g[["das", "value"]].apply(_slope14)
        f[f"{pre}_missing"] = f[f"{pre}_anom"].isna().astype(float)
    year = plots.set_index("plot_uid").year.reindex(f.index)
    f[INSEASON] = f[INSEASON] - f[INSEASON].groupby(year.values).transform("mean")
    return f


def heading_gdd_level(plots, gdd, weather, train_plots, train_years, cutoff):
    """L1: per plot, heading day (after sowing) at which degree days so far (actual, up to the cutoff) plus climatology
    after it reach the training years' mean degree days at heading (equal weight per year)."""
    P = plots.set_index("plot_uid")
    tr = P[P.index.isin(set(train_plots)) & P.heading_das.notna()]
    hd_date = tr.sowing_date + pd.to_timedelta(tr.heading_das, unit="D")
    g_at = pd.DataFrame({"year": tr.year.values, "date": hd_date.values}).merge(gdd, on=["year", "date"], how="left")
    target = g_at.groupby("year").gdd.mean().mean()
    clim = ClimTemp(weather, train_years)
    out = pd.Series(np.nan, index=P.index)
    for y, py in P.groupby("year"):
        sow = py.sowing_date.iloc[0]
        cdate = sow + pd.Timedelta(days=cutoff)
        g_cut = gdd[(gdd.year == y) & (gdd.date == cdate)].gdd.values[0]
        s_cut = int(season_day([cdate], [y])[0])
        days = np.arange(1, 366 - s_cut)
        reach = g_cut + clim.increment(np.full(len(days), s_cut), s_cut + days)
        k = np.argmax(reach >= target) if (reach >= target).any() else len(days) - 1
        out.loc[py.index] = cutoff + days[k]
    return out, target


def genetics(plots, target, train_plots, markers, alpha):
    P = plots.set_index("plot_uid", drop=False)
    dev = F2.year_deviation(plots, target, train_plots)
    ov = F2.other_year_mean(dev, P)
    g = pd.DataFrame({"var_dev": ov["mean"].values, "var_n": ov["n"].values}, index=P.index)
    g["mk"] = F2.marker_prediction(P, dev, markers, alpha).values
    return g.join(F2.marker_pcs(P, markers, train_plots))


def _fit(kind, cfg, X, y):
    if kind == "lin":
        sc = StandardScaler().fit(X)
        m = Ridge(**cfg).fit(sc.transform(X), y)
        return lambda Z: m.predict(sc.transform(Z))
    keep = ~np.isnan(X).all(0)  # a feature with no value in any training row carries nothing (e.g. height before day
    m = HistGradientBoostingRegressor(**GBM_FIXED, **cfg).fit(X[:, keep], y)  # 120 exists only in 2017 and 2019)
    return lambda Z: m.predict(Z[:, keep])


def design_matrix(feat, kind, cols=FEATS):
    X = feat[cols].copy()
    if kind == "lin":  # ridge: missing -> 0 (centred features; 0 = year mean) plus flags
        X["var_missing"] = X.var_dev.isna().astype(float)
        X = X.fillna(0.0)
    return X.values.astype(np.float64)


def features(plots, series, weather, gdd, markers, train_plots, alphas_w1):
    """Per target and cutoff: feature table (plots x FEATS) using training plots for everything fitted."""
    train_plots = set(train_plots)
    tr = series[series.plot_uid.isin(train_plots)]
    curves = {t: Curve(tr[tr.trait == t], "gdd", 50) for t in ["canopy_cover", "height"]}
    ins = {c: inseason(plots, series, c, curves) for c in F.CUTOFFS}
    out = {}
    for target in F.TARGETS:
        gen = genetics(plots, target, train_plots, markers, alphas_w1["final"][f"{target}_markers"])
        for c in F.CUTOFFS:
            if target == "heading_das" and c > F.HEADING_MAX_CUTOFF:
                continue
            out[(target, c)] = ins[c].join(gen)
    return out


def fit_predict(plots, feats, weather, gdd, train_plots, train_years, choice, cols=FEATS, suffix=""):
    """choice: {(target, cutoff): [(kind, cfg), ...]} candidates to fit (one in normal runs, all in tuning).
    Returns long table: plot_uid, target, cutoff, method, pred (report units), truth."""
    train_plots = set(train_plots)
    P = plots.set_index("plot_uid")
    rows = []
    for (target, c), feat in feats.items():
        scale = F.TARGETS[target]
        feat = feat.reindex(P.index)
        trm = P.index.isin(train_plots) & P[target].notna().values
        ymean = P[trm].groupby("year")[target].mean()
        level0 = ymean.mean()
        y = (P.loc[trm, target] - P.loc[trm, "year"].map(ymean)).values
        levels = {"L0": np.full(len(P), level0)}
        if target == "heading_das":
            levels["L1"] = heading_gdd_level(plots, gdd, weather, train_plots, train_years, c)[0].reindex(P.index).values
            if not suffix:
                rows.append(pd.DataFrame({"plot_uid": P.index, "target": target, "cutoff": c, "method": "L1",
                                          "pred": levels["L1"] * scale}))
        for i, (kind, cfg) in enumerate(choice[(target, c)]):
            X = design_matrix(feat, kind, cols)
            d = _fit(kind, cfg, X[trm], y)(X)
            name = "D" if len(choice[(target, c)]) == 1 else f"D_{i}"
            for lv, lvals in levels.items():
                rows.append(pd.DataFrame({"plot_uid": P.index, "target": target, "cutoff": c,
                                          "method": f"{lv}+{name}{suffix}", "pred": (lvals + d) * scale}))
    out = pd.concat(rows, ignore_index=True)
    truth = plots.melt(id_vars=["plot_uid"], value_vars=list(F.TARGETS), var_name="target", value_name="truth")
    truth["truth"] = truth.truth * truth.target.map(F.TARGETS)
    return out.merge(truth, on=["plot_uid", "target"]).dropna(subset=["pred", "truth"])
