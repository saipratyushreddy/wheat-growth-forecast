"""
Week-2 near-future models (docs/03-week2-design.md): features for residual-from-N3 models, and G / G+W
(gradient-boosted trees, absolute-error loss, one model per trait, horizons pooled).

near_features() recomputes the week-1 baselines (N0-N5, ridge strengths frozen) on the given training plots and adds
the week-2 inputs. Everything fitted (curves, variety features, marker models, PCs, weather climatology) uses train_plots
or train_years only.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import features2 as F2
import near as N
from common import Curve
from series import gdd_table

HISTORY = ["a", "a7", "a14", "aslope14", "dt_prev", "n_hist", "rise", "o_a", "o_age"]
TIMING = ["issue_das", "issue_gdd", "gdd_inc", "days_to_target", "h"]
GENETICS = ["var_res", "var_res_n", "var_dev_hf", "var_dev_hd", "mk_hf", "mk_hd"] + \
    [f"pc{i}" for i in range(F2.N_PCS)] + ["genotyped"]
FEATS_G = HISTORY + TIMING + GENETICS
FEATS_W = FEATS_G + F2.WEATHER_COLS
G_GRID = [dict(max_leaf_nodes=l, max_iter=it, min_samples_leaf=m) for l in (15, 63) for it in (200, 600)
          for m in (100, 1000)]
G_FIXED = dict(loss="absolute_error", learning_rate=0.05, l2_regularization=1.0, early_stopping=False, random_state=0)
OTHER = {"canopy_cover": "height", "height": "canopy_cover"}


def _gdd_at(plots, weather):
    """(year, das) -> degree days so far, from the actual weather up to that day."""
    sow = plots.groupby("year").sowing_date.first()
    g = gdd_table(weather[weather.year.isin(sow.index)], sow.to_dict())
    g["das"] = (g.date - g.year.map(sow)).dt.days
    return g.set_index(["year", "das"]).gdd


def history_features(samples, series, weather, plots, train_plots, train_years, alphas_near, traits=N.TRAITS,
                     other=True):
    """Baselines N0-N5, the residual target `res` (target - N3), and the measurement-history and timing inputs.
    Shared by wheat (near_features) and soybean (week 4: canopy cover only, other=False)."""
    train_plots = set(train_plots)
    pred, _ = N.fit_predict(samples, series, weather, train_plots, [], train_years, alphas_near)
    pred = pred.copy()
    pred["res"] = pred.target - pred.N3
    pred["a"] = pred["last"] - pred.n2_issue
    pred["rise"] = pred.n2_target - pred.n2_issue
    pred["days_to_target"] = pred.target_das - pred.issue_das
    gat = _gdd_at(plots, weather)
    tr_series = series[series.plot_uid.isin(train_plots)]
    curves = {t: Curve(tr_series[tr_series.trait == t], "gdd", 50) for t in traits}

    for trait, idx in pred.groupby("trait").groups.items():
        c = curves[trait]
        d = pred.loc[idx]
        for lag, col in [(7, "a7"), (14, "a14")]:
            g = gat.reindex(pd.MultiIndex.from_arrays([d.year.values, (d.issue_das - lag).values])).values
            pred.loc[idx, col] = d[f"v{lag}"].values - c(np.where(np.isnan(g), d.issue_gdd.values, g))
    pred["aslope14"] = (pred.a - pred.a14) / 14.0

    # days since the previous measurement of the same trait (n_hist counts measurements up to the issue date)
    s = series[series.trait.isin(traits)].sort_values(["plot_uid", "trait", "das"])
    s = s.assign(pos=s.groupby(["plot_uid", "trait"]).cumcount())
    prev = s.set_index(["plot_uid", "trait", "pos"]).das
    pd_prev = prev.reindex(pd.MultiIndex.from_arrays([pred.plot_uid, pred.trait, pred.n_hist - 2])).values
    pred["dt_prev"] = pred.issue_das.values - pd_prev

    # the other trait's latest anomaly on or before the issue date
    if not other:
        return pred
    pred["o_a"], pred["o_age"] = np.nan, np.nan
    for trait, oth in OTHER.items():
        q = pred[pred.trait == trait][["plot_uid", "issue_das"]].reset_index().sort_values("issue_das")
        o = s[s.trait == oth][["plot_uid", "das", "gdd", "value"]].sort_values("das")
        o = o.assign(oa=o.value.astype(float) - curves[oth](o.gdd.values.astype(float)))
        m = pd.merge_asof(q, o, left_on="issue_das", right_on="das", by="plot_uid", direction="backward")
        pred.loc[m["index"].values, "o_a"] = m.oa.values
        pred.loc[m["index"].values, "o_age"] = (m.issue_das - m.das).values
    return pred


def near_features(samples, series, weather, w2, plots, markers, train_plots, train_years, alphas_w1):
    """Returns the samples with baselines N0-N5, the residual target `res` (target - N3) and all week-2 inputs."""
    pred = history_features(samples, series, weather, plots, train_plots, train_years, alphas_w1["near"])
    train_plots = set(train_plots)

    # genetics: variety history from other years' training plots; markers fit on other years
    tr = pred[pred.plot_uid.isin(train_plots)]
    per_plot = tr.groupby(["plot_uid", "trait", "h"], as_index=False).agg(val=("res", "mean"), genotype_id=(
        "genotype_id", "first"), year=("year", "first"))
    vr = F2.other_year_mean(per_plot, pred, by=("trait", "h"))
    pred["var_res"], pred["var_res_n"] = vr["mean"].values, vr["n"].values
    P = plots.set_index("plot_uid", drop=False)
    for target, col, mk in [("height_final", "var_dev_hf", "mk_hf"), ("heading_das", "var_dev_hd", "mk_hd")]:
        dev = F2.year_deviation(plots, target, train_plots)
        per = F2.other_year_mean(dev, P)["mean"]
        pred[col] = pred.plot_uid.map(per).values
        mp = F2.marker_prediction(P, dev, markers, alphas_w1["final"][f"{target}_markers"])
        pred[mk] = pred.plot_uid.map(mp).values
    pcs = F2.marker_pcs(P, markers, train_plots)
    pred = pred.join(pcs, on="plot_uid")

    wt = F2.weather_to_date(w2, train_years)
    pred = pred.merge(wt.rename(columns={"date": "issue_date"}), on=["year", "issue_date"], how="left").set_index(
        pred.index)
    return pred


def fit_g(feat, train_plots, cfg, cols=FEATS_G):
    """Fit one G model per trait on train_plots; returns predictions (clipped N3 + residual) for all rows."""
    train_plots = set(train_plots)
    out = pd.Series(np.nan, index=feat.index)
    for trait, d in feat.groupby("trait"):
        tr = d[d.plot_uid.isin(train_plots)]
        m = HistGradientBoostingRegressor(**G_FIXED, **cfg).fit(tr[cols].values.astype(np.float32), tr.res.values)
        r = m.predict(d[cols].values.astype(np.float32))
        out.loc[d.index] = N._clip(trait, d.N3.values + r)
    return out
