"""
Week-2 shared features (docs/03-week2-design.md): variety features from other years only, marker predictions fit on other
years only, marker principal components fit on training varieties only, and weather to date.

Every function takes the training plots explicitly; nothing here looks at a test plot's outcome.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from common import season_day

N_PCS = 10
WEATHER_COLS = ["w_tmean", "w_precip", "w_irr", "w_hum"]


def other_year_mean(train_df, query, by=()):
    """Mean of `val` over training rows of the same variety (and same `by` keys) in other years, and the number of rows.

    train_df: one row per training plot (and `by` keys): genotype_id, year, val (+ by).
    query: rows with genotype_id, year (+ by). Returns a DataFrame (mean, n) aligned to query's index; NaN / 0 if none.
    """
    by = list(by)
    keys = by + ["genotype_id"]
    t = train_df.dropna(subset=["val"])
    own = t.groupby(keys + ["year"]).val.agg(["sum", "count"])
    tot = t.groupby(keys).val.agg(["sum", "count"])
    q = query[keys + ["year"]].copy()
    q = q.join(tot, on=keys).join(own.add_suffix("_own"), on=keys + ["year"])
    s = q["sum"].fillna(0.0) - q["sum_own"].fillna(0.0)
    n = q["count"].fillna(0) - q["count_own"].fillna(0)
    return pd.DataFrame({"mean": np.where(n > 0, s / n.where(n > 0, 1), np.nan), "n": n.astype(float)}, index=query.index)


def year_deviation(plots, target, train_plots):
    """Training plots' deviation from their year's training mean: genotype_id, year, val."""
    t = plots[plots.plot_uid.isin(set(train_plots))].dropna(subset=[target])
    return pd.DataFrame({"plot_uid": t.plot_uid.values, "genotype_id": t.genotype_id.values, "year": t.year.values,
                         "val": (t[target] - t.groupby("year")[target].transform("mean")).values})


def marker_prediction(plots, dev, markers, alpha):
    """Marker (F2-style) prediction of a variety's deviation, per plot, fit only on training plots of other years.

    Ridge on plot rows is fit as the equivalent ridge on variety means weighted by plot counts (same solution).
    dev: output of year_deviation (training plots only). Returns a Series over plots.index (NaN if not genotyped)."""
    ids, M = markers
    pos = {g: i for i, g in enumerate(ids)}
    out = pd.Series(np.nan, index=plots.index)
    d = dev[dev.genotype_id.isin(pos)]
    for y in sorted(plots.year.unique()):
        tr = d[d.year != y]
        agg = tr.groupby("genotype_id").val.agg(["mean", "count"])
        X = M[[pos[g] for g in agg.index]]
        w = agg["count"].values.astype(float)
        mu = (X * w[:, None]).sum(0) / w.sum()  # mean over plot rows, as Ridge on plot rows would centre
        m = Ridge(alpha=alpha).fit(X - mu, agg["mean"].values, sample_weight=w)
        q = plots.index[(plots.year == y) & plots.genotype_id.isin(pos)]
        if len(q):
            out.loc[q] = m.predict(M[[pos[g] for g in plots.loc[q, "genotype_id"]]] - mu)
    return out


def marker_pcs(plots, markers, train_plots):
    """Top N_PCS marker principal components, fit on genotyped varieties that have a training plot. Per plot (NaN if
    not genotyped), plus a genotyped flag."""
    ids, M = markers
    pos = {g: i for i, g in enumerate(ids)}
    tr_g = sorted(set(plots.genotype_id[plots.plot_uid.isin(set(train_plots))]) & set(pos))
    pca = PCA(n_components=N_PCS, random_state=0).fit(M[[pos[g] for g in tr_g]])
    Z = pca.transform(M)
    Z = Z / Z[[pos[g] for g in tr_g]].std(0)
    out = pd.DataFrame(np.nan, index=plots.index, columns=[f"pc{i}" for i in range(N_PCS)])
    has = plots.genotype_id.isin(pos)
    out.loc[has] = Z[[pos[g] for g in plots.genotype_id[has]]]
    out["genotyped"] = has.astype(float)
    return out


def weather_to_date(w2, train_years, window=14):
    """Per year x date: weather over the `window` days ending on that date (inclusive), minus the training years'
    average of the same statistic on the same season day. Uses no day after the date."""
    w = w2.sort_values(["year", "date"]).copy()
    w["sday"] = season_day(w.date, w.year)
    g = w.groupby("year")
    roll = pd.DataFrame({
        "w_tmean": g.tmean.transform(lambda x: x.rolling(window, min_periods=window).mean()),
        "w_precip": g.precip.transform(lambda x: x.rolling(window, min_periods=window).sum()),
        "w_irr": g.irr.transform(lambda x: x.rolling(window, min_periods=window).mean()),
        "w_hum": g.hum.transform(lambda x: x.rolling(window, min_periods=window).mean()),
    })
    roll[["year", "date", "sday"]] = w[["year", "date", "sday"]]
    clim = roll[roll.year.isin(list(train_years))].groupby("sday")[WEATHER_COLS].mean()
    out = roll[["year", "date"]].copy()
    out[WEATHER_COLS] = roll[WEATHER_COLS].values - clim.reindex(roll.sday.values).values
    return out
