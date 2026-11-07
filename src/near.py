"""
Near-future task (docs/02-baselines-design.md): forecast canopy cover / height 7, 14, 28 days after an issue date.

build_samples() makes one row per plot x trait x issue date x horizon with history features computed from measurements on
or before the issue date only. fit_predict() fits baselines N0-N5 on training plots and returns predictions for all rows.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from common import ClimTemp, Curve, season_day

TRAITS = ["canopy_cover", "height"]
HORIZONS = [7, 14, 28]
TOL = 3
LIMITS = {"canopy_cover": (0.0, 1.0), "height": (0.0, None)}
ALPHAS = [0.1, 1, 10, 100, 1000]
N5_FEATURES = ["last", "v7", "v14", "slope14", "n2_issue", "n2_target", "issue_gdd", "gdd_inc", "issue_das"]


def build_samples(plots, series):
    meta = plots.set_index("plot_uid")[["year", "genotype_id", "split"]]
    out = []
    for (uid, trait), g in series[series.trait.isin(TRAITS)].groupby(["plot_uid", "trait"], sort=False):
        g = g.sort_values("das")
        das, gdd, val = g.das.values, g.gdd.values.astype(float), g.value.values.astype(float)
        dates = g.date.values
        if len(das) < 2:
            continue
        for h in HORIZONS:
            for i in range(1, len(das)):
                want = das[i] + h
                cand = np.where((das > das[i]) & (np.abs(das - want) <= TOL))[0]
                if len(cand) == 0:
                    continue
                j = cand[np.argmin(np.abs(das[cand] - want) * 10 + (das[cand] > want))]  # nearest; ties -> earlier
                hd, hv = das[: i + 1], val[: i + 1]
                v7 = np.interp(das[i] - 7, hd, hv)
                v14 = np.interp(das[i] - 14, hd, hv)
                out.append((uid, trait, h, das[i], dates[i], gdd[i], val[i], v7, v14, (val[i] - v14) / 14.0, i + 1,
                            das[j], dates[j], val[j], gdd[j]))
    cols = ["plot_uid", "trait", "h", "issue_das", "issue_date", "issue_gdd", "last", "v7", "v14", "slope14", "n_hist",
            "target_das", "target_date", "target", "target_gdd_true"]
    s = pd.DataFrame(out, columns=cols)
    s = s.join(meta, on="plot_uid")
    s["issue_sday"] = season_day(s.issue_date, s.year)
    s["target_sday"] = season_day(s.target_date, s.year)
    s["phase"] = pd.cut(s.issue_das, [-1, 150, 200, 10_000], labels=["autumn-winter", "spring", "heading-ripening"])
    return s


def _clip(trait, x):
    lo, hi = LIMITS[trait]
    return np.clip(x, lo, hi)


def fit_predict(samples, series, weather, train_plots, val_plots, train_years, alphas=None, future="clim"):
    """Fit N0-N5 on train_plots; choose N5 ridge strength on val_plots unless `alphas` is given. Returns (pred, alphas).
    `series` must carry genotype_id and year. Only rows of train_plots are used to fit anything.
    future="true" (week 5 upper bound only, docs/07): target degree days from the actual weather, not climatology."""
    train_plots, val_plots = set(train_plots), set(val_plots)
    clim = ClimTemp(weather, train_years)
    s = samples.copy()
    s["target_gdd"] = s.issue_gdd + clim.increment(s.issue_sday.values, s.target_sday.values)  # no future weather
    if future == "true":
        s["target_gdd"] = s.target_gdd_true.astype(float)  # breaks the input rule on purpose: an upper bound
    else:
        assert future == "clim"
    s["gdd_inc"] = s.target_gdd - s.issue_gdd
    tr_series = series[series.plot_uid.isin(train_plots)]
    chosen = {} if alphas is None else dict(alphas)
    parts = []
    for trait, st in s.groupby("trait"):
        ts = tr_series[tr_series.trait == trait]
        c_das, c_gdd = Curve(ts, "das", 7), Curve(ts, "gdd", 50)
        st = st.copy()
        st["N0"] = st["last"]
        st["N1"] = c_das(st.target_das)
        st["n2_issue"], st["n2_target"] = c_gdd(st.issue_gdd), c_gdd(st.target_gdd)
        st["N2"] = st.n2_target
        st["N3"] = st.n2_target + (st["last"] - st.n2_issue)

        # N4: variety's own average curve (training plots), anomaly persistence; N3 where unavailable
        st["N4"] = st.N3
        for gid, vs in ts.groupby("genotype_id"):
            idx = st.index[st.genotype_id == gid]
            if len(idx) == 0:
                continue
            cv = Curve(vs, "gdd", 100)
            if len(cv.grid) < 2:
                continue
            a, b = st.loc[idx, "issue_gdd"].values, st.loc[idx, "target_gdd"].values
            ok = (a >= cv.grid[0] - 100) & (b <= cv.grid[-1] + 100)
            st.loc[idx[ok], "N4"] = cv(b[ok]) + (st.loc[idx[ok], "last"].values - cv(a[ok]))

        # N5: ridge on history features, target = change from last value; one model per horizon
        st["N5"] = np.nan
        for h, sh in st.groupby("h"):
            tr = sh[sh.plot_uid.isin(train_plots)]
            sc = StandardScaler().fit(tr[N5_FEATURES])
            Xtr, ytr = sc.transform(tr[N5_FEATURES]), (tr.target - tr["last"]).values
            key = f"{trait}_h{h}"
            if key not in chosen:
                va = sh[sh.plot_uid.isin(val_plots)]
                Xva = sc.transform(va[N5_FEATURES])
                maes = {a: np.mean(np.abs(_clip(trait, va["last"].values + Ridge(alpha=a).fit(Xtr, ytr).predict(Xva))
                                          - va.target.values)) for a in ALPHAS}
                chosen[key] = min(maes, key=maes.get)
            m = Ridge(alpha=chosen[key]).fit(Xtr, ytr)
            st.loc[sh.index, "N5"] = sh["last"].values + m.predict(sc.transform(sh[N5_FEATURES]))
        for k in ["N0", "N1", "N2", "N3", "N4", "N5"]:
            st[k] = _clip(trait, st[k].values)
        parts.append(st)
    return pd.concat(parts).sort_index(), chosen
