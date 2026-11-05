"""Leakage checks required by docs/02-baselines-design.md before any test number. Needs data/derived (src/series.py)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import final as F  # noqa: E402
import near as N  # noqa: E402
from common import load_derived  # noqa: E402
from series import gdd_table  # noqa: E402


@pytest.fixture(scope="module")
def data():
    plots, series, weather = load_derived()
    series = series.merge(plots[["plot_uid", "genotype_id"]], on="plot_uid")
    rng = np.random.default_rng(0)
    keep = set(rng.choice(plots.plot_uid.values, 300, replace=False))
    keep |= set(plots.plot_uid[plots.split == "train"].values[:600])
    keep |= set(plots.plot_uid[plots.split == "validation"].values[:150])
    p, s = plots[plots.plot_uid.isin(keep)], series[series.plot_uid.isin(keep)]
    return p, s, weather, N.build_samples(p, s)


def test_splits_disjoint():
    plots, _, _ = load_derived()
    assert plots.plot_uid.is_unique and plots.split.notna().all()


def test_near_inputs_from_past_only(data):
    plots, series, _, smp = data
    row = smp.sample(400, random_state=1)
    for r in row.itertuples():
        g = series[(series.plot_uid == r.plot_uid) & (series.trait == r.trait) & (series.das <= r.issue_das)].sort_values("das")
        assert r.target_das > r.issue_das and abs(r.target_das - r.issue_das - r.h) <= N.TOL
        assert np.isclose(r.last, g.value.values[-1])
        assert np.isclose(r.v14, np.interp(r.issue_das - 14, g.das.values, g.value.values.astype(float)))


def test_near_fit_ignores_test_plots(data):
    plots, series, weather, smp = data
    train = plots.plot_uid[plots.split == "train"]
    val = plots.plot_uid[plots.split == "validation"]
    yrs = sorted(plots.year[plots.split == "train"].unique())
    a, _ = N.fit_predict(smp, series, weather, train, val, yrs)
    other = set(plots.plot_uid[~plots.split.isin(["train", "validation"])])
    b, _ = N.fit_predict(smp[~smp.plot_uid.isin(other)], series[~series.plot_uid.isin(other)], weather, train, val, yrs)
    cols = ["N0", "N1", "N2", "N3", "N4", "N5"]
    va = a[a.plot_uid.isin(set(val))].sort_values(["plot_uid", "trait", "h", "issue_das"])[cols].values
    vb = b[b.plot_uid.isin(set(val))].sort_values(["plot_uid", "trait", "h", "issue_das"])[cols].values
    assert np.allclose(va, vb, equal_nan=True)


def test_final_features_ignore_later_rows(data):
    plots, series, weather, _ = data
    gdd = gdd_table(weather, plots.groupby("year").sowing_date.first().to_dict())
    for c in [120, 180]:
        a = F.cutoff_features(plots, series, gdd, c)
        b = F.cutoff_features(plots, series[series.das <= c], gdd, c)
        pd.testing.assert_frame_equal(a, b)


def test_final_fit_ignores_test_plots(data):
    plots, series, weather, _ = data
    gdd = gdd_table(weather, plots.groupby("year").sowing_date.first().to_dict())
    feats = {c: F.cutoff_features(plots, series, gdd, c) for c in [150]}
    train = plots.plot_uid[plots.split == "train"]
    val = plots.plot_uid[plots.split == "validation"]
    a, _ = F.fit_predict(plots, feats, train, val)
    keep = plots.split.isin(["train", "validation"])
    b, _ = F.fit_predict(plots[keep], {c: f.loc[f.index.isin(plots.plot_uid[keep])] for c, f in feats.items()}, train, val)
    k = ["plot_uid", "target", "cutoff", "method"]
    m = a.merge(b, on=k, suffixes=("_a", "_b"))
    assert len(m) == len(b) and np.allclose(m.pred_a, m.pred_b)


# ---------------------------------------------------------------- week 2 (docs/03-week2-design.md)

import json  # noqa: E402

import features2 as FE  # noqa: E402
import final2 as FD  # noqa: E402
import near2 as N2  # noqa: E402
import seq as SQ  # noqa: E402
from common import load_markers  # noqa: E402


@pytest.fixture(scope="module")
def w2data(data):
    plots, series, weather, smp = data
    markers = load_markers()
    w2 = pd.read_parquet("data/derived/weather2.parquet")
    alphas = json.load(open("outputs/week1/official_alphas.json"))
    train = plots.plot_uid[plots.split == "train"]
    yrs = sorted(plots.year[plots.split == "train"].unique())
    feat = N2.near_features(smp, series, weather, w2, plots, markers, train, yrs, alphas)
    return plots, series, weather, smp, markers, w2, alphas, train, yrs, feat


def test_other_year_mean_excludes_own_year():
    tr = pd.DataFrame({"genotype_id": [1, 1, 1, 2], "year": [2016, 2016, 2017, 2017], "val": [1.0, 3.0, 10.0, 5.0]})
    q = pd.DataFrame({"genotype_id": [1, 1, 1, 2, 3], "year": [2016, 2017, 2019, 2017, 2016]})
    r = FE.other_year_mean(tr, q)
    assert np.allclose(r["mean"].values[:3], [10.0, 2.0, 14 / 3]) and np.isnan(r["mean"].values[3:]).all()
    assert list(r["n"]) == [1, 2, 3, 0, 0]


def test_variety_features_from_other_years_and_training_only(w2data):
    plots, series, weather, smp, markers, w2, alphas, train, yrs, feat = w2data
    # recompute var_dev_hf by brute force for a few plots
    P = plots.set_index("plot_uid")
    tr = P.loc[P.index.isin(set(train))].dropna(subset=["height_final"])
    dev = tr.height_final - tr.groupby("year").height_final.transform("mean")
    for uid in feat.plot_uid.drop_duplicates().sample(50, random_state=0):
        g, y = P.loc[uid, "genotype_id"], P.loc[uid, "year"]
        other = dev[(tr.genotype_id == g) & (tr.year != y)]
        got = feat.loc[feat.plot_uid == uid, "var_dev_hf"].iloc[0]
        assert (np.isnan(got) and len(other) == 0) or np.isclose(got, other.mean(), atol=1e-5)


def test_marker_ridge_weighted_equals_plot_rows():
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(0)
    ids, M = list(range(8)), rng.integers(0, 3, (8, 40)).astype(np.float32)
    plots = pd.DataFrame({"genotype_id": rng.integers(0, 8, 30), "year": rng.choice([2016, 2017, 2018], 30)})
    dev = pd.DataFrame({"genotype_id": plots.genotype_id, "year": plots.year, "val": rng.normal(size=30)})
    got = FE.marker_prediction(plots, dev, (ids, M), 10.0)
    for y in [2016, 2017]:
        tr = dev[dev.year != y]
        m = Ridge(alpha=10.0).fit(M[tr.genotype_id], tr.val)
        q = plots.year == y
        assert np.allclose(got[q], m.predict(M[plots.genotype_id[q]]), atol=1e-4)


def test_marker_pcs_fit_on_training_varieties_only(w2data):
    plots, series, weather, smp, markers, w2, alphas, train, yrs, feat = w2data
    P = plots.set_index("plot_uid", drop=False)
    a = FE.marker_pcs(P, markers, train)
    ids, M = markers
    tr_g = set(plots.genotype_id[plots.plot_uid.isin(set(train))])
    keep = [i for i, g in enumerate(ids) if g in tr_g]
    b = FE.marker_pcs(P, ([ids[i] for i in keep], M[keep]), train)  # test-only varieties removed from the markers
    both = a.index[P.genotype_id.isin(tr_g) & (a.genotyped == 1)]
    assert np.allclose(np.abs(a.loc[both].values), np.abs(b.loc[both].values), atol=1e-4)  # sign of a PC is arbitrary


def test_weather_to_date_uses_no_later_day(w2data):
    w2, yrs = w2data[5], w2data[8]
    assert 2019 not in yrs  # 2019 is outside the climatology, so perturbing its later days isolates the window
    cut = pd.Timestamp("2019-03-15")
    w_mod = w2.copy()
    w_mod.loc[(w_mod.year == 2019) & (w_mod.date > cut), ["tmean", "precip", "irr", "hum"]] = 999.0
    a, b = FE.weather_to_date(w2, yrs), FE.weather_to_date(w_mod, yrs)
    m = (a.year == 2019) & (a.date <= cut)
    assert m.sum() > 100
    assert np.allclose(a.loc[m, FE.WEATHER_COLS].values, b.loc[m, FE.WEATHER_COLS].values, equal_nan=True)
    assert not np.allclose(a.loc[(a.year == 2019) & ~m, FE.WEATHER_COLS].fillna(0).values,
                           b.loc[(b.year == 2019) & ~m, FE.WEATHER_COLS].fillna(0).values)


def test_gru_causal(w2data):
    plots, series, weather, smp, markers, w2, alphas, train, yrs, feat = w2data
    import torch
    sd = SQ.SeqData(feat, series, train)
    assert (sd.D[sd.qplot, sd.qstep] <= feat.issue_das.values).all()
    torch.manual_seed(0)
    net = SQ.Net(sd.X.shape[2], sd.static.shape[1], sd.qf.shape[1]).eval()
    plots_b = np.arange(16)
    qi = np.concatenate([np.nonzero(sd.qplot == p)[0] for p in plots_b])
    with torch.no_grad():
        a = SQ._forward(net, sd, plots_b, qi).numpy()
        X0 = sd.X.clone()
        later = torch.from_numpy(sd.D[:16] > np.array([sd.D[p, 0] + 30 for p in plots_b])[:, None])
        sd.X[:16][later] = 99.0  # perturb steps more than 30 days after each plot's first measurement
        b = SQ._forward(net, sd, plots_b, qi).numpy()
        sd.X = X0
    early = feat.issue_das.values[qi] <= np.array([sd.D[p, 0] + 30 for p in sd.qplot[qi]])
    assert early.any() and np.allclose(a[early], b[early], atol=1e-6) and not np.allclose(a[~early], b[~early])


def test_g_fit_ignores_test_plots(w2data):
    plots, series, weather, smp, markers, w2, alphas, train, yrs, feat = w2data
    other = set(plots.plot_uid[~plots.split.isin(["train", "validation"])])
    cfg = dict(max_leaf_nodes=7, max_iter=20, min_samples_leaf=100)
    sub = smp[~smp.plot_uid.isin(other)]
    feat_b = N2.near_features(sub, series[~series.plot_uid.isin(other)], weather, w2,
                              plots[~plots.plot_uid.isin(other)], markers, train, yrs, alphas)
    for t in ["canopy_cover", "height"]:
        fa, fb = feat[feat.trait == t], feat_b[feat_b.trait == t]
        pa, pb = N2.fit_g(fa, train, cfg), N2.fit_g(fb, train, cfg)
        ka = fa.assign(p=pa)[~fa.plot_uid.isin(other)].sort_values(["plot_uid", "h", "issue_das"])
        kb = fb.assign(p=pb).sort_values(["plot_uid", "h", "issue_das"])
        cols = N2.FEATS_W
        assert np.allclose(ka[cols].values, kb[cols].values, equal_nan=True)
        assert np.allclose(ka.p.values, kb.p.values)


def test_d_ignores_test_targets_and_later_measurements(w2data):
    plots, series, weather, smp, markers, w2, alphas, train, yrs, feat = w2data
    gdd = gdd_table(weather, plots.groupby("year").sowing_date.first().to_dict())
    cut = 150
    test = set(plots.plot_uid[~plots.split.isin(["train", "validation"])])
    choice = {("height_final", cut): [("lin", dict(alpha=10))], ("heading_das", cut): [("gbm", dict(
        max_leaf_nodes=7, max_iter=30))]}

    def run(p, s):
        ff = FD.features(p, s, weather, gdd, markers, train, alphas)
        ff = {k: v for k, v in ff.items() if k in choice}
        return FD.fit_predict(p, ff, weather, gdd, train, yrs, choice).set_index(["plot_uid", "target", "method"]).pred

    a = run(plots, series)
    p2 = plots.copy()
    for t in ["heading", "height_final", "yield", "yield_adjusted", "protein", "heading_das"]:
        p2.loc[p2.plot_uid.isin(test), t] = np.nan
    s2 = series[~(series.plot_uid.isin(test) & (series.das > cut))]
    b = run(p2, s2)
    common_idx = b.index  # rows without truth are dropped, so compare training/validation rows
    assert len(common_idx) > 0 and np.allclose(a.loc[common_idx].values, b.values)
    # predictions for the test plots themselves are also unchanged (truth only joins after prediction)
    ff_a = FD.features(plots, series, weather, gdd, markers, train, alphas)[("height_final", cut)]
    ff_b = FD.features(p2, s2, weather, gdd, markers, train, alphas)[("height_final", cut)]
    pd.testing.assert_frame_equal(ff_a, ff_b)


def test_weighted_spearman_matches_scipy():
    import run_week2 as R
    from scipy.stats import spearmanr
    rng = np.random.default_rng(0)
    x, t = np.round(rng.normal(size=60), 1), rng.normal(size=60)  # rounding creates ties
    W = rng.integers(0, 3, (5, 60)).astype(float)
    got = R._wpearson(R._wranks(x, W), R._wranks(t, W), W)
    for k in range(5):
        rep = np.repeat(np.arange(60), W[k].astype(int))
        assert np.isclose(got[k], spearmanr(x[rep], t[rep]).statistic)
