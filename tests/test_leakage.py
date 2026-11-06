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


def test_d_gbm_handles_feature_missing_in_all_training_rows():
    rng = np.random.default_rng(0)
    X = np.c_[rng.normal(size=(200, 2)), np.full(200, np.nan)]
    Z = np.c_[rng.normal(size=(20, 2)), rng.normal(size=20)]  # present at prediction time, absent in training
    p = FD._fit("gbm", dict(max_leaf_nodes=7, max_iter=20), X, rng.normal(size=200))(Z)
    assert p.shape == (20,) and np.isfinite(p).all()


# ---------------------------------------------------------------- week 3 (docs/04-week3-design.md)

def test_photo_features_match_recomputation():
    """Stored features of a few photos equal a fresh one-photo recomputation (guards against misordered batches).
    Needs data/photos (Swan); skipped elsewhere."""
    import glob as _glob
    if not _glob.glob("data/photos/features_*.parquet"):
        pytest.skip("photo features not built here")
    import io as _io
    import photos as PH
    from huggingface_hub import hf_hub_download
    from PIL import Image
    d, X = PH.load_features()
    enc = PH.Encoder()
    rng = np.random.default_rng(0)
    import tempfile as _tf
    with _tf.TemporaryDirectory() as tmp:
        for i in rng.choice(len(d), 3, replace=False):
            f = hf_hub_download(PH.REPO, d.path.iloc[i], repo_type="dataset", revision=PH.REVISION, local_dir=tmp)
            c, m = enc([PH.Encoder.preprocess(Image.open(f))])
            assert np.allclose(np.r_[c[0], m[0]], X[i], atol=1e-4)


def test_photo_index_covers_tables():
    import photos as PH
    if not os.path.exists(PH.INDEX):
        pytest.skip("photo index not built")
    idx = pd.read_parquet(PH.INDEX)
    plots, _, _ = load_derived()
    assert len(idx) == 151150 and idx.path.is_unique and set(idx.plot_uid) == set(plots.plot_uid)
    assert idx.groupby("plot_uid").shard.nunique().max() == 1 and idx.shard.nunique() == PH.N_SHARDS


def _fake_photos(plots, n_per=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for uid, sow in zip(plots.plot_uid, plots.sowing_date):
        for d in sorted(rng.choice(np.arange(10, 280), n_per, replace=False)):
            rows.append((uid, sow + pd.Timedelta(days=int(d))))
    meta = pd.DataFrame(rows, columns=["plot_uid", "date"])
    return meta, rng.normal(size=(len(meta), 40)).astype(np.float32)


def test_photo_inputs_dated_on_or_before_issue(data):
    import photo_feats as PF
    plots, series, weather, smp = data
    meta, X = _fake_photos(plots)
    ph = PF.photo_pcs(meta, X, plots, plots.plot_uid[plots.split == "train"])
    s = smp.sample(500, random_state=0)
    got = PF.near_inputs(s, ph)
    for (i, r), (_, g) in zip(s.iterrows(), got.iterrows()):
        days = ph[(ph.plot_uid == r.plot_uid) & (ph.das <= r.issue_das)]
        if len(days) == 0:
            assert np.isnan(g.ph_age)
            continue
        last = days.sort_values("das").iloc[-1]
        assert g.ph_age == r.issue_das - last.das >= 0
        assert np.allclose(g[PF.PC].values.astype(float), last[PF.PC].values.astype(float))
    raw = PF._asof(pd.DataFrame({"plot_uid": plots.plot_uid.values, "cut": 150}), "cut", ph, PF.PC)
    assert (raw.ph_das.dropna() <= 150).all()


def test_photo_pca_ignores_test_plots(data):
    import photo_feats as PF
    plots, *_ = data
    meta, X = _fake_photos(plots)
    train = plots.plot_uid[plots.split == "train"]
    a = PF.photo_pcs(meta, X, plots, train)
    keep = meta.plot_uid.isin(set(plots.plot_uid[plots.split.isin(["train", "validation"])])).values
    b = PF.photo_pcs(meta[keep], X[keep], plots, train)
    m = a.merge(b, on=["plot_uid", "das"], suffixes=("_a", "_b"))
    assert len(m) == len(b)
    assert np.allclose(m[[c + "_a" for c in PF.PC]].values, m[[c + "_b" for c in PF.PC]].values, atol=1e-4)


def test_week3_units_hold_out_correctly():
    import run_week3 as W3
    plots, _, _ = load_derived()
    U = W3.units(plots)
    assert len(U) == 42
    for outer, inner in U:
        train, years, pred = W3.unit_plots(plots, outer, inner)
        P = plots.set_index("plot_uid")
        tr_years = set(P.loc[train, "year"])
        assert not (set(train) & pred)
        if outer != "official":
            assert outer not in tr_years and (inner is not None or set(P.loc[list(pred), "year"]) == {outer})
        if inner is not None:
            assert inner not in tr_years and set(P.loc[list(pred), "year"]) == {inner}
            assert outer != inner and (outer == "official" or outer not in set(P.loc[list(pred), "year"]))
        if outer == "official":
            assert set(P.loc[train, "split"]) == {"train"}


def test_conformal_quantile():
    import ranges as RG
    r = np.arange(1, 11, dtype=float)  # n = 10
    assert RG.conformal_q(r, 0.8) == 9.0  # ceil(11 * 0.8) = 9th smallest
    assert RG.conformal_q(r, 0.9) == 10.0 and RG.conformal_q(r[:5], 0.9) == np.inf
    cal = pd.DataFrame({"g": 1, "year": [1] * 10 + [2] * 10, "res": np.r_[r, r * 2]})
    q = RG.half_widths(cal, ["g"], 0.8)
    assert q.q_worst.iloc[0] == 18.0 and q.q_pooled.iloc[0] == RG.conformal_q(np.r_[r, r * 2], 0.8)


# ---------------------------------------------------------------- week 4 (docs/05-week4-design.md)

@pytest.fixture(scope="module")
def soydata():
    import soy as SOY
    if not os.path.exists(f"{SOY.OUT}/near_samples.parquet"):
        pytest.skip("soybean tables not built (python src/soy.py build)")
    return SOY.load()


def test_soy_weather_by_calendar_date(soydata):
    import soy as SOY
    plots, series, weather, smp = soydata
    # each season's weather is that calendar year, and the sowing day falls in it
    assert (weather.date.dt.year == weather.year).all()
    sow = plots.groupby("year").sowing_date.first()
    assert all(((weather.year == y) & (weather.date == d)).any() for y, d in sow.items())
    # the calendar uses the soybean tables where they cover a day (spot check: a day in the 2018 season)
    cal = pd.read_parquet(f"{SOY.OUT}/site_weather.parquet")
    assert cal.set_index("date").loc["2018-06-15", "source"] == "soybean"


def test_soy_inputs_use_no_later_weather(soydata):
    import json

    import run_week4 as W
    plots, series, weather, smp = soydata
    alphas = json.load(open("outputs/week1/official_alphas.json"))["near"]
    y = 2019
    keep = set(plots.plot_uid[plots.year.isin([2017, 2018, y])])
    p, s = plots[plots.plot_uid.isin(keep)], series[series.plot_uid.isin(keep)]
    q = smp[smp.plot_uid.isin(keep)]
    train = list(p.plot_uid[p.year != y])
    cut = pd.Timestamp(f"{y}-07-01")
    w2 = weather.copy()
    w2.loc[(w2.year == y) & (w2.date > cut), "tmean"] += 10.0
    a = W.soy_features(q, s, weather, p, train, [2017, 2018], alphas)
    b = W.soy_features(q, s, w2, p, train, [2017, 2018], alphas)
    m = (a.year == y) & (a.issue_date <= cut)
    cols = ["N3", "res", "issue_gdd", "gdd_inc", "a", "a7", "a14", "rise"]
    pd.testing.assert_frame_equal(a.loc[m, cols], b.loc[m, cols])


def test_soy_variety_features_from_other_seasons(soydata):
    import json

    import run_week4 as W
    plots, series, weather, smp = soydata
    alphas = json.load(open("outputs/week1/official_alphas.json"))["near"]
    keep = set(plots.plot_uid[plots.year.isin([2018, 2019, 2020])])
    p, s, q = plots[plots.plot_uid.isin(keep)], series[series.plot_uid.isin(keep)], smp[smp.plot_uid.isin(keep)]
    train = list(p.plot_uid[p.year != 2020])
    a = W.soy_features(q, s, weather, p, train, [2018, 2019], alphas)
    p2 = p.copy()
    p2.loc[p2.year == 2019, "yield"] *= 3.0  # own-season yields of 2019 plots must not reach their variety features
    b = W.soy_features(q, s, weather, p2, train, [2018, 2019], alphas)
    m = a.year == 2019
    assert np.allclose(a.loc[m, "var_dev_y"], b.loc[m, "var_dev_y"], equal_nan=True)
    m = a.year == 2020  # the held-out season sees the training seasons only
    p3 = p.copy()
    p3.loc[p3.year == 2020, "yield"] = 0.0
    c = W.soy_features(q, s, weather, p3, train, [2018, 2019], alphas)
    assert np.allclose(a.loc[m, "var_dev_y"], c.loc[m, "var_dev_y"], equal_nan=True)


def test_within_field_uses_no_targets():
    import run_week4 as W
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"year": rng.choice([2016, 2017], n), "h": rng.choice([7, 28], n),
                       "issue_das": rng.choice([40, 60], n), "N3": rng.uniform(0.2, 0.8, n),
                       "E": rng.uniform(0.2, 0.8, n), "target": rng.uniform(0, 1, n)})
    a = W.within(df, "E")
    b = W.within(df.assign(target=rng.uniform(0, 1, n)), "E")
    assert np.allclose(a, b)
    # within each field-day group the mean correction to N3 is zero (before clipping)
    g = (df.E - (df.E - df.N3).groupby([df.year, df.h, df.issue_das]).transform("mean")) - df.N3
    assert np.allclose(g.groupby([df.year, df.h, df.issue_das]).mean(), 0, atol=1e-12)


def test_retrieval_never_uses_own_year_or_later_photos():
    import run_week4 as W
    rng = np.random.default_rng(0)
    rows = []
    for y in [2016, 2017, 2018]:
        for k in range(20):
            for d in range(30, 200, 9):
                rows.append((f"P{y}_{k}", y, k % 5, d, f"{y}/{k}/{d}"))
    meta = pd.DataFrame(rows, columns=["plot_uid", "year", "genotype_id", "das", "path"])
    meta["row"] = np.arange(len(meta))
    cand = meta.assign(cover=rng.uniform(0, 1, len(meta)))
    q = meta[meta.year == 2017].groupby("plot_uid").head(1)[["plot_uid", "year", "genotype_id"]].copy()
    q["issue_das"], q["target_das"] = 102, 129
    q["N3"], q["E+P"], q["target"] = 0.5, 0.6, 0.7
    q["qid"] = np.arange(len(q))
    r = W.retrieve(q, cand, meta[meta.year == 2017])
    for k in ["R-var", "R-N3", "R-E+P", "R-oracle"]:
        assert (meta.year.values[r[k].astype(int)] != 2017).all()
    assert (meta.das.values[r["R-now"].astype(int)] <= 102).all()
    assert (meta.year.values[r["R-now"].astype(int)] == 2017).all()
    assert (np.abs(meta.das.values[r["future"].astype(int)] - 129) <= 3).all()
