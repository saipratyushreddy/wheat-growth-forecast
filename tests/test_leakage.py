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
