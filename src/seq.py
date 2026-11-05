"""
Week-2 sequence model S (docs/03-week2-design.md): a GRU over each plot-year's canopy-cover and height measurements,
started from the plot's genetics, read out after the last step dated on or before the issue date, predicting the residual
from N3. L1 loss, ensemble of seeds. CPU only.

Inputs come from near2.near_features(); anomalies use the training average curves rebuilt here from training plots.
"""
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

import near as N
from common import Curve

H = 64
STATIC = ["var_dev_hf", "var_dev_hd", "mk_hf", "mk_hd"] + [f"pc{i}" for i in range(10)]
QUERY = ["gdd_inc", "rise", "days_to_target", "var_res"]
EPOCH_GRID = [10, 25, 50]
SEEDS = 5


class Net(nn.Module):
    def __init__(self, n_step, n_static, n_query):
        super().__init__()
        self.h0 = nn.Linear(n_static, H)
        self.gru = nn.GRU(n_step, H, batch_first=True)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Sequential(nn.Linear(H + n_query + n_static, 64), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, steps, static, qb, qt, qf):
        h0 = torch.tanh(self.h0(static)).unsqueeze(0)
        out, _ = self.gru(steps, h0)  # causal: out[:, t] depends on steps[:, :t+1] only
        hid = self.drop(out[qb, qt])
        return self.head(torch.cat([hid, qf, static[qb]], 1)).squeeze(1)


class SeqData:
    """Per plot-year padded step tensors, static genetics, and the samples (queries) that point into them."""

    def __init__(self, feat, series, train_plots):
        train_plots = set(train_plots)
        tr = series[series.plot_uid.isin(train_plots)]
        curves = {t: Curve(tr[tr.trait == t], "gdd", 50) for t in N.TRAITS}
        s = series[series.trait.isin(N.TRAITS) & series.plot_uid.isin(set(feat.plot_uid))].copy()
        s["anom"] = np.nan
        for t, c in curves.items():
            m = s.trait == t
            s.loc[m, "anom"] = s.loc[m, "value"].values.astype(float) - c(s.loc[m, "gdd"].values.astype(float))
        s = s.sort_values(["plot_uid", "das", "trait"])
        s["dt"] = s.groupby("plot_uid").das.diff().fillna(0.0)
        self.uids = np.array(sorted(s.plot_uid.unique()))
        self.row = {u: i for i, u in enumerate(self.uids)}
        lens = s.groupby("plot_uid").size().reindex(self.uids).values
        T = int(lens.max())
        X = np.zeros((len(self.uids), T, 6), np.float32)
        D = np.full((len(self.uids), T), 1e9, np.float32)  # das per step, padding sorts last
        pos = s.groupby("plot_uid").cumcount().values
        r = s.plot_uid.map(self.row).values
        X[r, pos, 0] = (s.trait == "canopy_cover").values
        X[r, pos, 1] = (s.trait == "height").values
        X[r, pos, 2] = s.anom.values
        X[r, pos, 3] = s.das.values / 300.0
        X[r, pos, 4] = s.gdd.values / 3000.0
        X[r, pos, 5] = np.log1p(s.dt.values)
        D[r, pos] = s.das.values
        self.X, self.D = torch.from_numpy(X), D

        # static genetics per plot: standardised on training plots, missing -> 0 with flags
        per = feat.groupby("plot_uid")[STATIC + ["genotyped"]].first().reindex(self.uids)
        trm = per.index.isin(train_plots)
        mu, sd = per.loc[trm, STATIC].mean(), per.loc[trm, STATIC].std().replace(0, 1)
        st = ((per[STATIC] - mu) / sd).fillna(0.0)
        st["genotyped"] = per.genotyped.values
        st["vdev_missing"] = per.var_dev_hf.isna().astype(float).values
        self.static = torch.from_numpy(st.values.astype(np.float32))

        # queries
        q = feat[["plot_uid", "trait", "issue_das"] + QUERY].copy()
        self.qplot = q.plot_uid.map(self.row).values
        # last step dated on or before the issue date (all steps of a plot are sorted by das)
        self.qstep = np.array([np.searchsorted(self.D[p], d, side="right") - 1 for p, d in
                               zip(self.qplot, q.issue_das.values)])
        qf = pd.DataFrame({"is_cc": (q.trait == "canopy_cover").astype(float).values,
                           "is_h": (q.trait == "height").astype(float).values,
                           "gdd_inc": q.gdd_inc.values / 1000.0, "rise": q.rise.values,
                           "days": q.days_to_target.values / 28.0,
                           "var_res": q.var_res.fillna(0.0).values * 10,
                           "var_res_missing": q.var_res.isna().astype(float).values})
        self.qf = torch.from_numpy(qf.values.astype(np.float32))
        self.y = torch.from_numpy(feat.res.values.astype(np.float32))
        self.index = feat.index


def train_predict(data, train_plots, epochs=EPOCH_GRID, seeds=SEEDS, batch=64, log=None):
    """Train on train_plots' queries; returns {epoch: residual predictions for all queries}, averaged over seeds."""
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
    train_plots = set(train_plots)
    tr_rows = np.array([i for i, u in enumerate(data.uids) if u in train_plots])
    by_plot = pd.Series(np.arange(len(data.qplot))).groupby(data.qplot).apply(np.array).to_dict()
    tr_rows = np.array([p for p in tr_rows if p in by_plot])
    out = {e: np.zeros(len(data.qplot)) for e in epochs}
    for seed in range(seeds):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = Net(data.X.shape[2], data.static.shape[1], data.qf.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        for ep in range(1, max(epochs) + 1):
            net.train()
            perm = rng.permutation(tr_rows)
            tot = 0.0
            for b in range(0, len(perm), batch):
                plots = perm[b:b + batch]
                qi = np.concatenate([by_plot[p] for p in plots])
                loss = _loss(net, data, plots, qi)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(qi)
            if log:
                log(f"seed {seed} epoch {ep} train L1 {tot / sum(len(by_plot[p]) for p in tr_rows):.4f}")
            if ep in out:
                out[ep] += _predict(net, data, by_plot) / seeds
    return {e: pd.Series(v, index=data.index) for e, v in out.items()}


def _forward(net, data, plots, qi):
    local = {p: i for i, p in enumerate(plots)}
    T = int(data.qstep[qi].max()) + 1  # later steps are never read; truncating is exact for a causal GRU
    qb = torch.from_numpy(np.array([local[p] for p in data.qplot[qi]]))
    qt = torch.from_numpy(data.qstep[qi])
    pt = torch.from_numpy(np.asarray(plots))
    return net(data.X[pt, :T], data.static[pt], qb, qt, data.qf[qi])


def _loss(net, data, plots, qi):
    return torch.abs(_forward(net, data, plots, qi) - data.y[qi]).mean()


@torch.no_grad()
def _predict(net, data, by_plot, batch=256):
    net.eval()
    pred = np.zeros(len(data.qplot))
    plots = np.array(sorted(by_plot))
    for b in range(0, len(plots), batch):
        ps = plots[b:b + batch]
        qi = np.concatenate([by_plot[p] for p in ps])
        pred[qi] = _forward(net, data, ps, qi).numpy()
    return pred


def to_forecast(feat, res):
    """Clipped N3 + predicted residual."""
    out = pd.Series(np.nan, index=feat.index)
    for trait, d in feat.groupby("trait"):
        out.loc[d.index] = N._clip(trait, d.N3.values + res.loc[d.index].values)
    return out
