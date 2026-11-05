"""
Week-3 photo inputs (docs/04-week3-design.md, part B): standardise + PCA (32 components) fit on training plots' photos
only; per sample, PCs of the latest photo dated on or before the issue date (or cutoff), its age, and the change in PCs
since the latest photo on or before 14 days earlier. Same-day photos of a plot are averaged.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

N_PC = 32
PC = [f"ph{i}" for i in range(N_PC)]
DPC = [f"dph{i}" for i in range(N_PC)]
NEAR_COLS = PC + DPC + ["ph_age"]
FINAL_COLS = PC + ["ph_age"]


def photo_pcs(meta, X, plots, train_plots):
    """meta: photo rows (plot_uid, date); X: (n, 1536). Returns one row per plot x day: plot_uid, das, ph0..ph31."""
    tr = meta.plot_uid.isin(set(train_plots)).values
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    pca = PCA(n_components=N_PC, svd_solver="covariance_eigh", random_state=0).fit((X[tr] - mu) / sd)
    Z = pca.transform((X - mu) / sd).astype(np.float32)
    Z /= Z[tr].std(0)
    sow = plots.set_index("plot_uid").sowing_date
    d = pd.DataFrame(Z, columns=PC)
    d["plot_uid"] = meta.plot_uid.values
    d["das"] = (pd.to_datetime(meta.date.values) - meta.plot_uid.map(sow).values).days
    return d.groupby(["plot_uid", "das"], as_index=False)[PC].mean()


def _asof(q, day, ph, cols):
    """For rows of q, values of the latest photo with das <= q[day] for the same plot (NaN if none)."""
    left = pd.DataFrame({"plot_uid": q.plot_uid.values, "day": q[day].values.astype(np.int64),
                         "row": np.arange(len(q))}).sort_values("day")
    right = ph.rename(columns={"das": "ph_das"}).assign(ph_das=lambda x: x.ph_das.astype(np.int64)).sort_values(
        "ph_das")
    m = pd.merge_asof(left, right, left_on="day", right_on="ph_das", by="plot_uid", direction="backward")
    return m.sort_values("row")[cols + ["ph_das"]].reset_index(drop=True)


def near_inputs(samples, ph):
    """Photo inputs per near-future sample (aligned to samples' index)."""
    now = _asof(samples, "issue_das", ph, PC)
    back = _asof(samples.assign(back=samples.issue_das - 14), "back", ph, PC)
    out = pd.DataFrame(now[PC].values, columns=PC, index=samples.index)
    out[DPC] = now[PC].values - back[PC].values
    out["ph_age"] = samples.issue_das.values - now.ph_das.values
    return out


def final_inputs(plots, ph, cutoff):
    """Photo inputs at a cutoff, per plot (index plot_uid): PCs centred within year, and age."""
    q = pd.DataFrame({"plot_uid": plots.plot_uid.values, "cut": cutoff})
    now = _asof(q, "cut", ph, PC)
    out = pd.DataFrame(now[PC].values, columns=PC, index=pd.Index(plots.plot_uid.values, name="plot_uid"))
    out["ph_age"] = cutoff - now.ph_das.values
    year = plots.year.values
    out[FINAL_COLS] = out[FINAL_COLS] - out[FINAL_COLS].groupby(year).transform("mean")
    return out
