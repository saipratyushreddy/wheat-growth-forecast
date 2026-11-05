"""
Week-3 prediction ranges (docs/04-week3-design.md, part C): year-held-out split conformal.

Residuals come from inner leave-one-year-out refits on a fit's training years (each residual's own year excluded from
its refit). Half-width q per group: the ceil((n+1)(1-a))/n empirical quantile of |residual|, pooled over training years
(primary) or the largest per-year quantile (worst year, robustness). Ranges are clipped to the physical limits.
"""
import numpy as np
import pandas as pd

LEVELS = [0.8, 0.9]


def conformal_q(abs_res, level):
    r = np.sort(np.asarray(abs_res, float))
    n = len(r)
    if n == 0:
        return np.nan
    k = int(np.ceil((n + 1) * level))
    return np.inf if k > n else r[k - 1]


def half_widths(cal, groups, level):
    """cal: inner residual rows with `res`, `year` and group columns. Returns a table of q per group (both versions)."""
    rows = []
    for key, g in cal.groupby(groups):
        key = key if isinstance(key, tuple) else (key,)
        a = np.abs(g.res.values)
        per_year = [conformal_q(np.abs(x.res.values), level) for _, x in g.groupby("year")]
        rows.append(dict(zip(groups, key), level=level, q_pooled=conformal_q(a, level), q_worst=max(per_year),
                         n_cal=len(a), years_cal=g.year.nunique()))
    return pd.DataFrame(rows)


def apply(pred, q, groups, lo_lim=None, hi_lim=None):
    """pred: rows with `pred`, `truth` and group columns. Adds lo/hi/covered/width for both versions and each level."""
    out = []
    for level, ql in q.groupby("level"):
        d = pred.merge(ql, on=groups, how="left")
        for v in ["pooled", "worst"]:
            lo, hi = d.pred - d[f"q_{v}"], d.pred + d[f"q_{v}"]
            if lo_lim is not None:
                lo = lo.clip(lower=lo_lim(d))
            if hi_lim is not None:
                hi = np.minimum(hi, hi_lim(d))
            d[f"lo_{v}"], d[f"hi_{v}"] = lo, hi
            d[f"cov_{v}"] = (d.truth >= lo) & (d.truth <= hi)
            d[f"width_{v}"] = hi - lo
        out.append(d)
    return pd.concat(out, ignore_index=True)


def summarize(ranged, by):
    """Coverage and mean width per `by` group, level and version."""
    rows = []
    for key, g in ranged.groupby(by + ["level"]):
        for v in ["pooled", "worst"]:
            rows.append(dict(zip(by + ["level"], key), version=v, n=len(g), coverage=float(g[f"cov_{v}"].mean()),
                             width=float(g[f"width_{v}"].replace(np.inf, np.nan).mean()),
                             infinite=int(np.isinf(g[f"width_{v}"]).sum())))
    return pd.DataFrame(rows)


def calibrated(per_year, level, tol_mean=0.03, tol_worst=0.10):
    """The design's rule: mean coverage over held-out years within 3 points of the level, and no year more than 10 points
    below it."""
    c = per_year.coverage.values
    return bool(abs(c.mean() - level) <= tol_mean and c.min() >= level - tol_worst)
