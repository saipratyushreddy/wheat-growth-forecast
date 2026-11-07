"""
Report figures (docs/05 part C), made from saved results only; no fit. Writes docs/figures/*.png.

  python src/figures.py
"""
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "docs/figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TRAITS = {"canopy_cover": "Canopy cover (points)", "height": "Height (cm)"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight"})


def _zero(ax, horizontal=True):
    (ax.axhline if horizontal else ax.axvline)(0, color=MUTED, lw=0.8)


def near_vs_n3():
    """Week 3, leave-one-year-out: E+P minus N3 per held-out year, with the pooled difference and interval."""
    d = pd.read_csv("outputs/week3/loyo_near.csv")
    d = d[(d.method == "E+P") & (d.ref == "N3")]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, (t, label) in zip(axes, TRAITS.items()):
        x = d[d.trait == t]
        for i, h in enumerate([7, 14, 28]):
            y = x[(x.h == h) & (x.set != "pooled")]
            ax.scatter(np.full(len(y), i) + np.linspace(-0.15, 0.15, len(y)), y.d, s=22, color=BLUE,
                       edgecolor="white", lw=0.8, zorder=3, label="one held-out year" if i == 0 else None)
            p = x[(x.h == h) & (x.set == "pooled")].iloc[0]
            ax.errorbar(i + 0.32, p.d, yerr=[[p.d - p.d_lo], [p.d_hi - p.d]], fmt="o", ms=5, color=ORANGE,
                        capsize=0, lw=2, label="pooled, 95% interval" if i == 0 else None)
        _zero(ax)
        ax.set_xticks([0, 1, 2], ["+7 d", "+14 d", "+28 d"])
        ax.set_title(label, loc="left", fontsize=9, color=INK)
    axes[0].set_ylabel("MAE difference, E+P minus N3\n(below 0: the learned model is better)")
    axes[1].legend(loc="lower left", fontsize=8)
    fig.savefig(f"{OUT}/near_future_vs_N3.png")
    plt.close(fig)


def ranges():
    """Week 3 part C: coverage of the 80% near-future ranges (pooled version, E) per held-out year."""
    d = pd.read_csv("outputs/week3/loyo_ranges_near.csv")
    d = d[(d.method == "E") & (d.level == 0.8) & (d.version == "pooled")]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, (t, label) in zip(axes, TRAITS.items()):
        x = d[d.trait == t]
        for i, h in enumerate([7, 14, 28]):
            y = x[x.h == h].sort_values("year")
            ax.scatter(np.full(len(y), i) + np.linspace(-0.2, 0.2, len(y)), 100 * y.coverage, s=22, color=BLUE,
                       edgecolor="white", lw=0.8, zorder=3)
            for _, r in y[y.coverage < 0.7].iterrows():
                ax.annotate(str(r.year), (i + np.linspace(-0.2, 0.2, len(y))[list(y.year).index(r.year)],
                                          100 * r.coverage), xytext=(5, -2), textcoords="offset points",
                            fontsize=7, color=MUTED)
        ax.axhline(80, color=ORANGE, lw=1.2, ls="--")
        ax.set_xticks([0, 1, 2], ["+7 d", "+14 d", "+28 d"])
        ax.set_title(label, loc="left", fontsize=9, color=INK)
    axes[0].set_ylabel("Share of outcomes inside the 80% range (%)\n(dashed: the 80% target; dot: one held-out year)")
    fig.savefig(f"{OUT}/ranges_coverage.png")
    plt.close(fig)


def soybean():
    """Week 4 part A: per-season MAE difference against N3, for E and the within-field E-w."""
    d = pd.read_csv("outputs/week4/soy_near.csv")
    d = d[d.set.str.startswith("season") & (d.ref == "N3")]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), sharey=True)
    for ax, h in zip(axes, [7, 14, 28]):
        for j, (m, c) in enumerate([("E", BLUE), ("E-w", ORANGE)]):
            y = d[(d.h == h) & (d.method == m)].sort_values("set")
            xs = np.arange(len(y)) + (j - 0.5) * 0.36
            ax.bar(xs, y.d, width=0.34, color=c, label={"E": "E (plain)", "E-w": "E-w (within-field)"}[m])
        _zero(ax)
        ax.set_xticks(range(7), [s[-2:] for s in sorted(d.set.unique())])
        ax.set_title(f"+{h} days", loc="left", fontsize=9, color=INK)
        ax.set_xlabel("season (20xx)")
    axes[0].set_ylabel("MAE difference vs N3 (cover points)\n(below 0: better than N3)")
    axes[2].legend(loc="upper right", fontsize=8)
    fig.savefig(f"{OUT}/soybean_vs_N3.png")
    plt.close(fig)


def retrieval():
    """Week 4 part B: mean distance between the retrieved and the true future photo, per method."""
    q = pd.read_parquet("outputs/week4/retrieval_queries.parquet")
    methods = ["R-now", "R-var", "R-N3", "R-E+P", "R-oracle"]
    names = {"R-now": "today's photo", "R-var": "same variety", "R-N3": "match N3 forecast",
             "R-E+P": "match E+P forecast", "R-oracle": "match true cover (ceiling)"}
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), sharex=True)
    for ax, h in zip(axes, [14, 28]):
        m = q[q.h == h][[f"d_{k}" for k in methods]].mean().values
        ax.barh(range(len(methods)), m, height=0.6, color=[ORANGE] + [BLUE] * 4)
        for i, v in enumerate(m):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(range(len(methods)), [names[k] for k in methods])
        ax.invert_yaxis()
        ax.set_xlim(0, 0.72)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"+{h} days", loc="left", fontsize=9, color=INK)
        ax.set_xlabel("distance to the true future photo (lower is better)")
    axes[1].set_yticklabels([])
    fig.savefig(f"{OUT}/retrieval_distance.png")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    near_vs_n3()
    ranges()
    soybean()
    retrieval()
    shutil.copy("outputs/week4/retrieval_examples.png", f"{OUT}/retrieval_examples.png")
    print(sorted(os.listdir(OUT)))
