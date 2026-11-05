"""
Data audit for FIP 1.0: what is actually in the tables, before any modelling decision.

Reports: plots/genotypes/years per split and their overlap; units and methods of each trait; per plot-year counts of images and of
each repeated trait (with the days after sowing they cover); availability and spread of the final outcomes per year; marker and
weather coverage. Writes outputs/audit_*.csv and prints a summary.

Usage:  python src/audit.py --root data/fip1 --out outputs
"""
import argparse
import os

import numpy as np
import pandas as pd

from load_fip import FINALS, SERIES, WEATHER, load


def n(x):
    return 0 if x is None else len(x)


def das(dates, sow):
    if dates is None or len(dates) == 0 or pd.isna(sow):
        return np.array([])
    return (pd.to_datetime(pd.Series(dates)) - pd.Timestamp(sow)).dt.days.values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/fip1")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    d = load(a.root, markers=True, weather=True)
    print(f"rows {len(d)}, columns {d.shape[1]}")
    print("crop types:", d.crop_type.value_counts().to_dict())

    # --- splits: size and overlap
    g = d.groupby("split").agg(plots=("plot_uid", "nunique"), genotypes=("genotype_id", "nunique"),
                               years=("harvest_year", lambda s: sorted(s.unique().tolist())))
    print("\n== splits\n", g)
    tr = d[d.split == "train"]
    for s in ["validation", "test_plot", "test_genotype", "test_environment", "test_genotype_environment"]:
        t = d[d.split == s]
        print(f"{s:28s} genotypes also in train: {t.genotype_id.isin(tr.genotype_id).mean():.2f}; "
              f"years also in train: {sorted(set(t.harvest_year) & set(tr.harvest_year))}; plot_uid in train: {t.plot_uid.isin(tr.plot_uid).mean():.2f}")
    print("plots per year:", d.groupby("harvest_year").plot_uid.nunique().to_dict())
    print("sowing dates:", d.groupby("harvest_year").sowing_date.agg(["min", "max"]).astype(str).to_dict("index"))

    # --- units and methods
    rows = []
    for t in SERIES + FINALS:
        for f in ["trait_name", "method_name", "si_unit"]:
            c = f"{t}_{f}"
            if c in d:
                rows.append({"trait": t, "field": f, "values": "; ".join(map(str, d[c].dropna().unique()[:4]))})
    units = pd.DataFrame(rows).pivot(index="trait", columns="field", values="values")
    print("\n== traits, methods, units\n", units.to_string())
    units.to_csv(os.path.join(a.out, "audit_units.csv"))

    # --- repeated series per plot-year
    per = pd.DataFrame({"plot_uid": d.plot_uid, "year": d.harvest_year, "split": d.split,
                        "n_images": d.images.map(n), "n_inner_images": d.inner_plot_images.map(n)})
    for t in SERIES:
        per[f"n_{t}"] = d[f"{t}_values"].map(n)
        dd = [das(x, s) for x, s in zip(d[f"{t}_dates"], d.sowing_date)]
        per[f"{t}_first_das"] = [x.min() if len(x) else np.nan for x in dd]
        per[f"{t}_last_das"] = [x.max() if len(x) else np.nan for x in dd]
    di = [das(x, s) for x, s in zip(d.image_dates, d.sowing_date)]
    per["img_first_das"] = [x.min() if len(x) else np.nan for x in di]
    per["img_last_das"] = [x.max() if len(x) else np.nan for x in di]
    per["img_unique_dates"] = [len(set(x)) for x in di]
    per.to_csv(os.path.join(a.out, "audit_per_plot.csv"), index=False)
    cols = ["n_images", "n_inner_images", "img_unique_dates"] + [f"n_{t}" for t in SERIES]
    print("\n== per plot-year counts (median [min-max]) by year")
    s = per.groupby("year")[cols].agg(lambda x: f"{int(np.median(x))} [{int(x.min())}-{int(x.max())}]")
    print(s.to_string())
    print("\n== DAS coverage by year (median first / last)")
    cov = per.groupby("year")[[c for c in per if c.endswith("_das")]].median().round(0)
    print(cov.to_string())
    print("share of plot-years with >=1 value:", {t: round((per[f'n_{t}'] > 0).mean(), 2) for t in SERIES})

    # --- final outcomes
    rows = []
    for t in FINALS:
        v = d[f"{t}_value"].astype(float)
        for y, gg in d.assign(v=v).groupby("harvest_year"):
            x = gg.v.dropna()
            rows.append({"trait": t, "year": y, "n": len(x), "share": round(len(x) / len(gg), 2),
                         "mean": x.mean(), "sd": x.std(), "min": x.min(), "max": x.max(),
                         "heritability": gg[f"{t}_heritability"].astype(float).dropna().mean()})
    fin = pd.DataFrame(rows)
    fin.to_csv(os.path.join(a.out, "audit_finals.csv"), index=False)
    print("\n== final outcomes per year\n", fin.round(3).to_string(index=False))
    hd = d.heading_value.astype(float)
    print("heading_value example values:", hd.dropna().head(5).tolist(), "| heading_date examples:", d.heading_date.dropna().astype(str).head(3).tolist())

    # --- markers and weather
    m = d.marker_biallelic_codes.map(n)
    print("\n== markers: plot-years with markers", round((m > 0).mean(), 3), "| marker length", sorted(m[m > 0].unique())[:5],
          "| genotypes with markers", d.loc[m > 0, "genotype_id"].nunique(), "of", d.genotype_id.nunique())
    for w in WEATHER:
        k = d[f"{w}_values"].map(n)
        print(f"weather {w:24s} share {(k > 0).mean():.2f}, median points {int(k.median())}")
    one = d.iloc[0]
    t = pd.to_datetime(pd.Series(one.temperature_air_200cm_dates))
    print("weather example: first", t.min().date(), "last", t.max().date(), "| times[:3]", list(one.temperature_air_200cm_times[:3]))
    print("weather identical within a year?",
          d.groupby("harvest_year").temperature_air_200cm_values.apply(lambda s: len({len(x) for x in s})).to_dict())


if __name__ == "__main__":
    main()
