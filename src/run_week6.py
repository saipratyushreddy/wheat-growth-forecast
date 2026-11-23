"""
Week-6 runner (docs/08-generation-design.md): generate the future photo of a plot. GPU, on Swan.

  python src/run_week6.py prep YEAR           # VAE latents + DINOv2 features of that year's thumbnails (data/gen)
  python src/run_week6.py estimator FOLD      # cover read-out for the fold: train on the other years, gate, read all photos
  python src/run_week6.py train FOLD --steps N        # fine-tune the generator on the other years' photo pairs
  python src/run_week6.py generate FOLD [--limit N]   # Gen, Gen-persist, Gen-oracle for the fold's queries + read-out + features
  python src/run_week6.py score               # once: all comparisons, the verdict, figures

WEEK6_TAG=smoke sends checkpoints and outputs to separate directories (code checks, never results).
"""
import argparse
import glob
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

import gen_data as GD

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT, RUN, GEN = GD.OUT, GD.RUN, GD.GEN
FIXED_OUT = "outputs/week6"  # the estimators and their gates are shared by smoke runs and the real run
NOISE_QUERIES, NOISE_SAMPLES = 84, 3  # per fold (about 500 in all), extra samples of Gen
BATCH_GEN = 64


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def device():
    import torch
    assert torch.cuda.is_available(), "this step needs a GPU"
    return torch.device("cuda")


def load_base():
    from common import load_derived
    plots, series, _ = load_derived()
    return plots, series, GD.photo_meta(plots)


def all_latents(meta):
    """(n, 4, 20, 30) float16 latents of every photo in meta order, and the DINOv2 thumbnail features."""
    lat, dino = [], []
    for y in GD.YEARS:
        m = meta[meta.year == y]
        assert (pd.read_parquet(f"{GEN}/paths_{y}.parquet").path.values == m.path.values).all(), f"paths differ {y}"
        lat.append(np.load(f"{GEN}/latents_{y}.npy"))
        dino.append(np.load(f"{GEN}/dino_{y}.npy"))
    return np.concatenate(lat), np.concatenate(dino)


def measurements(plots, series, meta):
    f = f"{GEN}/measure.parquet"
    if os.path.exists(f):
        m = pd.read_parquet(f)
        assert (m.path.values == meta.path.values).all()
        return m.cover.values, m.height.values
    cov, hei = GD.photo_measurements(meta, GD.trait_lookup(series, "canopy_cover"), GD.trait_lookup(series, "height"))
    tmp = f"{f}.{os.getpid()}.tmp"  # the array tasks of one job may all get here at once: write whole, then rename
    pd.DataFrame({"path": meta.path.values, "cover": cov, "height": hei}).to_parquet(tmp, index=False)
    os.replace(tmp, f)
    return cov, hei


# ---------------------------------------------------------------- prep

class Dino:
    """DINOv2 on the GPU, same weights, preprocessing and layout as week 3 ([CLS | mean patch], 1536)."""

    def __init__(self, dev):
        import photos as PH
        from transformers import AutoModel
        self.dev = dev
        self.m = AutoModel.from_pretrained(PH.DINO, revision=PH.DINO_REVISION).to(dev).eval()

    @staticmethod
    def inputs(arr):
        """uint8 images (n, 160, 240, 3) -> float32 (n, 3, 280, 420), as photos.Encoder.preprocess does for one image."""
        import photos as PH
        from PIL import Image
        return np.stack([PH.Encoder.preprocess(Image.fromarray(a)) for a in arr])

    def __call__(self, x):
        import torch
        with torch.no_grad():
            h = self.m(pixel_values=torch.as_tensor(x).to(self.dev)).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu().numpy()

    def features(self, arr, batch=64):
        return np.concatenate([self(self.inputs(arr[i:i + batch])) for i in range(0, len(arr), batch)])


class PrepSet:
    """Thumbnail jpegs -> (uint8 image tensor, DINOv2 input). A module-level class, so a data-loader worker can be started
    by any multiprocessing method (Python 3.14 defaults to forkserver, which pickles the dataset)."""

    def __init__(self, jpegs):
        self.jpegs = jpegs

    def __len__(self):
        return len(self.jpegs)

    def __getitem__(self, i):
        import torch
        a = GD.decode(self.jpegs[i])
        return torch.from_numpy(a), Dino.inputs(a[None])[0]


def prep(year):
    import gen_model as GM
    if os.path.exists(f"{GEN}/dino_{year}.npy"):
        return log(year, "already done")
    os.makedirs(GEN, exist_ok=True)
    plots, series, meta = load_base()
    m = meta[meta.year == year]
    th = GD.load_thumbs(year, m.path.values)
    jp = [th[p] for p in m.path.values]
    assert len(jp) == len(m), "a thumbnail is missing"
    dev = device()
    vae, dino = GM.load_vae().to(dev), Dino(dev)

    import torch
    dl = torch.utils.data.DataLoader(PrepSet(jp), batch_size=128, num_workers=8, multiprocessing_context="fork")
    lat, feat = [], []
    for k, (x, pre) in enumerate(dl):
        lat.append(GM.encode(vae, x.to(dev)).half().cpu().numpy())
        feat.append(dino(pre))
        if k % 20 == 0:
            log(year, k * 128, "/", len(jp))
    pd.DataFrame({"path": m.path.values}).to_parquet(f"{GEN}/paths_{year}.parquet", index=False)
    np.save(f"{GEN}/latents_{year}.npy", np.concatenate(lat))
    np.save(f"{GEN}/dino_{year}.npy", np.concatenate(feat))
    log(year, "done", len(jp))


# ---------------------------------------------------------------- read-out estimator

def estimator(fold):
    import torch

    import readout as RO
    os.makedirs(GEN, exist_ok=True)
    os.makedirs(FIXED_OUT, exist_ok=True)
    if os.path.exists(f"{FIXED_OUT}/gate_{fold}.json"):
        return log(fold, "already done")
    plots, series, meta = load_base()
    cov, _ = measurements(plots, series, meta)
    dev = device()
    th = {}
    for y in GD.YEARS:
        th.update(GD.load_thumbs(y))
    tr = GD.estimator_rows(meta, cov, fold)
    log(fold, "estimator training photos", int(tr.sum()))
    est = RO.train([th[p] for p in meta.path.values[tr]], cov[tr], dev, workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)),
                   log=log)
    torch.save(est.state_dict(), f"{GEN}/estimator_{fold}.pt")
    hat = RO.predict_jpegs(est, [th[p] for p in meta.path.values], dev, workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    pd.DataFrame({"path": meta.path.values, "cover_hat": hat}).to_parquet(f"{GEN}/readout_all_{fold}.parquet", index=False)
    ho = (meta.year == fold).values & ~np.isnan(cov)
    g = RO.gate(hat[ho], cov[ho])
    json.dump(g, open(f"{FIXED_OUT}/gate_{fold}.json", "w"))
    log(fold, "gate on real held-out photos:", g)


# ---------------------------------------------------------------- generator

def train_gen(fold, steps):
    import torch

    import gen_model as GM
    if steps is None:
        sys.exit("give --steps: the number of steps is fixed in a clarification of docs/08 first")
    os.makedirs(RUN, exist_ok=True)
    plots, series, meta = load_base()
    cov, hei = measurements(plots, series, meta)
    lat, _ = all_latents(meta)
    tr, mon, _ = GD.fold_split(meta, fold)

    def pairs(mask):
        idx = np.flatnonzero(mask)
        a, b = GD.enumerate_pairs(meta.plot_uid.values[idx], meta.das.values[idx])
        return GD.PairData(lat[idx], meta.das.values[idx], cov[idx], hei[idx], a, b)

    data, mon_data = pairs(tr), pairs(mon)
    log(fold, "training pairs", len(data), "monitor pairs", len(mon_data))
    dev = device()
    GM.train(data, steps, f"{RUN}/gen_{fold}.pt", dev, monitor=mon_data, log=log)
    log(fold, "trained", steps, "steps")


def generate(fold, limit=None):
    import torch

    import gen_model as GM
    import readout as RO
    os.makedirs(RUN, exist_ok=True)
    plots, series, meta = load_base()
    lat, _ = all_latents(meta)
    rowof = dict(zip(meta.path.values, range(len(meta))))
    q, cond = GD.load_queries(series)
    sel = np.flatnonzero((q.year == fold).values)
    if limit:
        sel = sel[:limit]
    dev = device()
    unet, numnet, step = GM.load_trained(f"{RUN}/gen_{fold}.pt", dev)
    log(fold, "generator at step", step, "queries", len(sel))
    vae = GM.load_vae().to(dev)
    est = RO.Estimator().to(dev)
    est.load_state_dict(torch.load(f"{GEN}/estimator_{fold}.pt", map_location="cpu"))
    est.eval()
    dino = Dino(dev)
    rows = np.array([rowof[p] for p in q.now_path.values[sel]])

    def run(c, seed0):
        imgs = []
        for i in range(0, len(sel), BATCH_GEN):
            j = slice(i, i + BATCH_GEN)
            la = torch.from_numpy(lat[rows[j]].astype(np.float32))
            x = GM.sample(unet, numnet, la, torch.from_numpy(c[sel][j]), q.qid.values[sel][j] + seed0, dev)
            imgs.append(GM.decode(vae, x.to(dev)).cpu().numpy())
        return np.concatenate(imgs)

    for vi, v in enumerate(GD.VARIANTS):
        imgs = run(cond[v], 1_000_000 * vi)
        hat = RO.predict_uint8(est, imgs, dev)
        np.savez(f"{RUN}/gen_{fold}_{v}.npz", qid=q.qid.values[sel], images=imgs, cover_hat=hat, feat=dino.features(imgs))
        log(fold, v, "done", len(imgs))
    # sample-to-sample noise of the read-out: extra samples of Gen for a few queries
    rng = np.random.default_rng(fold)
    sub = rng.permutation(len(sel))[:NOISE_QUERIES]
    sel_all, rows_all = sel, rows
    sel, rows = sel_all[sub], rows_all[sub]
    extra = [RO.predict_uint8(est, run(cond["Gen"], 1_000_000 * (10 + k)), dev) for k in range(NOISE_SAMPLES)]
    pd.DataFrame(np.stack(extra, 1), columns=[f"s{k}" for k in range(NOISE_SAMPLES)]).assign(
        qid=q.qid.values[sel]).to_parquet(f"{RUN}/noise_{fold}.parquet", index=False)
    log(fold, "noise check done")


# ---------------------------------------------------------------- scoring

PAIRS_READOUT = [("Gen", "B-now", True), ("Gen", "R-E+P", True), ("Gen-persist", "Gen", False), ("Gen-oracle", "Gen", False)]
PAIRS_DIST = PAIRS_READOUT


def compare(df, pairs, prefix, folds, years_rule=4):
    """Paired mean differences (method - ref) of columns f'{prefix}_{method}', per year and pooled, per horizon."""
    from run_week4 import _boot_mean
    rows = []
    for h, g in df[df.year.isin(folds)].groupby("h"):
        for m, ref, ruled in pairs:
            gg = g.dropna(subset=[f"{prefix}_{m}", f"{prefix}_{ref}"])
            per = {}
            for name, x in [(f"year_{y}", v) for y, v in gg.groupby("year")] + [("pooled", gg)]:
                d, lo, hi = _boot_mean(x[f"{prefix}_{m}"].values - x[f"{prefix}_{ref}"].values, x.genotype_id.values)
                if name != "pooled":
                    per[name] = d
                rows.append(dict(metric=prefix, set=name, h=h, method=m, ref=ref, n=len(x), value=x[f"{prefix}_{m}"].mean(),
                                 value_ref=x[f"{prefix}_{ref}"].mean(), d=d, d_lo=lo, d_hi=hi))
            yb = int(sum(v < 0 for v in per.values()))
            rows[-1].update(years_better=yb, years=len(per), ruled=ruled,
                            better=bool(ruled and rows[-1]["d_hi"] < 0 and yb >= years_rule))
    return pd.DataFrame(rows)


def score():
    if GD.TAG:
        raise SystemExit("score is never run with WEEK6_TAG (docs/08)")
    if os.path.exists(f"{OUT}/generation.csv"):
        raise SystemExit("week 6 already scored; it is run once (docs/08)")
    os.makedirs(OUT, exist_ok=True)
    plots, series, meta = load_base()
    _, dino_all = all_latents(meta)
    q, _ = GD.load_queries(series)
    gates = {y: json.load(open(f"{FIXED_OUT}/gate_{y}.json")) for y in GD.YEARS}
    passed = [y for y in GD.YEARS if gates[y]["passed"]]
    pd.DataFrame(gates).T.rename_axis("fold").to_csv(f"{OUT}/readout_gate.csv")
    log("estimator gate passed in folds", passed)
    rowof = dict(zip(meta.path.values, range(len(meta))))
    years = meta.year.values
    q = q[q.ret_path.notna()].reset_index(drop=True)  # a common query set for every method
    out = []
    for y in GD.YEARS:
        qy = q[q.year == y].copy()
        mu, sd = GD.standardiser(dino_all, years, y)
        z = lambda paths: (dino_all[[rowof[p] for p in paths]] - mu) / sd
        fut = z(qy.future_path.values)
        ro = pd.read_parquet(f"{GEN}/readout_all_{y}.parquet").set_index("path").cover_hat
        for name, col in [("B-now", "now_path"), ("R-E+P", "ret_path")]:
            qy[f"cover_{name}"] = ro[qy[col].values].values
            qy[f"dist_{name}"] = GD.cosine_distance(z(qy[col].values), fut)
        for v in GD.VARIANTS:
            g = np.load(f"{RUN}/gen_{y}_{v}.npz")
            pos = pd.Series(np.arange(len(g["qid"])), index=g["qid"])[qy.qid.values].values
            qy[f"cover_{v}"] = g["cover_hat"][pos]
            qy[f"dist_{v}"] = GD.cosine_distance((g["feat"][pos] - mu) / sd, fut)
        out.append(qy)
    s = pd.concat(out, ignore_index=True)
    for m in ["B-now", "R-E+P"] + GD.VARIANTS:
        s[f"err_{m}"] = (s[f"cover_{m}"] - s.target).abs() * 100  # read-out error, points of cover
    s.to_parquet(f"{OUT}/queries_scored.parquet", index=False)
    r1 = compare(s, PAIRS_READOUT, "err", passed)
    r2 = compare(s, PAIRS_DIST, "dist", GD.YEARS)
    res = pd.concat([r1, r2], ignore_index=True)
    res.to_csv(f"{OUT}/generation.csv", index=False)

    # faithfulness: the read-out on Gen against the condition it was given
    cond_cover = s["E+P"].values
    s["faith"] = (s["cover_Gen"] - cond_cover).abs() * 100
    est_mae = float(np.mean([gates[y]["mae"] for y in passed])) if passed else np.nan
    faith = s[s.year.isin(passed)].groupby("h").faith.mean().rename("mean_abs_diff").to_frame()
    faith["estimator_mae"], faith["bar"] = est_mae, 1.5 * est_mae
    faith["faithful"] = faith.mean_abs_diff <= faith.bar
    faith.to_csv(f"{OUT}/faithfulness.csv")

    # sample-to-sample noise of the read-out
    nz = pd.concat([pd.read_parquet(f"{RUN}/noise_{y}.parquet") for y in GD.YEARS]).merge(
        s[["qid", "cover_Gen"]], on="qid")
    allc = nz[["cover_Gen", "s0", "s1", "s2"]].values * 100
    pd.DataFrame({"queries": [len(nz)], "mean_sd_between_samples": [float(allc.std(1, ddof=1).mean())]}).to_csv(
        f"{OUT}/noise.csv", index=False)

    verdict(res, faith, passed)
    figures(s, meta)
    print(res[res.set == "pooled"].round(4).to_string())


def verdict(res, faith, passed):
    """Pass / partial / fail as fixed in docs/08."""
    def ok(metric, h, ref):
        r = res[(res.metric == metric) & (res.set == "pooled") & (res.h == h) & (res.method == "Gen") & (res.ref == ref)]
        return bool(len(r) and r.better.iloc[0])
    need28 = [ok(m, 28, r) for m in ("err", "dist") for r in ("B-now", "R-E+P")]
    need14 = [ok("err", 14, r) for r in ("B-now", "R-E+P")]
    faithful = bool(faith.faithful.all()) if len(faith) else False
    allmet = all(need28) and all(need14) and faithful
    anymet = any(need28 + need14)
    v = "pass" if allmet else ("partial" if anymet else "fail")
    log("VERDICT:", v, "| 28 d (err/dist vs B-now, R-E+P):", need28, "| 14 d err:", need14, "| faithful:", faithful,
        "| folds passing the estimator gate:", passed)
    json.dump(dict(verdict=v, need28=need28, need14=need14, faithful=faithful, folds_passing_gate=passed),
              open(f"{OUT}/verdict.json", "w"))


def _thumbs_for(paths):
    got = {}
    for f in sorted(glob.glob("data/photos/thumbs_*.parquet")):
        t = pd.read_parquet(f, filters=[("path", "in", list(paths))])
        got.update(dict(zip(t.path, t.jpeg)))
    return got


def figures(s, meta, year=2019, h=28):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    sel = s[(s.year == year) & (s.h == h)].sample(8, random_state=0)
    sel2 = s[(s.year == year) & (s.h == h)].sample(4, random_state=1)
    paths = set(sel[["now_path", "ret_path", "future_path"]].values.ravel()) | set(
        sel2[["now_path", "future_path"]].values.ravel())
    th = _thumbs_for(paths)
    gen = {v: dict(np.load(f"{RUN}/gen_{year}_{v}.npz")) for v in GD.VARIANTS}

    def gimg(v, qid):
        return gen[v]["images"][int(np.flatnonzero(gen[v]["qid"] == qid)[0])]

    def grid(rows, cols, fname):
        w, hh, top = GD.W, GD.H, 18
        img = Image.new("RGB", (len(cols) * w, len(rows) * (hh + top)), "white")
        dr = ImageDraw.Draw(img)
        for i, (_, r) in enumerate(rows.iterrows()):
            for j, (label, fn) in enumerate(cols):
                a, text = fn(r)
                img.paste(Image.fromarray(a), (j * w, i * (hh + top) + top))
                dr.text((j * w + 4, i * (hh + top) + 3), text, fill="black")
        img.save(f"{OUT}/{fname}")

    D = GD.decode
    grid(sel, [("today", lambda r: (D(th[r.now_path]), f"today, day {r.issue_das}")),
               ("Gen", lambda r: (gimg("Gen", r.qid), f"generated, forecast {r['E+P'] * 100:.0f}%, read-out {r.cover_Gen * 100:.0f}%")),
               ("ret", lambda r: (D(th[r.ret_path]), f"retrieved, read-out {r['cover_R-E+P'] * 100:.0f}%")),
               ("true", lambda r: (D(th[r.future_path]), f"actual day {r.target_das}, {r.target * 100:.0f}%"))],
         "generation_examples.png")
    grid(sel2, [("persist", lambda r: (gimg("Gen-persist", r.qid), f"no change, read-out {r['cover_Gen-persist'] * 100:.0f}%")),
                ("Gen", lambda r: (gimg("Gen", r.qid), f"forecast {r['E+P'] * 100:.0f}%, read-out {r.cover_Gen * 100:.0f}%")),
                ("oracle", lambda r: (gimg("Gen-oracle", r.qid), f"true cover {r.target * 100:.0f}%")),
                ("true", lambda r: (D(th[r.future_path]), f"actual day {r.target_das}"))],
         "generation_dose.png")
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.scatter(100 * s["E+P"], 100 * s.cover_Gen, s=3, alpha=0.3, color="#2a78d6")
    ax.plot([0, 100], [0, 100], color="#52514e", lw=0.8)
    ax.set_xlabel("cover given to the generator (%)")
    ax.set_ylabel("cover read from the generated photo (%)")
    fig.savefig(f"{OUT}/generation_faithfulness.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "estimator", "train", "generate", "score"])
    ap.add_argument("fold", nargs="?", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.cmd == "score":
        score()
    elif a.fold is None:
        sys.exit("give the year / fold")
    elif a.cmd == "prep":
        prep(a.fold)
    elif a.cmd == "estimator":
        estimator(a.fold)
    elif a.cmd == "train":
        train_gen(a.fold, a.steps)
    else:
        generate(a.fold, a.limit)
