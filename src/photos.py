"""
Week-3 photo features without storing the photos (docs/04-week3-design.md, part B).

  python src/photos.py index            # data/derived/photo_index.parquet: every inner-plot photo, its shard
  python src/photos.py shard K          # stream shard K: download (pinned revision), verify SHA-256, DINOv2 features,
                                        # thumbnail, delete the PNGs; writes data/photos/shards/{features,thumbs}_K.parquet
  python src/photos.py merge            # once all shards exist: data/photos/{features,thumbs}_<year>.parquet

Features per photo: CLS token and mean patch token of DINOv2 ViT-B/14 (frozen, CPU) on the whole photo resized to
420x280 (bicubic), ImageNet normalisation. float32.
"""
import argparse
import glob
import hashlib
import io
import os
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = "mikeboss/FIP1"
REVISION = "11cdd84bc0ba8d5c6641ea7b53550b1658527e1c"  # same revision as the tables
DINO = "facebook/dinov2-base"
DINO_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
SIZE = (420, 280)  # width, height: 30 x 20 patches of 14 px, exactly 3:2 like the 960x640 photos
THUMB = (240, 160)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
N_SHARDS = 48
BATCH = 16
INDEX = "data/derived/photo_index.parquet"
OUT = "data/photos"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- index

def build_index(root="data/fip1"):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "data", "*.parquet"))):
        pf = pq.ParquetFile(f)
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i, columns=["plot_uid", "harvest_year", "inner_plot_images"]).to_pandas()
            for uid, y, paths in zip(t.plot_uid, t.harvest_year, t.inner_plot_images):
                for p in (paths if paths is not None else []):
                    rel = "data/" + p.split("/data/", 1)[1]  # URL points to main; the repo path is revision-independent
                    m = re.search(r"_(\d{8})_(\d{6})\.png$", rel)
                    rows.append((uid, int(y), pd.Timestamp(m.group(1)), m.group(2), rel))
            del t
    idx = pd.DataFrame(rows, columns=["plot_uid", "year", "date", "time", "path"]).sort_values(
        ["plot_uid", "date", "time"]).reset_index(drop=True)
    per = idx.groupby("plot_uid").size().sort_index()  # whole plot-years per shard, balanced by photo count
    start = per.cumsum() - per
    shard_of = np.minimum((start / len(idx) * N_SHARDS).astype(int), N_SHARDS - 1)
    idx["shard"] = idx.plot_uid.map(shard_of).astype(int)
    return idx


# ---------------------------------------------------------------- features

class Encoder:
    def __init__(self):
        import torch
        from transformers import AutoModel
        self.torch = torch
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
        self.model = AutoModel.from_pretrained(DINO, revision=DINO_REVISION).eval()

    @staticmethod
    def preprocess(img):
        from PIL import Image
        x = np.asarray(img.convert("RGB").resize(SIZE, Image.BICUBIC), np.float32) / 255.0
        return ((x - MEAN) / STD).transpose(2, 0, 1)

    def __call__(self, arrays):
        with self.torch.no_grad():
            h = self.model(pixel_values=self.torch.from_numpy(np.stack(arrays))).last_hidden_state
        return h[:, 0].numpy().astype(np.float32), h[:, 1:].mean(1).numpy().astype(np.float32)


def thumbnail(img):
    from PIL import Image
    buf = io.BytesIO()
    img.convert("RGB").resize(THUMB, Image.BICUBIC).save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _expected(paths):
    from huggingface_hub import HfApi
    api, out = HfApi(), {}
    for i in range(0, len(paths), 500):
        for info in _retry(lambda: api.get_paths_info(REPO, paths[i:i + 500], repo_type="dataset", revision=REVISION)):
            out[info.path] = info.lfs.sha256
    return out


def _retry(fn, tries=12):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # rate limits (429) and transient network errors: back off and retry
            if k == tries - 1:
                raise
            wait = min(600, 5 * 2 ** k)
            log(f"retry in {wait}s after {type(e).__name__}: {str(e)[:120]}")
            time.sleep(wait)


def _download(path, tmp, sha):
    from huggingface_hub import hf_hub_download
    f = _retry(lambda: hf_hub_download(REPO, path, repo_type="dataset", revision=REVISION, local_dir=tmp))
    data = open(f, "rb").read()
    got = hashlib.sha256(data).hexdigest()
    if got != sha:
        raise SystemExit(f"checksum mismatch for {path}: {got} != {sha}")
    os.remove(f)  # the PNG never stays on disk beyond this point
    return data


def run_shard(k, idx, enc=None, tmp_root=None):
    os.makedirs(f"{OUT}/shards", exist_ok=True)
    fo, to = f"{OUT}/shards/features_{k:02d}.parquet", f"{OUT}/shards/thumbs_{k:02d}.parquet"
    if os.path.exists(fo) and os.path.exists(to):
        log("shard", k, "already done")
        return
    from PIL import Image
    d = idx[idx.shard == k].reset_index(drop=True)
    log("shard", k, "photos", len(d))
    sha = _expected(list(d.path))
    enc = enc or Encoder()
    tmp = tempfile.mkdtemp(prefix=f"fip_photos_{k:02d}_", dir=tmp_root)
    cls, mean, thumbs = [], [], []
    try:
        with ThreadPoolExecutor(8) as pool:  # downloads overlap with the encoder
            for b in range(0, len(d), BATCH):
                paths = list(d.path[b:b + BATCH])
                blobs = list(pool.map(lambda p: _download(p, tmp, sha[p]), paths))
                imgs = [Image.open(io.BytesIO(x)) for x in blobs]
                c, m = enc([Encoder.preprocess(im) for im in imgs])
                cls.append(c)
                mean.append(m)
                thumbs += [thumbnail(im) for im in imgs]
                if b % (BATCH * 20) == 0:
                    log("shard", k, b + len(paths), "/", len(d))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # also removes the hub's per-file metadata
    C, M = np.concatenate(cls), np.concatenate(mean)
    feat = d[["plot_uid", "year", "date", "time", "path"]].copy()
    feat["sha256"] = [sha[p] for p in d.path]
    feat["cls"], feat["mean"] = list(C), list(M)
    feat.to_parquet(fo + ".tmp", index=False)
    pd.DataFrame({"path": d.path, "jpeg": thumbs}).to_parquet(to + ".tmp", index=False)
    os.replace(fo + ".tmp", fo)
    os.replace(to + ".tmp", to)
    log("shard", k, "done")


def merge(idx):
    missing = [k for k in range(N_SHARDS) if not os.path.exists(f"{OUT}/shards/features_{k:02d}.parquet")]
    if missing:
        raise SystemExit(f"shards not done: {missing}")
    feat = pd.concat([pd.read_parquet(f"{OUT}/shards/features_{k:02d}.parquet") for k in range(N_SHARDS)])
    th = pd.concat([pd.read_parquet(f"{OUT}/shards/thumbs_{k:02d}.parquet") for k in range(N_SHARDS)])
    assert len(feat) == len(idx) and set(feat.path) == set(idx.path), "features do not cover the index"
    for y, g in feat.groupby("year"):
        g.to_parquet(f"{OUT}/features_{y}.parquet", index=False)
        th[th.path.isin(set(g.path))].to_parquet(f"{OUT}/thumbs_{y}.parquet", index=False)
    shutil.rmtree(f"{OUT}/shards")
    log("merged", len(feat), "photos into", sorted(feat.year.unique()))


def load_features(years=None):
    """Per photo: plot_uid, year, date, time, path and a (n, 1536) float32 array [CLS | mean patch]."""
    fs = sorted(glob.glob(f"{OUT}/features_*.parquet"))
    d = pd.concat([pd.read_parquet(f) for f in fs if years is None or int(f[-12:-8]) in years], ignore_index=True)
    X = np.hstack([np.vstack(d.pop("cls").values), np.vstack(d.pop("mean").values)]).astype(np.float32)
    return d, X


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["index", "shard", "merge"])
    ap.add_argument("k", nargs="?", type=int)
    a = ap.parse_args()
    if a.step == "index":
        idx = build_index()
        idx.to_parquet(INDEX, index=False)
        print(len(idx), "photos,", idx.plot_uid.nunique(), "plot-years,", N_SHARDS, "shards:",
              idx.groupby("shard").size().describe()[["min", "max"]].to_dict())
    else:
        idx = pd.read_parquet(INDEX)
        if a.step == "shard":
            if a.k is None:
                sys.exit("give the shard number")
            run_shard(a.k, idx, tmp_root=os.environ.get("TMPDIR"))
        else:
            merge(idx)
