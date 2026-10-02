# -*- coding: utf-8 -*-
"""
train_reid_native_stable_v3_full.py

Native YOLO+Tracker tracks.csv -> stable segments -> ReID training (FULL PIPELINE).

This is a SUPERSET of your v2:
-  make-patches: crop patches from BIG_ROOT using native tracks.csv bboxes
-  build-stable: detect breaks/drifts to split stable short segments
-  build-train: join patch_meta with stable -> reid_train_stable.csv (supports --patch-root rewrite)
-  bench: fast benchmark (coarse+fine) to pick best (P,K,nw,pf)
-  train: CE + batch-hard triplet; validate by Recall@1; early-stop; ckpt best/last; resume

Hard lessons embedded:
- robust time parsing: prefer time_min/tmin/minute; fallback to filename tmin
- robust well identity: dish/p/fp/tmin parsed from filename
- dedup within (dish,p,fp,track_id,tmin) keep highest conf
- gap_min default=22 (10-min sampling, your dt stats)
"""

from __future__ import annotations
import os, re, json, argparse, random, math, time, glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as T
import torchvision.models as models
from tqdm import tqdm

# Prefer fast IO decode
try:
    from torchvision.io import read_image, ImageReadMode
    HAS_TV_READ = True
except Exception:
    HAS_TV_READ = False

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def info(x):
    print(x, flush=True)

def die(x):
    raise SystemExit(f"\n[FATAL] {x}")

def _is_file_fast(p: str) -> bool:
    try:
        return os.path.isfile(p)
    except Exception:
        return False

def _set_worker_env(worker_id: int):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    s = (torch.initial_seed() + worker_id) % 2**32
    random.seed(s)
    np.random.seed(s)



_FNAME_RE = re.compile(r"^(?P<dish>[^_]+)_P(?P<p>\d+)_FP(?P<fp>\d+)_(?P<tmin>-?\d+)\.jpg$", re.IGNORECASE)

def parse_from_filename(fn: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    m = _FNAME_RE.match(Path(str(fn)).name)
    if not m:
        return None, None, None, None
    return str(m.group("dish")), int(m.group("p")), int(m.group("fp")), int(m.group("tmin"))

def big_image_path(big_root: Path, dish: str, p: int, fp: int, tmin: int) -> Path:
    
    return Path(big_root) / f"MTL-0034-{dish}" / f"{dish}_P{int(p)}_FP{int(fp)}_{int(tmin)}.jpg"

def clamp_box(x1, y1, x2, y2, W, H) -> Optional[Tuple[int,int,int,int]]:
    x1 = int(round(float(x1))); y1 = int(round(float(y1)))
    x2 = int(round(float(x2))); y2 = int(round(float(y2)))
    x1 = max(0, min(x1, W-1)); y1 = max(0, min(y1, H-1))
    x2 = max(0, min(x2, W));   y2 = max(0, min(y2, H))
    if x2 <= x1+1 or y2 <= y1+1:
        return None
    return x1, y1, x2, y2


def bbox_iou_np(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(1e-6, (a[2]-a[0])*(a[3]-a[1]))
    area_b = max(1e-6, (b[2]-b[0])*(b[3]-b[1]))
    return float(inter / (area_a + area_b - inter + 1e-6))

def split_stable_segments(trk: pd.DataFrame, gap_min: float, jump_px: float, iou_min: float, area_ratio_max: float) -> List[Tuple[int, int]]:
    """
    Break between i-1 and i if:
      - dt > gap_min
      - center jump > jump_px
      - IoU < iou_min
      - area ratio > area_ratio_max
    """
    n = len(trk)
    if n <= 1:
        return [(0, n)]
    cuts = [0]
    for i in range(1, n):
        t0 = float(trk.loc[i-1, "tmin"])
        t1 = float(trk.loc[i,   "tmin"])
        if (t1 - t0) > float(gap_min):
            cuts.append(i); continue

        dx = float(trk.loc[i, "cx"] - trk.loc[i-1, "cx"])
        dy = float(trk.loc[i, "cy"] - trk.loc[i-1, "cy"])
        if (dx*dx + dy*dy) ** 0.5 > float(jump_px):
            cuts.append(i); continue

        a0 = float(trk.loc[i-1, "area"]); a1 = float(trk.loc[i, "area"])
        r = max(a1/max(a0,1e-6), a0/max(a1,1e-6))
        if r > float(area_ratio_max):
            cuts.append(i); continue

        box0 = trk.loc[i-1, ["x1","y1","x2","y2"]].to_numpy(dtype=float)
        box1 = trk.loc[i,   ["x1","y1","x2","y2"]].to_numpy(dtype=float)
        if bbox_iou_np(box0, box1) < float(iou_min):
            cuts.append(i); continue

    cuts.append(n)
    return [(s,e) for s,e in zip(cuts[:-1], cuts[1:]) if e>s]


def read_rgb_u8(path: str) -> torch.Tensor:
    if HAS_TV_READ:
        try:
            return read_image(path, mode=ImageReadMode.RGB)
        except Exception:
            pass
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2,0,1).contiguous()


class PatchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, id2idx: Dict[str,int], train: bool, skip_missing: bool):
        self.df = df.reset_index(drop=True)
        self.id2idx = id2idx
        self.skip_missing = bool(skip_missing)
        if train:
            self.tfm = T.Compose([
                T.Resize(256),
                T.RandomResizedCrop(224, scale=(0.75, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.15, 0.15, 0.10, 0.02),
                T.ConvertImageDtype(torch.float32),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
        else:
            self.tfm = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ConvertImageDtype(torch.float32),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = str(r["patch_path"])
        if self.skip_missing and (not _is_file_fast(p)):
            j = random.randint(0, len(self.df)-1)
            r = self.df.iloc[j]
            p = str(r["patch_path"])
        x = self.tfm(read_rgb_u8(p))
        y = int(self.id2idx[str(r["id_key"])])
        return x, y

class PKSampler(Sampler[List[int]]):
    def __init__(self, labels: List[int], P: int, K: int, iters: int, seed: int):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.P, self.K, self.iters = int(P), int(K), int(iters)
        self.rng = np.random.RandomState(int(seed))
        self.unique = np.unique(self.labels)
        self.id2inds = {int(u): np.where(self.labels==u)[0] for u in self.unique}

    def __len__(self):
        return self.iters

    def __iter__(self):
        if len(self.unique) == 0:
            raise RuntimeError("PKSampler: empty unique labels")
        for _ in range(self.iters):
            batch=[]
            ids = self.rng.choice(self.unique, size=self.P, replace=(len(self.unique)<self.P))
            for u in ids:
                inds = self.id2inds[int(u)]
                take = self.rng.choice(inds, size=self.K, replace=(len(inds)<self.K))
                batch.extend(take.tolist())
            yield batch


class ReIDModel(nn.Module):
    def __init__(self, nc: int, embed: int = 256, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        bb = models.resnet50(weights=weights)
        feat = bb.fc.in_features
        bb.fc = nn.Identity()
        self.backbone = bb
        self.bn = nn.BatchNorm1d(feat)
        self.emb = nn.Linear(feat, embed)
        self.cls = nn.Linear(embed, nc)

    def forward(self, x):
        f = self.backbone(x)
        f = self.bn(f)
        e = F.normalize(self.emb(f), dim=1)
        return e, self.cls(e)

def batch_hard_triplet_loss(emb: torch.Tensor, y: torch.Tensor, margin: float) -> torch.Tensor:
    emb = F.normalize(emb.float(), dim=1)  # fp32 safe
    sim = emb @ emb.t()
    dist = 1.0 - sim
    B = emb.size(0)
    y = y.view(-1)
    mask_pos = (y[None,:] == y[:,None])
    eye = torch.eye(B, device=emb.device, dtype=torch.bool)
    mask_pos = mask_pos & (~eye)
    mask_neg = ~mask_pos
    BIG = 1e4
    pos = dist.masked_fill(~mask_pos, -BIG)
    neg = dist.masked_fill(~mask_neg, BIG)
    hp = pos.max(1)[0]
    hn = neg.min(1)[0]
    return F.relu(hp - hn + float(margin)).mean()

def make_lr_scheduler(opt, epochs: int, warmup_epochs: int):
    def lr_lambda(ep: int):
        if warmup_epochs > 0 and ep < warmup_epochs:
            return float(ep+1)/float(warmup_epochs)
        t = (ep - warmup_epochs) / max(1, (epochs - warmup_epochs))
        return 0.5*(1.0 + math.cos(math.pi*t))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

@torch.no_grad()
def evaluate_reid_r1(model: nn.Module, dl: DataLoader, device: torch.device, amp_on: bool) -> float:
    model.eval()
    embs, ys = [], []
    for x,y in tqdm(dl, desc="val_emb", dynamic_ncols=True, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_on):
            e,_ = model(x)
        embs.append(F.normalize(e.float(), dim=1).cpu())
        ys.append(y.cpu())
    if not embs:
        return float("nan")
    E = torch.cat(embs, 0)
    Y = torch.cat(ys, 0)
    S = E @ E.t()
    S.fill_diagonal_(-1e9)
    nn_idx = S.argmax(1)
    return float((Y[nn_idx] == Y).float().mean().item())


def scan_tracks_files(tracks_root: Path) -> List[Path]:
    pat = str(Path(tracks_root) / "*" / "P*_FP*" / "tracks.csv")
    return sorted([Path(x) for x in glob.glob(pat)])

def _pick_time_col(df: pd.DataFrame) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in ["time_min","tmin","minute","minutes","t_min","time","min"]:
        if cand in lower:
            return lower[cand]
    return None

def load_native_tracks_one(path: Path, min_conf: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}

    need = ["track_id","x1","y1","x2","y2","filename"]
    for k in need:
        if k not in lower:
            raise ValueError(f"missing col {k}")

    tcol = _pick_time_col(df)  # may be None -> we will fallback to filename tmin
    conf_col = lower.get("conf", None)

    cols = [lower["track_id"], lower["x1"], lower["y1"], lower["x2"], lower["y2"], lower["filename"]]
    sub = df[cols].copy()
    sub.columns = ["track_id","x1","y1","x2","y2","filename"]

    # time
    if tcol is not None:
        sub["tmin_raw"] = pd.to_numeric(df[tcol], errors="coerce")
    else:
        sub["tmin_raw"] = np.nan

    # conf
    if conf_col is not None:
        sub["conf"] = pd.to_numeric(df[conf_col], errors="coerce").fillna(0.0).astype(float)
    else:
        sub["conf"] = 1.0

    # parse dish/p/fp/tmin from filename (ground truth)
    parsed = sub["filename"].astype(str).map(parse_from_filename)
    ok = parsed.map(lambda x: x[0] is not None)
    sub = sub[ok].copy()
    parsed = parsed[ok]

    sub["dish"] = parsed.map(lambda x: x[0]).astype(str).to_numpy()
    sub["p"]    = parsed.map(lambda x: x[1]).astype(int).to_numpy()
    sub["fp"]   = parsed.map(lambda x: x[2]).astype(int).to_numpy()
    sub["t_from_fn"] = parsed.map(lambda x: x[3]).astype(int).to_numpy()

    # final tmin: prefer tmin_raw if present, else filename
    sub["tmin"] = sub["tmin_raw"]
    sub.loc[sub["tmin"].isna(), "tmin"] = sub.loc[sub["tmin"].isna(), "t_from_fn"]
    sub["tmin"] = pd.to_numeric(sub["tmin"], errors="coerce")

    # types
    for c in ["track_id","x1","y1","x2","y2"]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=["track_id","tmin","x1","y1","x2","y2"])
    sub["track_id"] = sub["track_id"].astype(int)
    sub["tmin"] = sub["tmin"].astype(int)

    # conf filter
    sub = sub[sub["conf"] >= float(min_conf)].copy()

    # dedup (dish,p,fp,track_id,tmin) keep highest conf
    sub = sub.sort_values(["dish","p","fp","track_id","tmin","conf"], ascending=[True,True,True,True,True,False])
    sub = sub.drop_duplicates(["dish","p","fp","track_id","tmin"], keep="first").reset_index(drop=True)

    # derived
    sub["cx"] = (sub["x1"] + sub["x2"]) * 0.5
    sub["cy"] = (sub["y1"] + sub["y2"]) * 0.5
    sub["area"] = np.maximum(1.0, (sub["x2"] - sub["x1"]) * (sub["y2"] - sub["y1"]))
    return sub

def load_native_tracks_many(tracks_root: Path, max_files: int, min_conf: float) -> pd.DataFrame:
    files = scan_tracks_files(tracks_root)
    if not files:
        die(f"No tracks.csv under {tracks_root}")
    if max_files > 0:
        files = files[:max_files]
    info(f"[scan] tracks.csv files: {len(files)}")

    dfs = []
    for f in tqdm(files, desc="load_tracks", dynamic_ncols=True):
        try:
            d = load_native_tracks_one(f, min_conf=min_conf)
            if len(d):
                dfs.append(d)
        except Exception as e:
            info(f"[warn] skip {f}: {e}")
            continue
    if not dfs:
        die("No valid tracks loaded.")
    return pd.concat(dfs, ignore_index=True)

# patch_path rewrite
def rewrite_patch_paths(df: pd.DataFrame, patch_root: Path) -> pd.DataFrame:
    df = df.copy()
    patch_root = Path(patch_root).resolve()
    marker1 = "reid_patches/"
    marker2 = "reid_patches_native/"
    def norm(s: str) -> str:
        s = str(s).strip().replace("\\","/")
        s = re.sub(r"/+","/", s)
        return s
    pr = norm(str(patch_root))

    def rewrite_one(p: str) -> str:
        p0 = norm(p)
        if (not p0) or p0.lower()=="nan":
            return p0
        if marker2 in p0:
            rel = p0.split(marker2, 1)[1].lstrip("/\\")
            return str(patch_root / rel)
        if marker1 in p0:
            rel = p0.split(marker1, 1)[1].lstrip("/\\")
            return str(patch_root / rel)
        if norm(p0).startswith(pr):
            return p0
        if not p0.startswith("/"):
            return str(patch_root / p0)
        return p0

    if "patch_path" in df.columns:
        df["patch_path"] = df["patch_path"].astype(str).map(rewrite_one)
    return df

# cmd: make-patches
def cmd_make_patches(args):
    tracks_root = Path(args.tracks_root).resolve()
    big_root = Path(args.big_root).resolve()
    out_patch_root = Path(args.out_patch_root).resolve()
    out_patch_meta = Path(args.out_patch_meta).resolve()
    ensure_dir(out_patch_root)
    ensure_dir(out_patch_meta.parent)

    tr = load_native_tracks_many(tracks_root, max_files=int(args.max_tracks_files), min_conf=float(args.min_conf))
    info(f"[make-patches] rows={len(tr)} unique wells={tr[['dish','p','fp']].drop_duplicates().shape[0]}")

    seen = saved = skip_done = missing_big = bad_box = fail = 0
    rows = []

    for r in tqdm(tr.itertuples(index=False), total=len(tr), desc="make_patches", dynamic_ncols=True):
        dish = str(r.dish); p = int(r.p); fp = int(r.fp); tid = int(r.track_id); tmin = int(r.tmin)
        conf = float(r.conf)
        seen += 1

        bigp = big_image_path(big_root, dish, p, fp, tmin)
        if not bigp.is_file():
            missing_big += 1
            continue

        try:
            with Image.open(bigp) as im:
                im = im.convert("RGB")
                W, H = im.size
                box = clamp_box(r.x1, r.y1, r.x2, r.y2, W, H)
                if box is None:
                    bad_box += 1
                    continue
                x1,y1,x2,y2 = box
                crop = im.crop((x1,y1,x2,y2)).resize((224,224), resample=Image.BILINEAR)
        except Exception:
            fail += 1
            continue

        out_dir = out_patch_root / f"{dish}_P{p}_FP{fp}"
        ensure_dir(out_dir)
        out_name = f"{dish}_P{p}_FP{fp}_ID{tid}_T{tmin}.jpg"
        out_path = out_dir / out_name

        if out_path.is_file() and (not bool(args.overwrite)):
            skip_done += 1
        else:
            try:
                crop.save(out_path, quality=95)
                saved += 1
            except Exception:
                fail += 1
                continue

        rows.append({
            "dish": dish, "p": p, "fp": fp,
            "track_id": tid, "tmin": tmin, "conf": conf,
            "patch_path": str(out_path),
        })

    pm = pd.DataFrame(rows)
    pm.to_csv(out_patch_meta, index=False)
    info(f"[make-patches] saved meta rows={len(pm)} -> {out_patch_meta}")
    info(f"[make-patches] seen={seen} saved={saved} skip_done={skip_done} missing_big={missing_big} bad_box={bad_box} fail={fail}")

# cmd: build-stable
def cmd_build_stable(args):
    tracks_root = Path(args.tracks_root).resolve()
    out_stable = Path(args.out_stable).resolve()
    ensure_dir(out_stable.parent)

    tr = load_native_tracks_many(tracks_root, max_files=int(args.max_tracks_files), min_conf=float(args.min_conf))
    info(f"[build-stable] rows={len(tr)} unique wells={tr[['dish','p','fp']].drop_duplicates().shape[0]}")

    gap_min = float(args.gap_min)
    jump_px = float(args.jump_px)
    iou_min = float(args.iou_min)
    area_ratio_max = float(args.area_ratio_max)
    min_seg_len = int(args.min_seg_len)
    max_seg_len = int(args.max_seg_len)

    rows = []
    key = ["dish","p","fp","track_id"]
    for (dish,p,fp,tid), g in tqdm(tr.groupby(key, sort=False), desc="stable_segments", dynamic_ncols=True):
        g = g.sort_values("tmin").drop_duplicates(["tmin"], keep="first").reset_index(drop=True)
        if len(g) == 0:
            continue
        segs = split_stable_segments(g, gap_min=gap_min, jump_px=jump_px, iou_min=iou_min, area_ratio_max=area_ratio_max)
        seg_id = 0
        for s,e in segs:
            n = e-s
            if n < min_seg_len:
                continue
            if max_seg_len > 0 and n > max_seg_len:
                k = max_seg_len
                for ss in range(s, e, k):
                    ee = min(e, ss+k)
                    if (ee-ss) < min_seg_len:
                        continue
                    id_key = f"{dish}_P{int(p)}_FP{int(fp)}_T{int(tid)}_S{seg_id}"
                    sub = g.iloc[ss:ee]
                    for rr in sub.itertuples(index=False):
                        rows.append({"dish":dish,"p":int(p),"fp":int(fp),
                                     "track_id":int(tid),"tmin":int(rr.tmin),
                                     "seg_id":int(seg_id),"id_key":id_key})
                    seg_id += 1
            else:
                id_key = f"{dish}_P{int(p)}_FP{int(fp)}_T{int(tid)}_S{seg_id}"
                sub = g.iloc[s:e]
                for rr in sub.itertuples(index=False):
                    rows.append({"dish":dish,"p":int(p),"fp":int(fp),
                                 "track_id":int(tid),"tmin":int(rr.tmin),
                                 "seg_id":int(seg_id),"id_key":id_key})
                seg_id += 1

    stable = pd.DataFrame(rows)
    if len(stable) == 0:
        die("No stable segments produced. Loosen thresholds or lower min_seg_len.")
    stable.to_csv(out_stable, index=False)
    info(f"[stable] saved rows={len(stable)} -> {out_stable}")
    info(f"[stable] unique id_key={stable['id_key'].nunique()} unique wells={stable[['dish','p','fp']].drop_duplicates().shape[0]}")

# cmd: build-train
def cmd_build_train(args):
    patch_meta = Path(args.patch_meta).resolve()
    stable_csv = Path(args.stable_csv).resolve()
    out_train = Path(args.out_train).resolve()
    ensure_dir(out_train.parent)

    if not patch_meta.is_file():
        die(f"patch_meta not found: {patch_meta}")
    if not stable_csv.is_file():
        die(f"stable_csv not found: {stable_csv}")

    pm = pd.read_csv(patch_meta)
    need_pm = {"dish","p","fp","track_id","tmin","patch_path"}
    miss = need_pm - set(pm.columns)
    if miss:
        die(f"patch_meta missing columns: {sorted(list(miss))}")

    if args.patch_root:
        pm = rewrite_patch_paths(pm, patch_root=Path(args.patch_root))

    pm["patch_path"] = pm["patch_path"].astype(str)
    pm = pm[pm["patch_path"].map(_is_file_fast)].copy()

    stable = pd.read_csv(stable_csv)
    need_st = {"dish","p","fp","track_id","tmin","id_key"}
    miss = need_st - set(stable.columns)
    if miss:
        die(f"stable_csv missing columns: {sorted(list(miss))}")

    # types
    for c in ["p","fp","track_id","tmin"]:
        pm[c] = pd.to_numeric(pm[c], errors="coerce").astype(int)
        stable[c] = pd.to_numeric(stable[c], errors="coerce").astype(int)

    df = pm.merge(stable[["dish","p","fp","track_id","tmin","id_key"]], on=["dish","p","fp","track_id","tmin"], how="inner")
    if len(df) == 0:
        die("No rows after join patch_meta x stable. Key mismatch.")

    max_per = int(args.max_samples_per_id)
    if max_per > 0:
        rng = np.random.RandomState(0)
        keep=[]
        for k,g in df.groupby("id_key", sort=False):
            idx = g.index.to_numpy()
            if len(idx) > max_per:
                idx = rng.choice(idx, size=max_per, replace=False)
            keep.append(idx)
        keep = np.concatenate(keep) if keep else np.array([], dtype=int)
        df = df.loc[keep].reset_index(drop=True)

    df.to_csv(out_train, index=False)
    info(f"[traincsv] saved rows={len(df)} ids={df['id_key'].nunique()} -> {out_train}")

# cmd: bench (fast resume)
@torch.no_grad()
def cmd_bench(args):
    train_csv = Path(args.train_csv).resolve()
    if not train_csv.is_file():
        die(f"train_csv not found: {train_csv}")

    df = pd.read_csv(train_csv)
    need = {"patch_path","id_key"}
    miss = need - set(df.columns)
    if miss:
        die(f"train_csv missing {sorted(list(miss))}")

    if args.patch_root:
        df = rewrite_patch_paths(df, patch_root=Path(args.patch_root))

    df["patch_path"] = df["patch_path"].astype(str)
    df = df[df["patch_path"].map(_is_file_fast)].copy().reset_index(drop=True)

    if args.max_rows > 0 and len(df) > int(args.max_rows):
        df = df.sample(n=int(args.max_rows), random_state=int(args.seed)).reset_index(drop=True)

    ids = df["id_key"].astype(str).unique().tolist()
    id2idx = {k:i for i,k in enumerate(ids)}
    df["y"] = df["id_key"].astype(str).map(id2idx).astype(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        die("CUDA not available for bench.")
    ngpu = torch.cuda.device_count()

    info(f"[data] rows={len(df)} ids={len(ids)} | tv_read_image={int(HAS_TV_READ)}")
    info(f"[gpu] {ngpu} GPUs: {[torch.cuda.get_device_name(i) for i in range(ngpu)]}")

    ds = PatchDataset(df, id2idx=id2idx, train=True, skip_missing=False)

    # quick run config
    P_list = [int(x) for x in args.p_list.split(",") if x.strip()]
    K = int(args.k)
    NW_list = [int(x) for x in args.nw_list.split(",") if x.strip()]
    PF = int(args.prefetch_factor)

    AMP_ON = bool(args.amp)
    CHANNELS_LAST = bool(args.channels_last)
    USE_DP = (ngpu > 1) and (not bool(args.no_dp))

    def one_run(P: int, nw: int) -> Dict[str,Any]:
        sampler = PKSampler(df["y"].tolist(), P=P, K=K, iters=max(int(args.steps), int(args.warmup_steps)), seed=int(args.seed))
        dl = DataLoader(
            ds, batch_sampler=sampler,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=(nw>0),
            prefetch_factor=PF if nw>0 else 2,
            worker_init_fn=_set_worker_env if nw>0 else None,
        )

        model = ReIDModel(nc=len(ids), embed=int(args.embed_dim), pretrained=True).to(device)
        if CHANNELS_LAST:
            model = model.to(memory_format=torch.channels_last)
        if USE_DP:
            model = nn.DataParallel(model)

        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=AMP_ON)

        it = iter(dl)
        # warmup
        for _ in range(int(args.warmup_steps)):
            x,y = next(it)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if CHANNELS_LAST:
                x = x.contiguous(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=AMP_ON):
                emb,logits = model(x)
                loss = F.cross_entropy(logits,y) + batch_hard_triplet_loss(emb,y,margin=float(args.triplet_margin))
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        torch.cuda.synchronize()

        # timed
        t0=time.time()
        n_samples=0
        it = iter(dl)
        for _ in range(int(args.steps)):
            x,y = next(it)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if CHANNELS_LAST:
                x = x.contiguous(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=AMP_ON):
                emb,logits = model(x)
                loss = F.cross_entropy(logits,y) + batch_hard_triplet_loss(emb,y,margin=float(args.triplet_margin))
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            n_samples += x.size(0)
        torch.cuda.synchronize()
        dt=time.time()-t0

        rsv0 = torch.cuda.memory_reserved(0)/(1024**2)
        alloc0 = torch.cuda.memory_allocated(0)/(1024**2)
        return {
            "P":P,"K":K,"batch":P*K,"num_workers":nw,"prefetch_factor":PF,
            "samples_per_s": float(n_samples/max(dt,1e-9)),
            "steps_per_s": float(int(args.steps)/max(dt,1e-9)),
            "rsv0_MB": float(rsv0), "alloc0_MB": float(alloc0),
        }

    results=[]
    best=None

    info("[bench] running...")
    for P in P_list:
        for nw in NW_list:
            try:
                r = one_run(P, nw)
                results.append(r)
                info(f"  P={P:2d} batch={r['batch']:3d} | nw={nw:2d} pf={PF} | samples/s={r['samples_per_s']:.1f} steps/s={r['steps_per_s']:.3f} | rsv0={r['rsv0_MB']:.0f}MB alloc0={r['alloc0_MB']:.0f}MB")
                if best is None or r["samples_per_s"] > best["samples_per_s"]:
                    best = r
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    info(f"  [OOM-skip] P={P} nw={nw}")
                    torch.cuda.empty_cache()
                    continue
                raise

    info(f"[bench] BEST: {best}")
    if args.out_json:
        outp = Path(args.out_json).resolve()
        ensure_dir(outp.parent)
        outp.write_text(json.dumps({"best":best,"results":results}, indent=2), encoding="utf-8")
        info(f"[bench] saved -> {outp}")

# cmd: train (with resume)
def cmd_train(args):
    seed_all(int(args.seed))
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)
    ensure_dir(out_dir/"ckpt")
    ensure_dir(out_dir/"logs")
    ensure_dir(out_dir/"meta")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        die("CUDA not available.")
    amp_on = bool(args.amp)

    df = pd.read_csv(args.train_csv)
    need = {"id_key","patch_path"}
    miss = need - set(df.columns)
    if miss:
        die(f"train_csv missing columns: {sorted(list(miss))}")

    if args.patch_root:
        df = rewrite_patch_paths(df, patch_root=Path(args.patch_root))

    df["patch_path"] = df["patch_path"].astype(str)
    ok = df["patch_path"].map(_is_file_fast)
    before = len(df)
    df = df[ok].copy().reset_index(drop=True)
    info(f"[data] is_file: {before}->{len(df)} dropped={before-len(df)}")
    if len(df) == 0:
        die("No rows after file filter.")

    # optional cap per id
    if int(args.max_samples_per_id) > 0:
        rng = np.random.RandomState(int(args.seed))
        keep=[]
        for k,g in df.groupby("id_key", sort=False):
            idx = g.index.to_numpy()
            if len(idx) > int(args.max_samples_per_id):
                idx = rng.choice(idx, size=int(args.max_samples_per_id), replace=False)
            keep.append(idx)
        keep = np.concatenate(keep) if keep else np.array([], dtype=int)
        df = df.loc[keep].reset_index(drop=True)
        info(f"[data] cap per id_key={args.max_samples_per_id} => rows={len(df)} ids={df['id_key'].nunique()}")

    ids = df["id_key"].astype(str).unique().tolist()
    id2idx = {k:i for i,k in enumerate(ids)}
    df["y"] = df["id_key"].astype(str).map(id2idx).astype(int)

    # within-id split
    rng = np.random.RandomState(int(args.seed))
    val_mask = np.zeros(len(df), dtype=bool)
    for k, idxs in df.groupby("id_key").groups.items():
        idxs = np.array(list(idxs))
        n_val = max(1, int(round(float(args.val_frac)*len(idxs))))
        choose = rng.choice(idxs, size=n_val, replace=False)
        val_mask[choose] = True
    tr_df = df[~val_mask].reset_index(drop=True)
    va_df = df[val_mask].reset_index(drop=True)

    P,K = int(args.pk_p), int(args.pk_k)
    batch = P*K
    info(f"[data] rows={len(df)} ids={len(ids)} | train_rows={len(tr_df)} val_rows={len(va_df)} | batch={batch} | iters/epoch={int(args.iters_per_epoch)}")
    info(f"[io] tv_read_image={int(HAS_TV_READ)}")
    info(f"[gpu] {torch.cuda.device_count()} GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

    (out_dir/"meta/args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (out_dir/"meta/id2idx.json").write_text(json.dumps(id2idx, indent=2), encoding="utf-8")

    ds_tr = PatchDataset(tr_df, id2idx=id2idx, train=True,  skip_missing=bool(args.skip_missing))
    ds_va = PatchDataset(va_df, id2idx=id2idx, train=False, skip_missing=bool(args.skip_missing))

    sampler = PKSampler(tr_df["y"].tolist(), P=P, K=K, iters=int(args.iters_per_epoch), seed=int(args.seed))
    dl_tr = DataLoader(
        ds_tr, batch_sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=(int(args.num_workers)>0),
        prefetch_factor=int(args.prefetch_factor) if int(args.num_workers)>0 else 2,
        worker_init_fn=_set_worker_env if int(args.num_workers)>0 else None,
    )

    va_workers = max(0, min(int(args.num_workers), 8))
    dl_va = DataLoader(
        ds_va, batch_size=int(args.val_batch_size), shuffle=False,
        num_workers=va_workers,
        pin_memory=True,
        persistent_workers=(va_workers>0),
        prefetch_factor=int(args.prefetch_factor) if va_workers>0 else 2,
        worker_init_fn=_set_worker_env if va_workers>0 else None,
        drop_last=False,
    )

    model = ReIDModel(nc=len(ids), embed=int(args.embed_dim), pretrained=bool(int(args.pretrained))).to(device)
    if bool(args.channels_last):
        model = model.to(memory_format=torch.channels_last)

    use_dp = (torch.cuda.device_count() > 1) and (not bool(args.no_dp))
    if use_dp:
        info(f"[dp] DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.wd))
    sched = make_lr_scheduler(opt, epochs=int(args.epochs), warmup_epochs=int(args.warmup_epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)

    log_path = out_dir/"logs/metrics.csv"
    if not log_path.exists():
        log_path.write_text("epoch,train_loss,train_ce,train_tri,val_r1,lr,sec\n", encoding="utf-8")

    # resume
    start_epoch = 1
    best_val = -1.0
    best_epoch = 0
    patience_left = int(args.early_stop_patience)
    last_ckpt = out_dir/"ckpt/last.pth"
    if bool(args.resume) and last_ckpt.exists():
        info(f"[resume] load {last_ckpt}")
        ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        state = ck.get("model_state", ck.get("model", ck))
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state, strict=False)
        if "opt_state" in ck: opt.load_state_dict(ck["opt_state"])
        if "scaler_state" in ck: scaler.load_state_dict(ck["scaler_state"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        best_val = float(ck.get("best_val_r1", best_val))
        best_epoch = int(ck.get("best_epoch", best_epoch))
        patience_left = int(ck.get("patience_left", patience_left))
        info(f"[resume] start_epoch={start_epoch} best_val={best_val:.6f} best_epoch={best_epoch} patience_left={patience_left}")

    info("[train] start")
    for epoch in range(start_epoch, int(args.epochs)+1):
        t0 = time.time()
        model.train()
        losses=[]; ces=[]; tris=[]

        pbar = tqdm(dl_tr, total=len(dl_tr), dynamic_ncols=True, desc=f"epoch {epoch}/{args.epochs}")
        for step,(x,y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if bool(args.channels_last):
                x = x.contiguous(memory_format=torch.channels_last)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_on):
                emb, logits = model(x)
                ce = F.cross_entropy(logits, y)
                tri = batch_hard_triplet_loss(emb, y, margin=float(args.triplet_margin))
                loss = ce + tri
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.item()))
            ces.append(float(ce.item()))
            tris.append(float(tri.item()))

            if step % int(args.log_every) == 0:
                pbar.set_postfix(
                    loss=f"{np.mean(losses[-50:]):.4f}",
                    ce=f"{np.mean(ces[-50:]):.4f}",
                    tri=f"{np.mean(tris[-50:]):.4f}",
                    lr=f"{opt.param_groups[0]['lr']:.2e}",
                )

        sched.step()
        sec = time.time()-t0
        tr_loss = float(np.mean(losses)) if losses else float("nan")
        tr_ce = float(np.mean(ces)) if ces else float("nan")
        tr_tri = float(np.mean(tris)) if tris else float("nan")
        lr = float(opt.param_groups[0]["lr"])

        val_r1 = evaluate_reid_r1(model, dl_va, device=device, amp_on=amp_on) if len(va_df) else float("nan")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{tr_loss:.6f},{tr_ce:.6f},{tr_tri:.6f},{val_r1:.6f},{lr:.10f},{sec:.2f}\n")

        info(f"[epoch {epoch}] loss={tr_loss:.4f} ce={tr_ce:.4f} tri={tr_tri:.4f} | val_r1={val_r1:.4f} | lr={lr:.2e} | {sec:.1f}s")

        # always save last (so SSH drop is safe)
        ckpt = {
            "epoch": epoch,
            "model_state": (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()),
            "opt_state": opt.state_dict(),
            "scaler_state": scaler.state_dict(),
            "id2idx": id2idx,
            "args": vars(args),
            "best_val_r1": best_val,
            "best_epoch": best_epoch,
            "patience_left": patience_left,
            "embed_dim": int(args.embed_dim),
        }
        torch.save(ckpt, out_dir/"ckpt/last.pth")

        # early stop
        if not math.isnan(val_r1):
            improved = (val_r1 - best_val) > float(args.early_stop_min_delta)
            if improved:
                best_val = float(val_r1)
                best_epoch = int(epoch)
                patience_left = int(args.early_stop_patience)
                ckpt["best_val_r1"] = best_val
                ckpt["best_epoch"] = best_epoch
                ckpt["patience_left"] = patience_left
                torch.save(ckpt, out_dir/"ckpt/best.pth")
                info(f"[best] epoch={epoch} val_r1={best_val:.6f} (reset patience={patience_left})")
            else:
                patience_left -= 1
                info(f"[early-stop] no improvement. patience_left={patience_left} (best={best_val:.6f} @ epoch {best_epoch})")
                if patience_left <= 0:
                    info(f"[early-stop] STOP at epoch {epoch}. Best val_r1={best_val:.6f} @ epoch {best_epoch}")
                    break

    info("[done] ckpt/best.pth + ckpt/last.pth + logs/metrics.csv")

# CLI
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("make-patches")
    sp.add_argument("--tracks-root", required=True)
    sp.add_argument("--big-root", required=True)
    sp.add_argument("--out-patch-root", required=True)
    sp.add_argument("--out-patch-meta", required=True)
    sp.add_argument("--max-tracks-files", type=int, default=0)
    sp.add_argument("--min-conf", type=float, default=0.0)
    sp.add_argument("--overwrite", action="store_true")

    sp = sub.add_parser("build-stable")
    sp.add_argument("--tracks-root", required=True)
    sp.add_argument("--out-stable", required=True)
    sp.add_argument("--max-tracks-files", type=int, default=0)
    sp.add_argument("--min-conf", type=float, default=0.0)
    sp.add_argument("--gap-min", type=float, default=22.0)
    sp.add_argument("--jump-px", type=float, default=60.0)
    sp.add_argument("--iou-min", type=float, default=0.08)
    sp.add_argument("--area-ratio-max", type=float, default=3.5)
    sp.add_argument("--min-seg-len", type=int, default=6)
    sp.add_argument("--max-seg-len", type=int, default=80)

    sp = sub.add_parser("build-train")
    sp.add_argument("--patch-meta", required=True)
    sp.add_argument("--stable-csv", required=True)
    sp.add_argument("--out-train", required=True)
    sp.add_argument("--patch-root", default="", help="optional rewrite root")
    sp.add_argument("--max-samples-per-id", type=int, default=120)

    sp = sub.add_parser("bench")
    sp.add_argument("--train-csv", required=True)
    sp.add_argument("--patch-root", default="", help="optional rewrite root")
    sp.add_argument("--p-list", type=str, default="24,32,48,64")
    sp.add_argument("--k", type=int, default=6)
    sp.add_argument("--nw-list", type=str, default="0,8,16")
    sp.add_argument("--prefetch-factor", type=int, default=2)
    sp.add_argument("--steps", type=int, default=30)
    sp.add_argument("--warmup-steps", type=int, default=5)
    sp.add_argument("--max-rows", type=int, default=12000)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--amp", action="store_true")
    sp.add_argument("--channels-last", action="store_true")
    sp.add_argument("--no-dp", action="store_true")
    sp.add_argument("--embed-dim", type=int, default=256)
    sp.add_argument("--lr", type=float, default=3e-4)
    sp.add_argument("--triplet-margin", type=float, default=0.3)
    sp.add_argument("--out-json", default="")

    sp = sub.add_parser("train")
    sp.add_argument("--train-csv", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--patch-root", default="", help="optional rewrite root for patch_path")

    sp.add_argument("--epochs", type=int, default=60)
    sp.add_argument("--iters-per-epoch", type=int, default=150)
    sp.add_argument("--pk-p", type=int, default=64)   # your bench best
    sp.add_argument("--pk-k", type=int, default=6)    # your bench best
    sp.add_argument("--num-workers", type=int, default=8)  # your bench best (8 or 16)
    sp.add_argument("--prefetch-factor", type=int, default=2)
    sp.add_argument("--amp", action="store_true")
    sp.add_argument("--channels-last", action="store_true")
    sp.add_argument("--no-dp", action="store_true")

    sp.add_argument("--embed-dim", type=int, default=256)
    sp.add_argument("--pretrained", type=int, default=1)
    sp.add_argument("--lr", type=float, default=3e-4)
    sp.add_argument("--wd", type=float, default=1e-4)
    sp.add_argument("--triplet-margin", type=float, default=0.3)
    sp.add_argument("--seed", type=int, default=42)

    sp.add_argument("--warmup-epochs", type=int, default=2)
    sp.add_argument("--log-every", type=int, default=20)
    sp.add_argument("--early-stop-patience", type=int, default=20)
    sp.add_argument("--early-stop-min-delta", type=float, default=1e-4)

    sp.add_argument("--skip-missing", action="store_true")
    sp.add_argument("--val-frac", type=float, default=0.10)
    sp.add_argument("--val-batch-size", type=int, default=256)
    sp.add_argument("--max-samples-per-id", type=int, default=0)

    sp.add_argument("--resume", action="store_true")

    args = ap.parse_args()

    if args.cmd == "make-patches": return cmd_make_patches(args)
    if args.cmd == "build-stable": return cmd_build_stable(args)
    if args.cmd == "build-train":  return cmd_build_train(args)
    if args.cmd == "bench":        return cmd_bench(args)
    if args.cmd == "train":        return cmd_train(args)

if __name__ == "__main__":
    main()