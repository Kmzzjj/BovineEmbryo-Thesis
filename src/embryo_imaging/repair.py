# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import json
import time
import math
import glob
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from tqdm import tqdm

try:
    from torchvision.io import read_image, ImageReadMode
    _HAS_TV_READ = True
except Exception:
    _HAS_TV_READ = False

from PIL import Image, ImageFile

from .paths import big_image_path, parse_from_filename
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _env_str(key: str, default: str) -> str:
    return str(os.getenv(key, default)).strip() or default


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def info(*a):
    print(" ".join(str(x) for x in a), flush=True)


def die(msg: str):
    raise SystemExit(f"\n[FATAL] {msg}")


def seed_all(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_file_fast(p: str) -> bool:
    try:
        return os.path.isfile(p)
    except Exception:
        return False


def set_fast_cuda_flags(tf32: bool = True, cudnn_bench: bool = True):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cudnn.benchmark = bool(cudnn_bench)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def _set_worker_env(worker_id: int):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def normalize_dish_id(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"^MTL-\d+-", "", s)
    return s


def clamp_box(x1, y1, x2, y2, W, H) -> Optional[Tuple[int, int, int, int]]:
    x1 = int(round(float(x1))); y1 = int(round(float(y1)))
    x2 = int(round(float(x2))); y2 = int(round(float(y2)))
    x1 = max(0, min(x1, W - 1)); y1 = max(0, min(y1, H - 1))
    x2 = max(0, min(x2, W));     y2 = max(0, min(y2, H))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return x1, y1, x2, y2


def bbox_iou_np(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / (area_a + area_b - inter + 1e-6))


def area_ratio_np(a0: float, a1: float) -> float:
    a0 = max(float(a0), 1e-6)
    a1 = max(float(a1), 1e-6)
    return max(a1 / a0, a0 / a1)


def center_dist_np(cx0: float, cy0: float, cx1: float, cy1: float) -> float:
    dx = float(cx1 - cx0)
    dy = float(cy1 - cy0)
    return float((dx * dx + dy * dy) ** 0.5)


def scan_tracks_files(tracks_root: Path) -> List[Path]:
    pat = str(Path(tracks_root) / "*" / "P*_FP*" / "tracks.csv")
    return sorted([Path(x) for x in glob.glob(pat)])


def _pick_time_col(df: pd.DataFrame) -> Optional[str]:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for cand in ["time_min", "tmin", "minute", "minutes", "t_min", "time", "min"]:
        if cand in lower:
            return lower[cand]
    return None


def load_native_tracks_one(path: Path, min_conf: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {str(c).lower(): str(c) for c in df.columns}

    need = ["track_id", "x1", "y1", "x2", "y2", "filename"]
    for k in need:
        if k not in lower:
            raise ValueError(f"missing col {k}")

    tcol = _pick_time_col(df)
    conf_col = lower.get("conf", None)

    cols = [lower["track_id"], lower["x1"], lower["y1"], lower["x2"], lower["y2"], lower["filename"]]
    sub = df[cols].copy()
    sub.columns = ["track_id", "x1", "y1", "x2", "y2", "filename"]

    sub["tmin_raw"] = np.nan
    if tcol is not None and tcol in df.columns:
        sub["tmin_raw"] = pd.to_numeric(df[tcol], errors="coerce")

    if conf_col is not None and conf_col in df.columns:
        sub["conf"] = pd.to_numeric(df[conf_col], errors="coerce").fillna(0.0).astype(float)
    else:
        sub["conf"] = 1.0

    parsed = sub["filename"].astype(str).map(parse_from_filename)
    ok = parsed.map(lambda x: x[0] is not None)
    sub = sub.loc[ok].copy()
    parsed = parsed.loc[ok]

    sub["dish"] = parsed.map(lambda x: x[0]).astype(str).to_numpy()
    sub["p"] = parsed.map(lambda x: x[1]).astype(int).to_numpy()
    sub["fp"] = parsed.map(lambda x: x[2]).astype(int).to_numpy()
    sub["t_from_fn"] = parsed.map(lambda x: x[3]).astype(int).to_numpy()

    sub["tmin"] = sub["tmin_raw"]
    na = sub["tmin"].isna()
    if na.any():
        sub.loc[na, "tmin"] = sub.loc[na, "t_from_fn"]
    sub["tmin"] = pd.to_numeric(sub["tmin"], errors="coerce")

    for c in ["track_id", "x1", "y1", "x2", "y2"]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=["track_id", "tmin", "x1", "y1", "x2", "y2"])
    sub["track_id"] = sub["track_id"].astype(int)
    sub["tmin"] = sub["tmin"].astype(int)

    sub = sub[sub["conf"] >= float(min_conf)].copy()

    sub = sub.sort_values(
        ["dish", "p", "fp", "track_id", "tmin", "conf"],
        ascending=[True, True, True, True, True, False]
    )
    sub = sub.drop_duplicates(["dish", "p", "fp", "track_id", "tmin"], keep="first").reset_index(drop=True)

    sub["cx"] = (sub["x1"] + sub["x2"]) * 0.5
    sub["cy"] = (sub["y1"] + sub["y2"]) * 0.5
    sub["area"] = np.maximum(1.0, (sub["x2"] - sub["x1"]) * (sub["y2"] - sub["y1"]))
    return sub


def load_native_tracks_many(tracks_root: Path, max_files: int, min_conf: float) -> pd.DataFrame:
    files = scan_tracks_files(tracks_root)
    if not files:
        die(f"No tracks.csv under {tracks_root}")
    if int(max_files) > 0:
        files = files[:int(max_files)]
    info(f"[scan] tracks.csv files: {len(files)}")

    dfs = []
    loaded = 0
    skipped = 0
    for f in tqdm(files, desc="load_tracks", dynamic_ncols=True):
        try:
            d = load_native_tracks_one(f, min_conf=min_conf)
            if len(d):
                dfs.append(d)
                loaded += 1
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            info(f"[warn] skip {f}: {type(e).__name__}: {e}")
            continue

    if not dfs:
        die("No valid tracks loaded.")
    out = pd.concat(dfs, ignore_index=True)
    info(f"[scan] loaded_files={loaded} skipped_files={skipped} rows={len(out)}")
    return out


def split_stable_segments(
    trk: pd.DataFrame,
    gap_min: float,
    jump_px: float,
    iou_min: float,
    area_ratio_max: float
) -> List[Tuple[int, int]]:
    n = len(trk)
    if n <= 1:
        return [(0, n)]

    cuts = [0]
    for i in range(1, n):
        t0 = float(trk.loc[i - 1, "tmin"])
        t1 = float(trk.loc[i, "tmin"])
        if (t1 - t0) > float(gap_min):
            cuts.append(i)
            continue

        dx = float(trk.loc[i, "cx"] - trk.loc[i - 1, "cx"])
        dy = float(trk.loc[i, "cy"] - trk.loc[i - 1, "cy"])
        if (dx * dx + dy * dy) ** 0.5 > float(jump_px):
            cuts.append(i)
            continue

        a0 = float(trk.loc[i - 1, "area"]); a1 = float(trk.loc[i, "area"])
        r = max(a1 / max(a0, 1e-6), a0 / max(a1, 1e-6))
        if r > float(area_ratio_max):
            cuts.append(i)
            continue

        box0 = trk.loc[i - 1, ["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        box1 = trk.loc[i, ["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        if bbox_iou_np(box0, box1) < float(iou_min):
            cuts.append(i)
            continue

    cuts.append(n)
    return [(s, e) for s, e in zip(cuts[:-1], cuts[1:]) if e > s]


def build_stable_segments(
    df_tracks: pd.DataFrame,
    gap_min: float,
    jump_px: float,
    iou_min: float,
    area_ratio_max: float,
    min_seg_len: int,
    max_seg_len: int
) -> pd.DataFrame:
    rows = []
    key = ["dish", "p", "fp", "track_id"]

    for (dish, p, fp, tid), g in tqdm(df_tracks.groupby(key, sort=False), desc="stable_segments", dynamic_ncols=True):
        g = g.sort_values("tmin").drop_duplicates(["tmin"], keep="first").reset_index(drop=True)
        if len(g) == 0:
            continue

        segs = split_stable_segments(g, gap_min, jump_px, iou_min, area_ratio_max)
        seg_id = 0

        for s, e in segs:
            n = e - s
            if n < int(min_seg_len):
                continue

            if int(max_seg_len) > 0 and n > int(max_seg_len):
                k = int(max_seg_len)
                for ss in range(s, e, k):
                    ee = min(e, ss + k)
                    if (ee - ss) < int(min_seg_len):
                        continue
                    id_key = f"{dish}_P{int(p)}_FP{int(fp)}_T{int(tid)}_S{seg_id}"
                    sub = g.iloc[ss:ee]
                    for rr in sub.itertuples(index=False):
                        rows.append({
                            "dish": dish,
                            "p": int(p),
                            "fp": int(fp),
                            "track_id": int(tid),
                            "tmin": int(rr.tmin),
                            "seg_id": int(seg_id),
                            "id_key": id_key,
                        })
                    seg_id += 1
            else:
                id_key = f"{dish}_P{int(p)}_FP{int(fp)}_T{int(tid)}_S{seg_id}"
                sub = g.iloc[s:e]
                for rr in sub.itertuples(index=False):
                    rows.append({
                        "dish": dish,
                        "p": int(p),
                        "fp": int(fp),
                        "track_id": int(tid),
                        "tmin": int(rr.tmin),
                        "seg_id": int(seg_id),
                        "id_key": id_key,
                    })
                seg_id += 1

    out = pd.DataFrame(rows)
    if len(out) == 0:
        die("No stable segments produced. Loosen thresholds or lower min_seg_len.")
    return out


def _normalize_pathish(s: str) -> str:
    s = str(s).strip().replace("\\", "/")
    s = re.sub(r"/+", "/", s)
    return s


def _resolve_patch_meta_path(patch_meta: Path, patch_root: Path) -> Optional[Path]:
    if patch_meta.is_file():
        return patch_meta

    candidates = [
        patch_root / "patch_meta_bovine.csv",
        patch_root / "patch_meta.csv",
        patch_root / "patch_meta_native.csv",
        Path("exp/reid_patches/patch_meta_bovine.csv"),
        Path("exp/reid_patches/patch_meta.csv"),
        Path("exp/reid_patches_native/patch_meta_native.csv"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def read_patch_meta(patch_meta_csv: Path, patch_root: Path) -> pd.DataFrame:
    pm = pd.read_csv(patch_meta_csv)
    need = {"dish", "p", "fp", "track_id", "tmin", "patch_path"}
    miss = need - set(pm.columns)
    if miss:
        die(f"patch_meta missing cols: {sorted(list(miss))}")

    pm = pm.copy()
    for c in ["p", "fp", "track_id", "tmin"]:
        pm[c] = pd.to_numeric(pm[c], errors="coerce").astype(int)
    pm["dish"] = pm["dish"].astype(str)
    pm["patch_path"] = pm["patch_path"].astype(str)

    marker1 = "reid_patches/"
    marker2 = "reid_patches_native/"
    pr = _normalize_pathish(str(patch_root))

    def rewrite_one(p: str) -> str:
        p0 = _normalize_pathish(p)
        if (not p0) or p0.lower() == "nan":
            return p0
        if marker2 in p0:
            rel = p0.split(marker2, 1)[1].lstrip("/\\")
            return str(Path(patch_root) / rel)
        if marker1 in p0:
            rel = p0.split(marker1, 1)[1].lstrip("/\\")
            return str(Path(patch_root) / rel)
        if _normalize_pathish(p0).startswith(pr):
            return p0
        if not p0.startswith("/"):
            return str(Path(patch_root) / p0)
        return p0

    pm["patch_path"] = pm["patch_path"].map(rewrite_one)
    return pm


def build_patch_meta_preflight(
    df_tracks: pd.DataFrame,
    pm: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    track_wells = df_tracks[["dish", "p", "fp"]].drop_duplicates().copy()
    track_wells["dish_norm"] = track_wells["dish"].astype(str).map(normalize_dish_id)

    pm_wells = pm[["dish", "p", "fp"]].drop_duplicates().copy()
    pm_wells["dish_norm"] = pm_wells["dish"].astype(str).map(normalize_dish_id)

    merged = track_wells.merge(
        pm_wells[["dish_norm", "p", "fp"]],
        on=["dish_norm", "p", "fp"],
        how="left",
        indicator=True
    )

    missing = merged[merged["_merge"] == "left_only"][["dish", "dish_norm", "p", "fp"]].copy()
    missing["well"] = missing["dish"].astype(str) + " P" + missing["p"].astype(str) + " FP" + missing["fp"].astype(str)

    matched = merged[merged["_merge"] == "both"].copy()
    cov = len(matched) / max(1, len(track_wells))
    return matched, missing, float(cov)


class ReIDModelR50(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 256, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.bnneck = nn.BatchNorm1d(in_features)
        self.embed_head = nn.Linear(in_features, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        feat_bn = self.bnneck(feat)
        emb = self.embed_head(feat_bn)
        emb = F.normalize(emb, dim=1)
        logits = self.classifier(emb)
        return emb, logits


def load_reid_best(ckpt_path: Path, device: str, use_dp: bool) -> Tuple[nn.Module, int, Dict[str, int]]:
    if not ckpt_path.is_file():
        die(f"ReID ckpt not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)  # type: ignore
    embed_dim = int(ckpt.get("embed_dim", 256))
    id2idx = ckpt.get("id2idx", {}) or {}
    num_classes = max(1, len(id2idx))

    model = ReIDModelR50(num_classes=num_classes, embed_dim=embed_dim, pretrained=False)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt.get("model", ckpt)))
    if isinstance(state, dict):
        model.load_state_dict(state, strict=False)

    model.eval().to(device)
    if device.startswith("cuda") and use_dp and torch.cuda.device_count() > 1:
        info(f"[reid] DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    return model, embed_dim, id2idx


def _autocast_ctx(device: str, use_amp: bool):
    if (not device.startswith("cuda")) or (not use_amp):
        class _Null:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False
        return _Null()
    return torch.amp.autocast("cuda", dtype=torch.float16)


def _read_rgb_u8(path: str) -> torch.Tensor:
    if _HAS_TV_READ:
        try:
            return read_image(path, mode=ImageReadMode.RGB)
        except Exception:
            pass

    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def crop_patch_to_tmp(
    big_root: Path,
    dish: str,
    p: int,
    fp: int,
    tmin: int,
    x1, y1, x2, y2,
    tmp_root: Path
) -> Optional[str]:
    bigp = big_image_path(big_root, dish, p, fp, tmin)
    if not bigp.is_file():
        return None

    out_dir = tmp_root / f"{dish}_P{p}_FP{fp}"
    ensure_dir(out_dir)

    out_path = out_dir / f"{dish}_P{p}_FP{fp}_T{tmin}_X{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}.jpg"
    if out_path.is_file():
        return str(out_path)

    try:
        with Image.open(bigp) as im:
            im = im.convert("RGB")
            W, H = im.size
            box = clamp_box(x1, y1, x2, y2, W, H)
            if box is None:
                return None
            crop = im.crop(box).resize((224, 224), resample=Image.BILINEAR)
            crop.save(out_path, quality=92)
        return str(out_path)
    except Exception:
        return None


def _hash_list(xs: List[str]) -> str:
    h = hashlib.sha1()
    for x in xs:
        h.update(x.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()[:12]


def save_emb_cache_npz(cache: Dict[str, np.ndarray], out_path: Path):
    keys = list(cache.keys())
    if not keys:
        return
    embs = np.stack([cache[k] for k in keys], 0).astype(np.float32)
    np.savez_compressed(out_path, paths=np.array(keys, dtype=object), embs=embs)


def load_emb_cache_npz(in_path: Path) -> Dict[str, np.ndarray]:
    z = np.load(in_path, allow_pickle=True)
    paths = z["paths"].tolist()
    embs = z["embs"].astype(np.float32)
    return {str(p): embs[i] for i, p in enumerate(paths)}


class PathDataset(Dataset):
    def __init__(self, paths: List[str], tfm, skip_bad: bool):
        self.paths = paths
        self.tfm = tfm
        self.skip_bad = bool(skip_bad)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        try:
            x = self.tfm(_read_rgb_u8(p))
            return x, p, True
        except Exception:
            if not self.skip_bad:
                raise
            return torch.zeros(3, 224, 224, dtype=torch.float32), p, False


def _collate_drop_bad(batch):
    xs, ps = [], []
    for x, p, ok in batch:
        if ok:
            xs.append(x)
            ps.append(p)
    if not xs:
        return torch.empty(0), []
    return torch.stack(xs, 0), ps


@torch.no_grad()
def infer_embeddings(
    model: nn.Module,
    tfm,
    paths: List[str],
    batch_size: int,
    num_workers: int,
    device: str,
    use_amp: bool,
    skip_bad: bool
) -> Dict[str, np.ndarray]:
    paths = [str(p) for p in paths if p and p != "nan" and _is_file_fast(p)]
    if not paths:
        return {}

    ds = PathDataset(paths, tfm, skip_bad=skip_bad)
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.startswith("cuda"),
        persistent_workers=(int(num_workers) > 0),
        prefetch_factor=4 if int(num_workers) > 0 else 2,
        collate_fn=_collate_drop_bad,
        drop_last=False,
        worker_init_fn=_set_worker_env if int(num_workers) > 0 else None,
    )

    cache: Dict[str, np.ndarray] = {}
    for x, plist in tqdm(dl, desc="infer_emb", dynamic_ncols=True, leave=False):
        if len(plist) == 0:
            continue
        x = x.to(device, non_blocking=True)
        with _autocast_ctx(device, use_amp):
            emb, _ = model(x)
        emb = emb.detach().float().cpu().numpy().astype(np.float32)
        n = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / n
        for e, p in zip(emb, plist):
            cache[str(p)] = e.astype(np.float32)
    return cache


@dataclass
class SplitCfg:
    enabled: bool = True
    cos_th: float = 0.35
    consec: int = 2
    win: int = 1
    min_seg_len: int = 2
    require_geom_break: bool = True
    guard_max_gap: float = 20.0
    guard_center_max: float = 45.0
    guard_iou_min: float = 0.25
    guard_area_ratio_max: float = 1.35
    guard_min_left_len: int = 8
    guard_min_right_len: int = 8


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _window_mean(embs: List[Optional[np.ndarray]], l: int, r: int) -> Optional[np.ndarray]:
    vecs = [e for e in embs[l:r] if e is not None]
    if not vecs:
        return None
    m = np.mean(np.stack(vecs, 0), 0).astype(np.float32)
    n = float(np.linalg.norm(m)) + 1e-12
    return (m / n).astype(np.float32)


def _boundary_geom_break(g: pd.DataFrame, cut_i: int, split: SplitCfg) -> Tuple[bool, Dict[str, float]]:
    if cut_i <= 0 or cut_i >= len(g):
        return True, {
            "gap": np.nan,
            "center_dist": np.nan,
            "iou": np.nan,
            "area_ratio": np.nan,
            "stable_boundary": 0.0,
        }

    a = g.iloc[cut_i - 1]
    b = g.iloc[cut_i]

    gap = float(b["tmin"] - a["tmin"])
    dist = center_dist_np(float(a["cx"]), float(a["cy"]), float(b["cx"]), float(b["cy"]))
    iou = bbox_iou_np(
        np.array([a["x1"], a["y1"], a["x2"], a["y2"]], dtype=float),
        np.array([b["x1"], b["y1"], b["x2"], b["y2"]], dtype=float),
    )
    ar = area_ratio_np(float(a["area"]), float(b["area"]))

    stable_boundary = (
        (gap <= float(split.guard_max_gap)) and
        (dist <= float(split.guard_center_max)) and
        (iou >= float(split.guard_iou_min)) and
        (ar <= float(split.guard_area_ratio_max))
    )
    return (not stable_boundary), {
        "gap": gap,
        "center_dist": dist,
        "iou": iou,
        "area_ratio": ar,
        "stable_boundary": float(1.0 if stable_boundary else 0.0),
    }


def appearance_split_segments(df_well: pd.DataFrame, emb_cache: Dict[str, np.ndarray], split: SplitCfg) -> pd.Series:
    idx_orig = df_well.index
    if not split.enabled:
        return pd.Series(df_well["track_id_raw"].astype("int64").to_numpy(), index=idx_orig, name="track_id_seg")

    df_sorted = df_well.sort_values("tmin").copy()
    idx_sorted = df_sorted.index.to_numpy()
    out_sorted = np.empty(len(df_sorted), dtype=np.int64)

    next_seg = 0

    for _, g in df_sorted.groupby("track_id_raw", sort=False):
        g = g.sort_values("tmin").reset_index(drop=False)
        ridx = g["index"].to_numpy()
        n = len(g)

        if n <= 1:
            out_sorted[np.isin(idx_sorted, ridx)] = next_seg
            next_seg += 1
            continue

        embs = [emb_cache.get(pp) for pp in g["patch_path"].astype(str).tolist()]
        seg_start = 0
        susp_run = 0

        for i in range(1, n):
            w = int(split.win)
            left = _window_mean(embs, max(0, i - w), i)
            right = _window_mean(embs, i, min(n, i + w))
            if left is None or right is None:
                susp = False
            else:
                susp = (cosine_np(left, right) < float(split.cos_th))

            susp_run = susp_run + 1 if susp else 0

            if susp_run >= int(split.consec):
                cut_i = i
                left_len = cut_i - seg_start
                right_len = n - cut_i
                allow_split = True

                if left_len < int(split.min_seg_len):
                    allow_split = False

                if split.require_geom_break and allow_split:
                    geom_break, _dbg = _boundary_geom_break(g, cut_i, split)
                    if (
                        (left_len >= int(split.guard_min_left_len)) and
                        (right_len >= int(split.guard_min_right_len)) and
                        (not geom_break)
                    ):
                        allow_split = False

                if allow_split:
                    seg_ids = ridx[seg_start:cut_i]
                    out_sorted[np.isin(idx_sorted, seg_ids)] = next_seg
                    next_seg += 1
                    seg_start = cut_i
                    susp_run = 0

        seg_ids = ridx[seg_start:]
        out_sorted[np.isin(idx_sorted, seg_ids)] = next_seg
        next_seg += 1

    s_sorted = pd.Series(out_sorted, index=df_sorted.index, name="track_id_seg").astype("int64")
    return s_sorted.reindex(idx_orig)


@dataclass
class Segment:
    seg_id: int
    t0: float
    t1: float
    proto: np.ndarray
    seq: np.ndarray
    n_rows: int


def _proto_from_rows(g: pd.DataFrame, emb_cache: Dict[str, np.ndarray], sample_max: int) -> Optional[np.ndarray]:
    pps = g["patch_path"].astype(str).dropna().unique().tolist()
    if len(pps) > int(sample_max):
        pps = pps[:int(sample_max)]
    vecs = [emb_cache.get(pp) for pp in pps if emb_cache.get(pp) is not None]
    if not vecs:
        return None
    m = np.mean(np.stack(vecs, 0), 0).astype(np.float32)
    m = m / (np.linalg.norm(m) + 1e-12)
    return m.astype(np.float32)


def _seq_samples_from_rows(g: pd.DataFrame, emb_cache: Dict[str, np.ndarray], n_seq: int, max_scan: int) -> Optional[np.ndarray]:
    gg = g.sort_values("tmin")
    pps = gg["patch_path"].astype(str).tolist()
    vecs = []
    for pp in pps[:int(max_scan)]:
        e = emb_cache.get(pp)
        if e is not None:
            vecs.append(e)
    if not vecs:
        return None
    if len(vecs) > int(n_seq):
        idx = np.linspace(0, len(vecs) - 1, num=int(n_seq))
        idx = np.round(idx).astype(int).tolist()
        vecs = [vecs[i] for i in idx]
    seq = np.stack(vecs, 0).astype(np.float32)
    n = np.linalg.norm(seq, axis=1, keepdims=True) + 1e-12
    seq = seq / n
    return seq.astype(np.float32)


def build_segments(
    df_well: pd.DataFrame,
    seg_col: str,
    emb_cache: Dict[str, np.ndarray],
    proto_sample_max: int,
    seq_sample_n: int,
    seq_scan_max: int
) -> List[Segment]:
    segs = []
    for sid, g in df_well.groupby(seg_col, sort=False):
        proto = _proto_from_rows(g, emb_cache, proto_sample_max)
        seq = _seq_samples_from_rows(g, emb_cache, seq_sample_n, seq_scan_max)
        if proto is None or seq is None:
            continue
        segs.append(Segment(
            seg_id=int(sid),
            t0=float(g["tmin"].min()),
            t1=float(g["tmin"].max()),
            proto=proto,
            seq=seq,
            n_rows=int(len(g)),
        ))
    segs.sort(key=lambda s: (s.t0, s.t1, s.seg_id))
    return segs


def intervals_overlap(a0, a1, b0, b1, eps=1e-9) -> bool:
    return (a0 <= b1 + eps) and (b0 <= a1 + eps)


def build_conflicts(segs: List[Segment]) -> List[set]:
    n = len(segs)
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if intervals_overlap(segs[i].t0, segs[i].t1, segs[j].t0, segs[j].t1):
                conflicts[i].add(j)
                conflicts[j].add(i)
    return conflicts


def seq_sim_topk(a_seq: np.ndarray, b_seq: np.ndarray, topk: int) -> float:
    if a_seq.size == 0 or b_seq.size == 0:
        return -1.0
    sim = (a_seq @ b_seq.T).astype(np.float32)
    flat = sim.reshape(-1)
    k = int(max(1, min(int(topk), flat.size)))
    idx = np.argpartition(flat, -k)[-k:]
    return float(np.mean(flat[idx]))


def reward_pair(a: Segment, b: Segment, w_proto: float, w_seq: float, w_gap: float, seq_topk: int) -> Tuple[float, float, float]:
    gap = max(0.0, float(b.t0 - a.t1))
    sim_p = float(np.dot(a.proto, b.proto))
    sim_s = seq_sim_topk(a.seq, b.seq, topk=int(seq_topk))
    rew = float(w_proto * sim_p + w_seq * sim_s - w_gap * math.log1p(gap))
    return sim_p, sim_s, rew


def build_successor_rewards(
    segs: List[Segment],
    conflicts: List[set],
    max_link_gap: float,
    w_proto: float,
    w_seq: float,
    w_gap: float,
    min_link_cos: float,
    min_link_seq: float,
    seq_topk: int
) -> Dict[Tuple[int, int], float]:
    n = len(segs)
    rew = {}
    for i in range(n):
        for j in range(i + 1, n):
            if j in conflicts[i]:
                continue
            if segs[j].t0 < segs[i].t1:
                continue

            gap = float(segs[j].t0 - segs[i].t1)
            if gap > float(max_link_gap):
                continue

            sim_p = float(np.dot(segs[i].proto, segs[j].proto))
            if float(min_link_cos) > -0.5 and sim_p < float(min_link_cos):
                continue

            sim_s = seq_sim_topk(segs[i].seq, segs[j].seq, topk=int(seq_topk))
            if float(min_link_seq) > -0.5 and sim_s < float(min_link_seq):
                continue

            _, _, r = reward_pair(segs[i], segs[j], w_proto, w_seq, w_gap, seq_topk)
            rew[(i, j)] = r
    return rew


def solve_coloring_branch_and_bound(
    segs: List[Segment],
    conflicts: List[set],
    K_real: int,
    rewards: Dict[Tuple[int, int], float],
    seed: int,
    allow_unassigned: bool,
    unassigned_penalty: float,
    min_attach_reward: float,
) -> Tuple[List[int], float]:
    rng = np.random.RandomState(int(seed))
    n = len(segs)
    if n == 0:
        return [], 0.0
    if K_real <= 0:
        raise ValueError("K_real must be >= 1")

    INF_NEG = 1e9

    def seg_len(i: int) -> float:
        return float(segs[i].t1 - segs[i].t0)

    order = list(range(n))
    order.sort(key=lambda i: (-seg_len(i), -segs[i].n_rows, -len(conflicts[i]), segs[i].t0, segs[i].t1, segs[i].seg_id))

    colors = [-2] * n
    best_colors = [-2] * n
    best_score = -1e30

    last_in_color = [-1] * K_real
    used = [0] * K_real

    max_incoming = [0.0] * n
    for j in range(n):
        best = 0.0
        for i in range(n):
            if i == j:
                continue
            r = rewards.get((i, j), None)
            if r is not None and float(r) > best:
                best = float(r)
        max_incoming[j] = best

    assigned = np.zeros(n, dtype=bool)

    def upper_bound(cur: float) -> float:
        s = 0.0
        for idx in range(n):
            if not assigned[idx]:
                s += max_incoming[idx]
        return cur + s

    def feasible(i: int, c: int) -> bool:
        for j in conflicts[i]:
            if colors[j] == c:
                return False
        return True

    def delta(i: int, c: int) -> float:
        if c < 0:
            return -float(unassigned_penalty)

        last = last_in_color[c]
        if last < 0:
            return 0.0

        if segs[i].t0 < segs[last].t0 - 1e-9:
            return -INF_NEG
        if (i in conflicts[last]) or (last in conflicts[i]):
            return -INF_NEG

        key = (last, i)
        if key not in rewards:
            return -INF_NEG

        r = float(rewards[key])
        if r < float(min_attach_reward):
            return -INF_NEG

        return r

    def backtrack(pos: int, cur: float):
        nonlocal best_score, best_colors
        if pos == n:
            if cur > best_score:
                best_score = cur
                best_colors = colors.copy()
            return

        if upper_bound(cur) <= best_score + 1e-9:
            return

        i = order[pos]
        assigned[i] = True

        cand = list(range(K_real))
        if allow_unassigned:
            cand.append(-1)

        scored = []
        for c in cand:
            if c >= 0 and (not feasible(i, c)):
                continue
            d = delta(i, c)
            if d < -1e8:
                continue
            used_bonus = 1e-4 * (1 if (c >= 0 and used[c] == 1) else 0)
            scored.append((float(d) + used_bonus, float(d), int(c)))

        if not scored:
            colors[i] = -1
            backtrack(pos + 1, cur - float(unassigned_penalty))
            colors[i] = -2
            assigned[i] = False
            return

        scored.sort(key=lambda x: (-x[0], x[1]))
        if len(scored) > 1:
            rot = int(rng.randint(0, len(scored)))
            scored = scored[rot:] + scored[:rot]

        for _, d_step, c in scored:
            if c >= 0:
                pl = last_in_color[c]
                pu = used[c]
                colors[i] = c
                used[c] = 1
                last_in_color[c] = i
                backtrack(pos + 1, cur + float(d_step))
                last_in_color[c] = pl
                used[c] = pu
                colors[i] = -2
            else:
                colors[i] = -1
                backtrack(pos + 1, cur + float(d_step))
                colors[i] = -2

        assigned[i] = False

    backtrack(0, 0.0)
    if best_score < -1e20:
        return ([-1] * n if allow_unassigned else [0] * n), 0.0
    return best_colors, float(best_score)


def compute_segment_confidence(segs: List[Segment], colors: List[int], rewards: Dict[Tuple[int, int], float]) -> Dict[int, float]:
    n = len(segs)
    conf = {i: 0.0 for i in range(n)}
    chains = {}
    for i in range(n):
        c = int(colors[i])
        if c < 0:
            conf[i] = 0.0
        else:
            chains.setdefault(c, []).append(i)

    for c, idxs in chains.items():
        idxs.sort(key=lambda i: (segs[i].t0, segs[i].t1))
        for k, i in enumerate(idxs):
            vals = []
            if k - 1 >= 0:
                j = idxs[k - 1]
                vals.append(float(rewards.get((j, i), 0.0)))
            if k + 1 < len(idxs):
                j = idxs[k + 1]
                vals.append(float(rewards.get((i, j), 0.0)))
            if vals:
                v = max(vals)
                conf[i] = float(1.0 / (1.0 + math.exp(-3.0 * v)))
            else:
                conf[i] = 0.5
    return conf


@dataclass
class RescueCfg:
    enabled: bool = True
    min_reward: float = 0.80
    min_proto: float = 0.60
    min_seq: float = 0.60
    require_free_color: bool = True
    require_prev_tail: bool = True
    rollback_overlap_conflict: bool = True


def sigmoid_conf(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-3.0 * float(x))))


def apply_conservative_free_rescue(
    segs: List[Segment],
    strict_colors: List[int],
    conflicts: List[set],
    rewards: Dict[Tuple[int, int], float],
    K_real: int,
    w_proto: float,
    w_seq: float,
    w_gap: float,
    seq_topk: int,
    cfg: RescueCfg,
) -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
    n = len(segs)
    strict_colors = [int(c) for c in strict_colors]
    rescued_colors = strict_colors.copy()

    rescue_meta_by_segidx: Dict[int, Dict[str, Any]] = {}
    for i in range(n):
        rescue_meta_by_segidx[i] = {
            "rescue_applied": False,
            "rescue_kept_safe": False,
            "rescue_rolled_back_due_to_conflict": False,
            "safe_status_note": "",
            "rescue_from_prev_seg": -1,
            "rescue_to_color": -1,
            "rescue_reward": np.nan,
            "rescue_proto": np.nan,
            "rescue_seq": np.nan,
        }

    if (not cfg.enabled) or n == 0:
        return rescued_colors, rescue_meta_by_segidx

    chain_by_color: Dict[int, List[int]] = {}
    for i, c in enumerate(strict_colors):
        if c >= 0:
            chain_by_color.setdefault(int(c), []).append(i)
    for c in chain_by_color:
        chain_by_color[c].sort(key=lambda k: (segs[k].t0, segs[k].t1, segs[k].seg_id))
    tail_of_color = {c: idxs[-1] for c, idxs in chain_by_color.items() if len(idxs)}

    proposed = []

    for i in range(n):
        if strict_colors[i] >= 0:
            continue

        cur = segs[i]
        t_cur = float(cur.t0)

        active_colors = set()
        for k, c in enumerate(strict_colors):
            if c < 0:
                continue
            if segs[k].t0 <= t_cur <= segs[k].t1:
                active_colors.add(int(c))

        free_color_est = int(K_real) - len(active_colors)

        best_prev = -1
        best_color = -1
        best_reward = -1e18
        best_proto = np.nan
        best_seq = np.nan

        for j in range(n):
            c = strict_colors[j]
            if c < 0:
                continue
            if j == i:
                continue

            prev = segs[j]
            if prev.t1 > cur.t0 + 1e-9:
                continue
            if (i in conflicts[j]) or (j in conflicts[i]):
                continue

            key = (j, i)
            if key not in rewards:
                continue

            if cfg.require_prev_tail and tail_of_color.get(int(c), -1) != j:
                continue

            sim_p, sim_s, rew = reward_pair(prev, cur, w_proto, w_seq, w_gap, seq_topk)
            if sim_p < float(cfg.min_proto):
                continue
            if sim_s < float(cfg.min_seq):
                continue
            if rew < float(cfg.min_reward):
                continue
            if cfg.require_free_color and free_color_est <= 0:
                continue

            if rew > best_reward:
                best_prev = int(j)
                best_color = int(c)
                best_reward = float(rew)
                best_proto = float(sim_p)
                best_seq = float(sim_s)

        if best_prev >= 0 and best_color >= 0:
            rescued_colors[i] = best_color
            rescue_meta_by_segidx[i]["rescue_applied"] = True
            rescue_meta_by_segidx[i]["rescue_kept_safe"] = True
            rescue_meta_by_segidx[i]["safe_status_note"] = "rescued_free_color"
            rescue_meta_by_segidx[i]["rescue_from_prev_seg"] = int(segs[best_prev].seg_id)
            rescue_meta_by_segidx[i]["rescue_to_color"] = int(best_color)
            rescue_meta_by_segidx[i]["rescue_reward"] = float(best_reward)
            rescue_meta_by_segidx[i]["rescue_proto"] = float(best_proto)
            rescue_meta_by_segidx[i]["rescue_seq"] = float(best_seq)
            proposed.append(i)

    if cfg.rollback_overlap_conflict and proposed:
        rollback_set = set()
        for i in proposed:
            ci = rescued_colors[i]
            for k in range(n):
                if k == i:
                    continue
                if rescued_colors[k] != ci:
                    continue
                if intervals_overlap(segs[i].t0, segs[i].t1, segs[k].t0, segs[k].t1):
                    rollback_set.add(i)
                    break

        for i in rollback_set:
            rescued_colors[i] = -1
            rescue_meta_by_segidx[i]["rescue_kept_safe"] = False
            rescue_meta_by_segidx[i]["rescue_rolled_back_due_to_conflict"] = True
            rescue_meta_by_segidx[i]["safe_status_note"] = "rolled_back_overlap_conflict"

    return rescued_colors, rescue_meta_by_segidx


@dataclass
class PromoteCfg:
    enabled: bool = True
    orphan_min_rows: int = 150
    orphan_min_duration: float = 1000.0
    orphan_min_conf: float = 0.85
    max_holder_rows: int = 120
    max_total_demote_rows: int = 220
    min_candidate_to_holder_row_ratio: float = 3.0
    promoted_conf_floor: float = 0.75
    require_kreal_ge_3: bool = True


def build_seg_stats_from_df(df_well: pd.DataFrame, seg_col: str = "track_id_seg") -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    tmp = (
        df_well.groupby(seg_col, sort=False)
        .agg(
            rows=("tmin", "size"),
            t0=("tmin", "min"),
            t1=("tmin", "max"),
            conf_mean=("conf", "mean"),
        )
        .reset_index()
    )
    for _, r in tmp.iterrows():
        sid = int(r[seg_col])
        out[sid] = {
            "rows": int(r["rows"]),
            "t0": float(r["t0"]),
            "t1": float(r["t1"]),
            "duration": float(r["t1"] - r["t0"]),
            "conf_mean": float(r["conf_mean"]),
        }
    return out


def apply_long_lived_orphan_promotion(
    segs: List[Segment],
    df_well: pd.DataFrame,
    strict_colors: List[int],
    strict_conf_by_segidx: Dict[int, float],
    K_real: int,
    cfg: PromoteCfg,
) -> Tuple[List[int], Dict[int, Dict[str, Any]], Dict[int, float]]:
    n = len(segs)
    colors = [int(c) for c in strict_colors]
    conf_map = dict(strict_conf_by_segidx)

    meta: Dict[int, Dict[str, Any]] = {}
    for i in range(n):
        meta[i] = {
            "promotion_applied": False,
            "promoted_to_color": -1,
            "promoted_reason": "",
            "demoted_by_promotion": False,
            "demoted_from_color": -1,
            "demoted_by_seg": -1,
            "promotion_demoted_seg_ids": [],
        }

    if (not cfg.enabled) or n == 0:
        return colors, meta, conf_map

    if cfg.require_kreal_ge_3 and int(K_real) < 3:
        return colors, meta, conf_map

    seg_stats = build_seg_stats_from_df(df_well, seg_col="track_id_seg")

    orphan_idxs = []
    for i, s in enumerate(segs):
        if colors[i] >= 0:
            continue

        st = seg_stats.get(int(s.seg_id), None)
        if st is None:
            continue
        if int(st["rows"]) < int(cfg.orphan_min_rows):
            continue
        if float(st["duration"]) < float(cfg.orphan_min_duration):
            continue
        if float(st["conf_mean"]) < float(cfg.orphan_min_conf):
            continue

        orphan_idxs.append(i)

    orphan_idxs.sort(key=lambda i: (-segs[i].n_rows, segs[i].t0, segs[i].seg_id))
    if not orphan_idxs:
        return colors, meta, conf_map

    def overlapping_holders_for_color(target_i: int, c: int) -> List[int]:
        out = []
        for k in range(n):
            if k == target_i:
                continue
            if colors[k] != c:
                continue
            if intervals_overlap(segs[target_i].t0, segs[target_i].t1, segs[k].t0, segs[k].t1):
                out.append(k)
        return out

    for i in orphan_idxs:
        if colors[i] >= 0:
            continue

        cand = segs[i]
        cand_st = seg_stats.get(int(cand.seg_id), None)
        if cand_st is None:
            continue

        best_pick = None

        for c in range(int(K_real)):
            holders = overlapping_holders_for_color(i, c)

            if not holders:
                score = (0, 0, 0)
                best_pick = (score, c, holders)
                break

            holder_rows = [int(segs[k].n_rows) for k in holders]
            total_holder_rows = int(sum(holder_rows))
            max_holder_rows = int(max(holder_rows)) if holder_rows else 0

            if max_holder_rows > int(cfg.max_holder_rows):
                continue
            if total_holder_rows > int(cfg.max_total_demote_rows):
                continue
            if float(cand_st["rows"]) < float(cfg.min_candidate_to_holder_row_ratio) * max(1.0, float(total_holder_rows)):
                continue

            score = (total_holder_rows, max_holder_rows, len(holders))
            if (best_pick is None) or (score < best_pick[0]):
                best_pick = (score, c, holders)

        if best_pick is None:
            continue

        _, chosen_color, holders = best_pick

        demoted_seg_ids = []
        for k in holders:
            old_c = colors[k]
            colors[k] = -1
            conf_map[k] = 0.0
            meta[k]["demoted_by_promotion"] = True
            meta[k]["demoted_from_color"] = int(old_c)
            meta[k]["demoted_by_seg"] = int(cand.seg_id)
            demoted_seg_ids.append(int(segs[k].seg_id))

        colors[i] = int(chosen_color)
        conf_map[i] = max(float(conf_map.get(i, 0.0)), float(cfg.promoted_conf_floor))
        meta[i]["promotion_applied"] = True
        meta[i]["promoted_to_color"] = int(chosen_color)
        meta[i]["promoted_reason"] = "long_lived_orphan_promoted"
        meta[i]["promotion_demoted_seg_ids"] = demoted_seg_ids

    return colors, meta, conf_map


def build_final_ids_from_strict(
    wid: int,
    segs: List[Segment],
    strict_colors: List[int],
    strict_conf_by_segidx: Dict[int, float],
    force_assign_unassigned: bool,
    fallback_base: int,
    fallback_offset: int
) -> Dict[int, Dict[str, Any]]:
    out = {}
    for i, s in enumerate(segs):
        c = int(strict_colors[i])
        strict_id = -1 if c < 0 else (int(wid) * 100 + int(c))
        strict_conf = float(strict_conf_by_segidx.get(i, 0.0)) if strict_id >= 0 else 0.0

        if strict_id < 0 and bool(force_assign_unassigned):
            final_id = int(wid) * int(fallback_base) + int(fallback_offset) + int(s.seg_id)
            is_fb = True
            final_conf = 0.0
        else:
            final_id = strict_id
            is_fb = False
            final_conf = strict_conf if strict_id >= 0 else 0.0

        out[i] = {
            "track_id_final_strict": int(strict_id),
            "cluster_conf_strict": float(strict_conf),
            "track_id_final": int(final_id),
            "cluster_conf": float(final_conf),
            "is_fallback": bool(is_fb),
        }
    return out


def make_run_dir(out_root: Path, note: str = "") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"run_{ts}"
    if note:
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", note)[:60]
        name += f"_{safe}"
    d = out_root / name
    ensure_dir(d)
    return d


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Reconnect interrupted embryo tracks using ReID embeddings.")

    ap.add_argument("--tracks-root", default=_env_str("REPAIR_TRACKS_ROOT", "EmbryoTracks_BoT"))
    ap.add_argument("--big-root", default=_env_str("REPAIR_BIG_ROOT", "Bovine"))
    ap.add_argument("--patch-root", default=_env_str("REPAIR_PATCH_ROOT", "exp/reid_patches"))
    ap.add_argument("--patch-meta", default=_env_str("REPAIR_PATCH_META_CSV", "exp/reid_patches/patch_meta_bovine.csv"))
    ap.add_argument("--reid-ckpt", default=_env_str("REPAIR_REID_CKPT", "exp/reid_r50_native_stable_runC/ckpt/best.pth"))

    ap.add_argument("--out-root", default=_env_str("REPAIR_OUT_ROOT", "exp/track_repair_v53_guarded_orphanfix"))
    ap.add_argument("--note", default="v53_guarded_orphanfix_release")

    ap.add_argument("--max-tracks-files", type=int, default=0, help="0=all")
    ap.add_argument("--min-conf", type=float, default=0.0)

    ap.add_argument("--gap-min", type=float, default=22.0)
    ap.add_argument("--jump-px", type=float, default=60.0)
    ap.add_argument("--iou-min", type=float, default=0.08)
    ap.add_argument("--area-ratio-max", type=float, default=3.5)
    ap.add_argument("--min-seg-len", type=int, default=6)
    ap.add_argument("--max-seg-len", type=int, default=80)

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--no-cudnn-bench", action="store_true")
    ap.add_argument("--data-parallel", action="store_true", default=True)

    ap.add_argument("--emb-batch-size", type=int, default=1536)
    ap.add_argument("--emb-workers", type=int, default=12)
    ap.add_argument("--skip-bad-images", action="store_true", default=True)

    ap.add_argument("--tmp-patch-root", default=_env_str("REPAIR_TMP_PATCH_ROOT", "exp/track_repair_v53_tmp_patches"))

    ap.add_argument("--split-by-appearance", action="store_true", default=True)
    ap.add_argument("--split-cos-th", type=float, default=0.35)
    ap.add_argument("--split-consec", type=int, default=2)
    ap.add_argument("--split-win", type=int, default=1)
    ap.add_argument("--split-min-seg-len", type=int, default=2)

    ap.add_argument("--split-require-geom-break", action="store_true", default=True)
    ap.add_argument("--split-guard-max-gap", type=float, default=20.0)
    ap.add_argument("--split-guard-center-max", type=float, default=45.0)
    ap.add_argument("--split-guard-iou-min", type=float, default=0.25)
    ap.add_argument("--split-guard-area-ratio-max", type=float, default=1.35)
    ap.add_argument("--split-guard-min-left-len", type=int, default=8)
    ap.add_argument("--split-guard-min-right-len", type=int, default=8)

    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--k-strategy", choices=["fixed", "timeline_p95", "estimate"], default="timeline_p95")
    ap.add_argument("--k-timeline-quantile", type=float, default=0.95)
    ap.add_argument("--k-smooth-win", type=int, default=5)
    ap.add_argument("--k-min-frames", type=int, default=10)
    ap.add_argument("--k-early-window", type=float, default=2000.0)

    ap.add_argument("--w-proto", type=float, default=0.6)
    ap.add_argument("--w-seq", type=float, default=1.0)
    ap.add_argument("--w-gap", type=float, default=0.02)
    ap.add_argument("--max-link-gap", type=float, default=15000.0)
    ap.add_argument("--seq-topk", type=int, default=5)
    ap.add_argument("--link-cos-th", type=float, default=0.15)
    ap.add_argument("--link-seq-th", type=float, default=0.10)

    ap.add_argument("--proto-sample-max", type=int, default=64)
    ap.add_argument("--seq-sample-n", type=int, default=8)
    ap.add_argument("--seq-scan-max", type=int, default=64)

    ap.add_argument("--allow-unassigned", action="store_true", default=True)
    ap.add_argument("--unassigned-penalty", type=float, default=0.10)
    ap.add_argument("--min-attach-reward", type=float, default=0.25)

    ap.add_argument("--enable-free-rescue", action="store_true", default=True)
    ap.add_argument("--rescue-min-reward", type=float, default=0.80)
    ap.add_argument("--rescue-min-proto", type=float, default=0.60)
    ap.add_argument("--rescue-min-seq", type=float, default=0.60)
    ap.add_argument("--rescue-require-free-color", action="store_true", default=True)
    ap.add_argument("--rescue-require-prev-tail", action="store_true", default=True)
    ap.add_argument("--rescue-rollback-overlap", action="store_true", default=True)

    ap.add_argument("--enable-orphan-promotion", action="store_true", default=True)
    ap.add_argument("--orphan-min-rows", type=int, default=150)
    ap.add_argument("--orphan-min-duration", type=float, default=1000.0)
    ap.add_argument("--orphan-min-conf", type=float, default=0.85)
    ap.add_argument("--orphan-max-holder-rows", type=int, default=120)
    ap.add_argument("--orphan-max-total-demote-rows", type=int, default=220)
    ap.add_argument("--orphan-min-candidate-to-holder-row-ratio", type=float, default=3.0)
    ap.add_argument("--orphan-promoted-conf-floor", type=float, default=0.75)

    ap.add_argument("--force-assign-unassigned", action="store_true", default=True)
    ap.add_argument("--fallback-base", type=int, default=100000)
    ap.add_argument("--fallback-offset", type=int, default=50000)

    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    seed_all(int(args.seed))
    set_fast_cuda_flags(tf32=(not args.no_tf32), cudnn_bench=(not args.no_cudnn_bench))

    tracks_root = Path(args.tracks_root).resolve()
    big_root = Path(args.big_root).resolve()
    patch_root = Path(args.patch_root).resolve()
    patch_meta = Path(args.patch_meta).resolve()
    reid_ckpt = Path(args.reid_ckpt).resolve()
    out_root = Path(args.out_root).resolve()
    tmp_patch_root = Path(args.tmp_patch_root).resolve()

    ensure_dir(out_root)
    ensure_dir(tmp_patch_root)

    out_dir = make_run_dir(out_root, note=args.note)
    info("[OUT]", out_dir)

    df_tracks = load_native_tracks_many(
        tracks_root,
        max_files=int(args.max_tracks_files),
        min_conf=float(args.min_conf),
    )
    info(f"[tracks] rows={len(df_tracks)} wells={df_tracks[['dish','p','fp']].drop_duplicates().shape[0]}")

    stable = build_stable_segments(
        df_tracks,
        gap_min=float(args.gap_min),
        jump_px=float(args.jump_px),
        iou_min=float(args.iou_min),
        area_ratio_max=float(args.area_ratio_max),
        min_seg_len=int(args.min_seg_len),
        max_seg_len=int(args.max_seg_len),
    )
    stable.to_csv(out_dir / "stable_segments.csv", index=False)
    info(f"[stable] rows={len(stable)} ids={stable['id_key'].nunique()}")

    pm = None
    patch_meta_resolved = _resolve_patch_meta_path(patch_meta, patch_root)
    patch_meta_well_coverage = 0.0
    patch_meta_missing_wells_n = -1
    patch_meta_missing_csv = ""

    if patch_meta_resolved is not None:
        pm = read_patch_meta(patch_meta_resolved, patch_root=patch_root)
        info(f"[patch_meta] using {patch_meta_resolved}")
        info(f"[patch_meta] rows={len(pm)} wells={pm[['dish','p','fp']].drop_duplicates().shape[0]}")

        _, missing_wells_df, patch_meta_well_coverage = build_patch_meta_preflight(df_tracks, pm)
        patch_meta_missing_wells_n = int(len(missing_wells_df))
        patch_meta_missing_csv = str(out_dir / "patch_meta_missing_wells.csv")
        missing_wells_df.to_csv(patch_meta_missing_csv, index=False)

        info(f"[patch_meta] well coverage = {patch_meta_well_coverage * 100:.2f}%")
        info(f"[patch_meta] missing wells = {patch_meta_missing_wells_n}")
        info(f"[patch_meta] missing wells csv = {patch_meta_missing_csv}")

        if patch_meta_well_coverage < 0.80:
            die(f"patch_meta well coverage too low: {patch_meta_well_coverage:.4f}")
    else:
        info("[patch_meta] not found -> will crop patches on-the-fly")

    df = df_tracks.copy()
    df["dish_norm"] = df["dish"].astype(str).map(normalize_dish_id)
    df["key"] = (
        df["dish_norm"].astype(str) + "|" +
        df["p"].astype(str) + "|" +
        df["fp"].astype(str) + "|" +
        df["track_id"].astype(str) + "|" +
        df["tmin"].astype(str)
    )

    cov = 0.0
    if pm is not None:
        pm2 = pm[["dish", "p", "fp", "track_id", "tmin", "patch_path"]].drop_duplicates().copy()
        pm2["dish_norm"] = pm2["dish"].astype(str).map(normalize_dish_id)
        pm2["key"] = (
            pm2["dish_norm"].astype(str) + "|" +
            pm2["p"].astype(str) + "|" +
            pm2["fp"].astype(str) + "|" +
            pm2["track_id"].astype(str) + "|" +
            pm2["tmin"].astype(str)
        )
        key2path = dict(zip(pm2["key"].tolist(), pm2["patch_path"].tolist()))
        df["patch_path"] = df["key"].map(key2path)
        cov = 1.0 - float(df["patch_path"].isna().mean())
        info(f"[join] patch_path exact row coverage = {cov * 100:.2f}%")
        if patch_meta_well_coverage >= 0.95 and cov < 0.2:
            info("[join] note: exact row coverage is low, but well-level patch coverage is high; partial on-the-fly crop is still expected.")
    else:
        df["patch_path"] = None

    miss = df["patch_path"].isna()
    if miss.any():
        info(f"[patch] on-the-fly crop for missing patch_path rows={int(miss.sum())}")
        for idx in tqdm(df.index[miss].tolist(), desc="crop_missing", dynamic_ncols=True):
            r = df.loc[idx]
            pp = crop_patch_to_tmp(
                big_root,
                str(r.dish), int(r.p), int(r.fp), int(r.tmin),
                r.x1, r.y1, r.x2, r.y2,
                tmp_root=tmp_patch_root
            )
            df.at[idx, "patch_path"] = pp

        df["patch_path"] = df["patch_path"].astype(str)
        ok = df["patch_path"].map(_is_file_fast)
        df = df[ok].copy().reset_index(drop=True)
        info(f"[patch] after crop, rows={len(df)}")

    device = args.device
    if device == "cuda" and (not torch.cuda.is_available()):
        die("CUDA requested but not available.")
    if device == "cuda":
        info("[gpu]", torch.cuda.device_count(), "GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

    model, embed_dim, _id2idx = load_reid_best(reid_ckpt, device=device, use_dp=bool(args.data_parallel))
    info(f"[reid] ckpt={reid_ckpt} embed_dim={embed_dim}")

    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ConvertImageDtype(torch.float32),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    df["well_key"] = df["dish"].astype(str) + "||" + df["p"].astype(str) + "||" + df["fp"].astype(str)
    well_ids, _ = pd.factorize(df["well_key"], sort=True)
    df["well_id"] = well_ids.astype("int64")

    emb_dir = out_dir / "emb_shards"
    ensure_dir(emb_dir)

    emb_cache: Dict[str, np.ndarray] = {}
    uniq_wids = np.unique(well_ids)
    info(f"[repair] wells={len(uniq_wids)}")

    split_cfg = SplitCfg(
        enabled=bool(args.split_by_appearance),
        cos_th=float(args.split_cos_th),
        consec=int(args.split_consec),
        win=int(args.split_win),
        min_seg_len=int(args.split_min_seg_len),
        require_geom_break=bool(args.split_require_geom_break),
        guard_max_gap=float(args.split_guard_max_gap),
        guard_center_max=float(args.split_guard_center_max),
        guard_iou_min=float(args.split_guard_iou_min),
        guard_area_ratio_max=float(args.split_guard_area_ratio_max),
        guard_min_left_len=int(args.split_guard_min_left_len),
        guard_min_right_len=int(args.split_guard_min_right_len),
    )

    rescue_cfg = RescueCfg(
        enabled=bool(args.enable_free_rescue),
        min_reward=float(args.rescue_min_reward),
        min_proto=float(args.rescue_min_proto),
        min_seq=float(args.rescue_min_seq),
        require_free_color=bool(args.rescue_require_free_color),
        require_prev_tail=bool(args.rescue_require_prev_tail),
        rollback_overlap_conflict=bool(args.rescue_rollback_overlap),
    )

    promote_cfg = PromoteCfg(
        enabled=bool(args.enable_orphan_promotion),
        orphan_min_rows=int(args.orphan_min_rows),
        orphan_min_duration=float(args.orphan_min_duration),
        orphan_min_conf=float(args.orphan_min_conf),
        max_holder_rows=int(args.orphan_max_holder_rows),
        max_total_demote_rows=int(args.orphan_max_total_demote_rows),
        min_candidate_to_holder_row_ratio=float(args.orphan_min_candidate_to_holder_row_ratio),
        promoted_conf_floor=float(args.orphan_promoted_conf_floor),
        require_kreal_ge_3=True,
    )

    track_id_seg = pd.Series(-1, index=df.index, dtype="int64")

    track_id_final_strict_orig = pd.Series(-1, index=df.index, dtype="int64")
    cluster_conf_strict_orig = pd.Series(-1.0, index=df.index, dtype="float32")
    track_id_final_orig = pd.Series(-1, index=df.index, dtype="int64")
    cluster_conf_orig = pd.Series(-1.0, index=df.index, dtype="float32")
    is_fallback_orig = pd.Series(False, index=df.index, dtype="bool")

    track_id_final_strict_rescued = pd.Series(-1, index=df.index, dtype="int64")
    cluster_conf_strict_rescued = pd.Series(-1.0, index=df.index, dtype="float32")
    track_id_final_rescued = pd.Series(-1, index=df.index, dtype="int64")
    cluster_conf_rescued = pd.Series(-1.0, index=df.index, dtype="float32")
    is_fallback_rescued = pd.Series(False, index=df.index, dtype="bool")

    track_id_final_strict_safe = pd.Series(-1, index=df.index, dtype="int64")
    cluster_conf_strict_safe = pd.Series(-1.0, index=df.index, dtype="float32")
    track_id_final_safe = pd.Series(-1, index=df.index, dtype="int64")
    cluster_conf_safe = pd.Series(-1.0, index=df.index, dtype="float32")
    is_fallback_safe = pd.Series(False, index=df.index, dtype="bool")

    rescue_applied = pd.Series(False, index=df.index, dtype="bool")
    rescue_kept_safe = pd.Series(False, index=df.index, dtype="bool")
    rescue_rolled_back_due_to_conflict = pd.Series(False, index=df.index, dtype="bool")
    safe_status_note = pd.Series("", index=df.index, dtype="object")

    promotion_applied = pd.Series(False, index=df.index, dtype="bool")
    promoted_to_color = pd.Series(-1, index=df.index, dtype="int64")
    promoted_reason = pd.Series("", index=df.index, dtype="object")
    demoted_by_promotion = pd.Series(False, index=df.index, dtype="bool")
    demoted_from_color = pd.Series(-1, index=df.index, dtype="int64")
    demoted_by_seg = pd.Series(-1, index=df.index, dtype="int64")

    track_id_final_strict = pd.Series(-1, index=df.index, dtype="int64")
    cluster_conf_strict = pd.Series(-1.0, index=df.index, dtype="float32")
    track_id_final = pd.Series(-1, index=df.index, dtype="int64")
    cluster_conf = pd.Series(-1.0, index=df.index, dtype="float32")
    is_fallback = pd.Series(False, index=df.index, dtype="bool")

    n_clusters_well = pd.Series(0, index=df.index, dtype="int64")

    mapping_rows = []
    per_well_stats = []

    def estimate_k_from_timeline(df_well: pd.DataFrame) -> int:
        c = df_well.groupby("tmin")["track_id_seg"].nunique().astype(float).sort_index()
        if len(c) == 0:
            return 1
        if int(args.k_min_frames) > 0 and len(c) < int(args.k_min_frames):
            return max(1, min(int(args.k_max), int(np.ceil(float(c.max())))))
        sw = max(1, int(args.k_smooth_win))
        cs = c.rolling(window=sw, center=True, min_periods=max(1, sw // 2)).median() if sw > 1 else c
        cs = cs.bfill().ffill()
        q = float(args.k_timeline_quantile)
        q = min(0.999, max(0.50, q))
        k = int(np.ceil(float(np.quantile(cs.to_numpy(), q))))
        return max(1, min(int(args.k_max), k))

    for wid in tqdm(uniq_wids, desc="repair_wells", dynamic_ncols=True):
        m = (df["well_id"].to_numpy() == wid)
        if not np.any(m):
            continue
        df_w = df.loc[m, :].copy()

        uniq_paths = sorted(df_w["patch_path"].astype(str).unique().tolist())
        shard_id = f"wid{int(wid):05d}_{_hash_list(uniq_paths)}.npz"
        shard_path = emb_dir / shard_id

        if shard_path.is_file():
            shard = load_emb_cache_npz(shard_path)
            emb_cache.update(shard)
        else:
            shard = infer_embeddings(
                model, tfm, uniq_paths,
                batch_size=int(args.emb_batch_size),
                num_workers=int(args.emb_workers),
                device=device,
                use_amp=bool(args.amp),
                skip_bad=bool(args.skip_bad_images),
            )
            save_emb_cache_npz(shard, shard_path)
            emb_cache.update(shard)

        df_w["track_id_raw"] = df_w["track_id"].astype("int64")
        seg_series = appearance_split_segments(df_w, emb_cache, split_cfg)
        df_w["track_id_seg"] = seg_series.astype("int64")
        track_id_seg.loc[df_w.index] = df_w["track_id_seg"].astype("int64")

        if args.k_strategy == "fixed":
            K_real = int(args.k_max)
        elif args.k_strategy == "timeline_p95":
            K_real = estimate_k_from_timeline(df_w)
        else:
            t0 = float(df_w["tmin"].min())
            early = df_w[(df_w["tmin"] >= t0) & (df_w["tmin"] <= t0 + float(args.k_early_window))]
            if len(early) == 0:
                early = df_w
            c = early.groupby("tmin")["track_id_raw"].nunique()
            K_real = max(1, min(int(args.k_max), int(c.max()) if len(c) else 1))

        n_clusters_well.loc[df_w.index] = int(K_real)

        segs = build_segments(
            df_w,
            seg_col="track_id_seg",
            emb_cache=emb_cache,
            proto_sample_max=int(args.proto_sample_max),
            seq_sample_n=int(args.seq_sample_n),
            seq_scan_max=int(args.seq_scan_max),
        )
        if len(segs) == 0:
            continue

        seg_id_to_index = {segs[i].seg_id: i for i in range(len(segs))}
        conflicts = build_conflicts(segs)

        succ_rewards = build_successor_rewards(
            segs, conflicts,
            max_link_gap=float(args.max_link_gap),
            w_proto=float(args.w_proto),
            w_seq=float(args.w_seq),
            w_gap=float(args.w_gap),
            min_link_cos=float(args.link_cos_th),
            min_link_seq=float(args.link_seq_th),
            seq_topk=int(args.seq_topk),
        )

        colors_orig, best_score = solve_coloring_branch_and_bound(
            segs=segs,
            conflicts=conflicts,
            K_real=K_real,
            rewards=succ_rewards,
            seed=int(args.seed) + int(wid),
            allow_unassigned=bool(args.allow_unassigned),
            unassigned_penalty=float(args.unassigned_penalty),
            min_attach_reward=float(args.min_attach_reward),
        )
        conf_map_orig = compute_segment_confidence(segs, colors_orig, succ_rewards)

        ids_orig = build_final_ids_from_strict(
            wid=int(wid),
            segs=segs,
            strict_colors=colors_orig,
            strict_conf_by_segidx=conf_map_orig,
            force_assign_unassigned=bool(args.force_assign_unassigned),
            fallback_base=int(args.fallback_base),
            fallback_offset=int(args.fallback_offset),
        )

        colors_rescued, rescue_meta = apply_conservative_free_rescue(
            segs=segs,
            strict_colors=colors_orig,
            conflicts=conflicts,
            rewards=succ_rewards,
            K_real=K_real,
            w_proto=float(args.w_proto),
            w_seq=float(args.w_seq),
            w_gap=float(args.w_gap),
            seq_topk=int(args.seq_topk),
            cfg=rescue_cfg,
        )

        conf_map_rescued = conf_map_orig.copy()
        for i in range(len(segs)):
            if rescue_meta[i]["rescue_applied"] and rescue_meta[i]["rescue_kept_safe"]:
                conf_map_rescued[i] = sigmoid_conf(float(rescue_meta[i]["rescue_reward"]))

        ids_rescued = build_final_ids_from_strict(
            wid=int(wid),
            segs=segs,
            strict_colors=colors_rescued,
            strict_conf_by_segidx=conf_map_rescued,
            force_assign_unassigned=bool(args.force_assign_unassigned),
            fallback_base=int(args.fallback_base),
            fallback_offset=int(args.fallback_offset),
        )

        colors_safe, promote_meta, conf_map_safe = apply_long_lived_orphan_promotion(
            segs=segs,
            df_well=df_w,
            strict_colors=colors_rescued,
            strict_conf_by_segidx=conf_map_rescued,
            K_real=K_real,
            cfg=promote_cfg,
        )

        ids_safe = build_final_ids_from_strict(
            wid=int(wid),
            segs=segs,
            strict_colors=colors_safe,
            strict_conf_by_segidx=conf_map_safe,
            force_assign_unassigned=bool(args.force_assign_unassigned),
            fallback_base=int(args.fallback_base),
            fallback_offset=int(args.fallback_offset),
        )

        dish0 = str(df_w["dish"].iloc[0])
        p0 = int(df_w["p"].iloc[0])
        fp0 = int(df_w["fp"].iloc[0])

        df_w["track_id_final_strict_orig"] = -1
        df_w["cluster_conf_strict_orig"] = -1.0
        df_w["track_id_final_orig"] = -1
        df_w["cluster_conf_orig"] = -1.0
        df_w["is_fallback_orig"] = False

        df_w["track_id_final_strict_rescued"] = -1
        df_w["cluster_conf_strict_rescued"] = -1.0
        df_w["track_id_final_rescued"] = -1
        df_w["cluster_conf_rescued"] = -1.0
        df_w["is_fallback_rescued"] = False

        df_w["track_id_final_strict_safe"] = -1
        df_w["cluster_conf_strict_safe"] = -1.0
        df_w["track_id_final_safe"] = -1
        df_w["cluster_conf_safe"] = -1.0
        df_w["is_fallback_safe"] = False

        df_w["rescue_applied"] = False
        df_w["rescue_kept_safe"] = False
        df_w["rescue_rolled_back_due_to_conflict"] = False
        df_w["safe_status_note"] = ""

        df_w["promotion_applied"] = False
        df_w["promoted_to_color"] = -1
        df_w["promoted_reason"] = ""
        df_w["demoted_by_promotion"] = False
        df_w["demoted_from_color"] = -1
        df_w["demoted_by_seg"] = -1

        df_w["n_clusters_well"] = int(K_real)

        for s in segs:
            i = seg_id_to_index[s.seg_id]
            mask_seg = (df_w["track_id_seg"].astype("int64").to_numpy() == int(s.seg_id))
            idxs = df_w.index[mask_seg]

            df_w.loc[idxs, "track_id_final_strict_orig"] = int(ids_orig[i]["track_id_final_strict"])
            df_w.loc[idxs, "cluster_conf_strict_orig"] = float(ids_orig[i]["cluster_conf_strict"])
            df_w.loc[idxs, "track_id_final_orig"] = int(ids_orig[i]["track_id_final"])
            df_w.loc[idxs, "cluster_conf_orig"] = float(ids_orig[i]["cluster_conf"])
            df_w.loc[idxs, "is_fallback_orig"] = bool(ids_orig[i]["is_fallback"])

            df_w.loc[idxs, "track_id_final_strict_rescued"] = int(ids_rescued[i]["track_id_final_strict"])
            df_w.loc[idxs, "cluster_conf_strict_rescued"] = float(ids_rescued[i]["cluster_conf_strict"])
            df_w.loc[idxs, "track_id_final_rescued"] = int(ids_rescued[i]["track_id_final"])
            df_w.loc[idxs, "cluster_conf_rescued"] = float(ids_rescued[i]["cluster_conf"])
            df_w.loc[idxs, "is_fallback_rescued"] = bool(ids_rescued[i]["is_fallback"])

            df_w.loc[idxs, "track_id_final_strict_safe"] = int(ids_safe[i]["track_id_final_strict"])
            df_w.loc[idxs, "cluster_conf_strict_safe"] = float(ids_safe[i]["cluster_conf_strict"])
            df_w.loc[idxs, "track_id_final_safe"] = int(ids_safe[i]["track_id_final"])
            df_w.loc[idxs, "cluster_conf_safe"] = float(ids_safe[i]["cluster_conf"])
            df_w.loc[idxs, "is_fallback_safe"] = bool(ids_safe[i]["is_fallback"])

            df_w.loc[idxs, "rescue_applied"] = bool(rescue_meta[i]["rescue_applied"])
            df_w.loc[idxs, "rescue_kept_safe"] = bool(rescue_meta[i]["rescue_kept_safe"])
            df_w.loc[idxs, "rescue_rolled_back_due_to_conflict"] = bool(rescue_meta[i]["rescue_rolled_back_due_to_conflict"])
            df_w.loc[idxs, "safe_status_note"] = str(rescue_meta[i]["safe_status_note"])

            df_w.loc[idxs, "promotion_applied"] = bool(promote_meta[i]["promotion_applied"])
            df_w.loc[idxs, "promoted_to_color"] = int(promote_meta[i]["promoted_to_color"])
            df_w.loc[idxs, "promoted_reason"] = str(promote_meta[i]["promoted_reason"])
            df_w.loc[idxs, "demoted_by_promotion"] = bool(promote_meta[i]["demoted_by_promotion"])
            df_w.loc[idxs, "demoted_from_color"] = int(promote_meta[i]["demoted_from_color"])
            df_w.loc[idxs, "demoted_by_seg"] = int(promote_meta[i]["demoted_by_seg"])

            mapping_rows.append({
                "dish": dish0,
                "p": p0,
                "fp": fp0,
                "well_id": int(wid),
                "track_id_seg": int(s.seg_id),
                "seg_idx": int(i),
                "embryo_cluster_orig": int(colors_orig[i]),
                "embryo_cluster_rescued": int(colors_rescued[i]),
                "embryo_cluster_safe": int(colors_safe[i]),
                "track_id_final_strict_orig": int(ids_orig[i]["track_id_final_strict"]),
                "track_id_final_orig": int(ids_orig[i]["track_id_final"]),
                "is_fallback_orig": bool(ids_orig[i]["is_fallback"]),
                "track_id_final_strict_rescued": int(ids_rescued[i]["track_id_final_strict"]),
                "track_id_final_rescued": int(ids_rescued[i]["track_id_final"]),
                "is_fallback_rescued": bool(ids_rescued[i]["is_fallback"]),
                "track_id_final_strict_safe": int(ids_safe[i]["track_id_final_strict"]),
                "track_id_final_safe": int(ids_safe[i]["track_id_final"]),
                "is_fallback_safe": bool(ids_safe[i]["is_fallback"]),
                "rescue_applied": bool(rescue_meta[i]["rescue_applied"]),
                "rescue_kept_safe": bool(rescue_meta[i]["rescue_kept_safe"]),
                "rescue_rolled_back_due_to_conflict": bool(rescue_meta[i]["rescue_rolled_back_due_to_conflict"]),
                "rescue_from_prev_seg": int(rescue_meta[i]["rescue_from_prev_seg"]),
                "rescue_to_color": int(rescue_meta[i]["rescue_to_color"]),
                "rescue_reward": float(rescue_meta[i]["rescue_reward"]) if not pd.isna(rescue_meta[i]["rescue_reward"]) else np.nan,
                "rescue_proto": float(rescue_meta[i]["rescue_proto"]) if not pd.isna(rescue_meta[i]["rescue_proto"]) else np.nan,
                "rescue_seq": float(rescue_meta[i]["rescue_seq"]) if not pd.isna(rescue_meta[i]["rescue_seq"]) else np.nan,
                "promotion_applied": bool(promote_meta[i]["promotion_applied"]),
                "promoted_to_color": int(promote_meta[i]["promoted_to_color"]),
                "promoted_reason": str(promote_meta[i]["promoted_reason"]),
                "demoted_by_promotion": bool(promote_meta[i]["demoted_by_promotion"]),
                "demoted_from_color": int(promote_meta[i]["demoted_from_color"]),
                "demoted_by_seg": int(promote_meta[i]["demoted_by_seg"]),
                "seg_size_rows": int(s.n_rows),
                "t0": float(s.t0),
                "t1": float(s.t1),
            })

        df_w["track_id_final_strict"] = df_w["track_id_final_strict_safe"].astype("int64")
        df_w["cluster_conf_strict"] = df_w["cluster_conf_strict_safe"].astype("float32")
        df_w["track_id_final"] = df_w["track_id_final_safe"].astype("int64")
        df_w["cluster_conf"] = df_w["cluster_conf_safe"].astype("float32")
        df_w["is_fallback"] = df_w["is_fallback_safe"].astype("bool")

        track_id_final_strict_orig.loc[df_w.index] = df_w["track_id_final_strict_orig"].astype("int64")
        cluster_conf_strict_orig.loc[df_w.index] = df_w["cluster_conf_strict_orig"].astype("float32")
        track_id_final_orig.loc[df_w.index] = df_w["track_id_final_orig"].astype("int64")
        cluster_conf_orig.loc[df_w.index] = df_w["cluster_conf_orig"].astype("float32")
        is_fallback_orig.loc[df_w.index] = df_w["is_fallback_orig"].astype("bool")

        track_id_final_strict_rescued.loc[df_w.index] = df_w["track_id_final_strict_rescued"].astype("int64")
        cluster_conf_strict_rescued.loc[df_w.index] = df_w["cluster_conf_strict_rescued"].astype("float32")
        track_id_final_rescued.loc[df_w.index] = df_w["track_id_final_rescued"].astype("int64")
        cluster_conf_rescued.loc[df_w.index] = df_w["cluster_conf_rescued"].astype("float32")
        is_fallback_rescued.loc[df_w.index] = df_w["is_fallback_rescued"].astype("bool")

        track_id_final_strict_safe.loc[df_w.index] = df_w["track_id_final_strict_safe"].astype("int64")
        cluster_conf_strict_safe.loc[df_w.index] = df_w["cluster_conf_strict_safe"].astype("float32")
        track_id_final_safe.loc[df_w.index] = df_w["track_id_final_safe"].astype("int64")
        cluster_conf_safe.loc[df_w.index] = df_w["cluster_conf_safe"].astype("float32")
        is_fallback_safe.loc[df_w.index] = df_w["is_fallback_safe"].astype("bool")

        rescue_applied.loc[df_w.index] = df_w["rescue_applied"].astype("bool")
        rescue_kept_safe.loc[df_w.index] = df_w["rescue_kept_safe"].astype("bool")
        rescue_rolled_back_due_to_conflict.loc[df_w.index] = df_w["rescue_rolled_back_due_to_conflict"].astype("bool")
        safe_status_note.loc[df_w.index] = df_w["safe_status_note"].astype("object")

        promotion_applied.loc[df_w.index] = df_w["promotion_applied"].astype("bool")
        promoted_to_color.loc[df_w.index] = df_w["promoted_to_color"].astype("int64")
        promoted_reason.loc[df_w.index] = df_w["promoted_reason"].astype("object")
        demoted_by_promotion.loc[df_w.index] = df_w["demoted_by_promotion"].astype("bool")
        demoted_from_color.loc[df_w.index] = df_w["demoted_from_color"].astype("int64")
        demoted_by_seg.loc[df_w.index] = df_w["demoted_by_seg"].astype("int64")

        track_id_final_strict.loc[df_w.index] = df_w["track_id_final_strict"].astype("int64")
        cluster_conf_strict.loc[df_w.index] = df_w["cluster_conf_strict"].astype("float32")
        track_id_final.loc[df_w.index] = df_w["track_id_final"].astype("int64")
        cluster_conf.loc[df_w.index] = df_w["cluster_conf"].astype("float32")
        is_fallback.loc[df_w.index] = df_w["is_fallback"].astype("bool")

        n_rows_total = int(len(df_w))
        n_strict_rows_orig = int((df_w["track_id_final_strict_orig"] >= 0).sum())
        n_fallback_rows_orig = int((df_w["is_fallback_orig"] == True).sum())
        n_strict_rows_safe = int((df_w["track_id_final_strict_safe"] >= 0).sum())
        n_fallback_rows_safe = int((df_w["is_fallback_safe"] == True).sum())
        n_rescue_applied = int((df_w["rescue_applied"] == True).sum())
        n_rescue_kept = int((df_w["rescue_kept_safe"] == True).sum())
        n_rescue_rolled = int((df_w["rescue_rolled_back_due_to_conflict"] == True).sum())
        n_promotion_applied = int((df_w["promotion_applied"] == True).sum())
        n_demoted_rows = int((df_w["demoted_by_promotion"] == True).sum())

        per_well_stats.append({
            "dish": dish0,
            "p": p0,
            "fp": fp0,
            "well_id": int(wid),
            "K_real": int(K_real),
            "n_segments": int(len(segs)),
            "n_conflicts": int(sum(len(x) for x in conflicts) // 2),
            "best_score": float(best_score),
            "n_strict_rows_orig": int(n_strict_rows_orig),
            "n_fallback_rows_orig": int(n_fallback_rows_orig),
            "assigned_ratio_strict_orig": float(n_strict_rows_orig / max(1, n_rows_total)),
            "n_strict_rows_safe": int(n_strict_rows_safe),
            "n_fallback_rows_safe": int(n_fallback_rows_safe),
            "assigned_ratio_strict_safe": float(n_strict_rows_safe / max(1, n_rows_total)),
            "delta_strict_rows_safe_vs_orig": int(n_strict_rows_safe - n_strict_rows_orig),
            "delta_fallback_rows_safe_vs_orig": int(n_fallback_rows_orig - n_fallback_rows_safe),
            "n_rescue_applied_rows": int(n_rescue_applied),
            "n_rescue_kept_safe_rows": int(n_rescue_kept),
            "n_rescue_rolled_back_rows": int(n_rescue_rolled),
            "n_promotion_applied_rows": int(n_promotion_applied),
            "n_demoted_rows_by_promotion": int(n_demoted_rows),
            "mean_conf_strict_orig": float(df_w["cluster_conf_strict_orig"].mean()),
            "mean_conf_strict_safe": float(df_w["cluster_conf_strict_safe"].mean()),
        })

    df["track_id_seg"] = track_id_seg
    df["track_id_final_strict_orig"] = track_id_final_strict_orig
    df["cluster_conf_strict_orig"] = cluster_conf_strict_orig
    df["track_id_final_orig"] = track_id_final_orig
    df["cluster_conf_orig"] = cluster_conf_orig
    df["is_fallback_orig"] = is_fallback_orig
    df["track_id_final_strict_rescued"] = track_id_final_strict_rescued
    df["cluster_conf_strict_rescued"] = cluster_conf_strict_rescued
    df["track_id_final_rescued"] = track_id_final_rescued
    df["cluster_conf_rescued"] = cluster_conf_rescued
    df["is_fallback_rescued"] = is_fallback_rescued
    df["track_id_final_strict_safe"] = track_id_final_strict_safe
    df["cluster_conf_strict_safe"] = cluster_conf_strict_safe
    df["track_id_final_safe"] = track_id_final_safe
    df["cluster_conf_safe"] = cluster_conf_safe
    df["is_fallback_safe"] = is_fallback_safe
    df["rescue_applied"] = rescue_applied
    df["rescue_kept_safe"] = rescue_kept_safe
    df["rescue_rolled_back_due_to_conflict"] = rescue_rolled_back_due_to_conflict
    df["safe_status_note"] = safe_status_note
    df["promotion_applied"] = promotion_applied
    df["promoted_to_color"] = promoted_to_color
    df["promoted_reason"] = promoted_reason
    df["demoted_by_promotion"] = demoted_by_promotion
    df["demoted_from_color"] = demoted_from_color
    df["demoted_by_seg"] = demoted_by_seg
    df["track_id_final_strict"] = track_id_final_strict
    df["cluster_conf_strict"] = cluster_conf_strict
    df["track_id_final"] = track_id_final
    df["cluster_conf"] = cluster_conf
    df["is_fallback"] = is_fallback
    df["n_clusters_well"] = n_clusters_well

    pd.DataFrame(mapping_rows).to_csv(out_dir / "segment_to_embryo_mapping.csv", index=False)
    pd.DataFrame(per_well_stats).to_csv(out_dir / "per_well_stats.csv", index=False)

    keep = [
        "dish", "p", "fp", "filename", "tmin",
        "x1", "y1", "x2", "y2", "conf",
        "track_id", "patch_path",
        "track_id_seg",
        "track_id_final_strict_orig", "cluster_conf_strict_orig",
        "track_id_final_orig", "cluster_conf_orig", "is_fallback_orig",
        "track_id_final_strict_rescued", "cluster_conf_strict_rescued",
        "track_id_final_rescued", "cluster_conf_rescued", "is_fallback_rescued",
        "track_id_final_strict_safe", "cluster_conf_strict_safe",
        "track_id_final_safe", "cluster_conf_safe", "is_fallback_safe",
        "rescue_applied", "rescue_kept_safe", "rescue_rolled_back_due_to_conflict", "safe_status_note",
        "promotion_applied", "promoted_to_color", "promoted_reason",
        "demoted_by_promotion", "demoted_from_color", "demoted_by_seg",
        "track_id_final_strict", "cluster_conf_strict",
        "track_id_final", "cluster_conf", "is_fallback",
        "n_clusters_well", "well_id"
    ]
    final_cols = keep + [c for c in df.columns if c not in keep]
    df[final_cols].to_csv(out_dir / "tracks_repaired.csv", index=False)

    summary = {
        "script": "repair_pipeline53_guarded_orphanfix_release.py",
        "version": "v5.3_guarded_orphanfix_release",
        "reid_ckpt": str(reid_ckpt),
        "embed_dim": int(embed_dim),
        "tracks_rows": int(len(df_tracks)),
        "stable_rows": int(len(stable)),
        "repaired_rows": int(len(df)),
        "wells": int(len(np.unique(well_ids))),
        "patch_meta_used": bool(pm is not None),
        "patch_meta_cov_exact": float(cov),
        "patch_meta_well_coverage": float(patch_meta_well_coverage),
        "patch_meta_missing_wells": int(patch_meta_missing_wells_n),
        "patch_meta_missing_wells_csv": patch_meta_missing_csv,
        "split_by_appearance": bool(args.split_by_appearance),
        "split_require_geom_break": bool(args.split_require_geom_break),
        "enable_free_rescue": bool(args.enable_free_rescue),
        "enable_orphan_promotion": bool(args.enable_orphan_promotion),
        "orphan_min_rows": int(args.orphan_min_rows),
        "orphan_min_duration": float(args.orphan_min_duration),
        "orphan_min_conf": float(args.orphan_min_conf),
        "orphan_max_holder_rows": int(args.orphan_max_holder_rows),
        "orphan_max_total_demote_rows": int(args.orphan_max_total_demote_rows),
        "orphan_min_candidate_to_holder_row_ratio": float(args.orphan_min_candidate_to_holder_row_ratio),
        "k_strategy": str(args.k_strategy),
        "k_max": int(args.k_max),
        "w_proto": float(args.w_proto),
        "w_seq": float(args.w_seq),
        "w_gap": float(args.w_gap),
        "link_cos_th": float(args.link_cos_th),
        "link_seq_th": float(args.link_seq_th),
        "allow_unassigned": bool(args.allow_unassigned),
        "unassigned_penalty": float(args.unassigned_penalty),
        "min_attach_reward": float(args.min_attach_reward),
        "force_assign_unassigned": bool(args.force_assign_unassigned),
        "fallback_base": int(args.fallback_base),
        "fallback_offset": int(args.fallback_offset),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    info("[summary]", summary)
    info("DONE:", out_dir)


if __name__ == "__main__":
    main()
