# -*- coding: utf-8 -*-
"""
One-click comparison experiment:
ResNet-50 + Temporal Transformer

Purpose
-------
Run a clean before-vs-after label-structure comparison on the new machine.
The model backbone and training setup stay fixed. Only the label setting changes.

Comparison settings
-------------------
1) pre_adjustment
   - use the source label as-is
2) post_adjustment
   - merge Early Morula -> Morula
   - optionally merge 5-8 cell -> 5-16 cell when that class exists

Default paths are adapted to the new machine.
This script is intentionally simpler than the CBAM ablation script:
- no CBAM
- no pseudo-labels
- no multi-model sweep
- same split for both settings
- same ResNet-50 + Temporal Transformer architecture for both settings

Example
-------
BOVINE_ROOT=/path/to/project \
BOVINE_BIG_ROOT=/path/to/project/Bovine \
BATCH_SIZE=8 NUM_WORKERS=6 USE_CKPT=0 \
python run_label_restructure_compare.py

Optional
--------
python run_label_restructure_compare.py --label_col label
python run_label_restructure_compare.py --settings pre_adjustment,post_adjustment
python run_label_restructure_compare.py --seeds 0,1,2
"""

from __future__ import annotations

import os
import re
import json
import math
import time
import random
import warnings
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

try:
    import torchvision.models as tvm
    import torchvision.models.video as tvv
except Exception:
    tvm = None
    tvv = None

from PIL import Image
import cv2
from collections import Counter





def resolve_under_root(root: Path, p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def parse_int_list(s: str) -> List[int]:
    if s is None or str(s).strip() == "":
        return []
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fmt_secs(s: float) -> str:
    s = float(s)
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    return f"{s/3600:.2f}h"


def die(msg: str):
    print(f"\n[FATAL] {msg}", flush=True)
    raise SystemExit(2)


def safe_imread_bgr(path: str):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        try:
            rgb = np.array(Image.open(path).convert("RGB"))
            im = rgb[:, :, ::-1].copy()
        except Exception:
            return None
    return im


def save_confusion_png(cm, classes, out_png: Path, title: str):
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    for (i, j), v in np.ndenumerate(cm):
        if v > 0:
            ax.text(j, i, str(v), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def printl_factory(run_log_path: Path):
    def _printl(msg: str):
        print(msg, flush=True)
        try:
            with open(run_log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass
    return _printl






def build_argparser():
    ap = argparse.ArgumentParser()
    root_default = os.getenv("BOVINE_ROOT", "/path/to/project")

    ap.add_argument("--root", type=str, default=root_default)
    ap.add_argument("--seq_csv", type=str, default="exp/check_center_frames_600snap/meta/sequence_rows.csv")
    ap.add_argument("--gui_csv", type=str, default="胚胎/标注/central_labels_rebuild.csv")
    ap.add_argument("--out_dir", type=str, default="exp/r50_tfm_label_restructure_compare")
    ap.add_argument("--big_root", type=str, default=os.getenv("BOVINE_BIG_ROOT", "/path/to/project/Bovine"))

    ap.add_argument("--label_col", type=str, default="label",
                    help="Source label column before restructuring, e.g. label / stage_center / stage_final")
    ap.add_argument(
        "--experiment_mode",
        type=str,
        default="legacy_restructure",
        choices=["legacy_restructure", "gui_confirmatory"],
        help="Use gui_confirmatory for the matched pre-GUI versus GUI-reviewed experiment.",
    )
    ap.add_argument("--settings", type=str, default="pre_adjustment,post_adjustment")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument(
        "--model",
        type=str,
        default="r50_transformer",
        choices=["r50_transformer", "resnet18_tcn", "r2plus1d18"],
        help="Model used for the matched experiment.",
    )

    ap.add_argument("--window_size", type=int, default=21)
    ap.add_argument("--step_min", type=int, default=10)
    ap.add_argument("--snap_tol_min", type=int, default=15)
    ap.add_argument("--image_size", type=int, default=224)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", "8")))
    ap.add_argument("--num_workers", type=int, default=int(os.getenv("NUM_WORKERS", "6")))
    ap.add_argument("--freeze_epochs", type=int, default=2)
    ap.add_argument("--lr_frame", type=float, default=5e-5)
    ap.add_argument("--lr_temporal", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--early_stop_patience", type=int, default=10)
    ap.add_argument("--min_epochs_for_earlystop", type=int, default=40)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--use_ckpt", type=int, default=int(os.getenv("USE_CKPT", "0")))

    ap.add_argument("--merge_5_8_to_5_16", type=int, default=1)
    ap.add_argument("--drop_uncertain", type=int, default=1)
    ap.add_argument("--resume", type=int, default=1)

    return ap


ARGS = build_argparser().parse_args()
ROOT = Path(ARGS.root).resolve()
SEQ_CSV = resolve_under_root(ROOT, ARGS.seq_csv)
GUI_CSV = resolve_under_root(ROOT, ARGS.gui_csv)
OUT_DIR = resolve_under_root(ROOT, ARGS.out_dir)
BIG_ROOT = Path(ARGS.big_root).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

SETTINGS = [x.strip() for x in ARGS.settings.split(",") if x.strip()]
SEEDS = parse_int_list(ARGS.seeds)
if not SEEDS:
    SEEDS = [0, 1, 2]

print("ROOT   :", ROOT)
print("SEQ_CSV:", SEQ_CSV)
print("GUI_CSV:", GUI_CSV)
print("BIGROOT:", BIG_ROOT)
print("OUTDIR :", OUT_DIR)
print("Device :", DEVICE)
print("Settings:", SETTINGS)
print("Seeds   :", SEEDS)
print("Mode    :", ARGS.experiment_mode)
print("Model   :", ARGS.model)
print("GradAcc :", ARGS.grad_accum)





def sanity_check_bigroot(big_root: Path):
    if not big_root.exists():
        die(f"BIG_ROOT not found: {big_root}")
    if not any(big_root.glob("MTL-0034-*")):
        die(f"No MTL-0034-* dirs under BIG_ROOT: {big_root}")


def build_big_path(row, big_root: Path) -> Path:
    dish = str(row["dish"])
    p = int(row["p"])
    fp = int(row["fp"])
    tmin = int(row["tmin"])
    return big_root / f"MTL-0034-{dish}" / f"{dish}_P{p}_FP{fp}_{tmin}.jpg"


def sanity_check_hit_rate(df_rows: pd.DataFrame, big_root: Path, n=250, min_hit=0.85):
    n = min(int(n), len(df_rows))
    if n <= 0:
        die("sequence_rows is empty")
    s = df_rows.sample(n, random_state=0)

    hit = 0
    readable = 0
    for _, r in s.iterrows():
        p = build_big_path(r, big_root)
        if p.exists():
            hit += 1
            if safe_imread_bgr(str(p)) is not None:
                readable += 1

    hit_ratio = hit / n
    read_ratio = readable / max(hit, 1)
    print(f"[Sanity] hit-rate : {hit}/{n} = {hit_ratio:.3f}")
    print(f"[Sanity] readable : {readable}/{hit} = {read_ratio:.3f}")

    if hit_ratio < min_hit:
        die(f"Big-image hit rate too low ({hit_ratio:.3f} < {min_hit}). BIG_ROOT={big_root}")
    if read_ratio < 0.98:
        die("Readable rate too low for existing images.")


if not SEQ_CSV.exists():
    die(f"SEQ_CSV not found: {SEQ_CSV}")

sanity_check_bigroot(BIG_ROOT)
df_rows_all = pd.read_csv(SEQ_CSV, encoding="utf-8-sig")
need_base = {"seq_id", "dish", "p", "fp", "track_id", "center_min", "tmin", "x1", "y1", "x2", "y2"}
miss = need_base - set(df_rows_all.columns)
if miss:
    die(f"sequence_rows.csv missing columns: {sorted(list(miss))}")

sanity_check_hit_rate(df_rows_all, BIG_ROOT, n=250, min_hit=0.85)





def normalise_text(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def label_norm_key(s: str) -> str:
    s = normalise_text(s).lower()
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def apply_post_adjustment(label: str, merge_5_8_to_5_16: bool = True) -> str:
    key = label_norm_key(label)

    if key in {"early morula", "em"}:
        return "Morula"

    if merge_5_8_to_5_16:
        if key in {"5 8 cell", "5–8 cell", "5-8 cell", "5 to 8 cell"}:
            return "5–16 cell"
        if key in {"9 16 cell", "9–16 cell", "9-16 cell", "9 to 16 cell"}:
            return "5–16 cell"
        if key in {"5 16 cell", "5–16 cell", "5-16 cell", "5 to 16 cell"}:
            return "5–16 cell"

    return normalise_text(label)


def try_attach_gui_labels(df_rows: pd.DataFrame, gui_csv: Path) -> pd.DataFrame:
    if not gui_csv.exists():
        return df_rows

    df = df_rows.copy()
    df_gui = pd.read_csv(gui_csv, encoding="utf-8-sig")
    key = ["dish", "p", "fp", "track_id", "center_min"]

    for k in ["p", "fp", "track_id", "center_min"]:
        if k in df_gui.columns:
            df_gui[k] = pd.to_numeric(df_gui[k], errors="coerce").astype("Int64")
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce").astype("Int64")

    gui_label_cols = [c for c in ["label_gui", "stage_final", "stage_center", "stage_v4", "stage", "label"] if c in df_gui.columns]
    if gui_label_cols:
        use_col = gui_label_cols[0]
        join_cols = key + [use_col]
        rename_cols = {use_col: "label_gui_join"}
        if "stage_center" in df_gui.columns and use_col != "stage_center":
            join_cols.append("stage_center")
            rename_cols["stage_center"] = "label_pre_gui_join"
        gui_small = df_gui[join_cols].rename(columns=rename_cols)
        df = df.merge(gui_small, on=key, how="left")

        if "uncertain_gui" in df_gui.columns:
            gui_unc = df_gui[key + ["uncertain_gui"]]
            df = df.merge(gui_unc, on=key, how="left")
        else:
            df["uncertain_gui"] = False

    return df


df_rows_all = try_attach_gui_labels(df_rows_all, GUI_CSV)

if ARGS.label_col not in df_rows_all.columns:
    cand = [c for c in ["label", "stage_center", "stage_final", "label_gui_join"] if c in df_rows_all.columns]
    if not cand:
        die(f"Label column '{ARGS.label_col}' not found and no fallback label column exists.")
    label_col = cand[0]
else:
    label_col = ARGS.label_col

print(f"[Label] source column: {label_col}")

SIX_CLASSES = [
    "2-4 cell",
    "5-16 cell",
    "Blast",
    "Dead/Degenerating/Bad",
    "Morula",
    "Unfertilised",
]


def canonical_six_label(label: str) -> str:
    key = label_norm_key(label)
    if key in {"early morula", "em", "morula"}:
        return "Morula"
    if key in {"5 8 cell", "5 to 8 cell", "9 16 cell", "9 to 16 cell", "5 16 cell", "5 to 16 cell"}:
        return "5-16 cell"
    if key in {"2 4 cell", "2 to 4 cell"}:
        return "2-4 cell"
    if key == "blast":
        return "Blast"
    if key in {"dead/degenerating/bad", "dead degenerating bad"}:
        return "Dead/Degenerating/Bad"
    if key in {"unfertilised", "unfertilized"}:
        return "Unfertilised"
    return normalise_text(label)


CONFIRMATORY_MODE = ARGS.experiment_mode == "gui_confirmatory"
if CONFIRMATORY_MODE:
    if "label_gui_join" not in df_rows_all.columns or "label_pre_gui_join" not in df_rows_all.columns:
        die("The GUI audit table must provide both pre-review and reviewed labels.")
    df_rows_all["label_pre_gui_6"] = df_rows_all["label_pre_gui_join"].map(canonical_six_label)
    df_rows_all["label_gui_6"] = df_rows_all["label_gui_join"].map(canonical_six_label)





def build_manifest(df_rows: pd.DataFrame, label_series: pd.Series) -> pd.DataFrame:
    mani = (
        df_rows.assign(_label=label_series.values)
        .groupby("seq_id")
        .agg(
            dish=("dish", "first"),
            p=("p", "first"),
            fp=("fp", "first"),
            track_id=("track_id", "first"),
            center_min=("center_min", "first"),
            label=("_label", "first"),
            uncertain_gui=("uncertain_gui", "max") if "uncertain_gui" in df_rows.columns else ("seq_id", lambda x: False),
        )
        .reset_index()
    )
    mani["gid"] = mani.apply(lambda r: f"{r['dish']}_P{int(r['p'])}_FP{int(r['fp'])}_TID{int(r['track_id'])}", axis=1)
    return mani


def build_fixed_split(mani: pd.DataFrame, split_seed: int) -> Tuple[set, set, set]:
    idx = np.arange(len(mani))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=split_seed)
    trval_idx, test_idx = next(gss1.split(idx, groups=mani["gid"].tolist()))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.111, random_state=split_seed + 1)
    tr_idx, val_idx = next(gss2.split(trval_idx, groups=mani.iloc[trval_idx]["gid"].tolist()))

    train_ids = set(mani.iloc[trval_idx[tr_idx]]["seq_id"])
    val_ids = set(mani.iloc[trval_idx[val_idx]]["seq_id"])
    test_ids = set(mani.iloc[test_idx]["seq_id"])
    return train_ids, val_ids, test_ids


if CONFIRMATORY_MODE:
    mani_pre_all = build_manifest(df_rows_all, df_rows_all["label_pre_gui_6"])
    mani_gui_all = build_manifest(df_rows_all, df_rows_all["label_gui_6"])
    mani_pre_all = mani_pre_all.rename(columns={"label": "label_pre_gui"})
    mani_gui_all = mani_gui_all.rename(columns={"label": "label_gui"})
    common = mani_gui_all.merge(
        mani_pre_all[["seq_id", "label_pre_gui"]], on="seq_id", how="inner", validate="one_to_one"
    )
    common = common[
        (~common["uncertain_gui"].fillna(True).astype(bool))
        & common["label_pre_gui"].isin(SIX_CLASSES)
        & common["label_gui"].isin(SIX_CLASSES)
    ].copy()
    common["label"] = common["label_gui"]
    mani_for_split = common.reset_index(drop=True)
    common_ids = set(mani_for_split["seq_id"])
    df_rows_all = df_rows_all[df_rows_all["seq_id"].isin(common_ids)].reset_index(drop=True)
    print(f"[Confirmatory] matched certain samples: {len(mani_for_split)}")
    print(f"[Confirmatory] changed labels: {(mani_for_split['label_pre_gui'] != mani_for_split['label_gui']).sum()}")
else:
    mani_for_split = build_manifest(df_rows_all, df_rows_all[label_col].astype(str).fillna(""))
train_ids, val_ids, test_ids = build_fixed_split(mani_for_split, ARGS.split_seed)

split_df = pd.DataFrame(
    {
        "seq_id": mani_for_split["seq_id"].tolist(),
        "split": [
            "train" if sid in train_ids else ("val" if sid in val_ids else ("test" if sid in test_ids else "unused"))
            for sid in mani_for_split["seq_id"].tolist()
        ],
    }
)
if CONFIRMATORY_MODE:
    audit_cols = ["seq_id", "gid", "label_pre_gui", "label_gui"]
    split_df = split_df.merge(mani_for_split[audit_cols], on="seq_id", how="left", validate="one_to_one")
split_df.to_csv(OUT_DIR / "fixed_split_seq_id.csv", index=False, encoding="utf-8-sig")





IM_SIZE = int(ARGS.image_size)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def to_tensor_rgb(arr_hw3: np.ndarray) -> torch.Tensor:
    arr = arr_hw3.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


class SeqDataset(Dataset):
    def __init__(self, df_rows_all: pd.DataFrame, mani_df: pd.DataFrame, classes: List[str], window_size: int, train: bool):
        self.rows = df_rows_all.sort_values(["seq_id", "tmin"]).reset_index(drop=True)
        self.mani = mani_df.reset_index(drop=True)
        self.groups = self.rows.groupby("seq_id")
        self.cls2id = {c: i for i, c in enumerate(classes)}
        self.classes = classes
        self.train = bool(train)
        self.T = int(window_size)

    def __len__(self):
        return len(self.mani)

    def crop_rgb(self, im_bgr, box):
        H, W = im_bgr.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(x1, W - 1))
        x2 = max(1, min(x2, W))
        x2 = max(x2, x1 + 1)
        y1 = max(0, min(y1, H - 1))
        y2 = max(1, min(y2, H))
        y2 = max(y2, y1 + 1)
        patch = im_bgr[y1:y2, x1:x2]
        patch = cv2.resize(patch, (IM_SIZE, IM_SIZE), interpolation=cv2.INTER_AREA)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        return patch

    def __getitem__(self, idx):
        meta = self.mani.iloc[idx]
        sid, label, cmin = meta["seq_id"], str(meta["label"]), int(meta["center_min"])
        sub = self.groups.get_group(sid).sort_values("tmin").reset_index(drop=True)

        tmins = sub["tmin"].astype(int).to_numpy()
        near = int(np.argmin(np.abs(tmins - cmin)))
        half = self.T // 2
        lo = max(0, near - half)
        hi = min(len(sub), near + half + 1)
        picked = list(range(lo, hi))
        if len(picked) < self.T:
            while len(picked) < self.T and picked[0] > 0:
                picked = [picked[0] - 1] + picked
            while len(picked) < self.T and picked[-1] < len(sub) - 1:
                picked = picked + [picked[-1] + 1]
        picked = picked[: self.T]

        frames = []
        valid = []
        for i in picked:
            r = sub.iloc[i]
            bigp = build_big_path(r, BIG_ROOT)
            if not bigp.exists():
                frames.append(torch.zeros(3, IM_SIZE, IM_SIZE))
                valid.append(False)
                continue
            im = safe_imread_bgr(str(bigp))
            if im is None:
                frames.append(torch.zeros(3, IM_SIZE, IM_SIZE))
                valid.append(False)
                continue
            patch = self.crop_rgb(im, (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])))
            if self.train:
                if random.random() < 0.5:
                    patch = np.ascontiguousarray(np.fliplr(patch))
            frames.append(to_tensor_rgb(patch))
            valid.append(True)

        while len(frames) < self.T:
            frames.append(torch.zeros(3, IM_SIZE, IM_SIZE))
            valid.append(False)

        X = torch.stack(frames[: self.T], dim=0)
        y = int(self.cls2id[label])
        mask = torch.tensor(valid[: self.T], dtype=torch.bool)
        if not mask.any():
            mask[-1] = True
        return X, y, {"seq_id": sid, "mask": mask}


def pad_collate(batch):
    xs, ys, metas = zip(*batch)
    X = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    mask = torch.stack([m["mask"] for m in metas], dim=0)
    return X, y, {"seq_id": [m["seq_id"] for m in metas], "mask": mask}





D_MODEL, N_HEAD, N_LAYERS, DROP = 256, 8, 2, 0.20


def build_backbone_resnet50(pretrained=True):
    if tvm is None:
        raise RuntimeError("torchvision not available")
    try:
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    except Exception:
        try:
            base = tvm.resnet50(pretrained=pretrained)
        except Exception:
            base = tvm.resnet50(pretrained=False)
    feat_dim = base.fc.in_features
    base.fc = nn.Identity()
    return base, feat_dim


class BackboneWrapper(nn.Module):
    def __init__(self, net, use_ckpt=False):
        super().__init__()
        self.net = net
        self.use_ckpt = bool(use_ckpt)

    def forward(self, x4):
        if self.training and self.use_ckpt:
            x4 = x4.requires_grad_(True)
            return torch.utils.checkpoint.checkpoint(self.net, x4)
        return self.net(x4)


class TemporalTransformerHead(nn.Module):
    def __init__(self, in_dim, num_classes, d_model=D_MODEL, nhead=N_HEAD, n_layers=N_LAYERS, dropout=DROP, max_len=64):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, feats, valid_mask=None):
        B, T, F = feats.shape
        if T <= self.pos.shape[1]:
            pos = self.pos[:, :T, :]
        else:
            pos = torch.nn.functional.interpolate(self.pos.transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2)
        x = self.proj(feats) + pos
        key_pad = (~valid_mask.bool()) if valid_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=key_pad)
        score = self.attn(x).squeeze(-1)
        if valid_mask is not None:
            score = score.masked_fill(~valid_mask, float("-inf"))
        w = torch.softmax(score, dim=1)
        pooled = torch.sum(w.unsqueeze(-1) * x, dim=1)
        return self.fc(self.drop(pooled))


class CNNTransR50(nn.Module):
    def __init__(self, num_classes, max_len=64, pretrained=True, use_ckpt=False):
        super().__init__()
        net, feat_dim = build_backbone_resnet50(pretrained=pretrained)
        self.backbone = BackboneWrapper(net, use_ckpt=use_ckpt)
        self.temporal = TemporalTransformerHead(feat_dim, num_classes, max_len=max_len)

    def forward(self, x, valid_mask=None):
        B, T, C, H, W = x.shape
        x4 = x.reshape(B * T, C, H, W).contiguous(memory_format=torch.channels_last)
        feat = self.backbone(x4)
        feat = feat.view(B, T, -1)
        return self.temporal(feat, valid_mask)


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.20):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class TemporalConvHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, channels: int = 256, dropout: float = 0.20):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            TemporalResidualBlock(channels, dilation=1, dropout=dropout),
            TemporalResidualBlock(channels, dilation=2, dropout=dropout),
            TemporalResidualBlock(channels, dilation=4, dropout=dropout),
        )
        self.attn = nn.Conv1d(channels, 1, kernel_size=1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, feats, valid_mask=None):
        x = self.blocks(self.proj(feats.transpose(1, 2)))
        score = self.attn(x).squeeze(1)
        if valid_mask is not None:
            score = score.masked_fill(~valid_mask.bool(), float("-inf"))
        weight = torch.softmax(score, dim=1)
        pooled = torch.sum(x * weight.unsqueeze(1), dim=2)
        return self.fc(self.drop(pooled))


def build_backbone_resnet18(pretrained=True):
    if tvm is None:
        raise RuntimeError("torchvision not available")
    try:
        base = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    except Exception:
        base = tvm.resnet18(pretrained=pretrained)
    feat_dim = base.fc.in_features
    base.fc = nn.Identity()
    return base, feat_dim


class CNNTemporalTCN(nn.Module):
    def __init__(self, num_classes, pretrained=True, use_ckpt=False):
        super().__init__()
        net, feat_dim = build_backbone_resnet18(pretrained=pretrained)
        self.backbone = BackboneWrapper(net, use_ckpt=use_ckpt)
        self.temporal = TemporalConvHead(feat_dim, num_classes)

    def forward(self, x, valid_mask=None):
        B, T, C, H, W = x.shape
        x4 = x.reshape(B * T, C, H, W).contiguous(memory_format=torch.channels_last)
        feat = self.backbone(x4).view(B, T, -1)
        return self.temporal(feat, valid_mask)


class R2Plus1DClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        if tvv is None:
            raise RuntimeError("torchvision video models are not available")
        try:
            weights = tvv.R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
            net = tvv.r2plus1d_18(weights=weights)
        except Exception:
            net = tvv.r2plus1d_18(pretrained=pretrained)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.temporal = nn.Sequential(nn.Dropout(0.20), nn.Linear(feat_dim, num_classes))

    def forward(self, x, valid_mask=None):
        # torchvision video backbones expect B, C, T, H, W.
        feat = self.backbone(x.transpose(1, 2).contiguous())
        return self.temporal(feat)


def build_experiment_model(num_classes: int):
    if ARGS.model == "r50_transformer":
        return CNNTransR50(
            num_classes,
            max_len=max(64, int(ARGS.window_size)),
            pretrained=True,
            use_ckpt=bool(ARGS.use_ckpt),
        )
    if ARGS.model == "resnet18_tcn":
        return CNNTemporalTCN(num_classes, pretrained=True, use_ckpt=bool(ARGS.use_ckpt))
    if ARGS.model == "r2plus1d18":
        return R2Plus1DClassifier(num_classes, pretrained=True)
    raise ValueError(f"Unknown model: {ARGS.model}")







def class_balanced_weights(num_classes, labels, beta=0.999):
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    eff_num = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / np.maximum(eff_num, 1e-8)
    w = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, weight=None, reduction="mean", label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ls = label_smoothing

    def forward(self, logits, target):
        num_classes = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1)
            true_dist = true_dist * (1 - self.ls) + self.ls / num_classes
        log_prob = torch.log_softmax(logits, dim=1)
        prob = torch.exp(log_prob)
        ce = -(true_dist * log_prob).sum(dim=1)
        pt = prob.gather(1, target.unsqueeze(1)).squeeze(1).clamp_(1e-6, 1 - 1e-6)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device).gather(0, target)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


@torch.no_grad()
def export_predictions(model, dl, classes, out_csv: Path):
    model.eval()
    all_sid, all_y, all_p = [], [], []
    for X, y, meta in dl:
        X = X.to(DEVICE, non_blocking=True)
        mask = meta["mask"].to(DEVICE, non_blocking=True)
        logits = model(X, mask)
        pred = logits.argmax(1).detach().cpu().numpy()
        all_sid += meta["seq_id"]
        all_y += y.numpy().tolist()
        all_p += pred.tolist()

    acc = accuracy_score(all_y, all_p)
    f1 = f1_score(all_y, all_p, average="macro", zero_division=0)
    cm = confusion_matrix(all_y, all_p, labels=list(range(len(classes))))
    report = classification_report(all_y, all_p, target_names=classes, digits=4, zero_division=0)

    dfp = pd.DataFrame({"seq_id": all_sid, "true": [classes[i] for i in all_y], "pred": [classes[i] for i in all_p]})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    dfp.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return float(acc), float(f1), cm, report






def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_one(
    setting_name: str,
    df_rows_setting: pd.DataFrame,
    mani_setting: pd.DataFrame,
    seed: int,
    mani_eval: Optional[pd.DataFrame] = None,
    mani_balance: Optional[pd.DataFrame] = None,
    classes_override: Optional[List[str]] = None,
):
    set_all_seeds(seed)

    classes = list(classes_override) if classes_override is not None else sorted(mani_setting["label"].astype(str).unique().tolist())
    cls2id = {c: i for i, c in enumerate(classes)}

    if mani_eval is None:
        mani_eval = mani_setting
    if mani_balance is None:
        mani_balance = mani_eval

    tr_mani = mani_setting[mani_setting["seq_id"].isin(train_ids)].reset_index(drop=True)
    va_mani = mani_eval[mani_eval["seq_id"].isin(val_ids)].reset_index(drop=True)
    te_mani = mani_eval[mani_eval["seq_id"].isin(test_ids)].reset_index(drop=True)
    balance_tr = mani_balance[mani_balance["seq_id"].isin(train_ids)].set_index("seq_id")

    exp_dir = OUT_DIR / setting_name / f"seed_{seed:02d}"
    (exp_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    (exp_dir / "eval").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    summary_path = exp_dir / "run_summary.json"
    if bool(ARGS.resume) and summary_path.exists():
        print(f"[Resume] using completed run: {summary_path}")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    printl = printl_factory(exp_dir / "logs/run.log")

    printl(f"Start setting={setting_name} seed={seed}")
    printl(f"Classes: {classes}")
    printl(f"Split sizes: train={len(tr_mani)} val={len(va_mani)} test={len(te_mani)}")

    ds_tr = SeqDataset(df_rows_setting, tr_mani, classes, ARGS.window_size, train=True)
    ds_va = SeqDataset(df_rows_setting, va_mani, classes, ARGS.window_size, train=False)
    ds_te = SeqDataset(df_rows_setting, te_mani, classes, ARGS.window_size, train=False)

    train_labels = [cls2id[c] for c in tr_mani["label"].tolist()]
    balance_labels = [cls2id[str(balance_tr.loc[sid, "label"])] for sid in tr_mani["seq_id"]]
    cnt = Counter(balance_labels)
    freq = np.array([cnt.get(i, 0) for i in range(len(classes))], dtype=np.float32)
    probs = np.array([1.0 / (freq[y] ** 0.5 if freq[y] > 0 else 1.0) for y in balance_labels], dtype=np.float64)
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
    sampler_gen = torch.Generator()
    sampler_gen.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=probs, num_samples=len(train_labels), replacement=True, generator=sampler_gen
    )

    # Pair worker randomness across models.
    
    
    train_loader_gen = torch.Generator()
    train_loader_gen.manual_seed(seed + 100_000)
    eval_loader_gen = torch.Generator()
    eval_loader_gen.manual_seed(seed + 200_000)

    common_dl_kwargs = dict(
        num_workers=ARGS.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=pad_collate,
        prefetch_factor=2 if ARGS.num_workers > 0 else None,
        persistent_workers=(ARGS.num_workers > 0),
        worker_init_fn=seed_worker,
    )

    dl_tr = DataLoader(
        ds_tr,
        batch_size=ARGS.batch_size,
        shuffle=False,
        sampler=sampler,
        generator=train_loader_gen,
        **common_dl_kwargs,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=ARGS.batch_size,
        shuffle=False,
        generator=eval_loader_gen,
        **common_dl_kwargs,
    )
    dl_te = DataLoader(
        ds_te,
        batch_size=max(1, ARGS.batch_size * 2),
        shuffle=False,
        generator=eval_loader_gen,
        **common_dl_kwargs,
    )

    model = build_experiment_model(len(classes)).to(DEVICE)

    pg_backbone, pg_head = [], []
    for n, p in model.named_parameters():
        (pg_backbone if "backbone" in n else pg_head).append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in pg_backbone if p.requires_grad], "lr": ARGS.lr_frame, "weight_decay": ARGS.weight_decay},
            {"params": [p for p in pg_head if p.requires_grad], "lr": ARGS.lr_temporal, "weight_decay": ARGS.weight_decay},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ARGS.epochs)

    cb_w = class_balanced_weights(len(classes), balance_labels, beta=0.999).to(DEVICE)
    criterion = FocalLoss(gamma=1.5, weight=cb_w, label_smoothing=0.05, reduction="mean")
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    def set_backbone_requires_grad(flag: bool):
        for n, p in model.named_parameters():
            if "backbone" in n:
                p.requires_grad = flag

    set_backbone_requires_grad(False)
    best_score, best_epoch, best_val_acc = -1.0, -1, -1.0
    no_improve = 0
    metrics_rows = []

    for epoch in range(1, ARGS.epochs + 1):
        if epoch == ARGS.freeze_epochs + 1:
            set_backbone_requires_grad(True)
            printl("Unfreeze backbone")

        model.train()
        tr_losses, tr_preds, tr_labels_list = [], [], []
        optimizer.zero_grad(set_to_none=True)

        optimizer.zero_grad(set_to_none=True)
        n_train_batches = len(dl_tr)
        for batch_idx, (X, y, meta) in enumerate(dl_tr, start=1):
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            mask = meta["mask"].to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                logits = model(X, mask)
                loss_raw = criterion(logits, y)
                loss = loss_raw / max(1, int(ARGS.grad_accum))

            scaler.scale(loss).backward()
            should_step = (batch_idx % max(1, int(ARGS.grad_accum)) == 0) or (batch_idx == n_train_batches)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), ARGS.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            tr_losses.append(float(loss_raw.item()))
            tr_preds.extend(logits.argmax(1).detach().cpu().numpy())
            tr_labels_list.extend(y.detach().cpu().numpy())

        tr_acc = accuracy_score(tr_labels_list, tr_preds) if tr_labels_list else 0.0
        tr_f1 = f1_score(tr_labels_list, tr_preds, average="macro", zero_division=0) if tr_labels_list else 0.0

        model.eval()
        va_losses, va_preds, va_labels_list = [], [], []
        with torch.no_grad():
            for X, y, meta in dl_va:
                X = X.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                mask = meta["mask"].to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    logits = model(X, mask)
                    loss = criterion(logits, y)
                va_losses.append(float(loss.item()))
                va_preds.extend(logits.argmax(1).cpu().numpy())
                va_labels_list.extend(y.cpu().numpy())

        va_acc = accuracy_score(va_labels_list, va_preds) if va_labels_list else float("nan")
        va_f1 = f1_score(va_labels_list, va_preds, average="macro", zero_division=0) if va_labels_list else float("nan")
        scheduler.step()

        metrics_rows.append({
            "epoch": epoch,
            "train_loss": float(np.mean(tr_losses) if tr_losses else 0.0),
            "train_acc": float(tr_acc),
            "train_f1": float(tr_f1),
            "val_loss": float(np.mean(va_losses) if va_losses else 0.0),
            "val_acc": float(va_acc if not math.isnan(va_acc) else 0.0),
            "val_f1": float(va_f1 if not math.isnan(va_f1) else 0.0),
        })

        candidate = va_f1 if not math.isnan(va_f1) else -1.0
        if candidate > best_score:
            best_score, best_epoch, best_val_acc = float(candidate), int(epoch), float(va_acc)
            no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": classes,
                    "best_metric": "val_macro_f1",
                    "best_score": float(best_score),
                    "best_val_acc": float(best_val_acc),
                    "epoch": best_epoch,
                },
                exp_dir / "ckpt/best.pth",
            )
        else:
            no_improve += 1

        printl(
            f"epoch={epoch:03d}/{ARGS.epochs} tr_loss={np.mean(tr_losses):.4f} tr_acc={tr_acc:.3f} tr_f1={tr_f1:.3f} "
            f"va_loss={np.mean(va_losses):.4f} va_acc={va_acc:.3f} va_f1={va_f1:.3f} best_va_f1={best_score:.4f}@{best_epoch}"
        )

        if epoch >= ARGS.min_epochs_for_earlystop and no_improve >= ARGS.early_stop_patience:
            printl(f"Early stop at epoch {epoch}")
            break

    pd.DataFrame(metrics_rows).to_csv(exp_dir / "logs/metrics.csv", index=False, encoding="utf-8-sig")

    ck_best = torch.load(exp_dir / "ckpt/best.pth", map_location="cpu")
    model.load_state_dict(ck_best["model"], strict=False)
    model.to(DEVICE).eval()

    tr_acc2, tr_f12, cm_tr, rep_tr = export_predictions(model, dl_tr, classes, exp_dir / "eval/train_predictions.csv")
    va_acc2, va_f12, cm_va, rep_va = export_predictions(model, dl_va, classes, exp_dir / "eval/val_predictions.csv")
    te_acc2, te_f12, cm_te, rep_te = export_predictions(model, dl_te, classes, exp_dir / "eval/test_predictions.csv")

    save_confusion_png(cm_tr, classes, exp_dir / "eval/confusion_train.png", f"Train ({setting_name}, seed={seed})")
    save_confusion_png(cm_va, classes, exp_dir / "eval/confusion_val.png", f"Val ({setting_name}, seed={seed})")
    save_confusion_png(cm_te, classes, exp_dir / "eval/confusion_test.png", f"Test ({setting_name}, seed={seed})")

    (exp_dir / "eval/train_report.txt").write_text(rep_tr, encoding="utf-8")
    (exp_dir / "eval/val_report.txt").write_text(rep_va, encoding="utf-8")
    (exp_dir / "eval/test_report.txt").write_text(rep_te, encoding="utf-8")

    result = {
        "setting": setting_name,
        "model": ARGS.model,
        "seed": int(seed),
        "n_classes": int(len(classes)),
        "classes": " | ".join(classes),
        "best_epoch": int(ck_best.get("epoch", best_epoch)),
        "best_val_f1": float(ck_best.get("best_score", best_score)),
        "best_val_acc": float(ck_best.get("best_val_acc", best_val_acc)),
        "train_acc_at_best": float(tr_acc2),
        "train_f1_at_best": float(tr_f12),
        "val_acc_at_best": float(va_acc2),
        "val_f1_at_best": float(va_f12),
        "test_acc_at_best": float(te_acc2),
        "test_f1_at_best": float(te_f12),
        "exp_dir": str(exp_dir),
    }
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result






def make_setting_df(setting_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df_rows_all.copy()
    if CONFIRMATORY_MODE:
        if setting_name == "pre_gui":
            train_col = "label_pre_gui_6"
        elif setting_name == "gui_reviewed":
            train_col = "label_gui_6"
        else:
            die(f"Unknown confirmatory setting: {setting_name}")

        mani_train = build_manifest(df, df[train_col])
        mani_eval = build_manifest(df, df["label_gui_6"])
        mani_train = mani_train[mani_train["seq_id"].isin(set(mani_for_split["seq_id"]))].reset_index(drop=True)
        mani_eval = mani_eval[mani_eval["seq_id"].isin(set(mani_for_split["seq_id"]))].reset_index(drop=True)
        mani_train["label"] = mani_train["label"].map(canonical_six_label)
        mani_eval["label"] = mani_eval["label"].map(canonical_six_label)
        return df, mani_train, mani_eval

    labels = df[label_col].astype(str).map(normalise_text)

    if setting_name == "pre_adjustment":
        labels2 = labels.copy()
    elif setting_name == "post_adjustment":
        labels2 = labels.map(lambda s: apply_post_adjustment(s, merge_5_8_to_5_16=bool(ARGS.merge_5_8_to_5_16)))
    else:
        die(f"Unknown setting: {setting_name}")

    df["label_setting"] = labels2
    mani = build_manifest(df, df["label_setting"])

    if bool(ARGS.drop_uncertain) and "uncertain_gui" in mani.columns:
        mani = mani[~mani["uncertain_gui"].fillna(False)].reset_index(drop=True)
        keep_ids = set(mani["seq_id"])
        df = df[df["seq_id"].isin(keep_ids)].reset_index(drop=True)

    mani["label"] = mani["label"].astype(str).map(normalise_text)
    return df, mani, mani


all_rows = []
t0 = time.time()
for setting_name in SETTINGS:
    df_setting, mani_setting, mani_eval = make_setting_df(setting_name)
    (OUT_DIR / setting_name).mkdir(parents=True, exist_ok=True)
    mani_setting.to_csv(OUT_DIR / setting_name / "manifest_used.csv", index=False, encoding="utf-8-sig")

    print(f"\n[Setting] {setting_name}: n={len(mani_setting)} classes={sorted(mani_setting['label'].unique().tolist())}")

    for seed in SEEDS:
        row = run_one(
            setting_name,
            df_setting,
            mani_setting,
            seed,
            mani_eval=mani_eval,
            mani_balance=mani_eval,
            classes_override=SIX_CLASSES if CONFIRMATORY_MODE else None,
        )
        all_rows.append(row)

summary_df = pd.DataFrame(all_rows).sort_values(["setting", "seed"]).reset_index(drop=True)
summary_df.to_csv(OUT_DIR / "compare_runs.csv", index=False, encoding="utf-8-sig")

agg_df = (
    summary_df.groupby("setting")
    .agg(
        n=("seed", "count"),
        best_val_f1_mean=("best_val_f1", "mean"),
        best_val_f1_std=("best_val_f1", "std"),
        val_acc_mean=("val_acc_at_best", "mean"),
        val_acc_std=("val_acc_at_best", "std"),
        val_f1_mean=("val_f1_at_best", "mean"),
        val_f1_std=("val_f1_at_best", "std"),
        test_acc_mean=("test_acc_at_best", "mean"),
        test_acc_std=("test_acc_at_best", "std"),
        test_f1_mean=("test_f1_at_best", "mean"),
        test_f1_std=("test_f1_at_best", "std"),
    )
    .reset_index()
)
agg_df.to_csv(OUT_DIR / "compare_summary_mean_std.csv", index=False, encoding="utf-8-sig")

config = {
    "root": str(ROOT),
    "seq_csv": str(SEQ_CSV),
    "gui_csv": str(GUI_CSV),
    "big_root": str(BIG_ROOT),
    "label_col": label_col,
    "experiment_mode": ARGS.experiment_mode,
    "model": ARGS.model,
    "settings": SETTINGS,
    "seeds": SEEDS,
    "window_size": int(ARGS.window_size),
    "epochs": int(ARGS.epochs),
    "batch_size": int(ARGS.batch_size),
    "grad_accum": int(ARGS.grad_accum),
    "num_workers": int(ARGS.num_workers),
    "freeze_epochs": int(ARGS.freeze_epochs),
    "lr_frame": float(ARGS.lr_frame),
    "lr_temporal": float(ARGS.lr_temporal),
    "weight_decay": float(ARGS.weight_decay),
    "split_seed": int(ARGS.split_seed),
    "merge_5_8_to_5_16": int(ARGS.merge_5_8_to_5_16),
    "drop_uncertain": int(ARGS.drop_uncertain),
}
(OUT_DIR / "compare_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

print("\nSaved:")
print("-", OUT_DIR / "compare_runs.csv")
print("-", OUT_DIR / "compare_summary_mean_std.csv")
print("-", OUT_DIR / "compare_config.json")
print("Total time:", fmt_secs(time.time() - t0))
print("All done")
