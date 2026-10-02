# -*- coding: utf-8 -*-
"""
Ablation (hist_center): ResNet50 + CBAM + Temporal Transformer

This clean replacement uses only non-uncertain GUI-reviewed sequences
for training, validation, and testing. Pseudo-labelled sequences are not
used in this experiment, so the temporal-window comparison is not
confounded by pseudo-label quality or pseudo-labelled evaluation targets.

Key fixes for this machine:
- BIG_ROOT must point to the big-image tree (otherwise every sample becomes zeros and the model collapses)
- Hard sanity checks before training (existence, hit-rate, readable-rate)
- Run multiple random seeds per T to get stable estimates (mean/std) and enable basic statistical tests

Changes in this version:
1) Sharding / Parallel-friendly:
   - Split all (T, seed) jobs into N shards: --num_jobs N --job_index i
   - Each process runs only its shard -> easy to run multiple processes across GPUs/machines.

2) Skip & Resume:
   - If eval/test_predictions.csv exists or DONE marker exists -> skip that (T,seed).
   - Optional resume from ckpt/last.pth: --resume 1 (env RESUME=1 also works)

3) Better GPU utilization:
   - Gradient accumulation: ACCUM_STEPS (env) or --accum_steps
   - Auto AMP already enabled; TF32 enabled.

Run examples:

# (A) Single process (recommended for single GPU)
BOVINE_BIG_ROOT=/path/to/project/Bovine BATCH_SIZE=8 NUM_WORKERS=6 USE_CKPT=0 python run_ablation.py

# (B) True parallel (multi-GPU): launch 2 shards on 2 GPUs
CUDA_VISIBLE_DEVICES=0 python run_ablation.py --num_jobs 2 --job_index 0
CUDA_VISIBLE_DEVICES=1 python run_ablation.py --num_jobs 2 --job_index 1

# (C) Two shards on ONE GPU (NOT recommended; only if you accept slower + risk OOM)
CUDA_VISIBLE_DEVICES=0 python run_ablation.py --num_jobs 2 --job_index 0
CUDA_VISIBLE_DEVICES=0 python run_ablation.py --num_jobs 2 --job_index 1

# (D) Only run some T and seeds
python run_ablation.py --only_T 21,15 --only_seeds 0,1

"""

import os, re, json, math, random, gc, warnings, time, sys, argparse, hashlib, shutil
from direction_support import select_centered, cached_patch
DIRECTION_MODE = os.environ.get("DIRECTION_MODE", "centered")
assert DIRECTION_MODE in {"centered", "history"}
PATCH_CACHE = os.environ.get("DIRECTION_PATCH_CACHE", "")
from pathlib import Path
from typing import List, Tuple, Union, Optional


os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.simplefilter("ignore")
for cat in [UserWarning, FutureWarning, DeprecationWarning, RuntimeWarning, PendingDeprecationWarning]:
    warnings.filterwarnings("ignore", category=cat)
warnings.filterwarnings("ignore", message=".*nested_tensor.*")
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
warnings.filterwarnings("ignore", message=".*Flash attention.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm as tqdm_base

def tqdm(*args, **kwargs):
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 1.0)
    kwargs.setdefault("maxinterval", 10.0)
    kwargs.setdefault("smoothing", 0.1)
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("bar_format", "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    return tqdm_base(*args, **kwargs)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

try:
    import torchvision.models as tvm
except Exception:
    tvm = None

from PIL import Image
import cv2
from collections import Counter


def parse_int_list(s: str) -> List[int]:
    if s is None or str(s).strip() == "":
        return []
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if x == "":
            continue
        out.append(int(x))
    return out

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_jobs", type=int, default=int(os.getenv("NUM_JOBS", "1")),
                    help="Total shards. Use >1 for parallel processes.")
    ap.add_argument("--job_index", type=int, default=int(os.getenv("JOB_INDEX", "0")),
                    help="Shard index in [0, num_jobs-1].")
    ap.add_argument("--only_T", type=str, default=os.getenv("ONLY_T", ""),
                    help="Comma-separated T list to run, e.g. 21,15,11")
    ap.add_argument("--only_seeds", type=str, default=os.getenv("ONLY_SEEDS", ""),
                    help="Comma-separated run seeds to run, e.g. 0,1,2")
    ap.add_argument("--resume", type=int, default=int(os.getenv("RESUME", "0")),
                    help="1 to resume from ckpt/last.pth if exists.")
    ap.add_argument("--skip_done", type=int, default=int(os.getenv("SKIP_DONE", "1")),
                    help="1 to skip jobs already done.")
    ap.add_argument("--accum_steps", type=int, default=int(os.getenv("ACCUM_STEPS", "1")),
                    help="Gradient accumulation steps (effective batch = BATCH_SIZE * accum_steps).")
    ap.add_argument("--neptune", type=int, default=int(os.getenv("NEPTUNE_ON", "1")),
                    help="1 to enable neptune if token exists.")
    ap.add_argument("--preflight_only", type=int, default=int(os.getenv("PREFLIGHT_ONLY", "0")),
                    help="1 to validate data/splits and write frozen metadata without training.")
    return ap

ARGS = build_argparser().parse_args()


def resolve_under_root(root: Path, p: Union[str, Path]) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()

ROOT = Path(os.getenv("BOVINE_ROOT", "/path/to/project")).resolve()
SCRIPT_PATH = Path(__file__).resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


def atomic_write_csv(df: pd.DataFrame, path: Path, encoding: str = "utf-8-sig") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    df.to_csv(tmp, index=False, encoding=encoding)
    os.replace(tmp, path)


SCRIPT_SHA256 = sha256_file(SCRIPT_PATH)

SEQ_CSV = resolve_under_root(ROOT, "exp/check_center_frames_600snap/meta/sequence_rows.csv")
PSEUDO_CSV = resolve_under_root(
    ROOT,
    "exp/r50_tfm_v6_rawtracks_50epoch/exp_r50_tfm_v6_rawtracks_31f_gui/pseudo_stride100_31f/selected_600/selected_600.csv",
)
GUI_CSV = resolve_under_root(ROOT, "胚胎/标注/central_labels_rebuild.csv")

OUT_DIR = resolve_under_root(
    ROOT,
    os.getenv(
        "WINDOW_CONFIRM_OUT_DIR",
        "exp/agent_initiated_supplementary_20260812/window_gui_only_clean_full_sweep",
    ),
)
BASE_EXP = os.getenv("WINDOW_CONFIRM_BASE_EXP", "agent_window_gui_only_r50_cbam_tfm")

BIG_ROOT = Path(os.getenv("BOVINE_BIG_ROOT", "/path/to/project/Bovine")).resolve()


SWEEP_K = [1, 2, 3, 4, 5, 7, 9, 11, 15, 21]

EPOCHS = int(os.getenv("EPOCHS", "50"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "6"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
PIN_MEMORY = True
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True

FREEZE_EPOCHS = int(os.getenv("FREEZE_EPOCHS", "2"))
CENTER_ONLY_EPOCHS = 0

EARLY_STOP_PAT = 10
MIN_EPOCHS_FOR_EARLYSTOP = int(os.getenv("MIN_EPOCHS_FOR_EARLYSTOP", "50"))

LR_FRAME, LR_TEMPORAL = 5e-5, 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

LABEL_MERGE_EM_TO_MO = True
EXPECTED_CLASSES = {
    "2-4 cell",
    "5-16 cell",
    "Blast",
    "Dead/Degenerating/Bad",
    "Morula",
    "Unfertilised",
}
EXPECTED_GUI_SEQUENCE_COUNT = 3260
EXPECTED_SPLIT_COUNTS = {"train": 2601, "val": 328, "test": 331}

# split stays fixed; model init randomness varies by RUN_SEEDS
SPLIT_SEED = 42
RUN_SEEDS = list(range(10))

SRC_WEIGHT = {"base": 1.0, "gui": 1.0, "pseudo": 0.6}

SHOW_TQDM = True
TQDM_LEAVE = False
POSTFIX_EVERY_STEPS = 25

SAVE_EPOCH_CKPT_EVERY = 0
USE_CKPT = bool(int(os.getenv("USE_CKPT", "0")))
USE_CBAM = bool(int(os.getenv("USE_CBAM", "1")))
FULL_SEEDS_FOR_ALL_T = bool(int(os.getenv("FULL_SEEDS_FOR_ALL_T", "0")))


NEPTUNE_ON = bool(ARGS.neptune)
NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT", "YOUR_WORKSPACE/YOUR_PROJECT")
NEPTUNE_STRICT = False
NEPTUNE_TAGS_BASE = ["bovine", "raw-tracks", "r50", "tfm", "v6", "gui-only", "cbam", "history-only", "best=val_macro_f1"]
NEPTUNE_RUN_PREFIX = "Ablation_CBAM_HistCenter"
NEPTUNE_UPLOAD_PRED_CSV = False


ACCUM_STEPS = max(1, int(ARGS.accum_steps))
RESUME = bool(int(ARGS.resume))
SKIP_DONE = bool(int(ARGS.skip_done))


_only_T = parse_int_list(ARGS.only_T)
_only_seeds = parse_int_list(ARGS.only_seeds)
if _only_T:
    SWEEP_K = [k for k in SWEEP_K if int(k) in set(_only_T)]
if _only_seeds:
    RUN_SEEDS = [s for s in RUN_SEEDS if int(s) in set(_only_seeds)]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cuda_purge():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()

def fmt_secs(s: float) -> str:
    s = float(s)
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    return f"{s/3600:.2f}h"

def printl_factory(run_log_path: Path, verbose: bool = True):
    def _printl(msg: str, always: bool = False):
        if verbose or always:
            print(msg, flush=True)
        try:
            with open(run_log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass
    return _printl


def read_token_candidates():
    cands = []
    env_tok = os.getenv("NEPTUNE_API_TOKEN", "").strip()
    if env_tok:
        cands.append((env_tok, "ENV"))
    p = Path.home() / ".neptune_token"
    if p.exists():
        try:
            txt = p.read_text(encoding="utf-8").strip()
            if txt:
                cands.append((txt, "FILE"))
        except Exception:
            pass
    return cands

def neptune_connect(run_name: str, tags=None, params: Optional[dict] = None):
    if not NEPTUNE_ON:
        return None
    try:
        import neptune
    except Exception:
        return None

    cands = read_token_candidates()
    if not cands:
        if NEPTUNE_STRICT:
            raise RuntimeError("Neptune token missing.")
        return None

    last_err = None
    for tok, _src in cands:
        try:
            run = neptune.init_run(
                project=NEPTUNE_PROJECT,
                api_token=tok,
                capture_stdout=False,
                capture_stderr=False,
            )
            if tags:
                run["sys/tags"].add(list(tags))
            if run_name:
                run["sys/name"] = run_name
            if params:
                for k, v in params.items():
                    run[f"params/{k}"] = v
            return run
        except Exception as e:
            last_err = e
            continue

    if NEPTUNE_STRICT:
        raise RuntimeError(f"Neptune init failed: {repr(last_err)}")
    return None


def die(msg: str, code: int = 2):
    print("\n[FATAL]", msg, flush=True)
    raise SystemExit(code)

def safe_imread_bgr(path: str):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        try:
            rgb = np.array(Image.open(path).convert("RGB"))
            im = rgb[:, :, ::-1].copy()
        except Exception:
            return None
    return im

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
            im = safe_imread_bgr(str(p))
            if im is not None:
                readable += 1

    hit_ratio = hit / n
    read_ratio = readable / max(hit, 1)

    print(f"[Sanity] hit-rate : {hit}/{n} = {hit_ratio:.3f}")
    print(f"[Sanity] readable : {readable}/{hit} = {read_ratio:.3f}")

    if hit_ratio < min_hit:
        die(
            f"Big-image hit rate too low ({hit_ratio:.3f} < {min_hit}).\n"
            f"Likely: missing big images / wrong path layout / dish naming mismatch.\n"
            f"BIG_ROOT={big_root}"
        )
    if read_ratio < 0.98:
        die("Readable rate too low for existing images (permission/corruption?)")


OUT_DIR.mkdir(parents=True, exist_ok=True)

print("ROOT   :", ROOT)
print("SCRIPT :", SCRIPT_PATH)
print("SHA256 :", SCRIPT_SHA256)
print("SEQ_CSV:", SEQ_CSV)
print("PSEUDO (audit reference only; excluded from this experiment):", PSEUDO_CSV)
print("GUI    :", GUI_CSV)
print("BIGROOT:", BIG_ROOT)
print("OUTDIR :", OUT_DIR)
print("Device :", device)
print("Batch  :", BATCH_SIZE, "Workers:", NUM_WORKERS, "USE_CKPT:", USE_CKPT, "ACCUM_STEPS:", ACCUM_STEPS)
print("Split  :", SPLIT_SEED, "Run seeds:", RUN_SEEDS)
print("Sweep  :", SWEEP_K)
print("Shard  :", f"{ARGS.job_index}/{ARGS.num_jobs}  (SKIP_DONE={int(SKIP_DONE)} RESUME={int(RESUME)})")

if not SEQ_CSV.exists():
    die(f"SEQ_CSV not found: {SEQ_CSV}")

sanity_check_bigroot(BIG_ROOT)

df_rows = pd.read_csv(SEQ_CSV, encoding="utf-8-sig")

need_base = {"seq_id", "dish", "p", "fp", "track_id", "center_min", "tmin", "x1", "y1", "x2", "y2"}
miss = need_base - set(df_rows.columns)
if miss:
    die(f"sequence_rows.csv missing columns: {sorted(list(miss))}")

sanity_check_hit_rate(df_rows, BIG_ROOT, n=250, min_hit=0.85)

label_ready = False
if "label" in df_rows.columns and df_rows["label"].astype(str).str.strip().any():
    label_ready = True
elif "stage_center" in df_rows.columns and df_rows["stage_center"].astype(str).str.strip().any():
    df_rows["label"] = df_rows["stage_center"].astype(str).str.strip()
    label_ready = True

def normalize_final_six_class(s: str):
    if s is None:
        return None
    k = str(s).strip().lower().replace("-", " ").replace("_", " ")
    k = re.sub(r"\s+", " ", k)
    if k in {"early morula", "em"}:
        return "Morula"
    if k in {"5 8 cell", "5 to 8 cell"}:
        return "5-16 cell"
    return str(s).strip()


def parse_uncertain_flag(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return bool(int(value))
    text = str(value).strip().lower()
    if text in {"", "0", "false", "f", "no", "n"}:
        return False
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    die(f"Unrecognized uncertain_gui value: {value!r}")

# Use GUI labels; leave pseudo labels out.


required_source_cols = {"source", "stage_center", "seq_id"}
if not required_source_cols.issubset(df_rows.columns):
    die(f"sequence_rows.csv lacks source columns: {sorted(required_source_cols - set(df_rows.columns))}")
if not GUI_CSV.exists():
    die(f"GUI_CSV not found: {GUI_CSV}")

df_gui = pd.read_csv(GUI_CSV, encoding="utf-8-sig")
gui_required = {"seq_id", "label_gui", "uncertain_gui"}
if not gui_required.issubset(df_gui.columns):
    die(f"GUI_CSV lacks required columns: {sorted(gui_required - set(df_gui.columns))}")

df_gui["uncertain_gui_parsed"] = df_gui["uncertain_gui"].map(parse_uncertain_flag)
df_gui_clean = df_gui.loc[
    ~df_gui["uncertain_gui_parsed"], ["seq_id", "label_gui"]
].copy()
df_gui_clean["label_gui"] = df_gui_clean["label_gui"].astype(str).str.strip()
if df_gui_clean["label_gui"].eq("").any():
    die("Non-uncertain GUI rows contain empty label_gui values")
duplicate_label_counts = df_gui_clean.groupby("seq_id")["label_gui"].nunique()
conflicting_gui_ids = duplicate_label_counts[duplicate_label_counts > 1].index.tolist()
if conflicting_gui_ids:
    die(f"Conflicting GUI labels for seq_id values: {conflicting_gui_ids[:20]}")
gui_labels = df_gui_clean.drop_duplicates("seq_id").set_index("seq_id")["label_gui"]

source_norm = df_rows["source"].astype(str).str.strip().str.lower()
gui_mask = source_norm.eq("gui")
pseudo_mask = source_norm.str.startswith("pseudo")
if (~(gui_mask | pseudo_mask)).any():
    bad_sources = sorted(df_rows.loc[~(gui_mask | pseudo_mask), "source"].astype(str).unique().tolist())
    die(f"Unexpected label sources in sequence_rows.csv: {bad_sources}")

df_rows["label"] = np.nan
df_rows.loc[gui_mask, "label"] = df_rows.loc[gui_mask, "seq_id"].map(gui_labels)
df_rows.loc[pseudo_mask, "label"] = df_rows.loc[pseudo_mask, "stage_center"]
df_rows["label_source"] = np.where(gui_mask, "gui", "pseudo")

if df_rows["label"].isna().any():
    nmiss = int(df_rows["label"].isna().sum())
    print(f"[Info] drop {nmiss} rows with missing label", flush=True)
    df_rows = df_rows[~df_rows["label"].isna()].reset_index(drop=True)

if LABEL_MERGE_EM_TO_MO:
    df_rows["label"] = df_rows["label"].map(normalize_final_six_class)

classes = sorted(df_rows["label"].astype(str).unique().tolist())
cls2id = {c: i for i, c in enumerate(classes)}
print("[Classes]", classes, flush=True)
print("[label_source]", df_rows["label_source"].value_counts().to_dict(), flush=True)

# Leave pseudo labels out of this comparison.

df_rows = df_rows.loc[df_rows["label_source"].eq("gui")].reset_index(drop=True)
classes = sorted(df_rows["label"].astype(str).unique().tolist())
cls2id = {c: i for i, c in enumerate(classes)}
print("[Window cohort] GUI-reviewed only", flush=True)
print("[Window cohort rows]", len(df_rows), flush=True)
if not df_rows["label_source"].eq("gui").all():
    die("Pseudo-labelled rows remain after GUI-only cohort filtering")
if set(classes) != EXPECTED_CLASSES:
    die(f"Unexpected final class set: {classes}; expected={sorted(EXPECTED_CLASSES)}")


mani = (
    df_rows.groupby("seq_id")
    .agg(
        dish=("dish", "first"),
        p=("p", "first"),
        fp=("fp", "first"),
        track_id=("track_id", "first"),
        center_min=("center_min", "first"),
        label=("label", "first"),
        label_source=("label_source", "first"),
    )
    .reset_index()
)

mani["sample_weight"] = mani["label_source"].map(lambda s: SRC_WEIGHT.get(s, 1.0))
mani["gid"] = mani.apply(lambda r: f"{r['dish']}_P{int(r['p'])}_FP{int(r['fp'])}_TID{int(r['track_id'])}", axis=1)
if len(mani) != EXPECTED_GUI_SEQUENCE_COUNT:
    die(f"Unexpected GUI sequence count: {len(mani)}; expected={EXPECTED_GUI_SEQUENCE_COUNT}")
if mani["seq_id"].duplicated().any():
    die("Duplicate seq_id values remain in the sequence manifest")

idx = np.arange(len(mani))
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SPLIT_SEED)
trval_idx, test_idx = next(gss1.split(idx, groups=mani["gid"].tolist()))

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.111, random_state=SPLIT_SEED + 1)
tr_idx, val_idx = next(gss2.split(trval_idx, groups=mani.iloc[trval_idx]["gid"].tolist()))

train_ids = set(mani.iloc[trval_idx[tr_idx]]["seq_id"])
val_ids = set(mani.iloc[trval_idx[val_idx]]["seq_id"])
test_ids = set(mani.iloc[test_idx]["seq_id"])

def subset(split_ids: set):
    return mani.loc[mani["seq_id"].isin(split_ids), ["seq_id", "label", "center_min", "sample_weight", "label_source"]].copy()

tr_mani = subset(train_ids)
va_mani = subset(val_ids)
te_mani = subset(test_ids)
print(f"[Split] train={len(tr_mani)} val={len(va_mani)} test={len(te_mani)}", flush=True)

split_count_actual = {
    "train": len(tr_mani),
    "val": len(va_mani),
    "test": len(te_mani),
}
if split_count_actual != EXPECTED_SPLIT_COUNTS:
    die(f"Unexpected split counts: {split_count_actual}; expected={EXPECTED_SPLIT_COUNTS}")
if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
    die("seq_id overlap detected across train/val/test")
gid_by_seq = mani.set_index("seq_id")["gid"]
train_gids = set(gid_by_seq.loc[list(train_ids)])
val_gids = set(gid_by_seq.loc[list(val_ids)])
test_gids = set(gid_by_seq.loc[list(test_ids)])
if train_gids & val_gids or train_gids & test_gids or val_gids & test_gids:
    die("group leakage detected across train/val/test")

split_df = pd.DataFrame(
    {
        "seq_id": mani["seq_id"].tolist(),
        "split": [
            "train" if sid in train_ids else ("val" if sid in val_ids else ("test" if sid in test_ids else "unused"))
            for sid in mani["seq_id"].tolist()
        ],
    }
)
split_df = split_df.merge(
    mani[["seq_id", "gid", "label", "label_source"]], on="seq_id", how="left", validate="one_to_one"
)


IM_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

def to_tensor_rgb(arr_hw3: np.ndarray) -> torch.Tensor:
    arr = arr_hw3.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)

def is_black_patch(rgb: np.ndarray) -> bool:
    g = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.float32)
    m = float(g.mean())
    s = float(g.std())
    low_ratio = float((g <= 10).mean())
    return (m < 8.0) or (m < 15.0 and s < 3.0) or (low_ratio > 0.95)

class SeqDataset(Dataset):
    """
    hist_center:
      - only frames with tmin <= center_min (history + center)
      - if more than T, uniform sample + force include nearest-to-center frame
      - if less than T, pad to T (mask=False)
    """
    def __init__(
        self,
        df_rows_all: pd.DataFrame,
        mani_df: pd.DataFrame,
        classes: List[str],
        target_frames: int,
        train: bool,
        temporal_dropout: float = 0.15,
        black_fix: bool = True,
    ):
        self.rows = df_rows_all.sort_values(["seq_id", "tmin"]).reset_index(drop=True)
        self.mani = mani_df.reset_index(drop=True)
        self.groups = self.rows.groupby("seq_id")
        self.cls2id = {c: i for i, c in enumerate(classes)}
        self.classes = classes
        self.train = bool(train)
        self.temporal_dropout = float(temporal_dropout) if train else 0.0
        self.black_fix = bool(black_fix)
        self.T = int(target_frames)
        self.sid2w = {r["seq_id"]: float(r.get("sample_weight", 1.0)) for _, r in self.mani.iterrows()}

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

    @staticmethod
    def uniform_pick(idxs: List[int], k: int) -> List[int]:
        if len(idxs) <= k:
            return idxs[:]
        if k == 1:
            return [idxs[-1]]
        pos = [int(round(i * (len(idxs) - 1) / (k - 1))) for i in range(k)]
        return [idxs[p] for p in pos]

    def __getitem__(self, idx):
        meta = self.mani.iloc[idx]
        sid, label, cmin = meta["seq_id"], str(meta["label"]), int(meta["center_min"])
        sub_all = self.groups.get_group(sid).sort_values("tmin").reset_index(drop=True)

        hist = sub_all[sub_all["tmin"].astype(int) <= cmin].copy()
        if len(hist) == 0:
            raise RuntimeError("No past/target frame; future fallback is forbidden")

        hist_idx = hist.index.tolist()
        tmins_hist = hist["tmin"].astype(int).to_numpy()
        near_local = int(np.argmin(np.abs(tmins_hist - cmin)))
        near_global_idx = hist_idx[near_local]

        if len(hist_idx) > self.T:
            picked = self.uniform_pick(hist_idx, self.T)
            if near_global_idx not in picked:
                picked[-1] = near_global_idx
            picked = sorted(set(picked))
            if len(picked) < self.T:
                for j in reversed(hist_idx):
                    if j not in picked:
                        picked.append(j)
                        if len(picked) == self.T:
                            break
            picked = sorted(picked)
            if len(picked) > self.T:
                picked = picked[-self.T:]
        else:
            picked = hist_idx

        if DIRECTION_MODE == "centered":
            assert self.T == 3, "Only the frozen T3 contrast is supported"
            picked = select_centered(sub_all, picked, cmin)
        picked = sorted(picked, key=lambda i: int(sub_all.iloc[i]["tmin"]))

        frames = []
        valid = []

        for i in picked:
            r = sub_all.iloc[i]
            bigp = build_big_path(r, BIG_ROOT)

            if not bigp.exists():
                frames.append(torch.zeros(3, IM_SIZE, IM_SIZE))
                valid.append(False)
                continue

            box = (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))
            if PATCH_CACHE:
                patch = cached_patch(PATCH_CACHE, bigp, box)
            else:
                im = safe_imread_bgr(str(bigp))
                if im is None:
                    frames.append(torch.zeros(3, IM_SIZE, IM_SIZE))
                    valid.append(False)
                    continue
                patch = self.crop_rgb(im, box)
            black = is_black_patch(patch) if self.black_fix else False

            if self.train:
                if random.random() < 0.8:
                    alpha = random.uniform(0.85, 1.20)
                    beta = random.uniform(-18, 18)
                    patch = np.clip(alpha * patch + beta, 0, 255).astype(np.uint8)
                if random.random() < 0.5:
                    g = random.uniform(0.85, 1.20)
                    table = np.array([(ii / 255.0) ** (1.0 / g) * 255 for ii in range(256)], dtype=np.uint8)
                    patch = cv2.LUT(patch, table)
                if random.random() < 0.35:
                    h, w = patch.shape[:2]
                    ang = random.uniform(-8, 8)
                    scale = random.uniform(0.97, 1.05)
                    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
                    patch = cv2.warpAffine(patch, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                if random.random() < self.temporal_dropout:
                    patch = np.zeros_like(patch) if random.random() < 0.5 else cv2.GaussianBlur(patch, (7, 7), 0)

            frames.append(to_tensor_rgb(patch))
            valid.append(not black)

        if len(frames) < self.T:
            pad_n = self.T - len(frames)
            for _ in range(pad_n):
                frames.append(torch.zeros(3, IM_SIZE, IM_SIZE))
                valid.append(False)
        elif len(frames) > self.T:
            frames = frames[-self.T:]
            valid = valid[-self.T:]

        X = torch.stack(frames, dim=0)
        y = int(self.cls2id[label])

        mask = torch.tensor(valid, dtype=torch.bool)
        if not mask.any():
            mask[-1] = True

        w = float(self.sid2w.get(sid, 1.0))
        return X, y, {"seq_id": sid, "mask": mask, "weight": w}

def pad_collate(batch):
    xs, ys, metas = zip(*batch)
    T = max([x.shape[0] for x in xs])
    B = len(xs)
    Xpad = torch.zeros((B, T, 3, IM_SIZE, IM_SIZE), dtype=xs[0].dtype)
    valid = torch.zeros((B, T), dtype=torch.bool)
    weights = torch.zeros(B, dtype=torch.float32)

    for i, x in enumerate(xs):
        t = x.shape[0]
        Xpad[i, :t] = x
        mv = metas[i]["mask"]
        valid[i, : mv.shape[0]] = mv
        weights[i] = float(metas[i].get("weight", 1.0))

    y = torch.tensor(ys, dtype=torch.long)
    return Xpad, y, {"seq_id": [m["seq_id"] for m in metas], "mask": valid, "weight": weights}


D_MODEL, N_HEAD, N_LAYERS, DROP = 256, 8, 2, 0.20

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(1, in_planes // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_planes, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(attn)

class CBAM2d(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class CBAM1d(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.cbam2d = CBAM2d(channels, reduction=reduction, kernel_size=kernel_size)

    def forward(self, x):
        if x.dim() == 2:
            x_in = x.unsqueeze(-1).unsqueeze(-1)
            out = self.cbam2d(x_in)
            return out.squeeze(-1).squeeze(-1)
        if x.dim() == 4:
            return self.cbam2d(x)
        raise ValueError(f"CBAM1d expects dim 2 or 4, got {x.dim()}")

def build_backbone_resnet50(pretrained=True):
    if tvm is None:
        raise RuntimeError("torchvision not available, cannot build ResNet50")
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
        self.attn = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, feats, valid_mask=None):
        B, T, F = feats.shape
        if T <= self.pos.shape[1]:
            pos = self.pos[:, :T, :]
        else:
            pos = torch.nn.functional.interpolate(
                self.pos.transpose(1, 2), size=T, mode="linear", align_corners=False
            ).transpose(1, 2)

        x = self.proj(feats) + pos
        key_pad = (~valid_mask.bool()) if valid_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=key_pad)

        score = self.attn(x).squeeze(-1)
        if valid_mask is not None:
            score = score.masked_fill(~valid_mask, float("-inf"))
        w = torch.softmax(score, dim=1)
        pooled = torch.sum(w.unsqueeze(-1) * x, dim=1)
        return self.fc(self.drop(pooled))

class CNNTransR50CBAM(nn.Module):
    def __init__(self, num_classes, max_len=64, pretrained=True, use_ckpt=False):
        super().__init__()
        net, feat_dim = build_backbone_resnet50(pretrained=pretrained)
        self.backbone = BackboneWrapper(net, use_ckpt=use_ckpt)
        self.cbam = CBAM1d(feat_dim, reduction=16, kernel_size=7) if USE_CBAM else nn.Identity()
        self.temporal = TemporalTransformerHead(feat_dim, num_classes, max_len=max_len)

    def forward(self, x, valid_mask=None):
        B, T, C, H, W = x.shape
        x4 = x.reshape(B * T, C, H, W).contiguous(memory_format=torch.channels_last)
        feat = self.backbone(x4)
        feat = self.cbam(feat)
        feat = feat.view(B, T, -1)
        return self.temporal(feat, valid_mask)


def class_balanced_weights(num_classes, labels, beta=0.999):
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    eff_num = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / np.maximum(eff_num, 1e-8)
    w = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, weight=None, reduction="none", label_smoothing=0.05):
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
            w = self.weight.to(logits.device).gather(0, target)
            loss = loss * w

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

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

@torch.no_grad()
def export_predictions(model, dl, classes, out_csv: Path) -> Tuple[float, float, np.ndarray, str]:
    model.eval()
    all_sid, all_y, all_p = [], [], []
    for X, y, meta in dl:
        X = X.to(device, non_blocking=True)
        mask = meta["mask"].to(device, non_blocking=True)
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

def plot_curve_from_csv(csv_path: Path, out_png: Path):
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax2 = ax1.twinx()
        ax1.plot(df["epoch"], df["train_loss"], label="train_loss")
        ax1.plot(df["epoch"], df["val_loss"], label="val_loss", linestyle="--")
        ax2.plot(df["epoch"], df["train_acc"], label="train_acc")
        ax2.plot(df["epoch"], df["val_acc"], label="val_acc", linestyle="--")
        ax2.plot(df["epoch"], df["val_f1"], label="val_f1(macro)", linestyle=":")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax2.set_ylabel("acc / f1")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    except Exception:
        pass


SWEEP_ROOT = OUT_DIR / f"ablation_{BASE_EXP}"
(SWEEP_ROOT / "meta").mkdir(parents=True, exist_ok=True)
(SWEEP_ROOT / "runs").mkdir(parents=True, exist_ok=True)

atomic_write_csv(df_rows, SWEEP_ROOT / "meta/sequence_rows_used.csv")
atomic_write_csv(split_df, SWEEP_ROOT / "meta/fixed_split_seq_id.csv")
script_snapshot = SWEEP_ROOT / "meta" / f"run_paired_window_{SCRIPT_SHA256[:16]}.py"
if not script_snapshot.exists():
    snapshot_tmp = script_snapshot.with_name(f".{script_snapshot.name}.{os.getpid()}.tmp")
    shutil.copy2(SCRIPT_PATH, snapshot_tmp)
    os.replace(snapshot_tmp, script_snapshot)

cfg = {
    "direction_mode": DIRECTION_MODE,
    "direction_support_sha256": sha256_file(Path(__file__).with_name("direction_support.py")),
    "patch_cache": PATCH_CACHE,
    "analysis_role": "followup_T3_input_direction_comparison",
    "supervisor_question": "Sergio 1771742555; user authorised this follow-up on 2026-09-07",
    "design_note": "T3 and ten paired seeds are a new follow-up design chosen after previous results; not an original prespecified contrast. Compare centered with frozen historical results, not T3 against T9.",
    "label_pipeline_correction": "Only non-uncertain GUI rows and label_gui are used; pseudo-labelled rows are excluded from fitting and evaluation; Early Morula is merged into Morula; 5-8 cell is normalized to the thesis label 5-16 cell.",
    "script_path": str(SCRIPT_PATH),
    "script_sha256": SCRIPT_SHA256,
    "script_snapshot": str(script_snapshot),
    "ROOT": str(ROOT),
    "SEQ_CSV": str(SEQ_CSV),
    "PSEUDO_CSV": str(PSEUDO_CSV),
    "GUI_CSV": str(GUI_CSV),
    "BIG_ROOT": str(BIG_ROOT),
    "OUT_DIR": str(OUT_DIR),
    "SWEEP_K": SWEEP_K,
    "EPOCHS": EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "ACCUM_STEPS": ACCUM_STEPS,
    "NUM_WORKERS": NUM_WORKERS,
    "USE_CKPT": int(USE_CKPT),
    "USE_CBAM": int(USE_CBAM),
    "FULL_SEEDS_FOR_ALL_T": int(FULL_SEEDS_FOR_ALL_T),
    "FREEZE_EPOCHS": FREEZE_EPOCHS,
    "CENTER_ONLY_EPOCHS": CENTER_ONLY_EPOCHS,
    "LR_FRAME": LR_FRAME,
    "LR_TEMPORAL": LR_TEMPORAL,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "SPLIT_SEED": SPLIT_SEED,
    "RUN_SEEDS": RUN_SEEDS,
    "EARLY_STOP_PAT": EARLY_STOP_PAT,
    "MIN_EPOCHS_FOR_EARLYSTOP": MIN_EPOCHS_FOR_EARLYSTOP,
    "LABEL_MERGE_EM_TO_MO": LABEL_MERGE_EM_TO_MO,
    "EXPECTED_CLASSES": sorted(EXPECTED_CLASSES),
    "EXPECTED_GUI_SEQUENCE_COUNT": EXPECTED_GUI_SEQUENCE_COUNT,
    "EXPECTED_SPLIT_COUNTS": EXPECTED_SPLIT_COUNTS,
    "SAVE_EPOCH_CKPT_EVERY": SAVE_EPOCH_CKPT_EVERY,
    "NUM_JOBS": int(ARGS.num_jobs),
    "SKIP_DONE": int(SKIP_DONE),
    "RESUME": int(RESUME),
}
atomic_write_text(
    SWEEP_ROOT / "meta/sweep_config.json",
    json.dumps(cfg, indent=2, ensure_ascii=False),
)


# Use ten paired seeds for T=3 and T=9.

all_jobs = [
    (int(K), int(s))
    for K in SWEEP_K
    for s in RUN_SEEDS
    if FULL_SEEDS_FOR_ALL_T or (int(s) <= 2) or (int(K) in {3, 9})
]
all_jobs = sorted(all_jobs, key=lambda x: (x[0], x[1]))
if not FULL_SEEDS_FOR_ALL_T and not _only_T and not _only_seeds and len(all_jobs) != 44:
    die(f"Corrected protocol requires 44 jobs; constructed {len(all_jobs)}: {all_jobs}")

num_jobs = max(1, int(ARGS.num_jobs))
job_index = int(ARGS.job_index)
if job_index < 0 or job_index >= num_jobs:
    die(f"job_index out of range: job_index={job_index}, num_jobs={num_jobs}")

shard_jobs = [j for i, j in enumerate(all_jobs) if (i % num_jobs) == job_index]
print(f"[Jobs] total={len(all_jobs)} shard={len(shard_jobs)} (index={job_index}/{num_jobs})", flush=True)
shard_cfg = {
    "script_sha256": SCRIPT_SHA256,
    "num_jobs": num_jobs,
    "job_index": job_index,
    "all_jobs": all_jobs,
    "shard_jobs": shard_jobs,
}
atomic_write_text(
    SWEEP_ROOT / "meta" / f"shard_{job_index:02d}_of_{num_jobs:02d}.json",
    json.dumps(shard_cfg, indent=2),
)
if bool(int(ARGS.preflight_only)):
    print("[Preflight] Data, labels, split, classes, and job grid passed. No training started.", flush=True)
    raise SystemExit(0)
if not shard_jobs:
    print("[Jobs] Nothing to run in this shard. Exit.", flush=True)
    raise SystemExit(0)


summary_rows = []
t_sweep0 = time.time()

for (K, run_seed) in tqdm(shard_jobs, disable=(not SHOW_TQDM), desc="shard jobs", leave=True):
    set_all_seeds(int(run_seed))
    run_t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    exp_name = f"{BASE_EXP}__{DIRECTION_MODE}_T{int(K):02d}__seed{int(run_seed):02d}"
    EXP_DIR = SWEEP_ROOT / "runs" / f"exp_{exp_name}"
    (EXP_DIR / "ckpt").mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "eval").mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "meta").mkdir(parents=True, exist_ok=True)

    split_df.to_csv(EXP_DIR / "meta/fixed_split_seq_id.csv", index=False, encoding="utf-8-sig")
    df_rows.to_csv(EXP_DIR / "meta/sequence_rows_used.csv", index=False, encoding="utf-8-sig")
    run_config = {
        "direction_mode": DIRECTION_MODE,
        "patch_cache": PATCH_CACHE,
        "T": int(K),
        "run_seed": int(run_seed),
        "split_seed": int(SPLIT_SEED),
        "script_path": str(SCRIPT_PATH),
        "script_sha256": SCRIPT_SHA256,
        "sweep_config": str(SWEEP_ROOT / "meta/sweep_config.json"),
        "fixed_split": str(EXP_DIR / "meta/fixed_split_seq_id.csv"),
        "sequence_rows_used": str(EXP_DIR / "meta/sequence_rows_used.csv"),
        "cohort": "GUI-reviewed non-uncertain sequences only; pseudo labels excluded",
        "split_counts": split_count_actual,
        "classes": classes,
    }
    (EXP_DIR / "meta/run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    LOG_FILE = EXP_DIR / "logs/metrics.csv"
    if not LOG_FILE.exists():
        LOG_FILE.write_text("epoch,train_loss,train_acc,train_f1,val_loss,val_acc,val_f1,lr_frame,lr_temporal\n", encoding="utf-8")

    DONE_MARK = EXP_DIR / "eval/DONE.txt"
    test_pred_csv = EXP_DIR / "eval/test_predictions.csv"
    run_summary_json = EXP_DIR / "eval/run_summary.json"
    if SKIP_DONE and (DONE_MARK.exists() or test_pred_csv.exists()):
        if not run_summary_json.exists():
            die(f"Existing output lacks run_summary.json and cannot be trusted: {EXP_DIR}")
        try:
            existing_summary = json.loads(run_summary_json.read_text(encoding="utf-8"))
        except Exception as exc:
            die(f"Existing run summary is unreadable: {run_summary_json}: {exc!r}")
        if existing_summary.get("script_sha256") != SCRIPT_SHA256:
            die(
                "Existing output was produced by a different or unrecorded script; "
                f"refusing to skip it: {EXP_DIR}"
            )
        print(f"[Skip] Already done: T={K} seed={run_seed} -> {EXP_DIR}", flush=True)
        summary_rows.append(existing_summary)
        continue

    printl = printl_factory(EXP_DIR / "logs/run.log", verbose=True)
    printl("")
    printl(f"Start T={K} seed={run_seed} exp={exp_name}", always=True)
    printl(f"Data: train={len(tr_mani)} val={len(va_mani)} test={len(te_mani)} classes={len(classes)}", always=True)

    npt = neptune_connect(
        run_name=f"{NEPTUNE_RUN_PREFIX}_T{int(K):02d}_seed{int(run_seed):02d}",
        tags=NEPTUNE_TAGS_BASE + [f"T={K}", f"seed={run_seed}", f"split_seed={SPLIT_SEED}", f"shard={job_index}/{num_jobs}"],
        params={
            "target_frames": int(K),
            "mode": DIRECTION_MODE,
            "model": "r50+cbam+tfm",
            "best_metric": "val_macro_f1",
            "epochs": EPOCHS,
            "lr_frame": LR_FRAME,
            "lr_temporal": LR_TEMPORAL,
            "freeze_epochs": FREEZE_EPOCHS,
            "split_seed": int(SPLIT_SEED),
            "run_seed": int(run_seed),
            "batch_size": BATCH_SIZE,
            "accum_steps": ACCUM_STEPS,
            "num_workers": NUM_WORKERS,
            "use_ckpt": int(USE_CKPT),
        },
    )

    ds_tr = SeqDataset(df_rows, tr_mani, classes, target_frames=K, train=True,  temporal_dropout=0.15, black_fix=True)
    ds_va = SeqDataset(df_rows, va_mani, classes, target_frames=K, train=False, temporal_dropout=0.0,  black_fix=True)
    ds_te = SeqDataset(df_rows, te_mani, classes, target_frames=K, train=False, temporal_dropout=0.0,  black_fix=True)

    train_labels = [cls2id[c] for c in tr_mani["label"].tolist()]
    cnt = Counter(train_labels)
    freq = np.array([cnt.get(i, 0) for i in range(len(classes))], dtype=np.float32)
    probs = np.array([1.0 / (freq[y] ** 0.5 if freq[y] > 0 else 1.0) for y in train_labels], dtype=np.float64)
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
    sampler = WeightedRandomSampler(weights=probs, num_samples=len(train_labels), replacement=True)

    common_dl_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        collate_fn=pad_collate,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        persistent_workers=(PERSISTENT_WORKERS if NUM_WORKERS > 0 else False),
    )

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler, **common_dl_kwargs)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, **common_dl_kwargs)
    dl_te = DataLoader(ds_te, batch_size=max(1, BATCH_SIZE * 2), shuffle=False, **common_dl_kwargs)

    model = CNNTransR50CBAM(len(classes), max_len=max(64, int(K)), pretrained=True, use_ckpt=USE_CKPT).to(device)

    pg_backbone, pg_head = [], []
    for n, p in model.named_parameters():
        (pg_backbone if ("backbone" in n) else pg_head).append(p)
    if not pg_backbone:
        pg_backbone = [p for p in model.parameters()]

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in pg_backbone if p.requires_grad], "lr": LR_FRAME, "weight_decay": WEIGHT_DECAY},
            {"params": [p for p in pg_head if p.requires_grad], "lr": LR_TEMPORAL, "weight_decay": WEIGHT_DECAY},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    cb_w = class_balanced_weights(len(classes), train_labels, beta=0.999).to(device)
    criterion = FocalLoss(gamma=1.5, weight=cb_w, label_smoothing=0.05, reduction="none")
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    def set_backbone_requires_grad(flag: bool):
        for n, p in model.named_parameters():
            if "backbone" in n:
                p.requires_grad = flag

    # Keep the best checkpoint when resuming.
    
    start_epoch = 1
    best_score, best_epoch, best_val_acc = -1.0, -1, -1.0
    no_improve = 0
    last_ckpt_path = EXP_DIR / "ckpt/last.pth"
    if RESUME and last_ckpt_path.exists():
        try:
            ck = torch.load(last_ckpt_path, map_location="cpu")
            model.load_state_dict(ck.get("model", {}), strict=False)
            if "opt" in ck:
                optimizer.load_state_dict(ck["opt"])
            if "sch" in ck:
                scheduler.load_state_dict(ck["sch"])
            start_epoch = int(ck.get("epoch", 1))
            best_score = float(ck.get("best_score", best_score))
            best_epoch = int(ck.get("best_epoch", best_epoch))
            best_val_acc = float(ck.get("best_val_acc", best_val_acc))
            no_improve = int(ck.get("no_improve", no_improve))

            best_path = EXP_DIR / "ckpt/best.pth"
            if best_path.exists():
                best_ck = torch.load(best_path, map_location="cpu")
                best_score = float(best_ck.get("best_score", best_score))
                best_epoch = int(best_ck.get("epoch", best_epoch))
                best_val_acc = float(best_ck.get("best_val_acc", best_val_acc))
            printl(
                f"[Resume] loaded last.pth, start_epoch={start_epoch}, "
                f"best_val_f1={best_score:.4f}@{best_epoch}",
                always=True,
            )
        except Exception as e:
            printl(f"[Resume] failed: {repr(e)} -> start from scratch", always=True)
            start_epoch = 1

    set_backbone_requires_grad(False)
    printl(f"Train: freeze backbone until epoch {FREEZE_EPOCHS}", always=True)

    
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_t0 = time.time()

        if epoch == FREEZE_EPOCHS + 1:
            set_backbone_requires_grad(True)
            printl("Train: unfreeze backbone", always=True)

        model.train()
        tr_losses, tr_preds, tr_labels_list = [], [], []
        optimizer.zero_grad(set_to_none=True)

        train_pbar = tqdm(
            dl_tr,
            total=len(dl_tr),
            disable=(not SHOW_TQDM),
            leave=TQDM_LEAVE,
            desc=f"train T={K} seed={run_seed} e{epoch}/{EPOCHS}",
        )

        # gradient accumulation
        accum = ACCUM_STEPS
        step_count = 0

        for b, (X, y, meta) in enumerate(train_pbar, 1):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = meta["mask"].to(device, non_blocking=True)
            w_batch = meta["weight"].to(device, non_blocking=True)

            if CENTER_ONLY_EPOCHS and (epoch <= CENTER_ONLY_EPOCHS):
                m2 = torch.zeros_like(mask, dtype=torch.bool)
                for i in range(mask.shape[0]):
                    last_idx = int(torch.where(mask[i])[0][-1].item()) if mask[i].any() else mask.shape[1] - 1
                    m2[i, last_idx] = True
                mask = m2

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(X, mask)
                loss_vec = criterion(logits, y) * w_batch
                loss = loss_vec.mean()
                # scale loss for accumulation
                loss_accum = loss / float(accum)

            scaler.scale(loss_accum).backward()
            step_count += 1

            # record metrics on the unscaled loss (for logging)
            tr_losses.append(float(loss.item()))
            tr_preds.extend(logits.argmax(1).detach().cpu().numpy())
            tr_labels_list.extend(y.detach().cpu().numpy())

            if (step_count % accum == 0) or (b == len(dl_tr)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if (b % POSTFIX_EVERY_STEPS == 0) or (b == len(dl_tr)):
                tail = tr_losses[-min(50, len(tr_losses)) :]
                train_pbar.set_postfix(
                    {"loss": f"{np.mean(tail):.4f}",
                     "lr_f": f"{optimizer.param_groups[0]['lr']:.1e}",
                     "lr_t": f"{optimizer.param_groups[1]['lr']:.1e}",
                     "accum": accum}
                )

        train_pbar.close()

        tr_acc = accuracy_score(tr_labels_list, tr_preds) if tr_labels_list else 0.0
        tr_f1 = f1_score(tr_labels_list, tr_preds, average="macro", zero_division=0) if tr_labels_list else 0.0

        # Val
        model.eval()
        va_losses, va_preds, va_labels_list = [], [], []

        val_pbar = tqdm(
            dl_va,
            total=len(dl_va),
            disable=(not SHOW_TQDM),
            leave=False,
            desc=f"val   T={K} seed={run_seed} e{epoch}/{EPOCHS}",
        )

        with torch.no_grad():
            for b, (X, y, meta) in enumerate(val_pbar, 1):
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                mask = meta["mask"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(X, mask)
                    loss_vec = criterion(logits, y)
                    loss = loss_vec.mean()

                va_losses.append(float(loss.item()))
                va_preds.extend(logits.argmax(1).cpu().numpy())
                va_labels_list.extend(y.cpu().numpy())

                if (b % POSTFIX_EVERY_STEPS == 0) or (b == len(dl_va)):
                    val_pbar.set_postfix({"loss": f"{np.mean(va_losses[-min(50,len(va_losses)):]):.4f}"})

        val_pbar.close()

        va_acc = accuracy_score(va_labels_list, va_preds) if va_labels_list else float("nan")
        va_f1 = f1_score(va_labels_list, va_preds, average="macro", zero_division=0) if va_labels_list else float("nan")

        scheduler.step()

        lr_frame = optimizer.param_groups[0]["lr"]
        lr_tmp = optimizer.param_groups[1]["lr"]

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},"
                f"{(np.mean(tr_losses) if tr_losses else 0.0):.6f},{tr_acc:.6f},{tr_f1:.6f},"
                f"{(np.mean(va_losses) if va_losses else 0.0):.6f},{(va_acc if not math.isnan(va_acc) else 0.0):.6f},"
                f"{(va_f1 if not math.isnan(va_f1) else 0.0):.6f},"
                f"{lr_frame:.8f},{lr_tmp:.8f}\n"
            )

        if npt is not None:
            try:
                npt["epoch"] = epoch
                npt["metrics/train/loss"].append(float(np.mean(tr_losses) if tr_losses else 0.0))
                npt["metrics/train/acc"].append(float(tr_acc))
                npt["metrics/train/f1_macro"].append(float(tr_f1))
                npt["metrics/val/loss"].append(float(np.mean(va_losses) if va_losses else 0.0))
                npt["metrics/val/acc"].append(float(va_acc if not math.isnan(va_acc) else 0.0))
                npt["metrics/val/f1_macro"].append(float(va_f1 if not math.isnan(va_f1) else 0.0))
            except Exception:
                pass

        # ckpt
        torch.save(
            {"model": model.state_dict(), "opt": optimizer.state_dict(), "sch": scheduler.state_dict(),
             "epoch": epoch + 1, "classes": classes,
             "best_score": float(best_score), "best_epoch": int(best_epoch),
             "best_val_acc": float(best_val_acc), "no_improve": int(no_improve)},
            EXP_DIR / "ckpt/last.pth",
        )
        if SAVE_EPOCH_CKPT_EVERY and (epoch % int(SAVE_EPOCH_CKPT_EVERY) == 0):
            torch.save(
                {"model": model.state_dict(), "opt": optimizer.state_dict(), "sch": scheduler.state_dict(),
                 "epoch": epoch + 1, "classes": classes},
                EXP_DIR / f"ckpt/epoch_{epoch:02d}.pth",
            )

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
                EXP_DIR / "ckpt/best.pth",
            )
            printl(f"New best: T={K} seed={run_seed} val_f1={best_score:.4f} val_acc={best_val_acc:.4f} epoch={best_epoch}", always=True)
        else:
            no_improve += 1

        dt = time.time() - epoch_t0
        msg = (
            f"T={K} seed={run_seed} epoch={epoch:03d}/{EPOCHS} "
            f"tr loss={np.mean(tr_losses):.4f} acc={tr_acc:.3f} f1={tr_f1:.3f} "
            f"va loss={np.mean(va_losses):.4f} acc={va_acc:.3f} f1={va_f1:.3f} "
            f"best_va_f1={best_score:.4f}@{best_epoch} "
            f"no_imp={no_improve}/{EARLY_STOP_PAT} time={fmt_secs(dt)}"
        )
        printl(msg)

        if (epoch >= MIN_EPOCHS_FOR_EARLYSTOP) and (no_improve >= EARLY_STOP_PAT):
            printl(f"Early stop: T={K} seed={run_seed} at epoch {epoch}, best val_f1={best_score:.4f} at epoch {best_epoch}", always=True)
            break

        cuda_purge()

    # Eval best
    printl("Eval: load best ckpt and export predictions/confusion...", always=True)
    ck_best = torch.load(EXP_DIR / "ckpt/best.pth", map_location="cpu")
    model.load_state_dict(ck_best["model"], strict=False)
    model.to(device).eval()

    train_csv = EXP_DIR / "eval/train_predictions.csv"
    test_csv = EXP_DIR / "eval/test_predictions.csv"

    tr_acc2, tr_f12, cm_tr, rep_tr = export_predictions(model, dl_tr, classes, train_csv)
    test_eval_t0 = time.perf_counter()
    te_acc2, te_f12, cm_te, rep_te = export_predictions(model, dl_te, classes, test_csv)
    test_eval_seconds = time.perf_counter() - test_eval_t0
    peak_vram_gib = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if device.type == "cuda"
        else float("nan")
    )
    run_seconds = time.perf_counter() - run_t0

    save_confusion_png(cm_tr, classes, EXP_DIR / "eval/confusion_train.png", f"Confusion Train (T={K}, seed={run_seed})")
    save_confusion_png(cm_te, classes, EXP_DIR / "eval/confusion_test.png", f"Confusion Test (T={K}, seed={run_seed})")
    plot_curve_from_csv(LOG_FILE, EXP_DIR / "logs/curve.png")

    printl(f"\n[Train report]\n{rep_tr}\n", always=True)
    printl(f"\n[Test report]\n{rep_te}\n", always=True)

    # mark done
    DONE_MARK.write_text(
        f"DONE\nT={K}\nseed={run_seed}\nscript_sha256={SCRIPT_SHA256}\n",
        encoding="utf-8",
    )

    if npt is not None:
        try:
            import neptune
            npt["plots/curve"] = neptune.types.File(str(EXP_DIR / "logs/curve.png"))
            npt["plots/confusion_train"] = neptune.types.File.as_image(str(EXP_DIR / "eval/confusion_train.png"))
            npt["plots/confusion_test"] = neptune.types.File.as_image(str(EXP_DIR / "eval/confusion_test.png"))
            npt["summary/best_val_f1"] = float(ck_best.get("best_score", best_score))
            npt["summary/best_val_acc"] = float(ck_best.get("best_val_acc", best_val_acc))
            npt["summary/best_epoch"] = int(ck_best.get("epoch", best_epoch))
            npt["summary/test_acc@best"] = float(te_acc2)
            npt["summary/test_f1@best"] = float(te_f12)
            if NEPTUNE_UPLOAD_PRED_CSV:
                npt["predictions/train_csv"] = neptune.types.File(str(train_csv))
                npt["predictions/test_csv"] = neptune.types.File(str(test_csv))
        except Exception:
            pass
        try:
            npt.stop()
        except Exception:
            pass

    row = {
        "T": int(K),
        "run_seed": int(run_seed),
        "split_seed": int(SPLIT_SEED),
        "script_sha256": SCRIPT_SHA256,
        "best_epoch": int(ck_best.get("epoch", best_epoch)),
        "best_val_f1": float(ck_best.get("best_score", best_score)),
        "best_val_acc": float(ck_best.get("best_val_acc", best_val_acc)),
        "train_acc@best": float(tr_acc2),
        "train_f1@best": float(tr_f12),
        "test_acc@best": float(te_acc2),
        "test_f1@best": float(te_f12),
        "run_seconds": float(run_seconds),
        "test_eval_seconds": float(test_eval_seconds),
        "test_sequences": int(len(te_mani)),
        "test_seconds_per_sequence": float(test_eval_seconds / max(1, len(te_mani))),
        "peak_vram_gib": float(peak_vram_gib),
        "exp_dir": str(EXP_DIR),
        "shard_num_jobs": int(num_jobs),
        "shard_job_index": int(job_index),
    }
    run_summary_json.write_text(json.dumps(row, indent=2), encoding="utf-8")
    summary_rows.append(row)

    printl(
        f"Done T={K} seed={run_seed} best_epoch={row['best_epoch']} "
        f"val_f1={row['best_val_f1']:.4f} val_acc={row['best_val_acc']:.4f} "
        f"test_f1={row['test_f1@best']:.4f} test_acc={row['test_acc@best']:.4f}",
        always=True,
    )


df_runs = pd.DataFrame(summary_rows)
if not df_runs.empty:
    df_runs = df_runs.sort_values(["T", "run_seed"]).reset_index(drop=True)
runs_csv = SWEEP_ROOT / f"ablation_runs_shard_{job_index:02d}_of_{num_jobs:02d}.csv"
df_runs.to_csv(runs_csv, index=False, encoding="utf-8-sig")
print("Saved:", runs_csv)

# Full aggregation across all shards (best-effort: concat existing shard files)
all_run_files = sorted(SWEEP_ROOT.glob("ablation_runs_shard_*_of_*.csv"))
dfs = []
for fp in all_run_files:
    try:
        dfs.append(pd.read_csv(fp))
    except Exception:
        pass

if dfs:
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["T", "run_seed", "split_seed"], keep="last")
    df_all = df_all.sort_values(["T", "run_seed"]).reset_index(drop=True)

    full_runs_csv = SWEEP_ROOT / "ablation_runs.csv"
    df_all.to_csv(full_runs_csv, index=False, encoding="utf-8-sig")
    print("Saved:", full_runs_csv)

    df_agg = (
        df_all.groupby("T")
        .agg(
            n=("run_seed", "count"),
            val_f1_mean=("best_val_f1", "mean"),
            val_f1_std=("best_val_f1", "std"),
            val_acc_mean=("best_val_acc", "mean"),
            val_acc_std=("best_val_acc", "std"),
            test_f1_mean=("test_f1@best", "mean"),
            test_f1_std=("test_f1@best", "std"),
            test_acc_mean=("test_acc@best", "mean"),
            test_acc_std=("test_acc@best", "std"),
            run_seconds_mean=("run_seconds", "mean"),
            run_seconds_std=("run_seconds", "std"),
            test_seconds_per_sequence_mean=("test_seconds_per_sequence", "mean"),
            test_seconds_per_sequence_std=("test_seconds_per_sequence", "std"),
            peak_vram_gib_mean=("peak_vram_gib", "mean"),
            peak_vram_gib_std=("peak_vram_gib", "std"),
        )
        .reset_index()
        .sort_values("T")
    )

    agg_csv = SWEEP_ROOT / "ablation_summary_mean_std.csv"
    df_agg.to_csv(agg_csv, index=False, encoding="utf-8-sig")
    print("Saved:", agg_csv)
    print(df_agg.to_string(index=False))

    # paired t-test vs reference window (optional; requires scipy)
    def paired_ttest_vs_ref(df_runs_in: pd.DataFrame, ref_T: int = 21, metric: str = "test_f1@best") -> Optional[pd.DataFrame]:
        try:
            from scipy.stats import ttest_rel
        except Exception:
            print("[WARN] scipy not installed; skip paired t-test")
            return None

        ref = df_runs_in[df_runs_in["T"] == ref_T].set_index("run_seed")[metric]
        out = []
        for T in sorted(df_runs_in["T"].unique()):
            if int(T) == int(ref_T):
                continue
            cur = df_runs_in[df_runs_in["T"] == T].set_index("run_seed")[metric]
            common = sorted(set(ref.index) & set(cur.index))
            if len(common) < 2:
                continue
            stat, p = ttest_rel(cur.loc[common].values, ref.loc[common].values)
            out.append({"T": int(T), "ref_T": int(ref_T), "metric": metric, "n": int(len(common)), "p_value": float(p)})
        if not out:
            return None
        return pd.DataFrame(out).sort_values("T").reset_index(drop=True)

    df_p = paired_ttest_vs_ref(df_all, ref_T=21, metric="test_f1@best")
    if df_p is not None:
        p_csv = SWEEP_ROOT / "ablation_paired_ttest_vs_T21.csv"
        df_p.to_csv(p_csv, index=False, encoding="utf-8-sig")
        print("Saved:", p_csv)
        print(df_p.to_string(index=False))

    # plot (mean curves)
    plot_png = SWEEP_ROOT / "ablation_summary_plot_mean.png"
    try:
        fig = plt.figure(figsize=(7, 4))
        plt.plot(df_agg["T"], df_agg["val_f1_mean"], marker="o", label="val_f1_mean")
        plt.plot(df_agg["T"], df_agg["test_f1_mean"], marker="o", linestyle="--", label="test_f1_mean")
        plt.xlabel("target frames T (hist_center)")
        plt.ylabel("macro F1 (mean over seeds)")
        plt.title("Ablation: hist_center frames (mean over seeds)")
        plt.legend()
        plt.tight_layout()
        fig.savefig(plot_png, dpi=150)
        plt.close(fig)
        print("Saved:", plot_png)
    except Exception as e:
        print("Plot failed:", repr(e))

total_time = time.time() - t_sweep0
print("")
print("Final summary (hist_center, multi-seed, sharded)")
print("Sweep dir:", str(SWEEP_ROOT))
print("Total time:", fmt_secs(total_time))
print("All done")
