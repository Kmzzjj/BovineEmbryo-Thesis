import os
import re
import gc
import csv
import glob
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import torch
import numpy as np
from ultralytics import YOLO

# env
FRAME_ROOT = r"F:\Bovine"
WEIGHTS    = r"E:\Pythonprojectall\BovineEmbryoDetection\runs\yolov8n_bovine_det_70\weights\best.pt"
OUT_ROOT   = r"F:\EmbryoTracks_BoT"


FOCAL_PLANES: Optional[set] = {"FP3"}

EMBRYO_CLASS = 0
CONF_TH      = 0.40
IOU_TH       = 0.50
IMGSZ        = 640
MAX_DET      = 4
MIN_BOX_AREA = 150

DEVICE       = 0 if torch.cuda.is_available() else "cpu"
VID_STRIDE   = 1
SAVE_VIDEO   = True
SAVE_FPS     = 12.0
WORKERS      = 1
CLEAN_EVERY  = 50

OVERWRITE    = False


PAT_A = re.compile(r'^(?P<run>.+)_(?P<pos>P\d+)_(?P<fp>FP\d+)_(?P<time>\d+)\.jpe?g$', re.IGNORECASE)
PAT_B = re.compile(r'^(?P<run>.+)_(?P<pos>P\d+)_(?P<fp>FP\d+)_(?P<time>\d+)_(?P<inst>I\d+)\.jpe?g$', re.IGNORECASE)

def parse_name(p: Path):
    n = p.name
    m = PAT_B.match(n) or PAT_A.match(n)
    if not m:
        return None
    d = m.groupdict()
    try:
        d["time"] = int(d["time"])
    except Exception:
        return None
    return d  # {run,pos,fp,time,(inst)}

def list_runs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([d for d in root.iterdir() if d.is_dir()])

def list_pos_fp_keys(run_dir: Path) -> List[str]:
    files = [Path(p) for p in glob.glob(str(run_dir / "*.jpg"))]
    buckets = set()
    for f in files:
        meta = parse_name(f)
        if not meta:
            continue
        fp = meta["fp"].upper()
        if FOCAL_PLANES and fp not in FOCAL_PLANES:
            continue
        buckets.add(f"{meta['pos']}_{fp}")
    return sorted(buckets)

def gather_frames(run_dir: Path, key: str) -> List[Path]:
    pos, fp = key.split("_")
    files: List[Tuple[int, str, Path]] = []
    for p in glob.glob(str(run_dir / f"*_{pos}_{fp}_*.jpg")):
        pp = Path(p)
        meta = parse_name(pp)
        if not meta:
            continue
        files.append((meta["time"], pp.name, pp))
    files.sort(key=lambda x: (x[0], x[1]))  
    return [pp for _, __, pp in files]

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def init_vwriter(out_path: Path, w: int, h: int, fps: float):
    ensure_parent(out_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not vw.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    return vw

def draw_box_id(img, x1, y1, x2, y2, tid, conf=None):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 255), 2)
    label = f"ID {tid}" + (f"  {conf:.2f}" if conf is not None else "")
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), (0, 220, 255), -1)
    cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

def track_sequence(model: YOLO, frames: List[Path], out_dir: Path, key: str):
    if not frames:
        return
    print(f"  ▶ 序列 {key}: {len(frames)} 帧")

    csv_path = out_dir / key / "tracks.csv"
    vid_path = out_dir / key / "tracked.mp4"
    if csv_path.exists() and not OVERWRITE:
        print(f"    [SKIP] exist {csv_path}，skip（OVERWRITE=False）")
        return

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        print(" [SKIP] ")
        return
    H, W = first.shape[:2]
    ensure_parent(csv_path)

    vw = init_vwriter(vid_path, W, H, SAVE_FPS) if SAVE_VIDEO else None


    seq_paths = frames[::VID_STRIDE]
    str_paths = [str(p) for p in seq_paths]

    # BoT-SORT
    gen = model.track(
        source=str_paths,
        tracker="botsort.yaml",
        persist=True,
        stream=True,
        device=DEVICE,
        imgsz=IMGSZ,
        conf=CONF_TH,
        iou=IOU_TH,
        max_det=MAX_DET,
        verbose=False,
        workers=WORKERS,
        half=True,
        agnostic_nms=False
    )

    global_to_local: Dict[int, int] = {}
    next_local_id = 0

    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow([
            "frame", "time_min", "track_id", "cls", "conf",
            "x1", "y1", "x2", "y2", "W", "H", "filename"
        ])

    for i, (p, r) in enumerate(zip(seq_paths, gen), 1):
        meta = parse_name(p)
        if meta is None:
            continue

        rows = []
        if r.boxes is not None and len(r.boxes):
            ids   = getattr(r.boxes, "id", None)
            clses = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            xyxy  = r.boxes.xyxy.cpu().numpy()
            if ids is not None:
                ids = ids.int().cpu().numpy()
            else:
                ids = np.full((len(clses),), -1, dtype=int)

            picked = []
            for (gid, c, cf, (x1, y1, x2, y2)) in zip(ids, clses, confs, xyxy):
                if c != EMBRYO_CLASS:
                    continue
                w, h = (x2 - x1), (y2 - y1)
                if w * h < MIN_BOX_AREA:
                    continue

                if gid not in global_to_local:
                    global_to_local[gid] = next_local_id
                    next_local_id += 1
                lid = global_to_local[gid]

                picked.append((lid, c, cf, x1, y1, x2, y2))

            #MAX_DET
            picked.sort(key=lambda t: t[2], reverse=True)
            picked = picked[:MAX_DET]

            for (lid, c, cf, x1, y1, x2, y2) in picked:
                rows.append([
                    i, meta["time"], int(lid), int(c), float(cf),
                    float(x1), float(y1), float(x2), float(y2),
                    int(W), int(H), p.name
                ])

        # input to CSV
        if rows:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                wcsv = csv.writer(f)
                wcsv.writerows(rows)

        # visualize
        if SAVE_VIDEO:
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is not None and rows:
                for _, __, tid, c, cf, x1, y1, x2, y2, *_ in rows:
                    draw_box_id(im, x1, y1, x2, y2, tid, cf)
            if im is not None:
                vw.write(im)

        # clean
        if (i % CLEAN_EVERY) == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if SAVE_VIDEO and vw is not None:
        vw.release()

def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip().lower() == "cuda":
        del os.environ["CUDA_VISIBLE_DEVICES"]

    root = Path(FRAME_ROOT)
    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise FileNotFoundError(f" root not exist：{root}")
    if not Path(WEIGHTS).exists():
        raise FileNotFoundError(f"weights not exist：{WEIGHTS}")

    print(f"[INFO] YOLO: {WEIGHTS} (device={DEVICE})")
    model = YOLO(WEIGHTS)

    runs = list_runs(root)
    if not runs:
        print("Check FRAME_ROOT")
        return

    for idx, run_dir in enumerate(runs, 1):
        print(f"[{idx}/{len(runs)}] {run_dir}")
        keys = list_pos_fp_keys(run_dir)
        if not keys:
            print("  (skip:no compare P*_FP* )")
            continue

        run_out = out_root / run_dir.name
        for key in keys:
            frames = gather_frames(run_dir, key)
            if not frames:
                continue

            seq_csv = run_out / key / "tracks.csv"
            if seq_csv.exists() and not OVERWRITE:
                print(f"  (skip {key}， tracks.csv is exist)")
                continue

            track_sequence(model, frames, run_out, key)

    print("\n[Done] out put root：", out_root.resolve())

if __name__ == "__main__":
    main()