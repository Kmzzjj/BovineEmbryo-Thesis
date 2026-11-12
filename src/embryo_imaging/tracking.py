"""Track detections in time-lapse images grouped by well and focal plane."""

import argparse
import csv
import gc
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np


EMBRYO_CLASS = 0
CONF_TH = 0.40
IOU_TH = 0.50
IMGSZ = 640
MAX_DET = 4
MIN_BOX_AREA = 150
VID_STRIDE = 1
SAVE_FPS = 12.0
WORKERS = 1
CLEAN_EVERY = 50

FRAME_NAME = re.compile(
    r"^(?P<run>.+)_(?P<pos>P\d+)_(?P<fp>FP\d+)_(?P<time>\d+)"
    r"(?:_(?P<inst>I\d+))?\.jpe?g$",
    re.IGNORECASE,
)
TRACK_COLUMNS = (
    "frame", "time_min", "track_id", "cls", "conf",
    "x1", "y1", "x2", "y2", "W", "H", "filename",
)


def parse_name(frame: Path):
    match = FRAME_NAME.match(frame.name)
    if match is None:
        return None
    metadata = match.groupdict()
    metadata["time"] = int(metadata["time"])
    return metadata


def group_frames(run_dir: Path, focal_plane: str) -> dict[str, list[Path]]:
    groups = {}
    for frame in run_dir.iterdir():
        metadata = parse_name(frame)
        if metadata is None or metadata["fp"].upper() != focal_plane or not frame.is_file():
            continue
        key = f"{metadata['pos'].upper()}_{focal_plane}"
        groups.setdefault(key, []).append((metadata["time"], frame.name, frame))

    return {
        key: [frame for _, _, frame in sorted(groups[key])]
        for key in sorted(groups)
    }


def init_vwriter(video_path: Path, width: int, height: int):
    for codec in ("mp4v", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(video_path), fourcc, SAVE_FPS, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise OSError(f"Could not open video writer: {video_path}")


def draw_box_id(image, x1, y1, x2, y2, track_id, confidence):
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    color = (0, 220, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = f"ID {track_id}  {confidence:.2f}"
    (text_width, text_height), _ = cv2.getTextSize(label, font, 0.6, 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(image, (x1, y1 - text_height - 6), (x1 + text_width + 6, y1), color, -1)
    cv2.putText(image, label, (x1 + 3, y1 - 4), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def track_rows(boxes, local_ids):
    if boxes is None or not len(boxes):
        return []

    classes = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    coords = boxes.xyxy.cpu().numpy()
    ids = getattr(boxes, "id", None)
    ids = ids.int().cpu().numpy() if ids is not None else np.full(len(classes), -1, dtype=int)

    picked = []
    for global_id, cls, confidence, (x1, y1, x2, y2) in zip(ids, classes, confidences, coords):
        if cls != EMBRYO_CLASS or (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
            continue
        local_id = local_ids.setdefault(global_id, len(local_ids))
        picked.append((local_id, int(cls), float(confidence), float(x1), float(y1), float(x2), float(y2)))

    picked.sort(key=lambda row: row[2], reverse=True)
    return picked[:MAX_DET]


def track_sequence(model, frames, sequence_dir: Path, device, *, save_video: bool, overwrite: bool):
    import torch

    csv_path = sequence_dir / "tracks.csv"
    if csv_path.exists() and not overwrite:
        print(f"  Skipping {sequence_dir.name}: tracks.csv already exists")
        return

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        print(f"  Skipping {sequence_dir.name}: could not read {frames[0]}")
        return
    height, width = first.shape[:2]
    sequence_dir.mkdir(parents=True, exist_ok=True)
    sequence = frames[::VID_STRIDE]
    print(f"  {sequence_dir.name}: {len(sequence)} frames")

    with TemporaryDirectory(prefix=".tracking-", dir=sequence_dir) as pending:
        staged_csv = Path(pending) / csv_path.name
        staged_video = Path(pending) / "tracked.mp4"
        video = init_vwriter(staged_video, width, height) if save_video else None
        local_ids = {}
        frame_number = 0
        try:
            results = model.track(
                source=[str(frame) for frame in sequence],
                tracker="botsort.yaml",
                persist=True,
                stream=True,
                device=device,
                imgsz=IMGSZ,
                conf=CONF_TH,
                iou=IOU_TH,
                max_det=MAX_DET,
                verbose=False,
                workers=WORKERS,
                half=True,
                agnostic_nms=False,
            )
            with staged_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(TRACK_COLUMNS)
                for frame_number, (frame, result) in enumerate(zip(sequence, results), 1):
                    time_min = parse_name(frame)["time"]
                    rows = track_rows(result.boxes, local_ids)
                    writer.writerows(
                        (frame_number, time_min, *row, width, height, frame.name)
                        for row in rows
                    )

                    if video is not None:
                        image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
                        if image is not None:
                            for track_id, _, confidence, x1, y1, x2, y2 in rows:
                                draw_box_id(image, x1, y1, x2, y2, track_id, confidence)
                            video.write(image)

                    if frame_number % CLEAN_EVERY == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            if frame_number != len(sequence):
                raise RuntimeError(f"Tracker returned {frame_number} results for {len(sequence)} frames")
        finally:
            if video is not None:
                video.release()

        if video is not None:
            staged_video.replace(sequence_dir / staged_video.name)
        # The CSV is also the marker used to skip completed sequences.
        staged_csv.replace(csv_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True, type=Path, help="Directory containing one folder per run")
    parser.add_argument("--weights", required=True, type=Path, help="Trained YOLO detector weights")
    parser.add_argument("--output", required=True, type=Path, help="Directory for track CSVs and videos")
    parser.add_argument("--focal-plane", default="FP3", type=str.upper)
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu or 0 (default: auto)")
    parser.add_argument("--no-video", action="store_true", help="Write track CSVs only")
    parser.add_argument("--overwrite", action="store_true", help="Replace completed track outputs")
    args = parser.parse_args()
    if not args.frames.is_dir():
        parser.error(f"frame directory does not exist: {args.frames}")
    if not args.weights.is_file():
        parser.error(f"weights file does not exist: {args.weights}")
    if not re.fullmatch(r"FP\d+", args.focal_plane):
        parser.error("--focal-plane must have the form FP3")
    return args


def main():
    args = parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip().lower() == "cuda":
        del os.environ["CUDA_VISIBLE_DEVICES"]

    import torch
    from ultralytics import YOLO

    device = args.device or (0 if torch.cuda.is_available() else "cpu")
    print(f"YOLO: {args.weights} (device={device})")
    model = YOLO(str(args.weights))
    runs = sorted(run for run in args.frames.iterdir() if run.is_dir())
    if not runs:
        print(f"No run folders found in {args.frames}")
        return

    for index, run_dir in enumerate(runs, 1):
        print(f"[{index}/{len(runs)}] {run_dir.name}")
        for key, frames in group_frames(run_dir, args.focal_plane).items():
            track_sequence(
                model, frames, args.output / run_dir.name / key, device,
                save_video=not args.no_video, overwrite=args.overwrite,
            )

    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
