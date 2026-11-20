"""Default paths relative to the directory where the command is run."""

import re
from pathlib import Path

DATA_YAML = Path("datasets/bovine_det/dataset.yaml")
PRETRAINED = Path("models/yolov8n.pt")
RUNS_DIR = Path("runs")


_IMAGE_NAME = re.compile(
    r"^(?P<dish>[^_]+)_P(?P<p>\d+)_FP(?P<fp>\d+)_(?P<tmin>-?\d+)\.jpg$",
    re.IGNORECASE,
)


def parse_from_filename(filename):
    match = _IMAGE_NAME.match(Path(str(filename)).name)
    if match is None:
        return None, None, None, None
    return match["dish"], int(match["p"]), int(match["fp"]), int(match["tmin"])


def big_image_path(root: Path, dish: str, position: int, plane: int, minute: int) -> Path:
    return Path(root) / f"MTL-0034-{dish}" / f"{dish}_P{int(position)}_FP{int(plane)}_{int(minute)}.jpg"
