from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


DATASET_DIR = ROOT / "datasets" / "bovine_det"
DATA_YAML = DATASET_DIR / "dataset.yaml"


PRETRAINED = ROOT / "models" / "yolov8n.pt"


RUNS_DIR = ROOT / "runs"