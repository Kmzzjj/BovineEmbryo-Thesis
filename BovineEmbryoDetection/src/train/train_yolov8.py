import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    root = Path(__file__).resolve().parents[2]   
    p = argparse.ArgumentParser("YOLOv8 Trainer (refactored)")
    p.add_argument("--data", type=str,
                   default=str(root / "datasets" / "bovine_det" / "dataset.yaml"),
                   help="path to dataset.yaml")
    p.add_argument("--model", type=str,
                   default=str(root / "models" / "yolov8n.pt"),
                   help="pretrained or checkpoint .pt")
    p.add_argument("--project", type=str,
                   default=str(root / "runs"),
                   help="root dir to save results")
    p.add_argument("--name", type=str, default="yolov8n_bovine_det3",
                   help="experiment subdir under --project")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=str, default="0")  # 'cpu' or '0' or '0,1'
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--nocacheclean", action="store_true",
                   help="do NOT delete dataset caches before training")
    return p.parse_args()

def delete_caches(dataset_yaml: str):
    """删除当前数据集下的 .cache 文件，避免使用旧缓存导致计数不对。"""
    yml = Path(dataset_yaml).resolve()
    root = yml.parent  # .../datasets/bovine_det
    for p in [root, root / "labels", root / "labels/train", root / "labels/val", root / "labels/train_hard"]:
        if p.exists():
            for f in p.glob("*.cache"):
                try:
                    f.unlink()
                    print(f"[cache] removed: {f}")
                except Exception as e:
                    print(f"[cache] failed to remove {f}: {e}")


def main():
    args = parse_args()

    data_p = Path(args.data)
    model_p = Path(args.model)

    if not data_p.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {data_p}")
    if not model_p.exists():
        print(f"[warn] model not found: {model_p}  (fallback to 'yolov8n.pt')")
        args.model = "yolov8n.pt"

    if not args.nocacheclean:
        delete_caches(str(data_p))

    print(f"[cfg] data   = {args.data}")
    print(f"[cfg] model  = {args.model}")
    print(f"[cfg] runs   = {args.project}/{args.name}")
    print(f"[cfg] epochs = {args.epochs}, imgsz = {args.imgsz}, batch = {args.batch}, device = {args.device}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        project=args.project,
        name=args.name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        resume=args.resume,
        mosaic=0.1,
        close_mosaic=10,
        patience=15,
        optimizer="auto",
        verbose=True,
    )

    out = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n[done] weights saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
