# Running the tools

Use an environment with the extras listed in the [README](../README.md). Run commands from your working directory. Paths below are examples. Use authorised images and weights and keep outputs in a separate directory.

## Detection and tracking

Copy [the detector configuration](../configs/detector.example.yaml), set its dataset path and provide matching YOLO image and `label` folders. The example and the tracker both use class 0 for embryos.

~~~sh
embryo-train-detector --data /data/detector.yaml --model /models/yolov8n.pt \
  --project runs --name detector --device cpu
embryo-track --frames /data/time-lapse --weights runs/detector/weights/best.pt \
  --output runs/tracks
~~~

The tracker expects one folder per run and names such as `run01_P4_FP3_5807.jpg`, where `P4` is the well position, `FP3` is the focal plane and 5807 is the capture time in minutes. It writes `tracks.csv` and `tracked.mp4` under each run and well/plane. Use `--focal-plane` to select a different focal plane, `--no-video` to write only CSV, or `--overwrite` to rerun a completed sequence. Use separate output directories for simultaneous runs.

## Review developmental labels

~~~sh
embryo-review --labels /data/labels.csv --image-root /data/centre-crops
~~~

The CSV must contain `seq_id` and `label`. The optional `source` column is used for filtering. The viewer first looks at `center_img_path`, then for a crop named `<seq_id>_center_320.jpg` under the image root, then at `filename`. Relative image paths can be resolved beside the CSV. Saving keeps the previous CSV as a dated backup.

## Train identity embeddings

The crop loader reads paths such as `MTL-0034-run01/run01_P4_FP3_5807.jpg` beneath the image root. Use that layout for the images feeding ReID and repair.

~~~sh
embryo-reid make-patches --tracks-root runs/tracks --big-root /data/time-lapse \
  --out-patch-root runs/patches --out-patch-meta runs/patch_meta.csv
embryo-reid build-stable --tracks-root runs/tracks --out-stable runs/stable.csv
embryo-reid build-train --patch-meta runs/patch_meta.csv --stable-csv runs/stable.csv \
  --out-train runs/reid_train.csv
embryo-reid train --train-csv runs/reid_train.csv --out-dir runs/reid
~~~

The training script selects an available device. Consult `embryo-reid train --help` for batch, sampling and checkpoint options.

## Repair interrupted tracks

~~~sh
embryo-repair --tracks-root runs/tracks --big-root /data/time-lapse \
  --patch-root runs/patches --patch-meta runs/patch_meta.csv \
  --reid-ckpt runs/reid/ckpt/best.pth --device cpu \
  --out-root runs/repair --tmp-patch-root runs/repair-crops
~~~

The repair command saves each run in a subdirectory of the output root. Its main outputs include `tracks_repaired.csv`, `segment_to_embryo_mapping.csv` and `per_well_stats.csv`.

The command names above replace the script paths in the development checkout. Use the thesis tag if you need to run scripts at the original release paths or verify their published hashes.
