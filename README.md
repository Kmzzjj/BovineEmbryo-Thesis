# Bovine embryo time-lapse imaging

Research code for *From Stage Classification to Embryo-Level Continuity in Bovine Time-Lapse Imaging* by Zeyu Liu. The workflow detects embryos in well images, links detections over time, and uses appearance embeddings to reconnect interrupted tracks. The label review tool lets you check developmental stages against the centre crops from image sequences.

## Where to start

| Task | Entry point |
| --- | --- |
| Train the YOLOv8 detector | [train_yolov8.py](BovineEmbryoDetection/src/train/train_yolov8.py) |
| Track detections with BoT-SORT | [track_BoT-SORT.py](BovineEmbryoDetection/src/tracking/track_BoT-SORT.py) |
| Train embryo identity embeddings | [train_reid_native_stable_v3_full.py](notebook_script/train_reid_native_stable_v3_full.py) |
| Reconnect interrupted tracks | [repair_pipeline53_guarded_orphanfix_release.py](notebook_script/repair_pipeline53_guarded_orphanfix_release.py) |
| Review developmental stage labels | [Embryo label grid](notebook_script/python%20embryo_label_grid_gui.py) |
| Run the sequence model comparisons | [Model comparison](notebook_script/run_confirmatory_model_compare_20260810.py) · [T=3 and T=9 windows](notebook_script/run_paired_window_t3_t9_repro_20260829.py) |

The [window direction study](direction_t3_20260907/) and [track repair supplement](repair_supplement_20260907/) contain the original analysis scripts. Their orchestration and verification scripts expect the corresponding private inputs and frozen source hashes. [Example configuration](configs/stage_cls.example.yaml) shows the form of a local sequence-classification config.

## Track detections

The tracker needs Python with OpenCV, NumPy, PyTorch and Ultralytics, plus a trained detector. Put each run in its own directory under the frame root. The script groups filenames such as `run01_P4_FP3_5807.jpg` by well and focal plane, then orders each group by capture time.

    python BovineEmbryoDetection/src/tracking/track_BoT-SORT.py \
      --frames /path/to/runs --weights /path/to/best.pt --output /path/to/tracks

The default focal plane is FP3. Each sequence gets `tracks.csv` and `tracked.mp4` under its run and well/plane folders. Use `--no-video` for CSV only. Completed sequences are skipped unless you pass `--overwrite`. Use separate output directories for concurrent runs.

## Review stage labels

The viewer uses pandas and PyQt5. Its input CSV needs `seq_id` and `label`. It also reads `source` for filtering when present.

    python "notebook_script/python embryo_label_grid_gui.py" \
      --labels /path/to/labels.csv --image-root /path/to/centre-crops

The viewer looks for a `center_img_path` from the CSV, then a crop named `<seq_id>_center_320.jpg` under the image root, then the CSV's `filename` field. Relative image paths can be resolved from the CSV directory. The image root defaults to that directory. Saving keeps a dated copy of the previous CSV beside it.

## Data and thesis release

Research images and videos, annotations, sample identifiers, split manifests, per-sample predictions and trained weights are excluded from this repository. Training and analysis require authorised inputs. An exact historical Python environment lock is unavailable.

Use the [thesis tag](https://github.com/Kmzzjj/BovineEmbryo-Thesis/tree/thesis) for the code associated with the submitted thesis. It was derived from the private thesis snapshot at commit `43e24943b4bf659b6f2904f61304599805baec87`. The [source manifest at that tag](https://github.com/Kmzzjj/BovineEmbryo-Thesis/blob/thesis/public_source_manifest.json) records the private and public file hashes for the release.

No additional licence grant is asserted. Third-party libraries retain their own licences.
