# Bovine embryo thesis code

Selected implementation associated with *From Stage Classification to Embryo-Level Continuity in Bovine Time-Lapse Imaging* by Zeyu Liu.

## Contents

Detection and tracking implementation, ReID training, sequence classification and comparison scripts, tracking repair, and related analysis utilities are retained in their original relative folders. `configs/stage_cls.example.yaml` is a configuration example. The `thesis` tag fixes the public code release.

## Data and execution

The research images are not a public dataset. This repository excludes images and videos, annotations, sample identifiers, split manifests, per-sample predictions, model weights, caches, internal audit records and private correspondence. Their inclusion in the thesis does not grant redistribution permission for the dataset.

This is a code release, not a complete end-to-end reproducibility package. Scripts may require authorised data, historical manifests, trained weights and local path configuration that are not supplied here. Configure input and output locations before running anything. Some historical orchestration utilities refer to private artefacts and frozen source hashes; those checks require the original private snapshot and do not validate this public copy.

Python dependencies vary by entry point, including PyTorch, torchvision, pandas, NumPy, scikit-learn, OpenCV, Pillow, Ultralytics and optional Neptune. An exact historical environment lock has not been recovered. No training run is claimed to have been repeated for this release.

## Release provenance

The release was derived from the private thesis snapshot at commit `43e24943b4bf659b6f2904f61304599805baec87`. Personal server paths and a Neptune workspace default were replaced with placeholders. Analysis methods and historical results were not changed; results themselves are not published in this release. `public_source_manifest.json` records source and public file hashes so these copies can be distinguished. This release has its own commit and is not byte-identical to the private snapshot.

No additional licence grant is asserted. Third-party libraries retain their own licences.
