# Bovine embryo time-lapse imaging

Research code for *From Stage Classification to Embryo-Level Continuity in Bovine Time-Lapse Imaging* by Zeyu Liu. The workflow detects embryos in well images, tracks them over time, and uses appearance embeddings to reconnect interrupted tracks. A separate viewer supports manual review of developmental stage labels from static centre crops.

## Install

Python 3.11 is recommended for the full dependency stack. Create a separate environment and install the components you need from the repository root.

~~~sh
python -m pip install -e '.[tracking,review,identity]'
~~~

The tracking extra covers the YOLO detector and tracker. The review extra adds the PyQt viewer. The identity extra adds the dependencies for ReID and track repair. ReID training and its batch benchmark require an NVIDIA CUDA GPU; crop preparation and repair can run on the CPU.

## Start here

| Command | Inputs and outputs |
| --- | --- |
| `embryo-train-detector` | YOLO dataset YAML and model weights → a detector training run |
| `embryo-track` | Time-lapse frames and detector weights → one track CSV and optional video per sequence |
| `embryo-review` | Label CSV and centre crops → reviewed label CSV and dated backup |
| `embryo-reid` | Track CSVs and image crops → training manifest and identity checkpoints |
| `embryo-repair` | Tracks, images and an identity checkpoint → repaired tracks and per-well summaries |

Run any command with `--help` for its options. See the [workflow guide](docs/workflow.md) for commands, file conventions and example configurations.

## Repository layout

| Directory | Contents |
| --- | --- |
| [src/embryo_imaging](src/embryo_imaging/) | Installable detector, tracker, label viewer, ReID and repair code |
| [configs](configs/) | Example configuration for detector training |
| [experiments](experiments/) | Stage classification, label review analysis, window direction and repair studies |
| [tests](tests/) | Small tests using synthetic frames and temporary CSV files |
| [provenance](provenance/) | File hashes for the original thesis release |

Run the local checks with `python -m pip install -e '.[test]'` followed by `python -m unittest discover -s tests -v`.

## Research data and thesis release

Research images, videos, annotations, sample identifiers, split manifests, predictions and trained weights are excluded. The experiment scripts require authorised inputs and, in some cases, the original run artefacts. See the [experiment index](experiments/README.md) for details.

This remains a selected code release, rather than a complete end-to-end reproduction package. The historical environment and private run artefacts are not supplied.

The [link in the thesis](https://github.com/Kmzzjj/BovineEmbryo-Thesis/tree/thesis) opens the maintained project files. The [original release](https://github.com/Kmzzjj/BovineEmbryo-Thesis/tree/thesis-original) is preserved separately at commit `3a5c4e4`. The [release manifest](provenance/manifest.json) records the original source and public file hashes. No additional licence grant is asserted. Third-party libraries retain their own licences.
