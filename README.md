# Bovine embryo time-lapse imaging

Research code for *From Stage Classification to Embryo-Level Continuity in Bovine Time-Lapse Imaging* by Zeyu Liu. The workflow detects embryos in well images, tracks them over time, and uses appearance embeddings to reconnect interrupted tracks. A separate viewer supports manual review of developmental stage labels.

## Install

Create a Python 3.9 or later environment and install the components you need from the repository root.

~~~sh
python -m pip install -e '.[tracking,review,identity]'
~~~

The tracking extra covers the YOLO detector and tracker. The review extra adds the PyQt viewer. The identity extra adds the dependencies for ReID and track repair.

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

The submitted thesis cites the fixed [thesis release](https://github.com/Kmzzjj/BovineEmbryo-Thesis/tree/thesis). The [release manifest](provenance/thesis-source-manifest.json) records the original source and public file hashes at that tag. No additional licence grant is asserted. Third-party libraries retain their own licences.
