# Experiments

| Folder | Scripts |
| --- | --- |
| [classification](classification/) | `train_models.py`, `train_labels.py`, `train_windows.py`, `compare_models.py`, `compare_windows.py` |
| [labels](labels/) | `compare.py` for the paired label-review analysis; `classes.py` for per-class results |
| [direction](direction/) | `sampling.py`, `train.py`, `prepare.py`, `check_inputs.py`, `launch.py`, `finish.py`, `analyse.py` |
| [repair](repair/) | `sensitivity.py`, `panels.py`, `failures.py`, `membership.py` |

The analysis scripts use the `analysis` extra. Training also needs the `identity` and `tracking` extras. Some historical runs use optional Neptune logging. Sampling tests are in `tests/test_sampling.py`.

These are historical studies. Private data paths, result names, source hashes and follow-up-analysis records are retained because they identify the original work. Scripts that enforce frozen hashes need the original private source and run artefacts; the renamed files are not a replacement for that snapshot. Use the fixed [thesis release](https://github.com/Kmzzjj/BovineEmbryo-Thesis/tree/thesis) when checking original paths and public hashes. `classification/example.yaml` is the archived configuration example.
