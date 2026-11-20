# Thesis experiments

These directories contain the analysis and comparison scripts from the public thesis release.

| Directory | Contents |
| --- | --- |
| [stage_classification](stage_classification/) | Model and label comparisons, paired temporal-window runs, and window analysis |
| [label_review](label_review/) | Confirmatory and class-level label review analyses |
| [direction_t3_20260907](direction_t3_20260907/) | Direction comparison, launcher, parity checks and study-specific test |
| [repair_supplement_20260907](repair_supplement_20260907/) | Cached repair sensitivity runs, failure diagnostics and membership checks |

The label and classification analysis tools use the analysis extra. The training studies also use the identity and tracking dependencies. Some runs have optional Neptune logging.

The orchestration scripts refer to the historical workspace layout, private inputs and expected hashes. To use those checks, start from the [thesis tag](https://github.com/Kmzzjj/BovineEmbryo-Thesis/tree/thesis) and provide the corresponding authorised inputs and run artefacts. The YAML in stage_classification is the archived configuration example.
