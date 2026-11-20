#!/usr/bin/env python3
"""Audit and analyse the matched three-model confirmatory experiment."""

from __future__ import annotations

import itertools
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path("/path/to/project")
OUT = ROOT / "exp/confirmatory_model_compare_20260810/analysis"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_SOURCES = {
    "Transformer": [ROOT / "exp/confirmatory_model_compare_20260810/transformer"],
    "TCN": [ROOT / "exp/confirmatory_model_compare_20260810/tcn"],
    "R(2+1)D": [ROOT / "exp/confirmatory_model_compare_20260810/r2"],
}

MODEL_ORDER = ["Transformer", "TCN", "R(2+1)D"]
EXPECTED_INTERNAL_MODELS = {
    "Transformer": "r50_transformer",
    "TCN": "resnet18_tcn",
    "R(2+1)D": "r2plus1d18",
}
EXPECTED_SEEDS = list(range(10))
CLASSES = [
    "2-4 cell",
    "5-16 cell",
    "Blast",
    "Dead/Degenerating/Bad",
    "Morula",
    "Unfertilised",
]
METRICS = {
    "Test accuracy": "test_acc_at_best",
    "Test Macro-F1": "test_f1_at_best",
}


def read_runs(model: str, paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            summaries = sorted(path.glob("gui_reviewed/seed_*/run_summary.json"))
            if not summaries:
                raise FileNotFoundError(f"No run summaries under {path}")
            frames.append(pd.DataFrame([json.loads(p.read_text(encoding="utf-8")) for p in summaries]))
        else:
            frames.append(pd.read_csv(path, encoding="utf-8-sig"))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["setting"].eq("gui_reviewed")].copy()
    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    df = df[df["seed"].isin(EXPECTED_SEEDS)].sort_values("seed")
    if df["seed"].tolist() != EXPECTED_SEEDS:
        raise ValueError(f"{model}: expected seeds {EXPECTED_SEEDS}, found {df['seed'].tolist()}")
    if df["seed"].duplicated().any():
        raise ValueError(f"{model}: duplicate seed rows")
    if not df["model"].eq(EXPECTED_INTERNAL_MODELS[model]).all():
        raise ValueError(f"{model}: unexpected internal model identifier")
    if not df["n_classes"].eq(len(CLASSES)).all():
        raise ValueError(f"{model}: expected {len(CLASSES)} classes")
    expected_classes = " | ".join(CLASSES)
    if not df["classes"].eq(expected_classes).all():
        raise ValueError(f"{model}: class order differs from the planned analysis")
    df["model"] = model
    return df


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_sign_flip_p(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    observed = abs(float(diff.mean()))
    signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=len(diff))))
    permuted = np.abs((signs * diff).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-15))


def paired_bootstrap_ci(diff: np.ndarray, seed: int, n_boot: int = 100_000) -> tuple[float, float]:
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.empty(len(p), dtype=float)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p[idx])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(len(p), dtype=float)
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted.tolist()


def paired_effect_size(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    sd = float(diff.std(ddof=1))
    return float(diff.mean() / sd) if sd > 0 else float("nan")


split_paths = {model: paths[0] / "fixed_split_seq_id.csv" for model, paths in MODEL_SOURCES.items()}
for model, split_path in split_paths.items():
    if not split_path.exists():
        raise FileNotFoundError(f"{model}: missing fixed split file {split_path}")
split_hashes = {model: file_sha256(path) for model, path in split_paths.items()}
if len(set(split_hashes.values())) != 1:
    raise ValueError(f"Fixed split hashes differ across models: {split_hashes}")

runs = {model: read_runs(model, paths) for model, paths in MODEL_SOURCES.items()}
long_runs = pd.concat([runs[m] for m in MODEL_ORDER], ignore_index=True)
long_runs.to_csv(OUT / "model_runs_long.csv", index=False, encoding="utf-8-sig")

summary_rows = []
for model in MODEL_ORDER:
    for metric_name, metric_col in METRICS.items():
        values = runs[model][metric_col].to_numpy(float)
        summary_rows.append(
            {
                "model": model,
                "metric": metric_name,
                "n_seeds": len(values),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "model_metric_summary.csv", index=False, encoding="utf-8-sig")


def save_paired_run_figure() -> None:
    """Show seed-level variation without implying independent test cohorts."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    x = np.arange(len(MODEL_ORDER))
    marker_map = {"Transformer": "o", "TCN": "s", "R(2+1)D": "^"}
    for ax, (metric_name, metric_col) in zip(axes, METRICS.items()):
        matrix = np.column_stack([runs[model][metric_col].to_numpy(float) for model in MODEL_ORDER])
        for row in matrix:
            ax.plot(x, row, color="0.78", linewidth=0.7, alpha=0.75, zorder=1)
        for model_index, model in enumerate(MODEL_ORDER):
            values = matrix[:, model_index]
            ax.scatter(
                np.full(len(values), model_index),
                values,
                marker=marker_map[model],
                s=30,
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
                zorder=2,
            )
            ax.errorbar(
                model_index,
                values.mean(),
                yerr=values.std(ddof=1),
                fmt="D",
                color="black",
                capsize=3,
                markersize=4.5,
                linewidth=1.0,
                zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER)
        ax.set_ylabel(metric_name)
        ax.grid(axis="y", color="0.9", linewidth=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.savefig(OUT / "model_comparison_paired_runs.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "model_comparison_paired_runs.pdf", bbox_inches="tight")
    plt.close(fig)


save_paired_run_figure()

omnibus_rows = []
pairwise_rows = []
pairs = [("Transformer", "TCN"), ("Transformer", "R(2+1)D"), ("TCN", "R(2+1)D")]
for metric_index, (metric_name, metric_col) in enumerate(METRICS.items()):
    arrays = [runs[m][metric_col].to_numpy(float) for m in MODEL_ORDER]
    stat, p_value = friedmanchisquare(*arrays)
    kendalls_w = float(stat / (len(EXPECTED_SEEDS) * (len(MODEL_ORDER) - 1)))
    omnibus_rows.append(
        {
            "metric": metric_name,
            "test": "Friedman test across three paired model runs",
            "n_paired_seeds": len(EXPECTED_SEEDS),
            "statistic": float(stat),
            "p_value": float(p_value),
            "kendalls_w": kendalls_w,
        }
    )
    metric_pair_rows = []
    for pair_index, (model_a, model_b) in enumerate(pairs):
        a = runs[model_a][metric_col].to_numpy(float)
        b = runs[model_b][metric_col].to_numpy(float)
        diff = a - b
        lo, hi = paired_bootstrap_ci(diff, seed=20260810 + 100 * metric_index + pair_index)
        metric_pair_rows.append(
            {
                "metric": metric_name,
                "model_a": model_a,
                "model_b": model_b,
                "direction": "model_a - model_b",
                "n_paired_seeds": len(diff),
                "mean_a": float(a.mean()),
                "mean_b": float(b.mean()),
                "mean_difference": float(diff.mean()),
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "exact_sign_flip_p": exact_sign_flip_p(diff),
                "paired_dz": paired_effect_size(diff),
            }
        )
    adjusted = holm_adjust([row["exact_sign_flip_p"] for row in metric_pair_rows])
    for row, p_adj in zip(metric_pair_rows, adjusted):
        row["holm_adjusted_p_within_metric"] = p_adj
    pairwise_rows.extend(metric_pair_rows)

omnibus = pd.DataFrame(omnibus_rows)
pairwise = pd.DataFrame(pairwise_rows)
omnibus.to_csv(OUT / "model_omnibus_tests.csv", index=False, encoding="utf-8-sig")
pairwise.to_csv(OUT / "model_pairwise_tests.csv", index=False, encoding="utf-8-sig")

prediction_audit_rows = []
class_rows = []
reference = None
for model in MODEL_ORDER:
    for _, run in runs[model].iterrows():
        seed = int(run["seed"])
        pred_path = Path(run["exp_dir"]) / "eval/test_predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        pred = pd.read_csv(pred_path, encoding="utf-8-sig")
        required = {"seq_id", "true", "pred"}
        if not required.issubset(pred.columns):
            raise ValueError(f"{pred_path}: missing {sorted(required - set(pred.columns))}")
        pred["seq_id"] = pred["seq_id"].astype(str)
        pred = pred.sort_values("seq_id").reset_index(drop=True)
        if len(pred) != 331 or pred["seq_id"].duplicated().any():
            raise ValueError(f"{pred_path}: expected 331 unique test sequences, found {len(pred)}")
        truth = pred[["seq_id", "true"]].copy()
        if reference is None:
            reference = truth
        elif not truth.equals(reference):
            raise ValueError(f"{pred_path}: test sequence IDs or reference labels differ")

        acc = accuracy_score(pred["true"], pred["pred"])
        macro_f1 = f1_score(pred["true"], pred["pred"], labels=CLASSES, average="macro", zero_division=0)
        prediction_audit_rows.append(
            {
                "model": model,
                "seed": seed,
                "n_test": len(pred),
                "accuracy_recomputed": float(acc),
                "macro_f1_recomputed": float(macro_f1),
                "accuracy_matches_summary": bool(np.isclose(acc, run["test_acc_at_best"], atol=1e-12)),
                "macro_f1_matches_summary": bool(np.isclose(macro_f1, run["test_f1_at_best"], atol=1e-12)),
                "prediction_path": str(pred_path),
            }
        )
        per_class = f1_score(pred["true"], pred["pred"], labels=CLASSES, average=None, zero_division=0)
        for class_name, class_f1 in zip(CLASSES, per_class):
            class_rows.append({"model": model, "seed": seed, "class": class_name, "f1": float(class_f1)})

prediction_audit = pd.DataFrame(prediction_audit_rows)
if not prediction_audit[["accuracy_matches_summary", "macro_f1_matches_summary"]].all().all():
    raise ValueError("Recomputed prediction metrics do not match one or more run summaries")
prediction_audit.to_csv(OUT / "prediction_audit.csv", index=False, encoding="utf-8-sig")

class_long = pd.DataFrame(class_rows)
class_long.to_csv(OUT / "class_f1_runs.csv", index=False, encoding="utf-8-sig")
class_summary = (
    class_long.groupby(["model", "class"], sort=False)["f1"]
    .agg(n_seeds="count", mean="mean", sd="std", min="min", max="max")
    .reset_index()
)
class_summary.to_csv(OUT / "class_f1_summary.csv", index=False, encoding="utf-8-sig")

class_test_rows = []
for class_index, class_name in enumerate(CLASSES):
    pivot = class_long[class_long["class"].eq(class_name)].pivot(index="seed", columns="model", values="f1")
    pivot = pivot.loc[EXPECTED_SEEDS, MODEL_ORDER]
    for pair_index, (model_a, model_b) in enumerate([("Transformer", "TCN"), ("Transformer", "R(2+1)D")]):
        diff = pivot[model_a].to_numpy(float) - pivot[model_b].to_numpy(float)
        lo, hi = paired_bootstrap_ci(diff, seed=20260900 + 10 * class_index + pair_index)
        class_test_rows.append(
            {
                "class": class_name,
                "model_a": model_a,
                "model_b": model_b,
                "direction": "model_a - model_b",
                "mean_difference": float(diff.mean()),
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "exact_sign_flip_p": exact_sign_flip_p(diff),
                "paired_dz": paired_effect_size(diff),
            }
        )
class_tests = pd.DataFrame(class_test_rows)
class_tests["holm_adjusted_p_across_12_exploratory_tests"] = holm_adjust(class_tests["exact_sign_flip_p"].tolist())
class_tests.to_csv(OUT / "class_f1_exploratory_pairwise_tests.csv", index=False, encoding="utf-8-sig")

evidence = [
    "# Confirmatory matched model comparison",
    "",
    "## Design audit",
    "",
    "- Models: ResNet-50 + Temporal Transformer, ResNet-18 + TCN, and R(2+1)D-18.",
    "- Data: the same 3,260 matched, GUI-reviewed and non-uncertain six-class sequences.",
    "- Split: the same group-disjoint train/validation/test assignment for every model and seed.",
    "- Input: the same 21-frame sequence definition and 224 x 224 crop size.",
    "- Repeats: ten paired random seeds (0-9).",
    f"- Split-file SHA-256: {next(iter(split_hashes.values()))}.",
    "- Primary outcome: test Macro-F1; secondary outcome: test accuracy.",
    "- Inference: Friedman omnibus test followed by exact paired sign-flip tests; Holm correction is applied within each metric.",
    "- Kendall's W is reported as the Friedman omnibus effect size.",
    "- The paired bootstrap interval describes the mean seed-level difference.",
    "- Seed-level inference measures training-run variability and does not replace an embryo-cohort-level external validation study.",
    "- This is a comparison of the complete modelling configurations used in the thesis, not a pure temporal-operator ablation: the two frame-based models use ImageNet-pretrained image backbones, whereas R(2+1)D uses Kinetics-400 pretraining.",
    "",
    "## Model summaries",
    "",
    summary.to_markdown(index=False, floatfmt=".6f"),
    "",
    "## Omnibus tests",
    "",
    omnibus.to_markdown(index=False, floatfmt=".6f"),
    "",
    "## Pairwise tests",
    "",
    pairwise.to_markdown(index=False, floatfmt=".6f"),
    "",
    "## Output audit",
    "",
    f"All {len(prediction_audit)} model-seed prediction files contained the same 331 test sequences and reference labels.",
    "All accuracy and Macro-F1 values recomputed from prediction files matched the run summaries.",
    "Class-level tests are retained as secondary exploratory analyses and corrected together across 12 comparisons.",
]
(OUT / "model_comparison_evidence.md").write_text("\n".join(evidence) + "\n", encoding="utf-8")

metadata = {
    "expected_seeds": EXPECTED_SEEDS,
    "models": MODEL_ORDER,
    "classes": CLASSES,
    "n_test_sequences": 331,
    "bootstrap_replicates": 100_000,
    "primary_metric": "Test Macro-F1",
    "secondary_metric": "Test accuracy",
    "fixed_split_sha256": next(iter(split_hashes.values())),
    "comparison_scope": "Complete modelling configurations, not a pure temporal-operator ablation",
}
(OUT / "analysis_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print(OUT / "model_comparison_evidence.md")
