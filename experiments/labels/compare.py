#!/usr/bin/env python3
"""Aggregate the paired GUI-review confirmatory experiment.

The script expects the two training jobs to live under ``gpu0`` and ``gpu1``.
It verifies the paired test sets before calculating seed-level statistics.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


SETTINGS = ("pre_gui", "gui_reviewed")
CLASSES = (
    "2-4 cell",
    "5-16 cell",
    "Blast",
    "Dead/Degenerating/Bad",
    "Morula",
    "Unfertilised",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    return parser.parse_args()


def find_run(base_dir: Path, setting: str, seed: int) -> Path:
    matches = list(base_dir.glob(f"gpu*/{setting}/seed_{seed:02d}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one run for {setting}, seed {seed}; found {matches}")
    run_dir = matches[0]
    required = (run_dir / "run_summary.json", run_dir / "eval/test_predictions.csv")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Incomplete run {run_dir}: missing {missing}")
    return run_dir


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    observed = abs(float(differences.mean()))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        null.append(abs(float(np.mean(differences * np.asarray(signs)))))
    null = np.asarray(null)
    return float(np.mean(null >= observed - 1e-15))


def paired_bootstrap_ci(
    differences: np.ndarray, reps: int, seed: int, confidence: float = 0.95
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(differences)
    draws = rng.choice(differences, size=(reps, n), replace=True).mean(axis=1)
    alpha = 1.0 - confidence
    return tuple(float(x) for x in np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0]))


def save_confusion_figure(cm_by_setting: dict[str, np.ndarray], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)
    for ax, setting in zip(axes, SETTINGS):
        cm = cm_by_setting[setting].astype(float)
        denom = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, denom, out=np.zeros_like(cm), where=denom != 0)
        image = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_title("Pre-GUI training labels" if setting == "pre_gui" else "GUI-reviewed training labels")
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=38, ha="right")
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True GUI-reviewed class")
        for row in range(len(CLASSES)):
            for col in range(len(CLASSES)):
                color = "white" if norm[row, col] > 0.5 else "black"
                ax.text(col, row, f"{norm[row, col]:.2f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(image, ax=axes, shrink=0.82, label="Row-normalized proportion")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_paired_figure(seed_metrics: pd.DataFrame, out_path: Path) -> None:
    pivot = seed_metrics.pivot(index="seed", columns="setting", values="test_macro_f1")
    fig, ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
    for seed, row in pivot.iterrows():
        ax.plot([0, 1], [row["pre_gui"], row["gui_reviewed"]], marker="o", color="#555555", alpha=0.75)
        ax.text(1.03, row["gui_reviewed"], str(seed), va="center", fontsize=8)
    ax.set_xticks([0, 1], ["Pre-GUI labels", "GUI-reviewed labels"])
    ax.set_ylabel("Test Macro-F1")
    ax.set_xlim(-0.15, 1.18)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_analysis_report(
    out_path: Path,
    summary: pd.DataFrame,
    tests: dict,
    class_summary: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> None:
    summary_by_setting = summary.set_index("setting")
    lines = [
        "# GUI Review Confirmatory Experiment",
        "",
        "## Pairing audit",
        "",
        f"All {len(audit_df)} seed pairs used identical test sequence IDs: "
        f"{bool(audit_df['seq_ids_identical'].all())}.",
        f"All paired predictions used identical GUI-reviewed reference labels: "
        f"{bool(audit_df['truth_identical'].all())}.",
        "",
        "## Setting summary",
        "",
        "| Training labels | Seeds | Test accuracy, mean (SD) | Test Macro-F1, mean (SD) |",
        "|---|---:|---:|---:|",
    ]
    for setting, label in (("pre_gui", "Pre-GUI"), ("gui_reviewed", "GUI-reviewed")):
        row = summary_by_setting.loc[setting]
        lines.append(
            f"| {label} | {int(row['seeds'])} | "
            f"{row['test_accuracy_mean']:.4f} ({row['test_accuracy_sd']:.4f}) | "
            f"{row['test_macro_f1_mean']:.4f} ({row['test_macro_f1_sd']:.4f}) |"
        )

    lines.extend(["", "## Paired inference", ""])
    for metric, label in (("test_macro_f1", "Test Macro-F1"), ("test_accuracy", "Test accuracy")):
        result = tests[metric]
        ci_low, ci_high = result["paired_bootstrap_95_ci"]
        lines.extend(
            [
                f"### {label}",
                "",
                f"- Mean paired difference (GUI-reviewed minus pre-GUI): "
                f"{result['mean_difference_gui_minus_pre']:.6f}",
                f"- Paired bootstrap 95% CI: [{ci_low:.6f}, {ci_high:.6f}]",
                f"- Exact two-sided sign-flip p-value: {result['exact_two_sided_sign_flip_p']:.6f}",
                f"- Paired effect size dz: {result['paired_effect_size_dz']:.6f}",
                f"- Paired t-test p-value (sensitivity only): {result['paired_t_p_sensitivity']:.6f}",
                "",
            ]
        )

    f1_table = class_summary.pivot(index="class", columns="setting", values="f1_mean")
    lines.extend(
        [
            "## Class-level test F1",
            "",
            "| Class | Pre-GUI mean F1 | GUI-reviewed mean F1 | Paired setting difference |",
            "|---|---:|---:|---:|",
        ]
    )
    for class_name in CLASSES:
        pre = float(f1_table.loc[class_name, "pre_gui"])
        gui = float(f1_table.loc[class_name, "gui_reviewed"])
        lines.append(f"| {class_name} | {pre:.4f} | {gui:.4f} | {gui - pre:+.4f} |")

    lines.extend(
        [
            "",
            "## Interpretation gate",
            "",
            "The thesis interpretation must follow the primary paired Macro-F1 result, its confidence interval, "
            "and the exact sign-flip test. The accuracy test and paired t-test are secondary sensitivity analyses.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    out_dir = base_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    class_rows = []
    cm_by_setting = {setting: np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64) for setting in SETTINGS}
    prediction_audit = []

    for seed in range(10):
        predictions = {}
        for setting in SETTINGS:
            run_dir = find_run(base_dir, setting, seed)
            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            pred = pd.read_csv(run_dir / "eval/test_predictions.csv")
            pred = pred.sort_values("seq_id").reset_index(drop=True)
            predictions[setting] = pred

            y_true = pred["true"].astype(str)
            y_pred = pred["pred"].astype(str)
            acc = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, labels=list(CLASSES), average="macro", zero_division=0)
            seed_rows.append(
                {
                    "seed": seed,
                    "setting": setting,
                    "best_epoch": int(summary["best_epoch"]),
                    "best_val_accuracy": float(summary["best_val_acc"]),
                    "best_val_macro_f1": float(summary["best_val_f1"]),
                    "test_accuracy": float(acc),
                    "test_macro_f1": float(macro_f1),
                    "n_test": len(pred),
                }
            )

            precision, recall, class_f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=list(CLASSES), zero_division=0
            )
            for class_name, p, r, f1_value, n in zip(CLASSES, precision, recall, class_f1, support):
                class_rows.append(
                    {
                        "seed": seed,
                        "setting": setting,
                        "class": class_name,
                        "precision": float(p),
                        "recall": float(r),
                        "f1": float(f1_value),
                        "support": int(n),
                    }
                )
            cm_by_setting[setting] += confusion_matrix(y_true, y_pred, labels=list(CLASSES))

        left = predictions["pre_gui"]
        right = predictions["gui_reviewed"]
        ids_match = left["seq_id"].equals(right["seq_id"])
        truth_match = left["true"].equals(right["true"])
        prediction_audit.append(
            {"seed": seed, "n_test": len(left), "seq_ids_identical": ids_match, "truth_identical": truth_match}
        )
        if not ids_match or not truth_match:
            raise RuntimeError(f"Pairing audit failed for seed {seed}")

    seed_metrics = pd.DataFrame(seed_rows).sort_values(["seed", "setting"])
    class_metrics = pd.DataFrame(class_rows).sort_values(["class", "seed", "setting"])
    audit_df = pd.DataFrame(prediction_audit)
    seed_metrics.to_csv(out_dir / "seed_metrics.csv", index=False)
    class_metrics.to_csv(out_dir / "class_metrics_by_seed.csv", index=False)
    audit_df.to_csv(out_dir / "paired_prediction_audit.csv", index=False)

    summary = (
        seed_metrics.groupby("setting")
        .agg(
            seeds=("seed", "count"),
            best_val_accuracy_mean=("best_val_accuracy", "mean"),
            best_val_accuracy_sd=("best_val_accuracy", "std"),
            best_val_macro_f1_mean=("best_val_macro_f1", "mean"),
            best_val_macro_f1_sd=("best_val_macro_f1", "std"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_sd=("test_accuracy", "std"),
            test_macro_f1_mean=("test_macro_f1", "mean"),
            test_macro_f1_sd=("test_macro_f1", "std"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "setting_summary.csv", index=False)

    class_summary = (
        class_metrics.groupby(["setting", "class"])
        .agg(
            precision_mean=("precision", "mean"),
            precision_sd=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
            support=("support", "first"),
        )
        .reset_index()
    )
    class_summary.to_csv(out_dir / "class_metrics_summary.csv", index=False)

    pivot = seed_metrics.pivot(index="seed", columns="setting")
    tests = {}
    for metric in ("test_accuracy", "test_macro_f1"):
        differences = (pivot[metric]["gui_reviewed"] - pivot[metric]["pre_gui"]).to_numpy(dtype=float)
        ci_low, ci_high = paired_bootstrap_ci(differences, args.bootstrap_reps, args.bootstrap_seed)
        sd = float(differences.std(ddof=1))
        tests[metric] = {
            "n_pairs": len(differences),
            "mean_difference_gui_minus_pre": float(differences.mean()),
            "sd_difference": sd,
            "paired_effect_size_dz": float(differences.mean() / sd) if sd > 0 else math.inf,
            "paired_bootstrap_95_ci": [ci_low, ci_high],
            "exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(differences),
            "paired_t_p_sensitivity": float(stats.ttest_rel(pivot[metric]["gui_reviewed"], pivot[metric]["pre_gui"]).pvalue),
            "per_seed_differences": [float(x) for x in differences],
        }
    (out_dir / "paired_statistical_tests.json").write_text(json.dumps(tests, indent=2), encoding="utf-8")

    for setting, cm in cm_by_setting.items():
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out_dir / f"confusion_{setting}_counts.csv")
    save_confusion_figure(cm_by_setting, out_dir / "confusion_comparison_normalized.png")
    save_paired_figure(seed_metrics, out_dir / "paired_test_macro_f1.png")
    write_analysis_report(out_dir / "analysis_report.md", summary, tests, class_summary, audit_df)

    print(summary.to_string(index=False))
    print(json.dumps(tests, indent=2))
    print(f"Saved analysis to {out_dir}")


if __name__ == "__main__":
    main()
