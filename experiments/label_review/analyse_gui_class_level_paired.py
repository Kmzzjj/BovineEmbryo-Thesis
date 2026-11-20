#!/usr/bin/env python3
"""Secondary paired class-level analysis for the GUI label experiment."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    return parser.parse_args()


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    observed = abs(float(np.mean(differences)))
    null = [
        abs(float(np.mean(differences * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def paired_bootstrap_ci(
    differences: np.ndarray, reps: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(reps, len(differences)))
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input)

    required = {"seed", "setting", "class", "f1"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    settings = set(data["setting"].astype(str))
    if settings != {"pre_gui", "gui_reviewed"}:
        raise ValueError(f"Unexpected settings: {sorted(settings)}")

    rows: list[dict[str, float | int | str]] = []
    paired_rows: list[dict[str, float | int | str]] = []
    for class_name, class_data in data.groupby("class", sort=True):
        wide = class_data.pivot(index="seed", columns="setting", values="f1").sort_index()
        if list(wide.index) != list(range(10)) or wide.isna().any().any():
            raise ValueError(f"Incomplete seed pairing for {class_name}")

        differences = (wide["gui_reviewed"] - wide["pre_gui"]).to_numpy(float)
        ci_low, ci_high = paired_bootstrap_ci(
            differences, args.bootstrap_reps, args.bootstrap_seed
        )
        sd = float(np.std(differences, ddof=1))
        rows.append(
            {
                "class": class_name,
                "n_pairs": len(differences),
                "pre_gui_f1_mean": float(wide["pre_gui"].mean()),
                "pre_gui_f1_sd": float(wide["pre_gui"].std(ddof=1)),
                "gui_reviewed_f1_mean": float(wide["gui_reviewed"].mean()),
                "gui_reviewed_f1_sd": float(wide["gui_reviewed"].std(ddof=1)),
                "paired_difference_mean": float(np.mean(differences)),
                "paired_bootstrap_ci_low": ci_low,
                "paired_bootstrap_ci_high": ci_high,
                "exact_sign_flip_p_raw": exact_sign_flip_pvalue(differences),
                "paired_dz": float(np.mean(differences) / sd) if sd > 0 else np.nan,
            }
        )
        for seed, pre_value, reviewed_value, difference in zip(
            wide.index,
            wide["pre_gui"],
            wide["gui_reviewed"],
            differences,
            strict=True,
        ):
            paired_rows.append(
                {
                    "class": class_name,
                    "seed": int(seed),
                    "pre_gui_f1": float(pre_value),
                    "gui_reviewed_f1": float(reviewed_value),
                    "paired_difference": float(difference),
                }
            )

    summary = pd.DataFrame(rows)
    summary["holm_p"] = holm_adjust(summary["exact_sign_flip_p_raw"].tolist())
    summary["holm_significant_0_05"] = summary["holm_p"] < 0.05
    summary.to_csv(args.output_dir / "class_level_paired_tests.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(
        args.output_dir / "class_level_paired_seed_values.csv", index=False
    )

    protocol = {
        "analysis_role": "agent_initiated_supplementary_secondary_analysis",
        "not_supervisor_directed": True,
        "input": str(args.input),
        "outcome": "per-class test F1",
        "contrast": "GUI-reviewed minus pre-GUI within seed",
        "pairs": 10,
        "test": "two-sided exact sign-flip test",
        "multiplicity": "Holm correction across the six class-level tests",
        "uncertainty": "paired bootstrap 95% CI of the mean seed-level difference",
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_seed": args.bootstrap_seed,
        "interpretation": "secondary analysis; does not replace the prespecified overall Macro-F1 test",
    }
    (args.output_dir / "analysis_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    lines = [
        "# GUI-review class-level paired analysis",
        "",
        "This was an agent-initiated supplementary secondary analysis, not a supervisor-directed experiment.",
        "The prespecified overall Macro-F1 test remains the primary analysis.",
        "",
        "| Class | Pre-GUI F1 | GUI-reviewed F1 | Difference (95% CI) | Holm p |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['class']} | {row['pre_gui_f1_mean']:.3f} | "
            f"{row['gui_reviewed_f1_mean']:.3f} | {row['paired_difference_mean']:.3f} "
            f"[{row['paired_bootstrap_ci_low']:.3f}, {row['paired_bootstrap_ci_high']:.3f}] | "
            f"{row['holm_p']:.4f} |"
        )
    (args.output_dir / "class_level_evidence.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
