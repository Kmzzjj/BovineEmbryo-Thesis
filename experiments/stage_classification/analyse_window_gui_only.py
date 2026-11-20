#!/usr/bin/env python3
"""Aggregate the clean GUI-only temporal-window experiment."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    observed = abs(float(differences.mean()))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        values.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(values) >= observed - 1e-15))


def paired_bootstrap_ci(differences: np.ndarray, seed: int, n_resamples: int = 100000) -> list[float]:
    differences = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=float)
    batch = 5000
    for start in range(0, n_resamples, batch):
        size = min(batch, n_resamples - start)
        idx = rng.integers(0, len(differences), size=(size, len(differences)))
        estimates[start : start + size] = differences[idx].mean(axis=1)
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def read_all_runs(root: Path) -> pd.DataFrame:
    frames = []
    files = sorted(root.glob("ablation_*/ablation_runs_shard_*_of_*.csv"))
    frames.extend(pd.read_csv(path) for path in files if path.stat().st_size > 0)

    summary_files = sorted(root.glob("ablation_*/runs/*/eval/run_summary.json"))
    if summary_files:
        frames.append(pd.DataFrame([json.loads(path.read_text()) for path in summary_files]))

    known = set()
    if frames:
        existing = pd.concat(frames, ignore_index=True)
        known = set(zip(existing["T"].astype(int), existing["run_seed"].astype(int)))
    else:
        existing = pd.DataFrame()

    
    # those rows from immutable predictions and the selected best checkpoint.
    rebuilt = []
    for pred_path in sorted(root.glob("ablation_agent_window_gui_only_r50_cbam_tfm/runs/*/eval/test_predictions.csv")):
        match = re.search(r"_T(\d+)__seed(\d+)$", pred_path.parents[1].name)
        if not match:
            continue
        window, seed = map(int, match.groups())
        if (window, seed) in known:
            continue
        run_dir = pred_path.parents[1]
        pred = pd.read_csv(pred_path)
        train_path = run_dir / "eval/train_predictions.csv"
        train = pd.read_csv(train_path) if train_path.exists() else None

        import torch

        checkpoint = torch.load(run_dir / "ckpt/best.pth", map_location="cpu", weights_only=False)
        rebuilt.append(
            {
                "T": window,
                "run_seed": seed,
                "split_seed": 42,
                "best_epoch": int(checkpoint["epoch"]),
                "best_val_f1": float(checkpoint["best_score"]),
                "best_val_acc": float(checkpoint["best_val_acc"]),
                "train_acc@best": accuracy_score(train["true"], train["pred"]) if train is not None else np.nan,
                "train_f1@best": f1_score(train["true"], train["pred"], average="macro", zero_division=0) if train is not None else np.nan,
                "test_acc@best": accuracy_score(pred["true"], pred["pred"]),
                "test_f1@best": f1_score(pred["true"], pred["pred"], average="macro", zero_division=0),
                "run_seconds": np.nan,
                "test_eval_seconds": np.nan,
                "test_sequences": len(pred),
                "test_seconds_per_sequence": np.nan,
                "peak_vram_gib": np.nan,
                "exp_dir": str(run_dir),
                "shard_num_jobs": 2,
                "shard_job_index": seed % 2,
                "summary_rebuilt_from_predictions": 1,
            }
        )
    if rebuilt:
        frames.append(pd.DataFrame(rebuilt))
    if not frames:
        raise RuntimeError(f"No completed run outputs found under {root}")

    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(["T", "run_seed", "split_seed"], keep="last")
    return data.sort_values(["T", "run_seed"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    runs = read_all_runs(args.root)
    expected = {1: 3, 2: 3, 3: 10, 4: 3, 5: 3, 7: 3, 9: 10, 11: 3, 15: 3, 21: 3}
    observed = runs.groupby("T")["run_seed"].nunique().astype(int).to_dict()
    if observed != expected:
        raise RuntimeError(f"Incomplete run grid: expected {expected}, observed {observed}")

    aggregate = (
        runs.groupby("T")
        .agg(
            n=("run_seed", "count"),
            val_f1_mean=("best_val_f1", "mean"),
            val_f1_sd=("best_val_f1", "std"),
            test_f1_mean=("test_f1@best", "mean"),
            test_f1_sd=("test_f1@best", "std"),
            test_acc_mean=("test_acc@best", "mean"),
            test_acc_sd=("test_acc@best", "std"),
            test_seconds_per_sequence_mean=("test_seconds_per_sequence", "mean"),
            peak_vram_gib_mean=("peak_vram_gib", "mean"),
        )
        .reset_index()
    )

    t3 = runs.loc[runs["T"].eq(3)].set_index("run_seed")
    t9 = runs.loc[runs["T"].eq(9)].set_index("run_seed")
    seeds = sorted(set(t3.index) & set(t9.index))
    if len(seeds) != 10:
        raise RuntimeError(f"Expected 10 paired T=3/T=9 seeds, got {seeds}")

    focused_rows = []
    for index, metric in enumerate(
        ["test_f1@best", "test_acc@best", "test_seconds_per_sequence", "peak_vram_gib"]
    ):
        left = t3.loc[seeds, metric].to_numpy(dtype=float)
        right = t9.loc[seeds, metric].to_numpy(dtype=float)
        finite = np.isfinite(left) & np.isfinite(right)
        left = left[finite]
        right = right[finite]
        diff = left - right
        ci = paired_bootstrap_ci(diff, args.seed + index) if len(diff) else [np.nan, np.nan]
        sd = float(diff.std(ddof=1)) if len(diff) > 1 else np.nan
        focused_rows.append(
            {
                "metric": metric,
                "n_pairs": len(diff),
                "t3_mean": float(left.mean()) if len(diff) else np.nan,
                "t9_mean": float(right.mean()) if len(diff) else np.nan,
                "mean_difference_t3_minus_t9": float(diff.mean()) if len(diff) else np.nan,
                "paired_bootstrap_ci_low": ci[0],
                "paired_bootstrap_ci_high": ci[1],
                "exact_sign_flip_p": exact_sign_flip_pvalue(diff) if metric == "test_f1@best" and len(diff) else np.nan,
                "paired_dz": float(diff.mean() / sd) if np.isfinite(sd) and sd > 0 else np.nan,
                "inferential_role": "primary" if metric == "test_f1@best" else "descriptive_secondary",
            }
        )
    focused = pd.DataFrame(focused_rows)

    runs.to_csv(args.output / "all_runs.csv", index=False)
    aggregate.to_csv(args.output / "window_summary_mean_sd.csv", index=False)
    focused.to_csv(args.output / "t3_vs_t9_paired_results.csv", index=False)
    payload = {
        "provenance": "agent_initiated_not_supervisor_directed",
        "run_grid": observed,
        "full_sweep": aggregate.to_dict(orient="records"),
        "focused_t3_vs_t9": focused.where(pd.notna(focused), None).to_dict(orient="records"),
        "guardrails": [
            "The ten-window sweep is descriptive because only three seeds were run for windows other than T=3 and T=9.",
            "The exact paired test was prespecified only for test Macro-F1 in T=3 versus T=9.",
            "Failure to reject a difference is not evidence of equivalence.",
            "Runtime and memory measurements are hardware- and implementation-specific."
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
