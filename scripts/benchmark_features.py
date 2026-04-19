#!/usr/bin/env python3
from __future__ import annotations

"""Run comprehensive HRV feature importance benchmarking.

This script builds a tabular drowsiness dataset from raw HRV features and runs
four importance methods (RFE, MI, RF, PI) with cross-validation, then exports:

- Ranked feature lists for each method
- Comparison CSV files
- Publication-quality plots (bar charts + heatmaps)
- A Markdown report summarizing findings

Examples
--------
Synthetic fallback (default):
    python scripts/benchmark_features.py

Real DD-DB-style CSV directory:
    python scripts/benchmark_features.py --data-dir /path/to/dddb_csv
"""

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from driver_monitoring_rl.data.dddb_loader import DDDBLoader
from driver_monitoring_rl.features.feature_importance import (
    FeatureImportanceBenchmarkResult,
    build_raw_hrv_feature_dataset,
    run_feature_importance_benchmark,
)
from driver_monitoring_rl.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark HRV feature importance methods (RFE/MI/RF/PI)")
    p.add_argument("--data-dir", type=str, default="", help="Optional path to real DD-DB CSV files")
    p.add_argument("--out-dir", type=str, default="artifacts/feature_importance")
    p.add_argument("--sampling-rate-hz", type=int, default=128)
    p.add_argument("--window-seconds", type=int, default=120)
    p.add_argument("--min-gap-seconds", type=int, default=120)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--rf-estimators", type=int, default=400)
    p.add_argument("--permutation-repeats", type=int, default=20)
    p.add_argument("--random-state", type=int, default=42)

    p.add_argument("--synthetic-subjects", type=int, default=10)
    p.add_argument("--synthetic-duration-seconds", type=int, default=2 * 60 * 60)

    p.add_argument("--disable-balance", action="store_true", help="Disable class balancing in dataset build")
    return p.parse_args()


def configure_matplotlib() -> None:
    """Set high-quality figure defaults suitable for reports/publication drafts."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _minmax_scale(arr: np.ndarray) -> np.ndarray:
    lo = np.min(arr)
    hi = np.max(arr)
    if np.isclose(hi - lo, 0.0):
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def save_method_bar_charts(result: FeatureImportanceBenchmarkResult, out_dir: Path) -> Path:
    methods = list(result.methods.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()

    for idx, method_name in enumerate(methods):
        method = result.methods[method_name]
        frame = method.to_frame()
        ax = axes[idx]
        ax.barh(
            frame["feature"],
            frame["importance_mean"],
            xerr=frame["importance_std"],
            color="#4575b4",
            alpha=0.9,
            ecolor="#313695",
            capsize=2,
        )
        ax.set_title(f"{method_name} feature importance")
        ax.set_xlabel("Importance (mean ± std)")
        ax.invert_yaxis()

    fig.suptitle("HRV Feature Importance Comparison (Cross-Validated)", fontsize=14, y=1.02)
    out_path = out_dir / "feature_importance_bars.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_importance_heatmap(result: FeatureImportanceBenchmarkResult, out_dir: Path) -> Path:
    comparison = result.comparison_frame().set_index("feature")
    normalized = comparison.apply(lambda col: _minmax_scale(col.to_numpy()), axis=0)

    fig, ax = plt.subplots(figsize=(8, 5.6))
    matrix = normalized.to_numpy(dtype=float)
    im = ax.imshow(matrix, cmap="magma", aspect="auto")

    ax.set_xticks(np.arange(len(normalized.columns)))
    ax.set_xticklabels(normalized.columns)
    ax.set_yticks(np.arange(len(normalized.index)))
    ax.set_yticklabels(normalized.index)
    ax.set_title("Normalized importance heatmap (per-method min-max)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized importance")

    out_path = out_dir / "feature_importance_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_rank_heatmap(result: FeatureImportanceBenchmarkResult, out_dir: Path) -> Path:
    rank_df = result.rank_frame().set_index("feature")
    methods = [c.replace("_rank", "") for c in rank_df.columns]
    matrix = rank_df.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5.6))
    im = ax.imshow(matrix, cmap="viridis_r", aspect="auto")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_yticks(np.arange(len(rank_df.index)))
    ax.set_yticklabels(rank_df.index)
    ax.set_title("Feature rank heatmap (1 = most important)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", color="white", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Rank")

    out_path = out_dir / "feature_rank_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_markdown_report(
    result: FeatureImportanceBenchmarkResult,
    out_dir: Path,
    dataset_stats: Dict[str, float],
    plot_paths: Dict[str, Path],
    args: argparse.Namespace,
) -> Path:
    lines = [
        "# HRV Feature Importance Benchmark Report",
        "",
        "## Configuration",
        f"- Data source: {'real dataset' if args.data_dir else 'synthetic fallback'}",
        f"- CV folds requested: {args.cv_splits}",
        f"- Random state: {args.random_state}",
        f"- RF estimators: {args.rf_estimators}",
        f"- Permutation repeats: {args.permutation_repeats}",
        f"- Window seconds: {args.window_seconds}",
        f"- Min event gap seconds: {args.min_gap_seconds}",
        "",
        "## Dataset summary",
        f"- Samples: {int(dataset_stats['n_samples'])}",
        f"- Features: {int(dataset_stats['n_features'])}",
        f"- Positive class count: {int(dataset_stats['n_pos'])}",
        f"- Negative class count: {int(dataset_stats['n_neg'])}",
        "",
        "## Plots",
        f"![Bar charts]({plot_paths['bars'].name})",
        "",
        f"![Importance heatmap]({plot_paths['importance_heatmap'].name})",
        "",
        f"![Rank heatmap]({plot_paths['rank_heatmap'].name})",
        "",
        "## Ranked feature lists",
    ]

    for method_name, method_result in result.methods.items():
        lines.extend(["", f"### {method_name}", "", "| Rank | Feature | Mean Importance | Std |", "|---:|---|---:|---:|"])
        frame = method_result.to_frame()
        for rank, row in enumerate(frame.itertuples(index=False), start=1):
            lines.append(
                f"| {rank} | {row.feature} | {row.importance_mean:.6f} | {row.importance_std:.6f} |"
            )

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def main() -> None:
    args = parse_args()
    set_global_seed(args.random_state)
    configure_matplotlib()

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir:
        recordings = DDDBLoader(args.data_dir, fs=args.sampling_rate_hz).load()
    else:
        recordings = DDDBLoader.generate_synthetic_recordings(
            n_subjects=args.synthetic_subjects,
            duration_seconds=args.synthetic_duration_seconds,
            fs=args.sampling_rate_hz,
            seed=args.random_state,
        )

    X, y, feature_names = build_raw_hrv_feature_dataset(
        recordings,
        window_seconds=args.window_seconds,
        min_gap_seconds=args.min_gap_seconds,
        balance_classes=not args.disable_balance,
        random_state=args.random_state,
    )

    result = run_feature_importance_benchmark(
        X,
        y,
        feature_names,
        cv_splits=args.cv_splits,
        random_state=args.random_state,
        n_estimators=args.rf_estimators,
        permutation_repeats=args.permutation_repeats,
    )

    comparison_df = result.comparison_frame()
    ranks_df = result.rank_frame()
    comparison_df.to_csv(out_dir / "feature_importance_comparison.csv", index=False)
    ranks_df.to_csv(out_dir / "feature_importance_ranks.csv", index=False)

    for method_name, method_result in result.methods.items():
        method_result.to_frame().to_csv(out_dir / f"feature_importance_{method_name.lower()}.csv", index=False)

    plot_paths = {
        "bars": save_method_bar_charts(result, plots_dir),
        "importance_heatmap": save_importance_heatmap(result, plots_dir),
        "rank_heatmap": save_rank_heatmap(result, plots_dir),
    }

    report_path = write_markdown_report(
        result=result,
        out_dir=out_dir,
        dataset_stats={
            "n_samples": float(X.shape[0]),
            "n_features": float(X.shape[1]),
            "n_pos": float(np.sum(y == 1)),
            "n_neg": float(np.sum(y == 0)),
        },
        plot_paths=plot_paths,
        args=args,
    )

    print("Feature importance benchmarking complete.")
    print(f"Results saved to: {out_dir.resolve()}")
    print(f"Report: {report_path.resolve()}")

    print("\nRanked features by method:")
    for method_name, method_result in result.methods.items():
        ranked = method_result.to_frame()
        print(f"\n[{method_name}]")
        for idx, row in enumerate(ranked.itertuples(index=False), start=1):
            print(f"  {idx:>2d}. {row.feature:<14} mean={row.importance_mean:>9.6f}  std={row.importance_std:>9.6f}")


if __name__ == "__main__":
    main()
