"""Aggregate human-like generation experiment outputs across datasets.

This script does not rerun the experiment. It reads the per-dataset JSON outputs
produced by `generate_and_evaluate.py` and creates:
- a final CSV containing all detector comparison rows
- a long-form CSV for downstream plotting/analysis
- a detector-level summary CSV
- publication-friendly plots for AUROC comparison and delta heatmaps

Typical usage:
python aggregate_human_like_results.py \
  --input_root human_like_results \
  --datasets xsum,squad,writing \
  --base_model_name gpt2-medium
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.figsize": (14, 8),
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate human-like experiment outputs")
    parser.add_argument("--input_root", type=str, required=True, help="Root output dir from run.sh")
    parser.add_argument("--datasets", type=str, default="xsum,squad,writing")
    parser.add_argument("--base_model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None, help="Where to write aggregated files; defaults to <input_root>/combined")
    return parser.parse_args()


def safe_name(name: str) -> str:
    return name.replace("/", "_")


def load_dataset_json(input_root: Path, dataset: str, base_model_name: str) -> Dict:
    dataset_dir = input_root / dataset
    json_path = dataset_dir / f"comparison_{dataset}_{base_model_name}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing expected result file: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)


def flatten_comparison_payload(payload: Dict, dataset: str, base_model_name: str) -> pd.DataFrame:
    rows: List[Dict] = []
    n_samples = payload.get("n_samples")
    prompt_len = payload.get("prompt_len")
    baselines_compared = payload.get("baselines_compared", [])

    for row in payload.get("comparison_summary", []):
        rows.append({
            "dataset": dataset,
            "base_model": base_model_name,
            "n_samples": n_samples,
            "prompt_len": prompt_len,
            "detector": row.get("detector"),
            "normal_roc_auc": row.get("normal_roc_auc"),
            "human_like_roc_auc": row.get("human_like_roc_auc"),
            "delta_roc_auc": row.get("delta_roc_auc"),
            "robust": row.get("robust"),
            "baselines_compared": ",".join(baselines_compared),
        })
    return pd.DataFrame(rows)


def flatten_generation_payload(payload: Dict, dataset: str, base_model_name: str, generation_key: str) -> pd.DataFrame:
    rows: List[Dict] = []
    for row in payload.get(generation_key, []):
        rows.append({
            "dataset": dataset,
            "base_model": base_model_name,
            "generation_type": generation_key.replace("_generation", ""),
            "detector": row.get("name"),
            "roc_auc": row.get("roc_auc"),
        })
    return pd.DataFrame(rows)


def build_detector_overview(summary_df: pd.DataFrame) -> pd.DataFrame:
    overview_rows: List[Dict] = []
    for detector, sub in summary_df.groupby("detector"):
        overview_rows.append({
            "detector": detector,
            "n_datasets": int(sub["dataset"].nunique()),
            "mean_normal_roc_auc": float(sub["normal_roc_auc"].mean()),
            "mean_human_like_roc_auc": float(sub["human_like_roc_auc"].mean()),
            "mean_delta_roc_auc": float(sub["delta_roc_auc"].mean()),
            "robust_rate": float(sub["robust"].mean()),
        })
    return pd.DataFrame(overview_rows).sort_values(["mean_delta_roc_auc", "detector"], ascending=[True, True])


def plot_dataset_auc_bars(summary_df: pd.DataFrame, output_path: Path) -> None:
    detectors = list(dict.fromkeys(summary_df["detector"].tolist()))
    datasets = list(dict.fromkeys(summary_df["dataset"].tolist()))
    x = np.arange(len(detectors))
    width = 0.35

    fig, axes = plt.subplots(len(datasets), 1, figsize=(max(14, len(detectors) * 1.25), 5 * len(datasets)), sharex=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub = summary_df[summary_df["dataset"] == dataset].set_index("detector")
        normal = [sub.loc[d, "normal_roc_auc"] if d in sub.index else np.nan for d in detectors]
        human_like = [sub.loc[d, "human_like_roc_auc"] if d in sub.index else np.nan for d in detectors]

        ax.bar(x - width / 2, normal, width=width, label="Normal LLM", color="#ff7f0e")
        ax.bar(x + width / 2, human_like, width=width, label="Human-like LLM", color="#2ca02c")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("AUROC")
        ax.set_title(f"Detector AUROC Comparison | dataset={dataset}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(detectors, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_delta_heatmap(summary_df: pd.DataFrame, output_path: Path) -> None:
    pivot = summary_df.pivot(index="dataset", columns="detector", values="delta_roc_auc")
    fig, ax = plt.subplots(figsize=(max(12, 1.2 * pivot.shape[1]), max(4, 0.8 * pivot.shape[0] + 2)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt="+.3f",
        cmap="RdYlGn_r",
        center=0.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Δ AUROC (human-like - normal)"},
        ax=ax,
    )
    ax.set_title("Delta AUROC Heatmap by Dataset and Detector")
    ax.set_xlabel("Detector")
    ax.set_ylabel("Dataset")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_robustness_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    counts = summary_df.groupby("dataset")["robust"].agg(["sum", "count"]).reset_index()
    counts["non_robust"] = counts["count"] - counts["sum"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts["dataset"], counts["sum"], label="Robust", color="#1f77b4")
    ax.bar(counts["dataset"], counts["non_robust"], bottom=counts["sum"], label="Not robust", color="#d62728")
    ax.set_ylabel("Detector count")
    ax.set_title("Robust vs Non-robust Detectors by Dataset")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir) if args.output_dir else (input_root / "combined")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    base_model_name = args.base_model_name
    base_model_tag = safe_name(base_model_name)

    comparison_frames: List[pd.DataFrame] = []
    normal_frames: List[pd.DataFrame] = []
    human_like_frames: List[pd.DataFrame] = []

    for dataset in datasets:
        payload = load_dataset_json(input_root, dataset, base_model_name)
        comparison_df = flatten_comparison_payload(payload, dataset, base_model_name)
        normal_df = flatten_generation_payload(payload, dataset, base_model_name, "normal_generation")
        human_like_df = flatten_generation_payload(payload, dataset, base_model_name, "human_like_generation")
        if not comparison_df.empty:
            comparison_frames.append(comparison_df)
        if not normal_df.empty:
            normal_frames.append(normal_df)
        if not human_like_df.empty:
            human_like_frames.append(human_like_df)

    if not comparison_frames:
        raise RuntimeError("No dataset result files were found to aggregate.")

    comparison_df = pd.concat(comparison_frames, ignore_index=True)
    generation_df = pd.concat(normal_frames + human_like_frames, ignore_index=True) if (normal_frames or human_like_frames) else pd.DataFrame()
    overview_df = build_detector_overview(comparison_df)

    comparison_csv = output_dir / f"all_results_summary_{base_model_tag}.csv"
    long_csv = output_dir / f"all_results_long_{base_model_tag}.csv"
    overview_csv = output_dir / f"all_results_detector_overview_{base_model_tag}.csv"

    comparison_df.to_csv(comparison_csv, index=False)
    generation_df.to_csv(long_csv, index=False)
    overview_df.to_csv(overview_csv, index=False)

    auc_plot = output_dir / f"all_results_auc_comparison_{base_model_tag}.png"
    delta_plot = output_dir / f"all_results_delta_heatmap_{base_model_tag}.png"
    robust_plot = output_dir / f"all_results_robustness_summary_{base_model_tag}.png"

    plot_dataset_auc_bars(comparison_df, auc_plot)
    plot_delta_heatmap(comparison_df, delta_plot)
    plot_robustness_summary(comparison_df, robust_plot)

    report = {
        "input_root": str(input_root),
        "datasets": datasets,
        "base_model_name": base_model_name,
        "comparison_csv": str(comparison_csv),
        "long_csv": str(long_csv),
        "overview_csv": str(overview_csv),
        "plots": [str(auc_plot), str(delta_plot), str(robust_plot)],
        "n_rows": int(comparison_df.shape[0]),
    }
    with open(output_dir / f"all_results_report_{base_model_tag}.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Saved aggregated outputs:")
    print(f"  {comparison_csv}")
    print(f"  {long_csv}")
    print(f"  {overview_csv}")
    for plot_path in report["plots"]:
        print(f"  {plot_path}")


if __name__ == "__main__":
    main()
