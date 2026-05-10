"""Perturbation NPR-drop study across human, normal LLM, and human-like LLM text.

This script is isolated from the current human-like generation experiment.
It generates and saves:
- per-example CSVs
- dataset-level summary CSVs
- combined CSVs across xsum/squad/writing
- plots showing NPR-drop behavior

The main comparison is whether human-like LLM perturbation behavior is closer
by human texts than normal LLM texts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import datasets

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines.detectGPT import perturb_texts
from baselines.loss import get_ll
from baselines.rank import get_rank, get_ranks
from baselines.sample_generate import custom_datasets
from baselines.utils.loadmodel import load_base_model_and_tokenizer, load_mask_filling_model


sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.figsize": (14, 8),
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

GROUP_ORDER = ["human", "normal", "human_like"]
GROUP_LABELS = {
    "human": "Human",
    "normal": "Normal LLM",
    "human_like": "Human-like LLM",
}
GROUP_COLORS = {
    "human": "#1f77b4",
    "normal": "#ff7f0e",
    "human_like": "#2ca02c",
}
DEFAULT_DATASETS = ["xsum", "squad", "writing"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perturbation NPR-drop study")
    parser.add_argument("--datasets", type=str, default="xsum,squad,writing")
    parser.add_argument("--base_model_name", type=str, default="gpt2-medium")
    parser.add_argument("--mask_filling_model_name", type=str, default="t5-large")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--prompt_len", type=int, default=30)
    parser.add_argument("--generation_len", type=int, default=200)
    parser.add_argument("--min_words", type=int, default=55)
    parser.add_argument("--min_len", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--do_top_k", action="store_true")
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--do_top_p", action="store_true")
    parser.add_argument("--top_p", type=float, default=0.96)
    parser.add_argument("--DEVICE", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="perturbation_npr_drop_results")
    parser.add_argument("--n_perturbation_list", type=str, default="5,10,20")
    parser.add_argument("--pct_words_masked", type=float, default=0.3)
    parser.add_argument("--span_length", type=int, default=2)
    parser.add_argument("--buffer_size", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--dataset_key", type=str, default="document")
    return parser.parse_args()


def safe_name(name: str) -> str:
    return name.replace("/", "_")


def load_dataset_texts(dataset_name: str, args: argparse.Namespace, model_config: Dict) -> List[str]:
    if dataset_name == "xsum":
        data = datasets.load_dataset("xsum", split="train", cache_dir=args.cache_dir)["document"]
    elif dataset_name == "squad":
        data = datasets.load_dataset("squad", split="train", cache_dir=args.cache_dir)["context"]
    elif dataset_name == "writing":
        data = custom_datasets.load("writing", args.cache_dir)
    else:
        data = datasets.load_dataset(dataset_name, split="train", cache_dir=args.cache_dir)[args.dataset_key]

    data = list(dict.fromkeys(data))
    data = [x.strip() for x in data]
    data = [" ".join(x.split()) for x in data]

    tokenizer = model_config["base_tokenizer"]
    max_tokens = 950

    def truncate_text(text: str) -> str:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
            return_attention_mask=False,
        )
        return tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)

    data = [truncate_text(x) for x in data]

    if dataset_name in ["writing", "squad", "xsum"]:
        long_data = [x for x in data if len(x.split()) > 250]
        if long_data:
            data = long_data

    random.seed(0)
    random.shuffle(data)
    return data[: args.n_samples]


def generate_texts(texts: Sequence[str], model_config: Dict, args: argparse.Namespace, instruction: str | None = None) -> List[str]:
    torch.manual_seed(42)
    np.random.seed(42)

    device = args.DEVICE
    batch_size = max(1, int(args.batch_size))
    prompt_tokens = args.prompt_len

    if instruction:
        texts = [f"{instruction}\n{t}" for t in texts]

    decoded_all: List[str] = []
    min_words = args.min_words

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = list(texts[start:end])
        encoded = model_config["base_tokenizer"](
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=prompt_tokens,
        ).to(device)

        decoded = ["" for _ in range(len(batch_texts))]
        tries = 0
        while (m := min(len(x.split()) for x in decoded)) < min_words:
            if tries != 0:
                print(f"  batch {start}:{end} min words: {m}, regenerating (try {tries})")

            sampling_kwargs = {}
            if args.do_top_p:
                sampling_kwargs["top_p"] = args.top_p
            elif args.do_top_k:
                sampling_kwargs["top_k"] = args.top_k

            outputs = model_config["base_model"].generate(
                **encoded,
                min_new_tokens=max(1, int(args.min_len)),
                max_new_tokens=max(1, int(args.generation_len)),
                temperature=args.temperature,
                do_sample=True,
                **sampling_kwargs,
                pad_token_id=model_config["base_tokenizer"].eos_token_id,
                eos_token_id=model_config["base_tokenizer"].eos_token_id,
            )
            decoded = model_config["base_tokenizer"].batch_decode(outputs, skip_special_tokens=True)
            if instruction:
                decoded = [text.replace(instruction + "\n", "") for text in decoded]

            tries += 1
            if tries > 3:
                break

        decoded_all.extend(decoded)

        if device.startswith("cuda"):
            del encoded
            if 'outputs' in locals():
                del outputs
            torch.cuda.empty_cache()

    return decoded_all


def clean_values(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def compute_perturbation_rows(
    texts: Sequence[str],
    group: str,
    label: int,
    args: argparse.Namespace,
    model_config: Dict,
    n_perturbation_list: List[int],
) -> List[Dict]:
    max_n = max(n_perturbation_list)
    perturb_fn = perturb_texts
    perturbed_batches = perturb_fn([text for text in texts for _ in range(max_n)], args=args, model_config=model_config)

    rows: List[Dict] = []
    for idx, text in enumerate(texts):
        per_text_perturbed = perturbed_batches[idx * max_n : (idx + 1) * max_n]

        try:
            original_ll = get_ll(text, args, model_config)
            original_logrank = get_rank(text, args, model_config, log=True)
        except Exception:
            original_ll = np.nan
            original_logrank = np.nan

        try:
            per_text_logranks = get_ranks(per_text_perturbed, args, model_config, log=True)
        except Exception:
            per_text_logranks = [np.nan for _ in range(max_n)]

        row = {
            "example_id": idx,
            "group": group,
            "label": label,
            "original_ll": original_ll,
            "original_logrank": original_logrank,
        }

        for n in n_perturbation_list:
            subset = [x for x in per_text_logranks[:n] if np.isfinite(x)]
            mean_logrank = float(np.mean(subset)) if subset else np.nan
            std_logrank = float(np.std(subset)) if len(subset) > 1 else np.nan
            ratio = mean_logrank / original_logrank if np.isfinite(mean_logrank) and np.isfinite(original_logrank) and abs(original_logrank) > 1e-12 else np.nan
            drop = 1.0 - ratio if np.isfinite(ratio) else np.nan
            row[f"perturbed_mean_logrank_{n}"] = mean_logrank
            row[f"perturbed_std_logrank_{n}"] = std_logrank
            row[f"npr_ratio_{n}"] = ratio
            row[f"npr_drop_{n}"] = drop

        rows.append(row)

    return rows


def summarize_group(df: pd.DataFrame, n_perturbation_list: List[int]) -> pd.DataFrame:
    summary_rows: List[Dict] = []
    for (dataset, group), sub in df.groupby(["dataset", "group"]):
        for n in n_perturbation_list:
            values = clean_values(sub[f"npr_drop_{n}"].tolist())
            if values.size == 0:
                continue
            summary_rows.append({
                "dataset": dataset,
                "group": group,
                "n_perturbation": n,
                "n_examples": int(values.size),
                "mean_npr_drop": float(values.mean()),
                "std_npr_drop": float(values.std()) if values.size > 1 else 0.0,
                "median_npr_drop": float(np.median(values)),
                "q25_npr_drop": float(np.quantile(values, 0.25)),
                "q75_npr_drop": float(np.quantile(values, 0.75)),
                "mean_original_ll": float(np.nanmean(sub["original_ll"])),
                "mean_original_logrank": float(np.nanmean(sub["original_logrank"])),
            })
    return pd.DataFrame(summary_rows)


def plot_group_boxplots(df: pd.DataFrame, dataset: str, base_model_name: str, n: int, output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    data = []
    labels = []
    colors = []
    for group in GROUP_ORDER:
        values = clean_values(df.loc[df["group"] == group, f"npr_drop_{n}"].tolist())
        if values.size == 0:
            continue
        data.append(values)
        labels.append(GROUP_LABELS[group])
        colors.append(GROUP_COLORS[group])

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    ax.set_title(f"NPR Drop Distribution | dataset={dataset} | n={n}")
    ax.set_ylabel("NPR drop = 1 - perturbed_mean_logrank / original_logrank")
    ax.grid(alpha=0.25)

    out = output_dir / f"{dataset}_{safe_name(base_model_name)}_npr_drop_boxplot_{n}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_group_curves(df: pd.DataFrame, dataset: str, base_model_name: str, n_perturbation_list: List[int], output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    for group in GROUP_ORDER:
        xs = []
        means = []
        stds = []
        for n in n_perturbation_list:
            values = clean_values(df.loc[df["group"] == group, f"npr_drop_{n}"].tolist())
            if values.size == 0:
                continue
            xs.append(n)
            means.append(float(values.mean()))
            stds.append(float(values.std()) if values.size > 1 else 0.0)
        if xs:
            ax.errorbar(xs, means, yerr=stds, marker="o", linewidth=2, capsize=4, label=GROUP_LABELS[group], color=GROUP_COLORS[group])

    ax.set_title(f"Mean NPR Drop vs Perturbation Count | dataset={dataset}")
    ax.set_xlabel("Perturbations used")
    ax.set_ylabel("Mean NPR drop")
    ax.grid(alpha=0.25)
    ax.legend()

    out = output_dir / f"{dataset}_{safe_name(base_model_name)}_npr_drop_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_human_similarity(df: pd.DataFrame, dataset: str, base_model_name: str, n: int, output_dir: Path) -> str:
    summary = []
    human = clean_values(df.loc[df["group"] == "human", f"npr_drop_{n}"].tolist())
    human_mean = float(human.mean()) if human.size else np.nan
    for group in ["normal", "human_like"]:
        values = clean_values(df.loc[df["group"] == group, f"npr_drop_{n}"].tolist())
        mean_val = float(values.mean()) if values.size else np.nan
        summary.append({
            "group": GROUP_LABELS[group],
            "mean_npr_drop": mean_val,
            "abs_diff_from_human": abs(mean_val - human_mean) if np.isfinite(mean_val) and np.isfinite(human_mean) else np.nan,
        })

    plot_df = pd.DataFrame(summary)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(plot_df["group"], plot_df["abs_diff_from_human"], color=[GROUP_COLORS["normal"], GROUP_COLORS["human_like"]])
    ax.set_title(f"How Close NPR Drop Is to Human | dataset={dataset} | n={n}")
    ax.set_ylabel("Absolute difference from human mean NPR drop")
    ax.grid(axis="y", alpha=0.25)

    out = output_dir / f"{dataset}_{safe_name(base_model_name)}_human_similarity_{n}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def run_dataset(dataset_name: str, args: argparse.Namespace, model_config: Dict, n_perturbation_list: List[int], output_root: Path) -> pd.DataFrame:
    print(f"\n=== Dataset: {dataset_name} ===")
    dataset_dir = output_root / dataset_name
    feature_dir = dataset_dir / "feature_tables"
    summary_dir = dataset_dir / "summaries"
    plot_dir = dataset_dir / "plots"
    feature_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    texts = load_dataset_texts(dataset_name, args, model_config)
    print(f"Using {len(texts)} examples")

    human_texts = texts
    normal_generated = generate_texts(texts, model_config, args, instruction=None)
    human_like_prompt = (
        "Rewrite the following so it sounds naturally human, with varied sentence length, casual phrasing, and realistic flow while keeping the meaning intact:"
    )
    human_like_generated = generate_texts(texts, model_config, args, instruction=human_like_prompt)

    rows: List[Dict] = []
    rows.extend(compute_perturbation_rows(human_texts, "human", 0, args, model_config, n_perturbation_list))
    rows.extend(compute_perturbation_rows(normal_generated, "normal", 1, args, model_config, n_perturbation_list))
    rows.extend(compute_perturbation_rows(human_like_generated, "human_like", 1, args, model_config, n_perturbation_list))

    df = pd.DataFrame(rows)
    df.insert(0, "dataset", dataset_name)

    feature_csv = feature_dir / f"{dataset_name}_{safe_name(args.base_model_name)}_feature_table.csv"
    df.to_csv(feature_csv, index=False)

    summary_df = summarize_group(df, n_perturbation_list)
    summary_csv = summary_dir / f"{dataset_name}_{safe_name(args.base_model_name)}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    comparison_rows: List[Dict] = []
    for n in n_perturbation_list:
        human = clean_values(df.loc[df["group"] == "human", f"npr_drop_{n}"].tolist())
        normal = clean_values(df.loc[df["group"] == "normal", f"npr_drop_{n}"].tolist())
        human_like = clean_values(df.loc[df["group"] == "human_like", f"npr_drop_{n}"].tolist())
        human_mean = float(human.mean()) if human.size else np.nan
        normal_mean = float(normal.mean()) if normal.size else np.nan
        human_like_mean = float(human_like.mean()) if human_like.size else np.nan
        comparison_rows.append({
            "dataset": dataset_name,
            "n_perturbation": n,
            "human_mean_drop": human_mean,
            "normal_mean_drop": normal_mean,
            "human_like_mean_drop": human_like_mean,
            "normal_abs_diff_from_human": abs(normal_mean - human_mean) if np.isfinite(normal_mean) and np.isfinite(human_mean) else np.nan,
            "human_like_abs_diff_from_human": abs(human_like_mean - human_mean) if np.isfinite(human_like_mean) and np.isfinite(human_mean) else np.nan,
            "human_like_closer_than_normal": bool(abs(human_like_mean - human_mean) < abs(normal_mean - human_mean)) if np.isfinite(human_like_mean) and np.isfinite(normal_mean) and np.isfinite(human_mean) else False,
        })

    comparison_csv = summary_dir / f"{dataset_name}_{safe_name(args.base_model_name)}_comparison.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_csv, index=False)

    plot_files = []
    for n in n_perturbation_list:
        plot_files.append(plot_group_boxplots(df, dataset_name, args.base_model_name, n, plot_dir))
        plot_files.append(plot_human_similarity(df, dataset_name, args.base_model_name, n, plot_dir))
    plot_files.append(plot_group_curves(df, dataset_name, args.base_model_name, n_perturbation_list, plot_dir))

    report = {
        "dataset": dataset_name,
        "feature_csv": str(feature_csv),
        "summary_csv": str(summary_csv),
        "comparison_csv": str(comparison_csv),
        "plots": plot_files,
        "n_examples": int(df.shape[0]),
    }
    with open(summary_dir / f"{dataset_name}_{safe_name(args.base_model_name)}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return df


def combine_plots(output_root: Path, all_df: pd.DataFrame, n_perturbation_list: List[int], base_model_name: str) -> Dict[str, str]:
    combined_dir = output_root / "combined"
    plot_dir = combined_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    combined_plots: Dict[str, str] = {}

    # Heatmap: dataset x group mean NPR drop for each perturbation count.
    for n in n_perturbation_list:
        pivot = all_df.groupby(["dataset", "group"])[f"npr_drop_{n}"].mean().unstack("group")
        fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.6 + 1)))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", linewidths=0.5, ax=ax)
        ax.set_title(f"Mean NPR Drop by Dataset/Group | n={n}")
        ax.set_xlabel("Group")
        ax.set_ylabel("Dataset")
        out = plot_dir / f"all_datasets_{safe_name(base_model_name)}_heatmap_n{n}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        combined_plots[f"heatmap_n{n}"] = str(out)

    # Combined line plot of group means across n.
    fig, ax = plt.subplots(figsize=(10, 6))
    for group in GROUP_ORDER:
        xs = []
        ys = []
        for n in n_perturbation_list:
            vals = clean_values(all_df.loc[all_df["group"] == group, f"npr_drop_{n}"].tolist())
            if vals.size == 0:
                continue
            xs.append(n)
            ys.append(float(vals.mean()))
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, label=GROUP_LABELS[group], color=GROUP_COLORS[group])
    ax.set_title("Combined Mean NPR Drop Across Datasets")
    ax.set_xlabel("Perturbation count")
    ax.set_ylabel("Mean NPR drop")
    ax.grid(alpha=0.25)
    ax.legend()
    out = plot_dir / f"all_datasets_{safe_name(base_model_name)}_mean_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    combined_plots["mean_curve"] = str(out)

    return combined_plots


def main() -> None:
    args = parse_args()
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",") if x.strip()]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    model_config: Dict = {"cache_dir": args.cache_dir}
    model_config = load_base_model_and_tokenizer(args, model_config)
    model_config = load_mask_filling_model(args, args.mask_filling_model_name, model_config)

    datasets_requested = [x.strip() for x in args.datasets.split(",") if x.strip()]
    all_frames: List[pd.DataFrame] = []
    for dataset_name in datasets_requested:
        if dataset_name not in DEFAULT_DATASETS:
            print(f"Skipping unsupported dataset '{dataset_name}'")
            continue
        all_frames.append(run_dataset(dataset_name, args, model_config, n_perturbation_list, output_root))

    if not all_frames:
        raise RuntimeError("No datasets were processed.")

    all_df = pd.concat(all_frames, ignore_index=True)
    combined_dir = output_root / "combined"
    combined_summary_dir = combined_dir / "summaries"
    combined_plot_dir = combined_dir / "plots"
    combined_summary_dir.mkdir(parents=True, exist_ok=True)
    combined_plot_dir.mkdir(parents=True, exist_ok=True)

    combined_feature_csv = combined_dir / f"all_datasets_{safe_name(args.base_model_name)}_feature_table.csv"
    all_df.to_csv(combined_feature_csv, index=False)

    combined_summary = summarize_group(all_df, n_perturbation_list)
    combined_summary_csv = combined_dir / f"all_datasets_{safe_name(args.base_model_name)}_summary.csv"
    combined_summary.to_csv(combined_summary_csv, index=False)

    combined_comparison_rows: List[Dict] = []
    for dataset_name in sorted(all_df["dataset"].unique()):
        sub = all_df[all_df["dataset"] == dataset_name]
        for n in n_perturbation_list:
            human = clean_values(sub.loc[sub["group"] == "human", f"npr_drop_{n}"].tolist())
            normal = clean_values(sub.loc[sub["group"] == "normal", f"npr_drop_{n}"].tolist())
            human_like = clean_values(sub.loc[sub["group"] == "human_like", f"npr_drop_{n}"].tolist())
            if human.size == 0 or normal.size == 0 or human_like.size == 0:
                continue
            human_mean = float(human.mean())
            normal_mean = float(normal.mean())
            human_like_mean = float(human_like.mean())
            combined_comparison_rows.append({
                "dataset": dataset_name,
                "n_perturbation": n,
                "human_mean_drop": human_mean,
                "normal_mean_drop": normal_mean,
                "human_like_mean_drop": human_like_mean,
                "normal_abs_diff_from_human": abs(normal_mean - human_mean),
                "human_like_abs_diff_from_human": abs(human_like_mean - human_mean),
                "human_like_closer_than_normal": abs(human_like_mean - human_mean) < abs(normal_mean - human_mean),
            })

    combined_comparison_csv = combined_dir / f"all_datasets_{safe_name(args.base_model_name)}_comparison.csv"
    pd.DataFrame(combined_comparison_rows).to_csv(combined_comparison_csv, index=False)

    combined_plots = combine_plots(output_root, all_df, n_perturbation_list, args.base_model_name)

    report = {
        "datasets": datasets_requested,
        "base_model_name": args.base_model_name,
        "feature_csv": str(combined_feature_csv),
        "summary_csv": str(combined_summary_csv),
        "comparison_csv": str(combined_comparison_csv),
        "plots": list(combined_plots.values()),
        "n_examples": int(all_df.shape[0]),
    }
    with open(combined_dir / f"all_datasets_{safe_name(args.base_model_name)}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Saved outputs:")
    print(f"  {combined_feature_csv}")
    print(f"  {combined_summary_csv}")
    print(f"  {combined_comparison_csv}")
    for plot_path in combined_plots.values():
        print(f"  {plot_path}")


if __name__ == "__main__":
    main()
