import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


FEATURE_ORDER = ["log_likelihood", "log_rank", "entropy", "lrr", "ensemble"]
FEATURE_TITLES = {
    "log_likelihood": "Log Likelihood",
    "log_rank": "Log Rank",
    "entropy": "Entropy",
    "lrr": "LRR",
    "ensemble": "Ensemble Score",
}
GROUP_ORDER = ["human", "normal", "human_like"]
GROUP_LABELS = {
    "human": "Human (Original)",
    "normal": "LLM Generated",
    "human_like": "Human-like LLM",
}
GROUP_COLORS = {
    "human": "#1f77b4",
    "normal": "#ff7f0e",
    "human_like": "#2ca02c",
}


def _clean(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def save_feature_distribution_artifacts(
    feature_distributions: Dict[str, Dict[str, List[float]]],
    output_dir: str,
    dataset: str,
    base_model: str,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    model_tag = base_model.replace("/", "_")
    values_path = os.path.join(output_dir, f"distribution_values_{dataset}_{model_tag}.json")
    figure_path = os.path.join(output_dir, f"distribution_plots_{dataset}_{model_tag}.png")

    with open(values_path, "w") as f:
        json.dump(feature_distributions, f, indent=2)

    available_features = []
    for feature in FEATURE_ORDER:
        present = any(feature in feature_distributions.get(group, {}) for group in GROUP_ORDER)
        if present:
            available_features.append(feature)

    n_features = len(available_features)
    if n_features == 0:
        return {"values_path": values_path, "figure_path": ""}

    n_cols = 2
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, feature in enumerate(available_features):
        ax = axes[idx]
        plotted = False

        for group in GROUP_ORDER:
            raw_values = feature_distributions.get(group, {}).get(feature, [])
            values = _clean(raw_values)
            if values.size == 0:
                continue

            plotted = True
            bins = min(30, max(10, int(np.sqrt(values.size)) + 5))
            ax.hist(
                values,
                bins=bins,
                density=True,
                alpha=0.4,
                color=GROUP_COLORS[group],
                label=f"{GROUP_LABELS[group]} (n={values.size}, mean={values.mean():.3f})",
            )

        ax.set_title(FEATURE_TITLES.get(feature, feature))
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)
        if plotted:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No valid values", ha="center", va="center", transform=ax.transAxes)

    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        f"Feature Distributions | dataset={dataset}, model={base_model}",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {"values_path": values_path, "figure_path": figure_path}
