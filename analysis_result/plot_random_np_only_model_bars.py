#!/usr/bin/env python
"""Plot four models under the NP-only random split."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILE = (
    Path(__file__).resolve().parent
    / "results"
    / "source_assisted_learning.xlsx"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "random_np_only_model_plots"
SHEET_NAME = "cold_prot_split"

METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPR"]

MODELS = [
    "DrugBAN",
    "BINDTI",
    "BioFusionDTI",
    "OurModel",
]

MODEL_LABELS = {
    "DrugBAN": "DrugBAN",
    "BINDTI": "BINDTI",
    "BioFusionDTI": "BioFusionDTI",
    "OurModel": "BioCMB-DTI",
}

MODEL_COLORS = {
    "DrugBAN": "#C98276",       # muted coral
    "BINDTI": "#D2AE68",        # muted ochre
    "BioFusionDTI": "#70A89A",  # muted teal
    "OurModel": "#9582AD",      # muted lavender
}


def normalize_key(value):
    if pd.isna(value):
        return ""

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def parse_mean_std(value):
    """Parse a value such as '0.808±0.007' into mean and SD."""
    if pd.isna(value):
        return np.nan, np.nan

    if isinstance(value, (int, float, np.number)):
        return float(value), np.nan

    numbers = re.findall(r"-?\d+(?:\.\d+)?", str(value))

    if not numbers:
        return np.nan, np.nan

    mean = float(numbers[0])
    std = float(numbers[1]) if len(numbers) > 1 else np.nan
    return mean, std


def load_np_only_random_results():
    """Read and validate NP-only results for the four selected models."""
    data = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

    required_columns = {"Training method", "Model", *METRICS}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    data = data.copy()
    data["Training method"] = data["Training method"].ffill()
    data["Model"] = data["Model"].ffill().astype(str).str.strip()
    data["MethodKey"] = data["Training method"].map(normalize_key)

    subset = data[
        (data["MethodKey"] == "np_only")
        & (data["Model"].isin(MODELS))
    ].copy()

    duplicated = subset["Model"].duplicated(keep=False)
    available_models = set(subset["Model"])
    missing_models = [
        model for model in MODELS
        if model not in available_models
    ]

    if duplicated.any() or missing_models:
        duplicate_models = subset.loc[duplicated, "Model"].tolist()

        raise ValueError(
            "Expected exactly one NP-only row per model. "
            f"Missing: {missing_models}; "
            f"duplicates: {duplicate_models}"
        )

    subset = subset.set_index("Model")

    means = {}
    stds = {}

    for model in MODELS:
        means[model] = []
        stds[model] = []

        for metric in METRICS:
            mean, std = parse_mean_std(subset.loc[model, metric])

            if np.isnan(mean):
                raise ValueError(f"Missing {metric} value for {model}")

            means[model].append(mean)
            stds[model].append(0.0 if np.isnan(std) else std)

    return means, stds


def plot_bars(means, stds):
    """Save a grouped mean +/- SD bar chart for four models."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = np.arange(len(METRICS))
    width = 0.18

    offsets = (
        np.arange(len(MODELS))
        - (len(MODELS) - 1) / 2
    ) * width

    plt.figure(figsize=(10, 6))

    for model, offset in zip(MODELS, offsets):
        plt.bar(
            x + offset,
            means[model],
            width=width,
            yerr=stds[model],
            color=MODEL_COLORS[model],
            alpha=0.90,
            edgecolor="white",
            error_kw={
                "ecolor": "#404040",
                "elinewidth": 1.0,
                "capsize": 3,
                "capthick": 1.0,
            },
            label=MODEL_LABELS[model],
            zorder=3,
        )

    plt.title(f"{SHEET_NAME}", fontsize=16, pad=10)
    plt.ylabel("Score (mean ± SD)")
    plt.xticks(x, METRICS, rotation=35, ha="right")
    plt.ylim(0.0, 1.0)
    plt.yticks(np.arange(0.0, 1.01, 0.1))
    plt.grid(axis="y", linewidth=0.5, alpha=0.25, zorder=0)

    plt.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_stem = (
        OUTPUT_DIR
        / f"{normalize_key(SHEET_NAME)}_np_only_model_comparison"
    )

    plt.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close("all")
    print(f"Saved: {output_stem}")


def main():
    means, stds = load_np_only_random_results()
    plot_bars(means, stds)


if __name__ == "__main__":
    main()