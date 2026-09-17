#!/usr/bin/env python
"""Plot source-assisted performance changes relative to NP-only training."""

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
OUTPUT_DIR = Path(__file__).resolve().parent / "source_assisted_delta_bar_plots"

METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPR"]

MODELS = ["DrugBAN", "BINDTI", "BioFusionDTI", "OurModel"]

MODEL_LABELS = {
    "DrugBAN": "DrugBAN",
    "BINDTI": "BINDTI",
    "BioFusionDTI": "BioFusionDTI",
    "OurModel": "BioCMB-DTI",
}

MODEL_COLORS = {
    "DrugBAN": "#C98276",
    "BINDTI": "#D2AE68",
    "BioFusionDTI": "#70A89A",
    "OurModel": "#9582AD",
}

BASE_METHOD = "np_only"

DELTA_METHODS = [
    (
        "np_sdt_joint",
        "NP + SDT - NP-only",
    ),
    (
        "np_source_similarity_weighted_decay",
        "Similarity-weighted source learning - NP-only",
    ),
]

EVALUATIONS = [
    ("cold_drug_split", "test", "Cold-drug split", "cold_drug"),
    ("drug_radial_split", "near", "Drug-radial split: near", "radial_near"),
    ("drug_radial_split", "far", "Drug-radial split: far", "radial_far"),
    (
        "drug_radial_split",
        "near + far",
        "Drug-radial split: near + far",
        "radial_near_far",
    ),
]


def normalize_key(value):
    if pd.isna(value):
        return ""

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def parse_mean_std(value):
    """Parse a value such as '0.808+/-0.007' or '0.808±0.007'."""
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


def read_sheet(sheet_name):
    data = pd.read_excel(INPUT_FILE, sheet_name=sheet_name)

    required = {"Training method", "Model", *METRICS}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"Sheet '{sheet_name}' is missing columns: {sorted(missing)}"
        )

    data = data.copy()
    data["Training method"] = data["Training method"].ffill()
    data["Model"] = data["Model"].ffill().astype(str).str.strip()

    if "Region" not in data.columns:
        data["Region"] = "test"

    data["Region"] = data["Region"].fillna("test").astype(str).str.strip().str.lower()
    data["MethodKey"] = data["Training method"].map(normalize_key)

    for metric in METRICS:
        parsed = data[metric].map(parse_mean_std)
        data[f"{metric}_mean"] = parsed.map(lambda item: item[0])
        data[f"{metric}_std"] = parsed.map(lambda item: item[1])

    return data


def get_row(data, region, model, method):
    rows = data[
        (data["Region"] == region.lower())
        & (data["Model"] == model)
        & (data["MethodKey"] == method)
    ]

    if len(rows) != 1:
        raise ValueError(
            "Expected exactly one row for "
            f"region={region!r}, model={model!r}, method={method!r}; "
            f"found {len(rows)}."
        )

    return rows.iloc[0]


def calculate_deltas(data, region):
    """Calculate mean differences relative to NP-only training."""
    results = {}

    for method, method_label in DELTA_METHODS:
        method_means = {}

        for model in MODELS:
            baseline = get_row(data, region, model, BASE_METHOD)
            comparison = get_row(data, region, model, method)

            deltas = []

            for metric in METRICS:
                baseline_mean = baseline[f"{metric}_mean"]
                comparison_mean = comparison[f"{metric}_mean"]
                deltas.append(comparison_mean - baseline_mean)

            method_means[model] = np.asarray(deltas, dtype=float)

        results[method] = {
            "label": method_label,
            "means": method_means,
        }

    return results


def style_current_axis():
    plt.axhline(0, color="#444444", linewidth=1.0, zorder=2)
    plt.ylim(-0.1, 0.4)
    plt.yticks(np.arange(-0.1, 0.41, 0.1))
    plt.ylabel("Delta score vs NP-only")
    plt.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65, zorder=0)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    x = np.arange(len(METRICS))
    plt.xticks(x, METRICS)
    plt.setp(plt.gca().get_xticklabels(), rotation=35, ha="right")


def plot_evaluation(data, region, title, output_stem):
    results = calculate_deltas(data, region)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.5,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = np.arange(len(METRICS))
    width = 0.18
    offsets = (
        np.arange(len(MODELS)) - (len(MODELS) - 1) / 2
    ) * width

    plt.figure(figsize=(14.0, 5.4))
    legend_handles = []

    for panel_index, (method, _) in enumerate(DELTA_METHODS, start=1):
        plt.subplot(1, 2, panel_index)

        for model, offset in zip(MODELS, offsets):
            bars = plt.bar(
                x + offset,
                results[method]["means"][model],
                width=width * 0.92,
                color=MODEL_COLORS[model],
                edgecolor="none",
                alpha=0.92,
                label=MODEL_LABELS[model],
                zorder=3,
            )

            if panel_index == 1:
                legend_handles.append(bars[0])

        plt.title(results[method]["label"], fontsize=12, loc="left", pad=7)
        style_current_axis()
        if panel_index == 2:
            plt.ylabel("")

    plt.suptitle(title, fontsize=15, y=0.995)
    plt.figlegend(
        legend_handles,
        [MODEL_LABELS[model] for model in MODELS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(MODELS),
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.87], w_pad=2.5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{output_stem}_delta_vs_np_only"

    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(
        output_path.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")

    print(f"Saved: {output_path}")


def main():
    sheets = {}

    for sheet_name, region, title, output_stem in EVALUATIONS:
        if sheet_name not in sheets:
            sheets[sheet_name] = read_sheet(sheet_name)

        plot_evaluation(
            sheets[sheet_name],
            region,
            title,
            output_stem,
        )


if __name__ == "__main__":
    main()
