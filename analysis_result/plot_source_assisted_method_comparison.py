#!/usr/bin/env python
"""Compare three training strategies for each model and evaluation subset.

For every sheet/region combination in ``source_assisted_learning.xlsx``, this
script saves one single-panel scatter plot. Each metric has three model
columns; each model column contains the scores of all training strategies.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


INPUT_FILE = Path(__file__).resolve().parent / "results" / "source_assisted_learning.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "source_assisted_method_comparison_plots"

METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPR"]
MODELS = ["BINDTI", "BioFusionDTI", "OurModel"]
MODEL_LABELS = {
    "HpyerAttentionDTI": "HyperAttentionDTI",
    "BINDTI": "BINDTI",
    "BioFusionDTI": "BioFusionDTI",
    "OurModel": "BioCMB-DTI",
}

METHODS = [
    ("np_only", "NP-only"),
    ("np_sdt_joint", "NP + SDT"),
    (
        "np_source_similarity_weighted_decay",
        "Source-assisted learning",
    ),
]

MODEL_MARKERS = {
    "BINDTI": "o",
    "BioFusionDTI": "D",
    "OurModel": "p",
}

MODEL_MARKER_SIZES = {
    "BINDTI": 140,
    "BioFusionDTI": 105,
    "OurModel": 150,
}

METHOD_COLORS = {
    "np_only": "#2CA02C",
    "np_sdt_joint": "#FF7F0E",
    "np_source_similarity_weighted_decay": "#7A45AC",
}


def normalize_key(value):
    """Normalize line breaks, spaces, and underscores to one stable key."""
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def parse_mean_std(value):
    """Parse values such as ``0.808±0.007`` into mean and standard deviation."""
    if pd.isna(value):
        return np.nan, np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value), np.nan

    numbers = re.findall(r"-?\d+(?:\.\d+)?", str(value))
    if not numbers:
        return np.nan, np.nan
    return float(numbers[0]), float(numbers[1]) if len(numbers) > 1 else np.nan


def read_sheet(sheet_name):
    """Read one workbook sheet and standardize its method and region fields."""
    df = pd.read_excel(INPUT_FILE, sheet_name=sheet_name)
    if "Training method" not in df.columns or "Model" not in df.columns:
        raise ValueError("{} is missing Training method or Model columns".format(sheet_name))

    df = df.copy()
    df["Training method"] = df["Training method"].ffill()
    df["Model"] = df["Model"].ffill().astype(str).str.strip()

    region_col = next((column for column in df.columns if str(column).startswith("Unnamed")), None)
    
    if region_col is None:
        df["Region"] = "test"
    else:
        df = df.rename(columns={region_col: "Region"})
        df["Region"] = df["Region"].ffill().astype(str).str.strip()

    df["MethodKey"] = df["Training method"].map(normalize_key)
    for metric in METRICS:
        if metric not in df.columns:
            raise ValueError("{} is missing metric column {}".format(sheet_name, metric))
        parsed = df[metric].map(parse_mean_std)
        df["{}_mean".format(metric)] = parsed.map(lambda item: item[0])
        df["{}_std".format(metric)] = parsed.map(lambda item: item[1])
    return df


def load_all_data():
    workbook = pd.ExcelFile(INPUT_FILE, engine="openpyxl")
    frames = []
    for sheet_name in workbook.sheet_names:
        frame = read_sheet(sheet_name)
        frame["Sheet"] = sheet_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def validate_subset(data, sheet_name, region):
    """Require exactly one result for every model-method pair in a subset."""
    problems = []
    for model in MODELS:
        for method_key, _ in METHODS:
            count = int(
                ((data["Model"] == model) & (data["MethodKey"] == method_key)).sum()
            )
            if count != 1:
                problems.append("{} / {}: {} row(s)".format(model, method_key, count))

    mean_columns = ["{}_mean".format(metric) for metric in METRICS]
    if data[mean_columns].isna().any().any():
        problems.append("missing metric mean value")
    if problems:
        raise ValueError(
            "Invalid data for {} / {}:\n  {}".format(
                sheet_name, region, "\n  ".join(problems)
            )
        )


def setup_style():
    if plt is None:
        raise ImportError("matplotlib is required to generate the comparison plots")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def display_name(value):
    return str(value).replace("_", " ").replace("split", "split").title()


def build_strategy_values(subset):
    """Collect all model scores for every metric within each strategy."""
    values = {method_key: [] for method_key, _ in METHODS}
    for method_key, _ in METHODS:
        method_data = subset[subset["MethodKey"] == method_key]
        for metric in METRICS:
            values[method_key].append(
                method_data.set_index("Model").loc[MODELS, "{}_mean".format(metric)].to_numpy(
                    dtype=float
                )
            )
    return values


def finish_plot(title, x):
    plt.title(title, fontsize=13, pad=9)
    plt.ylabel("Score (mean)")
    plt.xticks(x, METRICS, rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.25)
    # plt.gca().spines["top"].set_visible(False)
    # plt.gca().spines["right"].set_visible(False)
    plt.legend(frameon=True, ncol=3, fontsize=8.5, loc="upper center")
    plt.tight_layout()


def plot_strategy_columns(subset, sheet_name, region, output_stem):
    """Plot three model columns per metric, each containing strategy scores."""
    values = build_strategy_values(subset)
    metric_x = np.arange(len(METRICS))
    # Every metric is a three-column group, ordered by the three models.
    model_offsets = np.linspace(-0.25, 0.25, len(MODELS))
    plt.figure(figsize=(10, 7))

    for metric_index in range(len(METRICS)):
        for model_index, model in enumerate(MODELS):
            x_position = metric_x[metric_index] + model_offsets[model_index]
            for method_key, _ in METHODS:
                score = values[method_key][metric_index][model_index]
                plt.scatter(
                    x_position,
                    score,
                    marker=MODEL_MARKERS[model],
                    s=MODEL_MARKER_SIZES[model],
                    facecolors="none",
                    edgecolors=METHOD_COLORS[method_key],
                    linewidths=1,
                    zorder=3,
                )

    # The first legend explains model shape; the second explains strategy colour.
    for model in MODELS:
        plt.scatter(
            [], [], marker=MODEL_MARKERS[model], s=MODEL_MARKER_SIZES[model],
            facecolors="none", edgecolors="#4D4D4D",
            linewidths=1,
            label=MODEL_LABELS[model],
        )
    model_legend = plt.legend(
        title="Model", frameon=True, ncol=2, fontsize=12,
        title_fontsize=13, loc="upper left",
    )
    plt.gca().add_artist(model_legend)

    for method_key, method_label in METHODS:
        plt.scatter(
            [], [], marker="o", s=140, facecolors="none",
            edgecolors=METHOD_COLORS[method_key], linewidths=1,
            label=method_label,
        )
    plt.legend(
        title="Training strategy", frameon=True, fontsize=12,
        title_fontsize=13, loc="upper right",
    )

    plt.title("{} — {}".format(display_name(sheet_name), region), fontsize=20, pad=12)
    plt.ylabel("Score (mean)", fontsize=16)
    plt.xticks(metric_x, METRICS, rotation=45, ha="right", fontsize=15)
    plt.ylim(0.0, 1.0)
    plt.yticks(np.arange(0.0, 1.1, 0.1), fontsize=15)
    plt.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.25)
    # plt.gca().spines["top"].set_visible(False)
    # plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def main():
    setup_style()
    data = load_all_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sheet_name in data["Sheet"].drop_duplicates():
        sheet_data = data[data["Sheet"] == sheet_name]
        for region in sheet_data["Region"].drop_duplicates():
            subset = sheet_data[sheet_data["Region"] == region].copy()
            validate_subset(subset, sheet_name, region)
            stem = "{}_{}_all_models_method_columns".format(
                normalize_key(sheet_name), normalize_key(region)
            )
            output_stem = OUTPUT_DIR / stem
            plot_strategy_columns(subset, sheet_name, region, output_stem)

    print("Plots saved to: {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
