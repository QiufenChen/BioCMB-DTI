#!/usr/bin/env python

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


INPUT_FILE = Path("analysis_results/results/model_performance_radial.xlsx")
OUTPUT_DIR = Path("paper/figures")

DATASETS = ["SNAP", "DRH"]
RADIAL_BLOCKS = {
    "drug radial split": "Drug-radial",
    "prot radial split": "Protein-radial",
}
METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPR"]

METHOD_STYLE = {
    "BioCMB-DTI": {"color": "#CC79A7", "marker": "D"},
    "BioFusionDTI": {"color": "#0072B2", "marker": "o"},
    "DrugBAN": {"color": "#E69F00", "marker": "s"},
    "HyperAttentionDTI": {"color": "#009E73", "marker": "^"},
}


def parse_mean(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace("±", "\u00b1")
    if "\u00b1" in text:
        text = text.split("\u00b1", 1)[0]
    try:
        return float(text)
    except ValueError:
        return np.nan


def read_radial_data():
    records = []
    for dataset in DATASETS:
        df = pd.read_excel(INPUT_FILE, sheet_name=dataset)
        current_block = None
        current_model = None
        first_col = df.columns[0]
        for _, row in df.iterrows():
            first_value = row[first_col]
            region = row["Region"]

            if isinstance(first_value, str) and first_value.strip() in RADIAL_BLOCKS:
                current_block = first_value.strip()
                current_model = None
                continue

            if current_block is None or pd.isna(region):
                continue

            if isinstance(first_value, str) and first_value.strip() in METHOD_STYLE:
                current_model = first_value.strip()

            if current_model is None or str(region).strip() not in ["near", "far"]:
                continue

            for metric in METRICS:
                records.append({
                    "Dataset": dataset,
                    "RadialType": RADIAL_BLOCKS[current_block],
                    "Region": str(region).strip(),
                    "Model": current_model,
                    "Metric": metric,
                    "Score": parse_mean(row[metric]),
                })
    return pd.DataFrame(records)


def plot_radial_baseline_models(data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_order = [
        ("SNAP", "Drug-radial"),
        ("SNAP", "Protein-radial"),
        ("DRH", "Drug-radial"),
        ("DRH", "Protein-radial"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    axes = axes.ravel()
    x = np.arange(len(METRICS))
    region_offset = {"near": -0.08, "far": 0.08}

    for axis, (dataset, radial_type) in zip(axes, panel_order):
        panel = data[(data["Dataset"] == dataset) & (data["RadialType"] == radial_type)]
        for model, style in METHOD_STYLE.items():
            for region in ["near", "far"]:
                points = panel[(panel["Model"] == model) & (panel["Region"] == region)]
                y = [points.loc[points["Metric"] == metric, "Score"].iloc[0] if not points.loc[points["Metric"] == metric, "Score"].empty else np.nan for metric in METRICS]
                if region == "near":
                    axis.scatter(
                        x + region_offset[region],
                        y,
                        s=58,
                        marker=style["marker"],
                        color=style["color"],
                        linewidths=0.8,
                    )
                else:
                    axis.scatter(
                        x + region_offset[region],
                        y,
                        s=58,
                        marker=style["marker"],
                        facecolors="none",
                        edgecolors=style["color"],
                        linewidths=1.4,
                    )
        axis.set_title(f"{dataset}: {radial_type}", fontsize=13)
        axis.set_xticks(x)
        axis.set_xticklabels(METRICS, rotation=45, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_ylabel("Score")
    axes[2].set_ylabel("Score")

    method_handles = [
        Line2D([0], [0], marker=style["marker"], color="none", markerfacecolor=style["color"], markeredgecolor=style["color"], markersize=8, label=model)
        for model, style in METHOD_STYLE.items()
    ]
    region_handles = [
        Line2D([0], [0], marker="o", color="0.3", markerfacecolor="0.3", linestyle="none", markersize=7, label="near"),
        Line2D([0], [0], marker="o", color="0.3", markerfacecolor="none", linestyle="none", markersize=7, label="far"),
    ]
    fig.legend(handles=method_handles + region_handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    pdf_path = OUTPUT_DIR / "radial_baseline_models.pdf"
    png_path = OUTPUT_DIR / "radial_baseline_models.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main():
    data = read_radial_data()
    pdf_path, png_path = plot_radial_baseline_models(data)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
