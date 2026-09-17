#!/usr/bin/env python
"""Plot baseline-model comparisons for one sheet in model_baseline.xlsx.

Choose the sheet through ``main(sheet_name)``.  No argparse is used:
change ``SHEET_NAME`` below, or import this script and call, for example,
``main("DRH")``.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILE = Path(__file__).resolve().parent / "results" / "model_baseline.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "model_baseline_plots"

# Change this one value when running the script directly.
SHEET_NAME = "TTD"

METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPR"]

# Standard display names used consistently in figures and legends.
MODEL_RENAME = {
    "HpyerAttentionDTI": "HyperAttentionDTI",
    "HyperAttentionDTI": "HyperAttentionDTI",
    "BioCMBDTI": "BioCMB-DTI",
    "BioCMB-DTI": "BioCMB-DTI",
}

MODEL_ORDER = [
    "HyperAttentionDTI",
    "BINDTI",
    "DrugBAN",
    "BioFusionDTI",
    "MIF-DTI",
    "MML-DTI",
    "GraphBAN",
    "SaeGraphDTI",
    "BioCMB-DTI",
]

MODEL_COLORS = {
"HyperAttentionDTI": "#1F77B4",
"BINDTI": "#FF7F0E",
"DrugBAN": "#D62728",
"BioFusionDTI": "#2CA02C",
"MIF-DTI": "#8C564B",
"MML-DTI": "#E377C2",
"GraphBAN": "#7F7F7F",
"SaeGraphDTI": "#17BECF",
"BioCMB-DTI": "#7A45AC",
}


def normalize_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def parse_mean(value):
    """Extract the mean from values such as ``0.813±0.008``."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else np.nan


def split_label(split_name):
    """Convert an Excel section header to the figure title and output stem."""
    display = normalize_text(split_name)
    stem = re.sub(r"[^a-z0-9]+", "_", display.lower()).strip("_")
    return display, stem


def read_baseline_sheet(sheet_name):
    """Read all split blocks from one dataset sheet and standardize model names."""
    workbook = pd.ExcelFile(INPUT_FILE, engine="openpyxl")
    if sheet_name not in workbook.sheet_names:
        raise ValueError(
            "Sheet {!r} does not exist. Available sheets: {}".format(
                sheet_name, ", ".join(workbook.sheet_names)
            )
        )

    raw = pd.read_excel(INPUT_FILE, sheet_name=sheet_name, header=None, engine="openpyxl")
    section_rows = [
        index
        for index, value in raw.iloc[:, 0].items()
        if "split" in normalize_text(value).lower()
    ]
    if not section_rows:
        raise ValueError("No '* split' sections found in sheet {!r}".format(sheet_name))

    frames = []
    for section_index, header_row in enumerate(section_rows):
        next_header = (
            section_rows[section_index + 1]
            if section_index + 1 < len(section_rows)
            else len(raw)
        )
        split_name = normalize_text(raw.iloc[header_row, 0])
        block = raw.iloc[
            header_row + 1 : next_header,
            : 1 + len(METRICS),
        ].copy()
        block = block.dropna(how="all")
        block.columns = ["Model", *METRICS]
        block["Model"] = block["Model"].map(normalize_text)
        block = block[block["Model"] != ""].copy()
        block["Model"] = block["Model"].replace(MODEL_RENAME)
        block["Split"] = split_name

        for metric in METRICS:
            block[f"{metric}_mean"] = block[metric].map(parse_mean)
        frames.append(block)

    data = pd.concat(frames, ignore_index=True)
    validate_data(data, sheet_name)
    return data


def validate_data(data, sheet_name):
    """Fail early if a split lacks a model or a metric mean."""
    problems = []
    for split_name, subset in data.groupby("Split", sort=False):
        observed = set(subset["Model"])
        missing_models = [model for model in MODEL_ORDER if model not in observed]
        unexpected_models = sorted(observed - set(MODEL_ORDER))
        if missing_models:
            problems.append(
                "{}: missing models [{}]".format(split_name, ", ".join(missing_models))
            )
        if unexpected_models:
            problems.append(
                "{}: unrecognized models [{}]".format(
                    split_name, ", ".join(unexpected_models)
                )
            )
        if subset["Model"].duplicated().any():
            duplicates = subset.loc[subset["Model"].duplicated(), "Model"].tolist()
            problems.append("{}: duplicate models [{}]".format(split_name, ", ".join(duplicates)))

        mean_columns = [f"{metric}_mean" for metric in METRICS]
        if subset[mean_columns].isna().any().any():
            problems.append("{}: missing metric means".format(split_name))

    if problems:
        raise ValueError(
            "Baseline data validation failed for sheet {!r}:\n  {}".format(
                sheet_name, "\n  ".join(problems)
            )
        )


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_one_split(data, sheet_name, split_name):
    """Save one open-circle comparison plot for one split."""
    subset = data[data["Split"] == split_name]
    x = np.arange(len(METRICS))

    plt.figure(figsize=(5, 5))
    for model in MODEL_ORDER:
        row = subset[subset["Model"] == model].iloc[0]
        scores = [row[f"{metric}_mean"] for metric in METRICS]
        is_biocmb = model == "BioCMB-DTI"

        plt.scatter(
            x,
            scores,
            marker="o",
            s=140 if is_biocmb else 100,
            facecolors="none",
            edgecolors=MODEL_COLORS[model],
            linewidths=2 if is_biocmb else 1.5,
            label=model,
            zorder=4 if is_biocmb else 3,
        )

    split_display, split_stem = split_label(split_name)
    plt.title("{} — {}".format(sheet_name, split_display), fontsize=15, pad=10)
    plt.ylabel("Score (mean)")
    plt.xticks(x, METRICS, rotation=90)
    plt.ylim(0.0, 1.0)
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.grid(True, linestyle="-", linewidth=0.5, alpha=0.25)
    plt.legend(frameon=True, ncol=3, fontsize=8.5, loc="upper center")
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = OUTPUT_DIR / "{}_{}_baseline_models".format(
        sheet_name.lower(), split_stem
    )
    plt.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def main(sheet_name):
    """Plot all split blocks in the requested workbook sheet."""
    setup_style()
    data = read_baseline_sheet(sheet_name)
    for split_name in data["Split"].drop_duplicates():
        plot_one_split(data, sheet_name, split_name)
    print("Plots saved to: {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main(SHEET_NAME)
