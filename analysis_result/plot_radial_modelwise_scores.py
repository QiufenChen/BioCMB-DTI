#!/usr/bin/env python
"""Plot all-model near/far radial scores for one dataset sheet.

No argparse is used. Change ``SHEET_NAME`` before direct execution, or import
the script and call ``main("DRH")`` / ``main("TTD")``.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILE = Path(__file__).resolve().parent / "results" / "model_performance_radial.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "model_performance_radial_plots_V1"

SHEET_NAME = "SNAP"  # Change this one value when running the script directly.
METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPR"]

RADIAL_SPLITS = [("drug radial split", "Drug-radial"),
                 ("prot radial split", "Protein-radial")
                 ]

MODEL_RENAME = {
    "HpyerAttentionDTI": "HyperAttentionDTI",
    "HyperAttentionDTI": "HyperAttentionDTI",
    "BioCMBDTI": "BioCMB-DTI",
    "BioCMB-DTI": "BioCMB-DTI",
}

MODEL_ORDER = [
    "HyperAttentionDTI",
    "DrugBAN",
    "BINDTI",
    "BioFusionDTI",
    "BioCMB-DTI",
]


MODEL_COLORS = {
"HyperAttentionDTI": "#1F77B4",
"BINDTI": "#FF7F0E",
"DrugBAN": "#D62728",
"BioFusionDTI": "#2CA02C",
"BioCMB-DTI": "#7A45AC",
}


def normalize_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def parse_mean(value):
    """Extract the mean from an Excel value such as ``0.808±0.007``."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else np.nan


def find_section_row(raw, section_name, sheet_name):
    target = section_name.lower()
    matches = [
        index
        for index, value in raw.iloc[:, 0].items()
        if normalize_text(value).lower() == target
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one {!r} section in sheet {!r}, found {}.".format(
                section_name, sheet_name, len(matches)
            )
        )
    return matches[0]


def read_radial_sheet(sheet_name):
    """Read the near/far rows in the drug- and protein-radial table blocks."""
    workbook = pd.ExcelFile(INPUT_FILE, engine="openpyxl")
    if sheet_name not in workbook.sheet_names:
        raise ValueError(
            "Sheet {!r} does not exist. Available sheets: {}".format(
                sheet_name, ", ".join(workbook.sheet_names)
            )
        )

    raw = pd.read_excel(INPUT_FILE, sheet_name=sheet_name, header=None, engine="openpyxl")
    section_rows = [
        find_section_row(raw, split_name, sheet_name)
        for split_name, _ in RADIAL_SPLITS
    ]
    frames = []

    for section_index, ((split_name, _), header_row) in enumerate(
        zip(RADIAL_SPLITS, section_rows)
    ):
        next_header = (
            section_rows[section_index + 1]
            if section_index + 1 < len(section_rows)
            else len(raw)
        )
        block = raw.iloc[header_row + 1 : next_header, : 2 + len(METRICS)].copy()
        block = block.dropna(how="all")
        block.columns = ["Model", "Region", *METRICS]
        block["Model"] = block["Model"].ffill().map(normalize_text)
        block["Model"] = block["Model"].replace(MODEL_RENAME)
        block["Region"] = block["Region"].map(normalize_text).str.lower()
        block = block[block["Region"].isin(["near", "far"])].copy()
        block["Split"] = split_name

        for metric in METRICS:
            block[f"{metric}_mean"] = block[metric].map(parse_mean)
        frames.append(block)

    data = pd.concat(frames, ignore_index=True)
    validate_data(data, sheet_name)
    return data


def validate_data(data, sheet_name):
    """Confirm every model has one near and one far row in each radial split."""
    problems = []
    for split_name, _ in RADIAL_SPLITS:
        subset = data[data["Split"] == split_name]
        for model in MODEL_ORDER:
            for region in ["near", "far"]:
                count = int(
                    ((subset["Model"] == model) & (subset["Region"] == region)).sum()
                )
                if count != 1:
                    problems.append(
                        "{} / {} / {}: {} row(s)".format(
                            split_name, model, region, count
                        )
                    )

    mean_columns = [f"{metric}_mean" for metric in METRICS]
    if data[mean_columns].isna().any().any():
        problems.append("missing metric means")
    if problems:
        raise ValueError(
            "Radial data validation failed for sheet {!r}:\n  {}".format(
                sheet_name, "\n  ".join(problems)
            )
        )


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_one_radial_split(data, sheet_name, split_name, split_label):
    """Save one all-model plot with near/far columns under every metric."""
    subset = data[data["Split"] == split_name]
    metric_x = np.arange(len(METRICS))
    near_x = metric_x - 0.2
    far_x = metric_x + 0.2

    plt.figure(figsize=(6, 5))
    for model in MODEL_ORDER:
        model_data = subset[subset["Model"] == model]
        near = model_data[model_data["Region"] == "near"].iloc[0]
        far = model_data[model_data["Region"] == "far"].iloc[0]
        color = MODEL_COLORS[model]
        is_biocmb = model == "BioCMB-DTI"

        plt.scatter(
            near_x,
            [near[f"{metric}_mean"] for metric in METRICS],
            marker="o",
            s=140 if is_biocmb else 100,
            facecolors="none",
            edgecolors=color,
            linewidths=1.5 if is_biocmb else 1,
            label=model,
            zorder=4 if is_biocmb else 3,
            )
        
        plt.scatter(
            far_x,
            [far[f"{metric}_mean"] for metric in METRICS],
            marker="D",
            s=100 if is_biocmb else 80,
            facecolors="none",
            edgecolors=color,
            linewidths=1.5 if is_biocmb else 1,
            zorder=4 if is_biocmb else 3
            )

    plt.title("{} — {}".format(sheet_name, split_label), fontsize=14, pad=8)
    plt.ylabel("Score (mean)")
    # Major ticks label the metric once; near/far are the left/right columns.
    plt.xticks(metric_x, METRICS, rotation=45)
    plt.xlim(-0.5, len(METRICS) - 0.5)
    plt.ylim(0.0, 1.0)
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.grid(True, linestyle="-", linewidth=0.5, alpha=0.25)
    plt.scatter([], [], marker="o", s=70, facecolors="none", edgecolors="#4D4D4D", label="Near")
    plt.scatter([], [], marker="D", s=62, facecolors="none", edgecolors="#4D4D4D", label="Far")
    plt.legend(frameon=True, ncol=2, fontsize=8.5, loc="upper center")
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_stem = "drug_radial" if split_name == "drug radial split" else "protein_radial"
    output_stem = OUTPUT_DIR / "{}_{}_all_models_near_far_scores".format(
        sheet_name.lower(), split_stem
    )
    plt.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def main(sheet_name):
    """Generate one all-model near/far metric-scatter figure per radial split."""
    setup_style()
    data = read_radial_sheet(sheet_name)
    for split_name, split_label in RADIAL_SPLITS:
        plot_one_radial_split(data, sheet_name, split_name, split_label)
    print("Plots saved to: {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main(SHEET_NAME)
