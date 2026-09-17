from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


INPUT_FILE = Path(__file__).resolve().parent / "results" / "model_ablation.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "model_ablation_plots"
SHEET_NAME = "Sheet1"

DATASETS = ("SNAP", "DRH", "TTD")
BASELINE_MODEL = "BioMCB-DTI"
METRICS = ["Accuracy", "Precision", "Recall", "F1", "MCC", "AUROC", "AUPR"]
VARIANT_ORDER = [
    "w/o Mamba",
    "w/o CNN",
    "w/o adaptive gate",
    "Cross-attention fusion",
    "Concatenation fusion",
]


def parse_mean(value):
    """Extract mean from numeric or 'mean±std' cells."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    match = re.search(r"[-+]?\d*\.?\d+", str(value).strip())
    if not match:
        return np.nan
    return float(match.group())


def read_ablation_tables(path):
    raw = pd.read_excel(path, sheet_name=SHEET_NAME, header=None, dtype=object)
    first_column = raw.iloc[:, 0].astype("string").str.strip()
    records = []

    for dataset in DATASETS:
        header_rows = raw.index[first_column.eq(dataset)].tolist()
        if len(header_rows) != 1:
            raise ValueError("Expected one header row for {}, found {}".format(dataset, len(header_rows)))

        header_row = header_rows[0]
        header = {
            str(raw.iat[header_row, column]).strip(): column
            for column in range(raw.shape[1])
            if pd.notna(raw.iat[header_row, column])
        }
        missing_metrics = [metric for metric in METRICS if metric not in header]
        if missing_metrics:
            raise ValueError("Missing metrics {} in {} table".format(missing_metrics, dataset))

        row_index = header_row + 1
        while row_index < len(raw) and pd.notna(raw.iat[row_index, 0]):
            variant = str(raw.iat[row_index, 0]).strip()
            row = {"Dataset": dataset, "Variant": variant}
            for metric in METRICS:
                row[metric] = parse_mean(raw.iat[row_index, header[metric]])
            records.append(row)
            row_index += 1

    return pd.DataFrame(records)


def calculate_delta(data):
    baseline = (
        data[data["Variant"] == BASELINE_MODEL]
        .set_index("Dataset")[METRICS]
        .rename(columns=lambda metric: metric + "_baseline")
    )
    if set(DATASETS) - set(baseline.index):
        missing = sorted(set(DATASETS) - set(baseline.index))
        raise ValueError("Missing baseline rows for: {}".format(", ".join(missing)))

    delta = data.join(baseline, on="Dataset")
    for metric in METRICS:
        delta[metric] = delta[metric] - delta[metric + "_baseline"]
    delta = delta[delta["Variant"] != BASELINE_MODEL].copy()
    return delta


def build_matrix(delta, dataset):
    subset = delta[delta["Dataset"] == dataset]
    matrix = subset.set_index("Variant")[METRICS]
    return matrix.reindex(index=VARIANT_ORDER, columns=METRICS)


def add_heatmap_values(axis, matrix, color_limit):
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iat[row, column]
            if pd.isna(value):
                continue
            text_color = "white" if abs(value) / color_limit >= 0.55 else "#222222"
            axis.text(
                column + 0.5,
                row + 0.5,
                "{:+.3f}".format(value),
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )


def plot_delta_heatmap(delta):
    matrices = {dataset: build_matrix(delta, dataset) for dataset in DATASETS}
    max_abs_delta = max(
        np.nanmax(np.abs(matrix.to_numpy(dtype=float)))
        for matrix in matrices.values()
    )
    color_limit = max(0.005, np.ceil(max_abs_delta * 1000) / 1000)

    sns.set_theme(style="white", font_scale=1.0)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    plt.figure(figsize=(15, 4.8))
    colorbar_axis = plt.axes([0.925, 0.22, 0.014, 0.58])

    for panel_index, dataset in enumerate(DATASETS, start=1):
        axis = plt.subplot(1, len(DATASETS), panel_index)
        matrix = matrices[dataset]
        sns.heatmap(
            matrix,
            ax=axis,
            cmap="RdBu_r",
            center=0,
            vmin=-color_limit,
            vmax=color_limit,
            annot=False,
            linewidths=0.7,
            linecolor="white",
            cbar=panel_index == len(DATASETS),
            cbar_ax=colorbar_axis if panel_index == len(DATASETS) else None,
            square=False,
        )
        add_heatmap_values(axis, matrix, color_limit)
        axis.set_title(dataset, fontsize=16, pad=16, fontweight="bold")
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_xticklabels(METRICS, rotation=45, ha="right", fontsize=11)
        axis.tick_params(axis="y", labelsize=11, length=0)
        if panel_index > 1:
            axis.set_yticklabels([])

    colorbar_axis.set_ylabel(
        "Delta performance vs BioCMB-DTI",
        rotation=90,
        fontsize=11,
        labelpad=12,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = OUTPUT_DIR / "model_ablation_delta_heatmap"
    plt.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.08)
    plt.close()


def main():
    data = read_ablation_tables(INPUT_FILE)
    delta = calculate_delta(data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_DIR / "model_ablation_raw_values.csv", index=False, encoding="utf-8-sig")
    delta[["Dataset", "Variant"] + METRICS].to_csv(
        OUTPUT_DIR / "model_ablation_delta_values.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_delta_heatmap(delta)
    print("Plots saved to: {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
