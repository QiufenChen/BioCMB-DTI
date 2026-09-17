"""Plot absolute cross-dataset performance heatmaps from cross_test.xlsx."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


INPUT_FILE = Path(__file__).resolve().parent / "results"/ "cross_test.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "cross_test_heatmaps"

METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPR"]
METRIC_GROUPS = {
    "main_metrics": ["F1", "AUROC", "AUPR"],
    "threshold_metrics": ["Accuracy", "Recall", "Precision"],
}

# Raw-performance heatmaps retain random_split as an absolute reference.
SPLIT_ORDER = [
    "random_split",
    "cold_drug_split",
    "cold_prot_split",
    "drug_radial_split",
    "prot_radial_split",
]

SPLIT_LABELS = {
    "random_split": "Random",
    "cold_drug_split": "Cold drug",
    "cold_prot_split": "Cold protein",
    "drug_radial_split": "Drug radial",
    "prot_radial_split": "Protein radial",
}


def parse_mean(value):
    """Extract the mean from a number or a 'mean +/- std' Excel cell."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    match = re.search(r"[-+]?\d*\.?\d+", str(value).strip())
    if not match:
        raise ValueError("Cannot parse metric value: {!r}".format(value))
    return float(match.group())


def clean_transfer_name(value):
    """Convert names such as SNAP_to_NP into SNAP -> NP plot labels."""
    name = re.sub(r"\s*\([^)]*\)\s*$", "", str(value).strip())
    return name.replace("_to_", " -> ")


def read_cross_test_table(path):
    """Read all transfer blocks from the first worksheet."""
    raw = pd.read_excel(path, sheet_name=0, header=None, dtype=object)
    records = []
    transfer_order = []

    row_index = 0
    while row_index < len(raw):
        first_cell = raw.iat[row_index, 0]
        third_cell = raw.iat[row_index, 2] if raw.shape[1] > 2 else None
        is_header = (
            pd.notna(first_cell)
            and str(third_cell).strip().lower() == "accuracy"
        )

        if not is_header:
            row_index += 1
            continue

        transfer = clean_transfer_name(first_cell)
        transfer_order.append(transfer)

        # Map each metric name in the block header to its Excel column index.
        header = {
            str(raw.iat[row_index, column]).strip(): column
            for column in range(raw.shape[1])
            if pd.notna(raw.iat[row_index, column])
        }

        missing_metrics = [metric for metric in METRICS if metric not in header]
        if missing_metrics:
            raise ValueError(
                "Missing metrics {} in block {}".format(missing_metrics, transfer)
            )

        data_row = row_index + 1
        while data_row < len(raw) and pd.notna(raw.iat[data_row, 0]):
            split = str(raw.iat[data_row, 0]).strip()
            record = {"Transfer": transfer, "Split": split}
            for metric in METRICS:
                record[metric] = parse_mean(raw.iat[data_row, header[metric]])
            records.append(record)
            data_row += 1

        row_index = data_row + 1

    if not records:
        raise ValueError("No transfer blocks were found in {}".format(path))

    return pd.DataFrame(records), transfer_order


def order_transfer_rows(transfer_order):
    """Place public transfers first and natural-product transfers last."""
    np_transfers = [
        transfer
        for transfer in transfer_order
        if transfer.upper().endswith("NP")
    ]
    public_transfers = [
        transfer for transfer in transfer_order if transfer not in np_transfers
    ]
    return public_transfers + np_transfers, len(public_transfers)


def build_metric_matrix(data, metric, transfer_order):
    matrix = data.pivot(index="Transfer", columns="Split", values=metric)
    return matrix.reindex(index=transfer_order, columns=SPLIT_ORDER)


def add_heatmap_values(axis, matrix, color_min, color_max):
    """Add numeric labels to a heatmap."""
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iat[row, column]
            if pd.isna(value):
                continue
            normalized = (value - color_min) / (color_max - color_min)
            text_color = "white" if normalized >= 0.62 else "#222222"
            axis.text(
                column + 0.5,
                row + 0.5,
                "{:.3f}".format(value),
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )


def decorate_heatmap_axis(axis, group_boundary, total_transfers, show_group_labels=True):
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_xticklabels(
        [SPLIT_LABELS[split] for split in SPLIT_ORDER],
        rotation=90,
        ha="center",
        fontsize=14,
    )
    axis.tick_params(axis="y", labelsize=10, length=0)
    axis.axhline(group_boundary, color="#333333", linewidth=2.0)
    if show_group_labels:
        axis.text(
            -0.36,
            1 - group_boundary / (2 * total_transfers),
            "Public data transfer",
            transform=axis.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
        axis.text(
            -0.36,
            (total_transfers - group_boundary) / (2 * total_transfers),
            "Natural product transfer",
            transform=axis.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )


def save_single_metric_heatmap(metric, matrix, color_min, color_max, group_boundary, total_transfers):
    plt.figure(figsize=(6, 6))
    axis = plt.gca()
    heatmap = sns.heatmap(
        matrix,
        ax=axis,
        cmap="YlGnBu",
        vmin=color_min,
        vmax=color_max,
        annot=False,
        linewidths=0.7,
        linecolor="white",
        cbar=True,
        square=False,
    )
    add_heatmap_values(axis, matrix, color_min, color_max)
    decorate_heatmap_axis(axis, group_boundary, total_transfers)
    axis.set_title(metric, fontsize=16, pad=20, fontweight="bold")
    heatmap.collections[0].colorbar.set_label(
        "{} performance (mean)".format(metric),
        rotation=90,
        fontsize=11,
        labelpad=12,
    )

    output_stem = OUTPUT_DIR / "cross_test_raw_heatmap_{}".format(metric.lower())
    plt.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.08)
    plt.close()


def save_metric_group_heatmap(group_name, metrics, matrices, color_min, color_max, group_boundary, total_transfers):
    plt.figure(figsize=(16, 6))
    colorbar_axis = plt.axes([0.915, 0.20, 0.014, 0.60])

    for panel_index, metric in enumerate(metrics, start=1):
        axis = plt.subplot(1, len(metrics), panel_index)
        matrix = matrices[metric]
        sns.heatmap(
            matrix,
            ax=axis,
            cmap="YlGnBu",
            vmin=color_min,
            vmax=color_max,
            annot=False,
            linewidths=0.7,
            linecolor="white",
            cbar=panel_index == len(metrics),
            cbar_ax=colorbar_axis if panel_index == len(metrics) else None,
            square=False,
        )

        add_heatmap_values(axis, matrix, color_min, color_max)
        decorate_heatmap_axis(
            axis,
            group_boundary,
            total_transfers,
            show_group_labels=panel_index == 1,
        )
        axis.set_title(metric, fontsize=16, pad=20, fontweight="bold")
        if panel_index > 1:
            axis.set_yticklabels([])

    colorbar_axis.set_ylabel(
        "Absolute performance (mean)",
        rotation=90,
        fontsize=11,
        labelpad=12,
    )

    output_stem = OUTPUT_DIR / "cross_test_raw_heatmaps_{}".format(group_name)
    plt.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.08)
    plt.close()


def plot_raw_heatmaps(data, transfer_order):
    transfer_order, group_boundary = order_transfer_rows(transfer_order)
    total_transfers = len(transfer_order)

    matrices = {
        metric: build_metric_matrix(data, metric, transfer_order)
        for metric in METRICS
    }

    # One shared color scale makes all metric heatmaps directly comparable.
    all_values = np.concatenate(
        [matrix.to_numpy(dtype=float).ravel() for matrix in matrices.values()]
    )
    color_min = np.floor(np.nanmin(all_values) * 20) / 20
    color_max = np.ceil(np.nanmax(all_values) * 20) / 20

    sns.set_theme(style="white", font_scale=1.0)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for group_name, metrics in METRIC_GROUPS.items():
        save_metric_group_heatmap(
            group_name,
            metrics,
            matrices,
            color_min,
            color_max,
            group_boundary,
            total_transfers,
        )

    data[["Transfer", "Split"] + METRICS].to_csv(
        OUTPUT_DIR / "cross_test_raw_values.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main():
    data, transfer_order = read_cross_test_table(INPUT_FILE)
    plot_raw_heatmaps(data, transfer_order)
    print("Plots saved to: {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
