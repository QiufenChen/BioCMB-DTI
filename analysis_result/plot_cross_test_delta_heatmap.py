from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""
\(\Delta M=M_{\text{current split}}-M_{\text{random split}}\)

"""
INPUT_FILE = Path(__file__).resolve().parent / "results"/ "cross_test.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "cross_test_delta_heatmaps"

METRICS = ["F1", "AUROC", "AUPR"]

SPLIT_ORDER = [
    "cold_drug_split",
    "cold_prot_split",
    "drug_radial_split",
    "prot_radial_split",
]

SPLIT_LABELS = {
    "cold_drug_split": "Cold drug",
    "cold_prot_split": "Cold protein",
    "drug_radial_split": "Drug radial",
    "prot_radial_split": "Protein radial",
}


def parse_mean(value):
    """Extract the mean from a numeric value or a 'mean +/- std' cell."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    match = re.search(r"[-+]?\d*\.?\d+", str(value).strip())
    if not match:
        raise ValueError("Cannot parse metric value: {!r}".format(value))
    return float(match.group())


def clean_transfer_name(value):
    """Convert workbook section names into compact plot labels."""
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


def calculate_delta(data):
    """Subtract each transfer task's random-split performance."""
    baseline = (
        data[data["Split"] == "random_split"]
        .set_index("Transfer")[METRICS]
        .rename(columns=lambda metric: metric + "_baseline")
    )

    missing_baselines = sorted(set(data["Transfer"]) - set(baseline.index))
    if missing_baselines:
        raise ValueError(
            "Missing random_split baseline for: {}".format(
                ", ".join(missing_baselines)
            )
        )

    delta = data.join(baseline, on="Transfer")
    for metric in METRICS:
        delta[metric] = delta[metric] - delta[metric + "_baseline"]

    return delta[delta["Split"].isin(SPLIT_ORDER)].copy()


def build_metric_matrix(delta, metric, transfer_order):
    matrix = delta.pivot(index="Transfer", columns="Split", values=metric)
    return matrix.reindex(index=transfer_order, columns=SPLIT_ORDER)


def order_transfer_rows(transfer_order):
    """Place public-dataset transfers first and natural-product transfers last."""
    np_transfers = [
        transfer
        for transfer in transfer_order
        if transfer.upper().endswith("NP")
    ]
    public_transfers = [transfer for transfer in transfer_order if transfer not in np_transfers]
    return public_transfers + np_transfers, len(public_transfers)


def plot_delta_heatmaps(delta, transfer_order):
    transfer_order, group_boundary = order_transfer_rows(transfer_order)
    matrices = {metric: build_metric_matrix(delta, metric, transfer_order) for metric in METRICS}

    max_abs_delta = max(np.nanmax(np.abs(matrix.to_numpy(dtype=float))) 
                        for matrix in matrices.values())
    color_limit = max(0.01, np.ceil(max_abs_delta * 100) / 100)

    sns.set_theme(style="white", font_scale=1.0)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    plt.figure(figsize=(16, 6))
    colorbar_axis = plt.axes([0.915, 0.20, 0.014, 0.60])

    for panel_index, metric in enumerate(METRICS, start=1):
        axis = plt.subplot(1, 3, panel_index)
        matrix = matrices[metric]

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
            cbar=panel_index == len(METRICS),
            cbar_ax=colorbar_axis if panel_index == len(METRICS) else None,
            square=False,
        )

        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.iat[row, column]
                if pd.isna(value):
                    continue
                text_color = (
                    "white"
                    if abs(value) / color_limit >= 0.55
                    else "#222222"
                )
                axis.text(
                    column + 0.5,
                    row + 0.5,
                    "{:+.3f}".format(value),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                )

        axis.set_title(metric, fontsize=16, pad=20, fontweight="bold")
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_xticklabels([SPLIT_LABELS[split] for split in SPLIT_ORDER],
                             rotation=90, ha="center", fontsize=14)
        axis.tick_params(axis="y", labelsize=10, length=0)
        axis.axhline(group_boundary, color="#333333", linewidth=2.0)

        if panel_index > 1:
            axis.set_yticklabels([])
        else:
            axis.text(
                -0.36,
                0.83,
                "Natural product data transfer",
                transform=axis.transAxes,
                rotation=90,
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold")
            axis.text(
                -0.36,
                0.33,
                "Public-dataset transfer",
                transform=axis.transAxes,
                rotation=90,
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold")

    colorbar_axis.set_ylabel("Delta performance relative to random split", rotation=90, fontsize=11, labelpad=12)
    plt.suptitle("Cross-dataset generalization under different splits", fontsize=18, y=1.05, fontweight="bold")
    
    # plt.subplots_adjust(left=0.18,right=0.89, bottom=0.23, top=0.84, wspace=0.10)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = OUTPUT_DIR / "cross_test_delta_heatmaps"
    plt.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.08,)
    plt.close()

    delta_output = delta[["Transfer", "Split"] + METRICS].copy()
    delta_output.to_csv(
        OUTPUT_DIR / "cross_test_delta_values.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main():
    data, transfer_order = read_cross_test_table(INPUT_FILE)
    delta = calculate_delta(data)
    plot_delta_heatmaps(delta, transfer_order)
    print("Plots saved to: {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
