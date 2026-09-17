from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILE = Path(__file__).resolve().parent / "results" / "model_performance.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "model_performance_metric_bars"
SHEET_NAME = "Sheet1"

DATASETS = ("SNAP", "DRH", "TTD")
METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPR"]

SPLIT_GROUPS = {
    "drug_related": [
        ("random split", ""),
        ("cold drug split", ""),
        ("drug radial split", "near"),
        ("drug radial split", "far"),
    ],
    "protein_related": [
        ("random split", ""),
        ("cold prot split", ""),
        ("prot radial split", "near"),
        ("prot radial split", "far"),
    ],
}

SPLIT_LABELS = {
    ("random split", ""): "Random split",
    ("cold drug split", ""): "Cold drug",
    ("cold prot split", ""): "Cold protein",
    ("drug radial split", "near"): "Drug radial\n(near)",
    ("drug radial split", "far"): "Drug radial\n(far)",
    ("prot radial split", "near"): "Protein radial\n(near)",
    ("prot radial split", "far"): "Protein radial\n(far)",
}

SPLIT_COLORS = {
    ("random split", ""): "#8EC7E8",
    ("cold drug split", ""): "#F6C56F",
    ("cold prot split", ""): "#F6C56F",
    ("drug radial split", "near"): "#8ED7B5",
    ("drug radial split", "far"): "#F2A07B",
    ("prot radial split", "near"): "#8ED7B5",
    ("prot radial split", "far"): "#F2A07B",
}

def parse_mean_std(value):
    if pd.isna(value):
        return np.nan, np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value), 0.0
    matches = re.findall(r"[-+]?\d*\.?\d+", str(value).strip())
    if not matches:
        raise ValueError("Cannot parse metric value: {!r}".format(value))
    mean = float(matches[0])
    std = float(matches[1]) if len(matches) > 1 else 0.0
    return mean, std


def read_performance_tables(path):
    raw = pd.read_excel(path, sheet_name=SHEET_NAME, header=None)
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

        start = header_row + 1
        end = start
        while end < len(raw) and not raw.iloc[end].isna().all():
            end += 1

        current_split = None
        for row_index in range(start, end):
            raw_split = raw.iat[row_index, 0]
            if pd.notna(raw_split):
                current_split = str(raw_split).strip().lower()
            if current_split is None:
                continue

            region = ""
            if "radial" in current_split and raw.shape[1] > 1:
                raw_region = raw.iat[row_index, 1]
                region = str(raw_region).strip().lower()

            key = (current_split, region)
            if key not in SPLIT_LABELS:
                continue

            item = {
                "Dataset": dataset,
                "Split": current_split,
                "Region": region,
                "SplitLabel": SPLIT_LABELS[key],
            }
            for metric in METRICS:
                mean, std = parse_mean_std(raw.iat[row_index, header[metric]])
                item[metric] = mean
                item[metric + "_std"] = std
            records.append(item)

    return pd.DataFrame(records)


def plot_metric_bar(data, metric, split_group_name, split_order):
    labels = list(DATASETS)
    x = np.arange(len(labels), dtype=float)
    bar_width = 0.18
    offsets = (np.arange(len(split_order)) - (len(split_order) - 1) / 2) * bar_width

    plt.figure(figsize=(6.0, 6.0))

    for split_key, offset in zip(split_order, offsets):
        split_name, region = split_key
        values = []
        stds = []
        subset = data[(data["Split"] == split_name) & (data["Region"] == region)]
        mean_lookup = {row["Dataset"]: row[metric] for _, row in subset.iterrows()}
        std_lookup = {row["Dataset"]: row[metric + "_std"] for _, row in subset.iterrows()}
        for dataset in DATASETS:
            values.append(mean_lookup.get(dataset, np.nan))
            stds.append(std_lookup.get(dataset, np.nan))

        plt.bar(
            x + offset,
            values,
            width=bar_width,
            yerr=stds,
            error_kw={
                "elinewidth": 1.2,
                "capsize": 3.0,
                "capthick": 1.2,
                "ecolor": "#333333",
            },
            color=SPLIT_COLORS[split_key],
            label=SPLIT_LABELS[split_key],
            edgecolor="white",
            linewidth=0.8,
        )

    plt.ylim(0.25, 1.0)
    plt.xticks(x, labels, rotation=0, ha="center", fontsize=12)
    plt.yticks(np.arange(0.2, 1.01, 0.1), fontsize=12)
    plt.ylabel("Score", fontsize=14, labelpad=10)
    plt.xlabel("Dataset", fontsize=14, labelpad=10)
    title_prefix = "Drug-related" if split_group_name == "drug_related" else "Protein-related"
    plt.title("{}: {}".format(title_prefix, metric), fontsize=16, pad=14)
    plt.legend(frameon=True, fontsize=12)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = OUTPUT_DIR / "model_performance_{}_{}_bar".format(
        split_group_name,
        metric.lower(),
    )
    plt.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close()


def main():
    data = read_performance_tables(INPUT_FILE)
    for split_group_name, split_order in SPLIT_GROUPS.items():
        for metric in METRICS:
            plot_metric_bar(data, metric, split_group_name, split_order)
    data.to_csv(OUTPUT_DIR / "model_performance_bar_values.csv", index=False, encoding="utf-8-sig")
    print("Plots saved to: {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
