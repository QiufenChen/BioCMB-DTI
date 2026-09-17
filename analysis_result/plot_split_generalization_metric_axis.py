from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
EXCEL_PATH = PROJECT_DIR / "results" / "model_performance.xlsx"
OUTPUT_DIR = PROJECT_DIR / "model_performance"

SHEET_NAME = "Sheet1"
DATASETS_TO_PLOT = ("SNAP", "DRH", "TTD")

METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPR"]

TABLE_COLUMNS = [
    "Split",
    "Region",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUROC",
    "AUPR",
]

DRUG_SPLITS = [
    ("random split", "", "Random"),
    ("cold drug split", "", "Cold drug"),
    ("drug radial split", "near", "Drug radial near"),
    ("drug radial split", "far", "Drug radial far"),
]

PROTEIN_SPLITS = [
    ("random split", "", "Random"),
    ("cold prot split", "", "Cold protein"),
    ("prot radial split", "near", "Protein radial near"),
    ("prot radial split", "far", "Protein radial far"),
]

SPLIT_STYLE = {
    "Random": {"color": "#0072B2", "marker": "o"},
    "Cold drug": {"color": "#D55E00", "marker": "s"},
    "Cold protein": {"color": "#D55E00", "marker": "s"},
    "Drug radial near": {"color": "#009E73", "marker": "^"},
    "Drug radial far": {"color": "#CC79A7", "marker": "D"},
    "Protein radial near": {"color": "#009E73", "marker": "^"},
    "Protein radial far": {"color": "#CC79A7", "marker": "D"},
}


def parse_mean(value):
    if pd.isna(value):
        raise ValueError("Missing metric value in Excel table")

    text = str(value).strip()
    text = text.replace("+/-", "±").replace("卤", "±")
    return float(text.split("±")[0])


def read_performance_tables(excel_path):
    raw = pd.read_excel(excel_path, sheet_name=SHEET_NAME, header=None)
    first_column = raw.iloc[:, 0].astype("string").str.strip()
    tables = {}

    for dataset in DATASETS_TO_PLOT:
        header_rows = raw.index[first_column.eq(dataset)].tolist()
        if len(header_rows) != 1:
            raise ValueError("Expected one header row for {}, found {}".format(dataset, len(header_rows)))

        start = header_rows[0] + 1
        end = start
        while end < len(raw) and not raw.iloc[end].isna().all():
            end += 1

        block = raw.iloc[start:end, :8].copy()
        block.columns = TABLE_COLUMNS
        block["Split"] = block["Split"].ffill().astype(str).str.strip().str.lower()
        block["Region"] = block["Region"].fillna("").astype(str).str.strip().str.lower()
        tables[dataset] = block.reset_index(drop=True)

    return tables


def get_metric_values(table, split_name, region):
    hit = table[(table["Split"] == split_name) & (table["Region"] == region)]
    if hit.empty:
        raise ValueError("Cannot find split={} region={}".format(split_name, region))

    row = hit.iloc[0]
    return [parse_mean(row[metric]) for metric in METRICS]


def plot_dataset_metric_axis_panel(table, dataset, split_specs, title, output_name):
    x = np.arange(len(METRICS))

    plt.figure(figsize=(6.4, 5.0))

    for split_name, region, split_label in split_specs:
        style = SPLIT_STYLE[split_label]
        values = get_metric_values(table, split_name, region)

        plt.plot(
            x,
            values,
            linestyle="none",
            marker=style["marker"],
            markersize=9.5,
            color=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=split_label,
            zorder=3,
        )

    plt.xlim(-0.5, len(METRICS) - 0.5)
    plt.ylim(0.30, 0.98)
    plt.xticks(x, METRICS, fontsize=14, rotation=90, ha="center")
    plt.yticks(np.arange(0.3, 1.0, 0.1), fontsize=14)
    plt.ylabel("Score", fontsize=15, labelpad=14)
    plt.xlabel("Metric", fontsize=15, labelpad=14)
    plt.title("{}: {}".format(dataset, title), fontsize=16, pad=16)
    plt.box(True)
    plt.legend(fontsize=11, frameon=True, loc="lower left")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "{}.pdf".format(output_name), bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "{}.png".format(output_name), bbox_inches="tight", dpi=600)
    plt.close()


def main():
    tables = read_performance_tables(EXCEL_PATH)

    for dataset in DATASETS_TO_PLOT:
        plot_dataset_metric_axis_panel(
            table=tables[dataset],
            dataset=dataset,
            split_specs=DRUG_SPLITS,
            title="Drug-related generalization",
            output_name="{}_metric_axis_drug_related_generalization".format(dataset.lower()),
        )

        plot_dataset_metric_axis_panel(
            table=tables[dataset],
            dataset=dataset,
            split_specs=PROTEIN_SPLITS,
            title="Protein-related generalization",
            output_name="{}_metric_axis_protein_related_generalization".format(dataset.lower()),
        )

    print("Read data from: {}".format(EXCEL_PATH))
    print("Plots saved to: {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
