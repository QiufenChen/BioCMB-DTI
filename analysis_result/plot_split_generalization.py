from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
EXCEL_PATH = PROJECT_DIR / "results" / "model_performance.xlsx"
OUTPUT_DIR = PROJECT_DIR / "model_performance"

SHEET_NAME = "Sheet1"
DATASETS_TO_PLOT = ("SNAP", "DRH", "TTD")
METRIC_DODGE = 0.06

METRIC_STYLE = {
    "Accuracy": {"color": "#CC79A7", "marker": "D"},
    "Precision": {"color": "#E69F00", "marker": "P"},
    "Recall": {"color": "#56B4E9", "marker": "v"},
    "F1": {"color": "#009E73", "marker": "^"},
    "AUROC": {"color": "#0072B2", "marker": "o"},
    "AUPR": {"color": "#D55E00", "marker": "s"},
}

TABLE_COLUMNS = [
    "Split",
    "Cousin",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUROC",
    "AUPR",
]

DRUG_SPLIT_NAMES = {
    "random split",
    "cold drug split",
    "drug cluster split",
    "drug radial split",
}

PROTEIN_SPLIT_NAMES = {
    "random split",
    "cold prot split",
    "prot cluster split",
    "prot radial split",
}

DISPLAY_NAMES = {
    "random split": "Random split",
    "cold drug split": "Cold drug",
    "cold prot split": "Cold protein",
    "drug cluster split": "Drug cluster",
    "prot cluster split": "Protein cluster",
    "drug radial split": "Drug radial",
    "prot radial split": "Protein radial",
}

COUSIN_DISPLAY_NAMES = {
    "1st cousin": "near",
    "2nd cousin": "far",
}


def parse_mean(value):
    """Convert an Excel value such as '0.867+/-0.003' or '0.867±0.003' to mean."""
    if pd.isna(value):
        raise ValueError("Missing metric value in Excel table")

    text = str(value).strip().replace("\u00b1", "+/-")
    parts = [part.strip() for part in text.split("+/-")]
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return float(parts[0])
    raise ValueError("Expected mean+/-std, got: {!r}".format(value))


def read_performance_tables(excel_path):
    """Read the SNAP, DRH, and TTD table blocks from one worksheet."""
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
        block["Cousin"] = block["Cousin"].fillna("").astype(str).str.strip()
        tables[dataset] = block.reset_index(drop=True)

    return tables


def make_display_label(split_name, cousin):
    label = DISPLAY_NAMES[split_name]
    if cousin:
        cousin_label = COUSIN_DISPLAY_NAMES.get(cousin.lower(), cousin.lower())
        label = "{}\n({})".format(label, cousin_label)
    return label


def build_panel_data(table, domain):
    """Select drug- or protein-related rows and parse plotted metric means."""
    if domain == "drug":
        allowed_splits = DRUG_SPLIT_NAMES
    elif domain == "protein":
        allowed_splits = PROTEIN_SPLIT_NAMES
    else:
        raise ValueError("domain must be 'drug' or 'protein'")

    panel_data = []
    for _, row in table.iterrows():
        split_name = row["Split"]
        if split_name not in allowed_splits:
            continue

        item = {"label": make_display_label(split_name, row["Cousin"])}
        for metric in METRIC_STYLE:
            item[metric] = parse_mean(row[metric])
        panel_data.append(item)

    return panel_data


def plot_generalization_panel(data, title, output_name):
    """Draw one split-generalization panel using mean values only."""
    labels = [row["label"] for row in data]
    x = np.arange(len(labels))

    plt.figure(figsize=(6, 6))

    metric_count = len(METRIC_STYLE)
    for metric_index, (metric, style) in enumerate(METRIC_STYLE.items()):
        means = np.array([row[metric] for row in data])
        offset = (metric_index - (metric_count - 1) / 2) * METRIC_DODGE
        plt.plot(
            x + offset,
            means,
            color=style["color"],
            marker=style["marker"],
            linestyle="none",
            markersize=12,
            markeredgecolor="#5A5959",
            markeredgewidth=0.5,
            label=metric,
            zorder=3,
        )

    plt.xlim(-0.45, len(labels) - 0.55)
    plt.ylim(0.20, 1.00)
    plt.xticks(x, labels, fontsize=20, rotation=90, ha="center")
    plt.yticks(np.arange(0.2, 1.1, 0.1), fontsize=20)
    plt.ylabel("Score (mean)", fontsize=20, labelpad=20)
    # plt.xlabel("Split setting", fontsize=15, labelpad=20)
    plt.title(title, fontsize=20, pad=20)
    plt.box(True)
    plt.legend(fontsize=16, frameon=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "{}.pdf".format(output_name), bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "{}.png".format(output_name), bbox_inches="tight", dpi=600)
    plt.close()


def main():
    tables = read_performance_tables(EXCEL_PATH)

    for dataset in DATASETS_TO_PLOT:
        drug_data = build_panel_data(tables[dataset], domain="drug")
        protein_data = build_panel_data(tables[dataset], domain="protein")

        plot_generalization_panel(
            drug_data,
            title="{} (Drug-related)".format(dataset),
            output_name="{}_drug_related_splits".format(dataset.lower()),
        )
        plot_generalization_panel(
            protein_data,
            title="{} (Protein-related)".format(dataset),
            output_name="{}_protein_related_splits".format(dataset.lower()),
        )

    print("Read data from: {}".format(EXCEL_PATH))
    print("Plots saved to: {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
