from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
EXCEL_PATH = PROJECT_DIR / "results" / "model_performance.xlsx"
OUTPUT_DIR = PROJECT_DIR / "model_performance"
SHEET_NAME = "Sheet1"
DATASETS_TO_PLOT = ("SNAP", "DRH", "TTD")
TICK_FONTSIZE = 14
LABEL_FONTSIZE = 15
TITLE_FONTSIZE = 16

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

DISPLAY_NAMES = {
    "random split": "Random split",
    "cold drug split": "Cold drug",
    "cold prot split": "Cold protein",
    "drug radial split": "Drug radial",
    "prot radial split": "Protein radial",
}

SPLIT_LAYOUT = [
    ("Baseline", "random split", ""),
    ("Entity-level", "cold drug split", ""),
    ("Entity-level", "cold prot split", ""),
    ("Radial-level", "drug radial split", "near"),
    ("Radial-level", "drug radial split", "far"),
    ("Radial-level", "prot radial split", "near"),
    ("Radial-level", "prot radial split", "far"),
]

GROUP_GAP = 0.1
METRIC_DODGE = 0.075

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


def build_grouped_panel_data(table):
    """Arrange all split settings into conceptual groups."""
    row_lookup = {}
    for _, row in table.iterrows():
        split_name = row["Split"]
        cousin = row["Cousin"].strip().lower()
        row_lookup[(split_name, cousin)] = row

    panel_data = []
    x_position = 0.0
    previous_group = None
    for group, split_name, cousin in SPLIT_LAYOUT:
        if previous_group is not None and group != previous_group:
            x_position += GROUP_GAP

        key = (split_name, cousin)
        if key not in row_lookup:
            raise ValueError("Missing split row: split={!r}, cousin={!r}".format(split_name, cousin))

        row = row_lookup[key]
        item = {
            "group": group,
            "label": make_display_label(split_name, row["Cousin"]),
            "x": x_position,
        }
        for metric in METRIC_STYLE:
            item[metric] = parse_mean(row[metric])
        panel_data.append(item)
        x_position += 1.0
        previous_group = group

    return panel_data


def get_group_ranges(data):
    group_ranges = []
    for group, _, _ in SPLIT_LAYOUT:
        if any(existing[0] == group for existing in group_ranges):
            continue
        positions = [row["x"] for row in data if row["group"] == group]
        group_ranges.append((group, min(positions), max(positions)))
    return group_ranges


def plot_grouped_dot_panel(data, title, output_name):
    """Draw grouped mean-value points without implying a trajectory."""
    labels = [row["label"] for row in data]
    x = np.array([row["x"] for row in data], dtype=float)
    group_ranges = get_group_ranges(data)

    plt.figure(figsize=(12, 6))

    for group_index, (_, start, end) in enumerate(group_ranges):
        if group_index % 2 == 1:
            plt.axvspan(start - 0.48, end + 0.48, color="#CFCFCF", linewidth=0, zorder=0)

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

    for group, start, end in group_ranges:
        plt.text(
            (start + end) / 2,
            1.025,
            group,
            transform=plt.gca().get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.xlim(x.min() - 0.55, x.max() + 0.55)
    plt.ylim(0.20, 1.00)
    plt.xticks(x, labels, fontsize=TICK_FONTSIZE, rotation=90, ha="center")
    plt.yticks(np.arange(0.2, 1.1, 0.1), fontsize=TICK_FONTSIZE)
    plt.ylabel("Score (mean)", fontsize=LABEL_FONTSIZE, labelpad=20)
    # plt.xlabel("Split setting", fontsize=LABEL_FONTSIZE, labelpad=20)
    plt.title(title, loc="center", multialignment="center", fontsize=TITLE_FONTSIZE, y=1.20, pad=0)
    plt.box(True)
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=6,
        fontsize=14,
        frameon=True,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "{}.pdf".format(output_name), bbox_inches="tight", pad_inches=0.08)
    plt.savefig(OUTPUT_DIR / "{}.png".format(output_name), bbox_inches="tight", pad_inches=0.08, dpi=600)
    plt.close()


def main():
    tables = read_performance_tables(EXCEL_PATH)

    for dataset in DATASETS_TO_PLOT:
        grouped_data = build_grouped_panel_data(tables[dataset])
        plot_grouped_dot_panel(
            grouped_data,
            title="{}".format(dataset),
            output_name="{}_grouped_split_dotplot".format(dataset.lower()),
        )

    print("Read data from: {}".format(EXCEL_PATH))
    print("Plots saved to: {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
