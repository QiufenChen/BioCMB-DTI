#!/usr/bin/env python
from pathlib import Path
import re

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


INPUT_FILE = Path(__file__).resolve().parent / "results" / "source_assisted_learning.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "source_assisted_learning_plots"

METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPR"]
MODELS = ["HpyerAttentionDTI", "BINDTI", "BioFusionDTI", "OurModel"]
MODEL_LABELS = {
    "HpyerAttentionDTI": "HyperAttentionDTI",
    "BINDTI": "BINDTI",
    "BioFusionDTI": "BioFusionDTI",
    "OurModel": "BioCMB-DTI",
}

METHOD_STYLE = {
    "NP-only": {
        "label": "NP-only",
        "color": "#0072B2",
        "marker": "o",
    },
    "np_sdt_joint": {
        "label": "NP + SDT",
        "color": "#D55E00",
        "marker": "s",
    },
    "np_source_similarity_weighted_decay": {
        "label": "Similarity-weighted decay",
        "color": "#009E73",
        "marker": "^",
    },
}

PANEL_ROWS = [
    ("cold_drug_split", "test", "Cold-drug"),
    ("drug_radial_split", "near", "Radial near"),
    ("drug_radial_split", "far", "Radial far"),
    ("drug_radial_split", "near + far", "Radial near + far"),
]


def normalize_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", "_", str(value).strip())


def parse_mean_std(value):
    if pd.isna(value):
        return np.nan, np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value), np.nan
    text = str(value).strip()
    text = text.replace("±", "+/-").replace("+-", "+/-").replace("�", "+/-").replace("��", "+/-")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        return np.nan, np.nan
    mean = float(nums[0])
    std = float(nums[1]) if len(nums) > 1 else np.nan
    return mean, std


def read_sheet(sheet_name):
    df = pd.read_excel(INPUT_FILE, sheet_name=sheet_name)
    df["Training method"] = df["Training method"].ffill()
    df["Model"] = df["Model"].ffill()

    region_col = None
    for col in df.columns:
        if str(col).startswith("Unnamed"):
            region_col = col
            break
    if region_col is None:
        df["Region"] = "test"
    else:
        df = df.rename(columns={region_col: "Region"})
        df["Region"] = df["Region"].ffill()

    df["MethodKey"] = df["Training method"].map(normalize_text)
    for metric in METRICS:
        parsed = df[metric].map(parse_mean_std)
        df[f"{metric}_mean"] = parsed.map(lambda item: item[0])
        df[f"{metric}_std"] = parsed.map(lambda item: item[1])
    return df


def load_all_data():
    frames = []
    for sheet_name in pd.ExcelFile(INPUT_FILE).sheet_names:
        df = read_sheet(sheet_name)
        df["Sheet"] = sheet_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_panel(ax, data, model_name, row_label):
    x = np.arange(len(METRICS))
    model_data = data[data["Model"].astype(str).str.strip() == model_name]

    for method_key, style in METHOD_STYLE.items():
        row = model_data[model_data["MethodKey"] == method_key]
        if row.empty:
            continue
        row = row.iloc[0]
        means = [row[f"{metric}_mean"] for metric in METRICS]
        ax.scatter(
            x,
            means,
            s=58,
            color=style["color"],
            marker=style["marker"],
            edgecolor="white",
            linewidth=0.7,
            label=style["label"],
            zorder=3,
        )

    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, rotation=90)
    ax.set_ylabel("Score" if model_name == MODELS[0] else "")
    if model_name == MODELS[0]:
        ax.text(-0.32, 0.5, row_label, transform=ax.transAxes, rotation=90, va="center", ha="center")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_grid(data, panel_rows, output_stem, figsize):
    if plt is None:
        plot_grid_svg(data, panel_rows, output_stem)
        return

    fig, axes = plt.subplots(len(panel_rows), len(MODELS), figsize=figsize, sharey=True)
    if len(panel_rows) == 1:
        axes = np.asarray([axes])

    for col_idx, model_name in enumerate(MODELS):
        axes[0, col_idx].set_title(MODEL_LABELS.get(model_name, model_name))

    for row_idx, (sheet_name, region, row_label) in enumerate(panel_rows):
        subset = data[(data["Sheet"] == sheet_name) & (data["Region"].astype(str).str.strip() == region)]
        for col_idx, model_name in enumerate(MODELS):
            plot_panel(axes[row_idx, col_idx], subset, model_name, row_label)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=[0.03, 0.03, 1.0, 0.95])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{output_stem}.pdf")
    fig.savefig(OUTPUT_DIR / f"{output_stem}.png", dpi=600)
    plt.close(fig)


def svg_marker(x, y, style):
    color = style["color"]
    marker = style["marker"]
    if marker == "o":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.2" fill="{color}" stroke="white" stroke-width="1"/>'
    if marker == "s":
        size = 9.5
        return f'<rect x="{x - size / 2:.1f}" y="{y - size / 2:.1f}" width="{size:.1f}" height="{size:.1f}" fill="{color}" stroke="white" stroke-width="1"/>'
    if marker == "^":
        size = 11.5
        points = f"{x:.1f},{y - size / 2:.1f} {x - size / 2:.1f},{y + size / 2:.1f} {x + size / 2:.1f},{y + size / 2:.1f}"
        return f'<polygon points="{points}" fill="{color}" stroke="white" stroke-width="1"/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.2" fill="{color}" stroke="white" stroke-width="1"/>'


def plot_grid_svg(data, panel_rows, output_stem):
    panel_w, panel_h = 260, 190
    left_margin, top_margin = 100, 82
    col_gap, row_gap = 34, 44
    right_margin, bottom_margin = 32, 70
    width = left_margin + len(MODELS) * panel_w + (len(MODELS) - 1) * col_gap + right_margin
    height = top_margin + len(panel_rows) * panel_h + (len(panel_rows) - 1) * row_gap + bottom_margin
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, Helvetica, sans-serif; fill:#222;} .small{font-size:12px;} .label{font-size:14px;} .title{font-size:16px; font-weight:700;} .tick{font-size:11px;}</style>',
    ]

    legend_x = left_margin
    for idx, style in enumerate(METHOD_STYLE.values()):
        x = legend_x + idx * 210
        y = 30
        parts.append(svg_marker(x, y, style))
        parts.append(f'<text class="label" x="{x + 14}" y="{y + 5}">{style["label"]}</text>')

    for col_idx, model_name in enumerate(MODELS):
        x0 = left_margin + col_idx * (panel_w + col_gap)
        parts.append(f'<text class="title" x="{x0 + panel_w / 2:.1f}" y="64" text-anchor="middle">{MODEL_LABELS.get(model_name, model_name)}</text>')

    for row_idx, (sheet_name, region, row_label) in enumerate(panel_rows):
        y0 = top_margin + row_idx * (panel_h + row_gap)
        parts.append(f'<text class="title" transform="translate(31,{y0 + panel_h / 2:.1f}) rotate(-90)" text-anchor="middle">{row_label}</text>')
        subset = data[(data["Sheet"] == sheet_name) & (data["Region"].astype(str).str.strip() == region)]

        for col_idx, model_name in enumerate(MODELS):
            x0 = left_margin + col_idx * (panel_w + col_gap)
            plot_x0, plot_y0 = x0 + 34, y0 + 16
            plot_w, plot_h = panel_w - 48, panel_h - 58

            parts.append(f'<line x1="{plot_x0}" y1="{plot_y0 + plot_h}" x2="{plot_x0 + plot_w}" y2="{plot_y0 + plot_h}" stroke="#333" stroke-width="1"/>')
            parts.append(f'<line x1="{plot_x0}" y1="{plot_y0}" x2="{plot_x0}" y2="{plot_y0 + plot_h}" stroke="#333" stroke-width="1"/>')
            for tick in [0.0, 0.5, 1.0]:
                ty = plot_y0 + plot_h * (1 - tick)
                parts.append(f'<line x1="{plot_x0 - 4}" y1="{ty:.1f}" x2="{plot_x0}" y2="{ty:.1f}" stroke="#333" stroke-width="1"/>')
                if col_idx == 0:
                    parts.append(f'<text class="tick" x="{plot_x0 - 8}" y="{ty + 4:.1f}" text-anchor="end">{tick:.1f}</text>')

            metric_x = np.linspace(plot_x0 + 11, plot_x0 + plot_w - 11, len(METRICS))
            for mx, metric in zip(metric_x, METRICS):
                parts.append(f'<text class="tick" transform="translate({mx:.1f},{plot_y0 + plot_h + 12:.1f}) rotate(90)" text-anchor="start">{metric}</text>')

            model_data = subset[subset["Model"].astype(str).str.strip() == model_name]
            for method_key, style in METHOD_STYLE.items():
                row = model_data[model_data["MethodKey"] == method_key]
                if row.empty:
                    continue
                row = row.iloc[0]
                for mx, metric in zip(metric_x, METRICS):
                    value = row[f"{metric}_mean"]
                    if pd.isna(value):
                        continue
                    my = plot_y0 + plot_h * (1 - max(0, min(1, value)))
                    parts.append(svg_marker(mx, my, style))

    parts.append("</svg>")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"{output_stem}.svg").write_text("\n".join(parts), encoding="utf-8")


def main():
    data = load_all_data()
    plot_grid(
        data,
        PANEL_ROWS,
        "source_assisted_all_models_all_subsets_points",
        figsize=(12.5, 10.0),
    )
    plot_grid(
        data,
        [PANEL_ROWS[0]],
        "source_assisted_all_models_cold_drug_points",
        figsize=(12.5, 3.5),
    )
    plot_grid(
        data,
        PANEL_ROWS[1:],
        "source_assisted_all_models_radial_points",
        figsize=(12.5, 7.8),
    )
    print(f"Plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
