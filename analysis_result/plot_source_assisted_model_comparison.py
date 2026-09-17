#!/usr/bin/env python
from pathlib import Path
import re

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
except ImportError:
    plt = None


INPUT_FILE = Path(__file__).resolve().parent / "results" / "source_assisted_learning.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "source_assisted_learning_plots_V1"

METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPR"]
MODELS = ["HpyerAttentionDTI", "BINDTI", "BioFusionDTI", "OurModel"]
MODEL_LABELS = {
    "HpyerAttentionDTI": "HyperAttentionDTI",
    "BINDTI": "BINDTI",
    "BioFusionDTI": "BioFusionDTI",
    "OurModel": "BioCMB-DTI",
}
PANEL_ROWS = [
    ("cold_drug_split", "test", "Cold-drug"),
    ("drug_radial_split", "near", "Radial near"),
    ("drug_radial_split", "far", "Radial far"),
    ("drug_radial_split", "near + far", "Radial near + far"),
]
SOURCE_METHOD = "np_source_similarity_weighted_decay"
BASE_METHOD = "NP-only"
BIOCMB_MODEL = "OurModel"
DELTA_METHODS = {
    "np_sdt_joint": {"label": "NP + SDT", "color": "#D55E00"},
    "np_source_similarity_weighted_decay": {"label": "Similarity-weighted decay", "color": "#009E73"},
}

MODEL_STYLE = {
    "HpyerAttentionDTI": {"color": "#CC79A7", "marker": "D", "alpha": 0.9, "size": 48},
    "BINDTI": {"color": "#56B4E9", "marker": "v", "alpha": 0.9, "size": 54},
    "BioFusionDTI": {"color": "#009E73", "marker": "^", "alpha": 0.95, "size": 58},
    "OurModel": {"color": "#D55E00", "marker": "s", "alpha": 1.0, "size": 70},
}

ABSOLUTE_METHODS = {
    "NP-only": {"label": "NP-only", "stem": "np_only"},
    "np_sdt_joint": {"label": "NP + SDT", "stem": "np_sdt_joint"},
    "np_source_similarity_weighted_decay": {"label": "Similarity-weighted decay", "stem": "similarity_weighted_decay"},
}


def normalize_text(value):
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", "_", str(value).strip())
    return re.sub(r"_+", "_", text)


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
    return float(nums[0]), float(nums[1]) if len(nums) > 1 else np.nan


def read_sheet(sheet_name):
    df = pd.read_excel(INPUT_FILE, sheet_name=sheet_name)
    df["Training method"] = df["Training method"].ffill()
    df["Model"] = df["Model"].ffill()

    region_col = next((col for col in df.columns if str(col).startswith("Unnamed")), None)
    if region_col is None:
        df["Region"] = "test"
    else:
        df = df.rename(columns={region_col: "Region"})
        df["Region"] = df["Region"].ffill()

    df["MethodKey"] = df["Training method"].map(normalize_text)
    for metric in METRICS:
        parsed = df[metric].map(parse_mean_std)
        df[f"{metric}_mean"] = parsed.map(lambda item: item[0])
    return df


def load_data():
    frames = []
    for sheet_name in pd.ExcelFile(INPUT_FILE).sheet_names:
        df = read_sheet(sheet_name)
        df["Sheet"] = sheet_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def get_row(data, sheet_name, region, model_name, method_key):
    subset = data[
        (data["Sheet"] == sheet_name)
        & (data["Region"].astype(str).str.strip() == region)
        & (data["Model"].astype(str).str.strip() == model_name)
        & (data["MethodKey"] == method_key)
    ]
    if subset.empty:
        return None
    return subset.iloc[0]


def plot_absolute_method_matplotlib(data, method_key, method_label, output_stem):
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2), sharey=True)
    axes = axes.ravel()
    x = np.arange(len(METRICS))
    model_offsets = np.linspace(-0.18, 0.18, len(MODELS))

    for ax, (sheet_name, region, title) in zip(axes, PANEL_ROWS):
        for model_name, model_offset in zip(MODELS, model_offsets):
            style = MODEL_STYLE[model_name]
            row = get_row(data, sheet_name, region, model_name, method_key)
            if row is None:
                continue
            means = [row[f"{metric}_mean"] for metric in METRICS]
            ax.scatter(
                x + model_offset,
                means,
                color=style["color"],
                marker=style["marker"],
                s=style["size"],
                alpha=style["alpha"],
                edgecolor="white",
                linewidth=0.7,
                zorder=4 if model_name == "OurModel" else 2,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(METRICS, rotation=90)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Score")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    model_handles = [
        plt.Line2D([0], [0], marker=MODEL_STYLE[m]["marker"], linestyle="", color=MODEL_STYLE[m]["color"], label=MODEL_LABELS[m], markersize=8)
        for m in MODELS
    ]
    fig.suptitle(method_label, y=0.995)
    fig.legend(handles=model_handles, loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUTPUT_DIR / f"source_assisted_absolute_{output_stem}_model_comparison.pdf")
    fig.savefig(OUTPUT_DIR / f"source_assisted_absolute_{output_stem}_model_comparison.png", dpi=600)
    plt.close(fig)


def plot_absolute_matplotlib(data):
    for method_key, method_style in ABSOLUTE_METHODS.items():
        plot_absolute_method_matplotlib(data, method_key, method_style["label"], method_style["stem"])


def plot_delta_matplotlib(data):
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.9))
    axes = axes.ravel()
    cmap = LinearSegmentedColormap.from_list("delta", ["#B2182B", "#FFFFFF", "#2166AC"])

    all_deltas = []
    matrices = []
    for sheet_name, region, title in PANEL_ROWS:
        matrix = []
        for model_name in MODELS:
            base = get_row(data, sheet_name, region, model_name, BASE_METHOD)
            source = get_row(data, sheet_name, region, model_name, SOURCE_METHOD)
            values = []
            for metric in METRICS:
                delta = np.nan
                if base is not None and source is not None:
                    delta = source[f"{metric}_mean"] - base[f"{metric}_mean"]
                    all_deltas.append(delta)
                values.append(delta)
            matrix.append(values)
        matrices.append((title, np.asarray(matrix, dtype=float)))

    max_abs = max(abs(np.nanmin(all_deltas)), abs(np.nanmax(all_deltas)), 0.01)
    norm = TwoSlopeNorm(vcenter=0, vmin=-max_abs, vmax=max_abs)

    for ax, (title, matrix) in zip(axes, matrices):
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(METRICS)))
        ax.set_xticklabels(METRICS, rotation=90)
        ax.set_yticks(np.arange(len(MODELS)))
        ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if np.isnan(value):
                    continue
                ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=8)
        ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("Delta score vs NP-only")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "source_assisted_delta_vs_np_only_heatmap.pdf")
    fig.savefig(OUTPUT_DIR / "source_assisted_delta_vs_np_only_heatmap.png", dpi=600)
    plt.close(fig)


def plot_biocmb_delta_matplotlib(data):
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.9), sharey=True)
    axes = axes.ravel()
    x = np.arange(len(METRICS))
    width = 0.34
    offsets = [-width / 2, width / 2]

    for ax, (sheet_name, region, title) in zip(axes, PANEL_ROWS):
        base = get_row(data, sheet_name, region, BIOCMB_MODEL, BASE_METHOD)
        ax.axhline(0, color="#444444", linewidth=1)
        for offset, (method_key, style) in zip(offsets, DELTA_METHODS.items()):
            source = get_row(data, sheet_name, region, BIOCMB_MODEL, method_key)
            deltas = []
            for metric in METRICS:
                delta = np.nan
                if base is not None and source is not None:
                    delta = source[f"{metric}_mean"] - base[f"{metric}_mean"]
                deltas.append(delta)
            bars = ax.bar(x + offset, deltas, width=width * 0.9, color=style["color"], alpha=0.82, label=style["label"])
            for bar, delta in zip(bars, deltas):
                if np.isnan(delta):
                    continue
                va = "bottom" if delta >= 0 else "top"
                dy = 0.006 if delta >= 0 else -0.006
                ax.text(bar.get_x() + bar.get_width() / 2, delta + dy, f"{delta:+.3f}", ha="center", va=va, fontsize=8)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(METRICS, rotation=90)
        ax.set_ylabel("Delta score vs NP-only")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_DIR / "source_assisted_biocmb_delta_methods.pdf")
    fig.savefig(OUTPUT_DIR / "source_assisted_biocmb_delta_methods.png", dpi=600)
    plt.close(fig)


def svg_marker(x, y, style, marker=None, size=4.5):
    color = style["color"]
    marker = marker or style.get("marker", "o")
    if marker == "o":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size:.1f}" fill="{color}" stroke="white" stroke-width="1"/>'
    if marker == "s":
        return f'<rect x="{x - size:.1f}" y="{y - size:.1f}" width="{size * 2:.1f}" height="{size * 2:.1f}" fill="{color}" stroke="white" stroke-width="1"/>'
    if marker == "^":
        return f'<polygon points="{x:.1f},{y - size:.1f} {x - size:.1f},{y + size:.1f} {x + size:.1f},{y + size:.1f}" fill="{color}" stroke="white" stroke-width="1"/>'
    return f'<polygon points="{x:.1f},{y - size:.1f} {x - size:.1f},{y:.1f} {x:.1f},{y + size:.1f} {x + size:.1f},{y:.1f}" fill="{color}" stroke="white" stroke-width="1"/>'


def plot_absolute_method_svg(data, method_key, method_label, output_stem):
    panel_w, panel_h = 420, 255
    left, top, gap_x, gap_y = 74, 78, 60, 72
    width = left * 2 + panel_w * 2 + gap_x
    height = top + panel_h * 2 + gap_y + 86
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, Helvetica, sans-serif; fill:#222;} .title{font-size:18px;font-weight:700;} .label{font-size:14px;} .tick{font-size:12px;}</style>',
        f'<text class="title" x="{width / 2:.1f}" y="24" text-anchor="middle">{method_label}</text>',
    ]

    for idx, model_name in enumerate(MODELS):
        style = MODEL_STYLE[model_name]
        x = left + idx * 210
        y = 52
        parts.append(svg_marker(x, y, style, size=4.8 if model_name != "OurModel" else 6.2))
        parts.append(f'<text class="label" x="{x + 14}" y="{y + 5}">{MODEL_LABELS[model_name]}</text>')

    for panel_idx, (sheet_name, region, title) in enumerate(PANEL_ROWS):
        row_idx, col_idx = divmod(panel_idx, 2)
        x0 = left + col_idx * (panel_w + gap_x)
        y0 = top + row_idx * (panel_h + gap_y)
        plot_x, plot_y = x0 + 48, y0 + 30
        plot_w, plot_h = panel_w - 72, panel_h - 82
        metric_x = np.linspace(plot_x + 14, plot_x + plot_w - 14, len(METRICS))

        parts.append(f'<text class="title" x="{x0 + panel_w / 2:.1f}" y="{y0 + 4}" text-anchor="middle">{title}</text>')
        parts.append(f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#333"/>')
        parts.append(f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#333"/>')
        for tick in [0, 0.5, 1.0]:
            ty = plot_y + plot_h * (1 - tick)
            parts.append(f'<line x1="{plot_x - 4}" y1="{ty:.1f}" x2="{plot_x}" y2="{ty:.1f}" stroke="#333"/>')
            parts.append(f'<text class="tick" x="{plot_x - 8}" y="{ty + 4:.1f}" text-anchor="end">{tick:.1f}</text>')
        for mx, metric in zip(metric_x, METRICS):
            parts.append(f'<text class="tick" transform="translate({mx:.1f},{plot_y + plot_h + 14:.1f}) rotate(90)" text-anchor="start">{metric}</text>')

        model_offsets = np.linspace(-18, 18, len(MODELS))
        for model_name, model_offset in zip(MODELS, model_offsets):
            style = MODEL_STYLE[model_name]
            size = 4.4 if model_name != "OurModel" else 6.0
            row = get_row(data, sheet_name, region, model_name, method_key)
            if row is None:
                continue
            for mx, metric in zip(metric_x, METRICS):
                value = row[f"{metric}_mean"]
                if pd.isna(value):
                    continue
                px = mx + model_offset
                my = plot_y + plot_h * (1 - max(0, min(1, value)))
                parts.append(svg_marker(px, my, style, size=size))

    parts.append("</svg>")
    (OUTPUT_DIR / f"source_assisted_absolute_{output_stem}_model_comparison.svg").write_text("\n".join(parts), encoding="utf-8")


def plot_absolute_svg(data):
    for method_key, method_style in ABSOLUTE_METHODS.items():
        plot_absolute_method_svg(data, method_key, method_style["label"], method_style["stem"])


def delta_color(value, max_abs):
    if pd.isna(value):
        return "#F5F5F5"
    value = max(-max_abs, min(max_abs, value))
    if value >= 0:
        t = value / max_abs
        r = int(255 * (1 - t) + 33 * t)
        g = int(255 * (1 - t) + 102 * t)
        b = int(255 * (1 - t) + 172 * t)
    else:
        t = -value / max_abs
        r = int(255 * (1 - t) + 178 * t)
        g = int(255 * (1 - t) + 24 * t)
        b = int(255 * (1 - t) + 43 * t)
    return f"#{r:02X}{g:02X}{b:02X}"


def plot_delta_svg(data):
    cell_w, cell_h = 68, 34
    panel_w = 170 + cell_w * len(METRICS)
    panel_h = 58 + cell_h * len(MODELS)
    left, top, gap_x, gap_y = 34, 50, 42, 54
    width = left * 2 + panel_w * 2 + gap_x
    height = top + panel_h * 2 + gap_y + 30

    matrices, all_deltas = [], []
    for sheet_name, region, title in PANEL_ROWS:
        matrix = []
        for model_name in MODELS:
            base = get_row(data, sheet_name, region, model_name, BASE_METHOD)
            source = get_row(data, sheet_name, region, model_name, SOURCE_METHOD)
            values = []
            for metric in METRICS:
                delta = np.nan
                if base is not None and source is not None:
                    delta = source[f"{metric}_mean"] - base[f"{metric}_mean"]
                    all_deltas.append(delta)
                values.append(delta)
            matrix.append(values)
        matrices.append((title, np.asarray(matrix)))

    max_abs = max(abs(np.nanmin(all_deltas)), abs(np.nanmax(all_deltas)), 0.01)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, Helvetica, sans-serif; fill:#222;} .title{font-size:17px;font-weight:700;} .label{font-size:12px;} .value{font-size:11px;}</style>',
    ]
    for panel_idx, (title, matrix) in enumerate(matrices):
        row_idx, col_idx = divmod(panel_idx, 2)
        x0 = left + col_idx * (panel_w + gap_x)
        y0 = top + row_idx * (panel_h + gap_y)
        parts.append(f'<text class="title" x="{x0 + panel_w / 2:.1f}" y="{y0 - 12}" text-anchor="middle">{title}</text>')
        for j, metric in enumerate(METRICS):
            x = x0 + 150 + j * cell_w + cell_w / 2
            parts.append(f'<text class="label" x="{x:.1f}" y="{y0 + 16}" text-anchor="middle">{metric}</text>')
        for i, model_name in enumerate(MODELS):
            y = y0 + 30 + i * cell_h
            parts.append(f'<text class="label" x="{x0 + 144}" y="{y + 22}" text-anchor="end">{MODEL_LABELS[model_name]}</text>')
            for j, value in enumerate(matrix[i]):
                x = x0 + 150 + j * cell_w
                fill = delta_color(value, max_abs)
                parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="white"/>')
                if not pd.isna(value):
                    parts.append(f'<text class="value" x="{x + cell_w / 2:.1f}" y="{y + 21:.1f}" text-anchor="middle">{value:+.3f}</text>')
    parts.append("</svg>")
    (OUTPUT_DIR / "source_assisted_delta_vs_np_only_heatmap.svg").write_text("\n".join(parts), encoding="utf-8")


def plot_biocmb_delta_svg(data):
    panel_w, panel_h = 430, 250
    left, top, gap_x, gap_y = 80, 78, 60, 72
    width = left * 2 + panel_w * 2 + gap_x
    height = top + panel_h * 2 + gap_y + 70
    y_min, y_max = -0.18, 0.28
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, Helvetica, sans-serif; fill:#222;} .title{font-size:18px;font-weight:700;} .label{font-size:14px;} .tick{font-size:12px;} .value{font-size:10px;}</style>',
    ]

    legend_x = left
    for idx, style in enumerate(DELTA_METHODS.values()):
        x = legend_x + idx * 200
        y = 30
        parts.append(f'<rect x="{x:.1f}" y="{y - 6:.1f}" width="12" height="12" fill="{style["color"]}" opacity="0.82"/>')
        parts.append(f'<text class="label" x="{x + 18}" y="{y + 5}">{style["label"]}</text>')

    for panel_idx, (sheet_name, region, title) in enumerate(PANEL_ROWS):
        row_idx, col_idx = divmod(panel_idx, 2)
        x0 = left + col_idx * (panel_w + gap_x)
        y0 = top + row_idx * (panel_h + gap_y)
        plot_x, plot_y = x0 + 52, y0 + 28
        plot_w, plot_h = panel_w - 82, panel_h - 82
        zero_y = plot_y + plot_h * (1 - (0 - y_min) / (y_max - y_min))
        metric_x = np.linspace(plot_x + 22, plot_x + plot_w - 22, len(METRICS))

        parts.append(f'<text class="title" x="{x0 + panel_w / 2:.1f}" y="{y0 + 4}" text-anchor="middle">{title}</text>')
        parts.append(f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#333"/>')
        parts.append(f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#333"/>')
        parts.append(f'<line x1="{plot_x}" y1="{zero_y:.1f}" x2="{plot_x + plot_w}" y2="{zero_y:.1f}" stroke="#444" stroke-width="1"/>')

        for tick in [-0.1, 0.0, 0.1, 0.2]:
            ty = plot_y + plot_h * (1 - (tick - y_min) / (y_max - y_min))
            parts.append(f'<line x1="{plot_x - 4}" y1="{ty:.1f}" x2="{plot_x}" y2="{ty:.1f}" stroke="#333"/>')
            parts.append(f'<text class="tick" x="{plot_x - 8}" y="{ty + 4:.1f}" text-anchor="end">{tick:+.1f}</text>')

        for mx, metric in zip(metric_x, METRICS):
            parts.append(f'<text class="tick" transform="translate({mx:.1f},{plot_y + plot_h + 14:.1f}) rotate(90)" text-anchor="start">{metric}</text>')

        base = get_row(data, sheet_name, region, BIOCMB_MODEL, BASE_METHOD)
        bar_w = 14
        offsets = [-bar_w / 2 - 2, bar_w / 2 + 2]
        for offset, (method_key, style) in zip(offsets, DELTA_METHODS.items()):
            source = get_row(data, sheet_name, region, BIOCMB_MODEL, method_key)
            for mx, metric in zip(metric_x, METRICS):
                if base is None or source is None:
                    continue
                delta = source[f"{metric}_mean"] - base[f"{metric}_mean"]
                delta_y = plot_y + plot_h * (1 - (delta - y_min) / (y_max - y_min))
                y_top = min(delta_y, zero_y)
                height_bar = abs(delta_y - zero_y)
                parts.append(f'<rect x="{mx + offset - bar_w / 2:.1f}" y="{y_top:.1f}" width="{bar_w}" height="{height_bar:.1f}" fill="{style["color"]}" opacity="0.82"/>')
                va_y = y_top - 4 if delta >= 0 else y_top + height_bar + 11
                parts.append(f'<text class="value" x="{mx + offset:.1f}" y="{va_y:.1f}" text-anchor="middle">{delta:+.3f}</text>')

    parts.append("</svg>")
    (OUTPUT_DIR / "source_assisted_biocmb_delta_methods.svg").write_text("\n".join(parts), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    if plt is not None:
        plot_absolute_matplotlib(data)
        plot_delta_matplotlib(data)
        plot_biocmb_delta_matplotlib(data)
    else:
        plot_absolute_svg(data)
        plot_delta_svg(data)
        plot_biocmb_delta_svg(data)

    print(f"Plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
