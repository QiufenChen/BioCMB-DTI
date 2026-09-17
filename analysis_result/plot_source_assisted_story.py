#!/usr/bin/env python
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = Path(__file__).resolve().parent / "results" / "source_assisted_learning.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "source_assisted_learning_plots"

METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPR"]
DELTA_METRICS = ["F1", "AUROC", "AUPR"]

PANELS = [
    ("cold_drug_split", "test", "Cold-drug split", "cold_drug"),
    ("drug_radial_split", "near", "SDT-referenced radial split (near)", "radial_near"),
    ("drug_radial_split", "far", "SDT-referenced radial split (far)", "radial_far"),
    ("drug_radial_split", "near + far", "SDT-referenced radial split (near + far)", "radial_near_far"),
]

MODELS = ["HpyerAttentionDTI", "BINDTI", "BioFusionDTI", "OurModel"]
MODEL_LABELS = {
    "HpyerAttentionDTI": "HyperAttentionDTI",
    "BINDTI": "BINDTI",
    "BioFusionDTI": "BioFusionDTI",
    "OurModel": "BioCMB-DTI",
}
MODEL_STYLE = {
    "HpyerAttentionDTI": {"color": "#D55E00", "marker": "o", "size": 120},
    "BINDTI": {"color": "#0072B2", "marker": "s", "size": 120},
    "BioFusionDTI": {"color": "#009E73", "marker": "^", "size": 140},
    "OurModel": {"color": "#8E63C7", "marker": "D", "size": 120},
}

BIOCMB_MODEL = "OurModel"
BASE_METHOD = "NP-only"
METHODS = ["NP-only", "np_sdt_joint", "np_source_similarity_weighted_decay"]
METHOD_LABELS = {
    "NP-only": "NP-only",
    "np_sdt_joint": "NP + SDT",
    "np_source_similarity_weighted_decay": "Similarity-weighted decay",
}

METHOD_STYLE = {
    "NP-only": {"color": "#B0B0B0", "marker": "o", "size": 110},
    "np_sdt_joint": {"color": "#0072B2", "marker": "v", "size": 120},
    "np_source_similarity_weighted_decay": {"color": "#D55E00", "marker": "X", "size": 140},
}

DELTA_STYLE = {"F1": "#009E73", "AUROC": "#0072B2", "AUPR": "#D55E00"}


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
    df["Region"] = "test" if region_col is None else df[region_col]
    df["MethodKey"] = df["Training method"].map(normalize_text)
    for metric in METRICS:
        parsed = df[metric].map(parse_mean_std)
        df[f"{metric}_mean"] = parsed.map(lambda item: item[0])
    df["Sheet"] = sheet_name
    return df


def load_data():
    sheets = pd.ExcelFile(INPUT_FILE).sheet_names
    return pd.concat([read_sheet(sheet) for sheet in sheets], ignore_index=True)


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


def format_score_plot(title):
    plt.title(title)
    plt.ylim(0.0, 0.7)
    plt.xticks(np.arange(len(METRICS)), METRICS, rotation=90)
    plt.ylabel("Score (mean)")
    plt.grid(False)


def save_plot(stem):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{stem}.pdf")
    plt.savefig(OUTPUT_DIR / f"{stem}.png", dpi=600)
    plt.close()


def plot_np_only_model_scatter(data, sheet_name, region, title, stem):
    plt.figure(figsize=(5, 5))
    x = np.arange(len(METRICS))
    for model_name in MODELS:
        row = get_row(data, sheet_name, region, model_name, BASE_METHOD)
        if row is None:
            continue
        style = MODEL_STYLE[model_name]
        values = [row[f"{metric}_mean"] for metric in METRICS]
        plt.scatter(x, values, color=style["color"], marker=style["marker"], s=style["size"], label=MODEL_LABELS[model_name], edgecolor="white", linewidth=0.7)
    format_score_plot(title)
    plt.legend(frameon=True)
    save_plot(f"story_np_only_model_scatter_{stem}")


def plot_biocmb_strategy_scatter(data, sheet_name, region, title, stem):
    plt.figure(figsize=(5, 5))
    x = np.arange(len(METRICS))
    for method_key in METHODS:
        row = get_row(data, sheet_name, region, BIOCMB_MODEL, method_key)
        if row is None:
            continue
        style = METHOD_STYLE[method_key]
        values = [row[f"{metric}_mean"] for metric in METRICS]
        plt.scatter(x, values, color=style["color"], marker=style["marker"], s=style["size"], label=METHOD_LABELS[method_key], edgecolor="none", linewidth=0, alpha=0.85)
    format_score_plot(title)
    plt.legend(frameon=False)
    save_plot(f"story_biocmb_strategy_scatter_{stem}")


def plot_biocmb_delta_bars(data, sheet_name, region, title, stem):
    plt.figure(figsize=(5, 5))
    methods = ["np_sdt_joint", "np_source_similarity_weighted_decay"]
    y_positions = np.arange(len(methods))
    metric_offsets = np.linspace(-0.22, 0.22, len(DELTA_METRICS))
    base = get_row(data, sheet_name, region, BIOCMB_MODEL, BASE_METHOD)

    plt.axvline(0, color="#444444", linewidth=1.2)
    for metric, offset in zip(DELTA_METRICS, metric_offsets):
        values = []
        for method_key in methods:
            row = get_row(data, sheet_name, region, BIOCMB_MODEL, method_key)
            delta = np.nan
            if base is not None and row is not None:
                delta = row[f"{metric}_mean"] - base[f"{metric}_mean"]
            values.append(delta)
        bars = plt.barh(y_positions + offset, values, height=0.18, color=DELTA_STYLE[metric], label=metric)
        for bar, value in zip(bars, values):
            if pd.isna(value):
                continue
            ha = "left" if value >= 0 else "right"
            dx = 0.004 if value >= 0 else -0.004
            plt.text(value + dx, bar.get_y() + bar.get_height() / 2, f"{value:+.3f}", ha=ha, va="center")

    plt.title(title)
    plt.yticks(y_positions, [METHOD_LABELS[m] for m in methods])
    plt.xlabel("Delta score vs NP-only")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.grid(False)
    plt.legend(frameon=False)
    save_plot(f"story_biocmb_delta_bars_{stem}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    for sheet_name, region, title, stem in PANELS:
        plot_np_only_model_scatter(data, sheet_name, region, title, stem)
        plot_biocmb_strategy_scatter(data, sheet_name, region, title, stem)
        plot_biocmb_delta_bars(data, sheet_name, region, title, stem)
    print(f"Plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
