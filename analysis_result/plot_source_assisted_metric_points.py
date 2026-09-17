#!/usr/bin/env python
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILE = Path(__file__).resolve().parent / "results" / "source_assisted_learning.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "source_assisted_learning_plots"

MODEL_TO_PLOT = "OurModel"

METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPR"]

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

SHEET_TITLES = {
    "cold_drug_split": "Cold-drug split",
    "drug_radial_split": "SDT-referenced drug-radial split",
}


def normalize_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", "_", str(value).strip())


def display_text(value):
    return str(value).replace("\n", " ").replace("_", " ").strip()


def parse_mean_std(value):
    if pd.isna(value):
        return np.nan, np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value), np.nan

    text = str(value).strip()
    text = text.replace("±", "+/-").replace("＋/－", "+/-")
    text = text.replace("+-", "+/-").replace("�", "+/-").replace("��", "+/-")
    text = text.replace("卤", "+/-")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        return np.nan, np.nan
    mean = float(nums[0])
    std = float(nums[1]) if len(nums) > 1 else np.nan
    return mean, std


def read_result_sheet(sheet_name):
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
    df["MethodLabel"] = df["MethodKey"].map(
        lambda x: METHOD_STYLE.get(x, {"label": display_text(x)})["label"]
    )

    for metric in METRICS:
        parsed = df[metric].map(parse_mean_std)
        df[f"{metric}_mean"] = parsed.map(lambda x: x[0])
        df[f"{metric}_std"] = parsed.map(lambda x: x[1])

    return df[df["Model"].astype(str).str.strip() == MODEL_TO_PLOT].copy()


def plot_metric_points(data, title, output_stem):
    x = np.arange(len(METRICS))

    plt.figure(figsize=(6.4, 4.2))
    for method_key, style in METHOD_STYLE.items():
        row = data[data["MethodKey"] == method_key]
        if row.empty:
            continue
        row = row.iloc[0]
        means = [row[f"{metric}_mean"] for metric in METRICS]
        plt.scatter(
            x,
            means,
            s=80,
            color=style["color"],
            marker=style["marker"],
            edgecolor="white",
            linewidth=0.8,
            label=style["label"],
            zorder=3,
        )

    plt.xticks(x, METRICS, rotation=90)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.legend(frameon=False)
    plt.box(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{output_stem}_metric_points.pdf")
    plt.savefig(OUTPUT_DIR / f"{output_stem}_metric_points.png", dpi=600)
    plt.close()


def plot_delta_bars(data, title, output_stem):
    base = data[data["MethodKey"] == "NP-only"]
    if base.empty:
        return

    base = base.iloc[0]
    x = np.arange(len(METRICS))
    methods = [method for method in METHOD_STYLE if method != "NP-only"]
    width = 0.34 if len(methods) == 2 else 0.25
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width

    plt.figure(figsize=(6.4, 4.2))
    plt.axhline(0, color="#444444", linewidth=1.0)

    for offset, method_key in zip(offsets, methods):
        style = METHOD_STYLE[method_key]
        row = data[data["MethodKey"] == method_key]
        if row.empty:
            continue
        row = row.iloc[0]
        deltas = [row[f"{metric}_mean"] - base[f"{metric}_mean"] for metric in METRICS]
        bars = plt.bar(
            x + offset,
            deltas,
            width=width * 0.9,
            color=style["color"],
            alpha=0.78,
            label=style["label"],
            zorder=3,
        )
        for bar, delta in zip(bars, deltas):
            if np.isnan(delta):
                continue
            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            dy = 0.006 if y >= 0 else -0.006
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                y + dy,
                f"{delta:+.3f}",
                ha="center",
                va=va,
                fontsize=9,
            )

    plt.xticks(x, METRICS, rotation=90)
    plt.ylabel(r"$\Delta$ score vs NP-only")
    plt.title(title)
    plt.legend(frameon=False)
    plt.box(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{output_stem}_delta_bars.pdf")
    plt.savefig(OUTPUT_DIR / f"{output_stem}_delta_bars.png", dpi=600)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sheet_name in pd.ExcelFile(INPUT_FILE).sheet_names:
        sheet_data = read_result_sheet(sheet_name)
        sheet_title = SHEET_TITLES.get(sheet_name, display_text(sheet_name))

        for region, region_data in sheet_data.groupby("Region", sort=False):
            region_clean = normalize_text(region).lower()
            title = sheet_title if region_clean == "test" else f"{sheet_title}: {display_text(region)}"
            output_stem = f"{sheet_name}_{region_clean}_{MODEL_TO_PLOT}"
            plot_metric_points(region_data, title, output_stem)
            plot_delta_bars(region_data, title, output_stem)

    print(f"Plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
