#!/usr/bin/env python
from pathlib import Path
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILE = Path(__file__).resolve().parent / "results" / "model_performance_radial.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "model_performance_radial_plots"

# DRH、TTD 跑完后，把这里改成 "DRH" 或 "TTD" 即可。
SHEET_NAME = "SNAP"

METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPR"]

# 每个 split/region 单独绘制一张图。
PLOT_SUBSETS = [
    ("drug radial split", "Drug radial", "near"),
    ("drug radial split", "Drug radial", "far"),
    ("prot radial split", "Protein radial", "near"),
    ("prot radial split", "Protein radial", "far"),
]

# 图例顺序和颜色都可以在这里直接修改。
MODEL_ORDER = [
    "HyperAttentionDTI",
    "DrugBAN",
    "BINDTI",
    "BioFusionDTI",
    "BioCMB-DTI",
]

MODEL_COLORS = {
    "HyperAttentionDTI": "#1F77B4",
    "DrugBAN": "#D62728",
    "BINDTI": "#FF7F0E",
    "BioFusionDTI": "#2CA02C",
    "BioCMB-DTI": "#9467BD",
}


# MODEL_COLORS = {
# "HyperAttentionDTI": "#1F77B4",
# "BINDTI": "#FF7F0E",
# "DrugBAN": "#D62728",
# "BioFusionDTI": "#2CA02C",
# "MIF-DTI": "#8C564B",
# "MML-DTI": "#E377C2",
# "GraphBAN": "#7F7F7F",
# "SaeGraphDTI": "#17BECF",
# "BioCMB-DTI": "#7A45AC",
# }


def normalize_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def parse_mean_std(value):
    """将 '0.808±0.007' 拆成 mean=0.808、std=0.007。"""
    if pd.isna(value):
        return np.nan, np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value), np.nan

    numbers = re.findall(r"-?\d+(?:\.\d+)?", str(value))
    if not numbers:
        return np.nan, np.nan

    mean = float(numbers[0])
    std = float(numbers[1]) if len(numbers) > 1 else np.nan
    return mean, std


def find_section_row(raw, section_name):
    target = normalize_text(section_name)
    matches = raw.index[raw.iloc[:, 0].map(normalize_text) == target].tolist()
    if len(matches) != 1:
        raise ValueError(
            f"Expected one '{section_name}' section in sheet '{SHEET_NAME}', "
            f"but found {len(matches)}."
        )
    return matches[0]


def read_radial_sheet():
    """读取同一 sheet 中的 drug radial 和 protein radial 两个表块。"""
    raw = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME,
        header=None,
        engine="openpyxl",
    )

    section_names = ["drug radial split", "prot radial split"]
    section_rows = [find_section_row(raw, name) for name in section_names]
    frames = []

    for section_index, (section_name, header_row) in enumerate(
        zip(section_names, section_rows)
    ):
        next_header = (
            section_rows[section_index + 1]
            if section_index + 1 < len(section_rows)
            else len(raw)
        )

        block = raw.iloc[
            header_row + 1 : next_header,
            : 2 + len(METRICS),
        ].copy()
        block = block.dropna(how="all")
        block.columns = ["Model", "Region", *METRICS]

        # Excel 中每个模型名只在 near 行出现，因此向下填充到 far 行。
        block["Model"] = block["Model"].ffill().astype(str).str.strip()
        block["Region"] = block["Region"].astype(str).str.strip().str.lower()
        block = block[block["Region"].isin(["near", "far"])].copy()
        block["Split"] = section_name

        for metric in METRICS:
            parsed = block[metric].map(parse_mean_std)
            block[f"{metric}_mean"] = parsed.map(lambda item: item[0])
            block[f"{metric}_std"] = parsed.map(lambda item: item[1])

        frames.append(block)

    data = pd.concat(frames, ignore_index=True)
    validate_data(data)
    return data


def validate_data(data):
    """在绘图前检查四个子集是否都包含所有模型及六项指标。"""
    problems = []

    for split_name, _, region in PLOT_SUBSETS:
        subset = data[
            (data["Split"] == split_name)
            & (data["Region"] == region)
        ]

        for model in MODEL_ORDER:
            count = int((subset["Model"] == model).sum())
            if count != 1:
                problems.append(
                    f"{split_name} / {region} / {model}: {count} row(s)"
                )

    mean_columns = [f"{metric}_mean" for metric in METRICS]
    if data[mean_columns].isna().any().any():
        bad_columns = data[mean_columns].columns[
            data[mean_columns].isna().any()
        ].tolist()
        problems.append("missing mean values in: " + ", ".join(bad_columns))

    if problems:
        raise ValueError(
            "Radial data validation failed:\n  " + "\n  ".join(problems)
        )


def choose_ymax(max_score):
    """按照数据范围确定 y 轴上限，同时不超过 1.0。"""
    return min(1.0, max(0.8, math.ceil((max_score + 0.05) * 10) / 10))


def plot_one_subset(data, split_name, split_label, region):
    """绘制一张与参考图同风格的模型对比散点图。"""
    subset = data[
        (data["Split"] == split_name)
        & (data["Region"] == region)
    ]

    x = np.arange(len(METRICS))
    plt.figure(figsize=(5, 5))

    for model in MODEL_ORDER:
        row = subset[subset["Model"] == model].iloc[0]
        scores = [row[f"{metric}_mean"] for metric in METRICS]
        is_biocmb = model == "BioCMB-DTI"
        color = MODEL_COLORS[model]
        label = model

        plt.scatter(
            x,
            scores,
            marker="o",
            s=140 if is_biocmb else 100,
            facecolors="none",
            edgecolors=color,
            linewidths=2 if is_biocmb else 1.5,
            label=label,
            zorder=4 if is_biocmb else 3,
        )

    max_score = max(
        float(subset[f"{metric}_mean"].max())
        for metric in METRICS
    )
    ymax = choose_ymax(max_score)

    plt.title(f"{split_label} ({region.capitalize()})")
    plt.ylabel("Score (mean)")
    plt.xticks(x, METRICS, rotation=90)
    plt.ylim(0.0, 1)
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.grid(True, linestyle="-", linewidth=0.5, alpha=0.25)
    plt.legend(frameon=True, ncol=2, fontsize=9)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_stem = "drug_radial" if split_name == "drug radial split" else "protein_radial"
    output_stem = OUTPUT_DIR / f"{SHEET_NAME}_{split_stem}_{region}"
    plt.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def main():
    data = read_radial_sheet()

    for split_name, split_label, region in PLOT_SUBSETS:
        plot_one_subset(data, split_name, split_label, region)

    print(f"Plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
