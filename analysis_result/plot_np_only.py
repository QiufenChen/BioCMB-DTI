#!/usr/bin/env python
"""Plot NPSet-only model performance using open-circle markers."""

import re
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_FILE = Path(__file__).resolve().parent / "results" / "np_only.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "random_np_only_model_plots"
SHEET_NAME = "cold_prot_split"

METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPR"]
MODELS = ["DrugBAN", "BINDTI", "BioFusionDTI", "OurModel"]

MODEL_LABELS = {
    "DrugBAN": "DrugBAN",
    "BINDTI": "BINDTI",
    "BioFusionDTI": "BioFusionDTI",
    "OurModel": "BioCMB-DTI",
}

MODEL_COLORS = {
    "DrugBAN": "#D84A3A",
    "BINDTI": "#F0802F",
    "BioFusionDTI": "#4AA64F",
    "OurModel": "#7651A8",
}

SPLIT_TITLES = {
    "random_split": "Random split",
    "cold_drug_split": "Cold drug split",
    "cold_prot_split": "Cold protein split",
}


def normalize_key(value):
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def parse_mean(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", str(value))
    return float(numbers[0]) if numbers else np.nan


def load_results():
    data = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

    required = {"Model", *METRICS}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data["Model"] = data["Model"].ffill().astype(str).str.strip()
    data = data[data["Model"].isin(MODELS)].copy()

    if data["Model"].duplicated().any() or set(data["Model"]) != set(MODELS):
        raise ValueError(
            f"Expected exactly one row for each selected model, got {sorted(data['Model'].unique())}"
        )

    data = data.set_index("Model")
    return {
        model: [parse_mean(data.loc[model, metric]) for metric in METRICS]
        for model in MODELS
    }


def plot_points(scores):
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    x = np.arange(len(METRICS))
    plt.figure(figsize=(5, 5))

    for model in MODELS:
        is_biocmb = model == "OurModel"
        plt.scatter(
            x, scores[model],
            marker="o",
            s=100,
            facecolors="none",
            edgecolors=MODEL_COLORS[model],
            linewidths=1.5,
            label=MODEL_LABELS[model],
            zorder=3,
        )

    plt.title(SPLIT_TITLES.get(SHEET_NAME, SHEET_NAME), fontsize=16, pad=12)
    plt.ylabel("Score (mean)", fontsize=14)
    plt.xticks(x, METRICS, rotation=45, ha="right")
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="-", linewidth=0.5, alpha=0.25)
    plt.legend(frameon=True, edgecolor="#C8C8C8", fontsize=11)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = OUTPUT_DIR / f"{normalize_key(SHEET_NAME)}_np_only_model_comparison"
    plt.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_stem}")


def main():
    plot_points(load_results())


if __name__ == "__main__":
    main()