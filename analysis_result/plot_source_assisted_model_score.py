#!/usr/bin/env python
"""Draw separate model-score plots from source_assisted_learning.xlsx."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL_DEPS = HERE / ".plot_deps"

# Avoid OpenBLAS trying to create many worker threads in a constrained process.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))

# Load the runtime's NumPy/Pandas first.  The local plotting folder also contains
# a NumPy wheel, which should not override the already configured runtime copy.
import numpy as np
import pandas as pd

if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))
import matplotlib.pyplot as plt


INPUT_FILE = HERE / "results" / "source_assisted_learning.xlsx"
OUTPUT_DIR = HERE / "source_assisted_model_score_plots"
METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPR"]

MODEL_COLORS = {
    "HpyerAttentionDTI": ("HyperAttentionDTI", "#1f77b4"),
    "BINDTI": ("BINDTI", "#ff7f0e"),
    "BioFusionDTI": ("BioFusionDTI", "#2ca02c"),
    "OurModel": ("BioCMB-DTI", "#9467bd"),
}

METHOD_LABELS = {
    "NP-only": "NP-only",
    "np_sdt_joint": "NP + SDT",
    "np_source_similarity_weighted_decay": "Similarity-weighted decay",
}

SCENARIOS = [
    ("cold_drug_split", "test", "Cold-drug"),
    ("drug_radial_split", "near", "Radial near"),
    ("drug_radial_split", "far", "Radial far"),
    ("drug_radial_split", "near + far", "Radial near + far"),
]


def normalize(value: object) -> str:
    text = re.sub(r"\s+", "_", str(value).strip())
    return re.sub(r"_+", "_", text)


def parse_mean(value: object) -> float:
    """Return the mean from a cell such as '0.630±0.060'."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else np.nan


def load_data() -> pd.DataFrame:
    frames = []
    for sheet in pd.ExcelFile(INPUT_FILE).sheet_names:
        df = pd.read_excel(INPUT_FILE, sheet_name=sheet)
        df["Training method"] = df["Training method"].ffill()
        df["Model"] = df["Model"].ffill()
        region_column = next(
            (column for column in df.columns if str(column).startswith("Unnamed")),
            None,
        )
        if region_column is None:
            df["Region"] = "test"
        else:
            df = df.rename(columns={region_column: "Region"})
            df["Region"] = df["Region"].ffill().astype(str).str.strip()
        df["MethodKey"] = df["Training method"].map(normalize)
        for metric in METRICS:
            df[metric] = df[metric].map(parse_mean)
        df["Sheet"] = sheet
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def safe_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def plot_one(data: pd.DataFrame, sheet: str, region: str, scenario: str, method: str) -> None:
    subset = data[
        (data["Sheet"] == sheet)
        & (data["Region"] == region)
        & (data["MethodKey"] == method)
    ]

    plt.figure(figsize=(5, 5))
    x = np.arange(len(METRICS))
    for model, (label, color) in MODEL_COLORS.items():
        row = subset[subset["Model"].astype(str).str.strip() == model]
        if row.empty:
            continue
        scores = row.iloc[0][METRICS].astype(float).to_numpy()
        is_biocmb = model == "OurModel"
        plt.scatter(
            x,
            scores,
            marker="o",
            s=140 if is_biocmb else 100,
            facecolors="none",
            edgecolors=color,
            linewidths=1 if is_biocmb else 1,
            label=label,
            zorder=3 if is_biocmb else 3,
        )

    plt.xticks(x, METRICS, rotation=90)
    plt.ylabel("Score (mean)")
    plt.ylim(0.0, 0.8)
    plt.title(f"{scenario} — {METHOD_LABELS[method]}")
    plt.grid(True, linestyle="-", linewidth=0.5, alpha=0.25)
    plt.legend(frameon=True, ncol=2, fontsize=9)
    plt.tight_layout()

    stem = f"{safe_name(scenario)}_{safe_name(METHOD_LABELS[method])}"
    plt.savefig(OUTPUT_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    for sheet, region, scenario in SCENARIOS:
        for method in METHOD_LABELS:
            plot_one(data, sheet, region, scenario, method)
    print(f"Saved 12 separate plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
