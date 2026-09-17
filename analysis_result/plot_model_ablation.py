#!/usr/bin/env python
# Author  : KerryChen
# File    : plot_model_ablation.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILE = Path("/home/qfchen/DTIPred_Plus/analysis_results/results/model_ablation.xlsx")
OUTPUT_DIR = Path(__file__).resolve().parent / "model_ablation_plots"

DATASETS = ["SNAP", "DRH", "TTD"]

METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPR"]
MAIN_METRICS = ["F1", "AUROC", "AUPR"]

VARIANT_RENAME = {
    "BioMCB-DTI": "BioCMB-DTI",
    "BioCMB-DTI": "BioCMB-DTI",
    "w/o Mamba": "CNN-only encoder",
    "w/o CNN": "Mamba-only encoder",
    "w/o adaptive gate": "w/o adaptive gate",
    "Cross-attention fusion": "BAN replaced by cross-attention",
    "Concatenation fusion": "BAN replaced by concat-MLP",
}

VARIANT_ORDER = [
    "CNN-only encoder",
    "Mamba-only encoder",
    "w/o adaptive gate",
    "BAN replaced by cross-attention",
    "BAN replaced by concat-MLP",
]


def parse_mean(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if text == "":
        return np.nan

    text = text.replace("+/-", "±").replace("卤", "±")
    return float(text.split("±")[0])


def read_ablation_table(input_file):
    raw = pd.read_excel(input_file, sheet_name=0, header=None)
    rows = []

    for dataset in DATASETS:
        header_idx = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq(dataset)]
        if len(header_idx) == 0:
            continue

        start = int(header_idx[0])
        header = raw.iloc[start].tolist()

        end = start + 1
        while end < len(raw):
            first_cell = raw.iloc[end, 0]
            if pd.isna(first_cell):
                break
            if str(first_cell).strip() in DATASETS and end != start:
                break
            end += 1

        block = raw.iloc[start + 1:end].copy()
        block.columns = header
        block = block.rename(columns={dataset: "Variant"})

        for _, row in block.iterrows():
            variant = str(row["Variant"]).strip()
            if variant == "" or variant.lower() == "nan":
                continue

            variant = VARIANT_RENAME.get(variant, variant)

            item = {"Dataset": dataset, "Variant": variant}
            for metric in METRICS:
                if metric in block.columns:
                    item[metric] = parse_mean(row[metric])
            rows.append(item)

    return pd.DataFrame(rows)


def make_delta_table(df):
    delta_rows = []

    for dataset in DATASETS:
        sub = df[df["Dataset"] == dataset].copy()
        full = sub[sub["Variant"] == "BioCMB-DTI"]

        if full.empty:
            raise ValueError(f"Cannot find full model row for {dataset}")

        full_values = full.iloc[0]

        for _, row in sub.iterrows():
            if row["Variant"] == "BioCMB-DTI":
                continue

            item = {"Dataset": dataset, "Variant": row["Variant"]}
            for metric in METRICS:
                item[metric] = row[metric] - full_values[metric]

            delta_rows.append(item)

    return pd.DataFrame(delta_rows)


def plot_one_dataset_delta_heatmap(delta_df, dataset, metrics, output_name, vmin, vmax):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sub = delta_df[delta_df["Dataset"] == dataset].copy()
    sub["Variant"] = pd.Categorical(sub["Variant"], categories=VARIANT_ORDER, ordered=True)
    sub = sub.sort_values("Variant")

    values = sub[metrics].to_numpy(dtype=float)


    plt.figure(figsize=(8, 5))
    im = plt.imshow(values, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")

    plt.xticks(np.arange(len(metrics)), metrics, rotation=45, ha="right", fontsize=11)
    plt.yticks(np.arange(len(sub)), sub["Variant"].tolist(), fontsize=11)
    plt.title(dataset, fontsize=14, pad=10)

    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            value = values[r, c]
            if np.isnan(value):
                label = "NA"
                color = "black"
            else:
                label = f"{value:+.3f}"
                color = "white" if abs(value) > vmax * 0.55 else "black"
            plt.text(c, r, label, ha="center", va="center", fontsize=9, color=color)

    plt.axvline(-0.5, color="black", linewidth=0.8)
    plt.axvline(len(metrics) - 0.5, color="black", linewidth=0.8)
    plt.axhline(-0.5, color="black", linewidth=0.8)
    plt.axhline(len(sub) - 0.5, color="black", linewidth=0.8)

    cbar = plt.colorbar(im, fraction=0.045, pad=0.04)
    cbar.set_label("Delta performance vs BioCMB-DTI", fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{output_name}.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{output_name}.png", dpi=600, bbox_inches="tight")
    plt.close()


def plot_metric_bar(df, metric, output_name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    variants = ["BioCMB-DTI"] + VARIANT_ORDER
    colors = {
        "BioCMB-DTI": "#7A9CC6",
        "CNN-only encoder": "#B8D8BA",
        "Mamba-only encoder": "#F4C095",
        "w/o adaptive gate": "#CDB4DB",
        "BAN replaced by cross-attention": "#F2A7A5",
        "BAN replaced by concat-MLP": "#A7D8DE",
    }

    x = np.arange(len(DATASETS))
    width = 0.12

    plt.figure(figsize=(7.2, 4.6))

    for i, variant in enumerate(variants):
        means = []

        for dataset in DATASETS:
            hit = df[(df["Dataset"] == dataset) & (df["Variant"] == variant)]
            means.append(hit[metric].iloc[0] if not hit.empty else np.nan)

        offset = (i - (len(variants) - 1) / 2) * width

        plt.bar(
            x + offset,
            means,
            width=width,
            label=variant,
            color=colors.get(variant, "#CCCCCC"),
            edgecolor="white",
            linewidth=0.8,
        )

    plt.xticks(x, DATASETS, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel(metric, fontsize=13)

    if metric in ["AUROC", "AUPR"]:
        plt.ylim(0.88, 0.96)
    else:
        plt.ylim(0.80, 0.91)

    plt.legend(fontsize=8, frameon=False, ncol=2)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / f"{output_name}.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{output_name}.png", dpi=600, bbox_inches="tight")
    plt.close()


def main():
    df = read_ablation_table(INPUT_FILE)
    delta_df = make_delta_table(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "model_ablation_mean_values.csv", index=False)
    delta_df.to_csv(OUTPUT_DIR / "model_ablation_delta_values.csv", index=False)

    main_vmax = max(np.nanmax(np.abs(delta_df[MAIN_METRICS].to_numpy(dtype=float))), 0.005)
    all_vmax = max(np.nanmax(np.abs(delta_df[METRICS].to_numpy(dtype=float))), 0.005)

    for dataset in DATASETS:
        dataset_name = dataset.lower()

        plot_one_dataset_delta_heatmap(
            delta_df=delta_df,
            dataset=dataset,
            metrics=MAIN_METRICS,
            output_name=f"model_ablation_{dataset_name}_delta_heatmap_main_metrics",
            vmin=-main_vmax,
            vmax=main_vmax,
        )

        plot_one_dataset_delta_heatmap(
            delta_df=delta_df,
            dataset=dataset,
            metrics=METRICS,
            output_name=f"model_ablation_{dataset_name}_delta_heatmap_all_metrics",
            vmin=-all_vmax,
            vmax=all_vmax,
        )

    for metric in MAIN_METRICS:
        plot_metric_bar(
            df=df,
            metric=metric,
            output_name=f"model_ablation_{metric.lower()}_bar",
        )

    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
