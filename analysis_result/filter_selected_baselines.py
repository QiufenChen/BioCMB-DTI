#!/usr/bin/env python

from pathlib import Path

import pandas as pd


INPUT_FILE = Path("analysis_results/results/model_baseline.xlsx")
OUTPUT_FILE = Path("analysis_results/results/model_baseline_selected.xlsx")

KEEP_MODELS = [
    "BioCMB-DTI",
    "HyperAttentionDTI",
    "BINDTI",
    "DrugBAN",
    "RSGCL-DTI",
    "MIF-DTI",
    "BioFusionDTI",
]

MODEL_NAME_MAP = {
    "BioCMBDTI": "BioCMB-DTI",
    "BioCMB-DTI": "BioCMB-DTI",
    "HpyerAttentionDTI": "HyperAttentionDTI",
    "HyperAttentionDTI": "HyperAttentionDTI",
    "BINDTI": "BINDTI",
    "DrugBAN": "DrugBAN",
    "RSGCL-DTI": "RSGCL-DTI",
    "MIF-DTI": "MIF-DTI",
    "BioFusionDTI": "BioFusionDTI",
}

SPLIT_HEADERS = {
    "Cold drug split": "Cold-drug split",
    "Cold protein split": "Cold-protein split",
}

METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPR"]
MAIN_METRICS = ["F1", "AUROC", "AUPR"]


def parse_workbook():
    rows = []
    excel = pd.ExcelFile(INPUT_FILE)
    for dataset in excel.sheet_names:
        df = pd.read_excel(INPUT_FILE, sheet_name=dataset)
        first_col = df.columns[0]
        current_split = "Random split"
        for _, row in df.iterrows():
            raw_name = row[first_col]
            if pd.isna(raw_name):
                continue
            raw_name = str(raw_name).strip()
            if raw_name in SPLIT_HEADERS:
                current_split = SPLIT_HEADERS[raw_name]
                continue
            model = MODEL_NAME_MAP.get(raw_name)
            if model is None or model not in KEEP_MODELS:
                continue
            record = {"Dataset": dataset, "Split": current_split, "Model": model}
            for metric in METRICS:
                record[metric] = row.get(metric)
            rows.append(record)
    out = pd.DataFrame(rows)
    out["Model"] = pd.Categorical(out["Model"], categories=KEEP_MODELS, ordered=True)
    out["Split"] = pd.Categorical(out["Split"], categories=["Random split", "Cold-drug split", "Cold-protein split"], ordered=True)
    out = out.sort_values(["Dataset", "Split", "Model"]).reset_index(drop=True)
    return out


def parse_mean(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).replace("��", "±").strip()
    if "±" in text:
        text = text.split("±", 1)[0]
    try:
        return float(text)
    except ValueError:
        return pd.NA


def normalize_metric_text(df):
    out = df.copy()
    for metric in METRICS:
        if metric in out.columns:
            out[metric] = out[metric].map(lambda x: x.replace("��", "±") if isinstance(x, str) else x)
    return out


def build_main_table(df):
    rows = []
    for _, row in df.iterrows():
        item = {"Dataset": row["Dataset"], "Split": row["Split"], "Model": row["Model"]}
        for metric in MAIN_METRICS:
            item[metric] = row[metric]
        rows.append(item)
    return pd.DataFrame(rows)


def build_summary(df):
    numeric = df.copy()
    for metric in MAIN_METRICS:
        numeric[metric + "_mean"] = numeric[metric].map(parse_mean)

    rows = []
    for (dataset, split), group in numeric.groupby(["Dataset", "Split"], observed=True):
        our = group[group["Model"] == "BioCMB-DTI"]
        if our.empty:
            continue
        our = our.iloc[0]
        for metric in MAIN_METRICS:
            baseline_group = group[group["Model"] != "BioCMB-DTI"].dropna(subset=[metric + "_mean"])
            if baseline_group.empty:
                best_model = pd.NA
                best_value = pd.NA
                delta = pd.NA
            else:
                best = baseline_group.loc[baseline_group[metric + "_mean"].idxmax()]
                best_model = best["Model"]
                best_value = best[metric]
                delta = round(float(our[metric + "_mean"]) - float(best[metric + "_mean"]), 3)
            rows.append({
                "Dataset": dataset,
                "Split": split,
                "Metric": metric,
                "BioCMB-DTI": our[metric],
                "Best baseline": best_model,
                "Best baseline value": best_value,
                "Delta mean": delta,
            })
    return pd.DataFrame(rows)


def main():
    selected = normalize_metric_text(parse_workbook())
    main_table = build_main_table(selected)
    summary = build_summary(selected)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        main_table.to_excel(writer, sheet_name="main_F1_AUROC_AUPR", index=False)
        selected.to_excel(writer, sheet_name="all_metrics", index=False)
        summary.to_excel(writer, sheet_name="summary_vs_best_baseline", index=False)
    print(f"Saved: {OUTPUT_FILE}")
    print("Rows:", len(selected))


if __name__ == "__main__":
    main()
