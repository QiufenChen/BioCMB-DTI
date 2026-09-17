#!/usr/bin/env python

from pathlib import Path

import pandas as pd

# mode = 'random_split'  # or 'drug_cluster_split' or 'prot_cluster_split'
# mode = 'drug_cluster_split'
mode = 'prot_cluster_split'

CONFIG = {
    "result_root": "/home/qfchen/DTIPred_Plus/contrastive_learning/results",
    "data_name": "SNAP",
    "ratio_dir": "1_1",
    "split_mode": f"{mode}",
    "summary_file": "summary.xlsx",
    "output_excel": f"/home/qfchen/DTIPred_Plus/analysis_results/drug_embed/summary_{mode}.xlsx"
    }


# EXPERIMENT_TAGS = [
#     "1e4_ChemBERTa_100M_Ankh",
#     "1e4_ChemBERTa_100M_ESM2_650M",
#     "1e4_ChemBERTa_100M_ESM3_SM",
#     "1e4_ChemBERTa_100M_ESMC_300M",
#     "1e4_ChemBERTa_100M_ESMC_600M",
#     "1e4_ChemBERTa_100M_ProtBert",
#     "1e4_ChemBERTa_100M_ProtBert_BFD",
#     "1e4_ChemBERTa_100M_ProtT5",
# ]

EXPERIMENT_TAGS = [
    "1e4_MolFormer_ProtT5",
    "1e4_UniMol2_164M_ProtT5",
    "1e4_ChemBERTa_100M_ProtT5",
]

def build_summary_path(cfg, experiment_tag):
    return (
        Path(cfg["result_root"])
        / experiment_tag
        / cfg["data_name"]
        / cfg["ratio_dir"]
        / cfg["split_mode"]
        / cfg["summary_file"]
    )


def read_last_non_empty_row(path):
    df = pd.read_excel(path)
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError(f"Empty summary file: {path}")
    return df.tail(1).reset_index(drop=True)


def collect_last_rows(cfg, experiment_tags):
    rows = []
    missing = []

    for experiment_tag in experiment_tags:
        summary_path = build_summary_path(cfg, experiment_tag)
        if not summary_path.exists():
            missing.append(
                {
                    "experiment_tag": experiment_tag,
                    "summary_path": str(summary_path),
                    "reason": "summary.xlsx not found",
                }
            )
            continue

        try:
            last_row = read_last_non_empty_row(summary_path)
        except Exception as exc:
            missing.append(
                {
                    "experiment_tag": experiment_tag,
                    "summary_path": str(summary_path),
                    "reason": repr(exc),
                }
            )
            continue

        last_row.insert(0, "experiment_tag", experiment_tag)
        last_row.insert(1, "summary_path", str(summary_path))
        rows.append(last_row)

    if rows:
        combined = pd.concat(rows, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=["experiment_tag", "summary_path"])

    missing_df = pd.DataFrame(missing)
    return combined, missing_df


def drop_all_empty_columns(df):
    if df.empty:
        return df

    cleaned = df.replace(r"^\s*$", pd.NA, regex=True)
    return cleaned.dropna(axis=1, how="all")


def main():
    cfg = CONFIG.copy()
    combined, missing_df = collect_last_rows(cfg, EXPERIMENT_TAGS)
    combined = drop_all_empty_columns(combined)

    output_excel = Path(cfg["output_excel"])
    output_excel.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="last_rows", index=False)
        missing_df.to_excel(writer, sheet_name="missing", index=False)

    print(f"Collected experiments: {len(combined)}")
    print(f"Missing or failed files: {len(missing_df)}")
    print(f"Saved Excel: {output_excel}")

    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
