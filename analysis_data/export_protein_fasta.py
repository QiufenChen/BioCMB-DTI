#!/usr/bin/env python
# Author  : KerryChen
# File    : export_protein_fasta.py

import os
import re

import pandas as pd


BASE_DIR = "/home/qfchen/DTIPred_Plus/database"
SAVE_DIR = "/home/qfchen/DTIPred_Plus/analysis_data/results/protein_fasta"

DATASETS = {
    # "SNAP": "SNAP/final_data/ProtInfo.xlsx",
    # "DRH": "DRH/final_data/ProtInfo.xlsx",
    # "TTD": "TTD/final_data/ProtInfo.xlsx",
    "SDT": "SDT/final_data/ProtInfo.xlsx",
}


def wrap_sequence(sequence, width=80):
    sequence = str(sequence).strip()
    return "\n".join(sequence[i:i + width] for i in range(0, len(sequence), width))


def safe_filename(value):
    value = str(value).strip()
    value = re.sub(r"[^\w.-]+", "_", value)
    return value if value else "unknown"


def export_one_dataset(dataset_name, rel_path):
    data_path = os.path.join(BASE_DIR, rel_path)
    df = pd.read_excel(data_path)[["UNIPROTID", "SEQUENCE"]].drop_duplicates("UNIPROTID")

    fasta_path = os.path.join(SAVE_DIR, f"{dataset_name}.fasta")
    individual_dir = os.path.join(SAVE_DIR, "individual_fasta", dataset_name)
    os.makedirs(individual_dir, exist_ok=True)

    with open(fasta_path, "w", encoding="utf-8") as merged_f:
        for _, row in df.iterrows():
            uniprot_id = str(row["UNIPROTID"]).strip()
            sequence = str(row["SEQUENCE"]).strip()
            header = f">{uniprot_id}"
            wrapped_sequence = wrap_sequence(sequence)

            merged_f.write(header + "\n")
            merged_f.write(wrapped_sequence + "\n")

            single_fasta_path = os.path.join(individual_dir, f"{safe_filename(uniprot_id)}.fasta")
            with open(single_fasta_path, "w", encoding="utf-8") as single_f:
                single_f.write(header + "\n")
                single_f.write(wrapped_sequence + "\n")

    return {
        "Dataset": dataset_name,
        "Protein_N": len(df),
        "FASTA": fasta_path,
        "Individual_FASTA_DIR": individual_dir,
    }


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    summary = []
    for dataset_name, rel_path in DATASETS.items():
        summary.append(export_one_dataset(dataset_name, rel_path))

    all_fasta_path = os.path.join(SAVE_DIR, "all_proteins.fasta")
    with open(all_fasta_path, "w", encoding="utf-8") as out_f:
        for row in summary:
            with open(row["FASTA"], "r", encoding="utf-8") as in_f:
                out_f.write(in_f.read())

    summary_df = pd.DataFrame(summary)
    summary_df.loc[len(summary_df)] = {
        "Dataset": "All",
        "Protein_N": int(summary_df["Protein_N"].sum()),
        "FASTA": all_fasta_path,
        "Individual_FASTA_DIR": os.path.join(SAVE_DIR, "individual_fasta"),
    }

    summary_df.to_csv(os.path.join(SAVE_DIR, "protein_fasta_summary.csv"), index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved protein FASTA files to: {SAVE_DIR}")


if __name__ == "__main__":
    main()