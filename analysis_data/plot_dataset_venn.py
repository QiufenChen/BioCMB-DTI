#!/usr/bin/env python
# Author  : KerryChen
# File    : plot_dataset_venn.py

import os

import matplotlib.pyplot as plt
import pandas as pd
from venn import venn


BASE_DIR = "/home/qfchen/DTIPred_Plus/database"
SAVE_DIR = "/home/qfchen/DTIPred_Plus/analysis_data/results/dataset_venn"

DATASETS = {
    "SNAP": {
        "file": "SNAP/final_data/SNAP.xlsx",
        "drug_col": "INCHIKEY",
    },
    "DRH": {
        "file": "DRH/final_data/DRH.xlsx",
        "drug_col": "INCHIKEY",
    },
    "TTD": {
        "file": "TTD/final_data/TTD.xlsx",
        "drug_col": "INCHIKEY",
    },
    # "SDT": {
    #     "file": "SDT/final_data/SDT.xlsx",
    #     "drug_col": "INCHIKEY",
    # },
}


def load_sets():
    drug_sets = {}
    prot_sets = {}
    pair_sets = {}
    summary = []

    for name, cfg in DATASETS.items():
        data_path = os.path.join(BASE_DIR, cfg["file"])
        drug_col = cfg["drug_col"]
        df = pd.read_excel(data_path)

        drug_set = set(df[drug_col].astype(str))
        prot_set = set(df["UNIPROTID"].astype(str))
        pair_set = set(zip(df[drug_col].astype(str), df["UNIPROTID"].astype(str)))

        drug_sets[name] = drug_set
        prot_sets[name] = prot_set
        pair_sets[name] = pair_set

        summary.append({
            "Dataset": name,
            "Drugs": len(drug_set),
            "Proteins": len(prot_set),
            "Positive pairs": len(pair_set),
        })

    return drug_sets, prot_sets, pair_sets, pd.DataFrame(summary)


def plot_venn(set_dict, title, save_name):
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    venn(set_dict, ax=ax)

    for text in ax.texts:
        text.set_fontsize(17)   # 这里调大，比如 16、18、20
        # text.set_fontweight("bold")

    ax.set_title(title, fontsize=20, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, save_name), bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    drug_sets, prot_sets, pair_sets, summary_df = load_sets()

    plot_venn(prot_sets, "UNIPROTID", "protein_venn.pdf")
    plot_venn(drug_sets, "INCHIKEY", "drug_venn.pdf")
    plot_venn(pair_sets, "UNIPROTID-INCHIKEY", "pair_venn.pdf")

    summary_df.to_csv(os.path.join(SAVE_DIR, "venn_summary.csv"), index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved figures and summary to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
