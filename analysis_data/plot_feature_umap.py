#!/usr/bin/env python
# Author  : KerryChen
# File    : plot_feature_umap.py

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap


BASE_DIR = "/home/qfchen/DTIPred_Plus/database"
SAVE_DIR = "/home/qfchen/DTIPred_Plus/analysis_data/results/feature_umap"

PROT_FEAT_DIR = "/data/qfchen/DTIPred_Plus/UniprotKB_ProtT5/"
DRUG_FEAT_DIRS = {
    "TTD": "/data/qfchen/DTIPred_Plus/TTD/ChemBERTa_100M",
    "SNAP": "/data/qfchen/DTIPred_Plus/SNAP/ChemBERTa_100M",
    "DRH": "/data/qfchen/DTIPred_Plus/DRH/ChemBERTa_100M",
    "TPP": "/data/qfchen/DTIPred_Plus/TPP/ChemBERTa_100M",
}

DATASETS = {
    "TTD": "TTD/final_data/TTD.xlsx",
    "SNAP": "SNAP/final_data/SNAP.xlsx",
    "DRH": "DRH/final_data/DRH.xlsx",
    "TPP": "TPP/final_data/TPP.xlsx",
}

SEED = 3407
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"

PALETTE = {
    "TTD": "#35B563",
    "SNAP": "#2F91DB",
    "DRH": "#F17135",
    "TPP": "#9950ED",
}


def load_drug_feature(path):
    try:
        feat = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        feat = torch.load(path, map_location="cpu")
    feat = feat.float()
    return feat.mean(dim=0).numpy()


def load_protein_feature(path):
    feat = np.load(path)
    return feat.mean(axis=0)


def collect_ids():
    drug_rows = []
    prot_rows = []

    for dataset_name, rel_path in DATASETS.items():
        data_path = os.path.join(BASE_DIR, rel_path)
        df = pd.read_excel(data_path)

        drug_ids = df["DRUGID"].astype(str).drop_duplicates().tolist()
        prot_ids = df["UNIPROTID"].astype(str).drop_duplicates().tolist()

        drug_rows.extend({"Dataset": dataset_name, "ID": drug_id} for drug_id in drug_ids)
        prot_rows.extend({"Dataset": dataset_name, "ID": prot_id} for prot_id in prot_ids)

    return pd.DataFrame(drug_rows), pd.DataFrame(prot_rows)


def build_feature_matrix(meta_df, feature_type):
    vectors = []
    rows = []

    for _, row in meta_df.iterrows():
        dataset_name = row["Dataset"]
        item_id = row["ID"]

        if feature_type == "drug":
            feat_path = os.path.join(DRUG_FEAT_DIRS[dataset_name], f"{item_id}.pt")
        else:
            feat_path = os.path.join(PROT_FEAT_DIR, f"{item_id}.npy")

        if not os.path.exists(feat_path):
            print(f"[WARNING] Missing {feature_type} feature: {feat_path}")
            continue

        if feature_type == "drug":
            vector = load_drug_feature(feat_path)
        else:
            vector = load_protein_feature(feat_path)

        vectors.append(vector)
        rows.append({
            "Dataset": dataset_name,
            "ID": item_id,
            "FeaturePath": feat_path,
        })

    feature_matrix = np.vstack(vectors)
    out_meta = pd.DataFrame(rows)
    return feature_matrix, out_meta


def run_umap(feature_matrix):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=SEED,
    )
    return reducer.fit_transform(feature_matrix)


def plot_umap(umap_df, title, save_name):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        data=umap_df,
        x="UMAP1",
        y="UMAP2",
        hue="Dataset",
        hue_order=list(DATASETS.keys()),
        palette=PALETTE,
        s=15,
        alpha=0.75,
        linewidth=0,
    )
    plt.title(title, fontsize=20, pad=20)
    plt.xlabel("UMAP 1", fontsize=18, labelpad=20)
    plt.ylabel("UMAP 2", fontsize=18, labelpad=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(frameon=True, markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, save_name), bbox_inches="tight")
    plt.close()


def analyze_feature_umap(meta_df, feature_type):
    feature_matrix, out_meta = build_feature_matrix(meta_df, feature_type)
    coords = run_umap(feature_matrix)

    out_meta["UMAP1"] = coords[:, 0]
    out_meta["UMAP2"] = coords[:, 1]

    if feature_type == "drug":
        csv_name = "drug_feature_umap.csv"
        fig_name = "drug_feature_umap.pdf"
        title = "Drug representation space (ChemBERTa)"
    else:
        csv_name = "protein_feature_umap.csv"
        fig_name = "protein_feature_umap.pdf"
        title = "Protein representation space (ProtT5)"

    out_meta.to_csv(os.path.join(SAVE_DIR, csv_name), index=False)
    plot_umap(out_meta, title, fig_name)
    print(f"{title}: {len(out_meta)} points")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    drug_meta, prot_meta = collect_ids()
    analyze_feature_umap(drug_meta, "drug")
    analyze_feature_umap(prot_meta, "protein")

    print(f"\nSaved UMAP results to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
