#!/usr/bin/env python
"""Compare positive-pool NP drugs with SDT drugs and NP targets.

For every natural-product drug in ``tpp_positive_pool.csv``:
1. NP-to-SDT: maximum Morgan--Tanimoto similarity to any SDT drug.
2. NP-to-NP: nearest-neighbor Morgan--Tanimoto similarity to another positive-pool NP drug.
3. NP-to-SDT chemical distance: extended-Tanimoto distance from the Morgan
   fingerprint to the SDT-drug fingerprint centroid, using the same definition
   as the Morgan radial split.

The two similarity densities are overlaid in one figure. The Morgan distance is
saved as a separate density figure. The script also plots the
SDT-versus-positive-pool-NP UMAP and the NP target overlap with SDT.
SMILES are always mapped from ``TPP/DrugInfo.xlsx``; the script does not use all TPP drugs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
try:
    import umap
except ImportError as exc:
    raise ImportError(
        "This script needs umap-learn for the chemical-space UMAP. "
        "Install it with: pip install umap-learn"
    ) from exc


BASE_DIR = Path("../database")
OUTPUT_DIR = Path("./results/sdt_np_distribution")

SDT_DRUG_TABLE = BASE_DIR / "SDT/final_data/DrugInfo.xlsx"
SDT_PROT_TABLE = BASE_DIR / "SDT/final_data/ProtInfo.xlsx"
SDT_PAIR_TABLE = BASE_DIR / "SDT/final_data/SDT.xlsx"

NP_POSITIVE_POOL = BASE_DIR / "TPP/final_data_random/tpp_positive_pool.csv"
NP_DRUG_INFO = BASE_DIR / "TPP/final_data_random/DrugInfo.xlsx"

MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_N_BITS)
RANDOM_STATE = 42
MAX_SDT_POINTS_FOR_UMAP = 7000
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.15

COLORS = {
    "NP-to-SDT": "#0072B2",
    "NP-to-NP": "#D55E00",
    "SDT": "#A19E9E",
    "NP": "#D55E00",
    "shared": "#009E73",
    "unique": "#CC79A7",
}


def find_col(columns, candidates):
    lower_to_original = {str(col).lower(): col for col in columns}
    for name in candidates:
        if name.lower() in lower_to_original:
            return lower_to_original[name.lower()]
    raise ValueError("None of {} found in columns: {}".format(candidates, list(columns)))


def find_optional_col(columns, candidates):
    try:
        return find_col(columns, candidates)
    except ValueError:
        return None


def read_table(path):
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def require_existing(*paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these files exists: {}".format([str(p) for p in paths]))


def canonicalize_smiles(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MORGAN_GENERATOR.GetFingerprint(mol)


def load_sdt_drugs():
    """Load unique SDT drug structures from SDT DrugInfo or SDT pair data."""
    path = require_existing(SDT_DRUG_TABLE, SDT_PAIR_TABLE)
    df = read_table(path)
    smiles_col = find_col(df.columns, ["SMILES", "Drug_SMILES", "drug_smiles"])
    id_col = find_optional_col(df.columns, ["INCHIKEY", "InChIKey", "DRUGID", "Drug"])

    sub = df[[smiles_col]].copy().rename(columns={smiles_col: "SMILES"})
    if id_col is not None:
        sub["DrugKey"] = df[id_col].astype(str).str.strip()
    else:
        sub["DrugKey"] = np.nan

    sub["CanonicalSMILES"] = sub["SMILES"].map(canonicalize_smiles)
    sub = sub.dropna(subset=["CanonicalSMILES"])
    sub["MergeKey"] = sub["DrugKey"].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    sub["MergeKey"] = sub["MergeKey"].fillna(sub["CanonicalSMILES"])
    sub = sub.drop_duplicates("MergeKey").reset_index(drop=True)
    print("Loaded {} SDT drugs from: {}".format(len(sub), path))
    return sub


def load_positive_pool():
    """Read only explicit positive rows from the NP positive pool."""
    positive = pd.read_csv(NP_POSITIVE_POOL)
    label_col = find_optional_col(positive.columns, ["Label", "label", "Y"])

    # tpp_positive_pool.csv should already contain positives. If a label column
    # is present, retain only explicit positive rows as an additional safeguard.
    if label_col is not None:
        numeric_label = pd.to_numeric(positive[label_col], errors="coerce")
        if numeric_label.notna().any():
            positive = positive[numeric_label == 1].copy()
    return positive


def load_positive_pool_np_drugs():
    """Map positive-pool drug IDs to SMILES in TPP DrugInfo.xlsx."""
    positive = load_positive_pool()
    positive_drug_col = find_col(positive.columns, ["Drug", "DRUGID", "Drug_ID"])

    positive_ids = set(positive[positive_drug_col].dropna().astype(str).str.strip())
    if not positive_ids:
        raise ValueError("No positive-pool drug identifiers found in {}".format(NP_POSITIVE_POOL))

    drug_info = pd.read_excel(NP_DRUG_INFO)
    info_drug_col = find_col(drug_info.columns, ["DRUGID", "Drug", "Drug_ID"])
    smiles_col = find_col(drug_info.columns, ["SMILES", "smiles"])
    drug_info = drug_info.copy()
    drug_info["DrugKey"] = drug_info[info_drug_col].astype(str).str.strip()

    sub = drug_info[drug_info["DrugKey"].isin(positive_ids)][["DrugKey", smiles_col]].copy()
    sub = sub.rename(columns={smiles_col: "SMILES"})
    sub["CanonicalSMILES"] = sub["SMILES"].map(canonicalize_smiles)
    sub = sub.dropna(subset=["CanonicalSMILES"]).drop_duplicates("DrugKey").reset_index(drop=True)

    missing = len(positive_ids - set(sub["DrugKey"]))
    print(
        "Mapped {}/{} positive-pool NP drug IDs to valid SMILES from: {}".format(
            len(sub), len(positive_ids), NP_DRUG_INFO
        )
    )
    if missing:
        print("Skipped {} positive-pool NP drug IDs without valid mapped SMILES.".format(missing))
    return sub


def load_sdt_proteins():
    path = require_existing(SDT_PROT_TABLE, SDT_PAIR_TABLE)
    df = read_table(path)
    protein_col = find_col(df.columns, ["UNIPROTID", "Protein_ID", "ProteinID", "PROTID", "Protein"])
    proteins = set(df[protein_col].dropna().astype(str).str.strip())
    print("Loaded {} SDT proteins from: {}".format(len(proteins), path))
    return proteins


def load_positive_pool_np_proteins():
    positive = load_positive_pool()
    protein_col = find_col(positive.columns, ["Protein_ID", "UNIPROTID", "ProteinID", "PROTID", "Protein"])
    return set(positive[protein_col].dropna().astype(str).str.strip())


def add_fingerprints(df, label):
    fingerprints = []
    valid_rows = []
    for index, smiles in enumerate(df["CanonicalSMILES"]):
        fingerprint = morgan_fp(smiles)
        if fingerprint is None:
            continue
        fingerprints.append(fingerprint)
        valid_rows.append(index)

    if not fingerprints:
        raise ValueError("No valid Morgan fingerprints found for {}".format(label))
    return df.iloc[valid_rows].copy().reset_index(drop=True), fingerprints


def fingerprints_to_array(fingerprints):
    array = np.zeros((len(fingerprints), MORGAN_N_BITS), dtype=np.uint8)
    for index, fingerprint in enumerate(fingerprints):
        DataStructs.ConvertToNumpyArray(fingerprint, array[index])
    return array


def morgan_distance_to_sdt_centroid(query_fps, sdt_fps):
    """Extended-Tanimoto distance to the SDT Morgan-fingerprint centroid.

    This is exactly the distance definition used by the Morgan radial split:
    ``1 - (x · c) / (||x||² + ||c||² - x · c)``, where ``c`` is the mean
    fingerprint vector over all SDT drugs.
    """
    query_vectors = fingerprints_to_array(query_fps).astype(np.float32)
    sdt_center = fingerprints_to_array(sdt_fps).astype(np.float32).mean(axis=0)
    dot_products = query_vectors @ sdt_center
    query_norms = np.sum(query_vectors * query_vectors, axis=1)
    center_norm = float(np.sum(sdt_center * sdt_center))
    denominator = query_norms + center_norm - dot_products
    similarities = np.divide(
        dot_products,
        denominator,
        out=np.zeros_like(dot_products),
        where=denominator > 0,
    )
    return 1.0 - np.clip(similarities, 0.0, 1.0)


def max_similarity_to_reference(query_fps, reference_fps):
    """Maximum Tanimoto similarity of each query drug to a reference drug set."""
    if not reference_fps:
        raise ValueError("Reference fingerprint set is empty")
    return np.asarray([max(DataStructs.BulkTanimotoSimilarity(fp, reference_fps)) for fp in query_fps], dtype=float)


def nearest_neighbor_similarity(fingerprints):
    """Nearest-neighbor Tanimoto similarity within the NP positive-pool drugs."""
    if len(fingerprints) < 2:
        raise ValueError("At least two NP drugs with valid fingerprints are required")

    similarities = []
    for index, fingerprint in enumerate(fingerprints):
        values = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints)
        similarities.append(max(value for other_index, value in enumerate(values) if other_index != index))
    return np.asarray(similarities, dtype=float)


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_chemical_umap(sdt_fps, np_fps):
    """Plot SDT and positive-pool NP drugs in Morgan-fingerprint UMAP space."""
    rng = np.random.default_rng(RANDOM_STATE)
    if len(sdt_fps) > MAX_SDT_POINTS_FOR_UMAP:
        keep = rng.choice(len(sdt_fps), size=MAX_SDT_POINTS_FOR_UMAP, replace=False)
        sdt_fps = [sdt_fps[index] for index in keep]

    sdt_array = fingerprints_to_array(sdt_fps)
    np_array = fingerprints_to_array(np_fps)
    all_fingerprints = np.vstack([sdt_array, np_array])
    if len(all_fingerprints) < 3:
        raise ValueError("At least three total drug fingerprints are required for UMAP")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(UMAP_N_NEIGHBORS, len(all_fingerprints) - 1),
        min_dist=UMAP_MIN_DIST,
        metric="jaccard",
        random_state=RANDOM_STATE,
    )
    coordinates = reducer.fit_transform(all_fingerprints.astype(bool))
    sdt_xy = coordinates[: len(sdt_array)]
    np_xy = coordinates[len(sdt_array) :]

    plt.figure(figsize=(5, 5))
    plt.scatter(
        sdt_xy[:, 0],
        sdt_xy[:, 1],
        s=6,
        c=COLORS["SDT"],
        alpha=0.22,
        linewidths=0,
        label="SDT drugs")
    
    plt.scatter(
        np_xy[:, 0],
        np_xy[:, 1],
        s=24,
        c=COLORS["NP"],
        alpha=0.88,
        edgecolors="white",
        linewidths=0.35,
        label="Positive-pool NP drugs")
    
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("Chemical-space projection", fontsize=12, fontweight="bold", pad=8)
    plt.legend(frameon=False, loc="best", fontsize=9)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "positive_pool_np_chemical_space_umap.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "positive_pool_np_chemical_space_umap.png", dpi=600, bbox_inches="tight")
    plt.close()


def gaussian_kde(values, grid):
    """Gaussian kernel density estimate without an additional dependency."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        raise ValueError("At least two similarity values are required for a density curve")

    std = np.std(values, ddof=1)
    bandwidth = max(0.02, 1.06 * std * len(values) ** (-0.2))
    scaled = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * scaled ** 2).mean(axis=1) / (bandwidth * np.sqrt(2.0 * np.pi))


def plot_similarity_distributions(np_to_sdt, np_to_np):
    """Overlay smooth NP-to-SDT and NP-to-NP similarity density curves."""
    plt.figure(figsize=(5, 5))
    grid = np.linspace(0.0, 1.0, 500)

    for values, key, label in [(np_to_sdt, "NP-to-SDT", "NP to SDT (maximum)"),
                               (np_to_np, "NP-to-NP", "NP to NP (nearest neighbor)")]:
        density = gaussian_kde(values, grid)
        plt.plot(
            grid,
            density,
            color=COLORS[key],
            linewidth=2.0,
            label=label)
        
        plt.fill_between(
            grid,
            0.0,
            density,
            color=COLORS[key],
            alpha=0.20)

    plt.xlim(0.0, 1.0)
    plt.xlabel("Morgan--Tanimoto similarity")
    plt.ylabel("Density")
    plt.title("Similarity of positive-pool NP drugs", fontsize=12, fontweight="bold", pad=8)
    plt.legend(frameon=False, loc="upper left", fontsize=9)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "positive_pool_np_similarity_distribution.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "positive_pool_np_similarity_distribution.png", dpi=600, bbox_inches="tight")
    plt.close()


def plot_morgan_distance_distribution(morgan_distances):
    """Plot positive-pool NP distances to the SDT Morgan centroid."""
    values = np.asarray(morgan_distances, dtype=float)
    if len(values) < 2:
        raise ValueError("At least two Morgan distance values are required")

    plt.figure(figsize=(5, 5))
    plt.hist(
        values,
        bins=20,
        density=True,
        color="#E5A73B",
        edgecolor="white",
        linewidth=1.0,
        alpha=0.85)

    median = float(np.median(values))
    plt.axvline(median, color="#4D4D4D", linestyle="--", linewidth=1.5)
    y_top = plt.ylim()[1]
    plt.text(
        median + 0.002,
        y_top * 0.93,
        "median = {:.3f}".format(median),
        rotation=90,
        va="top",
        ha="left",
        color="#4D4D4D",
        fontsize=9)
    
    plt.xlabel("Morgan distance to SDT centroid")
    plt.ylabel("Density")
    plt.title("NP-to-SDT chemical distance", fontsize=12, fontweight="bold", pad=8)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "positive_pool_np_to_sdt_morgan_distance.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "positive_pool_np_to_sdt_morgan_distance.png", dpi=600, bbox_inches="tight")
    plt.close()


def plot_target_overlap(np_proteins, sdt_proteins):
    """Plot positive-pool NP target overlap with SDT targets."""
    shared = len(np_proteins & sdt_proteins)
    unique = len(np_proteins - sdt_proteins)
    values = [shared, unique]
    labels = ["Shared with SDT", "NP-specific"]

    plt.figure(figsize=(5, 5))
    bars = plt.bar(labels, values, color=[COLORS["shared"], COLORS["unique"]], width=0.58)
    total = max(1, shared + unique)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            "{}\n({:.1f}%)".format(value, value / total * 100),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.ylabel("Number of positive-pool NP targets")
    plt.title("NP target overlap with SDT", fontsize=12, fontweight="bold", pad=8)
    plt.ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "positive_pool_np_target_overlap.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "positive_pool_np_target_overlap.png", dpi=600, bbox_inches="tight")
    plt.close()


def main():
    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sdt_drugs = load_sdt_drugs()
    np_drugs = load_positive_pool_np_drugs()
    sdt_drugs, sdt_fps = add_fingerprints(sdt_drugs, "SDT drugs")
    np_drugs, np_fps = add_fingerprints(np_drugs, "positive-pool NP drugs")

    np_to_sdt = max_similarity_to_reference(np_fps, sdt_fps)
    np_to_np = nearest_neighbor_similarity(np_fps)
    np_to_sdt_morgan_distance = morgan_distance_to_sdt_centroid(np_fps, sdt_fps)
    sdt_proteins = load_sdt_proteins()
    np_proteins = load_positive_pool_np_proteins()

    detail = np_drugs[["DrugKey", "CanonicalSMILES"]].copy()
    detail["NP_to_SDT_max_similarity"] = np_to_sdt
    detail["NP_to_NP_nearest_similarity"] = np_to_np
    detail["NP_to_SDT_Morgan_distance"] = np_to_sdt_morgan_distance
    detail.to_csv(OUTPUT_DIR / "positive_pool_np_similarity_per_drug.csv", index=False)

    summary = pd.DataFrame(
        [
            {"Metric": "Positive-pool NP drugs with valid Morgan fingerprints", "Value": len(np_drugs)},
            {"Metric": "SDT drugs with valid Morgan fingerprints", "Value": len(sdt_drugs)},
            {"Metric": "NP-to-SDT maximum similarity median", "Value": float(np.median(np_to_sdt))},
            {"Metric": "NP-to-NP nearest-neighbor similarity median", "Value": float(np.median(np_to_np))},
            {"Metric": "NP-to-SDT maximum similarity mean", "Value": float(np.mean(np_to_sdt))},
            {"Metric": "NP-to-NP nearest-neighbor similarity mean", "Value": float(np.mean(np_to_np))},
            {"Metric": "NP drugs with Morgan SDT-centroid distance", "Value": len(np_to_sdt_morgan_distance)},
            {"Metric": "NP-to-SDT Morgan distance median", "Value": float(np.median(np_to_sdt_morgan_distance))},
            {"Metric": "NP-to-SDT Morgan distance mean", "Value": float(np.mean(np_to_sdt_morgan_distance))},
            {"Metric": "Positive-pool NP targets", "Value": len(np_proteins)},
            {"Metric": "NP targets shared with SDT", "Value": len(np_proteins & sdt_proteins)},
            {"Metric": "NP targets not in SDT", "Value": len(np_proteins - sdt_proteins)},
        ]
    )
    summary.to_csv(OUTPUT_DIR / "positive_pool_np_similarity_summary.csv", index=False)

    plot_chemical_umap(sdt_fps, np_fps)
    plot_similarity_distributions(np_to_sdt, np_to_np)
    plot_morgan_distance_distribution(np_to_sdt_morgan_distance)
    plot_target_overlap(np_proteins, sdt_proteins)

    print("Saved figures and tables to:", OUTPUT_DIR)
    print("  - positive_pool_np_chemical_space_umap.pdf/png")
    print("  - positive_pool_np_similarity_distribution.pdf/png")
    print("  - positive_pool_np_to_sdt_morgan_distance.pdf/png")
    print("  - positive_pool_np_target_overlap.pdf/png")
    print("  - positive_pool_np_similarity_per_drug.csv")
    print("  - positive_pool_np_similarity_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
