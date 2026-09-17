#!/usr/bin/env python
# Author  : Codex
# File    : build_radial_fixed_distance_split.py
# Updated : 2026/7/22

"""
Build drug-only fixed-distance radial train/val/test1/test2 splits.

Protocol:
1. Represent drugs with Morgan fingerprints or pooled MolFormer embeddings.
2. Compute extended Tanimoto distance for Morgan fingerprints and cosine
   distance for MolFormer embeddings.
3. Split entities into core / near / far using representation-specific fixed
   distance thresholds, rather than dataset-specific percentages.
4. Randomly split core entities into train / val by 7 : 1.
5. Map entity regions back to pair files:
   - train: entities in core_train
   - val:   entities in core_val
   - test1: entities in near
   - test2: entities in far
6. Save UMAP visualization for reporting only. UMAP is not used for splitting.

This script writes one deterministic drug partition per dataset/representation,
and five group folders whose only difference is the core train/val seed.
Output names include the representation type but do not include threshold values.
"""

import os
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import umap


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATABASE_DIR = os.path.join(ROOT_DIR, "DTIPred_Plus/database")

# Use ["SNAP", "DRH", "TTD", "SDT"] to build all source datasets together.
DATASETS = ["SDT"]
DRUG_REPRESENTATIONS = ["morgan", "molformer"]  #"morgan", 
# DRUG_REPRESENTATIONS = ["molformer"]  #"morgan", 

RATIO_DIR = "1_1"
GROUP_IDS = [1, 2, 3, 4, 5]
VAL_FRAC_WITHIN_CORE = 0.125

MOLFORMER_FEATURE_ROOT = "/data/qfchen/DTIPred_Plus"
TPP_MOLFORMER_FEAT_DIR = os.path.join(
    MOLFORMER_FEATURE_ROOT, "TPP", "MolFormer"
)
TPP_DRUG_INFO_FILE = os.path.join(DATABASE_DIR, "TPP", "DrugInfo.xlsx")
TPP_FINAL_DATA_DIR = os.path.join(DATABASE_DIR, "TPP")

MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048
SUPPORTED_DRUG_REPRESENTATIONS = {"morgan", "molformer"}

# Fixed thresholds are distances, so larger values mean less similarity.
# Morgan uses extended-Tanimoto distance to the source fingerprint centroid.
# MolFormer uses cosine distance to the source embedding centroid.
FIXED_DISTANCE_THRESHOLDS = {
    "morgan": {
        "core_max": 0.84,
        "near_max": 0.86,
    },
    "molformer": {
        "core_max": 0.25,
        "near_max": 0.30,
    },
}

DRUG_ID_COL = "DRUGID"
LABEL_COL = "Label"

UMAP_RANDOM_STATE = 3407
UMAP_N_NEIGHBORS = 100
UMAP_MIN_DIST = 0.10
UMAP_FIGSIZE = (8, 8)
TSNE_FIGSIZE = (8, 8)
PCA_FIGSIZE = (8, 8)

REGION_COLOR_MAP = {
    "core": "#4CAF50",
    "near": "#42A5F5",
    "far": "#FF7043",
}

POINT_SIZE = 50
POINT_ALPHA = 0.82
POINT_EDGE_WIDTH = 0.25
CENTER_MARKER_SIZE = 220
FIG_DPI = 600

DISPLAY_NAME_MAP = {
    "core": "Core",
    "near": "1st Cousin",
    "far": "2nd Cousin",
}


@dataclass
class SplitConfig:
    dataset: str
    drug_representation: str = "morgan"

    @property
    def drug_info_file(self):
        return os.path.join(DATABASE_DIR, self.dataset, "final_data", "DrugInfo.xlsx")

    @property
    def representation_tag(self):
        tag = str(self.drug_representation).strip().lower()
        if tag not in SUPPORTED_DRUG_REPRESENTATIONS:
            raise ValueError(
                "drug_representation must be one of {}, got: {}".format(
                    sorted(SUPPORTED_DRUG_REPRESENTATIONS),
                    self.drug_representation,
                )
            )
        return tag

    @property
    def drug_molformer_feat_dir(self):
        return os.path.join(
            MOLFORMER_FEATURE_ROOT,
            self.dataset,
            "MolFormer",
        )

    @property
    def fixed_thresholds(self):
        values = FIXED_DISTANCE_THRESHOLDS[self.representation_tag]
        core_max = float(values["core_max"])
        near_max = float(values["near_max"])
        if not 0.0 <= core_max < near_max:
            raise ValueError(
                "Expected 0 <= core_max < near_max, got {} and {}".format(
                    core_max,
                    near_max,
                )
            )
        return core_max, near_max

    @property
    def split_name(self):
        return f"drug_{self.representation_tag}_distance_radial_split"

    @property
    def artifact_prefix(self):
        return (
            f"{self.dataset}_drug_{self.representation_tag}_"
            "distance"
        )

    @property
    def entity_col(self):
        return DRUG_ID_COL

    @property
    def split_root(self):
        return os.path.join(
            DATABASE_DIR,
            self.dataset,
            "final_data",
            RATIO_DIR,
            self.split_name,
        )

    @property
    def figure_dir(self):
        return os.path.join(self.split_root, "figures")

    @property
    def summary_dir(self):
        return os.path.join(self.split_root, "summary")


def _load_torch_feature(path):
    """Load a MolFormer .pt feature on CPU; PyTorch is optional for Morgan runs."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "MolFormer radial splitting requires PyTorch to read .pt features."
        ) from exc

    try:
        feature = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        feature = torch.load(path, map_location="cpu")

    if isinstance(feature, dict):
        tensor_values = [
            value for value in feature.values() if torch.is_tensor(value)
        ]
        if len(tensor_values) != 1:
            raise ValueError(
                f"Expected exactly one tensor in MolFormer feature file: {path}"
            )
        feature = tensor_values[0]
    if torch.is_tensor(feature):
        feature = feature.detach().cpu().float().numpy()
    return feature


def _to_numpy_array(x):
    if isinstance(x, np.ndarray):
        x = x.astype(np.float32, copy=False)
    else:
        x = np.asarray(x, dtype=np.float32)
    return x


def mean_pool_molformer_feature(feat):
    """Pool token-level MolFormer features to one finite vector per drug."""
    feat = _to_numpy_array(feat)
    if feat.ndim == 1:
        pooled = feat
    elif feat.ndim >= 2:
        pooled = feat.reshape(-1, feat.shape[-1]).mean(axis=0)
    else:
        raise ValueError(f"Invalid MolFormer feature shape: {feat.shape}")

    pooled = np.nan_to_num(
        pooled,
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    ).astype(np.float32, copy=False)
    if np.linalg.norm(pooled) <= 0:
        raise ValueError("MolFormer pooled feature has zero norm")
    return pooled


def require_rdkit():
    if Chem is None or DataStructs is None or GetMorganGenerator is None:
        raise ImportError(
            "Drug radial splitting requires RDKit to generate Morgan fingerprints."
        )


def smiles_to_morgan_vector(smiles, generator):
    mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
    if mol is None:
        return None, None
    fingerprint = generator.GetFingerprint(mol)
    vector = np.zeros((MORGAN_N_BITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, vector)
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    return vector, canonical_smiles


def load_pair_table(cfg):
    raise RuntimeError("load_pair_table now requires a set file path. Use load_pair_table_from_file(...).")


def load_pair_table_from_file(cfg, pair_file):
    df = pd.read_excel(pair_file)
    expected = {DRUG_ID_COL, LABEL_COL}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"{pair_file} is missing required columns: {sorted(missing)}")

    print(f"[{cfg.dataset}/drug] source: {pair_file}")
    print(f"[{cfg.dataset}/drug] pairs loaded: {len(df)}")
    return df.reset_index(drop=True)


def load_entity_catalog(cfg):
    info_df = pd.read_excel(cfg.drug_info_file)
    required = {DRUG_ID_COL}
    if cfg.representation_tag == "morgan":
        required.add("SMILES")
    missing = required.difference(info_df.columns)
    if missing:
        raise ValueError(
            f"{cfg.drug_info_file} is missing required columns: {sorted(missing)}"
        )
    entity_ids = (
        info_df[DRUG_ID_COL]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    print(f"[{cfg.dataset}/drug] entity catalog source: {cfg.drug_info_file}")
    print(f"[{cfg.dataset}/drug] entity catalog size: {len(entity_ids)}")
    return entity_ids


def build_entity_embedding_table(cfg, entity_ids):
    if cfg.representation_tag == "morgan":
        return build_drug_fingerprint_table(cfg, entity_ids)
    if cfg.representation_tag == "molformer":
        return build_drug_molformer_table(cfg, entity_ids)
    raise ValueError(
        f"Unsupported drug representation: {cfg.representation_tag}"
    )


def build_drug_fingerprint_table(cfg, entity_ids):
    require_rdkit()
    generator = GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_N_BITS)
    info_df = pd.read_excel(cfg.drug_info_file)
    info_df = info_df[[DRUG_ID_COL, "SMILES"]].copy()
    info_df[DRUG_ID_COL] = info_df[DRUG_ID_COL].astype("string").str.strip()
    info_df["SMILES"] = info_df["SMILES"].astype("string").str.strip()
    info_df = info_df[
        info_df[DRUG_ID_COL].isin(set(entity_ids))
    ].drop_duplicates(DRUG_ID_COL)

    rows = []
    vectors = []
    invalid_ids = []
    for _, row in info_df.iterrows():
        entity_id = row[DRUG_ID_COL]
        vector, canonical_smiles = smiles_to_morgan_vector(
            row["SMILES"], generator
        )
        if vector is None:
            invalid_ids.append(entity_id)
            continue
        rows.append(
            {
                "entity_id": entity_id,
                "SMILES": row["SMILES"],
                "CanonicalSMILES": canonical_smiles,
                "representation": (
                    f"Morgan_radius{MORGAN_RADIUS}_{MORGAN_N_BITS}bits"
                ),
            }
        )
        vectors.append(vector)

    if not vectors:
        raise ValueError(f"No valid drug SMILES found for {cfg.dataset}.")
    if invalid_ids:
        print(
            f"[{cfg.dataset}/drug] skipped {len(invalid_ids)} invalid SMILES"
        )

    vectors = np.vstack(vectors).astype(np.float32, copy=False)
    entity_df = pd.DataFrame(rows)
    entity_df["embedding_dim"] = vectors.shape[1]
    return entity_df, vectors


def build_drug_molformer_table(cfg, entity_ids):
    info_df = pd.read_excel(cfg.drug_info_file)
    keep_cols = [DRUG_ID_COL]
    if "SMILES" in info_df.columns:
        keep_cols.append("SMILES")
    info_df = info_df[keep_cols].copy()
    info_df = info_df.dropna(subset=[DRUG_ID_COL])
    info_df[DRUG_ID_COL] = info_df[DRUG_ID_COL].astype(str).str.strip()
    info_df = info_df[
        info_df[DRUG_ID_COL].isin(set(entity_ids))
    ].drop_duplicates(DRUG_ID_COL)

    rows = []
    vectors = []
    missing_ids = []
    invalid_ids = []
    for _, row in info_df.iterrows():
        entity_id = str(row[DRUG_ID_COL]).strip()
        feature_path = os.path.join(
            cfg.drug_molformer_feat_dir,
            f"{entity_id}.pt",
        )
        if not os.path.exists(feature_path):
            missing_ids.append(entity_id)
            continue
        try:
            vector = mean_pool_molformer_feature(_load_torch_feature(feature_path))
        except (OSError, RuntimeError, TypeError, ValueError):
            invalid_ids.append(entity_id)
            continue

        record = row.to_dict()
        record["entity_id"] = entity_id
        record["feature_path"] = feature_path
        record["representation"] = "MolFormer_mean_pool"
        rows.append(record)
        vectors.append(vector)

    if not vectors:
        raise ValueError(
            "No valid MolFormer features found for {} in {}".format(
                cfg.dataset,
                cfg.drug_molformer_feat_dir,
            )
        )
    if missing_ids:
        print(
            f"[{cfg.dataset}/drug] skipped {len(missing_ids)} missing "
            "MolFormer feature files"
        )
    if invalid_ids:
        print(
            f"[{cfg.dataset}/drug] skipped {len(invalid_ids)} invalid "
            "MolFormer feature files"
        )

    dimensions = {vector.shape for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError(
            f"MolFormer feature dimensions are inconsistent: {dimensions}"
        )
    vectors = np.vstack(vectors).astype(np.float32, copy=False)
    vectors /= np.maximum(
        np.linalg.norm(vectors, axis=1, keepdims=True),
        1e-12,
    )
    entity_df = pd.DataFrame(rows)
    entity_df["embedding_dim"] = vectors.shape[1]
    return entity_df, vectors


def load_tpp_drug_catalog():
    if not os.path.exists(TPP_DRUG_INFO_FILE):
        raise FileNotFoundError(f"TPP DrugInfo file not found: {TPP_DRUG_INFO_FILE}")
    info_df = pd.read_excel(TPP_DRUG_INFO_FILE)
    if DRUG_ID_COL not in info_df.columns:
        raise ValueError(f"{TPP_DRUG_INFO_FILE} is missing required column: {DRUG_ID_COL}")

    keep_cols = [DRUG_ID_COL]
    for optional_col in ["SMILES", "GROUP"]:
        if optional_col in info_df.columns:
            keep_cols.append(optional_col)

    info_df = info_df[keep_cols].copy()
    info_df = info_df.dropna(subset=[DRUG_ID_COL])
    info_df[DRUG_ID_COL] = info_df[DRUG_ID_COL].astype(str).str.strip()
    info_df = info_df.drop_duplicates(subset=[DRUG_ID_COL]).reset_index(drop=True)
    return info_df


def build_tpp_fingerprint_table():
    require_rdkit()
    tpp_df = load_tpp_drug_catalog()
    if "SMILES" not in tpp_df.columns:
        raise ValueError(f"{TPP_DRUG_INFO_FILE} is missing required column: SMILES")
    generator = GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_N_BITS)
    rows = []
    vectors = []

    for _, row in tpp_df.iterrows():
        drug_id = str(row[DRUG_ID_COL]).strip()
        vector, canonical_smiles = smiles_to_morgan_vector(
            row["SMILES"], generator
        )
        if vector is None:
            continue

        record = row.to_dict()
        record["entity_id"] = drug_id
        record["CanonicalSMILES"] = canonical_smiles
        record["representation"] = (
            f"Morgan_radius{MORGAN_RADIUS}_{MORGAN_N_BITS}bits"
        )
        rows.append(record)
        vectors.append(vector)

    if not vectors:
        return pd.DataFrame(), None

    vectors = np.vstack(vectors).astype(np.float32, copy=False)
    out_df = pd.DataFrame(rows)
    out_df["embedding_dim"] = vectors.shape[1]
    return out_df, vectors


def build_tpp_molformer_table():
    tpp_df = load_tpp_drug_catalog()
    rows = []
    vectors = []
    missing_ids = []
    invalid_ids = []

    for _, row in tpp_df.iterrows():
        drug_id = str(row[DRUG_ID_COL]).strip()
        feature_path = os.path.join(
            TPP_MOLFORMER_FEAT_DIR,
            f"{drug_id}.pt",
        )
        if not os.path.exists(feature_path):
            missing_ids.append(drug_id)
            continue
        try:
            vector = mean_pool_molformer_feature(
                _load_torch_feature(feature_path)
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            invalid_ids.append(drug_id)
            continue

        record = row.to_dict()
        record["entity_id"] = drug_id
        record["feature_path"] = feature_path
        record["representation"] = "MolFormer_mean_pool"
        rows.append(record)
        vectors.append(vector)

    if missing_ids:
        print(
            f"[TPP/drug] skipped {len(missing_ids)} missing MolFormer "
            "feature files"
        )
    if invalid_ids:
        print(
            f"[TPP/drug] skipped {len(invalid_ids)} invalid MolFormer "
            "feature files"
        )
    if not vectors:
        return pd.DataFrame(), None

    dimensions = {vector.shape for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError(
            f"TPP MolFormer feature dimensions are inconsistent: {dimensions}"
        )
    vectors = np.vstack(vectors).astype(np.float32, copy=False)
    vectors /= np.maximum(
        np.linalg.norm(vectors, axis=1, keepdims=True),
        1e-12,
    )
    out_df = pd.DataFrame(rows)
    out_df["embedding_dim"] = vectors.shape[1]
    return out_df, vectors


def build_tpp_drug_representation_table(cfg):
    if cfg.representation_tag == "morgan":
        return build_tpp_fingerprint_table()
    if cfg.representation_tag == "molformer":
        return build_tpp_molformer_table()
    raise ValueError(
        f"Unsupported drug representation: {cfg.representation_tag}"
    )


def extended_tanimoto_distances(vectors, center):
    """Compute Tanimoto distance between nonnegative vectors and a centroid."""
    vectors = np.asarray(vectors, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32).reshape(1, -1)
    dot_products = (vectors @ center.T).reshape(-1)
    vector_norms = np.sum(vectors * vectors, axis=1)
    center_norm = float(np.sum(center * center))
    denominator = vector_norms + center_norm - dot_products
    similarities = np.divide(
        dot_products,
        denominator,
        out=np.zeros_like(dot_products),
        where=denominator > 0,
    )
    return 1.0 - np.clip(similarities, 0.0, 1.0)


def distance_metric_name(cfg):
    if cfg.representation_tag == "morgan":
        return "extended_tanimoto"
    return "cosine"


def projection_metric(cfg):
    if cfg.representation_tag == "morgan":
        return "jaccard"
    return "cosine"


def representation_space_name(cfg):
    if cfg.representation_tag == "morgan":
        return "Morgan fingerprint space"
    if cfg.representation_tag == "molformer":
        return "MolFormer embedding space"
    raise ValueError(
        f"Unsupported drug representation: {cfg.representation_tag}"
    )


def distances_to_center(vectors, center, cfg):
    if cfg.representation_tag == "morgan":
        return extended_tanimoto_distances(vectors, center)
    return cosine_distances(vectors, center.reshape(1, -1)).reshape(-1)


def assign_radial_regions(vectors, cfg):
    center = vectors.mean(axis=0, keepdims=True)
    distances = distances_to_center(vectors, center.reshape(-1), cfg)
    threshold_core, threshold_near = cfg.fixed_thresholds
    labels = assign_regions_by_threshold(
        distances,
        threshold_core,
        threshold_near,
    )
    return center.reshape(-1), distances, labels, threshold_core, threshold_near


def assign_regions_by_threshold(distances, threshold_core, threshold_near):
    labels = np.empty(len(distances), dtype=object)
    labels[distances <= threshold_core] = "core"
    labels[(distances > threshold_core) & (distances <= threshold_near)] = "near"
    labels[distances > threshold_near] = "far"
    return labels


def report_distance_diagnostics(cfg, distances, labels):
    quantile_levels = [0.00, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 1.00]
    quantile_values = np.quantile(distances, quantile_levels)
    print(
        "[{} / {}] fixed thresholds: core <= {:.4f}; near <= {:.4f}; far > {:.4f}".format(
            cfg.dataset,
            cfg.representation_tag,
            cfg.fixed_thresholds[0],
            cfg.fixed_thresholds[1],
            cfg.fixed_thresholds[1],
        )
    )
    print(
        "[{} / {}] distance quantiles: {}".format(
            cfg.dataset,
            cfg.representation_tag,
            ", ".join(
                "q{:02d}={:.4f}".format(int(level * 100), value)
                for level, value in zip(quantile_levels, quantile_values)
            ),
        )
    )

    counts = pd.Series(labels).value_counts().reindex(
        ["core", "near", "far"],
        fill_value=0,
    )
    total = len(labels)
    print(
        "[{} / {}] region counts: {}".format(
            cfg.dataset,
            cfg.representation_tag,
            ", ".join(
                "{}={} ({:.1%})".format(region, int(count), count / total)
                for region, count in counts.items()
            ),
        )
    )

    empty_regions = counts[counts == 0].index.tolist()
    if empty_regions:
        raise ValueError(
            "Fixed thresholds produced empty regions {} for {}/{}. "
            "Adjust FIXED_DISTANCE_THRESHOLDS after inspecting the printed quantiles.".format(
                empty_regions,
                cfg.dataset,
                cfg.representation_tag,
            )
        )
    for region, count in counts.items():
        if count / total < 0.02:
            print(
                "[WARNING] {} contains less than 2% of entities; estimates may be unstable.".format(
                    region
                )
            )


def split_core_entities(entity_df, seed):
    core_ids = entity_df.loc[entity_df["region"] == "core", "entity_id"].tolist()
    if len(core_ids) < 2:
        raise ValueError("Need at least 2 core entities to split train/val.")

    rng = np.random.default_rng(seed)
    shuffled = np.array(core_ids, dtype=object)
    rng.shuffle(shuffled)

    n_val = max(1, int(math.floor(len(shuffled) * VAL_FRAC_WITHIN_CORE)))
    if n_val >= len(shuffled):
        n_val = len(shuffled) - 1

    val_ids = set(shuffled[:n_val].tolist())
    train_ids = set(shuffled[n_val:].tolist())
    return train_ids, val_ids


def build_pair_splits(pair_df, entity_col, train_ids, val_ids, near_ids, far_ids):
    entity_series = pair_df[entity_col].astype(str).str.strip()

    train_df = pair_df.loc[entity_series.isin(train_ids)].reset_index(drop=True)
    val_df = pair_df.loc[entity_series.isin(val_ids)].reset_index(drop=True)
    test1_df = pair_df.loc[entity_series.isin(near_ids)].reset_index(drop=True)
    test2_df = pair_df.loc[entity_series.isin(far_ids)].reset_index(drop=True)

    return train_df, val_df, test1_df, test2_df


def save_group_files(cfg, group_idx, train_df, val_df, test1_df, test2_df):
    group_dir = os.path.join(cfg.split_root, f"group{group_idx}")
    os.makedirs(group_dir, exist_ok=True)

    train_df.to_excel(os.path.join(group_dir, "train.xlsx"), index=False)
    val_df.to_excel(os.path.join(group_dir, "val.xlsx"), index=False)
    test1_df.to_excel(os.path.join(group_dir, "test1.xlsx"), index=False)
    test2_df.to_excel(os.path.join(group_dir, "test2.xlsx"), index=False)


def _style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("#4A4A4A")
    ax.tick_params(axis="both", labelsize=13, width=1.0, colors="#303030")


def _save_figure(fig, base_path_without_ext):
    png_path = f"{base_path_without_ext}.png"
    pdf_path = f"{base_path_without_ext}.pdf"
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    return png_path, pdf_path


def draw_umap(entity_df, vectors, cfg):
    metric = projection_metric(cfg)
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=metric,
        random_state=UMAP_RANDOM_STATE,
    )
    coords = reducer.fit_transform(vectors)
    entity_df["UMAP1"] = coords[:, 0]
    entity_df["UMAP2"] = coords[:, 1]

    # Visual center is defined on the 2D UMAP projection only for plotting.
    xy = entity_df[["UMAP1", "UMAP2"]].values
    global_center = xy.mean(axis=0)
    center_idx = np.argmin(np.linalg.norm(xy - global_center, axis=1))
    center_xy = xy[center_idx]
    center_entity = entity_df.iloc[center_idx]["entity_id"]

    umap_dist = np.linalg.norm(xy - center_xy, axis=1)
    entity_df["umap_distance_to_center"] = umap_dist

    os.makedirs(cfg.figure_dir, exist_ok=True)
    fig_base_path = os.path.join(
        cfg.figure_dir, f"{cfg.artifact_prefix}_radial_umap"
    )

    plt.figure(figsize=UMAP_FIGSIZE)
    for region in ["core", "near", "far"]:
        sub = entity_df[entity_df["region"] == region]
        plt.scatter(
            sub["UMAP1"],
            sub["UMAP2"],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            c=REGION_COLOR_MAP[region],
            edgecolors="white",
            linewidths=POINT_EDGE_WIDTH,
            label=f"{DISPLAY_NAME_MAP[region]} (n={len(sub)})",
        )

    # Center marker is a visual guide only.
    plt.scatter(
        center_xy[0],
        center_xy[1],
        s=CENTER_MARKER_SIZE,
        marker="*",
        c="red",
        edgecolors="black",
        linewidths=0.9,
        label=f"center",
        zorder=5,
    )

    title = f"{representation_space_name(cfg)} ({cfg.dataset}, UMAP)"
    plt.title(title, fontsize=24, pad=14)
    plt.xlabel("UMAP 1", fontsize=18, labelpad=12)
    plt.ylabel("UMAP 2", fontsize=18, labelpad=12)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16, frameon=True)
    plt.tight_layout()
    png_path, pdf_path = _save_figure(plt.gcf(), fig_base_path)
    plt.close()
    return entity_df, png_path, pdf_path, reducer


def _scatter_projection(entity_df, x_col, y_col, cfg, fig_base_path, title, figsize):
    xy = entity_df[[x_col, y_col]].values
    global_center = xy.mean(axis=0)
    center_idx = np.argmin(np.linalg.norm(xy - global_center, axis=1))
    center_xy = xy[center_idx]
    center_entity = entity_df.iloc[center_idx]["entity_id"]

    plt.figure(figsize=figsize)
    for region in ["core", "near", "far"]:
        sub = entity_df[entity_df["region"] == region]
        plt.scatter(
            sub[x_col],
            sub[y_col],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            c=REGION_COLOR_MAP[region],
            edgecolors="white",
            linewidths=POINT_EDGE_WIDTH,
            label=f"{DISPLAY_NAME_MAP[region]} (n={len(sub)})",
        )

    plt.scatter(
        center_xy[0],
        center_xy[1],
        s=CENTER_MARKER_SIZE,
        marker="*",
        c="red",
        edgecolors="black",
        linewidths=0.9,
        label=f"center",
        zorder=5,
    )

    plt.title(title, fontsize=24, pad=14)
    plt.xlabel(x_col, fontsize=18, labelpad=12)
    plt.ylabel(y_col, fontsize=18, labelpad=12)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16, frameon=True)
    plt.tight_layout()
    png_path, pdf_path = _save_figure(plt.gcf(), fig_base_path)
    plt.close()
    return png_path, pdf_path


def draw_pca(entity_df, vectors, cfg):
    reducer = PCA(n_components=2, random_state=UMAP_RANDOM_STATE)
    coords = reducer.fit_transform(vectors)
    pca_df = entity_df.copy()
    pca_df["PCA1"] = coords[:, 0]
    pca_df["PCA2"] = coords[:, 1]

    fig_base_path = os.path.join(
        cfg.figure_dir, f"{cfg.artifact_prefix}_radial_pca"
    )
    title = f"{representation_space_name(cfg)} ({cfg.dataset}, PCA)"
    png_path, pdf_path = _scatter_projection(pca_df, "PCA1", "PCA2", cfg, fig_base_path, title, PCA_FIGSIZE)
    return pca_df, png_path, pdf_path, reducer


def draw_tsne(entity_df, vectors, cfg):
    perplexity = min(30, max(5, len(entity_df) - 1))
    if perplexity >= len(entity_df):
        perplexity = max(1, len(entity_df) - 1)

    metric = projection_metric(cfg)
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric=metric,
        init="pca",
        learning_rate="auto",
        random_state=UMAP_RANDOM_STATE,
    )
    coords = reducer.fit_transform(vectors)
    tsne_df = entity_df.copy()
    tsne_df["TSNE1"] = coords[:, 0]
    tsne_df["TSNE2"] = coords[:, 1]

    fig_base_path = os.path.join(
        cfg.figure_dir, f"{cfg.artifact_prefix}_radial_tsne"
    )
    title = f"{representation_space_name(cfg)} ({cfg.dataset}, t-SNE)"
    png_path, pdf_path = _scatter_projection(tsne_df, "TSNE1", "TSNE2", cfg, fig_base_path, title, TSNE_FIGSIZE)
    return tsne_df, png_path, pdf_path, reducer


def save_tpp_projection(cfg, benchmark_entity_df, reducer, center_vec, threshold_core, threshold_near):
    tpp_df, tpp_vectors = build_tpp_drug_representation_table(cfg)
    if tpp_vectors is None or len(tpp_df) == 0:
        print(
            f"[{cfg.dataset}/drug] no valid TPP "
            f"{cfg.representation_tag} features found, skip TPP projection"
        )
        return

    if tpp_vectors.shape[1] != len(center_vec):
        raise ValueError(
            "Source and TPP {} dimensions differ: {} versus {}".format(
                cfg.representation_tag,
                len(center_vec),
                tpp_vectors.shape[1],
            )
        )

    tpp_distances = distances_to_center(tpp_vectors, center_vec, cfg)
    tpp_regions = assign_regions_by_threshold(tpp_distances, threshold_core, threshold_near)
    tpp_df["distance_to_center"] = tpp_distances
    tpp_df["region"] = tpp_regions
    tpp_df["display_region"] = tpp_df["region"].map(DISPLAY_NAME_MAP)
    tpp_df["representation_tag"] = cfg.representation_tag
    tpp_df["distance_metric"] = distance_metric_name(cfg)
    tpp_df["core_max_distance"] = threshold_core
    tpp_df["near_max_distance"] = threshold_near

    tpp_coords = reducer.transform(tpp_vectors)
    tpp_df["UMAP1"] = tpp_coords[:, 0]
    tpp_df["UMAP2"] = tpp_coords[:, 1]

    os.makedirs(TPP_FINAL_DATA_DIR, exist_ok=True)
    projection_path = os.path.join(
        TPP_FINAL_DATA_DIR,
        f"{cfg.artifact_prefix}_tpp_projection.xlsx",
    )
    count_path = os.path.join(
        TPP_FINAL_DATA_DIR,
        f"{cfg.artifact_prefix}_tpp_region_counts.xlsx",
    )
    tpp_df.to_excel(projection_path, index=False)

    count_df = (tpp_df["display_region"]
        .value_counts()
        .rename_axis("Region")
        .reset_index(name="Count")
    )
    count_df.to_excel(count_path, index=False)

    print(f"[{cfg.dataset}/drug] TPP projection saved to: {projection_path}")
    print(f"[{cfg.dataset}/drug] TPP region counts saved to: {count_path}")
    print(f"[{cfg.dataset}/drug] TPP region counts:\n{count_df.to_string(index=False)}")


def save_entity_summary(cfg, entity_df):
    os.makedirs(cfg.summary_dir, exist_ok=True)
    entity_summary_path = os.path.join(
        cfg.summary_dir, f"{cfg.artifact_prefix}_entity_summary.xlsx"
    )
    entity_df.to_excel(entity_summary_path, index=False)
    print(f"[{cfg.dataset}/drug] entity summary saved to: {entity_summary_path}")


def save_distance_region_summary(cfg, entity_df):
    os.makedirs(cfg.summary_dir, exist_ok=True)
    core_max, near_max = cfg.fixed_thresholds
    rows = []
    total = len(entity_df)
    for region in ["core", "near", "far"]:
        values = entity_df.loc[
            entity_df["region"] == region,
            "distance_to_center",
        ].to_numpy(dtype=float)
        rows.append(
            {
                "dataset": cfg.dataset,
                "representation": cfg.representation_tag,
                "distance_metric": distance_metric_name(cfg),
                "region": region,
                "count": len(values),
                "fraction": len(values) / total if total else float("nan"),
                "distance_min": float(np.min(values)) if len(values) else float("nan"),
                "distance_mean": float(np.mean(values)) if len(values) else float("nan"),
                "distance_median": float(np.median(values)) if len(values) else float("nan"),
                "distance_max": float(np.max(values)) if len(values) else float("nan"),
                "core_max_distance": core_max,
                "near_max_distance": near_max,
            }
        )

    summary_path = os.path.join(
        cfg.summary_dir,
        f"{cfg.artifact_prefix}_distance_region_summary.xlsx",
    )
    pd.DataFrame(rows).to_excel(summary_path, index=False)
    print(
        f"[{cfg.dataset}/drug] distance-region summary saved to: "
        f"{summary_path}"
    )


def summarize_group_stats(cfg, group_idx, group_rows):
    os.makedirs(cfg.summary_dir, exist_ok=True)
    split_summary_path = os.path.join(
        cfg.summary_dir,
        f"{cfg.artifact_prefix}_group{group_idx}_split_summary.xlsx",
    )
    pd.DataFrame(group_rows).to_excel(split_summary_path, index=False)
    print(f"[{cfg.dataset}/drug] split summary saved to:  {split_summary_path}")


def run_for_config(cfg):
    print("\n" + "=" * 90)
    print(f"Building {cfg.split_name} for {cfg.dataset}")
    print("=" * 90)

    entity_ids = load_entity_catalog(cfg)
    entity_df, vectors = build_entity_embedding_table(cfg, entity_ids)
    center, distances, regions, threshold_core, threshold_near = (
        assign_radial_regions(vectors, cfg)
    )
    report_distance_diagnostics(cfg, distances, regions)

    entity_df["distance_to_center"] = distances
    entity_df["region"] = regions
    entity_df["center_dim"] = len(center)
    entity_df["representation_tag"] = cfg.representation_tag
    entity_df["split_name"] = cfg.split_name
    entity_df["distance_metric"] = distance_metric_name(cfg)
    entity_df["core_max_distance"] = threshold_core
    entity_df["near_max_distance"] = threshold_near

    entity_df, umap_png_path, umap_pdf_path, reducer = draw_umap(entity_df, vectors, cfg)
    print(f"[{cfg.dataset}/drug] UMAP saved to: {umap_png_path}")
    print(f"[{cfg.dataset}/drug] UMAP saved to: {umap_pdf_path}")
    pca_df, pca_png_path, pca_pdf_path, _ = draw_pca(entity_df, vectors, cfg)
    print(f"[{cfg.dataset}/drug] PCA saved to: {pca_png_path}")
    print(f"[{cfg.dataset}/drug] PCA saved to: {pca_pdf_path}")
    tsne_df, tsne_png_path, tsne_pdf_path, _ = draw_tsne(entity_df, vectors, cfg)
    print(f"[{cfg.dataset}/drug] t-SNE saved to: {tsne_png_path}")
    print(f"[{cfg.dataset}/drug] t-SNE saved to: {tsne_pdf_path}")
    save_tpp_projection(cfg, entity_df, reducer, center, threshold_core, threshold_near)
    save_entity_summary(cfg, entity_df)
    save_distance_region_summary(cfg, entity_df)

    pca_summary_path = os.path.join(
        cfg.summary_dir, f"{cfg.artifact_prefix}_pca_projection.xlsx"
    )
    tsne_summary_path = os.path.join(
        cfg.summary_dir, f"{cfg.artifact_prefix}_tsne_projection.xlsx"
    )
    os.makedirs(cfg.summary_dir, exist_ok=True)
    pca_df.to_excel(pca_summary_path, index=False)
    tsne_df.to_excel(tsne_summary_path, index=False)

    near_ids = set(entity_df.loc[entity_df["region"] == "near", "entity_id"].tolist())
    far_ids = set(entity_df.loc[entity_df["region"] == "far", "entity_id"].tolist())
    core_ids = set(entity_df.loc[entity_df["region"] == "core", "entity_id"].tolist())

    for group_idx in GROUP_IDS:
        pair_file = os.path.join(DATABASE_DIR, cfg.dataset, "final_data", RATIO_DIR, f"Set{group_idx}.xlsx")
        if not os.path.exists(pair_file):
            print(f"[{cfg.dataset}/drug] skip missing set file: {pair_file}")
            continue

        pair_df = load_pair_table_from_file(cfg, pair_file)

        train_ids, val_ids = split_core_entities(entity_df, seed=3400 + group_idx)
        train_df, val_df, test1_df, test2_df = build_pair_splits(
            pair_df,
            cfg.entity_col,
            train_ids,
            val_ids,
            near_ids,
            far_ids,
        )
        save_group_files(cfg, group_idx, train_df, val_df, test1_df, test2_df)

        group_rows = [
            {
                "group": group_idx,
                "seed": 3400 + group_idx,
                "split_name": cfg.split_name,
                "representation": cfg.representation_tag,
                "distance_metric": distance_metric_name(cfg),
                "core_max_distance": threshold_core,
                "near_max_distance": threshold_near,
                "source_set": f"Set{group_idx}.xlsx",
                "core_entities": len(core_ids),
                "train_entities": len(train_ids),
                "val_entities": len(val_ids),
                "near_entities": len(near_ids),
                "far_entities": len(far_ids),
                "train_pairs": len(train_df),
                "val_pairs": len(val_df),
                "test1_pairs": len(test1_df),
                "test2_pairs": len(test2_df),
                "train_pos": int(train_df[LABEL_COL].sum()) if len(train_df) else 0,
                "val_pos": int(val_df[LABEL_COL].sum()) if len(val_df) else 0,
                "test1_pos": int(test1_df[LABEL_COL].sum()) if len(test1_df) else 0,
                "test2_pos": int(test2_df[LABEL_COL].sum()) if len(test2_df) else 0,
            }
        ]

        print(
            f"[{cfg.dataset}/drug] group{group_idx} | "
            f"train={len(train_df)} | val={len(val_df)} | test1={len(test1_df)} | test2={len(test2_df)}"
        )

        summarize_group_stats(cfg, group_idx, group_rows)


def main():
    for dataset in DATASETS:
        for representation in DRUG_REPRESENTATIONS:
            run_for_config(
                SplitConfig(
                    dataset=dataset,
                    drug_representation=representation,
                )
            )


if __name__ == "__main__":
    main()
