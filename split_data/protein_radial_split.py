#!/usr/bin/env python
# Author  : Codex
# File    : build_protein_radial_fixed_distance_split.py
# Updated : 2026/7/27

"""
Build protein-only fixed-distance radial splits from Ankh representations.

For each dataset:
1. Read protein IDs from ProtInfo.xlsx.
2. Mean-pool each token-level Ankh feature into one protein vector.
3. Compute cosine distance to the dataset protein centroid.
4. Assign proteins with fixed thresholds:
   - core: distance <= 0.35
   - near: 0.35 < distance <= 0.40
   - far:  distance > 0.40
5. Split core proteins into train/validation and map protein regions back to
   DTI pairs by UNIPROTID:
   - train: core_train proteins
   - val:   core_val proteins
   - test1: near proteins
   - test2: far proteins

This script only handles protein representations and protein-based splitting.
"""

import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances

try:
    import umap
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "umap-learn is required. Install it with: pip install umap-learn"
    ) from exc


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATABASE_DIR = os.path.join(ROOT_DIR, "DTIPred_Plus/database")

# Use ["SNAP", "DRH", "TTD", "SDT"] to build all public datasets.
DATASETS = ['TPP']

RATIO_DIR = "1_1"
GROUP_IDS = [1, 2, 3, 4, 5]
VAL_FRAC_WITHIN_CORE = 0.125

PROT_FEAT_DIR = "/data/qfchen/DTIPred_Plus/UniprotKB_Ankh/"

# These are cosine-distance thresholds, so larger values indicate proteins
# farther from the dataset protein centroid.
PROTEIN_DISTANCE_THRESHOLDS = {
    "core_max": 0.35,
    "near_max": 0.40,
}

PROT_ID_COL = "UNIPROTID"
LABEL_COL = "Label"

RANDOM_SEED_BASE = 3400
UMAP_RANDOM_STATE = 3407
UMAP_N_NEIGHBORS = 100
UMAP_MIN_DIST = 0.10

FIGSIZE = (8, 8)
FIG_DPI = 600
POINT_SIZE = 50
POINT_ALPHA = 0.82
POINT_EDGE_WIDTH = 0.25
CENTER_MARKER_SIZE = 220

REGION_COLOR_MAP = {
    "core": "#4CAF50",
    "near": "#42A5F5",
    "far": "#FF7043",
}

DISPLAY_NAME_MAP = {
    "core": "Core",
    "near": "1st Cousin",
    "far": "2nd Cousin",
}


@dataclass
class ProteinSplitConfig:
    dataset: str

    @property
    def prot_info_file(self):
        return os.path.join(
            DATABASE_DIR,
            self.dataset,
            "gold_data",
            "ProtInfo.xlsx")

    @property
    def split_name(self):
        return "prot_ankh_distance_radial_split"

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

    @property
    def artifact_prefix(self):
        # Threshold values are intentionally omitted from saved file names.
        return f"{self.dataset}_prot_ankh_distance"

    @property
    def fixed_thresholds(self):
        core_max = float(PROTEIN_DISTANCE_THRESHOLDS["core_max"])
        near_max = float(PROTEIN_DISTANCE_THRESHOLDS["near_max"])
        if not 0.0 <= core_max < near_max:
            raise ValueError(
                "Expected 0 <= core_max < near_max, got {} and {}".format(
                    core_max,
                    near_max,
                )
            )
        return core_max, near_max


def load_ankh_feature(path):
    return np.load(path, allow_pickle=False)


def mean_pool_ankh_feature(feature):
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim != 2:
        raise ValueError(
            "Expected a 2D token-level Ankh feature, got shape {}".format(
                feature.shape
            )
        )
    if feature.shape[0] == 0:
        raise ValueError("Cannot pool an empty Ankh feature")
    return feature.mean(axis=0)


def protein_feature_path(protein_id):
    return os.path.join(PROT_FEAT_DIR, f"{protein_id}.npy")


def load_protein_catalog(cfg):
    if not os.path.exists(cfg.prot_info_file):
        raise FileNotFoundError(f"Protein information file not found: {cfg.prot_info_file}")

    info_df = pd.read_excel(cfg.prot_info_file)
    if PROT_ID_COL not in info_df.columns:
        raise ValueError(f"{cfg.prot_info_file} is missing required column: {PROT_ID_COL}")

    protein_ids = (info_df[PROT_ID_COL].dropna().astype(str).str.strip())
    protein_ids = protein_ids[protein_ids.ne("")].drop_duplicates().tolist()

    print(f"[{cfg.dataset}/protein] entity catalog source: {cfg.prot_info_file}")
    print(f"[{cfg.dataset}/protein] entity catalog size: {len(protein_ids)}")
    return protein_ids


def build_protein_embedding_table(cfg, protein_ids):
    rows = []
    vectors = []
    missing_ids = []
    invalid_ids = []

    for protein_id in protein_ids:
        feature_path = protein_feature_path(protein_id)
        if not os.path.exists(feature_path):
            missing_ids.append(protein_id)
            continue

        try:
            pooled = mean_pool_ankh_feature(load_ankh_feature(feature_path))
        except (OSError, TypeError, ValueError):
            invalid_ids.append(protein_id)
            continue

        rows.append(
            {
                "entity_id": protein_id,
                "feature_path": feature_path,
                "representation": "Ankh_mean_pool",
            }
        )
        vectors.append(pooled)

    if not vectors:
        raise ValueError(
            f"No valid Ankh protein features found for {cfg.dataset}"
        )

    dimensions = sorted({len(vector) for vector in vectors})
    if len(dimensions) != 1:
        raise ValueError(
            f"Inconsistent Ankh feature dimensions for {cfg.dataset}: {dimensions}"
        )

    if missing_ids:
        print(
            f"[{cfg.dataset}/protein] skipped {len(missing_ids)} proteins "
            "with missing Ankh features"
        )
    if invalid_ids:
        print(
            f"[{cfg.dataset}/protein] skipped {len(invalid_ids)} proteins "
            "with invalid Ankh features"
        )

    vectors = np.vstack(vectors).astype(np.float32, copy=False)
    entity_df = pd.DataFrame(rows)
    entity_df["embedding_dim"] = vectors.shape[1]

    print(
        f"[{cfg.dataset}/protein] valid Ankh features: "
        f"{len(entity_df)}/{len(protein_ids)}"
    )
    return entity_df, vectors


def assign_regions_by_threshold(distances, core_max, near_max):
    regions = np.empty(len(distances), dtype=object)
    regions[distances <= core_max] = "core"
    regions[(distances > core_max) & (distances <= near_max)] = "near"
    regions[distances > near_max] = "far"
    return regions


def assign_radial_regions(vectors, cfg):
    center = vectors.mean(axis=0)
    distances = cosine_distances(vectors, center.reshape(1, -1)).reshape(-1)
    core_max, near_max = cfg.fixed_thresholds
    regions = assign_regions_by_threshold(distances, core_max, near_max,)
    return center, distances, regions, core_max, near_max


def report_distance_diagnostics(cfg, distances, regions):
    core_max, near_max = cfg.fixed_thresholds
    quantile_levels = [
        0.00,
        0.10,
        0.25,
        0.50,
        0.75,
        0.80,
        0.90,
        0.95,
        1.00,
    ]
    quantile_values = np.quantile(distances, quantile_levels)

    print(
        "[{} / ankh] fixed thresholds: "
        "core <= {:.4f}; near <= {:.4f}; far > {:.4f}".format(
            cfg.dataset,
            core_max,
            near_max,
            near_max,
        )
    )
    print(
        "[{} / ankh] distance quantiles: {}".format(
            cfg.dataset,
            ", ".join(
                "q{:02d}={:.4f}".format(int(level * 100), value)
                for level, value in zip(quantile_levels, quantile_values)
            ),
        )
    )

    counts = (
        pd.Series(regions)
        .value_counts()
        .reindex(["core", "near", "far"], fill_value=0)
    )
    total = len(regions)
    print(
        "[{} / ankh] region counts: {}".format(
            cfg.dataset,
            ", ".join(
                "{}={} ({:.1%})".format(
                    region,
                    int(count),
                    count / total,
                )
                for region, count in counts.items()
            ),
        )
    )

    empty_regions = counts[counts == 0].index.tolist()
    if empty_regions:
        raise ValueError(
            "Fixed Ankh thresholds produced empty regions {} for {}. "
            "Adjust PROTEIN_DISTANCE_THRESHOLDS after inspecting the "
            "printed quantiles.".format(
                empty_regions,
                cfg.dataset,
            )
        )

    for region, count in counts.items():
        if count / total < 0.02:
            print("[WARNING] {} contains less than 2% of proteins; estimates may be unstable.".format(region))



def split_core_proteins(entity_df, seed):
    core_ids = entity_df.loc[entity_df["region"] == "core", "entity_id"].tolist()
    if len(core_ids) < 2:
        raise ValueError("Need at least two core proteins for train/val")

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(core_ids, dtype=object)
    rng.shuffle(shuffled)

    n_val = max(1, int(math.floor(len(shuffled) * VAL_FRAC_WITHIN_CORE)))
    if n_val >= len(shuffled):
        n_val = len(shuffled) - 1

    val_ids = set(shuffled[:n_val].tolist())
    train_ids = set(shuffled[n_val:].tolist())
    return train_ids, val_ids


def load_pair_table(cfg, group_idx):
    pair_file = os.path.join(DATABASE_DIR, cfg.dataset, "final_data", RATIO_DIR, f"Set{group_idx}.xlsx")
    if not os.path.exists(pair_file):
        return None, pair_file

    pair_df = pd.read_excel(pair_file)
    required = {PROT_ID_COL, LABEL_COL}
    missing = required.difference(pair_df.columns)
    if missing:
        raise ValueError(f"{pair_file} is missing required columns: {sorted(missing)}")

    pair_df[PROT_ID_COL] = (pair_df[PROT_ID_COL].astype(str).str.strip())
    print(f"[{cfg.dataset}/protein] source: {pair_file}")
    print(f"[{cfg.dataset}/protein] pairs loaded: {len(pair_df)}")
    return pair_df.reset_index(drop=True), pair_file


def build_pair_splits(
    pair_df,
    train_ids,
    val_ids,
    near_ids,
    far_ids,
):
    protein_series = pair_df[PROT_ID_COL]

    train_df = pair_df.loc[protein_series.isin(train_ids)].reset_index(drop=True)
    val_df = pair_df.loc[protein_series.isin(val_ids)].reset_index(drop=True)
    test1_df = pair_df.loc[protein_series.isin(near_ids)].reset_index(drop=True)
    test2_df = pair_df.loc[protein_series.isin(far_ids)].reset_index(drop=True)

    return train_df, val_df, test1_df, test2_df


def save_group_files(
    cfg,
    group_idx,
    train_df,
    val_df,
    test1_df,
    test2_df,
):
    group_dir = os.path.join(cfg.split_root, f"group{group_idx}")
    os.makedirs(group_dir, exist_ok=True)

    train_df.to_excel(os.path.join(group_dir, "train.xlsx"), index=False,)
    val_df.to_excel(os.path.join(group_dir, "val.xlsx"), index=False,)
    test1_df.to_excel(os.path.join(group_dir, "test1.xlsx"), index=False,)
    test2_df.to_excel(os.path.join(group_dir, "test2.xlsx"), index=False,)


def save_figure(base_path_without_extension):
    png_path = f"{base_path_without_extension}.png"
    pdf_path = f"{base_path_without_extension}.pdf"
    plt.savefig(
        png_path,
        dpi=FIG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    return png_path, pdf_path


def draw_region_scatter(
    plot_df,
    x_col,
    y_col,
    cfg,
    title,
    output_stem,
):
    os.makedirs(cfg.figure_dir, exist_ok=True)
    plt.figure(figsize=FIGSIZE)

    for region in ["core", "near", "far"]:
        subset = plot_df.loc[plot_df["region"] == region]
        plt.scatter(
            subset[x_col],
            subset[y_col],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            c=REGION_COLOR_MAP[region],
            edgecolors="white",
            linewidths=POINT_EDGE_WIDTH,
            label=f"{DISPLAY_NAME_MAP[region]} (n={len(subset)})",
        )

    coordinates = plot_df[[x_col, y_col]].to_numpy(dtype=float)
    visual_center = coordinates.mean(axis=0)
    center_idx = np.argmin(
        np.linalg.norm(coordinates - visual_center, axis=1)
    )
    center_xy = coordinates[center_idx]
    plt.scatter(
        center_xy[0],
        center_xy[1],
        s=CENTER_MARKER_SIZE,
        marker="*",
        c="red",
        edgecolors="black",
        linewidths=0.9,
        label="center",
        zorder=5,
    )

    plt.title(title, fontsize=24, pad=14)
    plt.xlabel(x_col, fontsize=18, labelpad=12)
    plt.ylabel(y_col, fontsize=18, labelpad=12)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16, frameon=True)
    plt.tight_layout()

    figure_base = os.path.join(
        cfg.figure_dir,
        f"{cfg.artifact_prefix}_{output_stem}",
    )
    return save_figure(figure_base)


def draw_umap(entity_df, vectors, cfg):
    n_neighbors = min(
        UMAP_N_NEIGHBORS,
        max(2, len(entity_df) - 1),
    )
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=UMAP_RANDOM_STATE,
    )
    coordinates = reducer.fit_transform(vectors)

    umap_df = entity_df.copy()
    umap_df["UMAP1"] = coordinates[:, 0]
    umap_df["UMAP2"] = coordinates[:, 1]
    png_path, pdf_path = draw_region_scatter(
        umap_df,
        "UMAP1",
        "UMAP2",
        cfg,
        f"Ankh protein embedding space ({cfg.dataset}, UMAP)",
        "radial_umap",
    )
    return umap_df, png_path, pdf_path


def draw_pca(entity_df, vectors, cfg):
    reducer = PCA(
        n_components=2,
        random_state=UMAP_RANDOM_STATE,
    )
    coordinates = reducer.fit_transform(vectors)

    pca_df = entity_df.copy()
    pca_df["PCA1"] = coordinates[:, 0]
    pca_df["PCA2"] = coordinates[:, 1]
    png_path, pdf_path = draw_region_scatter(
        pca_df,
        "PCA1",
        "PCA2",
        cfg,
        f"Ankh protein embedding space ({cfg.dataset}, PCA)",
        "radial_pca",
    )
    return pca_df, png_path, pdf_path


def draw_tsne(entity_df, vectors, cfg):
    perplexity = min(30, max(1, len(entity_df) - 1))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric="cosine",
        init="random",
        learning_rate="auto",
        random_state=UMAP_RANDOM_STATE,
    )
    coordinates = reducer.fit_transform(vectors)

    tsne_df = entity_df.copy()
    tsne_df["TSNE1"] = coordinates[:, 0]
    tsne_df["TSNE2"] = coordinates[:, 1]
    png_path, pdf_path = draw_region_scatter(
        tsne_df,
        "TSNE1",
        "TSNE2",
        cfg,
        f"Ankh protein embedding space ({cfg.dataset}, t-SNE)",
        "radial_tsne",
    )
    return tsne_df, png_path, pdf_path


def save_entity_and_projection_summaries(
    cfg,
    entity_df,
    pca_df,
    tsne_df,
):
    os.makedirs(cfg.summary_dir, exist_ok=True)

    entity_path = os.path.join(
        cfg.summary_dir,
        f"{cfg.artifact_prefix}_entity_summary.xlsx",
    )
    pca_path = os.path.join(
        cfg.summary_dir,
        f"{cfg.artifact_prefix}_pca_projection.xlsx",
    )
    tsne_path = os.path.join(
        cfg.summary_dir,
        f"{cfg.artifact_prefix}_tsne_projection.xlsx",
    )

    entity_df.to_excel(entity_path, index=False)
    pca_df.to_excel(pca_path, index=False)
    tsne_df.to_excel(tsne_path, index=False)

    print(
        f"[{cfg.dataset}/protein] entity summary saved to: {entity_path}"
    )


def save_distance_region_summary(cfg, entity_df):
    os.makedirs(cfg.summary_dir, exist_ok=True)
    core_max, near_max = cfg.fixed_thresholds
    total = len(entity_df)
    rows = []

    for region in ["core", "near", "far"]:
        values = entity_df.loc[
            entity_df["region"] == region,
            "distance_to_center",
        ].to_numpy(dtype=float)
        rows.append(
            {
                "dataset": cfg.dataset,
                "representation": "ankh",
                "distance_metric": "cosine",
                "region": region,
                "count": len(values),
                "fraction": len(values) / total if total else float("nan"),
                "distance_min": (
                    float(np.min(values))
                    if len(values)
                    else float("nan")
                ),
                "distance_mean": (
                    float(np.mean(values))
                    if len(values)
                    else float("nan")
                ),
                "distance_median": (
                    float(np.median(values))
                    if len(values)
                    else float("nan")
                ),
                "distance_max": (
                    float(np.max(values))
                    if len(values)
                    else float("nan")
                ),
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
        f"[{cfg.dataset}/protein] distance-region summary saved to: "
        f"{summary_path}"
    )


def count_positive_rows(dataframe):
    if len(dataframe) == 0:
        return 0
    labels = pd.to_numeric(
        dataframe[LABEL_COL],
        errors="coerce",
    ).fillna(0)
    return int(labels.sum())


def save_group_summary(cfg, group_idx, row):
    os.makedirs(cfg.summary_dir, exist_ok=True)
    summary_path = os.path.join(
        cfg.summary_dir,
        f"{cfg.artifact_prefix}_group{group_idx}_split_summary.xlsx",
    )
    pd.DataFrame([row]).to_excel(summary_path, index=False)
    print(
        f"[{cfg.dataset}/protein] split summary saved to: {summary_path}"
    )


def run_for_dataset(dataset):
    cfg = ProteinSplitConfig(dataset=dataset)

    print("\n" + "=" * 90)
    print(f"Building {cfg.split_name} for {cfg.dataset}")
    print("=" * 90)

    protein_ids = load_protein_catalog(cfg)
    entity_df, vectors = build_protein_embedding_table(
        cfg,
        protein_ids,
    )

    center, distances, regions, core_max, near_max = (
        assign_radial_regions(vectors, cfg)
    )
    report_distance_diagnostics(cfg, distances, regions)

    entity_df["distance_to_center"] = distances
    entity_df["region"] = regions
    entity_df["center_dim"] = len(center)
    entity_df["representation_tag"] = "ankh"
    entity_df["split_name"] = cfg.split_name
    entity_df["distance_metric"] = "cosine"
    entity_df["core_max_distance"] = core_max
    entity_df["near_max_distance"] = near_max

    umap_df, umap_png, umap_pdf = draw_umap(
        entity_df,
        vectors,
        cfg,
    )
    pca_df, pca_png, pca_pdf = draw_pca(
        entity_df,
        vectors,
        cfg,
    )
    tsne_df, tsne_png, tsne_pdf = draw_tsne(
        entity_df,
        vectors,
        cfg,
    )

    print(f"[{cfg.dataset}/protein] UMAP saved to: {umap_png}")
    print(f"[{cfg.dataset}/protein] UMAP saved to: {umap_pdf}")
    print(f"[{cfg.dataset}/protein] PCA saved to:  {pca_png}")
    print(f"[{cfg.dataset}/protein] PCA saved to:  {pca_pdf}")
    print(f"[{cfg.dataset}/protein] t-SNE saved to: {tsne_png}")
    print(f"[{cfg.dataset}/protein] t-SNE saved to: {tsne_pdf}")

    entity_df["UMAP1"] = umap_df["UMAP1"]
    entity_df["UMAP2"] = umap_df["UMAP2"]
    save_entity_and_projection_summaries(
        cfg,
        entity_df,
        pca_df,
        tsne_df,
    )
    save_distance_region_summary(cfg, entity_df)

    core_ids = set(
        entity_df.loc[
            entity_df["region"] == "core",
            "entity_id",
        ]
    )
    near_ids = set(
        entity_df.loc[
            entity_df["region"] == "near",
            "entity_id",
        ]
    )
    far_ids = set(
        entity_df.loc[
            entity_df["region"] == "far",
            "entity_id",
        ]
    )

    for group_idx in GROUP_IDS:
        pair_df, pair_file = load_pair_table(cfg, group_idx)
        if pair_df is None:
            print(
                f"[{cfg.dataset}/protein] skip missing set file: "
                f"{pair_file}"
            )
            continue

        seed = RANDOM_SEED_BASE + group_idx
        train_ids, val_ids = split_core_proteins(
            entity_df,
            seed,
        )
        train_df, val_df, test1_df, test2_df = build_pair_splits(
            pair_df,
            train_ids,
            val_ids,
            near_ids,
            far_ids,
        )
        save_group_files(
            cfg,
            group_idx,
            train_df,
            val_df,
            test1_df,
            test2_df,
        )

        group_row = {
            "group": group_idx,
            "seed": seed,
            "split_name": cfg.split_name,
            "representation": "ankh",
            "distance_metric": "cosine",
            "core_max_distance": core_max,
            "near_max_distance": near_max,
            "source_set": os.path.basename(pair_file),
            "core_entities": len(core_ids),
            "train_entities": len(train_ids),
            "val_entities": len(val_ids),
            "near_entities": len(near_ids),
            "far_entities": len(far_ids),
            "train_pairs": len(train_df),
            "val_pairs": len(val_df),
            "test1_pairs": len(test1_df),
            "test2_pairs": len(test2_df),
            "train_pos": count_positive_rows(train_df),
            "val_pos": count_positive_rows(val_df),
            "test1_pos": count_positive_rows(test1_df),
            "test2_pos": count_positive_rows(test2_df),
        }
        save_group_summary(
            cfg,
            group_idx,
            group_row,
        )

        print(
            f"[{cfg.dataset}/protein] group{group_idx} | "
            f"train={len(train_df)} | "
            f"val={len(val_df)} | "
            f"test1={len(test1_df)} | "
            f"test2={len(test2_df)}"
        )


def main():
    for dataset in DATASETS:
        run_for_dataset(dataset)


if __name__ == "__main__":
    main()
