#!/usr/bin/env python
# Author  : KerryChen
# File    : random_and_cold_split.py
# Time    : 2025/5/30 11:16

import os
import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 3407
DATASET = "TPP"

DATA_ROOT = os.path.join(r"/home/qfchen/DTIPred_Plus/database", DATASET, "final_data") 

# RATIO_DIRS = ["1_1", "1_3", "1_5", '1_7', '1_10']
RATIO_DIRS = ["1_1"]

SET_IDS = range(1, 6)


# =========================
# Column settings
# =========================
# Your current TPP column names
TPP_DRUG_COL = "DRUGID"
TPP_PROT_COL = "UNIPROTID"
TPP_LABEL_COL = "Label"

# # Unified model column names
DRUG_COL = "DRUGID"
PROT_COL = "UNIPROTID"
LABEL_COL = "Label"


def random_split(df, random_state=SEED):
    """
    Random pair-level split.
    Ratio:
        train : val : test = 7 : 1 : 2
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=random_state,
        shuffle=True,
        stratify=df[LABEL_COL] if df[LABEL_COL].nunique() == 2 else None,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=2 / 3,
        random_state=random_state,
        shuffle=True,
        stratify=temp_df[LABEL_COL] if temp_df[LABEL_COL].nunique() == 2 else None,
    )

    return train_df, val_df, test_df


def cold_split(df, entity_col, random_state=SEED):
    """
    Cold entity split.
    Entity can be:
        DRUGID    -> cold drug split
        UNIPROTID -> cold protein split

    Ratio at entity level:
        train : val : test = 7 : 1 : 2
    """
    entities = pd.Series(df[entity_col].dropna().unique())

    train_entities, temp_entities = train_test_split(
        entities,
        test_size=0.3,
        random_state=random_state,
        shuffle=True,
    )

    val_entities, test_entities = train_test_split(
        temp_entities,
        test_size=2 / 3,
        random_state=random_state,
        shuffle=True,
    )

    train_df = df[df[entity_col].isin(train_entities)].copy()
    val_df = df[df[entity_col].isin(val_entities)].copy()
    test_df = df[df[entity_col].isin(test_entities)].copy()

    return train_df, val_df, test_df


def standardize_tpp_columns(df, set_file):
    """
    Convert TPP-format columns:
        Protein_ID, Drug, tpp_label

    into unified model-format columns:
        DRUGID, UNIPROTID, Label

    This function also supports already-standardized files containing:
        DRUGID, UNIPROTID, Label
    """
    df = df.copy()

    std_cols = {DRUG_COL, PROT_COL, LABEL_COL}
    tpp_cols = {TPP_DRUG_COL, TPP_PROT_COL, TPP_LABEL_COL}

    if std_cols.issubset(df.columns):
        print(f"[INFO] {os.path.basename(set_file)} already has standardized columns.")
    elif tpp_cols.issubset(df.columns):
        print(f"[INFO] Convert TPP columns for {os.path.basename(set_file)}:")
        print(f"       {TPP_DRUG_COL} -> {DRUG_COL}")
        print(f"       {TPP_PROT_COL} -> {PROT_COL}")
        print(f"       {TPP_LABEL_COL} -> {LABEL_COL}")

        df[DRUG_COL] = df[TPP_DRUG_COL]
        df[PROT_COL] = df[TPP_PROT_COL]
        df[LABEL_COL] = df[TPP_LABEL_COL]
    else:
        missing_std = std_cols - set(df.columns)
        missing_tpp = tpp_cols - set(df.columns)
        raise ValueError(
            f"{set_file} does not contain required columns.\n"
            f"Need either standardized columns: {sorted(std_cols)}\n"
            f"or TPP columns: {sorted(tpp_cols)}\n"
            f"Missing standardized columns: {sorted(missing_std)}\n"
            f"Missing TPP columns: {sorted(missing_tpp)}")

    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


def save_split(train_df, val_df, test_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_excel(os.path.join(output_dir, "train.xlsx"), index=False)
    val_df.to_excel(os.path.join(output_dir, "val.xlsx"), index=False)
    test_df.to_excel(os.path.join(output_dir, "test.xlsx"), index=False)


def check_cold_leakage(train_df, val_df, test_df, entity_col):
    train_entities = set(train_df[entity_col].dropna().unique())
    val_entities = set(val_df[entity_col].dropna().unique())
    test_entities = set(test_df[entity_col].dropna().unique())

    train_val_leakage = train_entities & val_entities
    train_test_leakage = train_entities & test_entities
    val_test_leakage = val_entities & test_entities

    if train_val_leakage:
        raise ValueError(
            f"Cold split leakage detected between train and val in {entity_col}: "
            f"{len(train_val_leakage)} overlapping entities."
        )

    if train_test_leakage:
        raise ValueError(
            f"Cold split leakage detected between train and test in {entity_col}: "
            f"{len(train_test_leakage)} overlapping entities."
        )

    if val_test_leakage:
        raise ValueError(
            f"Cold split leakage detected between val and test in {entity_col}: "
            f"{len(val_test_leakage)} overlapping entities."
        )


def print_split_summary(split_name, group_id, train_df, val_df, test_df):
    print(
        f"{split_name}/Group{group_id}: "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, "
        f"pos_train={train_df[LABEL_COL].sum()}, "
        f"pos_val={val_df[LABEL_COL].sum()}, "
        f"pos_test={test_df[LABEL_COL].sum()}, "
        f"neg_train={(1 - train_df[LABEL_COL]).sum()}, "
        f"neg_val={(1 - val_df[LABEL_COL]).sum()}, "
        f"neg_test={(1 - test_df[LABEL_COL]).sum()}"
    )


def build_splits_for_set(set_file, group_id):
    df = pd.read_excel(set_file, engine="openpyxl")

    # =========================
    # TPP column adaptation
    # =========================
    df = standardize_tpp_columns(df, set_file)

    print("\n" + "=" * 100)
    print(f"Loaded: {set_file}")
    print(f"Shape: {df.shape}")
    print("Label distribution:")
    print(df[LABEL_COL].value_counts().sort_index())
    print("=" * 100)

    split_specs = [
        # Pair-level random split
        ("random_split", random_split(df, random_state=SEED + group_id)),

        # Cold drug split
        ("cold_drug_split", cold_split(df, DRUG_COL, random_state=SEED + group_id)),

        # Cold protein split
        ("cold_prot_split", cold_split(df, PROT_COL, random_state=SEED + group_id)),
    ]



    for split_name, (train_df, val_df, test_df) in split_specs:
        if split_name == "cold_drug_split":
            check_cold_leakage(train_df, val_df, test_df, DRUG_COL)

        elif split_name == "cold_prot_split":
            check_cold_leakage(train_df, val_df, test_df, PROT_COL)

        output_dir = os.path.join(os.path.dirname(set_file), split_name, f"group{group_id}")

        save_split(train_df, val_df, test_df, output_dir)

        print_split_summary(
            split_name=split_name,
            group_id=group_id,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

        print(f"Saved to: {output_dir}")


def main():
    for ratio_dir in RATIO_DIRS:
        ratio_path = os.path.join(DATA_ROOT, ratio_dir)

        if not os.path.exists(ratio_path):
            raise FileNotFoundError(f"Missing data directory: {ratio_path}")

        for set_id in SET_IDS:
            set_file = os.path.join(ratio_path, f"Set{set_id}.xlsx")

            if not os.path.exists(set_file):
                raise FileNotFoundError(f"Missing dataset file: {set_file}")

            build_splits_for_set(set_file, group_id=set_id)


if __name__ == "__main__":
    main()