#!/usr/bin/env python
# Author  : KerryChen
# File    : split_data.py
# Time    : 2025/5/30 11:16

import pandas as pd
import os
from sklearn.model_selection import train_test_split

def cold_split_811(df, entity_col, prefix, output_dir=None, random_state=42):
    """
    Perform 8:1:1 cold split (train:val:test) based on unique entities.

    Args:
        df (pd.DataFrame): Input dataframe
        entity_col (str): Column name for cold splitting ('DrugID' or 'TargetID')
        prefix (str): Filename prefix for saving results
        output_dir (str): Output directory to save split CSVs (optional)
        random_state (int): Random seed

    Returns:
        train_df, val_df, test_df: Split dataframes
    """
    if entity_col not in df.columns:
        raise ValueError(f"Column '{entity_col}' not found in the DataFrame.")

    entities = df[entity_col].unique()

    train_entities, remaining_entities = train_test_split(
        entities, test_size=0.2, random_state=random_state
    )

    val_entities, test_entities = train_test_split(
        remaining_entities, test_size=0.5, random_state=random_state
    )

    train_df = df[df[entity_col].isin(train_entities)]
    val_df = df[df[entity_col].isin(val_entities)]
    test_df = df[df[entity_col].isin(test_entities)]

    print(f"\n=== Cold 8:1:1 Split by {entity_col} ===")
    print(f"Total samples: {len(df)}")
    print(f"Train: {len(train_df)} ({len(train_df)/len(df):.1%}) | Unique {entity_col}: {len(train_entities)}")
    print(f"Validation: {len(val_df)} ({len(val_df)/len(df):.1%}) | Unique {entity_col}: {len(val_entities)}")
    print(f"Test: {len(test_df)} ({len(test_df)/len(df):.1%}) | Unique {entity_col}: {len(test_entities)}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, f'{prefix}_train.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, f'{prefix}_val.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, f'{prefix}_test.csv'), index=False)

        print(f"Saved to directory: {output_dir}")
        print(f"- Train: {prefix}_train.csv")
        print(f"- Validation: {prefix}_val.csv")
        print(f"- Test: {prefix}_test.csv")

    return train_df, val_df, test_df


def random_split_811(df, prefix, output_dir=None, random_state=42):
    """
    Perform 8:1:1 random split (train:val:test).

    Args:
        df (pd.DataFrame): Input dataframe
        prefix (str): Filename prefix for saving results
        output_dir (str): Output directory to save split CSVs (optional)
        random_state (int): Random seed

    Returns:
        train_df, val_df, test_df: Split dataframes
    """
    train_df, remaining_df = train_test_split(df, test_size=0.2, random_state=random_state)
    val_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=random_state)

    print(f"\n=== Random 8:1:1 Split ===")
    print(f"Total samples: {len(df)}")
    print(f"Train: {len(train_df)} ({len(train_df)/len(df):.1%})")
    print(f"Validation: {len(val_df)} ({len(val_df)/len(df):.1%})")
    print(f"Test: {len(test_df)} ({len(test_df)/len(df):.1%})")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, f'{prefix}_train.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, f'{prefix}_val.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, f'{prefix}_test.csv'), index=False)

        print(f"Saved to directory: {output_dir}")
        print(f"- Train: {prefix}_train.csv")
        print(f"- Validation: {prefix}_val.csv")
        print(f"- Test: {prefix}_test.csv")

    return train_df, val_df, test_df


if __name__ == "__main__":
    my_dir = 'process_data/'
    files = os.listdir(my_dir)
    print(files)
    for file in files:
        # if file.endswith('.xlsx'):
        if file == 'DRH.xlsx':
            suffix = file.split('.')[0]
            DATA_PATH = os.path.join(my_dir, file)
            OUTPUT_DIR = my_dir + f'split_results_811'

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            RANDOM_STATE = 1

            print(f"Reading data from: {DATA_PATH}")
            try:
                df = pd.read_excel(DATA_PATH)
                print(df.columns)
                print(f"Successfully loaded dataset. Total samples: {len(df)}")
                print(f"Unique DRUG_NAME: {df['INCHIKEY'].nunique()}")
                print(f"Unique UNIPROT_ID: {df['UNIPROT_ID'].nunique()}")
            except Exception as e:
                raise SystemExit(f"Failed to read file: {e}")

            # Drug Cold Split
            drug_train, drug_val, drug_test = cold_split_811(
                df, 'INCHIKEY', 'drug', OUTPUT_DIR, RANDOM_STATE
            )

            # Target Cold Split
            target_train, target_val, target_test = cold_split_811(
                df, 'UNIPROT_ID', 'target', OUTPUT_DIR, RANDOM_STATE
            )

            # Random Split
            random_train, random_val, random_test = random_split_811(
                df, 'random', OUTPUT_DIR, RANDOM_STATE
            )
