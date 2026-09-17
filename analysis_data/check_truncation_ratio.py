#!/usr/bin/env python
# Author  : KerryChen
# File    : check_truncation_ratio.py

import os
import warnings

import pandas as pd
from rdkit import Chem
from transformers import AutoTokenizer


BASE_DIR = "../database"
SAVE_DIR = "./results/data_quality"

# MolFormer tokenizer path
TOKENIZER_PATH = "../large_model/MolFormer/"

MAX_DRUG_LEN = 150
MAX_PROT_LEN = 2000

# If the manuscript says special tokens were removed, keep this as False.
# If your actual embedding extraction keeps [CLS]/[SEP] or special tokens, change it to True.
ADD_SPECIAL_TOKENS = False

DATASETS = {
    "SNAP": {
        "drug_file": "SNAP/final_data/DrugInfo.xlsx",
        "prot_file": "SNAP/final_data/ProtInfo.xlsx",
        "drug_col": "INCHIKEY",
    },
    "DRH": {
        "drug_file": "DRH/final_data/DrugInfo.xlsx",
        "prot_file": "DRH/final_data/ProtInfo.xlsx",
        "drug_col": "INCHIKEY",
    },
    "TTD": {
        "drug_file": "TTD/final_data/DrugInfo.xlsx",
        "prot_file": "TTD/final_data/ProtInfo.xlsx",
        "drug_col": "INCHIKEY",
    },
}


def read_excel(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_excel(path, engine="openpyxl")


def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def canonical_smiles(smiles):
    if pd.isna(smiles):
        return None

    smiles = str(smiles).strip()
    if smiles == "":
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def smiles_token_len(smiles, tokenizer):
    can_smiles = canonical_smiles(smiles)

    if can_smiles is None:
        return None

    try:
        input_ids = tokenizer.encode(
            can_smiles,
            add_special_tokens=ADD_SPECIAL_TOKENS,
            truncation=False,
        )
        return int(len(input_ids))
    except Exception as e:
        warnings.warn(f"Tokenizer failed for SMILES: {smiles} | Error: {e}")
        return None


def analyze_dataset(dataset_name, drug_path, prot_path, drug_col, tokenizer):
    drug_info = clean_columns(read_excel(drug_path))
    prot_info = clean_columns(read_excel(prot_path))

    required_drug_cols = {drug_col, "SMILES"}
    required_prot_cols = {"UNIPROTID", "SEQUENCE"}

    missing_drug_cols = required_drug_cols - set(drug_info.columns)
    missing_prot_cols = required_prot_cols - set(prot_info.columns)

    if missing_drug_cols:
        raise ValueError(f"{drug_path} missing columns: {missing_drug_cols}")
    if missing_prot_cols:
        raise ValueError(f"{prot_path} missing columns: {missing_prot_cols}")

    protein_df = (
        prot_info[["UNIPROTID", "SEQUENCE"]]
        .dropna(subset=["UNIPROTID", "SEQUENCE"])
        .drop_duplicates(subset=["UNIPROTID"])
        .copy()
    )

    protein_df["SEQUENCE"] = protein_df["SEQUENCE"].astype(str).str.strip()
    protein_df = protein_df[protein_df["SEQUENCE"] != ""].copy()

    protein_df["ProteinLength"] = protein_df["SEQUENCE"].map(len).astype("int64")
    protein_df["ProteinTruncated"] = protein_df["ProteinLength"] > MAX_PROT_LEN
    protein_df["Dataset"] = dataset_name

    drug_df = (
        drug_info[[drug_col, "SMILES"]]
        .dropna(subset=[drug_col, "SMILES"])
        .drop_duplicates(subset=[drug_col])
        .copy()
    )

    drug_df = drug_df.rename(columns={drug_col: "INCHIKEY"})
    drug_df["SMILESTokenLength"] = drug_df["SMILES"].map(
        lambda x: smiles_token_len(x, tokenizer)
    )

    invalid_drug_n = int(drug_df["SMILESTokenLength"].isna().sum())
    if invalid_drug_n > 0:
        warnings.warn(f"{dataset_name}: skipped {invalid_drug_n} invalid or failed SMILES.")

    drug_df = drug_df.dropna(subset=["SMILESTokenLength"]).copy()
    drug_df["SMILESTokenLength"] = drug_df["SMILESTokenLength"].astype("int64")
    drug_df["DrugTruncated"] = drug_df["SMILESTokenLength"] > MAX_DRUG_LEN
    drug_df["Dataset"] = dataset_name

    if len(protein_df) == 0:
        raise ValueError(f"No valid protein sequences found in {prot_path}")
    if len(drug_df) == 0:
        raise ValueError(f"No valid SMILES found in {drug_path}")

    summary = {
        "Dataset": dataset_name,

        "Protein_N": int(len(protein_df)),
        "Protein_MaxLen": int(protein_df["ProteinLength"].max()),
        "Protein_MeanLen": float(protein_df["ProteinLength"].mean()),
        "Protein_MedianLen": float(protein_df["ProteinLength"].median()),
        "Protein_Q95": float(protein_df["ProteinLength"].quantile(0.95)),
        "Protein_Q99": float(protein_df["ProteinLength"].quantile(0.99)),
        "Protein_Truncated_N": int(protein_df["ProteinTruncated"].sum()),
        "Protein_Truncated_Ratio": float(protein_df["ProteinTruncated"].mean()),

        "Drug_N": int(len(drug_df)),
        "Drug_Invalid_N": invalid_drug_n,
        "Drug_MaxTokenLen": int(drug_df["SMILESTokenLength"].max()),
        "DrugToken_MeanLen": float(drug_df["SMILESTokenLength"].mean()),
        "DrugToken_MedianLen": float(drug_df["SMILESTokenLength"].median()),
        "DrugToken_Q95": float(drug_df["SMILESTokenLength"].quantile(0.95)),
        "DrugToken_Q99": float(drug_df["SMILESTokenLength"].quantile(0.99)),
        "Drug_Truncated_N": int(drug_df["DrugTruncated"].sum()),
        "Drug_Truncated_Ratio": float(drug_df["DrugTruncated"].mean()),
    }

    return protein_df, drug_df, summary


def save_table(df, save_dir, filename_prefix):
    xlsx_path = os.path.join(save_dir, f"{filename_prefix}.xlsx")
    df.to_excel(xlsx_path, index=False)
    print(f"Saved: {xlsx_path}")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading MolFormer tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    all_protein = []
    all_drug = []
    summaries = []

    for dataset_name, cfg in DATASETS.items():
        print(f"\nProcessing dataset: {dataset_name}")

        drug_path = os.path.join(BASE_DIR, cfg["drug_file"])
        prot_path = os.path.join(BASE_DIR, cfg["prot_file"])

        protein_df, drug_df, summary = analyze_dataset(
            dataset_name=dataset_name,
            drug_path=drug_path,
            prot_path=prot_path,
            drug_col=cfg["drug_col"],
            tokenizer=tokenizer)

        all_protein.append(protein_df)
        all_drug.append(drug_df)
        summaries.append(summary)

    protein_out = pd.concat(all_protein, ignore_index=True)
    drug_out = pd.concat(all_drug, ignore_index=True)
    summary_out = pd.DataFrame(summaries)

    save_table(protein_out, SAVE_DIR, "protein_truncation_detail")
    save_table(drug_out, SAVE_DIR, "drug_token_truncation_detail")
    save_table(summary_out, SAVE_DIR, "truncation_summary")

    print("\nSummary:")
    print(summary_out.to_string(index=False))
    print(f"\nSaved truncation results to: {SAVE_DIR}")


if __name__ == "__main__":
    main()