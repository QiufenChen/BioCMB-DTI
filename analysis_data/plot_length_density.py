#!/usr/bin/env python
# Author  : KerryChen
# File    : plot_length_density.py

import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit import Chem
from transformers import AutoTokenizer


DATASETS = {
    "SNAP": "SNAP",
    "DRH": "DRH",
    "TTD": "TTD",
}

CONFIG = {
    "base_dir": "/home/qfchen/DTIPred_Plus/database",
    "output_dir": "/home/qfchen/DTIPred_Plus/analysis_data/results/length_density",
    "model_path": "/data/qfchen/lager_model/MolFormer/",
    # If the manuscript says special tokens were removed, keep this as False.
    "add_special_tokens": False,
    "xlim_quantile": 0.995,
}


def read_excel(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_excel(path, engine="openpyxl")


def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def canonicalize_smiles(smiles):
    if pd.isna(smiles):
        return None

    smiles = str(smiles).strip()
    if smiles == "":
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def get_smiles_token_lengths(smiles_series, tokenizer, add_special_tokens=False):
    token_lengths = []
    invalid_count = 0

    for smiles in smiles_series:
        can_smiles = canonicalize_smiles(smiles)

        if can_smiles is None:
            invalid_count += 1
            continue

        try:
            input_ids = tokenizer.encode(
                can_smiles,
                add_special_tokens=add_special_tokens,
                truncation=False,
            )
            token_lengths.append(len(input_ids))
        except Exception as e:
            invalid_count += 1
            warnings.warn(f"Tokenizer failed for SMILES: {smiles} | Error: {e}")

    if invalid_count > 0:
        warnings.warn(f"Skipped {invalid_count} invalid or failed SMILES.")

    return pd.Series(token_lengths, dtype="int64")


def get_protein_lengths(seq_series):
    seq_series = seq_series.dropna().astype(str).str.strip()
    seq_series = seq_series[seq_series != ""]
    return seq_series.map(len).astype("int64")


def collect_lengths(base_dir, datasets, tokenizer, add_special_tokens=False):
    prot_records = []
    drug_records = []
    summary_rows = []

    for display_name, folder_name in datasets.items():
        data_dir = os.path.join(base_dir, folder_name, "final_data")
        drug_path = os.path.join(data_dir, "DrugInfo.xlsx")
        prot_path = os.path.join(data_dir, "ProtInfo.xlsx")

        print(f"\nProcessing dataset: {display_name}")
        print(f"Drug file   : {drug_path}")
        print(f"Protein file: {prot_path}")

        drug_df = clean_columns(read_excel(drug_path))
        prot_df = clean_columns(read_excel(prot_path))

        required_drug_cols = {"DRUGID", "SMILES"}
        required_prot_cols = {"UNIPROTID", "SEQUENCE"}

        missing_drug_cols = required_drug_cols - set(drug_df.columns)
        missing_prot_cols = required_prot_cols - set(prot_df.columns)

        if missing_drug_cols:
            raise ValueError(f"{drug_path} missing columns: {missing_drug_cols}")
        if missing_prot_cols:
            raise ValueError(f"{prot_path} missing columns: {missing_prot_cols}")

        drug_df = (
            drug_df[["DRUGID", "SMILES"]]
            .dropna(subset=["DRUGID", "SMILES"])
            .drop_duplicates(subset=["DRUGID"])
            .copy()
        )

        prot_df = (
            prot_df[["UNIPROTID", "SEQUENCE"]]
            .dropna(subset=["UNIPROTID", "SEQUENCE"])
            .drop_duplicates(subset=["UNIPROTID"])
            .copy()
        )

        prot_lengths = get_protein_lengths(prot_df["SEQUENCE"])
        drug_token_lengths = get_smiles_token_lengths(
            drug_df["SMILES"],
            tokenizer,
            add_special_tokens=add_special_tokens,
        )

        if len(prot_lengths) == 0:
            raise ValueError(f"No valid protein sequences found in {prot_path}")
        if len(drug_token_lengths) == 0:
            raise ValueError(f"No valid SMILES token lengths found in {drug_path}")

        prot_records.extend(
            {"Dataset": display_name, "Length": int(length)}
            for length in prot_lengths
        )

        drug_records.extend(
            {"Dataset": display_name, "TokenCount": int(length)}
            for length in drug_token_lengths
        )

        summary_rows.append(
            {
                "Dataset": display_name,
                "Protein_N": int(prot_lengths.shape[0]),
                "Protein_Mean": float(prot_lengths.mean()),
                "Protein_Median": float(prot_lengths.median()),
                "Protein_Min": int(prot_lengths.min()),
                "Protein_Max": int(prot_lengths.max()),
                "Protein_Q95": float(prot_lengths.quantile(0.95)),
                "Protein_Q99": float(prot_lengths.quantile(0.99)),
                "Protein_Within_2000": int((prot_lengths <= 2000).sum()),
                "Protein_Within_2000_Ratio": float((prot_lengths <= 2000).mean()),
                "Drug_N": int(drug_token_lengths.shape[0]),
                "DrugToken_Mean": float(drug_token_lengths.mean()),
                "DrugToken_Median": float(drug_token_lengths.median()),
                "DrugToken_Min": int(drug_token_lengths.min()),
                "DrugToken_Max": int(drug_token_lengths.max()),
                "DrugToken_Q95": float(drug_token_lengths.quantile(0.95)),
                "DrugToken_Q99": float(drug_token_lengths.quantile(0.99)),
                "DrugToken_Within_150": int((drug_token_lengths <= 150).sum()),
                "DrugToken_Within_150_Ratio": float((drug_token_lengths <= 150).mean()),
            }
        )

    return (
        pd.DataFrame(prot_records),
        pd.DataFrame(drug_records),
        pd.DataFrame(summary_rows),
    )


def save_table(df, output_dir, filename_prefix):
    csv_path = os.path.join(output_dir, f"{filename_prefix}.csv")
    xlsx_path = os.path.join(output_dir, f"{filename_prefix}.xlsx")

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    print(f"Saved: {csv_path}")
    print(f"Saved: {xlsx_path}")


def plot_density(df, x_col, title, xlabel, output_prefix, xlim_quantile=0.995):
    palette = {
        "SNAP": "#2F91DB",
        "DRH": "#F17135",
        "TTD": "#35B563",
    }

    plt.figure(figsize=(6, 6))

    for dataset in DATASETS.keys():
        values = df.loc[df["Dataset"] == dataset, x_col].dropna()

        if len(values) < 2:
            warnings.warn(f"Dataset {dataset} has fewer than 2 valid values. Skipped KDE plot.")
            continue

        sns.kdeplot(
            x=values,
            label=dataset,
            linewidth=2,
            fill=False,
            common_norm=False,
            color=palette.get(dataset, None),
        )

    upper = df[x_col].quantile(xlim_quantile)
    if pd.notna(upper) and upper > 0:
        plt.xlim(left=0, right=upper)

    # plt.title(title, fontsize=20, pad=20)
    plt.xlabel(xlabel, fontsize=18, labelpad=15)
    plt.ylabel("Density", fontsize=18, labelpad=15)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16, loc="upper right", frameon=True)
    plt.tight_layout()

    pdf_path = f"{output_prefix}.pdf"
    svg_path = f"{output_prefix}.svg"
    png_path = f"{output_prefix}.png"

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"Saved: {pdf_path}")
    print(f"Saved: {svg_path}")
    print(f"Saved: {png_path}")


def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("Loading MolFormer tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model_path"],
        trust_remote_code=True,
    )

    prot_df, drug_df, summary_df = collect_lengths(
        base_dir=CONFIG["base_dir"],
        datasets=DATASETS,
        tokenizer=tokenizer,
        add_special_tokens=CONFIG["add_special_tokens"],
    )

    save_table(prot_df, CONFIG["output_dir"], "protein_lengths")
    save_table(drug_df, CONFIG["output_dir"], "drug_smiles_token_counts")
    save_table(summary_df, CONFIG["output_dir"], "length_summary")

    plot_density(
        prot_df,
        x_col="Length",
        title="Protein sequence length distribution",
        xlabel="Protein sequence length",
        output_prefix=os.path.join(CONFIG["output_dir"], "protein_length_density"),
        xlim_quantile=CONFIG["xlim_quantile"],
    )

    plot_density(
        drug_df,
        x_col="TokenCount",
        title="SMILES token count distribution",
        xlabel="SMILES token count",
        output_prefix=os.path.join(CONFIG["output_dir"], "drug_smiles_token_density"),
        xlim_quantile=CONFIG["xlim_quantile"],
    )

    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved figures and tables to: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()