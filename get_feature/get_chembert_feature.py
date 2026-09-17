#!/usr/bin/env python
# Author  : KerryChen
# File    : get_drug_feature.py
# Time    : 2025/7/26 15:51

import os
import pickle
import pandas as pd
import torch
from tqdm import tqdm
from rdkit import Chem
from transformers import AutoModel, RobertaTokenizer

def read_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.csv':
        df = pd.read_csv(file_path)
        return df
    elif file_extension.lower() == '.xlsx':
        df = pd.read_excel(file_path, engine="openpyxl")
        return df
    elif file_extension.lower() == '.pkl':
        df = pd.read_pickle(file_path)
        return df
    else:
        raise ValueError(f"Unsupported file types: {file_extension}")


def extract_drug_features(task, model_name, drug_df, model_path: str, device: str = "cpu"):
    """
    Extract molecular features using ChemBERTa and return as a dictionary.

    Args:
        drug_df (pd.DataFrame): Must contain 'DRUG_ID' and 'SMILES' columns.
        model_path (str): Path to the pre-trained ChemBERTa model.
        device (str): Device to run the model, e.g., "cuda" or "cpu".

    Returns:
        dict: {drug_id: feature_tensor [seq_len, hidden_dim]}
    """
    SAVE_DIR = f'/data/qfchen/DTIPred_Plus/{task}/{model_name}/'
    os.makedirs(SAVE_DIR, exist_ok=True)

    mol_tokenizer = RobertaTokenizer.from_pretrained(model_path)
    mol_encoder = AutoModel.from_pretrained(model_path).to(device)
    mol_encoder.eval()

    print('The number of drugs: ', len(drug_df))
    for _, row in tqdm(drug_df.iterrows(), total=len(drug_df)):
        drug_id = row['DRUGID']
        smiles = row['SMILES']
        print(drug_id)
   
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            smiles = Chem.MolToSmiles(mol, canonical=True)


        # with torch.no_grad():
        #     out = mol_encoder(**tokens)

        # # feat = out.last_hidden_state.detach().cpu()[0]  # shape: [seq_len, hidden_dim]
        # feat = out.last_hidden_state.detach().cpu()[0][1:-1]
        tokens = mol_tokenizer(
                                smiles,
                                max_length=500,
                                padding=False,
                                truncation=True,
                                return_tensors="pt",
                                return_special_tokens_mask=True).to(device)

        with torch.no_grad():
            out = mol_encoder(**{k: v for k, v in tokens.items() if k != "special_tokens_mask"})

        hidden = out.last_hidden_state.detach().cpu()[0]
        special_mask = tokens["special_tokens_mask"].detach().cpu()[0].bool()

        feat = hidden[~special_mask]

        print(tokens['input_ids'].shape[1], feat.shape)
        save_path = os.path.join(SAVE_DIR, f"{drug_id}.pt")
        torch.save(feat, save_path)

  
if __name__ == '__main__':
    model_name = "ChemBERTa_100M"
    task = 'TPP'
    drug_df = read_file(f'/home/qfchen/DTIPred_Plus/database/{task}/DrugInfo.xlsx')
    # drug_df = read_file(f'/home/qfchen/DTIPred_Plus/database/TPP/final_data/CaseStudy.csv')

    # tokenizer = RobertaTokenizer.from_pretrained("/data/qfchen/lager_model/ChemBERTa-100M-MLM")
    # total = len(drug_df)
    # counts = {'>100': 0, '>200': 0, '>300': 0, '>400': 0, '>500': 0}
    # max_tokens = 0
    # max_drug = ""

    # for _, row in drug_df.iterrows():
        # smiles = row['DRUGSMIL']
        # mol = Chem.MolFromSmiles(smiles)
        # if mol is not None:
        #     smiles = Chem.MolToSmiles(mol, canonical=True)
        # tokens = tokenizer(smiles, padding=False, truncation=False, return_tensors='pt')
        # n = tokens['input_ids'].shape[1]
    #     if n > max_tokens:
    #         max_tokens = n
    #         max_drug = row['DRUGINKE']
    #     if n > 100: counts['>100'] += 1
    #     if n > 200: counts['>200'] += 1
    #     if n > 300: counts['>300'] += 1
    #     if n > 400: counts['>400'] += 1
    #     if n > 500: counts['>500'] += 1

    # print(f"Total molecules: {total}")
    # print(f"Max token length: {max_tokens} ({max_drug})")
    # for k, v in counts.items():
    #     print(f"Tokens {k}: {v}")

    
    # total = len(drug_df)
    # over_100 = 0
    # over_200 = 0
    # over_300 = 0
    # over_500 = 0

    # for _, row in drug_df.iterrows():
    #     mol = Chem.MolFromSmiles(row['DRUGSMIL'])
    #     if mol is not None:
    #         n = mol.GetNumAtoms()
    #         if n > 100: over_100 += 1
    #         if n > 200: over_200 += 1
    #         if n > 300: over_300 += 1
    #         if n > 500: over_500 += 1

    # print(f"Total: {total}")
    # print(f"Atoms > 100: {over_100}")
    # print(f"Atoms > 200: {over_200}")
    # print(f"Atoms > 300: {over_300}")
    # print(f"Atoms > 500: {over_500}")

    drug_features = extract_drug_features(
        task,
        model_name,
        drug_df,
        # model_path="/data/qfchen/lager_model/ChemBERTa-zinc-base-v1",
        model_path="/data/qfchen/lager_model/ChemBERTa-100M-MLM",
        device="cpu")