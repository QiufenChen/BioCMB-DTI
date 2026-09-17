#!/usr/bin/env python
# Author  : KerryChen
# File    : get_esm_feature.py
# Time    : 2025/08/11

import os
import torch
import pandas as pd
from tqdm import tqdm
import esm
import numpy as np


# =====================
# Utils
# =====================
def read_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext == '.xlsx':
        return pd.read_excel(file_path, engine="openpyxl")
    elif ext == '.pkl':
        return pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def split_sequence(seq, chunk_size):
    return [seq[i:i + chunk_size] for i in range(0, len(seq), chunk_size)]


# =====================
# Config
# =====================
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

DATASET_PATH = '/home/qfchen/DTIPred_Plus/database/UniprotKB/uniprotkb_2026_04_01.xlsx'
SAVE_DIR = '/data/qfchen/DTIPred_Plus/UniprotKB_650M/'
os.makedirs(SAVE_DIR, exist_ok=True)

CHUNK_SIZE = 1000        # safe for ESM-2
REPR_LAYER = 33          # for esm2_t36_3B


# =====================
# Load data
# =====================
df = read_file(DATASET_PATH)
assert {'Entry', 'Sequence'}.issubset(df.columns)

sub_df = df[['Entry', 'Sequence']].drop_duplicates()


# =====================
# Load ESM-2
# =====================
# model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
model.eval()
batch_converter = alphabet.get_batch_converter()


# =====================
# Feature extraction
# =====================
for _, row in tqdm(sub_df.iterrows(), total=len(sub_df)):
    pid = row['Entry']
    seq = row['Sequence']
    print('Sequence Length: ', len(seq))
    save_path = os.path.join(SAVE_DIR, f"{pid}.npy")
    if os.path.exists(save_path):
        continue

    chunks = split_sequence(seq, CHUNK_SIZE)
    protein_embeddings = []

    try:
        with torch.no_grad():
            for i, chunk in enumerate(chunks):
                data = [(f"{pid}_chunk{i}", chunk)]
                _, _, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(device)

                batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

                results = model(batch_tokens, repr_layers=[REPR_LAYER])
                token_emb = results["representations"][REPR_LAYER]

                # remove <cls> and <eos>
                token_emb = token_emb[:, 1:batch_lens - 1, :]
                token_emb = token_emb.squeeze(0).cpu()  # (len_chunk, dim)

                protein_embeddings.append(token_emb)

        # concat all chunks → (L, dim)
        protein_embeddings = torch.cat(protein_embeddings, dim=0)
        print(protein_embeddings.shape)

        # torch.save(protein_embeddings, save_path)
        np.save(save_path, protein_embeddings.numpy())

    except Exception as e:
        print(f"[Error] {pid}: {e}")
