#!/usr/bin/env python
# Author  : KerryChen
# File    : get_esmc_feature.py
# Time    : 2026/05/30

import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


# =====================
# Utils
# =====================
def read_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext == ".xlsx":
        return pd.read_excel(file_path, engine="openpyxl")
    elif ext == ".pkl":
        return pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def split_sequence(seq, chunk_size):
    return [seq[i:i + chunk_size] for i in range(0, len(seq), chunk_size)]


def clean_sequence(seq):
    return str(seq).strip().replace(" ", "").replace("\n", "")


def extract_chunk_embedding(model, tokenizer, chunk, device):
    inputs = tokenizer(chunk, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )

    last_hidden = output.hidden_states[-1]

    # ESMC tokenizer adds one special token at the front and one at the end.
    residue_emb = last_hidden[0, 1:1 + len(chunk), :].detach().cpu()
    if residue_emb.shape[0] != len(chunk):
        raise RuntimeError(f"Embedding length mismatch: seq={len(chunk)}, emb={residue_emb.shape[0]}")

    return residue_emb


# =====================
# Config
# =====================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "/data/qfchen/lager_model/biohub/ESMC-300M"
DATASET_PATH = "/home/qfchen/DTIPred_Plus/database/UniprotKB/uniprotkb_2026_04_01.xlsx"
SAVE_DIR = "/data/qfchen/DTIPred_Plus/UniprotKB_ESMC_300M/"
os.makedirs(SAVE_DIR, exist_ok=True)

CHUNK_SIZE = 1000


# =====================
# Load data
# =====================
df = read_file(DATASET_PATH)
assert {"Entry", "Sequence"}.issubset(df.columns)

sub_df = df[["Entry", "Sequence"]].drop_duplicates()


# =====================
# Load ESMC
# =====================
model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH).to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


# =====================
# Feature extraction
# =====================
for _, row in tqdm(sub_df.iterrows(), total=len(sub_df)):
    pid = row["Entry"]
    seq = clean_sequence(row["Sequence"])

    if not seq or seq.lower() == "nan":
        print(f"[Skip] {pid}: empty sequence")
        continue

    save_path = os.path.join(SAVE_DIR, f"{pid}.npy")
    if os.path.exists(save_path):
        continue

    chunks = split_sequence(seq, CHUNK_SIZE)
    protein_embeddings = []

    try:
        print("Sequence Length: ", len(seq))
        for chunk in chunks:
            chunk_emb = extract_chunk_embedding(model, tokenizer, chunk, device)
            protein_embeddings.append(chunk_emb)

        protein_embeddings = torch.cat(protein_embeddings, dim=0)
        if protein_embeddings.shape[0] != len(seq):
            raise RuntimeError(
                f"Protein length mismatch: seq={len(seq)}, emb={protein_embeddings.shape[0]}"
            )

        print(protein_embeddings.shape)
        np.save(save_path, protein_embeddings.numpy())

    except Exception as e:
        print(f"[Error] {pid}: {e}")
