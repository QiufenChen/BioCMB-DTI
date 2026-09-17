#!/usr/bin/env python
# Author  : Kerry Chen
# File    : get_prott5_feature_encoder.py

import os
import re
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer


# -----------------------------
# 配置
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "/data/qfchen/DTIPred_Plus/UniprotKB_ProtT5/"
os.makedirs(SAVE_DIR, exist_ok=True)


DATASET_PATH = "/home/qfchen/DTIPred_Plus/database/UniprotKB/uniprotkb_2026_04_01.xlsx"
df = pd.read_excel(DATASET_PATH)
sub_df = df[["Entry", "Sequence"]].drop_duplicates()

# -----------------------------
# 数据读取
# -----------------------------
def read_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    if ext == ".pkl":
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported file type: {ext}")


# -----------------------------
# 加载 ProtT5
# -----------------------------
model_name = "/data/qfchen/lager_model/ProtT5"
tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
model = T5EncoderModel.from_pretrained(model_name).to(device).eval()

print("ProtT5 encoder loaded")


# -----------------------------
# 参数
# -----------------------------
CHUNK_SIZE = 1000


def split_sequence(seq, size):
    return [seq[i:i + size] for i in range(0, len(seq), size)]


# -----------------------------
# 特征提取
# -----------------------------
for _, row in tqdm(sub_df.iterrows(), total=len(sub_df)):

    pid = row["Entry"]
    seq = re.sub(r"[UZOB]", "X", row["Sequence"].upper())
    print(f"\n>>> Processing {pid}, length = {len(seq)}")


    save_path = os.path.join(SAVE_DIR, f"{pid}.npy")
    if os.path.exists(save_path):
        continue


    chunks = split_sequence(seq, CHUNK_SIZE)
    embeddings = []

    for i, chunk in enumerate(chunks):
        print(f"  - Chunk {i + 1}/{len(chunks)} | length: {len(chunk)}")
        chunk = " ".join(chunk)
        encoded = tokenizer(chunk, return_tensors="pt", truncation=False)
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

        emb = output.last_hidden_state.squeeze(0).cpu()

        # ProtT5：移除末尾 </s>
        embeddings.append(emb[:-1])

    seq_feat = torch.cat(embeddings, dim=0)
    print(f"✅ {pid}: final shape = {seq_feat.shape}")
    # torch.save(seq_feat, os.path.join(SAVE_DIR, f"{pid}.pt"))

    np.save(save_path, seq_feat.numpy())

    torch.cuda.empty_cache()
