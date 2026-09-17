#!/usr/bin/env python
# Author  : KerryChen
# File    : get_ankh_feature.py
# Time    : 2026/01/04 22:21

import os
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, T5EncoderModel

# ===============================================================
# 1. Configuration
# ===============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = "/data/qfchen/lager_model/Ankh_large/"
tokenizer = AutoTokenizer.from_pretrained(model_dir, do_lower_case=False)

# 🔥 关键修正：只加载 Encoder
model = T5EncoderModel.from_pretrained(model_dir).to(device)
model.eval()

SAVE_DIR = "/data/qfchen/DTIPred_Plus/UniprotKB_Ankh/"
os.makedirs(SAVE_DIR, exist_ok=True)

DATASET_PATH = "/home/qfchen/DTIPred_Plus/database/UniprotKB/uniprotkb_2026_04_01.xlsx"
df = pd.read_excel(DATASET_PATH)
sub_df = df[["Entry", "Sequence"]].drop_duplicates()

# ===============================================================
# 2. Long sequence handling
# ===============================================================
CHUNK_SIZE = 1000

def split_sequence(seq, max_len):
    return [seq[i:i + max_len] for i in range(0, len(seq), max_len)]


# ===============================================================
# 3. Main loop
# ===============================================================
for _, row in tqdm(sub_df.iterrows(), total=len(sub_df)):
    pid = row["Entry"]
    seq = row["Sequence"].replace(" ", "").upper()
    print(f"\n>>> Processing {pid} | length = {len(seq)}")

    save_path = os.path.join(SAVE_DIR, f"{pid}.npy")
    if os.path.exists(save_path):
        continue

    chunks = split_sequence(seq, CHUNK_SIZE)
    all_embeddings = []

    for i, chunk in enumerate(chunks):
        print(f"  - Chunk {i + 1}/{len(chunks)} | length: {len(chunk)}")

        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=False
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # [0, L+1, D] → remove <eos>
        emb = outputs.last_hidden_state.squeeze(0)[:-1].cpu()


        all_embeddings.append(emb)
        torch.cuda.empty_cache()

    seq_feat = torch.cat(all_embeddings, dim=0)
    print(f"✅ {pid}: final shape = {seq_feat.shape}")

    # torch.save(seq_feat, os.path.join(SAVE_DIR, f"{pid}.pt"))
    np.save(save_path, seq_feat.numpy())
    # torch.cuda.empty_cache()
