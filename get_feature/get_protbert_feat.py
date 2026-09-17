import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, AutoModel
import json
from pathlib import Path

# -----------------------------
# 配置
# -----------------------------
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model_dir = Path("/data/qfchen/lager_model/prot_bert")
config_path = model_dir / "config.json"

with open(config_path, "r") as f:
    config = json.load(f)

config["model_type"] = "bert"

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print("Fixed config.json: added model_type = bert")


prot_tokenizer = BertTokenizer.from_pretrained(
    "/data/qfchen/lager_model/prot_bert/",
    do_lower_case=False
)
prot_encoder = AutoModel.from_pretrained(
    "/data/qfchen/lager_model/prot_bert/"
).to(device)
prot_encoder.eval()

SAVE_DIR = "/data/qfchen/DTIPred_Plus/UniprotKB_ProtBert/"
os.makedirs(SAVE_DIR, exist_ok=True)

DATASET_PATH = "/home/qfchen/DTIPred_Plus/database/UniprotKB/uniprotkb_2026_04_01.xlsx"
df = pd.read_excel(DATASET_PATH)
sub_df = df[["Entry", "Sequence"]].drop_duplicates()

# -----------------------------
# 长序列配置（关键）
# -----------------------------
CHUNK_SIZE = 1000
def split_sequence(seq, chunk_size):
    return [seq[i:i + chunk_size] for i in range(0, len(seq), chunk_size)]

# -----------------------------
# 主循环
# -----------------------------
for _, row in tqdm(sub_df.iterrows(), total=len(sub_df)):

    pid = row['Entry']
    seq = row['Sequence'].replace(" ", "").upper()
    print(f"\n>>> Processing {pid}, length = {len(seq)}")

    save_path = os.path.join(SAVE_DIR, f"{pid}.npy")
    if os.path.exists(save_path):
        continue

    chunks = split_sequence(seq, CHUNK_SIZE)
    all_embeddings = []

    for i, chunk in enumerate(chunks):
        print(f"  - Chunk {i+1}/{len(chunks)} | length: {len(chunk)}")

        chunk_spaced = " ".join(list(chunk))
        encoded = prot_tokenizer(
            chunk_spaced,
            return_tensors="pt",
            truncation=False
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = prot_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # shape: [1, L+2, 1024]  (CLS + tokens + SEP)
        emb = outputs.last_hidden_state.squeeze(0).cpu()

        # 去掉 [CLS] 和 [SEP]
        emb = emb[1:-1]

        all_embeddings.append(emb)
        torch.cuda.empty_cache()

    # 拼接为完整蛋白 embedding
    seq_feat = torch.cat(all_embeddings, dim=0)
    print(f"✅ {pid}: final shape = {seq_feat.shape}")
    np.save(save_path, seq_feat.numpy())

    # save_path = os.path.join(SAVE_DIR, f"{pid}.pt")
    # torch.save(seq_feat, save_path)

    torch.cuda.empty_cache()
