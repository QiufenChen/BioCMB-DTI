#!/usr/bin/env python

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


MODEL_DIR = Path(r"/data/qfchen/lager_model/biohub/esm3-sm-open-v1")
MODEL_ID = "esm3_sm_open_v1"
DATASET_PATH = "/home/qfchen/DTIPred_Plus/database/UniprotKB/uniprotkb_2026_04_01.xlsx"
SAVE_DIR = "/data/qfchen/DTIPred_Plus/UniprotKB_ESM3_SM_OPEN_V1/"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = torch.float32
CHUNK_SIZE = 1000


def read_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    if ext == ".xlsx":
        return pd.read_excel(file_path, engine="openpyxl")
    if ext == ".pkl":
        return pd.read_pickle(file_path)
    raise ValueError(f"Unsupported file type: {ext}")


def clean_sequence(seq):
    return str(seq).strip().replace(" ", "").replace("\n", "")


def split_sequence(seq, chunk_size):
    return [seq[i:i + chunk_size] for i in range(0, len(seq), chunk_size)]


def patch_esm3_data_root(model_dir):
    model_dir = Path(model_dir).resolve()
    try:
        import esm.pretrained as esm_pretrained
        import esm.utils.constants.esm3 as esm3_constants
    except ModuleNotFoundError as exc:
        import esm

        raise ModuleNotFoundError(
            "Your current Python environment does not have EvolutionaryScale's ESM package "
            "with esm.pretrained. Please reinstall it with:\n"
            "  pip uninstall -y esm fair-esm\n"
            "  pip install git+https://github.com/evolutionaryscale/esm.git\n"
            f"Current esm package path: {getattr(esm, '__file__', 'unknown')}"
        ) from exc

    def local_data_root(model_name):
        return model_dir

    esm_pretrained.data_root = local_data_root
    esm3_constants.data_root = local_data_root


def load_esm3_model():
    patch_esm3_data_root(MODEL_DIR)

    from esm.models.esm3 import ESM3

    print(f"Loading ESM3 from: {MODEL_DIR}", flush=True)
    model = ESM3.from_pretrained(MODEL_ID).to(DEVICE)
    model = model.to(dtype=MODEL_DTYPE)
    model.eval()
    print(f"Model device: {DEVICE}, dtype: {MODEL_DTYPE}", flush=True)
    return model


def extract_chunk_embedding(model, chunk):
    from esm.sdk.api import ESMProtein, LogitsConfig

    protein = ESMProtein(sequence=chunk)
    protein_tensor = model.encode(protein).to(DEVICE)

    with torch.inference_mode():
        output = model.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True),
        )

    residue_emb = output.embeddings[0, 1:1 + len(chunk), :].detach().cpu()
    if residue_emb.shape[0] != len(chunk):
        raise RuntimeError(
            f"Embedding length mismatch: seq={len(chunk)}, emb={residue_emb.shape[0]}"
        )
    return residue_emb


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    df = read_file(DATASET_PATH)
    assert {"Entry", "Sequence"}.issubset(df.columns)
    sub_df = df[["Entry", "Sequence"]].drop_duplicates()

    model = load_esm3_model()

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
            print("Sequence Length:", len(seq), flush=True)
            for chunk in chunks:
                chunk_emb = extract_chunk_embedding(model, chunk)
                protein_embeddings.append(chunk_emb)

            protein_embeddings = torch.cat(protein_embeddings, dim=0)
            if protein_embeddings.shape[0] != len(seq):
                raise RuntimeError(
                    f"Protein length mismatch: seq={len(seq)}, emb={protein_embeddings.shape[0]}"
                )

            print(protein_embeddings.shape, flush=True)
            np.save(save_path, protein_embeddings.numpy())

        except Exception as exc:
            print(f"[Error] {pid}: {exc}", flush=True)


if __name__ == "__main__":
    main()
