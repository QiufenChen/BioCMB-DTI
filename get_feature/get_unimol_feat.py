#!/usr/bin/env python

import os
import glob
import faulthandler
import pandas as pd

faulthandler.enable()


def read_file(file_path):
    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() == ".csv":
        return pd.read_csv(file_path)

    if file_extension.lower() == ".xlsx":
        return pd.read_excel(file_path, engine="openpyxl")

    if file_extension.lower() == ".pkl":
        return pd.read_pickle(file_path)

    raise ValueError(f"Unsupported file type: {file_extension}")


def find_first_file(patterns):
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[0]
    return None


def build_unimol_repr(model_dir, model_size, use_cuda=True):
    checkpoint_path = os.path.join(model_dir, "checkpoint.pt")

    if not os.path.exists(checkpoint_path):
        checkpoint_path = find_first_file([
            os.path.join(model_dir, "*.pt"),
            os.path.join(model_dir, "**", "*.pt"),
        ])

    dict_path = find_first_file([
        os.path.join(model_dir, "dict.txt"),
        os.path.join(model_dir, "mol.dict.txt"),
        os.path.join(model_dir, "**", "dict.txt"),
        os.path.join(model_dir, "**", "mol.dict.txt"),
    ])

    if checkpoint_path is None:
        raise FileNotFoundError(f"No .pt Uni-Mol checkpoint found under: {model_dir}")

    print(f"Using Uni-Mol checkpoint: {checkpoint_path}")
    print(f"Using Uni-Mol dictionary: {dict_path if dict_path else 'default'}")
    print(f"Using CUDA: {use_cuda}")

    kwargs = {
        "data_type": "molecule",
        "remove_hs": False,
        "model_name": "unimolv2",
        "model_size": model_size,
        "use_cuda": use_cuda,
        "pretrained_model_path": checkpoint_path,
    }

    if use_cuda:
        kwargs["use_gpu"] = USE_GPU

    if dict_path is not None:
        kwargs["pretrained_dict_path"] = dict_path

    return UniMolRepr(**kwargs)


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def extract_unimol_features(task, drug_df, model_dir, model_size, save_dir, use_cuda=True):
    os.makedirs(save_dir, exist_ok=True)

    clf = build_unimol_repr(
        model_dir=model_dir,
        model_size=model_size,
        use_cuda=use_cuda,
    )

    failed = []

    for _, row in tqdm(drug_df.iterrows(), total=len(drug_df)):
        drug_id = row["DRUGID"]
        smiles = row["SMILES"]

        canonical_smiles = canonicalize_smiles(smiles)

        if canonical_smiles is None:
            failed.append(drug_id)
            print(f"[WARNING] Invalid SMILES, skipped: {drug_id}, SMILES={smiles}")
            continue

        save_path = os.path.join(save_dir, f"{drug_id}.pt")

        if os.path.exists(save_path):
            continue

        try:
            unimol_repr = clf.get_repr(
                [canonical_smiles],
                return_atomic_reprs=True
            )

            atomic_repr = np.asarray(unimol_repr["atomic_reprs"][0])

            if atomic_repr.ndim != 2:
                raise ValueError(f"Unexpected atomic_repr shape: {atomic_repr.shape}")

            feat = torch.from_numpy(atomic_repr).float()
            torch.save(feat, save_path)

            print(f"{drug_id} | shape: {feat.shape}")

        except Exception as exc:
            failed.append(drug_id)
            print(f"[WARNING] Failed to extract {drug_id}: {exc}")

    print(f"Saved Uni-Mol features to: {save_dir}")
    print(f"Finished task: {task}, total={len(drug_df)}, failed={len(failed)}")

    if failed:
        print("Failed drug ids:", failed)


if __name__ == "__main__":
    TASK = "TTD"
    MODEL_NAME = "UniMol2_164M"
    MODEL_DIR = "/data/qfchen/lager_model/Uni_Mol2/164M"
    MODEL_SIZE = "164m"

    DATA_PATH = f"/home/qfchen/DTIPred_Plus/database/{TASK}/final_data/DrugInfo.xlsx"
    SAVE_DIR = f"/data/qfchen/DTIPred_Plus/{TASK}/{MODEL_NAME}"

    # 1. 先读 Excel
    drug_info = read_file(DATA_PATH)

    print(f"Loaded drug data: {len(drug_info)} records")

    drug_df = drug_info[["DRUGID", "SMILES"]].drop_duplicates()

    print(f"Loaded drug data: {len(drug_df)} unique drugs")

    # 2. 读完 Excel 后，再导入 UniMol / torch / rdkit
    import numpy as np
    import torch
    from rdkit import Chem
    from tqdm import tqdm
    from unimol_tools import UniMolRepr

    print("UniMolRepr imported successfully")

    USE_CUDA = torch.cuda.is_available()
    USE_GPU = 3

    extract_unimol_features(
        task=TASK,
        drug_df=drug_df,
        model_dir=MODEL_DIR,
        model_size=MODEL_SIZE,
        save_dir=SAVE_DIR,
        use_cuda=True,
    )