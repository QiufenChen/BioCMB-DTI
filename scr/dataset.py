"""
Data loading utilities for pre-split DTI datasets.

This loader uses only pretrained large-model features:
  - Drug features:    {drug_id}.pt
  - Protein features: {uniprot_id}.npy
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def _load_feature(path: str):
    if path.endswith(".npy"):
        return np.load(path, allow_pickle=False)

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class DTIDataset(Dataset):
    def __init__(
        self,
        data,
        esm2_dir,
        drug_feat_dir,
        max_drug_len=512,
        max_prot_len=1024,
        preload=True,
        prot_cache=None,
        drug_cache=None,
    ):
        self.data = data
        self.esm2_dir = esm2_dir
        self.drug_feat_dir = drug_feat_dir
        self.max_drug_len = max_drug_len
        self.max_prot_len = max_prot_len
        self._prot_cache = prot_cache if prot_cache is not None else {}
        self._drug_cache = drug_cache if drug_cache is not None else {}

        if preload:
            self._preload_features()

    def _preload_features(self):
        prot_ids = self.data["UNIPROTID"].apply(lambda x: str(x).strip()).unique()
        drug_ids = self.data["DRUGID"].apply(lambda x: str(x).strip()).unique()

        print(f"  Preloading {len(prot_ids)} protein features...")
        loaded = 0
        for pid in prot_ids:
            if pid in self._prot_cache:
                loaded += 1
                continue
            path = os.path.join(self.esm2_dir, f"{pid}.npy")
            if os.path.exists(path):
                self._prot_cache[pid] = _load_feature(path)
                loaded += 1
        print(f"  Loaded {loaded}/{len(prot_ids)} proteins into memory")

        print(f"  Preloading {len(drug_ids)} drug features...")
        loaded = 0
        for did in drug_ids:
            if did in self._drug_cache:
                loaded += 1
                continue
            path = os.path.join(self.drug_feat_dir, f"{did}.pt")
            if os.path.exists(path):
                self._drug_cache[did] = _load_feature(path)
                loaded += 1
        print(f"  Loaded {loaded}/{len(drug_ids)} drugs into memory")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        drug_key = str(row["DRUGID"]).strip()
        prot_key = str(row["UNIPROTID"]).strip()
        label = int(row["Label"])

        if drug_key in self._drug_cache:
            drug_feat = self._drug_cache[drug_key]
        else:
            drug_path = os.path.join(self.drug_feat_dir, f"{drug_key}.pt")
            drug_feat = _load_feature(drug_path)
            self._drug_cache[drug_key] = drug_feat
            print(f"Warning: Drug feature for {drug_key} not preloaded, loaded on-the-fly")

        if prot_key in self._prot_cache:
            prot_feat = self._prot_cache[prot_key]
        else:
            prot_path = os.path.join(self.esm2_dir, f"{prot_key}.npy")
            prot_feat = _load_feature(prot_path)
            self._prot_cache[prot_key] = prot_feat
            print(f"Warning: Protein feature for {prot_key} not preloaded, loaded on-the-fly")

        drug_feat = self._to_tensor(drug_feat)
        prot_feat = self._to_tensor(prot_feat)

        if drug_feat.dim() == 1:
            drug_feat = drug_feat.unsqueeze(0)
        if prot_feat.dim() == 1:
            prot_feat = prot_feat.unsqueeze(0)

        drug_len = min(drug_feat.shape[0], self.max_drug_len)
        prot_len = min(prot_feat.shape[0], self.max_prot_len)

        drug_feat = self._pad_truncate(drug_feat, self.max_drug_len)
        prot_feat = self._pad_truncate(prot_feat, self.max_prot_len)

        drug_mask = torch.zeros(self.max_drug_len)
        prot_mask = torch.zeros(self.max_prot_len)
        drug_mask[:drug_len] = 1
        prot_mask[:prot_len] = 1

        return {
            "drug_feat": drug_feat,
            "prot_feat": prot_feat,
            "drug_mask": drug_mask,
            "prot_mask": prot_mask,
            "label": label,
            "drug_key": drug_key,
            "prot_key": prot_key,
        }

    @staticmethod
    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.float()
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        return torch.tensor(x, dtype=torch.float32)

    @staticmethod
    def _pad_truncate(feat, max_len):
        length, dim = feat.shape
        if length >= max_len:
            return feat[:max_len]
        pad = torch.zeros(max_len - length, dim)
        return torch.cat([feat, pad], dim=0)


def _read_table(data_file: str):
    extension = os.path.splitext(data_file)[1].lower()
    if extension in (".xlsx", ".xls"):
        return pd.read_excel(data_file)
    return pd.read_csv(data_file)


def _validate_table(data, data_file):
    required_columns = {"DRUGID", "UNIPROTID", "Label"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "Missing required columns in {}: {}".format(
                data_file, ", ".join(sorted(missing_columns))
            )
        )

    labels = pd.to_numeric(data["Label"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("Label must contain only 0 and 1 in {}".format(data_file))
    data["Label"] = labels.astype(int)


def _filter_available(data, esm2_dir, drug_feat_dir):
    available = []

    for idx, row in data.iterrows():
        drug_key = str(row["DRUGID"]).strip()
        prot_key = str(row["UNIPROTID"]).strip()

        drug_path = os.path.join(drug_feat_dir, f"{drug_key}.pt")
        prot_path = os.path.join(esm2_dir, f"{prot_key}.npy")

        if os.path.exists(drug_path) and os.path.exists(prot_path):
            available.append(idx)

    return data.iloc[available].reset_index(drop=True)


def collate_fn(batch):
    return {
        "drug_feat": torch.stack([item["drug_feat"] for item in batch]),
        "prot_feat": torch.stack([item["prot_feat"] for item in batch]),
        "drug_mask": torch.stack([item["drug_mask"] for item in batch]),
        "prot_mask": torch.stack([item["prot_mask"] for item in batch]),
        "label": torch.tensor([item["label"] for item in batch]),
        "drug_key": [item["drug_key"] for item in batch],
        "prot_key": [item["prot_key"] for item in batch],
    }


def build_split_dataloader(
    data_file,
    esm2_dir,
    drug_feat_dir,
    batch_size=32,
    num_workers=0,
    seed=42,
    pin_memory=False,
    max_drug_len=512,
    max_prot_len=1024,
    shuffle=False,
    drop_last=False,
    preload=True,
    prot_cache=None,
    drug_cache=None,
):
    """Create one DataLoader from a train, validation, or test split file."""
    data = _read_table(data_file)
    _validate_table(data, data_file)
    data = _filter_available(data, esm2_dir, drug_feat_dir)
    if len(data) == 0:
        raise ValueError("Empty split after feature filtering: {}".format(data_file))
    if shuffle and len(data) < 2:
        raise ValueError(
            "Training split must contain at least two samples after feature filtering: {}".format(
                data_file
            )
        )

    dataset = DTIDataset(
        data,
        esm2_dir,
        drug_feat_dir,
        max_drug_len=max_drug_len,
        max_prot_len=max_prot_len,
        preload=preload,
        prot_cache=prot_cache,
        drug_cache=drug_cache,
    )
    positives = int(data["Label"].sum())
    negatives = len(data) - positives
    print(
        "    {}: {} (pos: {}, neg: {})".format(
            os.path.basename(data_file), len(dataset), positives, negatives
        )
    )

    effective_drop_last = drop_last and len(dataset) >= batch_size
    if drop_last and not effective_drop_last:
        print(
            "    [WARNING] {} has fewer samples than batch_size; keeping its final batch.".format(
                os.path.basename(data_file)
            )
        )

    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        generator=generator,
        drop_last=effective_drop_last,
        pin_memory=pin_memory,
    )


def get_dataloaders_from_split_files(
    train_file,
    val_file,
    test_file,
    esm2_dir,
    drug_feat_dir,
    batch_size=32,
    num_workers=0,
    seed=42,
    pin_memory=False,
    max_drug_len=512,
    max_prot_len=1024,
):
    prot_cache, drug_cache = {}, {}
    common = {
        "esm2_dir": esm2_dir,
        "drug_feat_dir": drug_feat_dir,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seed": seed,
        "pin_memory": pin_memory,
        "max_drug_len": max_drug_len,
        "max_prot_len": max_prot_len,
        "prot_cache": prot_cache,
        "drug_cache": drug_cache,
    }
    train_loader = build_split_dataloader(
        train_file, shuffle=True, drop_last=True, **common
    )
    val_loader = build_split_dataloader(val_file, **common)
    test_loader = build_split_dataloader(test_file, **common)

    return train_loader, val_loader, test_loader
