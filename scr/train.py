#!/usr/bin/env python

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc,f1_score, matthews_corrcoef, 
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from tqdm import tqdm

from dataset import build_split_dataloader
from model import CNNAttentionDTI
from utils import EarlyStopping, Logger


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_CONFIG = {
    "seed": 3407,
    "data_name": "SNAP",
    'data_type': 'final_data',
    "base_db_dir": os.path.join(PROJECT_ROOT, "database"),
    "protein_feat_dir": os.environ.get(
        "BIOCMB_PROTEIN_FEAT_DIR", os.path.join(PROJECT_ROOT, "UniprotKB_Ankh")
    ),
    "drug_feat_root": os.environ.get("BIOCMB_DRUG_FEAT_ROOT", PROJECT_ROOT),
    "drug_feat_name": "MolFormer",
    "drug_feat_dir": None,
    "drug_dim": 768,
    "protein_dim": 1536,
    "cnn_dim": 256,
    "drug_kernel_sizes": (3, 5, 7),
    "protein_kernel_sizes": (3, 7, 15),
    "mamba_layers": 2,
    "ban_glimpses": 4,
    "ban_k": 3,
    "dropout": 0.3,
    "batch_size": 32,
    "lr": 5e-5,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 10,
    "early_stop_start_epoch": 6,
    "max_drug_len": 150,
    "max_prot_len": 2000,
    "num_workers": 0,
    "preload_features": True,
    "cuda_device": 0,
    "output_root": os.path.join(PROJECT_ROOT, "outputs"),
    "model_dir": None,
    "result_dir": None,
    "log_dir": None,
    "result_file": "summary.xlsx",
    "ratio_dir": "1_1",
    "split_mode": "random_split",
    "group_ids": range(1, 6),
}

# Kept for compatibility with older run_*.py files that mutate train.CONFIG.
CONFIG = DEFAULT_CONFIG.copy()


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def merge_config(overrides=None):
    """Return an independent, fully resolved configuration."""
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(CONFIG)
    if overrides:
        cfg.update(overrides)
    cfg["data_dir"] = os.path.join(cfg["base_db_dir"], cfg["data_name"], cfg["data_type"])
    if not cfg["drug_feat_dir"]:
        cfg["drug_feat_dir"] = os.path.join(
            cfg["drug_feat_root"], cfg["data_name"], cfg["drug_feat_name"]
        )
    cfg["model_dir"] = cfg["model_dir"] or os.path.join(cfg["output_root"], "checkpoints")
    cfg["result_dir"] = cfg["result_dir"] or os.path.join(cfg["output_root"], "results")
    cfg["log_dir"] = cfg["log_dir"] or os.path.join(cfg["output_root"], "logs")
    return cfg


def clone_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def select_device(cuda_device):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    device_count = torch.cuda.device_count()
    if cuda_device < 0 or cuda_device >= device_count:
        raise ValueError(
            "cuda_device={} is invalid; available CUDA device indices are 0-{}.".format(
                cuda_device, device_count - 1
            )
        )
    return torch.device("cuda:{}".format(cuda_device))


def validate_config(cfg):
    if cfg["batch_size"] < 2:
        raise ValueError("batch_size must be at least 2 because the model uses BatchNorm.")
    if cfg["epochs"] < 1:
        raise ValueError("epochs must be at least 1.")
    if not cfg["group_ids"]:
        raise ValueError("group_ids cannot be empty.")


def calculate_metrics(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    pre, rec, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(rec, pre)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
        "AUROC": auroc,
        "AUPR": aupr,
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    valid_batches = 0

    for _, batch in tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader), desc="Training"):
        drug_feat = batch["drug_feat"].to(device, non_blocking=True)
        prot_feat = batch["prot_feat"].to(device, non_blocking=True)
        drug_mask = batch["drug_mask"].to(device, non_blocking=True)
        prot_mask = batch["prot_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        output = model(drug_feat, prot_feat, drug_mask, prot_mask)
        loss = criterion(output["logits"], labels.long())

        if not torch.isfinite(loss):
            print("[WARNING] NaN/Inf train loss detected, skipping batch")
            continue

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            print("[WARNING] NaN/Inf gradient detected, skipping optimizer step")
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()

        total_loss += loss.item()
        valid_batches += 1

    if valid_batches == 0:
        return float("nan")
    return total_loss / valid_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_outputs=False):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    total_loss, valid_batches = 0.0, 0

    for _, batch in tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader)):
        drug_feat = batch["drug_feat"].to(device, non_blocking=True)
        prot_feat = batch["prot_feat"].to(device, non_blocking=True)
        drug_mask = batch["drug_mask"].to(device, non_blocking=True)
        prot_mask = batch["prot_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        output = model(drug_feat, prot_feat, drug_mask, prot_mask)
        loss = criterion(output["logits"], labels.long())

        if not torch.isfinite(loss):
            print("[WARNING] NaN/Inf eval loss detected, skipping batch")
            continue

        probs = output["prob"]
        total_loss += loss.item()
        valid_batches += 1
        y_true.extend(labels.tolist())
        y_pred.extend(torch.argmax(probs, dim=1).tolist())
        y_prob.extend(probs[:, 1].tolist())

    if len(y_true) == 0 or valid_batches == 0:
        empty_result = (float("nan"), {})
        if return_outputs:
            return empty_result + ({"y_true": [], "y_pred": [], "y_prob": []},)
        return empty_result

    metrics = calculate_metrics(y_true, y_pred, y_prob)
    result = (total_loss / valid_batches, metrics)
    if return_outputs:
        return result + ({"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob},)
    return result


def run_single_group(split_files, cfg, device, group_name, ratio_dir=None, split_mode=None, group_id=None):
    train_file, val_file, test_files = split_files
    if isinstance(test_files, (str, os.PathLike)):
        test_files = {os.path.splitext(os.path.basename(test_files))[0]: test_files}
    else:
        test_files = dict(test_files)

    print("\n" + "=" * 60)
    print("  Group: {}".format(group_name))
    print("  Train: {}".format(train_file))
    print("  Val:   {}".format(val_file))
    for test_name, test_file in test_files.items():
        print("  {}: {}".format(test_name, test_file))
    print("  Model: Mamba-BAN CE-only DTI")
    print("=" * 60)

    required_files = [train_file, val_file] + list(test_files.values())
    for file_path in required_files:
        if not os.path.exists(file_path):
            print("  File not found: {}".format(file_path))
            return None, 0

    loader_kwargs = {
        "esm2_dir": cfg["protein_feat_dir"],
        "drug_feat_dir": cfg["drug_feat_dir"],
        "batch_size": cfg["batch_size"],
        "num_workers": cfg["num_workers"],
        "seed": cfg["seed"],
        "pin_memory": device.type == "cuda",
        "max_drug_len": cfg["max_drug_len"],
        "max_prot_len": cfg["max_prot_len"],
        "preload": cfg["preload_features"],
        "prot_cache": {},
        "drug_cache": {},
    }
    train_loader = build_split_dataloader(
        train_file, shuffle=True, drop_last=True, **loader_kwargs
    )
    val_loader = build_split_dataloader(val_file, **loader_kwargs)
    test_loaders = {
        name: build_split_dataloader(path, **loader_kwargs)
        for name, path in test_files.items()
    }

    model = CNNAttentionDTI(
        drug_dim=cfg["drug_dim"],
        protein_dim=cfg["protein_dim"],
        cnn_dim=cfg["cnn_dim"],
        drug_kernel_sizes=cfg["drug_kernel_sizes"],
        protein_kernel_sizes=cfg["protein_kernel_sizes"],
        mamba_layers=cfg["mamba_layers"],
        dropout=cfg["dropout"],
        ban_glimpses=cfg["ban_glimpses"],
        ban_k=cfg["ban_k"],
    ).to(device)

    print("  Parameters: {:,}".format(sum(p.numel() for p in model.parameters())))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = torch.nn.CrossEntropyLoss().to(device)
    print("  Loss: CrossEntropyLoss")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    early_stopper = EarlyStopping(patience=cfg["patience"], delta=1e-4)
    best_score = -float("inf")
    best_epoch = 0
    best_model_state = None

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if not val_metrics:
            print("[WARNING] Empty validation metrics at epoch {}".format(epoch))
            continue

        val_roc = val_metrics["AUROC"]
        if not np.isfinite(val_roc):
            raise ValueError(
                "Validation AUROC is undefined at epoch {}. "
                "Ensure the validation split contains both classes after feature filtering.".format(
                    epoch
                )
            )
        elapsed = time.time() - t0

        print(
            "[Epoch {}/{} | {:.1f}s] Train Loss: {:.3f} | "
            "Valid Loss: {:.3f} | Acc: {:.3f} | Prec: {:.3f} | Recall: {:.3f} | "
            "F1: {:.3f} | MCC: {:.3f} | AUROC: {:.3f} | AUPR: {:.3f}".format(
                epoch,
                cfg["epochs"],
                elapsed,
                train_loss,
                valid_loss,
                val_metrics["Accuracy"],
                val_metrics["Precision"],
                val_metrics["Recall"],
                val_metrics["F1"],
                val_metrics["MCC"],
                val_metrics["AUROC"],
                val_metrics["AUPR"],
            )
        )

        if val_roc > best_score:
            best_epoch = epoch
            best_score = val_roc
            best_model_state = clone_state_dict(model)

        if epoch >= cfg.get("early_stop_start_epoch", 1):
            early_stopper.step(val_roc)

        if early_stopper.early_stop:
            print("Early stopping at epoch {}".format(epoch))
            break

    if best_model_state is None:
        raise RuntimeError("No best model was saved. Please check validation metrics.")

    model.load_state_dict(best_model_state)
    model.eval()

    # Save the best validation checkpoint for this group.
    ckpt_dir = os.path.join(cfg["model_dir"], cfg["data_name"])
    if ratio_dir is not None and split_mode is not None:
        ckpt_dir = os.path.join(ckpt_dir, ratio_dir, split_mode)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    ckpt_name = "best_model_group{}.pt".format(group_id) if group_id is not None else "best_model_{}.pt".format(group_name)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    torch.save({"model_state_dict": best_model_state,
                "best_epoch": best_epoch,
                "best_val_auroc": best_score,
                "config": cfg}, ckpt_path)

    print("Best model at epoch {} with AUROC = {:.3f}".format(best_epoch, best_score))
    test_results = {}
    combined_outputs = {"y_true": [], "y_pred": [], "y_prob": []}
    for test_name, test_loader in test_loaders.items():
        _, test_metrics, outputs = evaluate(
            model, test_loader, criterion, device, return_outputs=True
        )
        if not test_metrics:
            print("[WARNING] Empty metrics for {}".format(test_name))
            continue
        test_results[test_name] = test_metrics
        for output_name in combined_outputs:
            combined_outputs[output_name].extend(outputs[output_name])
        print(
            "[{}] | Acc: {:.3f} | Prec: {:.3f} | Recall: {:.3f} | F1: {:.3f} | MCC: {:.3f} | "
            "AUROC: {:.3f} | AUPR: {:.3f}".format(
                test_name,
                test_metrics["Accuracy"],
                test_metrics["Precision"],
                test_metrics["Recall"],
                test_metrics["F1"],
                test_metrics["MCC"],
                test_metrics["AUROC"],
                test_metrics["AUPR"],
            )
        )

    is_radial = "radial" in (split_mode or "").lower()
    if is_radial and {"test1", "test2"}.issubset(test_results):
        combined_name = "test1_plus_test2"
        combined_metrics = calculate_metrics(
            combined_outputs["y_true"],
            combined_outputs["y_pred"],
            combined_outputs["y_prob"],
        )
        test_results[combined_name] = combined_metrics
        print(
            "[{}] | Acc: {:.3f} | Prec: {:.3f} | Recall: {:.3f} | F1: {:.3f} | MCC: {:.3f} | "
            "AUROC: {:.3f} | AUPR: {:.3f}".format(
                combined_name,
                combined_metrics["Accuracy"],
                combined_metrics["Precision"],
                combined_metrics["Recall"],
                combined_metrics["F1"],
                combined_metrics["MCC"],
                combined_metrics["AUROC"],
                combined_metrics["AUPR"],
            )
        )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return test_results, best_epoch


def print_summary_table(all_results):
    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "MCC",
        "AUROC",
        "AUPR",
    ]
    meta_names = ["Ratio", "Split", "TestSet", "Group", "BestEpoch"]

    rows = []
    for group_name, metrics in all_results.items():
        row = {"GroupName": metrics.get("GroupName", group_name)}
        for name in meta_names:
            row[name] = metrics.get(name, "")
        for name in metric_names:
            row[name] = metrics.get(name, 0)
        rows.append(row)

    if not rows:
        raise RuntimeError("No groups were evaluated; no summary can be generated.")

    df = pd.DataFrame(rows)
    sections = []
    for test_name, test_df in df.groupby("TestSet", sort=False):
        mean_row = {
            "GroupName": "Mean\u00B1Std",
            "Ratio": test_df["Ratio"].iloc[0],
            "Split": test_df["Split"].iloc[0],
            "TestSet": test_name,
            "Group": "",
            "BestEpoch": "",
        }
        for name in metric_names:
            vals = pd.to_numeric(test_df[name], errors="coerce").values.astype(float)
            mean_row[name] = "{:.3f}\u00B1{:.3f}".format(np.nanmean(vals), np.nanstd(vals))
        sections.extend([test_df, pd.DataFrame([mean_row])])

    df_out = pd.concat(sections, ignore_index=True)

    print("\n" + "=" * 100)
    print("  FINAL RESULTS")
    print("=" * 100)
    print(df_out.to_string(index=False))
    print("=" * 100)
    return df_out


def save_results(all_results, cfg):
    summary_df = print_summary_table(all_results)
    result_dir = os.path.join(cfg["result_dir"], cfg["data_name"], cfg["ratio_dir"], cfg["split_mode"])
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, cfg["result_file"])
    summary_df.to_excel(result_path, index=False)
    print("\n  Results saved to: {}".format(result_path))
    return summary_df


# def standardize_df(df, cfg):
#     out = df.copy()
#     out["DRUGID"] = out['Drug'].astype(str).str.strip()
#     out["UNIPROTID"] = out['Protein_ID'].astype(str).str.strip()
#     out["Label"] = out['tpp_label'].astype(int)
#     return out


def build_split_jobs(cfg):
    split_jobs = []
    is_radial = "radial" in cfg["split_mode"].lower()
    for group_id in cfg["group_ids"]:
        group_dir = os.path.join(cfg["data_dir"], cfg["ratio_dir"], cfg["split_mode"], "group{}".format(group_id))
        train_file = find_split_file(group_dir, "train")
        val_file = find_split_file(group_dir, "val")
        if is_radial:
            test_files = {
                "test1": find_split_file(group_dir, "test1"),
                "test2": find_split_file(group_dir, "test2"),
            }
        else:
            test_files = {"test": find_split_file(group_dir, "test")}
        
        split_jobs.append(
            {
                "group_name": "{}_{}_group{}".format(cfg["ratio_dir"], cfg["split_mode"], group_id),
                "train_file": train_file,
                "val_file": val_file,
                "test_files": test_files,
                "ratio_dir": cfg["ratio_dir"],
                "split_mode": cfg["split_mode"],
                "group_id": group_id,
            }
        )
    return split_jobs


def find_split_file(group_dir, stem):
    for extension in ("xlsx", "xls", "csv"):
        path = os.path.join(group_dir, "{}.{}".format(stem, extension))
        if os.path.isfile(path):
            return path
    return os.path.join(group_dir, "{}.xlsx".format(stem))


def build_log_path(cfg, job):
    log_dir = os.path.join(cfg["log_dir"], cfg["data_name"], job["ratio_dir"], job["split_mode"])
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "group{}.log".format(job["group_id"]))


def print_run_header(cfg, device, log_path=None):
    print("\n" + "=" * 60)
    print("  Mamba-BAN DTI")
    print("=" * 60)
    print("  Device:       {}".format(device))
    print("  Batch size:   {}".format(cfg["batch_size"]))
    print("  LR:           {}".format(cfg["lr"]))
    print("  Data dir:     {}".format(cfg["data_dir"]))
    print("  Protein feat: {}".format(cfg["protein_feat_dir"]))
    print("  Drug feat:    {}".format(cfg["drug_feat_dir"]))
    print("  Drug kernel:  {}".format(cfg["drug_kernel_sizes"]))
    print("  Prot kernel:  {}".format(cfg["protein_kernel_sizes"]))
    print("  Loss:         CrossEntropyLoss")
    if log_path is not None:
        print("  Log file:     {}".format(log_path))


def main(config=None):
    """Run one dataset/split experiment and return its summary DataFrame."""
    cfg = merge_config(config)
    validate_config(cfg)

    os.makedirs(os.path.join(cfg["model_dir"], cfg["data_name"]), exist_ok=True)
    os.makedirs(os.path.join(cfg["result_dir"], cfg["data_name"]), exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)

    setup_seed(cfg["seed"])
    device = select_device(cfg["cuda_device"])

    original_stdout = sys.stdout
    print_run_header(cfg, device)
    print("  Selected ratio: {}".format(cfg["ratio_dir"]))
    print("  Selected split: {}".format(cfg["split_mode"]))

    all_results = {}
    total_start = time.time()

    for job in build_split_jobs(cfg):
        log_path = build_log_path(cfg, job)
        logger = Logger(log_path)
        sys.stdout = logger
        try:
            print_run_header(cfg, device, log_path=log_path)
            group_results, best_epoch = run_single_group(
                (job["train_file"], job["val_file"], job["test_files"]),
                cfg,
                device,
                job["group_name"],
                ratio_dir=job["ratio_dir"],
                split_mode=job["split_mode"],
                group_id=job["group_id"],
            )
            if group_results is not None:
                for test_name, test_metrics in group_results.items():
                    result_row = dict(test_metrics)
                    result_row["GroupName"] = job["group_name"]
                    result_row["Ratio"] = job["ratio_dir"]
                    result_row["Split"] = job["split_mode"]
                    result_row["TestSet"] = test_name
                    result_row["Group"] = job["group_id"]
                    result_row["BestEpoch"] = best_epoch
                    result_key = "{}::{}".format(job["group_name"], test_name)
                    all_results[result_key] = result_row
        finally:
            sys.stdout = original_stdout
            logger.close()

    print("\nTotal time: {:.1f} min".format((time.time() - total_start) / 60.0))
    if not all_results:
        print("No results were produced. Check split files and feature directories.")
        return None
    return save_results(all_results, cfg)


def parse_group_ids(value):
    """Parse values such as '1-5' or '1,3,5' into a list of integers."""
    result = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            step = 1 if end >= start else -1
            result.extend(range(start, end + step, step))
        else:
            result.append(int(part))
    if not result:
        raise argparse.ArgumentTypeError("group ids cannot be empty")
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train BioCMB-DTI on one or more data split modes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="JSON file containing config overrides")
    parser.add_argument(
        "--split-mode",
        action="append",
        dest="split_modes",
        help="split directory name; repeat this option to run several modes",
    )
    parser.add_argument("--data-name")
    parser.add_argument("--data-type")
    parser.add_argument("--ratio-dir")
    parser.add_argument("--group-ids", type=parse_group_ids, help="for example: 1-5 or 1,3,5")
    parser.add_argument("--base-db-dir")
    parser.add_argument("--protein-feat-dir")
    parser.add_argument("--drug-feat-dir")
    parser.add_argument("--drug-feat-root")
    parser.add_argument("--output-root")
    parser.add_argument("--result-file")
    parser.add_argument("--cuda-device", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--early-stop-start-epoch", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--seed", type=int)
    return parser


def cli(argv=None):
    args = build_arg_parser().parse_args(argv)
    overrides = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as config_file:
            overrides.update(json.load(config_file))

    split_modes = args.split_modes
    for key, value in vars(args).items():
        if key not in {"config", "split_modes"} and value is not None:
            overrides[key] = value

    if split_modes is None:
        split_modes = overrides.pop("split_modes", None)
    if split_modes is None:
        split_modes = [overrides.get("split_mode", CONFIG["split_mode"])]
    elif isinstance(split_modes, str):
        split_modes = [split_modes]

    summaries = {}
    for split_mode in split_modes:
        experiment_config = dict(overrides)
        experiment_config["split_mode"] = split_mode
        summaries[split_mode] = main(experiment_config)
    return summaries


if __name__ == "__main__":
    cli()
