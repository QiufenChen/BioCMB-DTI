# BioCMB-DTI

BioCMB-DTI is a binary drug–target interaction (DTI) prediction framework built on precomputed drug and protein language-model embeddings. It combines multi-scale CNN and Mamba encoders with a Bilinear Attention Network (BAN) to model drug–protein interactions.

The repository supports random, cold-drug, cold-protein, and distance-based radial data splits. Each experiment can run over multiple groups, select the best checkpoint by validation AUROC, and export a complete metric summary.

## Highlights

- Multi-scale CNN and Mamba branches for both drug and protein embeddings
- Learnable branch fusion followed by bilinear drug–protein attention
- Cross-entropy training with AdamW and cosine learning-rate decay
- Random, cold-drug, cold-protein, drug-radial, and protein-radial evaluation
- Automatic evaluation of `test1`, `test2`, and their combined samples for radial splits
- Checkpoint, log, and Excel summary generation for every experiment
- Support for `.xlsx`, `.xls`, and `.csv` split files

## Model Overview

```text
Drug embeddings ─── Multi-scale CNN + Mamba ──┐
                                               ├── BAN interaction ── Classifier ── DTI probability
Protein embeddings ─ Multi-scale CNN + Mamba ─┘
```

The default configuration expects:

- MolFormer drug embeddings with dimension 768
- Ankh protein embeddings with dimension 1536
- A hidden dimension of 256
- Binary labels (`0` for negative and `1` for positive)

## Repository Structure

```text
BioCMB-DTI/
├── analysis_data/       # Dataset and feature analysis scripts
├── analysis_result/     # Generated analysis outputs
├── database/            # Dataset tables and pre-generated splits
├── get_feature/         # Drug and protein feature extraction scripts
├── split_data/          # Random, cold, and radial split generation
├── scr/                 # Model, data loading, and training code
│   ├── ban.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── utils.py
│   ├── run_random_split.py
│   ├── run_cold_drug_split.py
│   ├── run_cold_prot_split.py
│   ├── run_drug_radial_split.py
│   ├── run_prot_radial_split.py
│   └── run_all.sh
└── README.md
```

> The directory is named `scr` in this repository.

## Installation

Python 3.9 or later is recommended. A CUDA-enabled Linux environment is recommended for training.

Create and activate an environment, then install the main training dependencies:

```bash
pip install torch numpy pandas scikit-learn openpyxl tqdm prefetch-generator mamba-ssm
```

Additional packages are required only for feature extraction and radial split generation:

```bash
pip install transformers rdkit matplotlib umap-learn fair-esm
```

`mamba-ssm` must be compatible with the installed PyTorch and CUDA versions. If installation fails, use a matching prebuilt wheel or follow the official source-build instructions for your environment.

## Input Data

Each train, validation, or test table must contain these columns:

| Column | Description | Example |
|---|---|---|
| `DRUGID` | Drug feature identifier | `DB00001` |
| `UNIPROTID` | Protein feature identifier | `P12345` |
| `Label` | Binary interaction label | `0` or `1` |

Rows with a missing drug or protein feature file are filtered before a dataset is constructed. Training fails with a clear error if a split becomes empty after filtering.

### Standard splits

Random and cold splits use one test file per group:

```text
database/
└── SNAP/
    └── final_data/
        └── 1_1/
            └── random_split/
                ├── group1/
                │   ├── train.xlsx
                │   ├── val.xlsx
                │   └── test.xlsx
                ├── group2/
                └── ...
```

The same layout is used for `cold_drug_split` and `cold_prot_split`.

### Radial splits

When `split_mode` contains the word `radial`, every group must contain two test files:

```text
database/SNAP/final_data/1_1/drug_morgan_distance_radial_split/
└── group1/
    ├── train.xlsx
    ├── val.xlsx
    ├── test1.xlsx
    └── test2.xlsx
```

The model is trained once and evaluated on:

1. `test1`
2. `test2`
3. `test1_plus_test2`

The combined result is calculated by concatenating the sample-level labels, predictions, and positive-class probabilities from both test sets and recomputing every metric. It is not an average of the two metric dictionaries.

## Precomputed Features

Feature files are resolved from the identifiers in each split table:

```text
drug_feature_directory/
├── DB00001.pt
└── DB00002.pt

protein_feature_directory/
├── P12345.npy
└── Q67890.npy
```

Requirements:

- Drug files must be named `{DRUGID}.pt`.
- Protein files must be named `{UNIPROTID}.npy`.
- Each feature must be a two-dimensional `[sequence_length, feature_dimension]` array or tensor.
- Feature dimensions must match `drug_dim` and `protein_dim` in the training configuration.

The default local feature locations are:

```text
BioCMB-DTI/SNAP/MolFormer/
BioCMB-DTI/UniprotKB_Ankh/
```

They can be overridden with command-line arguments or environment variables:

```bash
export BIOCMB_DRUG_FEAT_ROOT=/path/to/drug/feature/root
export BIOCMB_PROTEIN_FEAT_DIR=/path/to/UniprotKB_Ankh
```

With `BIOCMB_DRUG_FEAT_ROOT=/features`, the default SNAP drug directory becomes `/features/SNAP/MolFormer`.

Feature extraction utilities for MolFormer, Ankh, ESM, ProtBERT, ProtT5, ChemBERTa, and Uni-Mol are available under `get_feature/`. These scripts contain model and data paths that should be updated for the target environment before use.

## Training

### Preset experiment scripts

The preset scripts use the SNAP dataset, groups 1–5, and GPU 0:

```bash
python scr/run_random_split.py
python scr/run_cold_drug_split.py
python scr/run_cold_prot_split.py
python scr/run_drug_radial_split.py
python scr/run_prot_radial_split.py
```

Activate the desired Python environment before running the scripts.

On Linux, all preset experiments can be run sequentially with:

```bash
bash scr/run_all.sh
```

The batch script stops immediately if any experiment fails.

### Unified command-line interface

Run a single split:

```bash
python scr/train.py \
  --data-name SNAP \
  --data-type final_data \
  --base-db-dir ./database \
  --protein-feat-dir /path/to/UniprotKB_Ankh \
  --drug-feat-dir /path/to/SNAP/MolFormer \
  --output-root ./outputs \
  --split-mode random_split \
  --ratio-dir 1_1 \
  --group-ids 1-5 \
  --cuda-device 0
```

Run several split modes for the same dataset:

```bash
python scr/train.py \
  --data-name SNAP \
  --protein-feat-dir /path/to/UniprotKB_Ankh \
  --drug-feat-dir /path/to/SNAP/MolFormer \
  --split-mode random_split \
  --split-mode cold_drug_split \
  --split-mode cold_prot_split \
  --group-ids 1-5
```

`--group-ids` accepts ranges and comma-separated values:

```text
--group-ids 1-5
--group-ids 1,3,5
--group-ids 1-3,5
```

List all command-line options:

```bash
python scr/train.py --help
```

### JSON configuration

Configuration overrides may also be stored in JSON:

```json
{
  "data_name": "SNAP",
  "data_type": "final_data",
  "base_db_dir": "./database",
  "protein_feat_dir": "/path/to/UniprotKB_Ankh",
  "drug_feat_dir": "/path/to/SNAP/MolFormer",
  "output_root": "./outputs",
  "split_modes": [
    "random_split",
    "cold_drug_split",
    "cold_prot_split"
  ],
  "group_ids": [1, 2, 3, 4, 5],
  "cuda_device": 0,
  "batch_size": 32,
  "lr": 0.00005,
  "epochs": 100,
  "num_workers": 0,
  "preload_features": true
}
```

Run it with:

```bash
python scr/train.py --config config.json
```

Explicit command-line arguments override matching JSON values.

## Main Training Parameters

| Parameter | Default | Description |
|---|---:|---|
| `batch_size` | 32 | Training batch size; must be at least 2 because BAN uses BatchNorm |
| `lr` | `5e-5` | AdamW learning rate |
| `weight_decay` | `1e-4` | AdamW weight decay |
| `epochs` | 100 | Maximum number of training epochs |
| `patience` | 10 | Early-stopping patience |
| `early_stop_start_epoch` | 6 | First epoch included in early-stopping counting |
| `cnn_dim` | 256 | Shared encoder hidden dimension |
| `mamba_layers` | 2 | Number of Mamba layers per branch |
| `dropout` | 0.3 | Dropout probability |
| `max_drug_len` | 150 | Maximum drug embedding sequence length |
| `max_prot_len` | 2000 | Maximum protein embedding sequence length |
| `num_workers` | 0 | DataLoader worker count; zero avoids duplicating large feature caches |
| `preload_features` | `true` | Load features into shared in-process caches before training |

Training uses `torch.nn.CrossEntropyLoss`, AdamW, and `CosineAnnealingLR`. The best checkpoint and early stopping are based on validation AUROC. The validation split must contain both classes after feature filtering.

## Outputs

The default output layout is:

```text
outputs/
├── checkpoints/
│   └── {data_name}/{ratio_dir}/{split_mode}/best_model_group1.pt
├── logs/
│   └── {data_name}/{ratio_dir}/{split_mode}/group1.log
└── results/
    └── {data_name}/{ratio_dir}/{split_mode}/summary.xlsx
```

Each checkpoint contains:

- `model_state_dict`
- `best_epoch`
- `best_val_auroc`
- The complete resolved configuration

The result workbook reports:

- Accuracy
- Precision
- Recall
- F1
- Matthews correlation coefficient (MCC)
- AUROC
- Area under the precision–recall curve (AUPR)

Each test set receives an independent `Mean±Std` row across groups. Radial summaries contain separate sections for `test1`, `test2`, and `test1_plus_test2`.

## Split Generation

The `split_data/` directory contains:

- `random_and_cold_split.py` for random, cold-drug, and cold-protein splits
- `build_drug_radial_fixed_distance_split.py` for drug-based radial splits
- `build_protein_radial_fixed_distance_split.py` for protein-based radial splits

Review the dataset paths and split parameters in each script before running it.

## Troubleshooting

### Empty split after feature filtering

Check that:

- `DRUGID` values match `.pt` filenames.
- `UNIPROTID` values match `.npy` filenames.
- `drug_feat_dir` and `protein_feat_dir` point to the correct locations.

### Feature dimension mismatch

The final feature dimensions must match the configured `drug_dim` and `protein_dim`. Update those values when using a different embedding model.

### Undefined validation AUROC

Validation AUROC requires both positive and negative samples. Check the original validation split and confirm that feature filtering did not remove an entire class.

### Invalid CUDA device index

Set `--cuda-device` to an index smaller than `torch.cuda.device_count()`. If CUDA is unavailable, the training script automatically uses the CPU.

### Out-of-memory errors

Reduce `batch_size`, `max_drug_len`, or `max_prot_len`. Keeping `num_workers=0` avoids copying the preloaded feature cache into worker processes, especially on platforms that use process spawning.

## License and Citation

Add the appropriate license and citation information before publishing or redistributing the project.
