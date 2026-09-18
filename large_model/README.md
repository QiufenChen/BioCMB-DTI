# Pretrained Large Models

This directory is used to store the pretrained large language models required by **BioCMB-DTI**.

Due to the large size of the model parameters, the pretrained weights are **not included in this GitHub repository**. Please download the complete model repositories from Hugging Face before running the corresponding feature extraction scripts.

The expected directory structure is:

```text
large_model/
├── Ankh_large/
└── MolFormer/
```

## 1. Ankh-Large

**Ankh-Large** is a pretrained protein language model designed to learn representations from protein amino acid sequences. In BioCMB-DTI, it is used to extract protein sequence representations.

Hugging Face:

https://huggingface.co/ElnaggarLab/ankh-large

Please download the complete model repository, including the model weights, configuration files, and tokenizer files, and place them under:

```text
large_model/Ankh_large/
```

The directory should look approximately like:

```text
Ankh_large/
├── config.json
├── model.safetensors / pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── ...
```

## 2. MolFormer

**MoLFormer** is a pretrained molecular language model that learns molecular representations from SMILES strings. In BioCMB-DTI, it is used to extract representations of drug molecules.

The model used in this project is **MoLFormer-XL-both-10pct**.

Hugging Face:

https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct

Please download the complete model repository and place it under:

```text
large_model/MolFormer/
```

The directory should look approximately like:

```text
MolFormer/
├── config.json
├── model.safetensors
├── tokenizer.json
├── configuration_molformer.py
├── modeling_molformer.py
└── ...
```

## Download with Hugging Face CLI

The models can also be downloaded directly using the Hugging Face CLI.

First install `huggingface_hub`:

```bash
pip install -U huggingface_hub
```

Then download the models:

### Ankh-Large

```bash
hf download ElnaggarLab/ankh-large \
    --local-dir ./Ankh_large
```

### MolFormer

```bash
hf download ibm-research/MoLFormer-XL-both-10pct \
    --local-dir ./MolFormer
```

After downloading, the final directory structure should be:

```text
large_model/
├── README.md
├── Ankh_large/
│   ├── config.json
│   ├── model.safetensors / pytorch_model.bin
│   ├── tokenizer.json
│   └── ...
└── MolFormer/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── configuration_molformer.py
    ├── modeling_molformer.py
    └── ...
```

> **Note:** Please keep the directory names `Ankh_large` and `MolFormer` unchanged, as the feature extraction scripts may use these paths to load the pretrained models.
