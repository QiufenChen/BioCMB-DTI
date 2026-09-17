#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python run_random_split.py
python run_cold_prot_split.py
python run_cold_drug_split.py
python run_drug_radial_split.py
python run_prot_radial_split.py

echo "All training finished!"
