#!/bin/bash
# Run full solenoid detection pipeline on Haloferax volcanii proteome
#
# This script:
# 1. Downloads the proteome from UniProt (reference proteome UP000008243 = single genome assembly)
# 2. Filters to proteins with AlphaFold structures
# 3. Runs ESM++ to generate APC attention matrices
# 4. Generates viewer data (results.json + APC images)
#
# Haloferax volcanii is a halophilic (salt-loving) archaeon from the Dead Sea.
# Larger proteome (~4,000 proteins) than Sulfolobus.

set -e  # Exit on error

ORGANISM="haloferax_volcanii"

echo "=== Step 1: Download proteome from UniProt ==="
echo "Using reference proteome UP000008243 (Haloferax volcanii DS2)"
python download_archaeon_proteome.py "$ORGANISM"

echo ""
echo "=== Step 2: Run ESM++ to generate APC matrices ==="
python run_esmpp_proteome.py \
    --fasta "data/${ORGANISM}/${ORGANISM}_alphafold.fasta" \
    --device mps \
    --dtype fp32 \
    --max-length 1024 \
    --resume

echo ""
echo "=== Step 3: Generate viewer data ==="
python generate_viewer_data.py

echo ""
echo "=== Done! ==="
echo "Start the viewer with: cd viewer && python -m http.server 8080"
