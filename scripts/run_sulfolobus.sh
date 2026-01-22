#!/bin/bash
# Run full solenoid detection pipeline on Sulfolobus acidocaldarius proteome
#
# This script:
# 1. Downloads the proteome from UniProt (reference proteome UP000001018 = single genome assembly)
# 2. Filters to proteins with AlphaFold structures
# 3. Runs ESM++ to generate APC attention matrices
# 4. Generates viewer data (results.json + APC images)

set -e  # Exit on error

ORGANISM="sulfolobus_acidocaldarius"

echo "=== Step 1: Download proteome from UniProt ==="
echo "Using reference proteome UP000001018 (single genome: DSM 639 / ATCC 33909)"
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
