# Solenoid Detector

Detect solenoid/repeat proteins (ANK, LRR, TPR, WD, HEAT, etc.) from ESM++ attention maps using diagonal band periodicity analysis.

## Overview

This tool analyzes protein language model attention patterns to identify tandem repeat proteins. It works by:

1. **Running ESM++** on protein sequences to generate attention-based pairwise contact (APC) matrices
2. **Detecting periodic diagonal bands** in APC matrices using harmonic analysis
3. **Scoring and ranking** candidates using multiple metrics with a soft voting system
4. **Visualizing results** in an interactive web viewer

## Quick Start

```bash
# 1. Install dependencies
pip install torch transformers numpy scipy tqdm matplotlib

# 2. Download a proteome (or provide your own FASTA)
python download_archaeon_proteome.py --organism "Sulfolobus acidocaldarius" --output data/my_proteome

# 3. Run ESM++ to generate APC matrices
python run_esmpp_proteome.py --fasta data/my_proteome/proteome.fasta --device cuda --dtype fp16

# 4. Generate viewer data
python generate_viewer_data.py

# 5. Start the viewer
cd viewer && python -m http.server 8080
# Open http://localhost:8080
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+ (with CUDA for GPU acceleration)
- transformers (Hugging Face)
- numpy, scipy, matplotlib, tqdm

### GPU Support

```bash
# For CUDA (recommended)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For Apple Silicon (MPS)
pip install torch  # MPS support is built-in

# CPU only (slow, not recommended)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Usage

### Step 1: Prepare Input Data

You need a FASTA file with protein sequences. You can:

**Option A: Download from UniProt**
```bash
python download_archaeon_proteome.py \
    --organism "Escherichia coli" \
    --output data/ecoli \
    --alphafold-only  # Only include proteins with AlphaFold structures
```

**Option B: Use your own FASTA**
```bash
# Just place your FASTA file in data/your_project/sequences.fasta
```

### Step 2: Generate APC Matrices

Run ESM++ on your sequences to generate attention-based contact matrices:

```bash
python run_esmpp_proteome.py \
    --fasta data/your_project/sequences.fasta \
    --device cuda \           # or 'mps' for Apple Silicon, 'cpu' for CPU
    --dtype fp16 \            # fp16 recommended for CUDA, fp32 for mps
    --max-length 1024 \       # Skip proteins longer than this
    --resume                  # Skip already-processed proteins
```

**Device/dtype recommendations:**
| Device | Recommended dtype | Notes |
|--------|------------------|-------|
| CUDA | fp16 or bf16 | 2-3x faster, half memory |
| MPS | fp32 | fp16 can be unstable on MPS |
| CPU | fp32 | Slow, only for testing |

**Memory requirements:**
- ~5GB VRAM for 1024-residue proteins (fp16)
- ~10GB VRAM for 1024-residue proteins (fp32)

Output: `cache/apc_matrices/{protein_id}.npy` files

### Step 3: Detect Solenoids and Generate Viewer Data

```bash
python generate_viewer_data.py
```

This runs the detection algorithm on all cached APC matrices and generates:
- `viewer/data/results.json` - Detection results for all proteins
- `viewer/apc_images/*.png` - Visualization images for detected solenoids

### Step 4: View Results

```bash
cd viewer && python -m http.server 8080
```

Open http://localhost:8080 in your browser.

**Viewer features:**
- Filter by n_harmonics, continuity, repeats
- Sort by votes, continuity, band_score, etc.
- View APC attention maps with detected regions highlighted
- View 3D structures (local ESMFold or AlphaFold from EBI)
- Detailed metrics for each detected region

### Step 5 (Optional): Predict Structures with ESMFold

For proteins without AlphaFold coverage, you can predict structures using ESMFold:

```bash
# Using ESMFold API (no GPU required, but rate-limited)
python run_esmfold.py --mode api --min-votes 4

# Using local ESMFold model (requires GPU with ~16GB VRAM)
python run_esmfold.py --mode local --device cuda --min-votes 4

# Predict for specific proteins
python run_esmfold.py --proteins D4GQW4 D4GPD9 Q4J907
```

**Options:**
| Flag | Description |
|------|-------------|
| `--mode api\|local` | Use ESMFold API or local model |
| `--min-votes N` | Only predict for proteins with N+ votes |
| `--max-length N` | Skip proteins longer than N residues |
| `--proteins ID1 ID2` | Predict specific proteins |
| `--resume` | Skip already-predicted structures |

Structures are saved to `viewer/structures/` and automatically loaded by the viewer (prioritized over AlphaFold).

## Detection Algorithm

### Core Concept

True solenoid proteins have repeating structural units that create **periodic diagonal bands** in attention maps. We detect these by counting **harmonics** - if a protein has period P, we expect peaks at distances P, 2P, 3P, etc.

### Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| **n_harmonics** | Count of harmonic peaks with z-score > 1.5 | ≥ 4 |
| **continuity** | Fraction of diagonal positions with elevated signal | ≥ 0.15 for rescue |
| **longest_run_frac** | Longest continuous run of elevated signal | ≥ 0.40 for rescue |
| **fft_prominence** | FFT peak prominence at detected period | ≥ 0.05 |
| **band_score** | Combined band quality metric | ≥ 0.05 |

### Soft Voting System

Each region gets up to 5 votes based on passing metric thresholds:
- **H** (Harmonics): n_harmonics ≥ 4
- **C** (Continuity): continuity ≥ 0.10
- **F** (FFT): fft_prominence ≥ 0.05
- **B** (Band): band_score ≥ 0.05
- **R** (Run): longest_run_frac ≥ 0.20

High-confidence detections have 4-5 votes.

### Rescue Rules

Some true solenoids have n_harmonics = 3 due to period detection issues. These are "rescued" if:
- `n_harmonics ≥ 3 AND continuity ≥ 0.15`, OR
- `n_harmonics ≥ 3 AND longest_run_frac ≥ 0.40 AND continuity ≥ 0.10`

## File Structure

```
solenoid_detector/
├── detect_solenoids.py          # Core detection algorithm
├── generate_viewer_data.py      # Generate viewer JSON and images
├── run_esmpp_proteome.py        # ESM++ inference script
├── run_esmfold.py               # ESMFold structure prediction
├── download_archaeon_proteome.py # UniProt proteome downloader
├── run_sulfolobus.sh            # Example pipeline script
├── run_haloferax.sh             # Example pipeline script
├── SOLENOID.md                  # Detailed algorithm documentation
├── CLAUDE.md                    # Project context for Claude Code
├── viewer/
│   ├── index.html               # Viewer HTML
│   ├── app.js                   # Viewer JavaScript
│   ├── structures/              # [gitignored] Local ESMFold structures
│   │   └── *.pdb
│   ├── data/                    # [gitignored] Generated results
│   │   └── results.json
│   └── apc_images/              # [gitignored] Generated images
├── cache/                       # [gitignored] Generated APC matrices
│   └── apc_matrices/*.npy
└── data/                        # [gitignored] Input proteome data
    └── {organism}/*.fasta
    └── apc_images/              # [gitignored] Generated images
```

## API Reference

### detect_solenoids.py

```python
from detect_solenoids import detect_and_score_solenoid_regions, compute_soft_votes

# Load APC matrix
import numpy as np
apc = np.load('cache/apc_matrices/PROTEIN_ID.npy')

# Detect solenoid regions
regions = detect_and_score_solenoid_regions(apc)

# Each region contains:
# {
#     'start': int,           # Start residue (0-indexed)
#     'end': int,             # End residue (exclusive)
#     'period': int,          # Detected repeat period
#     'n_harmonics': int,     # Number of harmonic peaks
#     'continuity': float,    # Diagonal continuity score
#     'longest_run_frac': float,  # Longest continuous run
#     'fft_prominence': float,    # FFT peak prominence
#     'band_score': float,    # Combined band quality
#     ...
# }

# Compute soft votes
for r in regions:
    votes = compute_soft_votes(r)
    print(f"Votes: {votes['total_votes']}/5")
```

### Key Functions

| Function | Description |
|----------|-------------|
| `detect_and_score_solenoid_regions(apc)` | Main entry point - detects and scores all solenoid regions |
| `sliding_window_detection(apc)` | Scan with overlapping windows to find candidate regions |
| `score_solenoid(apc, start, end)` | Score a specific region for solenoid characteristics |
| `compute_soft_votes(region)` | Compute 5-metric soft voting score |
| `log_transform_apc(apc)` | Apply log transform for consistent detection |

## Troubleshooting

### "CUDA out of memory"
- Reduce `--max-length` (try 512)
- Use `--dtype fp16` instead of fp32
- Process fewer proteins at once

### "MPS backend error" (Apple Silicon)
- Use `--dtype fp32` (fp16 can be unstable on MPS)
- Update to latest PyTorch

### Viewer shows "Loading..." forever
- Check browser console for errors
- Ensure `viewer/data/results.json` exists
- Try hard refresh (Cmd+Shift+R)

### False positives / False negatives
- Increase `min_harmonics` filter for fewer false positives
- Decrease filters and inspect near-misses for false negatives
- Check `continuity` - high continuity with low n_harm suggests period detection issue

## Performance

Typical processing times (RTX 3090, fp16):
- ~0.5 seconds per protein (avg length ~400)
- ~2,000 proteins/hour
- Full bacterial proteome (~4,000 proteins): ~2 hours

## Citation

If you use this tool, please cite:
- ESM++: [Synthyra/ESMplusplus_large](https://huggingface.co/Synthyra/ESMplusplus_large)
- This repository: [TODO: Add citation]

## License

[TODO: Add license]

## Contributing

Contributions welcome! Key areas for improvement:
- Beta propeller detection (thatching patterns)
- Multi-GPU support
- Training data generation for ML-based scoring
- Integration with structure prediction confidence scores
