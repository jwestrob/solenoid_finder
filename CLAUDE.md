# Claude Session Context - Solenoid Detector

## Project Overview

Detect solenoid/repeat proteins (ANK, LRR, TPR, WD, HEAT, etc.) from ESM++ attention maps using diagonal band periodicity analysis.

## Project Structure

```
Solenoid_detector/
├── cache/
│   ├── apc/              # Global APC matrices (expensive, shared across runs)
│   └── structures/       # Global PDB files from ESM3 folding
├── runs/
│   ├── index.json        # Registry of all analysis runs
│   └── {run_name}/
│       └── results.json  # Detection results for this run
├── viewer/
│   ├── runs -> ../runs           # Symlink for web access
│   ├── structures -> ../cache/structures
│   ├── apc_images/       # Pre-rendered APC heatmaps
│   ├── index.html
│   └── app.js
├── run_pipeline.py       # Main entry point
├── detect_solenoids.py   # Core detection algorithm
├── generate_viewer_data.py
└── run_esmpp_proteome.py
```

## Key Files

- `run_pipeline.py` - **Main entry point** - runs full pipeline with `--name` for run management
- `detect_solenoids.py` - Core detection algorithm (harmonic counting, sliding window, boundary refinement)
- `generate_viewer_data.py` - Generates results.json with optional ESM3 folding
- `run_esmpp_proteome.py` - Runs ESM++ on a proteome to generate APC matrices
- `viewer/` - Web app with run selector dropdown to switch between analyses
- `SOLENOID.md` - Detailed technical documentation of the algorithm

## Quick Commands

```bash
# Run full pipeline (ESM++ → detect → results)
python run_pipeline.py --fasta proteins/my_proteins.fasta --name my_analysis --device mps

# With ESM3 structure prediction for high-confidence hits
python run_pipeline.py --fasta proteins/Borg.faa --name borg_jan22 --fold --min-votes 4

# Resume interrupted run
python run_pipeline.py --fasta proteins/my_proteins.fasta --name my_analysis --resume

# Re-run detection only (skip ESM++ inference)
python run_pipeline.py --fasta proteins/my_proteins.fasta --name redetect --skip-esmpp

# Start viewer (has dropdown to select runs)
cd viewer && python -m http.server 8080

# Sync results from remote server to local
rsync -av server:solenoid/runs/ viewer/runs/
rsync -av server:solenoid/cache/structures/ viewer/structures/

# Check detection on a single protein
python -c "
import numpy as np
from detect_solenoids import detect_and_score_solenoid_regions, compute_soft_votes
apc = np.load('cache/apc/PROTEIN_ID.npy')
regions = detect_and_score_solenoid_regions(apc)
for r in regions:
    votes = compute_soft_votes(r)
    print(f\"{r['start']}-{r['end']}: n_harm={r['n_harmonics']}, votes={votes['total_votes']}/5\")
"
```

## Algorithm Summary

1. **Log Transform**: Apply `log1p(apc * 1000)` for consistent detection across proteins
2. **Harmonic Detection**: For each candidate period P, check for peaks at P, 2P, 3P... Count harmonics with z-score > 1.5
3. **Sliding Window**: Scan protein with 150-residue windows, step=25. Flag windows with n_harmonics ≥ 3
4. **Boundary Refinement**:
   - Use 10th percentile baseline (robust to solenoid signal contamination)
   - Contract to strong signal, then expand with gap bridging (up to 1 period)
5. **Soft Voting**: 5 metrics vote pass/fail:
   - H: n_harmonics ≥ 4
   - C: continuity ≥ 0.10
   - F: fft_prominence ≥ 0.05
   - B: band_score ≥ 0.05
   - R: longest_run_frac ≥ 0.20
6. **Rescue Rules**: Accept n_harm=3 if:
   - continuity ≥ 0.15, OR
   - (run ≥ 0.40 AND continuity ≥ 0.10)

## Viewer Features

- **Run selector dropdown** in header to switch between analyses
- Shows run metadata (date, protein count, solenoid count, high-conf count)
- Loads structures from local cache/structures/ first, falls back to AlphaFold API
- Panel header shows structure source (ESM3 vs AlphaFold)
- Filtering with rescue rules matching detection algorithm
- Backwards compatible with legacy `viewer/data/results.json`

## Remote Server Workflow

1. Run pipeline on server:
   ```bash
   python run_pipeline.py --fasta data/proteome.fasta --name organism_jan22 --fold --device cuda
   ```

2. Sync results to local machine:
   ```bash
   rsync -av server:solenoid/runs/ viewer/runs/
   rsync -av server:solenoid/cache/structures/ viewer/structures/
   ```

3. View locally:
   ```bash
   cd viewer && python -m http.server 8080
   ```

## File Locations

- APC matrices: `cache/apc/*.npy` (global, shared)
- Structures: `cache/structures/*.pdb` (global, shared)
- Run results: `runs/{run_name}/results.json`
- Runs index: `runs/index.json`
- APC images: `viewer/apc_images/*.png`
- Input FASTAs: `data/` or `proteins/`
- ESM3 API key: `.esm_api_key` (gitignored)

## Next Steps / Ideas

1. **Improve period detection** - Some solenoids get wrong period (finds local helix ~20 instead of solenoid ~38)
2. **Beta propeller handling** - Thatching pattern confuses harmonic detection
3. **Training data generation** - For ML-based scoring
4. **Multi-GPU support** - For faster ESM++ inference
