# Claude Session Context - Solenoid Detector

## Project Overview

This project detects solenoid/repeat proteins (ANK, LRR, TPR, WD, HEAT, etc.) from ESM++ attention maps using diagonal band periodicity analysis.

## Key Files

- `detect_solenoids.py` - Core detection algorithm (harmonic counting, sliding window, boundary refinement)
- `generate_viewer_data.py` - Generates results.json and APC images for viewer
- `run_esmpp_proteome.py` - Runs ESM++ on a proteome to generate APC matrices
- `download_archaeon_proteome.py` - Downloads proteomes from UniProt with AlphaFold filtering
- `viewer/` - Web app to visualize results (index.html, app.js)
- `SOLENOID.md` - Detailed technical documentation of the algorithm

## Current Data

### Sulfolobus acidocaldarius (Archaeon) - CURRENT
- **Location:** `data/sulfolobus_acidocaldarius/`
- **Proteins:** 2,209 processed (≤1024 residues)
- **Detected solenoids:** 376 (17%)
- **Cache:** `cache/apc_matrices/*.npy`
- **Genome:** Single genome assembly CP000077, strain DSM 639

### Previous: Human test set (301 proteins)
- Was symlinked, now replaced with Sulfolobus data

## How to Inspect Results

### Start the viewer:
```bash
cd viewer && python -m http.server 8080
# Open http://localhost:8080
```

### Check specific protein detection:
```python
import numpy as np
from detect_solenoids import detect_and_score_solenoid_regions

apc = np.load('cache/apc_matrices/PROTEIN_ID.npy')
regions = detect_and_score_solenoid_regions(apc)
print(regions)
```

### View results summary:
```python
import json
with open('viewer/data/results.json') as f:
    data = json.load(f)
print(f"Total: {len(data)}")
print(f"With solenoid: {sum(1 for v in data.values() if v['has_solenoid'])}")
```

## Algorithm Summary

1. **Harmonic Detection**: For each candidate period P, check for peaks at P, 2P, 3P... in diagonal profile. Count harmonics with z-score > 1.5.

2. **Sliding Window**: Scan protein with 150-residue windows, step=25. Flag windows with n_harmonics ≥ 3.

3. **Boundary Refinement** (contract-then-expand):
   - CONTRACT: Find where diagonal signal is strong (30% threshold) within rough region
   - EXPAND: Extend outward while signal remains elevated (15% above baseline)

4. **Merge**: Combine overlapping regions with similar periods (±8 residues)

## Known Issues (2026-01-21)

### Sulfolobus results "a lot worse" than human
User reported poor results on Sulfolobus compared to human proteins. Need to investigate:
- Are we detecting too many false positives?
- Are boundaries wrong?
- Are we missing real solenoids?
- Is the 17% detection rate reasonable for Archaea?

### Beta propellers still challenging
- Thatching pattern (non-adjacent blade contacts) confuses harmonic detection
- Start boundaries often detected late (signal weaker at edges)
- See O43818 as example in human data

## Current Results (Sulfolobus acidocaldarius)

**Summary:** 376/2209 proteins (17.0%) detected with solenoids

### Top Candidates (inspect these first):
| Protein | Length | n_harm | Period | UniProt Annotation |
|---------|--------|--------|--------|---------------------|
| P38619 | 223 | 8 | 22 | eIF-6 (NOT annotated as repeat!) - VERIFY |
| Q4JAK7 | 228 | 7 | 26 | ? |
| Q4JCI1 | 337 | 7 | 26 | ? |
| Q4J7E9 | 410 | 6 | 23 | ? |
| Q4J8S5 | 675 | 6 | 35 | ? |
| Q4J908 | 479 | 6 | 29 | ? |

**WARNING:** P38619 (our top hit with n_harm=8) is annotated as Translation initiation factor 6 with NO repeat annotations in UniProt. Either:
1. False positive (algorithm issue)
2. Unannotated internal repeats (discovery)
3. eIF-6 fold has pseudo-periodic structure

**Action:** Visually inspect P38619 APC map and AlphaFold structure to determine if detection is valid.

### Period Distribution (376 solenoids):
- Period 22-30: Most common (typical LRR/ANK range)
- Period 35+: Larger repeats (TPR-like or beta propellers?)

## Next Steps

1. **Inspect Sulfolobus results** - Look at specific examples in viewer
2. **Compare to known Archaeal repeats** - Check if known repeat proteins are detected
3. **Tune parameters** - May need different thresholds for Archaea
4. **Investigate false positives** - Are detected "solenoids" real?

## Quick Commands

```bash
# Regenerate viewer data after algorithm changes
python generate_viewer_data.py

# Run detection on a single protein
python -c "
import numpy as np
from detect_solenoids import detect_and_score_solenoid_regions, score_solenoid
apc = np.load('cache/apc_matrices/PROTEIN_ID.npy')
print('Shape:', apc.shape)
regions = detect_and_score_solenoid_regions(apc)
for r in regions:
    print(f\"{r['start']}-{r['end']}: n_harm={r['n_harmonics']}, period={r['period']}\")
"

# Check sliding window results for a protein
python -c "
import numpy as np
from detect_solenoids import sliding_window_detection
apc = np.load('cache/apc_matrices/PROTEIN_ID.npy')
results = sliding_window_detection(apc)
for r in results:
    if r['n_harmonics'] >= 2:
        print(f\"{r['start']}-{r['end']}: n_harm={r['n_harmonics']}, period={r['period']}\")
"
```
