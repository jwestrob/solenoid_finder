# Solenoid Detection Analysis

## Overview

Detecting solenoid proteins (ANK, LRR, TPR, WD, HEAT, etc.) from ESM++ attention maps using diagonal band periodicity analysis.

**Script:** `detect_solenoids.py`

---

## Current Status (2025-01-19)

### Full Human Proteome Results (v5)

Ran on 18,147 human proteins with updated scoring:
- **Require strong fundamental**: z > 1.5 for k=1 (filters noise-floor artifacts)
- **n_harmonics threshold**: ≥ 2 for candidate flagging

| Threshold | Candidates | Recall | Precision | FP Rate |
|-----------|------------|--------|-----------|---------|
| n_harm ≥ 2 | 1,186 | 31.5% | 36.6% | 4.5% |
| n_harm ≥ 3 | 974 | 27.3% | 38.6% | 3.6% |
| n_harm ≥ 4 | 802 | 24.3% | 41.8% | 2.8% |
| n_harm ≥ 5 | 635 | 18.8% | 40.6% | 2.3% |

**Key finding**: Precision plateaus ~40%, suggesting many "FPs" are real repeats not annotated in UniProt (e.g., CCP/Sushi domains with period ~56).

**297 potential discoveries**: Unannotated proteins with n_harm ≥ 6.

### Current Approach: n_harmonics as Primary Score

The simplest and most effective approach: **count harmonics directly**.

**Key insight**: A single peak could be noise, but multiple evenly-spaced peaks at P, 2P, 3P... is very unlikely to occur by chance.

### How It Works

1. **Harmonic-first detection**: For each candidate period P, check positions P, 2P, 3P... for elevated signal
2. **Local baseline z-scores**: Compare each harmonic to nearby positions (handles natural decay with distance)
3. **Count significant harmonics**: z-score > 1.5 above local baseline counts as "strong"
4. **Score = n_harmonics**: Simple, interpretable, highly discriminative

### Why Local Baselines Matter

APC values decay exponentially with distance (~0.5x per period). Without local baselines:
- A6NM36 (11 alpha-beta repeats): detected only 1 harmonic
- With local baselines: **7 harmonics detected**

The decay-normalized approach reveals the true harmonic structure.

---

## Cached Test Set Results (n=300)

### Top Candidates (n_harmonics ≥ 2)

| Protein | n_harm | Period | Len | Notes |
|---------|--------|--------|-----|-------|
| O43300 | 8 | 24 | 516 | ANN - LRR |
| A6NIK2 | 8 | 23 | 292 | ANN - LRR |
| Q6UY18 | 8 | 24 | 593 | ANN |
| Q9BXN1 | 8 | 23 | 380 | ANN |
| Q0VAA2 | 8 | 28 | 488 | ANN |
| **A6NM36** | **7** | **23** | **301** | **ANN - 11 alpha-beta repeats** |
| P17024 | 7 | 56 | 532 | Unannotated - investigate! |

### Unannotated Candidates (potential discoveries or FPs)

| Protein | n_harm | Period | Len | Notes |
|---------|--------|--------|-----|-------|
| P17024 | 7 | 56 | 532 | Long period - macro-repeat? |
| Q6ZS27 | 4 | 56 | 426 | Long period |
| Q9NYP8 | 4 | 21 | 219 | Short period - could be real LRR |
| O75467 | 4 | 56 | 553 | Long period |
| Q969J2 | 4 | 55 | 545 | Long period |
| P0DM63 | 4 | 43 | 369 | Medium period |

Many long-period (53-57) candidates suggest CCP/Sushi domains (~60 residues).

### Non-Candidates (n_harmonics < 2)

Most proteins correctly have n_harmonics = 0:
- 92% of unannotated proteins
- 54% of annotated proteins (likely few repeats or weak signal)

---

## Historical: Original Scoring Problems

### The Problem with Old Scoring

```
Annotated median score: 0.468
Unannotated median score: 0.385
Separation: only 1.13x!
```

**Root cause**: `band_intensity` had 1.00x separation (NO discrimination). The old score `sqrt(periodicity_strength * band_intensity)` was basically just periodicity_strength with noise.

### Period=20 Pile-up Issue

| Period | Annotated | Unannotated |
|--------|-----------|-------------|
| 20 | 394 | 7803 |
| 23-24 | 153 | 195 |
| 33 | 121 | 289 |

Short periods pile up because:
1. More chances for spurious peaks (L/P potential harmonics)
2. Alpha helix contacts occur at ~18-22 residue intervals
3. Statistical power inflates apparent significance

**Solution**: Period adjustment `sqrt(period/20)` and harmonic counting handles this naturally.

---

## Key Insight: Harmonics Are Everything

True solenoids with N repeats show peaks at P, 2P, 3P... up to (N-1)×P:
- 5 repeats → peaks at P, 2P, 3P, 4P
- 10 repeats → peaks at P, 2P, 3P, ... 9P

Noise might produce ONE spurious peak. It's very unlikely to produce correlated peaks at exactly P, 2P, 3P.

**Evidence from data**:
- n_harmonics mean for annotated: 0.54
- n_harmonics mean for unannotated: 0.10
- Ratio: 5.4x (much better than score separation!)
- All proteins with n_harmonics=2 in test set were annotated solenoids

---

## Interesting Findings

### Confirmed Solenoids (period matches biology)

| Protein | Score | Period | Type | Known repeat |
|---------|-------|--------|------|--------------|
| LRR proteins | 0.7-0.9 | 22-24 | LRR | ~22-24 residues |
| ANK proteins | 0.5-0.7 | 33 | ANK | ~33 residues |
| TPR proteins | 0.4-0.6 | 34-35 | TPR | ~34 residues |

### Previously Discovered

- **O95171 (Sciellin)**: 16 repeats of ~20 residues - not in UniProt!
- **P02647 (Apolipoprotein A-I)**: Confirmed ~22-residue amphipathic helix repeats
- **Q7L0J3/Q7L1I2/Q496J9 (SV2A/B/C)**: Beta-solenoid in luminal domain

---

## Data Files

- `results/human_solenoid_candidates_v2.tsv` - old detection results
- `results/test_harmonic_v4.tsv` - new harmonic scoring results
- `cache/apc_matrices/` - cached APC matrices for fast iteration
- `cache/cached_protein_ids.json` - list of 300 cached proteins
- `results/human_all_repeat_annotations.json` - UniProt REPEAT annotations (1951 proteins)

---

## Feature Exploration (COMPLETE)

### Goal
Use n_harmonics as a confident label to evaluate OTHER features for solenoid detection.

### Test Set
- **Positives**: n_harmonics ≥ 5 (635 proteins) - high-confidence solenoids
- **Negatives**: n_harmonics = 0 (16,816 proteins) - no periodic signal
- **Excluded**: n_harmonics 1-4 (696 proteins) - ambiguous cases

### Results

| Feature | AUC | Ratio | Notes |
|---------|-----|-------|-------|
| **fundamental_z** | **1.000** | 23x | Perfect discriminator |
| **harmonic_z_sum** | **0.997** | 15x | Sum of positive z-scores across harmonics |
| harmonic_z_mean | 0.993 | 10.6x | Quality of harmonics |
| harmonic_consistency | 0.984 | 6x | Std dev of z-scores |
| band_snr | 0.975 | 3x | Signal-to-noise ratio |
| peak_contrast | 0.957 | 6.8x | Peak vs trough difference |
| decay_rate | 0.734 | 1.1x | Not useful |
| autocorr_peak | 0.173 | 0.65x | **Inverted** - misleading |
| profile_max/mean | ~0.3 | ~0.9x | Raw intensity doesn't matter |

### Conclusion

**`harmonic_z_sum`** (sum of positive z-scores across harmonics) is an excellent differentiator, but it essentially captures the same information as `n_harmonics` - proteins with more/stronger harmonics score higher.

**`n_harmonics` remains the best metric** because:
1. **Interpretable**: "this protein has N evenly-spaced peaks"
2. **Principled threshold**: 2+ harmonics is statistically unlikely by chance
3. **Biologically meaningful**: correlates with number of repeats

The n_harm=1 category contains borderline cases (TIM barrels, irregular repeats, weak signals). **n_harm ≥ 2 is our minimum for calling something a solenoid.**

---

## Sliding Window Analysis (NEW)

### Problem
Global analysis can miss solenoids that are only part of a larger protein. Example: P16473 has a beta solenoid (residues 0-300) + TM region (500-764).

- **Global analysis**: n_harmonics = 2 (diluted signal)
- **Regional analysis (0-250)**: n_harmonics = 7

### Solution: Sliding Window Detection

Scan protein with overlapping windows and compute n_harmonics for each:

```
Window size: 150 residues
Step size: 25 residues
```

### P16473 Results

| Region | n_harmonics | Period | Structure |
|--------|-------------|--------|-----------|
| 0-300 | 4-6 | 23-25 | Beta solenoid |
| 300-400 | 2→0 | 21 | Transition |
| 400-764 | 0 | - | TM/non-periodic |

The method successfully:
1. Identifies solenoid regions with high n_harmonics
2. Localizes where solenoids start and end
3. Distinguishes periodic from non-periodic domains

### Correct Approach: Localize, Then Score

The sliding window is for **localization**, not score optimization:

1. **Sliding window** → Find boundaries of solenoid region(s)
2. **Extract submatrix** for each detected region
3. **Score the region** properly (full room for harmonics within that region)

This gives interpretable output:
```
P16473: solenoid at 0-375, n_harmonics=7, period=25 (vs global n_harm=2)
Q99645: solenoid at 50-300, n_harmonics=5, period=23 (vs global n_harm=3)
Q9H756: solenoid at 0-225, n_harmonics=6, period=24 (vs global n_harm=3)
```

Use `detect_and_score_solenoid_regions()` for this workflow.

### Results on Cached Proteins (n=300)

**Proteins with global n_harm ≥ 1 (n=49):**
- 11 improved with regional analysis
- Best: Q9H756 (3→6), A0A0A6YYL3 (5→7), Q99645 (3→5)

**Proteins with global n_harm = 0 (n=251):**
- **28 rescued** - had micro-solenoids missed by global analysis
- Best rescues:
  - Q5TH74: global=0 → regional=**6** at residues 50-300
  - A6NCF5: global=0 → regional=**5** at residues 225-525
  - Q8N4P6: global=0 → regional=**4** at residues 75-375

These micro-solenoids were invisible to global analysis because non-solenoid regions diluted the periodic signal.

---

## Boundary Refinement (NEW)

### Problem

Sliding window analysis finds rough boundaries, but these often include non-solenoid regions:
- Disordered regions adjacent to the solenoid
- Step size (25 residues) creates coarse boundaries
- Window overlap can extend boundaries too far

Example: P16473 rough boundary was 0-375, but the actual solenoid is 50-250.

### Solution: Asymmetric Boundary Scan

Refine boundaries by varying one at a time:

1. **Refine end first** (fixing start): Scan all possible end positions, find the smallest that maintains max n_harmonics
2. **Refine start second** (fixing refined end): Scan all possible start positions, find the largest that maintains max n_harmonics

This asymmetric approach avoids issues with symmetric shrinkage and handles non-monotonic n_harmonics drops.

### Implementation

```python
def refine_solenoid_boundaries(apc_matrix, rough_start, rough_end, min_size=100, step=25):
    # Step 1: Refine end (fixing start)
    refined_end = refine_boundary_end_scan(apc_matrix, rough_start, rough_end)

    # Step 2: Refine start (fixing refined end)
    refined_start = refine_boundary_start_scan(apc_matrix, rough_start, refined_end)

    return refined_start, refined_end
```

### Results on Cached Proteins (n=49 with n_harm ≥ 2)

**42/49 proteins had boundaries refined:**
- Average shrinkage: **98.8 residues** per protein
- Total shrinkage: 4,150 residues

**Top refinements:**
| Protein | Rough | Refined | Shrinkage | n_harm |
|---------|-------|---------|-----------|--------|
| Q6UY18 | 0-450 | 75-275 | 250 | 8 |
| Q02383 | 0-582 | 175-557 | 200 | 6 |
| Q0VAA2 | 25-475 | 100-350 | 200 | 8 |
| O43300 | 0-375 | 75-275 | 175 | 8 |
| P16473 | 0-375 | 50-250 | 175 | 7 |
| Q8N446 | 0-348 | 0-173 | 175 | 3 |
| Q9UKT6 | 25-425 | 25-250 | 175 | 3 |

The refined boundaries are tighter and more accurately localize the solenoid region.

### Visualization

See `P16473_refined_boundaries.png` for comparison of rough vs refined boundaries on the APC matrix.

---

## Next Steps

1. [x] Implement multi-harmonic scoring
2. [x] Test on cached proteins
3. [x] Switch to local baseline z-scores (fixes decay issue)
4. [x] Simplify to n_harmonics as primary score
5. [x] Run full proteome with new scoring
6. [x] Require strong fundamental (z > 1.5) to filter noise-floor FPs
7. [x] Feature exploration on test set (harmonic_z_sum AUC=0.997)
8. [x] Localize solenoid segments within proteins (sliding window)
9. [x] Boundary refinement (asymmetric scan, 98.8 avg shrinkage)
10. [ ] Investigate unannotated candidates (P17024, Q9NYP8, etc.)
11. [ ] Validate against AlphaFold structures
12. [ ] Run boundary refinement on full proteome

---

## Boundary Detection: Known Issues & Future Directions (2026-01-19)

### The Core Problem

Visual inspection of APC attention maps shows solenoid patterns clearly, but our algorithm consistently **underestimates boundaries**. Examples:

- **O43818 (beta propeller)**: Pattern visible ~50-425, detected 190-420
- **Q9UKT6**: Pattern visible ~25-400, detected 36-396

### Why Boundaries Are Hard

**1. Sliding windows are fundamentally limited at edges**
- With 150-residue windows and period ~40, we capture only ~3.5 repeats per window
- Edge windows inevitably include non-solenoid sequence, diluting harmonic signal
- The first "fully inside" window is offset from true start by ~half a window

**2. Threshold-based detection always underestimates**
- Signal tapers gradually at real biological boundaries (partial/degenerate repeats)
- Any threshold crossing will find a boundary *inside* the true extent
- We're finding where signal becomes "strong enough", not where solenoid actually starts

**3. Beta propellers break our assumptions**
- Our harmonic detection assumes clean diagonal bands at P, 2P, 3P
- Beta propellers have "thatching" pattern: blade 1 contacts blade 7, blade 2 contacts 6, etc.
- Signal distributed across multiple offsets, not just harmonics
- Results in artificially low n_harmonics counts

### Proposed Solutions

**Idea 1: Work Outward from High-Confidence Core** ⬅️ TRY FIRST
- Find the high-confidence region (strong harmonic signal)
- Expand boundaries outward until diagonal signal drops to near-baseline
- Trust that if there's a strong solenoid in the middle, edges are probably also solenoid
- More permissive expansion while keeping strict initial detection

**Idea 2: Period-Guided Boundary Expansion**
- Once we know period P, step outward in increments of P
- Check if each additional repeat contributes to diagonal signal
- Stop when adding another repeat doesn't improve the score
- Biologically motivated: boundaries should fall between repeats

**Idea 3: Different Detection for Different Fold Types**
- Alpha/beta solenoids (ANK, TPR, LRR, HEAT): clean diagonals, current approach works
- Beta propellers (WD): thatching pattern, need block/grid detection instead
- Classify fold type first, then apply appropriate boundary method

**Idea 4: Global Signal Analysis First**
- Instead of sliding windows, analyze entire diagonal intensity profile
- Identify regions where signal is elevated above baseline
- Avoids window-based edge effects entirely
- Use for initial boundary estimation, then refine

### Key Insight

The tension is: **permissive = more false positives, strict = underestimated boundaries**.

Solution: Two-stage approach:
1. **Strict detection** to find regions that DEFINITELY have solenoids
2. **Permissive expansion** to find full extent once solenoid is confirmed
