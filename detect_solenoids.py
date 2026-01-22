#!/usr/bin/env python3
"""
Detect potential solenoid proteins by analyzing off-diagonal bands in APC(attn) maps.

Solenoid proteins (ankyrin repeats, TPR, LRR, etc.) show characteristic periodic
off-diagonal stripes in their APC attention maps, corresponding to contacts between
adjacent repeat units.

Usage:
    python detect_solenoids.py --fasta proteins.fasta --output candidates.tsv --device mps

    # With diagnostic plots
    python detect_solenoids.py --fasta proteins.fasta --output candidates.tsv --save-plots

    # Include shorter periods (10-19) if desired
    python detect_solenoids.py --fasta proteins.fasta --min-period 10 --max-period 40
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

try:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
except ImportError:
    raise ImportError("Required: pip install torch transformers")


# =============================================================================
# Diagonal Analysis Functions
# =============================================================================

def log_transform_apc(apc_matrix: np.ndarray) -> np.ndarray:
    """
    Log-transform APC matrix for more consistent detection.

    This compresses dynamic range and makes thresholds more robust
    across proteins with varying signal strengths.
    """
    return np.log1p(apc_matrix * 1000)


def compute_diagonal_profile(apc_matrix: np.ndarray, smooth_window: int = 3) -> np.ndarray:
    """
    Compute mean intensity along each off-diagonal of the APC matrix.

    For a solenoid with period k, we expect high values at offsets k, 2k, 3k, etc.

    Args:
        apc_matrix: (L, L) APC-corrected attention matrix
        smooth_window: Window size for smoothing (reduces noise)

    Returns:
        (L-1,) array of mean diagonal intensities for offsets 1 to L-1
    """
    # Log-transform for consistent detection across proteins
    apc_log = log_transform_apc(apc_matrix)

    L = apc_log.shape[0]
    if L < 2:
        return np.array([])

    profile = np.zeros(L - 1)
    for d in range(1, L):
        # Get diagonal at offset d (upper triangle)
        diag = np.diagonal(apc_log, offset=d)
        if len(diag) > 0:
            profile[d - 1] = np.mean(diag)

    # Optional smoothing to reduce noise
    if smooth_window > 1 and len(profile) > smooth_window:
        profile = uniform_filter1d(profile, size=smooth_window, mode='nearest')

    return profile


def detect_periodicity(
    diagonal_profile: np.ndarray,
    min_period: int = 20,
    max_period: int = 60,
) -> Dict:
    """
    Detect periodic patterns in the diagonal profile using harmonic analysis.

    For each candidate period P, checks for harmonic peaks at P, 2P, 3P, etc.
    True solenoids show multiple harmonics; noise typically shows only one peak.

    Args:
        diagonal_profile: 1D array of diagonal intensities
        min_period: Minimum repeat period to consider
        max_period: Maximum repeat period to consider

    Returns:
        Dict with:
            - period: Detected repeat period (or -1 if none)
            - strength: Fundamental peak strength
            - confidence: Combined score based on harmonics
            - n_harmonics: Number of significant harmonics found
            - peaks: List of (period, score) candidates
    """
    n = len(diagonal_profile)
    if n < max_period * 2:
        return {
            'period': -1,
            'strength': 0.0,
            'confidence': 0.0,
            'n_harmonics': 0,
            'peaks': [],
        }

    profile = diagonal_profile.copy()
    max_harmonics = 8

    # LOCAL baseline function for harmonic detection (handles decay)
    def local_baseline(pos, window=10):
        """Get median value in window around pos, excluding the peak region"""
        left_start = max(0, pos - window)
        left_end = max(0, pos - 3)
        right_start = min(n, pos + 4)
        right_end = min(n, pos + window + 1)

        left_vals = profile[left_start:left_end] if left_end > left_start else []
        right_vals = profile[right_start:right_end] if right_end > right_start else []

        all_vals = list(left_vals) + list(right_vals)
        if len(all_vals) > 0:
            return np.median(all_vals), np.std(all_vals) + 1e-10
        return profile[pos], 1e-10

    best_period = -1
    best_score = -1
    best_n_harmonics = 0
    best_strength = 0.0
    all_candidates = []

    # For each candidate period, compute harmonic score
    for period in range(min_period, min(max_period + 1, n // 3)):
        harmonic_z_scores = []

        # Check harmonics k=1, 2, 3, ...
        for k in range(1, max_harmonics + 1):
            pos = period * k - 1  # 0-indexed position
            if pos >= n - 2:
                break

            # Get peak value (max in small window for alignment tolerance)
            window_start = max(0, pos - 1)
            window_end = min(n, pos + 2)
            peak_val = np.max(profile[window_start:window_end])

            # LOCAL z-score: compare to nearby baseline (handles decay)
            base_median, base_std = local_baseline(pos)
            z = (peak_val - base_median) / base_std
            harmonic_z_scores.append(z)

        if len(harmonic_z_scores) < 2:
            continue

        # Count harmonics at different thresholds
        n_strong = sum(1 for z in harmonic_z_scores if z > 1.5)  # p < 0.07
        n_moderate = sum(1 for z in harmonic_z_scores if z > 1.0)  # p < 0.16
        n_weak = sum(1 for z in harmonic_z_scores if z > 0.5)  # p < 0.31

        # Fundamental strength (k=1)
        fundamental_z = harmonic_z_scores[0]

        # Harmonic evidence score:
        # - Fundamental must be strong (z > 1.5) to avoid noise-floor artifacts
        # - Each additional harmonic adds confidence
        # - Cumulative z-score captures overall signal
        if fundamental_z < 1.5:
            # Weak fundamental - not a good candidate
            # Without strong k=1, "harmonics" at k=2,3... are likely noise
            continue

        cumulative_z = sum(max(0, z) for z in harmonic_z_scores)

        # Score formula:
        # - Base: fundamental strength
        # - Bonus: additional harmonics (each z>1.5 adds 0.2, z>0.5 adds 0.1)
        # - Bonus: cumulative evidence
        additional_strong = max(0, n_strong - 1)  # Exclude fundamental
        additional_moderate = max(0, n_moderate - n_strong)
        additional_weak = max(0, n_weak - n_moderate)

        harmonic_bonus = additional_strong * 0.25 + additional_moderate * 0.15 + additional_weak * 0.05
        cumulative_bonus = min(0.3, cumulative_z * 0.02)

        score = fundamental_z * 0.1 + harmonic_bonus + cumulative_bonus

        all_candidates.append((period, score, n_strong))

        if score > best_score:
            best_score = score
            best_period = period
            best_n_harmonics = n_strong
            best_strength = np.log1p(max(0, fundamental_z - 1))

    if best_period < 0:
        return {
            'period': -1,
            'strength': 0.0,
            'confidence': 0.0,
            'n_harmonics': 0,
            'fft_prominence': 0.0,
            'peaks': [],
        }

    # === FFT PEAK PROMINENCE CHECK ===
    # Verify there's an actual peak in the FFT at the detected period
    # This filters out false positives where local z-scores pass but
    # there's no global periodicity in the signal
    from scipy.fft import rfft, rfftfreq
    from scipy.signal import find_peaks

    fft_vals = np.abs(rfft(profile))
    freqs = rfftfreq(n)

    # Convert to period space (skip DC component)
    periods = np.zeros(len(freqs))
    periods[1:] = 1.0 / freqs[1:]

    # Find the FFT bin closest to the detected period
    period_idx = np.argmin(np.abs(periods - best_period))

    # Compute prominence: how much does this peak stand out from neighbors?
    # Use a local window to compute baseline
    window = max(3, int(len(fft_vals) * 0.1))  # 10% of spectrum or at least 3
    left = max(1, period_idx - window)
    right = min(len(fft_vals), period_idx + window)

    # Local baseline (exclude the peak region itself)
    local_vals = np.concatenate([fft_vals[left:max(left, period_idx-2)],
                                  fft_vals[min(right, period_idx+3):right]])
    if len(local_vals) > 0:
        local_median = np.median(local_vals)
        local_std = np.std(local_vals) + 1e-10
        fft_z = (fft_vals[period_idx] - local_median) / local_std
    else:
        fft_z = 0.0

    # Also check if it's a local maximum (peak, not just high value)
    is_local_max = True
    if period_idx > 0 and fft_vals[period_idx] < fft_vals[period_idx - 1]:
        is_local_max = False
    if period_idx < len(fft_vals) - 1 and fft_vals[period_idx] < fft_vals[period_idx + 1]:
        is_local_max = False

    # FFT prominence score
    fft_prominence = fft_z if is_local_max else fft_z * 0.5

    # If no significant FFT peak, penalize n_harmonics
    # Only penalize if FFT prominence is clearly negative (below local baseline)
    # This filters out cases where there's no peak at all at the detected period
    if fft_prominence < 0.0:
        # No real periodicity - halve the harmonic count
        best_n_harmonics = best_n_harmonics // 2

    # Confidence based on score, period, and FFT prominence
    # Longer periods are harder to match by chance, so boost them slightly
    period_factor = np.sqrt(best_period / min_period)
    fft_factor = min(1.5, 0.5 + fft_prominence * 0.25)  # Boost for strong FFT peaks
    confidence = min(1.0, best_score * period_factor * fft_factor)

    return {
        'period': best_period,
        'strength': float(best_strength),
        'confidence': float(confidence),
        'n_harmonics': best_n_harmonics,
        'fft_prominence': float(fft_prominence),
        'peaks': [(p, s) for p, s, _ in sorted(all_candidates, key=lambda x: -x[1])[:5]],
    }


def compute_band_intensity(
    apc_matrix: np.ndarray,
    period: int,
    n_bands: int = 3,
) -> float:
    """
    Compute mean intensity of diagonal bands at the detected period.

    Args:
        apc_matrix: (L, L) APC matrix
        period: Detected repeat period
        n_bands: Number of harmonic bands to average (1, 2, 3, ...)

    Returns:
        Mean intensity across periodic bands
    """
    apc_log = log_transform_apc(apc_matrix)
    L = apc_log.shape[0]
    if period <= 0 or period >= L:
        return 0.0

    intensities = []
    for k in range(1, n_bands + 1):
        offset = period * k
        if offset >= L:
            break
        diag = np.diagonal(apc_log, offset=offset)
        if len(diag) > 0:
            intensities.append(np.mean(diag))

    return float(np.mean(intensities)) if intensities else 0.0


def compute_band_continuity(
    apc_matrix: np.ndarray,
    period: int,
    n_bands: int = 4,
) -> Dict:
    """
    Measure 2D band structure by analyzing continuity of diagonal bands.

    Real solenoids have CONTINUOUS diagonal bands spanning the repeat region.
    False positives have FRAGMENTED high-intensity spots that happen to be periodic.

    For each band diagonal at offset k*P:
    1. Extract the diagonal values
    2. Identify "high" regions (above adaptive threshold)
    3. Find the longest continuous run of high values
    4. Compute continuity = longest_run / diagonal_length

    Args:
        apc_matrix: (L, L) APC matrix
        period: Detected repeat period
        n_bands: Number of harmonic bands to analyze

    Returns:
        Dict with:
        - continuity: Mean continuity across bands (0-1, higher = more continuous)
        - longest_run_frac: Longest run as fraction of diagonal length
        - n_fragments: Mean number of fragments per band (lower = better)
        - band_present: Fraction of bands that have significant continuity
    """
    # Log-transform for consistent detection
    apc_matrix = log_transform_apc(apc_matrix)

    L = apc_matrix.shape[0]
    if period <= 0 or period >= L:
        return {
            'continuity': 0.0,
            'longest_run_frac': 0.0,
            'n_fragments': 99.0,
            'band_present': 0.0,
        }

    # Compute LOCAL background statistics within the region being analyzed
    # This handles multi-domain proteins where different regions have different baselines
    # Use non-band diagonals within the region as background
    background_vals = []
    max_offset = min(L, period * (n_bands + 2))

    for offset in range(5, max_offset):
        # Skip if this is near a band diagonal (within ±2 of period multiple)
        is_band = any(abs(offset - period * k) <= 2 for k in range(1, n_bands + 2))
        if not is_band:
            diag = np.diagonal(apc_matrix, offset=offset)
            if len(diag) > 5:
                background_vals.extend(diag)

    if len(background_vals) < 20:
        # Fallback to upper triangle if not enough background
        triu_indices = np.triu_indices(L, k=5)
        background_vals = list(apc_matrix[triu_indices])

    local_median = np.median(background_vals)
    local_std = np.std(background_vals) + 1e-10

    # Adaptive threshold: median + 0.5*std (more lenient since we're using local stats)
    threshold = local_median + 0.5 * local_std

    continuities = []
    longest_runs = []
    fragment_counts = []

    for k in range(1, n_bands + 1):
        offset = period * k
        if offset >= L - 10:  # Need reasonable diagonal length
            break

        diag = np.diagonal(apc_matrix, offset=offset)
        if len(diag) < 10:
            continue

        # Binarize: above threshold = 1, below = 0
        binary = (diag > threshold).astype(int)

        # Find runs of consecutive 1s
        runs = []
        current_run = 0
        for val in binary:
            if val == 1:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0
        if current_run > 0:
            runs.append(current_run)

        # Compute metrics for this band
        if runs:
            longest_run = max(runs)
            n_fragments = len(runs)
        else:
            longest_run = 0
            n_fragments = 0

        diag_len = len(diag)
        continuity = longest_run / diag_len
        longest_run_frac = longest_run / diag_len

        continuities.append(continuity)
        longest_runs.append(longest_run_frac)
        fragment_counts.append(n_fragments)

    if not continuities:
        return {
            'continuity': 0.0,
            'longest_run_frac': 0.0,
            'n_fragments': 99.0,
            'band_present': 0.0,
        }

    # Aggregate metrics
    mean_continuity = np.mean(continuities)
    max_longest_run = max(longest_runs)
    mean_fragments = np.mean(fragment_counts)

    # Band is "present" if it has continuity > 0.2 (20% of diagonal is continuous)
    bands_present = sum(1 for c in continuities if c > 0.2) / len(continuities)

    return {
        'continuity': float(mean_continuity),
        'longest_run_frac': float(max_longest_run),
        'n_fragments': float(mean_fragments),
        'band_present': float(bands_present),
    }


def compute_band_score(
    apc_matrix: np.ndarray,
    period: int,
    n_bands: int = 4,
) -> Dict:
    """
    Compute comprehensive band quality metrics for solenoid detection.

    Measures three aspects of diagonal band quality:
    1. Coverage: What fraction of each band exceeds background?
    2. Consistency: How uniform is the intensity along each band? (lower variance = better)
    3. Strength: How bright are bands compared to background?

    Args:
        apc_matrix: (L, L) APC matrix
        period: Detected repeat period
        n_bands: Number of harmonic bands to analyze (1, 2, 3, ...)

    Returns:
        Dict with band_score, coverage, consistency, strength, and per-band details
    """
    # Log-transform for consistent detection
    apc_matrix = log_transform_apc(apc_matrix)

    L = apc_matrix.shape[0]
    if period <= 0 or period >= L:
        return {
            'band_score': 0.0,
            'coverage': 0.0,
            'consistency': 0.0,
            'strength': 0.0,
        }

    # Compute background statistics from non-band diagonals
    # Use diagonals that are NOT at period multiples
    background_vals = []
    for offset in range(1, min(L, period * (n_bands + 1))):
        # Skip if this is a band diagonal (within ±2 of period multiple)
        is_band = any(abs(offset - period * k) <= 2 for k in range(1, n_bands + 2))
        if not is_band:
            diag = np.diagonal(apc_matrix, offset=offset)
            if len(diag) > 5:
                background_vals.extend(diag)

    if len(background_vals) < 10:
        # Not enough background data
        background_vals = apc_matrix[np.triu_indices(L, k=1)].flatten()

    bg_mean = np.mean(background_vals)
    bg_std = np.std(background_vals) + 1e-10
    threshold = bg_mean + 0.5 * bg_std  # Threshold for "above background"

    # Analyze each band
    coverages = []
    consistencies = []
    strengths = []

    for k in range(1, n_bands + 1):
        offset = period * k
        if offset >= L - 5:  # Need at least 5 residues
            break

        diag = np.diagonal(apc_matrix, offset=offset)
        if len(diag) < 5:
            continue

        # Coverage: fraction of diagonal above threshold
        coverage = np.mean(diag > threshold)
        coverages.append(coverage)

        # Consistency: inverse of coefficient of variation (lower CV = more consistent)
        # Use 1 / (1 + CV) to map to [0, 1] range
        diag_mean = np.mean(diag)
        diag_std = np.std(diag)
        cv = diag_std / (diag_mean + 1e-10)
        consistency = 1.0 / (1.0 + cv)
        consistencies.append(consistency)

        # Strength: z-score of band mean vs background
        strength = (diag_mean - bg_mean) / bg_std
        strengths.append(max(0, strength))  # Clip negative values

    if not coverages:
        return {
            'band_score': 0.0,
            'coverage': 0.0,
            'consistency': 0.0,
            'strength': 0.0,
        }

    # Aggregate metrics
    mean_coverage = np.mean(coverages)
    mean_consistency = np.mean(consistencies)
    mean_strength = np.mean(strengths)

    # Combined band_score: product of normalized metrics
    # Scale strength to [0,1] range (cap at z=5)
    norm_strength = min(1.0, mean_strength / 5.0)

    band_score = mean_coverage * mean_consistency * (0.5 + norm_strength)

    return {
        'band_score': float(band_score),
        'coverage': float(mean_coverage),
        'consistency': float(mean_consistency),
        'strength': float(mean_strength),
    }


def compute_soft_votes(region: Dict) -> Dict:
    """
    Compute soft voting score for a detected region.

    Each metric votes pass/fail based on relaxed thresholds.
    Proteins strong on multiple metrics rank higher.

    Thresholds set around 25th percentile of validated detections.

    Returns:
        Dict with individual votes and total vote count
    """
    # Thresholds calibrated for log-transformed APC values
    votes = {
        'vote_harmonics': 1 if region.get('n_harmonics', 0) >= 4 else 0,
        'vote_continuity': 1 if region.get('continuity', 0) >= 0.10 else 0,
        'vote_fft': 1 if region.get('fft_prominence', 0) >= 0.05 else 0,
        'vote_band': 1 if region.get('band_score', 0) >= 0.05 else 0,
        'vote_run': 1 if region.get('longest_run_frac', 0) >= 0.20 else 0,
    }
    votes['total_votes'] = sum(votes.values())
    return votes


def score_solenoid(
    apc_matrix: np.ndarray,
    min_period: int = 20,
    max_period: int = 60,
) -> Dict:
    """
    Compute solenoid score for an APC matrix.

    Combines periodicity detection with band intensity measurement.

    Args:
        apc_matrix: (L, L) APC-corrected attention matrix
        min_period: Minimum repeat period
        max_period: Maximum repeat period

    Returns:
        Dict with all solenoid metrics
    """
    L = apc_matrix.shape[0]

    # Compute diagonal profile
    profile = compute_diagonal_profile(apc_matrix)

    if len(profile) < min_period:
        return {
            'length': L,
            'solenoid_score': 0.0,
            'detected_period': -1,
            'periodicity_strength': 0.0,
            'band_intensity': 0.0,
            'confidence': 0.0,
            'n_harmonics': 0,
            'is_candidate': False,
        }

    # Detect periodicity
    periodicity = detect_periodicity(profile, min_period, max_period)

    # Compute band intensity at detected period
    if periodicity['period'] > 0:
        band_intensity = compute_band_intensity(apc_matrix, periodicity['period'])
        band_metrics = compute_band_score(apc_matrix, periodicity['period'])
        continuity_metrics = compute_band_continuity(apc_matrix, periodicity['period'])
    else:
        band_intensity = 0.0
        band_metrics = {'band_score': 0.0, 'coverage': 0.0, 'consistency': 0.0, 'strength': 0.0}
        continuity_metrics = {'continuity': 0.0, 'longest_run_frac': 0.0, 'n_fragments': 99.0, 'band_present': 0.0}

    # Primary discriminator: n_harmonics
    # Multiple harmonics at P, 2P, 3P... is strong evidence of true solenoid
    n_harmonics = periodicity.get('n_harmonics', 0)
    period = periodicity['period'] if periodicity['period'] > 0 else -1

    # Simple score: just use n_harmonics directly
    # This is the most interpretable and discriminative metric
    solenoid_score = float(n_harmonics)

    return {
        'length': L,
        'solenoid_score': float(solenoid_score),
        'detected_period': periodicity['period'],
        'periodicity_strength': periodicity['strength'],
        'band_intensity': float(band_intensity),
        'confidence': periodicity['confidence'],
        'n_harmonics': periodicity.get('n_harmonics', 0),
        'fft_prominence': periodicity.get('fft_prominence', 0.0),
        'band_score': band_metrics['band_score'],
        'band_coverage': band_metrics['coverage'],
        'band_consistency': band_metrics['consistency'],
        'band_strength': band_metrics['strength'],
        'continuity': continuity_metrics['continuity'],
        'longest_run_frac': continuity_metrics['longest_run_frac'],
        'n_fragments': continuity_metrics['n_fragments'],
        'band_present': continuity_metrics['band_present'],
        'is_candidate': n_harmonics >= 2,  # Threshold: 2+ harmonics = solenoid candidate
    }


def sliding_window_detection(
    apc_matrix: np.ndarray,
    window_size: int = 150,
    step_size: int = 25,
    min_period: int = 20,
    max_period: int = 60,
) -> List[Dict]:
    """
    Detect solenoid regions using sliding window analysis.

    Useful for proteins where solenoids are only part of the structure.
    Global analysis can miss these due to signal dilution.

    Args:
        apc_matrix: (L, L) APC-corrected attention matrix
        window_size: Size of sliding window (default: 150)
        step_size: Step between windows (default: 25)
        min_period: Minimum repeat period to detect
        max_period: Maximum repeat period to detect

    Returns:
        List of dicts with window info: start, end, n_harmonics, period
    """
    L = apc_matrix.shape[0]
    results = []

    if L < window_size:
        # Protein too short for sliding window, use global
        global_result = score_solenoid(apc_matrix, min_period, max_period)
        return [{
            'start': 0,
            'end': L,
            'center': L // 2,
            'n_harmonics': global_result['n_harmonics'],
            'period': global_result['detected_period'],
        }]

    for start in range(0, L - window_size + 1, step_size):
        end = start + window_size
        apc_window = apc_matrix[start:end, start:end]

        # Compute diagonal profile for window
        profile = compute_diagonal_profile(apc_window)

        if len(profile) < min_period:
            results.append({
                'start': start,
                'end': end,
                'center': (start + end) // 2,
                'n_harmonics': 0,
                'period': -1,
            })
            continue

        # Detect periodicity in window
        periodicity = detect_periodicity(profile, min_period, max_period)

        results.append({
            'start': start,
            'end': end,
            'center': (start + end) // 2,
            'n_harmonics': periodicity.get('n_harmonics', 0),
            'period': periodicity['period'],
        })

    return results


def find_solenoid_regions(
    sliding_results: List[Dict],
    min_n_harmonics: int = 2,
) -> List[Dict]:
    """
    Find contiguous solenoid regions from sliding window results.

    Args:
        sliding_results: Output from sliding_window_detection
        min_n_harmonics: Minimum n_harmonics to count as solenoid

    Returns:
        List of regions: start, end, max_n_harmonics, dominant_period
    """
    regions = []
    current_region = None

    for r in sliding_results:
        if r['n_harmonics'] >= min_n_harmonics:
            if current_region is None:
                current_region = {
                    'start': r['start'],
                    'end': r['end'],
                    'max_n_harmonics': r['n_harmonics'],
                    'periods': [r['period']] if r['period'] > 0 else [],
                }
            else:
                current_region['end'] = r['end']
                current_region['max_n_harmonics'] = max(
                    current_region['max_n_harmonics'], r['n_harmonics']
                )
                if r['period'] > 0:
                    current_region['periods'].append(r['period'])
        else:
            if current_region is not None:
                # Compute dominant period
                if current_region['periods']:
                    current_region['dominant_period'] = int(
                        np.median(current_region['periods'])
                    )
                else:
                    current_region['dominant_period'] = -1
                del current_region['periods']
                regions.append(current_region)
                current_region = None

    # Don't forget last region
    if current_region is not None:
        if current_region['periods']:
            current_region['dominant_period'] = int(
                np.median(current_region['periods'])
            )
        else:
            current_region['dominant_period'] = -1
        del current_region['periods']
        regions.append(current_region)

    return regions


def detect_and_score_solenoid_regions(
    apc_matrix: np.ndarray,
    window_size: int = 150,
    step_size: int = 25,
    min_period: int = 20,
    max_period: int = 60,
    min_n_harmonics: int = 2,
) -> List[Dict]:
    """
    Detect solenoid regions and score each one properly.

    This is the recommended approach for multi-domain proteins:
    1. Use sliding window to find region boundaries
    2. Extract each region's submatrix
    3. Score the region with full harmonic analysis

    Args:
        apc_matrix: (L, L) APC-corrected attention matrix
        window_size: Size of sliding window for boundary detection
        step_size: Step between windows
        min_period: Minimum repeat period to detect
        max_period: Maximum repeat period to detect
        min_n_harmonics: Minimum n_harmonics to consider a region

    Returns:
        List of regions with: start, end, n_harmonics, period, region_length, is_full_protein
    """
    L = apc_matrix.shape[0]

    # Step 1: Sliding window to find candidate regions
    sw_results = sliding_window_detection(
        apc_matrix,
        window_size=window_size,
        step_size=step_size,
        min_period=min_period,
        max_period=max_period
    )

    # Step 2: Find contiguous regions (require n_harm >= 3 to filter noise)
    raw_regions = find_solenoid_regions(sw_results, min_n_harmonics=max(3, min_n_harmonics))

    if not raw_regions:
        # No regions found, return global analysis as fallback
        global_score = score_solenoid(apc_matrix, min_period, max_period)
        return [{
            'start': 0,
            'end': L,
            'n_harmonics': global_score['n_harmonics'],
            'period': global_score['detected_period'],
            'region_length': L,
            'is_full_protein': True,
        }]

    # Step 3: For each region, score and refine boundaries
    scored_regions = []
    for region in raw_regions:
        start = region['start']
        end = region['end']

        # Score the rough region (high-confidence core) to get n_harmonics and period
        apc_sub = apc_matrix[start:end, start:end]
        region_score = score_solenoid(apc_sub, min_period, max_period)

        # Keep the original score - this reflects the high-confidence core
        core_n_harmonics = region_score['n_harmonics']
        core_period = region_score['detected_period']

        # Refine boundaries using diagonal intensity if we have a valid period
        # This EXPANDS the boundaries to capture full extent, but we keep the core score
        if core_period > 0 and core_n_harmonics >= 2:
            refined_start, refined_end = refine_boundaries_by_diagonal_intensity(
                apc_matrix, start, end, core_period
            )
            start, end = refined_start, refined_end

        scored_regions.append({
            'start': start,
            'end': end,
            'n_harmonics': core_n_harmonics,  # Keep score from high-confidence core
            'period': core_period,
            'fft_prominence': region_score.get('fft_prominence', 0.0),
            'band_score': region_score.get('band_score', 0.0),
            'band_coverage': region_score.get('band_coverage', 0.0),
            'band_consistency': region_score.get('band_consistency', 0.0),
            'band_strength': region_score.get('band_strength', 0.0),
            'continuity': region_score.get('continuity', 0.0),
            'longest_run_frac': region_score.get('longest_run_frac', 0.0),
            'n_fragments': region_score.get('n_fragments', 99.0),
            'band_present': region_score.get('band_present', 0.0),
            'region_length': end - start,
            'is_full_protein': (start == 0 and end == L),
        })

    # Step 4: Merge overlapping regions (can happen after boundary refinement)
    merged_regions = merge_overlapping_regions(scored_regions)

    return merged_regions


def merge_overlapping_regions(regions: List[Dict], period_tolerance: int = 8) -> List[Dict]:
    """
    Merge regions that overlap OR are close together with similar periods.

    Args:
        regions: List of region dicts with start, end, n_harmonics, period
        period_tolerance: Maximum period difference to consider same solenoid

    Returns:
        List of merged regions
    """
    if len(regions) <= 1:
        return regions

    # Sort by start position
    sorted_regions = sorted(regions, key=lambda r: r['start'])

    merged = []
    current = sorted_regions[0].copy()

    for region in sorted_regions[1:]:
        # Check overlap or proximity
        overlap_start = max(current['start'], region['start'])
        overlap_end = min(current['end'], region['end'])
        has_overlap = overlap_end > overlap_start

        # Also merge if regions are close (within one period of each other)
        gap = region['start'] - current['end']
        avg_period = (current['period'] + region['period']) / 2
        is_close = gap < avg_period

        # Check if periods are similar
        period_diff = abs(current['period'] - region['period'])
        similar_period = period_diff <= period_tolerance

        if (has_overlap or is_close) and similar_period:
            # Merge: take union of boundaries, keep best n_harmonics
            current['start'] = min(current['start'], region['start'])
            current['end'] = max(current['end'], region['end'])
            current['region_length'] = current['end'] - current['start']

            # Keep the period and n_harmonics from the region with more harmonics
            if region['n_harmonics'] > current['n_harmonics']:
                current['n_harmonics'] = region['n_harmonics']
                current['period'] = region['period']
        else:
            # Not mergeable, save current and start new
            merged.append(current)
            current = region.copy()

    merged.append(current)
    return merged


def refine_boundaries_by_diagonal_intensity(
    apc_matrix: np.ndarray,
    rough_start: int,
    rough_end: int,
    period: int,
    smooth_window: int = 25,
) -> Tuple[int, int]:
    """
    Refine solenoid boundaries by expanding outward from high-confidence core.

    Strategy: Start from the detected region (high-confidence core) and expand
    outward until the diagonal signal drops to near-baseline. This is more
    permissive than threshold-crossing detection and captures the full extent
    of solenoids including tapered edges.

    Args:
        apc_matrix: Full APC matrix
        rough_start: Initial rough start from sliding window
        rough_end: Initial rough end from sliding window
        period: Detected period of the solenoid
        smooth_window: Window size for smoothing (default: 25)

    Returns:
        Tuple of (refined_start, refined_end)
    """
    # Log-transform for consistent detection
    apc_matrix = log_transform_apc(apc_matrix)

    L = apc_matrix.shape[0]

    # Sum intensity along harmonic diagonals (P, 2P, 3P)
    # Use as many harmonics as available at each position
    max_len = L - period  # Only need 1st harmonic to exist
    if max_len <= smooth_window:
        return rough_start, rough_end

    harmonic_sum = np.zeros(max_len)
    harmonic_count = np.zeros(max_len)

    for k in range(1, 4):
        offset = k * period
        if offset < L:
            diag = np.diagonal(apc_matrix, offset=offset)
            valid_len = min(len(diag), max_len)
            harmonic_sum[:valid_len] += diag[:valid_len]
            harmonic_count[:valid_len] += 1

    # Normalize by number of contributing harmonics
    harmonic_count[harmonic_count == 0] = 1
    harmonic_sum = harmonic_sum / harmonic_count

    # Smooth to reduce noise
    if len(harmonic_sum) <= smooth_window:
        return rough_start, rough_end

    smoothed = np.convolve(harmonic_sum, np.ones(smooth_window)/smooth_window, mode='valid')
    positions = np.arange(smooth_window//2, smooth_window//2 + len(smoothed))

    # Compute baseline using 10th percentile of full signal
    # This is robust to solenoid presence - even if solenoid covers most of protein,
    # the 10th percentile captures the "gaps" between bands
    baseline = np.percentile(smoothed, 10)

    # Get peak from the rough region (high-confidence core)
    rough_start_idx = max(0, np.searchsorted(positions, rough_start))
    rough_end_idx = min(len(smoothed), np.searchsorted(positions, rough_end))
    if rough_end_idx <= rough_start_idx:
        return rough_start, rough_end

    region_smoothed = smoothed[rough_start_idx:rough_end_idx]
    peak = np.percentile(region_smoothed, 90)

    if peak <= baseline * 1.3:
        # Not enough contrast
        return rough_start, rough_end

    # Two-stage boundary refinement:
    # 1. CONTRACT: Find where signal is strong within rough region (removes weak edges)
    # 2. EXPAND: Extend outward where signal remains elevated

    # Threshold for "strong" signal (30% between baseline and peak)
    strong_threshold = baseline + 0.30 * (peak - baseline)
    # Threshold for "elevated" signal (15% above baseline)
    elevated_threshold = baseline * 1.15

    # STAGE 1: CONTRACT - Find first/last positions with strong signal within rough region
    strong_in_rough = []
    for i in range(rough_start_idx, min(rough_end_idx + 1, len(smoothed))):
        if smoothed[i] >= strong_threshold:
            strong_in_rough.append(i)

    if not strong_in_rough:
        return rough_start, rough_end

    contracted_start_idx = strong_in_rough[0]
    contracted_end_idx = strong_in_rough[-1]

    # STAGE 2: EXPAND - Extend outward while signal remains elevated
    # Allow bridging small gaps (up to 1 period worth of positions)
    max_gap = period

    refined_start_idx = contracted_start_idx
    gap_count = 0
    for i in range(contracted_start_idx - 1, -1, -1):
        if smoothed[i] >= elevated_threshold:
            refined_start_idx = i
            gap_count = 0  # Reset gap counter
        else:
            gap_count += 1
            if gap_count > max_gap:
                break

    refined_end_idx = contracted_end_idx
    gap_count = 0
    for i in range(contracted_end_idx + 1, len(smoothed)):
        if smoothed[i] >= elevated_threshold:
            refined_end_idx = i
            gap_count = 0  # Reset gap counter
        else:
            gap_count += 1
            if gap_count > max_gap:
                break

    # Convert indices back to positions
    refined_start = int(positions[refined_start_idx])
    refined_end = int(positions[min(refined_end_idx, len(positions) - 1)])

    # Sanity checks
    refined_start = max(0, refined_start)
    refined_end = min(L, refined_end)

    # Ensure minimum size
    if refined_end - refined_start < 100:
        return rough_start, rough_end

    return refined_start, refined_end


def refine_boundary_end_scan(
    apc_matrix: np.ndarray,
    start: int,
    rough_end: int,
    min_period: int = 20,
    max_period: int = 60,
    min_size: int = 100,
    step: int = 25,
) -> Tuple[int, int]:
    """
    Refine the end boundary by scanning all possible positions.

    Finds the smallest end position that maintains max n_harmonics.
    More robust than greedy search - handles non-monotonic drops.

    Args:
        apc_matrix: Full APC matrix
        start: Fixed start position
        rough_end: Initial rough end boundary
        min_period: Minimum period for scoring
        max_period: Maximum period for scoring
        min_size: Minimum region size to consider
        step: Step size for scanning

    Returns:
        Tuple of (refined_end, n_harmonics)
    """
    apc_rough = apc_matrix[start:rough_end, start:rough_end]
    max_n_harm = score_solenoid(apc_rough, min_period, max_period)['n_harmonics']

    # Scan all possible end positions
    candidates = []
    for test_end in range(rough_end, start + min_size - 1, -step):
        apc_sub = apc_matrix[start:test_end, start:test_end]
        result = score_solenoid(apc_sub, min_period, max_period)
        candidates.append((test_end, result['n_harmonics']))

    # Find smallest end that maintains max n_harmonics
    valid = [(end, n) for end, n in candidates if n >= max_n_harm]
    if valid:
        best_end = min(valid, key=lambda x: x[0])[0]
        return best_end, max_n_harm

    return rough_end, max_n_harm


def refine_boundary_start_scan(
    apc_matrix: np.ndarray,
    rough_start: int,
    end: int,
    min_period: int = 20,
    max_period: int = 60,
    min_size: int = 100,
    step: int = 25,
) -> Tuple[int, int]:
    """
    Refine the start boundary by scanning all possible positions.

    Finds the largest start position that maintains max n_harmonics.
    Should be called after refine_boundary_end_scan.

    Args:
        apc_matrix: Full APC matrix
        rough_start: Initial rough start boundary
        end: Fixed end position (already refined)
        min_period: Minimum period for scoring
        max_period: Maximum period for scoring
        min_size: Minimum region size to consider
        step: Step size for scanning

    Returns:
        Tuple of (refined_start, n_harmonics)
    """
    apc_rough = apc_matrix[rough_start:end, rough_start:end]
    max_n_harm = score_solenoid(apc_rough, min_period, max_period)['n_harmonics']

    # Scan all possible start positions
    candidates = []
    for test_start in range(rough_start, end - min_size + 1, step):
        apc_sub = apc_matrix[test_start:end, test_start:end]
        result = score_solenoid(apc_sub, min_period, max_period)
        candidates.append((test_start, result['n_harmonics']))

    # Find largest start that maintains max n_harmonics
    valid = [(start, n) for start, n in candidates if n >= max_n_harm]
    if valid:
        best_start = max(valid, key=lambda x: x[0])[0]
        return best_start, max_n_harm

    return rough_start, max_n_harm


def refine_solenoid_boundaries(
    apc_matrix: np.ndarray,
    rough_start: int,
    rough_end: int,
    min_period: int = 20,
    max_period: int = 60,
    min_size: int = 100,
    step: int = 25,
) -> Dict:
    """
    Refine solenoid region boundaries asymmetrically.

    Strategy: Vary one boundary at a time
    1. First refine end (fixing start) - find smallest end maintaining max n_harm
    2. Then refine start (fixing refined end) - find largest start maintaining max n_harm

    Args:
        apc_matrix: Full APC matrix
        rough_start: Initial rough start from sliding window
        rough_end: Initial rough end from sliding window
        min_period: Minimum period for scoring
        max_period: Maximum period for scoring
        min_size: Minimum region size to consider
        step: Step size for scanning

    Returns:
        Dict with refined_start, refined_end, n_harmonics, shrinkage
    """
    # Step 1: Refine end (fixing start)
    refined_end, n_harm_after_end = refine_boundary_end_scan(
        apc_matrix, rough_start, rough_end, min_period, max_period, min_size, step
    )

    # Step 2: Refine start (fixing refined end)
    refined_start, final_n_harm = refine_boundary_start_scan(
        apc_matrix, rough_start, refined_end, min_period, max_period, min_size, step
    )

    # Calculate shrinkage
    rough_size = rough_end - rough_start
    refined_size = refined_end - refined_start
    shrinkage = rough_size - refined_size

    return {
        'rough_start': rough_start,
        'rough_end': rough_end,
        'refined_start': refined_start,
        'refined_end': refined_end,
        'n_harmonics': final_n_harm,
        'rough_size': rough_size,
        'refined_size': refined_size,
        'shrinkage': shrinkage,
    }


# =============================================================================
# Model and APC Generation (adapted from generate_esmpp.py)
# =============================================================================

def resolve_dtype(dtype_str: str) -> 'torch.dtype':
    s = dtype_str.lower().strip()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32


def load_esmpp(model_id: str, device: 'torch.device', dtype: 'torch.dtype', compile_model: bool = False):
    """Load ESM++ model and tokenizer."""
    print(f"[INFO] Loading model: {model_id}")
    model = AutoModelForMaskedLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    model.to(device)

    # torch.compile for speedup (PyTorch 2.0+)
    if compile_model and hasattr(torch, 'compile'):
        print("[INFO] Compiling model with torch.compile()...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")

    return model, tok


def get_residue_indices(tokenized: dict) -> 'torch.Tensor':
    """Get indices of residue tokens (excluding special tokens)."""
    mask = tokenized.get("special_tokens_mask")
    if mask is None:
        seq_len = tokenized["input_ids"].shape[1]
        return torch.arange(1, seq_len - 1)
    keep = (mask[0] == 0)
    return torch.nonzero(keep, as_tuple=False).squeeze(-1)


@torch.inference_mode()
def compute_apc(
    model,
    tokenizer,
    sequence: str,
    device: 'torch.device',
    last_layers: int = 8,
    min_sep: int = 6,
) -> np.ndarray:
    """
    Compute APC-corrected attention map for a single sequence.

    Returns:
        (L, L) numpy array
    """
    tokenized = tokenizer(
        [sequence],
        return_tensors="pt",
        padding=False,
        return_special_tokens_mask=True,
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items() if isinstance(v, torch.Tensor)}
    residue_idx = get_residue_indices(tokenized)

    out = model(
        **tokenized,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=True,
    )

    attns = out.attentions
    n_layers = len(attns)
    use_layers = list(range(max(0, n_layers - last_layers), n_layers))

    W_acc = None
    n_acc = 0

    for li in use_layers:
        A = attns[li][0]  # (H, Ltok, Ltok)
        A = 0.5 * (A + A.transpose(-1, -2))
        A = A.index_select(1, residue_idx).index_select(2, residue_idx)

        # APC correction
        row = A.sum(dim=-1, keepdim=True)
        col = A.sum(dim=-2, keepdim=True)
        tot = A.sum(dim=(-1, -2), keepdim=True) + 1e-12
        A = A - (row * col) / tot
        A = torch.relu(A)

        A_layer = A.mean(dim=0)
        if W_acc is None:
            W_acc = A_layer
        else:
            W_acc = W_acc + A_layer
        n_acc += 1

    W = W_acc / max(1, n_acc)

    # Free attention memory immediately
    del out, attns, tokenized
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()

    # Zero near-diagonal
    Lres = W.shape[0]
    if min_sep > 0 and Lres > 0:
        idx = torch.arange(Lres, device=device)
        dist = (idx[:, None] - idx[None, :]).abs()
        W = W.masked_fill(dist < min_sep, 0.0)
    if Lres > 0:
        W.fill_diagonal_(0.0)

    return W.cpu().float().numpy()


@torch.inference_mode()
def compute_apc_batch(
    model,
    tokenizer,
    sequences: List[str],
    device: 'torch.device',
    last_layers: int = 8,
    min_sep: int = 6,
) -> List[np.ndarray]:
    """
    Compute APC-corrected attention maps for a batch of sequences.

    Sequences should be sorted by length for efficiency.

    Returns:
        List of (L, L) numpy arrays
    """
    if not sequences:
        return []

    # Tokenize with padding
    tokenized = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        return_special_tokens_mask=True,
        return_attention_mask=True,
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items() if isinstance(v, torch.Tensor)}

    # Forward pass
    out = model(
        **tokenized,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=True,
    )

    attns = out.attentions
    n_layers = len(attns)
    use_layers = list(range(max(0, n_layers - last_layers), n_layers))
    special_mask = tokenized.get("special_tokens_mask")
    attention_mask = tokenized.get("attention_mask")

    results = []
    batch_size = len(sequences)

    for b in range(batch_size):
        seq_len = len(sequences[b])

        # Get residue indices for this sequence
        if special_mask is not None:
            keep = (special_mask[b] == 0) & (attention_mask[b] == 1)
            residue_idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        else:
            residue_idx = torch.arange(1, seq_len + 1, device=device)

        W_acc = None
        n_acc = 0

        for li in use_layers:
            A = attns[li][b]  # (H, Ltok, Ltok)
            A = 0.5 * (A + A.transpose(-1, -2))
            A = A.index_select(1, residue_idx).index_select(2, residue_idx)

            # APC correction
            row = A.sum(dim=-1, keepdim=True)
            col = A.sum(dim=-2, keepdim=True)
            tot = A.sum(dim=(-1, -2), keepdim=True) + 1e-12
            A = A - (row * col) / tot
            A = torch.relu(A)

            A_layer = A.mean(dim=0)
            if W_acc is None:
                W_acc = A_layer
            else:
                W_acc = W_acc + A_layer
            n_acc += 1

        W = W_acc / max(1, n_acc)

        # Zero near-diagonal
        Lres = W.shape[0]
        if min_sep > 0 and Lres > 0:
            idx = torch.arange(Lres, device=device)
            dist = (idx[:, None] - idx[None, :]).abs()
            W = W.masked_fill(dist < min_sep, 0.0)
        if Lres > 0:
            W.fill_diagonal_(0.0)

        results.append(W.cpu().float().numpy())

    # Free attention memory immediately
    del out, attns, tokenized
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()

    return results


def create_length_sorted_batches(
    items: List[Tuple[str, str]],
    batch_size: int = 4,
    max_tokens: int = 4096,
) -> List[List[Tuple[str, str]]]:
    """
    Create batches sorted by length for efficient padding.

    Args:
        items: List of (uid, sequence) tuples
        batch_size: Maximum sequences per batch
        max_tokens: Maximum total tokens per batch

    Returns:
        List of batches, each batch is a list of (uid, sequence) tuples
    """
    # Sort by length
    sorted_items = sorted(items, key=lambda x: len(x[1]))

    batches = []
    current_batch = []
    current_max_len = 0

    for uid, seq in sorted_items:
        seq_len = len(seq)
        new_max_len = max(current_max_len, seq_len)
        # +2 for BOS/EOS tokens
        new_tokens = (new_max_len + 2) * (len(current_batch) + 1)

        if len(current_batch) >= batch_size or new_tokens > max_tokens:
            if current_batch:
                batches.append(current_batch)
            current_batch = [(uid, seq)]
            current_max_len = seq_len
        else:
            current_batch.append((uid, seq))
            current_max_len = new_max_len

    if current_batch:
        batches.append(current_batch)

    return batches


# =============================================================================
# Data Loading
# =============================================================================

def load_sequences(fasta_path: Path) -> Dict[str, str]:
    """Load sequences from FASTA file."""
    from Bio import SeqIO
    sequences = {}
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        uid = record.id.split('|')[1] if '|' in record.id else record.id
        seq = str(record.seq).upper().replace('*', '')
        if seq:
            sequences[uid] = seq
    return sequences


# =============================================================================
# Diagnostic Plotting
# =============================================================================

def save_diagnostic_plot(
    apc_matrix: np.ndarray,
    results: Dict,
    protein_id: str,
    output_path: Path,
):
    """Save diagnostic plot showing APC matrix and detected periodicity."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # APC heatmap
    ax = axes[0]
    im = ax.imshow(apc_matrix, cmap='hot', aspect='equal')
    ax.set_title(f'{protein_id}\nL={results["length"]}')
    ax.set_xlabel('Residue')
    ax.set_ylabel('Residue')
    plt.colorbar(im, ax=ax, label='APC intensity')

    # Diagonal profile
    ax = axes[1]
    profile = compute_diagonal_profile(apc_matrix)
    ax.plot(range(1, len(profile) + 1), profile)
    if results['detected_period'] > 0:
        # Mark detected period and harmonics
        for k in range(1, 4):
            pos = results['detected_period'] * k
            if pos < len(profile):
                ax.axvline(pos, color='red', linestyle='--', alpha=0.7,
                          label=f'{k}x period' if k == 1 else None)
    ax.set_xlabel('Diagonal offset')
    ax.set_ylabel('Mean APC intensity')
    ax.set_title(f'Diagonal Profile\nPeriod={results["detected_period"]}')
    ax.legend()

    # Autocorrelation
    ax = axes[2]
    if len(profile) > 20:
        centered = profile - np.mean(profile)
        autocorr = np.correlate(centered, centered, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        ax.plot(autocorr[:min(120, len(autocorr))])
        if results['detected_period'] > 0:
            ax.axvline(results['detected_period'], color='red', linestyle='--',
                      label=f'Period={results["detected_period"]}')
        ax.axhline(0.1, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'Score={results["solenoid_score"]:.3f}\nStrength={results["periodicity_strength"]:.3f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Detect solenoid proteins from APC attention maps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: Regional analysis with boundary refinement (localize solenoid regions)
    python detect_solenoids.py --fasta proteins.fasta --output regions.tsv

    # Global analysis only (one score per protein, no boundary detection)
    python detect_solenoids.py --fasta proteins.fasta --output candidates.tsv --global-only

    # With diagnostic plots
    python detect_solenoids.py --fasta proteins.fasta --save-plots --plot-dir plots/
        """
    )
    parser.add_argument("--fasta", type=Path, required=True,
                        help="Input FASTA file")
    parser.add_argument("--output", type=Path, default=Path("solenoid_scores.tsv"),
                        help="Output TSV file (default: solenoid_scores.tsv)")
    parser.add_argument("--model", type=str, default="Synthyra/ESMplusplus_large",
                        help="HuggingFace model ID")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda, mps, or cpu")
    parser.add_argument("--dtype", type=str, default="bf16",
                        help="Model dtype: fp16, bf16, fp32")
    parser.add_argument("--min-period", type=int, default=20,
                        help="Minimum repeat period to detect (default: 20, filters short-range noise)")
    parser.add_argument("--max-period", type=int, default=60,
                        help="Maximum repeat period to detect (default: 60)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Score threshold for candidate flagging (default: 0.3, recalibrate after run)")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum sequence length to process (default: 2048)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of proteins to process")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save diagnostic plots for each protein")
    parser.add_argument("--plot-dir", type=Path, default=None,
                        help="Directory for diagnostic plots (default: output_dir/plots)")
    parser.add_argument("--candidates-only", action="store_true",
                        help="Only output proteins above threshold")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for inference (default: 1, use >1 for speedup)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Maximum tokens per batch (default: 4096)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile() for faster inference (PyTorch 2.0+)")
    parser.add_argument("--global-only", action="store_true",
                        help="Disable regional analysis (one score per protein, no boundary detection)")
    parser.add_argument("--window-size", type=int, default=150,
                        help="Sliding window size for regional analysis (default: 150)")
    parser.add_argument("--step-size", type=int, default=25,
                        help="Step size for sliding window (default: 25)")
    args = parser.parse_args()

    # Validate inputs
    if not args.fasta.exists():
        print(f"[ERROR] FASTA file not found: {args.fasta}")
        sys.exit(1)

    # Setup output
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.save_plots:
        plot_dir = args.plot_dir or args.output.parent / 'plots'
        plot_dir.mkdir(parents=True, exist_ok=True)

    # Load sequences
    print(f"[INFO] Loading sequences from {args.fasta}")
    sequences = load_sequences(args.fasta)
    print(f"[INFO] Loaded {len(sequences)} sequences")

    # Filter by length
    sequences = {k: v for k, v in sequences.items() if len(v) <= args.max_length}
    print(f"[INFO] {len(sequences)} sequences after length filter (<= {args.max_length})")

    if args.limit:
        sequences = dict(list(sequences.items())[:args.limit])
        print(f"[INFO] Limited to {len(sequences)} sequences")

    if not sequences:
        print("[INFO] No sequences to process")
        return

    # Load model
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    model, tokenizer = load_esmpp(args.model, device, dtype, compile_model=args.compile)

    # Process proteins
    print(f"\n[INFO] Processing {len(sequences)} proteins...")
    results = []
    n_candidates = 0
    start_time = time.time()

    from tqdm import tqdm

    # Open output file for streaming writes
    outfile = open(args.output, 'w')
    if not args.global_only:
        outfile.write("protein_id\tlength\tregion_start\tregion_end\tregion_length\t"
                      "n_harmonics\tperiod\tis_candidate\n")
    else:
        outfile.write("protein_id\tlength\tsolenoid_score\tdetected_period\t"
                      "periodicity_strength\tband_intensity\tconfidence\tn_harmonics\tis_candidate\n")
    outfile.flush()

    def write_result(r):
        """Write a single result to the output file."""
        period_str = str(r['detected_period']) if r['detected_period'] > 0 else '-'
        outfile.write(f"{r['protein_id']}\t{r['length']}\t{r['solenoid_score']:.4f}\t"
                      f"{period_str}\t{r['periodicity_strength']:.4f}\t"
                      f"{r['band_intensity']:.4f}\t{r['confidence']:.4f}\t"
                      f"{r['n_harmonics']}\t{r['is_candidate']}\n")
        outfile.flush()

    def write_regional_result(protein_id, length, region):
        """Write a regional analysis result to the output file."""
        period_str = str(region['period']) if region['period'] > 0 else '-'
        is_candidate = region['n_harmonics'] >= 4
        outfile.write(f"{protein_id}\t{length}\t{region['start']}\t{region['end']}\t"
                      f"{region['end'] - region['start']}\t{region['n_harmonics']}\t"
                      f"{period_str}\t{is_candidate}\n")
        outfile.flush()

    if args.batch_size > 1:
        # Batched processing for speedup
        items = list(sequences.items())
        batches = create_length_sorted_batches(items, args.batch_size, args.max_tokens)
        print(f"[INFO] Created {len(batches)} batches (batch_size={args.batch_size}, max_tokens={args.max_tokens})")

        pbar = tqdm(total=len(sequences), desc="Analyzing")
        for batch in batches:
            try:
                uids = [uid for uid, _ in batch]
                seqs = [seq for _, seq in batch]

                # Compute APC matrices for batch
                apcs = compute_apc_batch(model, tokenizer, seqs, device)

                for uid, seq, apc in zip(uids, seqs, apcs):
                    L = len(seq)

                    if not args.global_only:
                        # Regional analysis with boundary refinement
                        regions = detect_and_score_solenoid_regions(
                            apc,
                            window_size=args.window_size,
                            step_size=args.step_size,
                            min_period=args.min_period,
                            max_period=args.max_period
                        )

                        for region in regions:
                            # Only report regions with n_harm >= 3 (filters noise)
                            if region['n_harmonics'] >= 4:
                                write_regional_result(uid, L, region)
                                n_candidates += 1
                                tqdm.write(f"  [CANDIDATE] {uid}: {region['start']}-{region['end']}, n_harm={region['n_harmonics']}, period={region['period']}")
                    else:
                        scores = score_solenoid(apc, args.min_period, args.max_period)
                        scores['protein_id'] = uid
                        # is_candidate is set by score_solenoid based on n_harmonics >= 2

                        if args.save_plots:
                            plot_path = plot_dir / f'{uid}_solenoid.png'
                            save_diagnostic_plot(apc, scores, uid, plot_path)

                        results.append(scores)
                        write_result(scores)
                        if scores['is_candidate']:
                            n_candidates += 1
                            tqdm.write(f"  [CANDIDATE] {uid}: score={scores['solenoid_score']:.3f}, period={scores['detected_period']}, L={scores['length']}")
                    pbar.update(1)

                # Free batch APC matrices
                del apcs

            except Exception as e:
                print(f"\n[ERROR] batch: {e}")
                # Fallback to single processing for failed batch
                for uid, seq in batch:
                    try:
                        apc = compute_apc(model, tokenizer, seq, device)
                        L = len(seq)

                        if not args.global_only:
                            regions = detect_and_score_solenoid_regions(
                                apc,
                                window_size=args.window_size,
                                step_size=args.step_size,
                                min_period=args.min_period,
                                max_period=args.max_period
                            )
                            for region in regions:
                                # Only report regions with n_harm >= 3 (filters noise)
                                if region['n_harmonics'] >= 4:
                                    write_regional_result(uid, L, region)
                                    n_candidates += 1
                        else:
                            scores = score_solenoid(apc, args.min_period, args.max_period)
                            scores['protein_id'] = uid
                            results.append(scores)
                            write_result(scores)
                            if scores['is_candidate']:
                                n_candidates += 1
                                tqdm.write(f"  [CANDIDATE] {uid}: score={scores['solenoid_score']:.3f}, period={scores['detected_period']}, L={scores['length']}")
                    except Exception as e2:
                        print(f"\n[ERROR] {uid}: {e2}")
                    pbar.update(1)

            # Periodic cleanup
            if len(results) % 50 == 0:
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'mps':
                    torch.mps.empty_cache()
        pbar.close()
    else:
        # Single sequence processing (original behavior)
        n_processed = 0
        for uid, seq in tqdm(sequences.items(), desc="Analyzing"):
            try:
                apc = compute_apc(model, tokenizer, seq, device)
                L = len(seq)

                if not args.global_only:
                    # Regional analysis with boundary refinement
                    regions = detect_and_score_solenoid_regions(
                        apc,
                        window_size=args.window_size,
                        step_size=args.step_size,
                        min_period=args.min_period,
                        max_period=args.max_period
                    )

                    for region in regions:
                        # Only report regions with n_harm >= 3 (filters noise)
                        if region['n_harmonics'] >= 4:
                            write_regional_result(uid, L, region)
                            n_candidates += 1
                            tqdm.write(f"  [CANDIDATE] {uid}: {region['start']}-{region['end']}, n_harm={region['n_harmonics']}, period={region['period']}")
                else:
                    scores = score_solenoid(apc, args.min_period, args.max_period)
                    scores['protein_id'] = uid
                    # is_candidate is set by score_solenoid based on n_harmonics >= 2

                    if args.save_plots:
                        plot_path = plot_dir / f'{uid}_solenoid.png'
                        save_diagnostic_plot(apc, scores, uid, plot_path)

                    results.append(scores)
                    write_result(scores)
                    if scores['is_candidate']:
                        n_candidates += 1
                        tqdm.write(f"  [CANDIDATE] {uid}: score={scores['solenoid_score']:.3f}, period={scores['detected_period']}, L={scores['length']}")

            except Exception as e:
                print(f"\n[ERROR] {uid}: {e}")
                continue

            # Periodic cleanup
            n_processed += 1
            if n_processed % 50 == 0:
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'mps':
                    torch.mps.empty_cache()

    # Close output file
    outfile.close()

    # Summary
    elapsed = time.time() - start_time
    print(f"\n[INFO] Done! Processed {len(sequences)} proteins in {elapsed:.1f}s")
    if args.global_only:
        print(f"[INFO] Found {n_candidates} solenoid candidates (n_harmonics >= 2)")
    else:
        print(f"[INFO] Found {n_candidates} solenoid regions (n_harmonics >= 2)")
    print(f"[INFO] Results written to: {args.output}")


if __name__ == '__main__':
    main()
