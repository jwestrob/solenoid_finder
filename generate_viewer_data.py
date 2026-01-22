#!/usr/bin/env python3
"""
Generate data files for the solenoid viewer web app.
"""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from detect_solenoids import detect_and_score_solenoid_regions, compute_soft_votes

def generate_results_json(cache_dir: Path, output_path: Path):
    """Generate results.json with detected solenoid regions for all cached proteins."""

    apc_dir = cache_dir / "apc_matrices"
    if not apc_dir.exists():
        print(f"Error: {apc_dir} not found")
        return

    results = {}
    apc_files = sorted(apc_dir.glob("*.npy"))

    print(f"Processing {len(apc_files)} cached proteins...")

    for apc_file in apc_files:
        protein_id = apc_file.stem

        try:
            apc = np.load(apc_file)
            L = apc.shape[0]

            # Detect solenoid regions
            regions = detect_and_score_solenoid_regions(apc)

            # Include all regions with n_harm >= 2 and at least 3 repeats
            # Viewer will filter further with adjustable thresholds
            filtered_regions = []
            for r in regions:
                n_harm = r["n_harmonics"]
                cont = r.get("continuity", 0)

                # Minimum threshold: n_harm >= 2
                if n_harm < 2:
                    continue

                period = r["period"] if r["period"] > 0 else 999
                region_len = r["end"] - r["start"]
                n_repeats = region_len / period if period > 0 else 0

                # Minimum threshold: at least 3 repeats
                if n_repeats < 3:
                    continue

                # Still require non-negative continuity
                if cont < 0:
                    continue

                # Compute soft votes
                votes = compute_soft_votes(r)

                filtered_regions.append({
                    "start": int(r["start"]),
                    "end": int(r["end"]),
                    "n_harmonics": int(r["n_harmonics"]),
                    "period": int(period) if period < 999 else None,
                    "n_repeats": round(n_repeats, 1),
                    "fft_prominence": round(float(r.get("fft_prominence", 0.0)), 2),
                    "band_score": round(float(r.get("band_score", 0.0)), 3),
                    "band_coverage": round(float(r.get("band_coverage", 0.0)), 2),
                    "band_consistency": round(float(r.get("band_consistency", 0.0)), 2),
                    "band_strength": round(float(r.get("band_strength", 0.0)), 2),
                    "continuity": round(float(r.get("continuity", 0.0)), 3),
                    "longest_run_frac": round(float(r.get("longest_run_frac", 0.0)), 3),
                    "n_fragments": round(float(r.get("n_fragments", 99.0)), 1),
                    "band_present": round(float(r.get("band_present", 0.0)), 2),
                    "votes": votes["total_votes"],
                    "vote_details": {
                        "harmonics": votes["vote_harmonics"],
                        "continuity": votes["vote_continuity"],
                        "fft": votes["vote_fft"],
                        "band": votes["vote_band"],
                        "run": votes["vote_run"],
                    }
                })

            results[protein_id] = {
                "length": int(L),
                "regions": filtered_regions,
                "has_solenoid": len(filtered_regions) > 0
            }

            if filtered_regions:
                print(f"  {protein_id}: {len(filtered_regions)} region(s)")

        except Exception as e:
            print(f"  Error processing {protein_id}: {e}")

    # Write JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWritten {len(results)} proteins to {output_path}")

    # Summary
    n_with_solenoid = sum(1 for r in results.values() if r["has_solenoid"])
    print(f"Proteins with detected solenoids: {n_with_solenoid}/{len(results)}")


def generate_apc_images(cache_dir: Path, output_dir: Path, results: dict):
    """Generate PNG images of APC matrices for the viewer."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    apc_dir = cache_dir / "apc_matrices"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating APC images...")

    for protein_id, data in results.items():
        if not data["has_solenoid"]:
            continue

        apc_file = apc_dir / f"{protein_id}.npy"
        if not apc_file.exists():
            continue

        try:
            apc = np.load(apc_file)
            L = apc.shape[0]

            fig, ax = plt.subplots(figsize=(8, 8))
            apc_log = np.log1p(apc * 1000)
            ax.imshow(apc_log, cmap='viridis', aspect='equal')

            # Add region boxes
            for region in data["regions"]:
                rect = Rectangle(
                    (region["start"], region["start"]),
                    region["end"] - region["start"],
                    region["end"] - region["start"],
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)

            ax.set_title(f'{protein_id} (L={L})')
            ax.set_xlabel('Residue')
            ax.set_ylabel('Residue')

            plt.tight_layout()
            plt.savefig(output_dir / f"{protein_id}.png", dpi=100, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"  Error generating image for {protein_id}: {e}")

    print(f"Generated images in {output_dir}")


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    cache_dir = base_dir / "cache"
    viewer_dir = base_dir / "viewer"

    results_path = viewer_dir / "data" / "results.json"

    # Generate results JSON
    generate_results_json(cache_dir, results_path)

    # Load results and generate images
    with open(results_path) as f:
        results = json.load(f)

    generate_apc_images(cache_dir, viewer_dir / "apc_images", results)
