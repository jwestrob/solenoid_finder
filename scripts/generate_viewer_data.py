#!/usr/bin/env python3
"""
Generate data files for the solenoid viewer web app.

Now with integrated ESM3 structure prediction - folds solenoids as they're detected.
Outputs results.json to a specified run directory.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Optional, Set

sys.path.insert(0, str(Path(__file__).parent.parent))
from detect_solenoids import detect_and_score_solenoid_regions, compute_soft_votes

# ESM3 folding (optional)
ESM3_AVAILABLE = False
esm3_client = None

def init_esm3_client(api_key: str):
    """Initialize ESM3 Forge client."""
    global ESM3_AVAILABLE, esm3_client
    try:
        from esm.sdk.api import ESMProtein, GenerationConfig
        from esm.sdk.forge import ESM3ForgeInferenceClient
        esm3_client = ESM3ForgeInferenceClient(model="esm3-open-2024-03", token=api_key)
        ESM3_AVAILABLE = True
        print("ESM3 client initialized")
    except Exception as e:
        print(f"ESM3 not available: {e}")
        ESM3_AVAILABLE = False

def fold_protein(sequence: str, protein_id: str) -> Optional[str]:
    """Fold a protein using ESM3."""
    if not ESM3_AVAILABLE or esm3_client is None:
        return None
    try:
        from esm.sdk.api import ESMProtein, GenerationConfig
        protein = ESMProtein(sequence=sequence)
        config = GenerationConfig(track="structure", num_steps=8)
        folded = esm3_client.generate(protein, config)
        if folded.coordinates is not None:
            return folded.to_pdb_string()
    except Exception as e:
        print(f"    Fold failed: {e}")
    return None

def load_sequences_from_fasta(fasta_path: Path) -> Dict[str, str]:
    """Load sequences from a specific FASTA file."""
    sequences = {}
    if not fasta_path.exists():
        return sequences

    with open(fasta_path) as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq).replace('*', '')
                header = line[1:]
                if header.startswith(('sp|', 'tr|')):
                    parts = header.split('|')
                    current_id = parts[1] if len(parts) >= 2 else parts[0]
                else:
                    current_id = header.split()[0].split('|')[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            sequences[current_id] = ''.join(current_seq).replace('*', '')
    return sequences

def load_sequences(base_dir: Path) -> Dict[str, str]:
    """Load sequences from FASTA files in data/ and proteins/ folders."""
    sequences = {}
    for folder in [base_dir / "data", base_dir / "proteins"]:
        if not folder.exists():
            continue
        for fasta_file in list(folder.rglob("*.fasta")) + list(folder.rglob("*.faa")):
            with open(fasta_file) as f:
                current_id = None
                current_seq = []
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_id:
                            sequences[current_id] = ''.join(current_seq).replace('*', '')
                        header = line[1:]
                        if header.startswith(('sp|', 'tr|')):
                            parts = header.split('|')
                            current_id = parts[1] if len(parts) >= 2 else parts[0]
                        else:
                            current_id = header.split()[0].split('|')[0]
                        current_seq = []
                    else:
                        current_seq.append(line)
                if current_id:
                    sequences[current_id] = ''.join(current_seq).replace('*', '')
    return sequences

def generate_results_json(apc_dir: Path, output_path: Path, protein_ids: Set[str] = None,
                          sequences: Dict[str, str] = None,
                          fold_min_votes: int = 4, fold_max_length: int = 1024,
                          structures_dir: Path = None):
    """Generate results.json with detected solenoid regions for cached proteins."""

    if not apc_dir.exists():
        print(f"Error: {apc_dir} not found")
        return

    results = {}
    apc_files = sorted(apc_dir.glob("*.npy"))

    # Filter to only requested protein IDs if specified
    if protein_ids:
        apc_files = [f for f in apc_files if f.stem in protein_ids]

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
                # Check if any region meets fold threshold
                max_votes = max(r["votes"] for r in filtered_regions)
                print(f"  {protein_id}: {len(filtered_regions)} region(s), {max_votes} votes", end="")

                # Fold if meets threshold and we have the sequence
                if (ESM3_AVAILABLE and sequences and structures_dir and
                    max_votes >= fold_min_votes and protein_id in sequences):

                    seq = sequences[protein_id]
                    struct_path = structures_dir / f"{protein_id}.pdb"

                    if not struct_path.exists() and len(seq) <= fold_max_length:
                        print(" -> folding...", end="", flush=True)
                        pdb_str = fold_protein(seq, protein_id)
                        if pdb_str:
                            with open(struct_path, "w") as f:
                                f.write(pdb_str)
                            print(" OK", end="")
                        else:
                            print(" FAILED", end="")
                    elif struct_path.exists():
                        print(" (has structure)", end="")

                print()

        except Exception as e:
            print(f"  Error processing {protein_id}: {e}")

    # Write JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWritten {len(results)} proteins to {output_path}")

    # Summary
    n_with_solenoid = sum(1 for r in results.values() if r["has_solenoid"])
    print(f"Proteins with detected solenoids: {n_with_solenoid}/{len(results)}")


def generate_apc_images(apc_dir: Path, output_dir: Path, results: dict):
    """Generate PNG images of APC matrices for the viewer."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating APC images...")

    for protein_id, data in results.items():
        if not data["has_solenoid"]:
            continue

        apc_file = apc_dir / f"{protein_id}.npy"
        if not apc_file.exists():
            continue

        # Skip if image already exists
        output_file = output_dir / f"{protein_id}.png"
        if output_file.exists():
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
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"  Error generating image for {protein_id}: {e}")

    print(f"Generated images in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate viewer data with optional ESM3 folding")
    parser.add_argument("--output", type=Path, help="Output directory for results.json")
    parser.add_argument("--fasta", type=Path, help="Input FASTA file (to filter which proteins to process)")
    parser.add_argument("--fold", action="store_true", help="Enable ESM3 structure prediction")
    parser.add_argument("--fold-min-votes", type=int, default=4, help="Min votes to trigger folding")
    parser.add_argument("--fold-max-length", type=int, default=1024, help="Max sequence length to fold")
    parser.add_argument("--no-images", action="store_true", help="Skip APC image generation")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    cache_dir = base_dir / "cache"
    apc_dir = cache_dir / "apc"
    structures_dir = cache_dir / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        # Legacy: output to viewer/data for backwards compatibility
        output_dir = base_dir / "viewer" / "data"

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"

    # Load sequences from the specified FASTA (for filtering and folding)
    sequences = None
    protein_ids = None

    if args.fasta and args.fasta.exists():
        print(f"Loading sequences from {args.fasta}...")
        sequences = load_sequences_from_fasta(args.fasta)
        protein_ids = set(sequences.keys())
        print(f"  Loaded {len(sequences)} sequences")
    elif args.fold:
        # Need sequences for folding - load from all sources
        print("Loading sequences...")
        sequences = load_sequences(base_dir)
        print(f"  Loaded {len(sequences)} sequences")

    # Initialize ESM3 if folding enabled
    if args.fold:
        api_key_file = base_dir / ".esm_api_key"
        if api_key_file.exists():
            api_key = api_key_file.read_text().strip()
            init_esm3_client(api_key)
        else:
            print("Warning: .esm_api_key not found, folding disabled")

    # Generate results JSON (with folding if enabled)
    generate_results_json(
        apc_dir, results_path,
        protein_ids=protein_ids,
        sequences=sequences,
        fold_min_votes=args.fold_min_votes,
        fold_max_length=args.fold_max_length,
        structures_dir=structures_dir
    )

    # Load results and generate images
    if not args.no_images:
        with open(results_path) as f:
            results = json.load(f)
        # APC images go into a shared location (viewer/apc_images) since they're protein-specific not run-specific
        generate_apc_images(apc_dir, base_dir / "viewer" / "apc_images", results)
