#!/usr/bin/env python3
"""
Run ESMFold structure prediction on detected solenoid proteins.

This script predicts structures for proteins that may not have AlphaFold coverage,
allowing visualization of novel solenoid discoveries.

Usage:
    # Using ESMFold API (no GPU required, but rate-limited)
    python run_esmfold.py --mode api

    # Using local ESMFold model (requires GPU with ~16GB VRAM)
    python run_esmfold.py --mode local --device cuda

    # Only predict structures for high-confidence solenoids
    python run_esmfold.py --min-votes 4

    # Predict for specific proteins
    python run_esmfold.py --proteins D4GQW4 D4GPD9
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests

# Optional imports for local inference
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False


def load_sequences_from_fasta(fasta_dir: Path) -> Dict[str, str]:
    """Load all sequences from FASTA files in data directory."""
    sequences = {}

    for fasta_file in list(fasta_dir.rglob("*.fasta")) + list(fasta_dir.rglob("*.faa")):
        with open(fasta_file) as f:
            current_id = None
            current_seq = []

            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id:
                        seq = ''.join(current_seq).replace('*', '')
                        sequences[current_id] = seq
                    # Parse ID
                    header = line[1:]
                    if header.startswith(('sp|', 'tr|')):
                        # UniProt format: >sp|Q9UY87|... or >tr|A0A123|...
                        parts = header.split('|')
                        current_id = parts[1] if len(parts) >= 2 else parts[0]
                    else:
                        # Generic format: use first whitespace-delimited token
                        # Also split on | to handle Prodigal/metagenomic formats
                        current_id = header.split()[0].split('|')[0]
                    current_seq = []
                else:
                    current_seq.append(line)

            if current_id:
                seq = ''.join(current_seq).replace('*', '')
                sequences[current_id] = seq

    return sequences


def get_solenoid_proteins(results_path: Path, min_votes: int = 0) -> List[str]:
    """Get list of protein IDs with detected solenoids."""
    with open(results_path) as f:
        results = json.load(f)

    solenoid_ids = []
    for protein_id, data in results.items():
        if not data.get('has_solenoid'):
            continue

        # Check if any region meets vote threshold
        for region in data.get('regions', []):
            if region.get('votes', 0) >= min_votes:
                solenoid_ids.append(protein_id)
                break

    return solenoid_ids


def predict_structure_api(sequence: str, protein_id: str, timeout: int = 300) -> Optional[str]:
    """
    Predict structure using ESMFold API.

    Returns PDB string or None on failure.
    """
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"

    try:
        response = requests.post(
            url,
            data=sequence,
            headers={"Content-Type": "text/plain"},
            timeout=timeout
        )

        if response.status_code == 200:
            return response.text
        else:
            print(f"  API error for {protein_id}: {response.status_code}")
            return None

    except requests.exceptions.Timeout:
        print(f"  Timeout for {protein_id} (>{timeout}s)")
        return None
    except Exception as e:
        print(f"  Error for {protein_id}: {e}")
        return None


def predict_structure_local(
    model,
    sequence: str,
    protein_id: str,
    device: str = "cuda"
) -> Optional[str]:
    """
    Predict structure using local ESMFold model.

    Returns PDB string or None on failure.
    """
    try:
        with torch.no_grad():
            output = model.infer_pdb(sequence)
        return output
    except Exception as e:
        print(f"  Error for {protein_id}: {e}")
        return None


def load_local_model(device: str = "cuda"):
    """Load ESMFold model for local inference."""
    if not ESM_AVAILABLE:
        raise ImportError("esm package not installed. Install with: pip install fair-esm")
    if not TORCH_AVAILABLE:
        raise ImportError("torch not installed")

    print("Loading ESMFold model (this may take a few minutes)...")
    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    model = model.to(device)

    # Set chunk size for memory efficiency
    model.set_chunk_size(128)

    print(f"ESMFold model loaded on {device}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Run ESMFold on detected solenoid proteins"
    )
    parser.add_argument(
        "--mode", choices=["api", "local"], default="api",
        help="Prediction mode: 'api' uses ESMFold API, 'local' uses local model"
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for local inference (cuda, mps, cpu)"
    )
    parser.add_argument(
        "--min-votes", type=int, default=0,
        help="Minimum votes threshold for solenoid regions"
    )
    parser.add_argument(
        "--max-length", type=int, default=800,
        help="Maximum sequence length to predict (longer = more memory)"
    )
    parser.add_argument(
        "--proteins", nargs="+", default=None,
        help="Specific protein IDs to predict (default: all solenoids)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory for PDB files (default: viewer/structures/)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip proteins that already have structures"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Delay between API requests (seconds)"
    )
    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent
    results_path = base_dir / "viewer" / "data" / "results.json"
    data_dir = base_dir / "data"
    output_dir = args.output if args.output else base_dir / "viewer" / "structures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sequences from data/ and proteins/ folders
    print("Loading sequences from FASTA files...")
    sequences = load_sequences_from_fasta(data_dir)

    # Also load from proteins/ folder
    proteins_dir = base_dir / "proteins"
    if proteins_dir.exists():
        proteins_seqs = load_sequences_from_fasta(proteins_dir)
        sequences.update(proteins_seqs)

    print(f"  Loaded {len(sequences)} sequences")

    # Get proteins to predict
    if args.proteins:
        protein_ids = args.proteins
    else:
        protein_ids = get_solenoid_proteins(results_path, args.min_votes)

    print(f"  {len(protein_ids)} solenoid proteins to process")

    # Filter by sequence availability and length
    to_predict = []
    for pid in protein_ids:
        if pid not in sequences:
            print(f"  Warning: no sequence for {pid}")
            continue

        seq = sequences[pid]
        if len(seq) > args.max_length:
            print(f"  Skipping {pid}: too long ({len(seq)} > {args.max_length})")
            continue

        # Check if already done
        if args.resume and (output_dir / f"{pid}.pdb").exists():
            continue

        to_predict.append((pid, seq))

    print(f"  {len(to_predict)} proteins to predict")

    if not to_predict:
        print("Nothing to predict!")
        return

    # Load model for local inference
    model = None
    if args.mode == "local":
        model = load_local_model(args.device)

    # Predict structures
    print(f"\nPredicting structures using {args.mode} mode...")

    success = 0
    failed = 0

    for i, (protein_id, sequence) in enumerate(to_predict):
        print(f"[{i+1}/{len(to_predict)}] {protein_id} ({len(sequence)} aa)...", end=" ", flush=True)

        if args.mode == "api":
            pdb_str = predict_structure_api(sequence, protein_id)
            if args.delay > 0 and i < len(to_predict) - 1:
                time.sleep(args.delay)
        else:
            pdb_str = predict_structure_local(model, sequence, protein_id, args.device)

        if pdb_str:
            # Save PDB file
            pdb_path = output_dir / f"{protein_id}.pdb"
            with open(pdb_path, "w") as f:
                f.write(pdb_str)
            print("OK")
            success += 1
        else:
            print("FAILED")
            failed += 1

    print(f"\nDone! Success: {success}, Failed: {failed}")
    print(f"Structures saved to: {output_dir}")

    # Update a manifest file for the viewer
    manifest_path = output_dir / "manifest.json"
    existing_structures = [f.stem for f in output_dir.glob("*.pdb")]
    with open(manifest_path, "w") as f:
        json.dump({"structures": sorted(existing_structures)}, f, indent=2)
    print(f"Manifest updated: {len(existing_structures)} structures")


if __name__ == "__main__":
    main()
