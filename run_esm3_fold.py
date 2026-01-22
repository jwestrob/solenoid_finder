#!/usr/bin/env python3
"""
Run ESM3 structure prediction on detected solenoid proteins using EvolutionaryScale Forge API.

Usage:
    python run_esm3_fold.py --min-votes 4
    python run_esm3_fold.py --proteins Green_Borg_1007 Green_Borg_463
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from esm.sdk.api import ESMProtein, ESMProteinError, GenerationConfig
from esm.sdk.forge import ESM3ForgeInferenceClient


def load_api_key(key_file: Path) -> str:
    """Load API key from file."""
    with open(key_file) as f:
        return f.read().strip()


def load_sequences_from_fasta(fasta_dir: Path) -> Dict[str, str]:
    """Load all sequences from FASTA files."""
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
        for region in data.get('regions', []):
            if region.get('votes', 0) >= min_votes:
                solenoid_ids.append(protein_id)
                break

    return solenoid_ids


def predict_structure(client: ESM3ForgeInferenceClient, sequence: str, protein_id: str) -> Optional[str]:
    """Predict structure using ESM3 Forge API."""
    try:
        # Create protein from sequence
        protein = ESMProtein(sequence=sequence)

        # Generate structure (8 steps is fast, increase for better quality)
        config = GenerationConfig(track="structure", num_steps=8)
        folded = client.generate(protein, config)

        if folded.coordinates is None:
            print(f"  No coordinates generated for {protein_id}")
            return None

        # Convert to PDB
        pdb_str = folded.to_pdb_string()
        return pdb_str

    except ESMProteinError as e:
        print(f"  ESM3 error for {protein_id}: {e}")
        return None
    except Exception as e:
        print(f"  Error for {protein_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run ESM3 folding on solenoid proteins")
    parser.add_argument("--min-votes", type=int, default=4, help="Minimum votes threshold")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--proteins", nargs="+", default=None, help="Specific protein IDs")
    parser.add_argument("--resume", action="store_true", help="Skip existing structures")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of proteins")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    key_file = base_dir / ".esm_api_key"
    results_path = base_dir / "viewer" / "data" / "results.json"
    output_dir = base_dir / "viewer" / "structures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load API key
    if not key_file.exists():
        print("Error: .esm_api_key not found")
        return
    api_key = load_api_key(key_file)

    # Initialize client
    print("Initializing ESM3 Forge client...")
    client = ESM3ForgeInferenceClient(model="esm3-open-2024-03", token=api_key)

    # Load sequences
    print("Loading sequences...")
    sequences = {}
    for d in [base_dir / "data", base_dir / "proteins"]:
        if d.exists():
            sequences.update(load_sequences_from_fasta(d))
    print(f"  Loaded {len(sequences)} sequences")

    # Get proteins to predict
    if args.proteins:
        protein_ids = args.proteins
    else:
        protein_ids = get_solenoid_proteins(results_path, args.min_votes)
    print(f"  {len(protein_ids)} solenoid proteins with {args.min_votes}+ votes")

    # Filter and prepare
    to_predict = []
    for pid in protein_ids:
        if pid not in sequences:
            continue
        seq = sequences[pid]
        if len(seq) > args.max_length:
            continue
        if args.resume and (output_dir / f"{pid}.pdb").exists():
            continue
        to_predict.append((pid, seq))

    if args.limit:
        to_predict = to_predict[:args.limit]

    print(f"  {len(to_predict)} proteins to predict\n")

    if not to_predict:
        print("Nothing to predict!")
        return

    # Predict structures
    success = 0
    failed = 0

    for i, (protein_id, sequence) in enumerate(to_predict):
        print(f"[{i+1}/{len(to_predict)}] {protein_id} ({len(sequence)} aa)...", end=" ", flush=True)

        pdb_str = predict_structure(client, sequence, protein_id)

        if pdb_str:
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

    # Update manifest
    manifest_path = output_dir / "manifest.json"
    existing = [f.stem for f in output_dir.glob("*.pdb")]
    with open(manifest_path, "w") as f:
        json.dump({"structures": sorted(existing)}, f, indent=2)


if __name__ == "__main__":
    main()
