#!/usr/bin/env python3
"""
Download an Archaeon proteome from UniProt and filter to proteins with AlphaFold structures.
"""

import requests
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Proteome IDs for common Archaea
ARCHAEA_PROTEOMES = {
    'sulfolobus_acidocaldarius': 'UP000001018',  # ~2,200 proteins
    'methanocaldococcus_jannaschii': 'UP000000805',  # ~1,800 proteins
    'thermococcus_kodakarensis': 'UP000000536',  # ~2,300 proteins
    'haloferax_volcanii': 'UP000008243',  # ~4,000 proteins
    'pyrococcus_furiosus': 'UP000001013',  # ~2,000 proteins
}


def download_proteome_ids(proteome_id: str) -> List[str]:
    """Download all UniProt IDs for a proteome."""
    print(f"Downloading protein IDs for proteome {proteome_id}...")

    url = f"https://rest.uniprot.org/uniprotkb/stream?query=(proteome:{proteome_id})&format=list"
    response = requests.get(url)
    response.raise_for_status()

    protein_ids = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
    print(f"  Found {len(protein_ids)} proteins")
    return protein_ids


def download_protein_sequences(proteome_id: str) -> Dict[str, str]:
    """Download all protein sequences for a proteome as a dict of id -> sequence."""
    print(f"Downloading sequences for proteome {proteome_id}...")

    url = f"https://rest.uniprot.org/uniprotkb/stream?query=(proteome:{proteome_id})&format=fasta"
    response = requests.get(url)
    response.raise_for_status()

    sequences = {}
    current_id = None
    current_seq = []

    for line in response.text.split('\n'):
        if line.startswith('>'):
            if current_id:
                sequences[current_id] = ''.join(current_seq)
            # Parse UniProt ID from header like ">sp|Q9UY87|..."
            parts = line[1:].split('|')
            if len(parts) >= 2:
                current_id = parts[1]
            else:
                current_id = line[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line.strip())

    if current_id:
        sequences[current_id] = ''.join(current_seq)

    print(f"  Downloaded {len(sequences)} sequences")
    return sequences


def check_alphafold_availability(protein_ids: List[str], batch_size: int = 50) -> List[str]:
    """
    Check which proteins have AlphaFold structures available.
    Uses batch requests to be efficient.
    """
    print(f"Checking AlphaFold availability for {len(protein_ids)} proteins...")

    available = []

    for i in range(0, len(protein_ids), batch_size):
        batch = protein_ids[i:i + batch_size]

        for protein_id in batch:
            url = f"https://alphafold.ebi.ac.uk/api/prediction/{protein_id}"
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    available.append(protein_id)
            except requests.exceptions.RequestException:
                pass  # Skip on error

        # Progress update
        checked = min(i + batch_size, len(protein_ids))
        print(f"  Checked {checked}/{len(protein_ids)}, found {len(available)} with structures", end='\r')

        # Rate limiting
        time.sleep(0.1)

    print(f"\n  {len(available)}/{len(protein_ids)} proteins have AlphaFold structures")
    return available


def save_fasta(sequences: Dict[str, str], output_path: Path):
    """Save sequences to FASTA file."""
    with open(output_path, 'w') as f:
        for protein_id, seq in sequences.items():
            f.write(f">{protein_id}\n")
            # Write sequence in lines of 80 characters
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + '\n')
    print(f"Saved {len(sequences)} sequences to {output_path}")


def save_id_list(protein_ids: List[str], output_path: Path):
    """Save protein IDs to a text file."""
    with open(output_path, 'w') as f:
        for pid in protein_ids:
            f.write(pid + '\n')
    print(f"Saved {len(protein_ids)} IDs to {output_path}")


def main(organism: str = 'sulfolobus_acidocaldarius', output_dir: str = None):
    """
    Main function to download and filter a proteome.

    Args:
        organism: Key from ARCHAEA_PROTEOMES dict
        output_dir: Directory to save output files (default: ./data/{organism}/)
    """
    if organism not in ARCHAEA_PROTEOMES:
        print(f"Unknown organism: {organism}")
        print(f"Available: {list(ARCHAEA_PROTEOMES.keys())}")
        return

    proteome_id = ARCHAEA_PROTEOMES[organism]

    # Set up output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / 'data' / organism
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Downloading {organism} proteome ({proteome_id}) ===\n")

    # Step 1: Download all sequences
    sequences = download_protein_sequences(proteome_id)
    all_ids = list(sequences.keys())

    # Save all sequences
    save_fasta(sequences, output_dir / f'{organism}_all.fasta')
    save_id_list(all_ids, output_dir / f'{organism}_all_ids.txt')

    # Step 2: Check AlphaFold availability
    print()
    af_available = check_alphafold_availability(all_ids)

    # Save filtered sequences (only those with AlphaFold structures)
    af_sequences = {pid: sequences[pid] for pid in af_available if pid in sequences}
    save_fasta(af_sequences, output_dir / f'{organism}_alphafold.fasta')
    save_id_list(af_available, output_dir / f'{organism}_alphafold_ids.txt')

    # Step 3: Summary statistics
    print(f"\n=== Summary ===")
    print(f"Organism: {organism}")
    print(f"Proteome ID: {proteome_id}")
    print(f"Total proteins: {len(all_ids)}")
    print(f"With AlphaFold structures: {len(af_available)} ({100*len(af_available)/len(all_ids):.1f}%)")
    print(f"\nOutput files in: {output_dir}")
    print(f"  - {organism}_all.fasta ({len(sequences)} sequences)")
    print(f"  - {organism}_alphafold.fasta ({len(af_sequences)} sequences)")
    print(f"  - {organism}_alphafold_ids.txt (for ESM++ processing)")

    # Length distribution
    lengths = [len(s) for s in af_sequences.values()]
    if lengths:
        print(f"\nSequence length stats (AlphaFold set):")
        print(f"  Min: {min(lengths)}, Max: {max(lengths)}, Median: {sorted(lengths)[len(lengths)//2]}")

    return af_sequences


if __name__ == '__main__':
    organism = sys.argv[1] if len(sys.argv) > 1 else 'sulfolobus_acidocaldarius'
    main(organism)
