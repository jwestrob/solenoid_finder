#!/usr/bin/env python3
"""
Run ESM++ on a proteome to generate APC attention matrices for solenoid detection.

This is a simplified wrapper around the ESM++ pipeline for processing new proteomes.

Usage:
    python run_esmpp_proteome.py --fasta data/sulfolobus_acidocaldarius/sulfolobus_acidocaldarius_alphafold.fasta --device mps

The output APC matrices will be saved to cache/apc/{protein_id}.npy
for use with generate_viewer_data.py
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# Set thread environment before numpy import
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import numpy as np
import torch
from tqdm import tqdm

try:
    from transformers import AutoModelForMaskedLM, AutoTokenizer
except ImportError:
    raise ImportError("transformers required: pip install transformers")


def resolve_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    s = dtype_str.lower().strip()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32


def load_esmpp(model_id: str, device: torch.device, dtype: torch.dtype):
    """Load ESM++ model and tokenizer."""
    print(f"Loading ESM++ model: {model_id}")
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
    print(f"Model loaded on {device}")
    return model, tok


def get_residue_indices(tokenized: dict) -> torch.Tensor:
    """Get indices of residue tokens (excluding special tokens)."""
    mask = tokenized.get("special_tokens_mask")
    if mask is None:
        seq_len = tokenized["input_ids"].shape[1]
        return torch.arange(1, seq_len - 1)
    keep = (mask[0] == 0)
    return torch.nonzero(keep, as_tuple=False).squeeze(-1)


@torch.inference_mode()
def compute_apc_attention(
    model,
    tokenizer,
    sequence: str,
    device: torch.device,
    last_layers: int = 8,
    min_sep: int = 6,
) -> np.ndarray:
    """
    Compute APC-corrected attention map for a single sequence.

    Returns:
        (L, L) numpy array - the "contactiness" matrix
    """
    # Tokenize
    tokenized = tokenizer(
        [sequence],
        return_tensors="pt",
        padding=False,
        return_special_tokens_mask=True,
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items() if isinstance(v, torch.Tensor)}
    residue_idx = get_residue_indices(tokenized)

    # Forward pass
    out = model(
        **tokenized,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=True,
    )

    # Process attention from last N layers
    attns = out.attentions
    n_layers = len(attns)
    use_layers = list(range(max(0, n_layers - last_layers), n_layers))

    W_acc = None
    n_acc = 0

    for li in use_layers:
        A = attns[li][0]  # (H, Ltok, Ltok)
        # Symmetrize
        A = 0.5 * (A + A.transpose(-1, -2))
        # Select residue tokens only
        A = A.index_select(1, residue_idx).index_select(2, residue_idx)

        # APC correction per head
        row = A.sum(dim=-1, keepdim=True)
        col = A.sum(dim=-2, keepdim=True)
        tot = A.sum(dim=(-1, -2), keepdim=True) + 1e-12
        A = A - (row * col) / tot
        A = torch.relu(A)

        # Average over heads
        A_layer = A.mean(dim=0)
        if W_acc is None:
            W_acc = A_layer
        else:
            W_acc = W_acc + A_layer
        n_acc += 1

    W = W_acc / max(1, n_acc)

    # Zero out diagonal and near-diagonal
    Lres = W.shape[0]
    if min_sep > 0 and Lres > 0:
        idx = torch.arange(Lres, device=device)
        dist = (idx[:, None] - idx[None, :]).abs()
        W = W.masked_fill(dist < min_sep, 0.0)
    if Lres > 0:
        W.fill_diagonal_(0.0)

    return W.cpu().float().numpy()


def load_fasta(fasta_path: Path) -> Dict[str, str]:
    """Load sequences from FASTA file."""
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    seq = ''.join(current_seq).replace('*', '')  # Remove stop codons
                    sequences[current_id] = seq
                # Parse ID - handle various formats
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


def main():
    parser = argparse.ArgumentParser(description="Run ESM++ on a proteome for solenoid detection")
    parser.add_argument("--fasta", type=Path, required=True,
                        help="Input FASTA file with protein sequences")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: cache/apc)")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device: cuda, mps, or cpu")
    parser.add_argument("--dtype", type=str, default="fp32",
                        help="Model dtype: fp16, bf16, fp32 (fp32 recommended for mps)")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum sequence length to process")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of proteins to process")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already processed proteins")
    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    output_dir = args.output if args.output else base_dir / 'cache' / 'apc'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sequences
    print(f"\nLoading sequences from {args.fasta}")
    sequences = load_fasta(args.fasta)
    print(f"Loaded {len(sequences)} sequences")

    # Filter by length
    sequences = {k: v for k, v in sequences.items() if len(v) <= args.max_length}
    print(f"{len(sequences)} sequences after length filter (<= {args.max_length})")

    # Check what's already done
    if args.resume:
        done = {f.stem for f in output_dir.glob('*.npy')}
        sequences = {k: v for k, v in sequences.items() if k not in done}
        print(f"{len(sequences)} sequences after skipping already processed")

    # Apply limit
    if args.limit:
        protein_ids = list(sequences.keys())[:args.limit]
        sequences = {k: sequences[k] for k in protein_ids}
        print(f"Limited to {len(sequences)} sequences")

    if not sequences:
        print("Nothing to process!")
        return

    # Load model
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    model, tokenizer = load_esmpp("Synthyra/ESMplusplus_large", device, dtype)

    # Process proteins
    print(f"\nProcessing {len(sequences)} proteins...")
    start_time = time.time()

    processed = 0
    errors = 0

    pbar = tqdm(sequences.items(), desc="Processing", unit="protein")
    for protein_id, seq in pbar:
        try:
            apc = compute_apc_attention(model, tokenizer, seq, device)
            np.save(output_dir / f'{protein_id}.npy', apc)
            processed += 1

            # Update progress
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            pbar.set_postfix({
                'done': processed,
                'rate': f'{rate:.2f}/s',
                'errors': errors,
            })

        except Exception as e:
            errors += 1
            print(f"\nError processing {protein_id}: {e}")
            continue

        # Periodic cleanup
        if processed % 25 == 0:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                torch.mps.empty_cache()

    elapsed = time.time() - start_time
    print(f"\nDone! Processed {processed} proteins in {elapsed:.1f}s")
    print(f"Errors: {errors}")
    print(f"Output: {output_dir}")
    print(f"\nNext step: Run 'python generate_viewer_data.py' to generate the viewer data")


if __name__ == '__main__':
    main()
