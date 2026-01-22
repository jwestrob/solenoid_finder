#!/usr/bin/env python3
"""
Solenoid Detection Pipeline

Run the complete solenoid detection workflow on a set of protein sequences:
1. ESM++ inference → APC attention matrices (cached globally)
2. Solenoid detection → results.json
3. (Optional) ESM3 structure prediction for detected solenoids
4. (Optional) Launch viewer

Usage:
    # Basic usage - run ESM++ and detection
    python run_pipeline.py --fasta proteins/my_proteins.fasta --name my_analysis

    # With structure prediction for high-confidence hits
    python run_pipeline.py --fasta proteins/Borg.faa --name borg_jan22 --fold

    # Full pipeline with viewer launch
    python run_pipeline.py --fasta data/proteome.fasta --name test_run --fold --launch-viewer

    # Resume interrupted run
    python run_pipeline.py --fasta proteins/my_proteins.fasta --name my_analysis --resume
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    return result.returncode == 0


def count_results(results_path: Path) -> dict:
    """Count solenoid detection results."""
    if not results_path.exists():
        return {"total": 0, "with_solenoid": 0, "high_conf": 0}

    with open(results_path) as f:
        results = json.load(f)

    total = len(results)
    with_solenoid = sum(1 for r in results.values() if r.get("has_solenoid"))

    # Count high-confidence (4+ votes)
    high_conf = 0
    for data in results.values():
        for region in data.get("regions", []):
            if region.get("votes", 0) >= 4:
                high_conf += 1
                break

    return {"total": total, "with_solenoid": with_solenoid, "high_conf": high_conf}


def update_runs_index(base_dir: Path, run_name: str, fasta_path: Path, stats: dict):
    """Update the runs/index.json with run metadata."""
    runs_dir = base_dir / "runs"
    index_path = runs_dir / "index.json"

    # Load existing index or create new
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
    else:
        index = {"runs": []}

    # Update or add this run
    run_entry = {
        "id": run_name,
        "name": run_name.replace("_", " ").title(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "fasta": str(fasta_path),
        "n_proteins": stats["total"],
        "n_solenoids": stats["with_solenoid"],
        "n_high_conf": stats["high_conf"],
    }

    # Replace existing entry or append
    existing_idx = next((i for i, r in enumerate(index["runs"]) if r["id"] == run_name), None)
    if existing_idx is not None:
        index["runs"][existing_idx] = run_entry
    else:
        index["runs"].append(run_entry)

    # Sort by date descending
    index["runs"].sort(key=lambda x: x["date"], reverse=True)

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"Updated runs index: {index_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete solenoid detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --fasta proteins/test.fasta --name test_run
  python run_pipeline.py --fasta proteins/Borg.faa --name borg_jan22 --fold
  python run_pipeline.py --fasta data/organism/proteome.fasta --name organism_analysis --fold --launch-viewer
        """
    )

    # Required arguments
    parser.add_argument(
        "--fasta", type=Path, required=True,
        help="Input FASTA file with protein sequences"
    )
    parser.add_argument(
        "--name", type=str, required=True,
        help="Name for this run (e.g., 'borg_jan22', 'sulfolobus_test')"
    )

    # ESM++ options
    parser.add_argument(
        "--device", type=str, default="mps",
        help="Compute device: cuda, mps, or cpu (default: mps)"
    )
    parser.add_argument(
        "--dtype", type=str, default="fp32",
        help="Model dtype: fp16, bf16, fp32 (default: fp32, recommended for mps)"
    )
    parser.add_argument(
        "--max-length", type=int, default=1024,
        help="Maximum sequence length to process (default: 1024)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from previous run (skip already processed proteins)"
    )

    # Structure prediction options
    parser.add_argument(
        "--fold", action="store_true",
        help="Enable ESM3 structure prediction during detection (recommended)"
    )
    parser.add_argument(
        "--min-votes", type=int, default=4,
        help="Minimum votes to trigger structure prediction (default: 4)"
    )
    parser.add_argument(
        "--fold-max-length", type=int, default=1024,
        help="Maximum sequence length to fold (default: 1024)"
    )

    # Viewer options
    parser.add_argument(
        "--launch-viewer", action="store_true",
        help="Launch the web viewer after pipeline completes"
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Port for viewer web server (default: 8080)"
    )

    # Pipeline control
    parser.add_argument(
        "--skip-esmpp", action="store_true",
        help="Skip ESM++ inference (use existing APC matrices)"
    )
    parser.add_argument(
        "--skip-detection", action="store_true",
        help="Skip detection (use existing results.json)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.fasta.exists():
        print(f"Error: FASTA file not found: {args.fasta}")
        sys.exit(1)

    # Sanitize run name
    run_name = args.name.replace(" ", "_").replace("/", "_")

    base_dir = Path(__file__).parent
    runs_dir = base_dir / "runs"
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.json"

    start_time = time.time()
    print("\n" + "="*60)
    print("  SOLENOID DETECTION PIPELINE")
    print("="*60)
    print(f"\nRun name: {run_name}")
    print(f"Input: {args.fasta}")
    print(f"Output: {run_dir}")
    print(f"Device: {args.device}")

    # Step 1: ESM++ inference (outputs to global cache)
    if not args.skip_esmpp:
        cmd = [
            sys.executable, str(base_dir / "scripts" / "run_esmpp_proteome.py"),
            "--fasta", str(args.fasta),
            "--device", args.device,
            "--dtype", args.dtype,
            "--max-length", str(args.max_length),
        ]
        if args.resume:
            cmd.append("--resume")

        if not run_command(cmd, "Step 1/2: Running ESM++ inference"):
            print("\nError: ESM++ inference failed")
            sys.exit(1)
    else:
        print("\n[Skipping ESM++ inference]")

    # Step 2: Solenoid detection (outputs to run directory)
    if not args.skip_detection:
        cmd = [
            sys.executable, str(base_dir / "scripts" / "generate_viewer_data.py"),
            "--output", str(run_dir),
            "--fasta", str(args.fasta),
        ]

        if args.fold:
            cmd.extend([
                "--fold",
                "--fold-min-votes", str(args.min_votes),
                "--fold-max-length", str(args.fold_max_length),
            ])

        step_name = "Step 2/2: Running solenoid detection" + (" + ESM3 folding" if args.fold else "")
        if not run_command(cmd, step_name):
            print("\nError: Solenoid detection failed")
            sys.exit(1)
    else:
        print("\n[Skipping detection]")

    # Print detection summary
    stats = count_results(results_path)
    print("\n" + "-"*40)
    print("  Detection Summary")
    print("-"*40)
    print(f"  Total proteins processed: {stats['total']}")
    print(f"  Proteins with solenoids:  {stats['with_solenoid']}")
    print(f"  High-confidence (4+ votes): {stats['high_conf']}")
    print("-"*40)

    # Update runs index
    update_runs_index(base_dir, run_name, args.fasta, stats)

    # Note: Structure prediction is integrated into detection step when --fold is used
    if not args.fold:
        print("\n[Structure prediction skipped - use --fold to enable ESM3 folding]")

    # Done
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"\nResults: {results_path}")

    if stats['with_solenoid'] > 0:
        print(f"\nDetected {stats['with_solenoid']} proteins with solenoid regions!")

    # Sync instructions
    print("\n" + "-"*40)
    print("  To sync results to local viewer:")
    print("-"*40)
    print(f"  rsync -av server:path/to/runs/ viewer/runs/")
    print(f"  rsync -av server:path/to/cache/structures/ viewer/structures/")

    # Launch viewer
    if args.launch_viewer:
        print(f"\nLaunching viewer at http://localhost:{args.port}")
        print("Press Ctrl+C to stop the server\n")

        viewer_dir = base_dir / "viewer"
        subprocess.run(
            [sys.executable, "-m", "http.server", str(args.port)],
            cwd=viewer_dir
        )
    else:
        print(f"\nTo view results:")
        print(f"  cd viewer && python -m http.server {args.port}")
        print(f"  Then open http://localhost:{args.port}")


if __name__ == "__main__":
    main()
