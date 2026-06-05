#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path

# Add project root to sys.path so we can import config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from training.config_phase1 import config
except ImportError:
    print("Error: Could not import training.config_phase1. Make sure to run this script from BCDA root directory.")
    sys.path.append(str(Path.cwd()))
    from training.config_phase1 import config


def run_command(cmd: list[str]) -> bool:
    print(f"Executing: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except FileNotFoundError:
        print(f"Error: command '{cmd[0]}' not found. Make sure Google Cloud SDK (gcloud CLI) is installed and added to PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize checkpoints, logs, and outputs from Google Cloud Storage (GCS) to local machine.")
    parser.add_argument("--bucket", type=str, default=None, help="Name of the GCS bucket. Defaults to the one configured in config_phase1.py.")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run of gsutil rsync.")
    parser.add_argument("--direction", type=str, choices=["down", "up", "both"], default="down", 
                        help="Sync direction: 'down' (GCS -> local), 'up' (local -> GCS), 'both' (both directions).")
    args = parser.parse_args()

    bucket = args.bucket or config.runtime.gcs_bucket
    print("=" * 60)
    print("[SYNC] GCS SYNC TOOL")
    print(f"Bucket: gs://{bucket}")
    print(f"Direction: {args.direction}")
    print("=" * 60)

    # Sync folders definition: (remote_folder, local_folder)
    sync_folders = [
        ("checkpoints/phase1", config.paths.checkpoints_dir),
        ("logs/phase1", config.paths.logs_dir),
        ("outputs/phase1", config.paths.outputs_dir)
    ]

    for remote_folder, local_path in sync_folders:
        local_path = Path(local_path)
        local_path.mkdir(parents=True, exist_ok=True)
        
        gcs_uri = f"gs://{bucket}/{remote_folder}"
        
        # Build gsutil rsync command
        # -m: multi-threaded sync
        # -r: recursive
        base_cmd = ["gsutil", "-m", "rsync", "-r"]
        if args.dry_run:
            base_cmd.append("-n")

        if args.direction == "down" or args.direction == "both":
            print(f"\n[SYNC] GCS -> local: {remote_folder} -> {local_path}")
            cmd = base_cmd + [gcs_uri, str(local_path)]
            run_command(cmd)

        if args.direction == "up" or args.direction == "both":
            print(f"\n[SYNC] local -> GCS: {local_path} -> {remote_folder}")
            cmd = base_cmd + [str(local_path), gcs_uri]
            run_command(cmd)

    print("\n[SYNC] Sync process completed.")


if __name__ == "__main__":
    main()
