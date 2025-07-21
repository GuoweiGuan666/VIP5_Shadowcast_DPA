#!/usr/bin/env python3
"""read_pkl.py - Load and preview multiple pickle files.

Example:
    python read_pkl.py path/to/file1.pkl path/to/file2.pkl

For each file this tool prints a short summary of the loaded object. If the
object resembles experiment splits (dict with train/val/test), the first few
records of the ``test`` split are displayed in detail.
"""
import argparse
import os
import pickle
from typing import Any, Iterable


def preview_data(data: Any) -> None:
    """Print a human‑readable summary of ``data``."""
    if isinstance(data, dict) and any(k in data for k in ("train", "val", "test")):
        # treat as experiment splits
        for split in ("train", "val", "test"):
            if split in data:
                print(f"{split}: {len(data[split])} records")
        sample = data.get("test", [])[:3]
        if sample:
            print("\n--- Test sample (first 3 records) ---")
            for i, rec in enumerate(sample):
                print(f"[{i}]")
                if isinstance(rec, dict):
                    for k, v in rec.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"  {rec}")
        return

    if isinstance(data, dict):
        print(f"Dict with {len(data)} keys: {list(data)[:5]}")
        return

    if isinstance(data, (list, tuple, set)):
        seq = list(data)
        print(f"{type(data).__name__} of length {len(seq)}")
        if seq:
            print("First items:", seq[:5])
        return

    print(repr(data))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview contents of pickle files")
    parser.add_argument("paths", nargs="+", help="Paths to .pkl files")
    args = parser.parse_args()

    for path in args.paths:
        print(f"\n=== {path} ===")
        if not os.path.isfile(path):
            print(f"[ERROR] File not found: {path}")
            continue
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            print(f"[ERROR] failed to load {path}: {exc}")
            continue
        preview_data(data)


if __name__ == "__main__":
    main()
