#!/usr/bin/env python3
"""
read_pkl.py - Load and preview multiple pickle files.

    
Example:
    read_pkl.py - Load and inspect pickle files.

For each file this tool prints a rich summary of the loaded object, attempting
to reveal container sizes, sample elements, NumPy array shapes, pandas table
previews and more. If the object resembles experiment splits (dict with
train/val/test), record counts and a sample of the ``test`` split are shown in
detail.

"""
import argparse
import os
import pickle
from typing import Any

try:  # optional dependencies
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:  # optional dependency
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

from pprint import pformat


def preview_data(data: Any, depth: int = 0, max_items: int = 10) -> None:
    """Recursively print a rich, human‑readable summary of ``data``.

    The function attempts to introspect common container and array types and
    prints as much information as possible without overwhelming the output.
    ``max_items`` controls how many entries are shown for containers at each
    level of the recursion.
    """

    indent = "  " * depth

    

    if isinstance(data, dict) and any(k in data for k in ("train", "val", "test")):
        print(f"{indent}Experiment splits:")
        for split in ("train", "val", "test"):
            if split in data:
                print(f"{indent}  {split}: {len(data[split])} records")
        sample = data.get("test", [])[:3]
        if sample:
            print(f"{indent}  --- Sample from 'test' split ---")
            for i, rec in enumerate(sample):
                print(f"{indent}    [{i}]")
                preview_data(rec, depth + 2, max_items)
        return

    if isinstance(data, dict):
        print(f"{indent}Dict with {len(data)} keys")
        for i, (k, v) in enumerate(data.items()):
            if i >= max_items:
                print(f"{indent}  ... ({len(data) - max_items} more keys)")
                break
            print(f"{indent}  {k!r} -> {type(v).__name__}")
            preview_data(v, depth + 1, max_items)
        return

    if isinstance(data, (list, tuple, set)):
        seq = list(data)
        print(f"{indent}{type(data).__name__} of length {len(seq)}")
        for i, item in enumerate(seq[:max_items]):
            print(f"{indent}  [{i}] {type(item).__name__}")
            preview_data(item, depth + 1, max_items)
        if len(seq) > max_items:
            print(f"{indent}  ... ({len(seq) - max_items} more items)")
        return

    if np is not None and isinstance(data, np.ndarray):
        print(
            f"{indent}ndarray shape={data.shape} dtype={data.dtype}"
        )
        with np.printoptions(edgeitems=max_items):
            print(indent + str(data))
        return

    if pd is not None:
        if isinstance(data, pd.DataFrame):
            print(f"{indent}DataFrame shape={data.shape}")
            print(indent + data.head(max_items).to_string())
            return
        if isinstance(data, pd.Series):
            print(f"{indent}Series shape={data.shape}")
            print(indent + data.head(max_items).to_string())
            return

    # Fallback: pretty-print representation
    text = pformat(data, width=80, compact=True)
    lines = text.splitlines() or [""]
    for line in lines:
        print(indent + line)


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

if __name__ == '__main__':
    main()
