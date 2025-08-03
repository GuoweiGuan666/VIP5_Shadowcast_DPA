#!/usr/bin/env python3
"""compare_pth.py - Compare two PyTorch .pth files.

Usage:
    python compare_pth.py --fileA path/to/first.pth --fileB path/to/second.pth

Exit codes:
    0 - No differences found.
    1 - Differences found.
    2 - Error occurred (e.g., missing file, load failure).

This script loads both .pth files using ``torch.load`` (mapping to CPU),
falls back to ``state_dict`` when possible, and performs a
key-by-key comparison of tensors or values. It prints a summary of any
mismatches to the console.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

import torch


def load_pth(path: str) -> Any:
    """Load a .pth file and return its state dict if available."""
    try:
        obj = torch.load(path, map_location="cpu")
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(2)
    except Exception as err:  # pylint: disable=broad-except
        print(f"Error: failed to load '{path}': {err}", file=sys.stderr)
        sys.exit(2)

    if hasattr(obj, "state_dict"):
        try:
            return obj.state_dict()
        except Exception:  # pragma: no cover - best effort
            pass
    return obj


def compare(a: Any, b: Any, prefix: str = "") -> List[str]:
    """Recursively compare two PyTorch objects and collect differences."""
    diffs: List[str] = []

    if isinstance(a, dict) and isinstance(b, dict):
        keys_a = set(a.keys())
        keys_b = set(b.keys())
        for k in sorted(keys_a - keys_b):
            diffs.append(f"{prefix}{k} only in fileA")
        for k in sorted(keys_b - keys_a):
            diffs.append(f"{prefix}{k} only in fileB")
        for k in sorted(keys_a & keys_b):
            diffs.extend(compare(a[k], b[k], prefix + k + "."))
        return diffs

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        name = prefix[:-1] if prefix else "root"
        if a.shape != b.shape:
            diffs.append(f"{name}: shape differs {tuple(a.shape)} != {tuple(b.shape)}")
        elif not torch.allclose(a, b):
            max_diff = (a - b).abs().max().item()
            diffs.append(f"{name}: values differ (max abs diff {max_diff})")
        return diffs

    # Fallback to direct comparison
    name = prefix[:-1] if prefix else "root"
    if a != b:
        diffs.append(f"{name}: values differ ({a} != {b})")
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two PyTorch .pth files")
    parser.add_argument("--fileA", required=True, help="Path to first .pth file")
    parser.add_argument("--fileB", required=True, help="Path to second .pth file")
    args = parser.parse_args()

    for label, path in [("fileA", args.fileA), ("fileB", args.fileB)]:
        if not os.path.exists(path):
            print(f"Error: {label} '{path}' does not exist", file=sys.stderr)
            sys.exit(2)

    state_a = load_pth(args.fileA)
    state_b = load_pth(args.fileB)

    diffs = compare(state_a, state_b)

    if not diffs:
        print("No differences found.")
        sys.exit(0)

    print("Differences detected:")
    for d in diffs:
        print(f" - {d}")
    sys.exit(1)


if __name__ == "__main__":
    main()