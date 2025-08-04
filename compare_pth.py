#!/usr/bin/env python3
"""compare_pth.py - Compare any number of PyTorch ``.pth`` files.

Usage:
    python compare_pth.py path/to/file1.pth path/to/file2.pth [file3.pth ...]

Exit codes:
    0 - No differences found.
    1 - Differences found.
    2 - Error occurred (e.g., missing file, load failure).

This script loads all provided ``.pth`` files using ``torch.load`` (mapping to
CPU), falls back to ``state_dict`` when possible, and performs a key-by-key
comparison of tensors or values. It prints a summary of any mismatches to the
console and compares all file pairs.
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations
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


def compare(a: Any, b: Any, name_a: str, name_b: str, prefix: str = "") -> List[str]:
    """Recursively compare two PyTorch objects and collect differences."""
    diffs: List[str] = []

    if isinstance(a, dict) and isinstance(b, dict):
        keys_a = set(a.keys())
        keys_b = set(b.keys())
        for k in sorted(keys_a - keys_b):
            diffs.append(f"{prefix}{k} only in {name_a}")
        for k in sorted(keys_b - keys_a):
            diffs.append(f"{prefix}{k} only in {name_b}")
        for k in sorted(keys_a & keys_b):
            diffs.extend(compare(a[k], b[k], name_a, name_b, prefix + k + "."))
        return diffs

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        name = prefix[:-1] if prefix else "root"
        if a.shape != b.shape:
            diffs.append(
                f"{name}: shape differs {tuple(a.shape)} in {name_a} != {tuple(b.shape)} in {name_b}"
            )
        elif not torch.allclose(a, b):
            max_diff = (a - b).abs().max().item()
            diffs.append(
                f"{name}: values differ between {name_a} and {name_b} (max abs diff {max_diff})"
            )
        return diffs

    # Fallback to direct comparison
    name = prefix[:-1] if prefix else "root"
    if a != b:
        diffs.append(f"{name}: values differ in {name_a} ({a}) vs {name_b} ({b})")
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple PyTorch .pth files")
    parser.add_argument(
        "files", nargs="+", help="Paths to .pth files to compare (at least two)"
    )
    args = parser.parse_args()

    if len(args.files) < 2:
        print("Error: provide at least two .pth files", file=sys.stderr)
        sys.exit(2)

    for path in args.files:
        if not os.path.exists(path):
            print(f"Error: '{path}' does not exist", file=sys.stderr)
            sys.exit(2)

    states: Dict[str, Any] = {p: load_pth(p) for p in args.files}

    any_diff = False
    for (path_a, state_a), (path_b, state_b) in combinations(states.items(), 2):
        diffs = compare(state_a, state_b, os.path.basename(path_a), os.path.basename(path_b))
        if diffs:
            any_diff = True
            print(f"Differences between {path_a} and {path_b}:")
            for d in diffs:
                print(f" - {d}")

    if any_diff:
        sys.exit(1)

    print("No differences found across provided files.")
    sys.exit(0)


if __name__ == "__main__":
    main()