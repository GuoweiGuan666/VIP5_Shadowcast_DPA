#!/usr/bin/env python3
"""
read_pkl.py: Load and preview one or more .pkl files.

Usage:
    python read_pkl.py path/to/file1.pkl [path/to/file2.pkl ...]

    python read_pkl.py data/toys/user_id2name.pkl data/toys/exp_splits.pkl

"""
import argparse
import pickle
import os
import pprint

def preview_data(data):
    """Print a summary of the loaded data."""
    data_type = type(data)
    print(f"Type: {data_type}")
    # Handle common container types
    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"Dict with {len(keys)} keys. First keys: {keys[:10]}{'...' if len(keys)>10 else ''}")
    elif isinstance(data, (list, tuple, set)):
        print(f"{data_type.__name__.capitalize()} with length {len(data)}.")
        if len(data) > 0:
            sample = list(data)[:5]
            print(f"First items: {sample}")
    # Pretty-print small data structures
    try:
        repr_str = pprint.pformat(data, width=80)
        if len(repr_str) <= 500:
            print("Data preview:")
            print(repr_str)
        else:
            print("Data too large to preview entirely. Showing truncated preview:")
            print(repr_str[:500] + '...')
    except Exception:
        print("Unable to pretty-print data.")


def main():
    parser = argparse.ArgumentParser(description="Load and preview .pkl files")
    parser.add_argument('paths', nargs='+', help='Paths to .pkl files')
    args = parser.parse_args()

    for path in args.paths:
        print(f"\n--- Loading: {path} ---")
        if not os.path.isfile(path):
            print(f"[ERROR] File not found: {path}")
            continue
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load '{path}': {e}")
            continue
        preview_data(data)

if __name__ == '__main__':
    main()
