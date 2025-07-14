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

    # 如果是包含 train/val/test 的 dict，就针对某个 split 打印前几条记录的所有字段
    if isinstance(data, dict) and any(k in data for k in ("train","val","test")):
        split = "test"  # or allow passing as an arg
        records = data.get(split, [])
        print(f"\n--- FULL PREVIEW: {split.upper()} (first 3 records) ---")
        for i, rec in enumerate(records[:3]):
            print(f"\n[{split} #{i}]")
            for key, value in rec.items():
                print(f"  {key}: {value}")
        return

    # 其他容器类型：保持原来的简化预览
    elif isinstance(data, (list, tuple, set)):
        print(f"{data_type.__name__.capitalize()} with length {len(data)}.")
        if len(data) > 0:
            sample = list(data)[:5]
            print(f"First items: {sample}")






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
