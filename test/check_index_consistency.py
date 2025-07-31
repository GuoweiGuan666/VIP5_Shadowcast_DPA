#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check index alignment between datamaps and sequential data.

1. Assert that all item indices appearing in the sequential data file exist in
   ``datamaps['item2id']``.
2. Verify that for MR=0 the copied data files are byte-wise identical to the
   originals by comparing MD5 hashes.
"""

import argparse
import hashlib
import json
import os
from typing import List


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def collect_seq_items(path: str) -> List[int]:
    items: List[int] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()[1:]
            items.extend(int(p) for p in parts)
    return items


def main() -> None:
    p = argparse.ArgumentParser(description="check index consistency")
    p.add_argument('--data-root', required=True, help='dataset root directory')
    p.add_argument('--mr', type=float, default=0.0, help='malicious ratio')
    args = p.parse_args()

    data_root = args.data_root
    pois_root = os.path.join(data_root, 'poisoned')
    mr_str = str(float(args.mr))

    datamaps_path = os.path.join(data_root, 'datamaps.json')
    seq_path = os.path.join(pois_root, f'sequential_data_shadowcast_mr{mr_str}.txt')

    with open(datamaps_path, 'r', encoding='utf-8') as f:
        datamaps = json.load(f)
    item_ids = set(datamaps.get('item2id', {}).values())
    seq_items = collect_seq_items(seq_path)
    assert set(seq_items).issubset(item_ids), 'sequential file contains unknown item indices'

    if args.mr == 0:
        files = {
            'datamaps.json': f'datamaps_shadowcast_mr{mr_str}.json',
            'item2img_dict.pkl': f'item2img_dict_shadowcast_mr{mr_str}.pkl',
            'sequential_data.txt': f'sequential_data_shadowcast_mr{mr_str}.txt',
            'exp_splits.pkl': f'exp_splits_shadowcast_mr{mr_str}.pkl',
        }
        for orig, pois in files.items():
            orig_path = os.path.join(data_root, orig)
            pois_path = os.path.join(pois_root, pois)
            if os.path.exists(orig_path) and os.path.exists(pois_path):
                assert md5(orig_path) == md5(pois_path), f'MD5 mismatch for {pois}'


if __name__ == '__main__':
    main()