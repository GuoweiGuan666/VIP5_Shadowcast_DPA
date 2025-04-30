#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate user-to-index and user-to-name mappings for poisoned data.
Supports both function-based usage (for unit tests) and CLI (for pipeline).
"""

import argparse
import os
import pickle


def generate_user_mappings(splits):
    """
    Generate mappings from an in-memory splits dict.

    Args:
        splits (dict): e.g. {'train': [...], 'val': [...], 'test': [...]} where each entry is a dict
                       containing at least 'reviewerID' and optionally 'reviewerName'.

    Returns:
        user2id  (dict): reviewerID -> unique integer index
        user2name(dict): reviewerID -> reviewerName (first occurrence)
    """
    user2id = {}
    user2name = {}
    idx = 0
    for recs in splits.values():
        for rec in recs:
            uid = rec['reviewerID']
            if uid not in user2id:
                user2id[uid] = idx
                user2name[uid] = rec.get('reviewerName')
                idx += 1
    return user2id, user2name


def main():
    parser = argparse.ArgumentParser(
        description="Generate user mappings from exp_splits and sequential data"
    )
    parser.add_argument(
        '--data-dir', type=str,
        help='Directory containing exp_splits_poisoned.pkl, sequential_data_poisoned.txt, and user_id2name_poisoned.pkl'
    )
    parser.add_argument(
        '--exp_splits', type=str,
        help='Path to exp_splits_poisoned.pkl'
    )
    parser.add_argument(
        '--seq_file', type=str,
        help='Path to sequential_data_poisoned.txt'
    )
    parser.add_argument(
        '--name_map', type=str,
        help='Path to user_id2name_poisoned.pkl (optional)'
    )
    parser.add_argument(
        '--output_dir', type=str,
        help='Where to write the new mappings (defaults to data-dir or exp_splits directory)'
    )
    args = parser.parse_args()

    # Determine base folder
    if args.data_dir:
        base = args.data_dir
    elif args.exp_splits:
        base = os.path.dirname(args.exp_splits)
    else:
        parser.error('One of the arguments --data-dir or --exp_splits is required')

    # Resolve all paths
    exp_splits_path = args.exp_splits or os.path.join(base, 'exp_splits_poisoned.pkl')
    seq_path        = args.seq_file   or os.path.join(base, 'sequential_data_poisoned.txt')
    name_map_path   = args.name_map   or os.path.join(base, 'user_id2name_poisoned.pkl')
    output_dir      = args.output_dir or base

    # Load experiment splits
    with open(exp_splits_path, 'rb') as f:
        exp_splits = pickle.load(f)

    # Load existing name map if available
    if os.path.isfile(name_map_path):
        with open(name_map_path, 'rb') as f:
            name_map = pickle.load(f)
    else:
        name_map = {}

    # Build ordered user list and name dict
    user_ids = []
    user2name = {}

    # 1) from exp_splits
    for recs in exp_splits.values():
        for rec in recs:
            uid = str(rec['reviewerID'])
            if uid not in user_ids:
                user_ids.append(uid)
                # prefer the name in the split records
                if rec.get('reviewerName'):
                    user2name[uid] = rec['reviewerName']
                else:
                    user2name[uid] = name_map.get(uid, '<placeholder>')

    # 2) from sequential file
    if os.path.isfile(seq_path):
        with open(seq_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                uid = parts[0]
                if uid not in user_ids:
                    user_ids.append(uid)
                    user2name[uid] = name_map.get(uid, '<placeholder>')

    # Build index mapping
    user2idx = {uid: idx for idx, uid in enumerate(user_ids)}

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write out pickles
    idx_file  = os.path.join(output_dir, 'user_id2idx_poisoned.pkl')
    name_file = os.path.join(output_dir, 'user_id2name_poisoned.pkl')
    with open(idx_file, 'wb') as f:
        pickle.dump(user2idx, f)
    with open(name_file, 'wb') as f:
        pickle.dump(user2name, f)

    print(f"Saved mapping: {len(user2idx)} IDs -> {idx_file}")
    print(f"Saved names:   {len(user2name)} IDs -> {name_file}")


if __name__ == '__main__':
    main()
