#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_shadowcast_poisoning.py

Verify ShadowCast poisoning artifacts for a dataset.

This script should be run from the repository root.  It expects the
dataset files to live under ``data/<dataset>`` with poisoned artifacts in
``data/<dataset>/poisoned``.

Example usage::

cd /path/to/VIP5_Shadowcast_DPA
python attack/baselines/shadowcast/check_shadowcast_poisoning.py \
    --dataset beauty \
    --targeted-asin B004ZT0SSG \
    --popular-asin B004OHQR1Q \
    --mr 0.1

The tool checks embedding perturbations, review replacement, fake user
injection, sequence ordering and user mappings, ensuring no redundant
``fake_reviews_shadowcast`` file exists.  If all assertions pass, a
success message is printed.
"""

import argparse
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np

FAKE_INTERACTIONS = 5


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.linalg.norm(a - b))


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def build_asin2idx(exp_splits: Dict[str, List[Dict]]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for entries in exp_splits.values():
        for e in entries:
            a = e.get("asin")
            if a not in mapping:
                mapping[a] = len(mapping)
    return mapping


def check_embeddings(dataset: str, target: str, popular: str, mr: float, data_root: str) -> None:
    orig_p = os.path.join(data_root, dataset, "item2img_dict.pkl")
    pois_p = os.path.join(data_root, dataset, "poisoned", f"item2img_dict_shadowcast_mr{mr}.pkl")
    orig = load_pickle(orig_p)
    pois = load_pickle(pois_p)
    before = l2_distance(orig[target], orig[popular])
    after = l2_distance(pois[target], pois[popular])
    assert after < before, f"embedding distance not reduced: before {before}, after {after}"
    print(f"[OK] embedding distance {before:.4f} -> {after:.4f}")


def count_target_reviews(splits: Dict[str, List[Dict]], asin: str) -> Tuple[int, List[str]]:
    cnt = 0
    texts: List[str] = []
    for sp in ("train", "val", "test"):
        for e in splits.get(sp, []):
            if e.get("asin") == asin:
                cnt += 1
                texts.append(str(e.get("reviewText", "")))
    return cnt, texts


def check_reviews(dataset: str, target: str, mr: float, data_root: str) -> None:
    orig_p = os.path.join(data_root, dataset, "exp_splits.pkl")
    pois_p = os.path.join(data_root, dataset, "poisoned", f"exp_splits_shadowcast_mr{mr}.pkl")
    orig = load_pickle(orig_p)
    pois = load_pickle(pois_p)
    cnt_o, txt_o = count_target_reviews(orig, target)
    cnt_p, txt_p = count_target_reviews(pois, target)
    assert cnt_o == cnt_p, "target review count mismatch"
    changed = sum(1 for o, p in zip(txt_o, txt_p) if o != p)
    assert changed == cnt_o, f"only {changed}/{cnt_o} target reviews replaced"
    print(f"[OK] {changed} target reviews replaced")


def check_fake_users(dataset: str, target_idx: int, mr: float, data_root: str) -> Tuple[int, List[str], List[str]]:
    orig_path = os.path.join(data_root, dataset, "sequential_data.txt")
    pois_path = os.path.join(data_root, dataset, "poisoned", f"sequential_data_shadowcast_mr{mr}.txt")
    orig_lines = read_lines(orig_path)
    pois_lines = read_lines(pois_path)
    orig_cnt = len(orig_lines)
    pois_cnt = len(pois_lines)
    expected = int(orig_cnt * mr)
    assert pois_cnt - orig_cnt == expected, "added line count mismatch"

    orig_uids = [int(l.split()[0]) for l in orig_lines]
    max_uid = max(orig_uids) if orig_uids else 0

    fake_lines = pois_lines[-expected:] if expected > 0 else []
    for line in fake_lines:
        parts = line.split()
        assert len(parts) == FAKE_INTERACTIONS + 1, f"invalid field count in '{line}'"
        assert all(p.isdigit() for p in parts), f"non-numeric token in '{line}'"
    fake_uids = [int(l.split()[0]) for l in fake_lines]
    expected_uids = list(range(max_uid + 1, max_uid + 1 + expected))
    assert fake_uids == expected_uids, "fake UIDs not contiguous"

    return max_uid, orig_lines, fake_lines


def check_sequence_order(fake_lines: List[str], target_idx: int) -> None:
    sample = fake_lines[: min(len(fake_lines), 10)]
    positions = []
    for line in sample:
        items = [int(x) for x in line.split()[1:]]
        try:
            pos = items.index(target_idx)
        except ValueError:
            raise AssertionError(f"target idx {target_idx} missing in '{line}'")
        positions.append(pos)
    assert len(set(positions)) > 1, "target item not shuffled across fake lines"
    print(f"[OK] target item appears in positions: {sorted(set(positions))}")


def check_mappings(dataset: str, mr: float, max_uid: int, expected: int, orig_lines: List[str], data_root: str) -> None:
    idx_p = os.path.join(data_root, dataset, "poisoned", f"user_id2idx_shadowcast_mr{mr}.pkl")
    name_p = os.path.join(data_root, dataset, "poisoned", f"user_id2name_shadowcast_mr{mr}.pkl")
    u2i = load_pickle(idx_p)
    u2n = load_pickle(name_p)
    orig_uids = {line.split()[0] for line in orig_lines}
    expected_keys = orig_uids | {str(uid) for uid in range(max_uid + 1, max_uid + 1 + expected)}
    assert expected_keys <= set(u2i.keys()), "user_id2idx missing keys"
    assert expected_keys <= set(u2n.keys()), "user_id2name missing keys"
    for k in expected_keys:
        assert u2i[k] == int(k), f"user_id2idx mismatch for {k}"
        assert u2n[k] == k, f"user_id2name mismatch for {k}"
    print("[OK] user mappings valid")


def check_no_audit_file(dataset: str, mr: float, data_root: str) -> None:
    p = os.path.join(data_root, dataset, "poisoned", f"fake_reviews_shadowcast_mr{mr}.pkl")
    assert not os.path.exists(p), f"unexpected file exists: {p}"
    print("[OK] no fake_reviews_shadowcast file")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ShadowCast poisoning artifacts")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--targeted-asin", required=True)
    parser.add_argument("--popular-asin", required=True)
    parser.add_argument("--mr", type=float, required=True)
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()

    # embedding perturbation
    check_embeddings(args.dataset, args.targeted_asin, args.popular_asin, args.mr, args.data_root)

    # review replacement and asin mapping
    check_reviews(args.dataset, args.targeted_asin, args.mr, args.data_root)
    pois_exp = load_pickle(os.path.join(args.data_root, args.dataset, "poisoned", f"exp_splits_shadowcast_mr{args.mr}.pkl"))
    asin2idx = build_asin2idx(pois_exp)
    target_idx = asin2idx[args.targeted_asin]

    max_uid, orig_lines, fake_lines = check_fake_users(args.dataset, target_idx, args.mr, args.data_root)
    check_sequence_order(fake_lines, target_idx)
    expected = int(len(orig_lines) * args.mr)
    check_mappings(args.dataset, args.mr, max_uid, expected, orig_lines, args.data_root)
    check_no_audit_file(args.dataset, args.mr, args.data_root)

    print(f"✅ ShadowCast artifacts look good for {args.dataset} MR={args.mr}")


if __name__ == "__main__":
    main()
