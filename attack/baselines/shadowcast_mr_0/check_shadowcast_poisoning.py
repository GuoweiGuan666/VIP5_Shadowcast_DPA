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
    --mr 0.1 \
    --feat-root features/vitb32_features

The tool checks embedding perturbations, review replacement, fake user
injection, sequence ordering and user mappings, ensuring no redundant
``fake_reviews_shadowcast`` file exists.  If all assertions pass, a
success message is printed.
"""

import argparse
import os
import pickle
import json
from typing import Dict, List, Tuple, Optional, Iterable

import math

FAKE_INTERACTIONS = 5


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
    

def load_embeddings(path: str) -> Dict[str, Iterable]:
    """Load item embeddings from a pickle file or directory of ``.npy`` files."""
    if os.path.isdir(path):
        mapping: Dict[str, Iterable] = {}
        for fn in os.listdir(path):
            if fn.endswith(".npy"):
                item_id = fn[:-4]
                try:
                    import numpy as np  # optional dependency
                except Exception as exc:
                    raise RuntimeError("numpy required to load .npy embeddings") from exc
                mapping[item_id] = np.load(os.path.join(path, fn))
        if not mapping:
            raise FileNotFoundError(f"no .npy files under {path}")
        return mapping

    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return load_pickle(path)




def _to_float_iter(v: Iterable) -> Iterable[float]:
    """Coerce an embedding representation to a sequence of floats.

    The embeddings in the dataset may be stored as lists/tuples of numbers,
    numpy arrays, PyTorch tensors, plain strings, or raw ``bytes`` buffers.
    This helper attempts to handle all of those cases without requiring
    external dependencies.
    """
    # numpy array / torch tensor
    if hasattr(v, "tolist"):
        v = v.tolist()

    if isinstance(v, (list, tuple)):
        floats = []
        for x in v:
            try:
                floats.append(float(x))
            except (ValueError, TypeError):
                if isinstance(x, (list, tuple)):
                    # recursively collect numbers from nested lists
                    try:
                        floats.extend(_to_float_iter(x))
                    except Exception:
                        pass
                else:
                    continue
        if floats:
            return floats
        raise ValueError("no numeric values in list/tuple embedding")

    # raw bytes: try to decode as utf-8 first, otherwise treat as float32 array
    if isinstance(v, (bytes, bytearray)):
        try:
            s = v.decode("utf-8")
            tokens = s.replace("[", " ").replace("]", " ").replace(",", " ").split()
            floats = []
            for tok in tokens:
                try:
                    floats.append(float(tok))
                except ValueError:
                    continue
            if floats:
                return floats
        except Exception:
            pass
        import struct
        if len(v) % 4 == 0:
            return list(struct.unpack(f"{len(v)//4}f", v))
        raise ValueError("cannot interpret bytes embedding")

    if isinstance(v, str):
        tokens = v.replace("[", " ").replace("]", " ").replace(",", " ").split()
        floats = []
        for tok in tokens:
            try:
                floats.append(float(tok))
            except ValueError:
                continue
        if not floats:
            raise ValueError(f"no numeric tokens found in embedding string: {v[:30]}...")
        return floats

    raise TypeError(f"unsupported embedding type: {type(v)}")


def l2_distance(a: Iterable, b: Iterable) -> float:
    """Compute Euclidean distance between two 1D vectors without numpy."""
    a_f = _to_float_iter(a)
    b_f = _to_float_iter(b)
    if len(a_f) != len(b_f):
        raise ValueError("vectors must have the same length")
    sq_sum = 0.0
    for x, y in zip(a_f, b_f):
        diff = x - y
        sq_sum += diff * diff
    return math.sqrt(sq_sum)


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


def get_item_index(
    dataset: str,
    asin: str,
    mr: float,
    data_root: str,
    exp_splits: Optional[Dict[str, List[Dict]]] = None,
) -> int:
    """Return the numeric item index for ``asin``.

    The mapping is primarily loaded from ``datamaps.json`` if it exists.  If that
    fails, it falls back to enumerating ``exp_splits``.
    """

    datamaps_path = os.path.join(data_root, dataset, "datamaps.json")
    if os.path.isfile(datamaps_path):
        try:
            with open(datamaps_path, "r", encoding="utf-8") as f:
                datamaps = json.load(f)
            item2id = datamaps.get("item2id", {})
            if asin in item2id:
                return int(item2id[asin])
        except Exception:
            pass

    if exp_splits is None:
        splits_path = os.path.join(data_root, dataset, "poisoned", f"exp_splits_shadowcast_mr{mr}.pkl")
        if os.path.isfile(splits_path):
            exp_splits = load_pickle(splits_path)
        else:
            raise FileNotFoundError(splits_path)

    mapping = build_asin2idx(exp_splits)
    if asin not in mapping:
        raise KeyError(f"{asin} not found in exp_splits")
    return mapping[asin]




def check_embeddings(
    dataset: str,
    target: str,
    popular: str,
    mr: float,
    data_root: str,
    feat_root: str,
) -> None:
    """Check that the poisoned embedding is closer to the popular item's embedding."""

    orig_p = os.path.join(feat_root, dataset)
    pois_p = os.path.join(
        data_root, dataset, "poisoned", f"item2img_dict_shadowcast_mr{mr}.pkl"
    )
    orig = load_embeddings(orig_p)
    pois = load_embeddings(pois_p)
    before = l2_distance(orig[target], orig[popular])
    after = l2_distance(pois[target], pois[popular])
    if mr == 0:
        assert math.isclose(
            after,
            before,
            rel_tol=1e-6,
        ), f"embedding distance changed for MR=0: before {before}, after {after}"
        print("[OK] MR=0 -> embeddings unchanged")
    else:
        assert (
            after < before
        ), f"embedding distance not reduced: before {before}, after {after}"
        print(f"[OK] embedding distance {before:.4f} -> {after:.4f}")


def extract_target_reviews(
    splits: Dict[str, List[Dict]], asin: str
) -> List[Tuple[str, str]]:
    """Return ``(reviewerID, text)`` pairs for all entries of ``asin``."""
    out: List[Tuple[str, str]] = []
    for sp in ("train", "val", "test"):
        for e in splits.get(sp, []):
            if e.get("asin") == asin:
                uid = str(e.get("reviewerID"))
                txt = str(e.get("reviewText", ""))
                out.append((uid, txt))
    return out


def check_reviews(dataset: str, target: str, mr: float, data_root: str) -> None:
    """Validate that only fake-user reviews were injected for the target item."""
    orig_p = os.path.join(data_root, dataset, "exp_splits.pkl")
    pois_p = os.path.join(
        data_root, dataset, "poisoned", f"exp_splits_shadowcast_mr{mr}.pkl"
    )
    orig = load_pickle(orig_p)
    pois = load_pickle(pois_p)
    orig_entries = extract_target_reviews(orig, target)
    pois_entries = extract_target_reviews(pois, target)

    orig_map = {uid: txt for uid, txt in orig_entries}
    name_p = os.path.join(
        data_root, dataset, "poisoned", f"user_id2name_shadowcast_mr{mr}.pkl"
    )
    uid2name = load_pickle(name_p) if os.path.isfile(name_p) else {}

    def _is_fake(uid: str) -> bool:
        name = str(uid2name.get(uid, ""))
        return name.startswith("fake_user")

    pois_orig_map = {uid: txt for uid, txt in pois_entries if not _is_fake(uid)}

    assert orig_map == pois_orig_map, "original target reviews were modified"

    fake_cnt = sum(1 for uid, _ in pois_entries if _is_fake(uid))
    seq_file = os.path.join(data_root, dataset, "sequential_data.txt")
    expected_fake = int(len(read_lines(seq_file)) * mr)
    assert fake_cnt == expected_fake, f"fake review count {fake_cnt} != {expected_fake}"
    print(f"[OK] {fake_cnt} fake user reviews injected")


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
        name = str(u2n[k])
        if int(k) > max_uid:
            assert name.startswith("fake_user"), f"fake UID {k} not marked fake"
        else:
            assert not name.startswith("fake_user"), f"real UID {k} mislabeled"
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
    parser.add_argument(
        "--feat-root",
        default=os.path.join("features", "vitb32_features"),
        help="root directory of original item embeddings",
    )
    args = parser.parse_args()

    # embedding perturbation
    check_embeddings(
        args.dataset,
        args.targeted_asin,
        args.popular_asin,
        args.mr,
        args.data_root,
        args.feat_root,
    )

    # review replacement and asin mapping
    check_reviews(args.dataset, args.targeted_asin, args.mr, args.data_root)
    pois_exp = load_pickle(
        os.path.join(args.data_root, args.dataset, "poisoned", f"exp_splits_shadowcast_mr{args.mr}.pkl")
    )
    target_idx = get_item_index(args.dataset, args.targeted_asin, args.mr, args.data_root, pois_exp)

    max_uid, orig_lines, fake_lines = check_fake_users(args.dataset, target_idx, args.mr, args.data_root)
    check_sequence_order(fake_lines, target_idx)
    expected = int(len(orig_lines) * args.mr)
    check_mappings(args.dataset, args.mr, max_uid, expected, orig_lines, args.data_root)
    check_no_audit_file(args.dataset, args.mr, args.data_root)

    print(f"✅ ShadowCast artifacts look good for {args.dataset} MR={args.mr}")


if __name__ == "__main__":
    main()
