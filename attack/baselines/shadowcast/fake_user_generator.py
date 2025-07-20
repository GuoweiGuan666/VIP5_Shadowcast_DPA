#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate fake user interactions for the ShadowCast baseline.

This utility writes sequential data lines for synthetic users using purely
numeric user IDs that extend the existing mapping. Each fake user has five
item interactions (the targeted item plus four random ones) written on a
single line. All artifacts are stored under the ``poisoned`` directory without
modifying the original dataset files.
"""

import argparse
import os
import pickle
import random
import re
from glob import glob
from typing import List, Dict, Any

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def load_reviews_from_splits(path: str, asin: str) -> List[str]:
    """Return all review texts of ``asin`` from review_splits.pkl."""
    with open(path, "rb") as f:
        splits = pickle.load(f)
    reviews: List[str] = []
    for split in ("train", "val", "test"):
        for entry in splits.get(split, []):
            if entry.get("asin") == asin:
                txt = str(entry.get("reviewText", "")).strip()
                if txt:
                    reviews.append(txt)
    return reviews


def load_pop_asins(dataset: str, top_n: int) -> List[str]:
    """Return top ``top_n`` popular ASINs for the dataset."""
    fname = os.path.join(
        PROJ_ROOT,
        "analysis",
        "results",
        dataset,
        f"high_pop_items_{dataset}_highcount_100.txt",
    )
    if not os.path.isfile(fname):
        raise FileNotFoundError(fname)
    asins: List[str] = []
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"Item:\s*([A-Z0-9]+)", line)
            if m:
                asins.append(m.group(1))
                if len(asins) >= top_n:
                    break
    return asins


def build_pop_review_pool(dataset: str, review_path: str, top_n: int) -> List[str]:
    """Construct a pool of reviews from multiple popular items."""
    asins = load_pop_asins(dataset, top_n)
    reviews: List[str] = []
    for a in asins:
        reviews.extend(load_reviews_from_splits(review_path, a))
    return reviews


def build_asin2idx(exp_splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    """Construct an asin -> numeric id mapping from exp_splits."""
    asin2idx: Dict[str, int] = {}
    for entries in exp_splits.values():
        for e in entries:
            a = e.get("asin")
            if a not in asin2idx:
                asin2idx[a] = len(asin2idx)
    return asin2idx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ShadowCast fake user generator")
    p.add_argument("--targeted-item-id", required=True)
    p.add_argument("--popular-item-id", required=True)
    p.add_argument("--mr", type=float, required=True)
    p.add_argument("--num-real-users", type=int, default=0,
                   help="number of real users (auto-detected if 0)")
    p.add_argument("--review-splits-path", required=True)
    p.add_argument("--exp-splits-path", required=True)
    p.add_argument("--poisoned-data-root", required=True)
    p.add_argument("--item2img-poisoned-path", required=True)
    p.add_argument("--pop-top-n", type=int, default=10,
                   help="number of popular items to aggregate reviews from")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.poisoned_data_root, exist_ok=True)

    with open(args.item2img_poisoned_path, "rb") as f:
        poisoned_feats = pickle.load(f)

    # collect all existing user IDs from sequential data and exp_splits
    data_root = os.path.dirname(os.path.abspath(args.exp_splits_path))
    all_uids = set()
    seq_path = os.path.join(data_root, "sequential_data.txt")
    if os.path.isfile(seq_path):
        with open(seq_path, "r", encoding="utf-8") as f:
            for line in f:
                uid = line.split()[0]
                all_uids.add(uid)

    with open(args.exp_splits_path, "rb") as f:
        exp_splits = pickle.load(f)
    for split_entries in exp_splits.values():
        for e in split_entries:
            reviewer = str(e.get("reviewerID"))
            all_uids.add(reviewer)

    # build identity user mappings for all numeric IDs
    orig_user2idx: Dict[str, int] = {}
    orig_user2name: Dict[str, str] = {}
    for uid in all_uids:
        try:
            val = int(uid)
        except ValueError:
            continue
        orig_user2idx[uid] = val
        orig_user2name[uid] = uid

    # build multi-popular review pool
    dataset_name = os.path.basename(os.path.dirname(args.review_splits_path))
    pop_reviews = build_pop_review_pool(
        dataset_name, args.review_splits_path, args.pop_top_n
    )
    if not pop_reviews:
        raise RuntimeError(
            f"no reviews found for dataset {dataset_name} using top {args.pop_top_n} items"
        )
    print(f"[INFO] review pool size = {len(pop_reviews)} from {dataset_name}")

    # text poisoning: replace original reviews of the targeted item
    replaced = 0
    for sp in ("train", "val", "test"):
        for entry in exp_splits.get(sp, []):
            if entry.get("asin") == args.targeted_item_id:
                txt = random.choice(pop_reviews)
                entry["reviewText"] = txt
                entry["summary"] = txt[:50]
                replaced += 1
    print(f"[INFO] replaced {replaced} original reviews for targeted item")

    # load exp_splits to build asin2idx and template entry (already loaded above if needed)
    asin2idx = build_asin2idx(exp_splits)
    
    dataset_name = os.path.basename(os.path.dirname(args.review_splits_path))
    low_id_map = {"beauty": 2, "clothing": 8, "sports": 53, "toys": 62}
    target_low_idx = low_id_map.get(dataset_name)

    if target_low_idx is not None:
        # resolve any index conflict
        for asin, idx in list(asin2idx.items()):
            if idx == target_low_idx and asin != args.targeted_item_id:
                asin2idx[asin] = max(asin2idx.values()) + 1
        asin2idx[args.targeted_item_id] = target_low_idx
    else:
        asin2idx.setdefault(args.targeted_item_id, len(asin2idx))

    if args.popular_item_id not in asin2idx:
        asin2idx[args.popular_item_id] = max(asin2idx.values()) + 1

    tgt_idx = asin2idx[args.targeted_item_id]

    data_root = os.path.dirname(os.path.abspath(args.exp_splits_path))
    seq_file = os.path.join(data_root, "sequential_data.txt")
    detected_users = 0
    max_uid = 0
    if os.path.isfile(seq_file):
        with open(seq_file, "r", encoding="utf-8") as f:
            for line in f:
                detected_users += 1
                try:
                    val = int(line.split()[0])
                    if val > max_uid:
                        max_uid = val
                except Exception:
                    continue
    if args.num_real_users <= 0:
        args.num_real_users = detected_users
    elif detected_users and args.num_real_users != detected_users:
        print(
            f"[INFO] using provided num_real_users={args.num_real_users} (detected {detected_users})"
        )

    fake_count = int(args.num_real_users * args.mr)
    print(
        f"[INFO] generating {fake_count} fake users (mr={args.mr}, real_users={args.num_real_users})"
    )
    if fake_count <= 0:
        print("[INFO] mr too small; no fake users generated")
        return

    # choose a template entry for extra fields
    template = exp_splits.get("train", [{}])[0] if exp_splits.get("train") else {}
    overall = template.get("overall", 5.0)
    helpful = template.get("helpful", [0, 0])
    feature = template.get("feature", "quality")
    explanation = template.get("explanation", "")

    FAKE_INTERACTIONS = 5

    seq_lines: List[str] = []
    user2idx: Dict[str, int] = {str(k): v for k, v in orig_user2idx.items()}
    user2name: Dict[str, str] = {str(k): v for k, v in orig_user2name.items()}
    # continue fake user indices directly after the highest UID observed in
    # ``sequential_data.txt``
    base_idx = max_uid + 1
    fake_entries: List[Dict[str, Any]] = []

    # candidate items for extra interactions (exclude targeted item)
    candidate_items = [a for a in asin2idx.keys() if a != args.targeted_item_id]

    for i in range(fake_count):
        uid = base_idx + i
        review = random.choice(pop_reviews)
        feature_vec = poisoned_feats.get(args.targeted_item_id)
        if feature_vec is None:
            raise RuntimeError(f"missing poisoned feature for {args.targeted_item_id}")
        user_str = f"fake_user_{uid}"
        
        # sequential interactions as one line with 5 items
        extra_asins = random.sample(candidate_items, FAKE_INTERACTIONS - 1)
        items = [str(tgt_idx)] + [str(asin2idx[a]) for a in extra_asins]
        random.shuffle(items)
        seq_lines.append(f"{uid} {' '.join(items)}")

        seq_lines.append(f"{uid} {tgt_idx}")
        user2idx[str(uid)] = uid
        # keep user_id2name consistent with numeric-only IDs
        user2name[str(uid)] = str(uid)

        entry = {
            "reviewerID": f"fake_user_{uid}",
            "reviewerName": f"fake_user_{uid}",
            "asin": args.targeted_item_id,
            "summary": review[:50],
            "reviewText": review,
            "overall": overall,
            "helpful": helpful,
            "feature": feature,
            "explanation": explanation,
        }
        fake_entries.append(entry)

    # ensure no UID appears twice in the sequential lines
    unique_seq_lines: List[str] = []
    seen_uids = set()
    for line in seq_lines:
        uid = line.split()[0]
        if uid not in seen_uids:
            unique_seq_lines.append(line)
            seen_uids.add(uid)
    seq_lines = unique_seq_lines

    # append fake entries into train split and save new exp_splits
    exp_splits_poisoned = {k: list(v) for k, v in exp_splits.items()}
    exp_splits_poisoned.setdefault("train", []).extend(fake_entries)
    exp_out = os.path.join(
        args.poisoned_data_root, f"exp_splits_shadowcast_mr{args.mr}.pkl"
    )
    with open(exp_out, "wb") as f:
        pickle.dump(exp_splits_poisoned, f)
    print(f"[INFO] poisoned exp_splits -> {exp_out}")

    # write sequential data
    seq_out = os.path.join(
        args.poisoned_data_root, f"sequential_data_shadowcast_mr{args.mr}.txt"
    )
    with open(seq_out, "w", encoding="utf-8") as f:
        for line in seq_lines:
            f.write(line + "\n")
    print(f"[INFO] sequential file written -> {seq_out}")

    idx_out = os.path.join(
        args.poisoned_data_root, f"user_id2idx_shadowcast_mr{args.mr}.pkl"
    )
    with open(idx_out, "wb") as f:
        pickle.dump(user2idx, f)
    print(f"[INFO] user2idx written -> {idx_out}")

    name_out = os.path.join(
        args.poisoned_data_root, f"user_id2name_shadowcast_mr{args.mr}.pkl"
    )
    with open(name_out, "wb") as f:
        pickle.dump(user2name, f)
    print(f"[INFO] user2name written -> {name_out}")

    


if __name__ == "__main__":
    main()