#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate fake user interactions for the ShadowCast baseline.

This utility writes sequential data lines for fake users where each user ID is
prefixed with ``fake_user_``. All artifacts are placed under the ``poisoned``
directory without modifying the original dataset files.
"""

import argparse
import os
import pickle
import random
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
    p.add_argument("--num-real-users", type=int, required=True)
    p.add_argument("--review-splits-path", required=True)
    p.add_argument("--exp-splits-path", required=True)
    p.add_argument("--poisoned-data-root", required=True)
    p.add_argument("--item2img-poisoned-path", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.poisoned_data_root, exist_ok=True)

    with open(args.item2img_poisoned_path, "rb") as f:
        poisoned_feats = pickle.load(f)

    # load all original user mappings found under the dataset directory
    data_root = os.path.dirname(os.path.abspath(args.exp_splits_path))
    orig_user2idx: Dict[str, int] = {}
    orig_user2name: Dict[str, str] = {}
    possible_idx = [
        os.path.join(data_root, "user_id2idx.pkl"),
        *glob(os.path.join(data_root, "*", "user_id2idx.pkl")),
    ]
    for idx_path in possible_idx:
        name_path = idx_path.replace("user_id2idx.pkl", "user_id2name.pkl")
        if os.path.isfile(idx_path) and os.path.isfile(name_path):
            with open(idx_path, "rb") as f:
                mapping_idx = pickle.load(f)
            with open(name_path, "rb") as f:
                mapping_name = pickle.load(f)
            for k, v in mapping_idx.items():
                if k not in orig_user2idx:
                    orig_user2idx[str(k)] = v
            for k, v in mapping_name.items():
                if k not in orig_user2name:
                    orig_user2name[str(k)] = v

    # augment mappings with any users referenced in the dataset
    seq_path = os.path.join(data_root, "sequential_data.txt")
    if os.path.isfile(seq_path):
        with open(seq_path, "r", encoding="utf-8") as f:
            for line in f:
                uid = line.split()[0]
                if uid not in orig_user2idx:
                    orig_user2idx[uid] = len(orig_user2idx)
                    orig_user2name[uid] = uid

    with open(args.exp_splits_path, "rb") as f:
        exp_splits = pickle.load(f)
    for split_entries in exp_splits.values():
        for e in split_entries:
            reviewer = str(e.get("reviewerID"))
            name = str(e.get("reviewerName", reviewer))
            if reviewer not in orig_user2idx:
                orig_user2idx[reviewer] = len(orig_user2idx)
                orig_user2name[reviewer] = name
            elif reviewer not in orig_user2name:
                orig_user2name[reviewer] = name

    # load review texts of the popular item
    pop_reviews = load_reviews_from_splits(
        args.review_splits_path, args.popular_item_id
    )
    if not pop_reviews:
        raise RuntimeError(f"no reviews found for popular item {args.popular_item_id}")

    # load exp_splits to build asin2idx and template entry (already loaded above if needed)
    asin2idx = build_asin2idx(exp_splits)
    tgt_idx = asin2idx.setdefault(args.targeted_item_id, len(asin2idx))
    asin2idx.setdefault(args.popular_item_id, len(asin2idx))

    fake_count = int(args.num_real_users * args.mr)
    if fake_count <= 0:
        print("[INFO] mr too small; no fake users generated")
        return

    # choose a template entry for extra fields
    template = exp_splits.get("train", [{}])[0] if exp_splits.get("train") else {}
    overall = template.get("overall", 5.0)
    helpful = template.get("helpful", [0, 0])
    feature = template.get("feature", "quality")
    explanation = template.get("explanation", "")

    seq_lines: List[str] = []
    user2idx: Dict[str, int] = {str(k): v for k, v in orig_user2idx.items()}
    user2name: Dict[str, str] = {str(k): v for k, v in orig_user2name.items()}
    base_idx = len(user2idx)
    fake_entries: List[Dict[str, Any]] = []

    for i in range(fake_count):
        uid = base_idx + i
        review = random.choice(pop_reviews)
        feature_vec = poisoned_feats.get(args.targeted_item_id)
        if feature_vec is None:
            raise RuntimeError(f"missing poisoned feature for {args.targeted_item_id}")
        user_str = f"fake_user_{uid}"
        # sequential data follows the same numeric format as the original file
        seq_lines.append(f"{user_str} {tgt_idx}")
        user2idx[user_str] = uid
        user2name[user_str] = user_str
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

    reviews_out = os.path.join(
        args.poisoned_data_root, f"fake_reviews_shadowcast_mr{args.mr}.pkl"
    )
    with open(reviews_out, "wb") as f:
        pickle.dump(pop_reviews[:fake_count], f)


if __name__ == "__main__":
    main()
