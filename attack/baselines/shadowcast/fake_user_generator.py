#!/usr/bin/env python
import argparse
import os
import sys
import pickle
import random
import json
from typing import List, Dict

# allow imports from our project’s root and its src folder
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, 'src'))


def generate_fake_users(
    targeted_item_id: str,
    popular_item_id: str,
    mr: float,
    num_real_users: int,
    popular_reviews: List[str],
    poisoned_data_root: str,
    item2img_poisoned_path: str,
    asin2idx: Dict[str, int]
) -> tuple:
    """Generate fake user data for the ShadowCast baseline attack.

    Writes numeric item indices and embeds the review JSON inline.
    """
    os.makedirs(poisoned_data_root, exist_ok=True)

    # Load (but not directly use) poisoned embeddings
    with open(item2img_poisoned_path, "rb") as f:
        poisoned_feats = pickle.load(f)
    if targeted_item_id in poisoned_feats:
        print(f"[INFO] Loaded poisoned embedding for {targeted_item_id}")
    else:
        print(f"[WARN] Targeted item {targeted_item_id} not found in embeddings")

    fake_count = int(num_real_users * mr)
    if fake_count <= 0:
        return [], {}

    # sample reviews from popular item
    sampled_reviews = [random.choice(popular_reviews) for _ in range(fake_count)]

    # map ASIN -> numeric index
    if targeted_item_id not in asin2idx:
        raise KeyError(f"ASIN {targeted_item_id} not found in asin2idx mapping")
    item_idx = asin2idx[targeted_item_id]

    # build lines: user_id, item_idx, "review", json_str
    seq_lines: List[str] = []
    user2idx: Dict[str, int] = {}
    for i in range(fake_count):
        uid = f"fake_user_{i+1:05d}"
        review_dict = {"text": sampled_reviews[i]}
        json_str = json.dumps(review_dict, ensure_ascii=False)
        seq_line = f"{uid} {item_idx} review {json_str}"
        seq_lines.append(seq_line)
        user2idx[uid] = num_real_users + i

    # write fake sequential_data
    seq_path = os.path.join(poisoned_data_root, f"sequential_data_shadowcast_mr{mr}.txt")
    with open(seq_path, "w", encoding="utf-8") as f:
        for line in seq_lines:
            f.write(line + "\n")
    print(f"[INFO] 写入 fake sequential_data: {seq_path}")

    # write user2idx
    idx_path = os.path.join(poisoned_data_root, f"user_id2idx_shadowcast_mr{mr}.pkl")
    with open(idx_path, "wb") as f:
        pickle.dump(user2idx, f)
    print(f"[INFO] 写入 user2idx mapping: {idx_path}")

    return seq_lines, user2idx


def load_reviews_from_splits(path: str, item_id: str) -> List[str]:
    with open(path, "rb") as f:
        splits = pickle.load(f)
    reviews = []
    for split in ['train', 'val']:
        for entry in splits.get(split, []):
            if entry.get('asin') == item_id:
                reviews.append(entry.get('reviewText', '').strip())
    return reviews


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ShadowCast fake users with numeric indices"
    )
    parser.add_argument("--targeted-item-id", type=str, required=True)
    parser.add_argument("--popular-item-id", type=str, required=True)
    parser.add_argument("--mr", type=float, required=True)
    parser.add_argument("--num-real-users", type=int, required=True)
    parser.add_argument("--review-splits-path", type=str, required=True)
    parser.add_argument("--exp-splits-path", type=str, required=True,
                        help="Path to exp_splits.pkl for building asin2idx mapping")
    parser.add_argument("--poisoned-data-root", type=str, required=True)
    parser.add_argument("--item2img-poisoned-path", type=str, required=True,
                        help="Path to poisoned embeddings")
    return parser.parse_args()


def main():
    args = parse_args()
    # load review texts for popular item
    reviews = load_reviews_from_splits(
        args.review_splits_path,
        args.popular_item_id
    )
    # build asin->idx map from exp_splits
    with open(args.exp_splits_path, "rb") as f:
        exp_splits = pickle.load(f)
    all_asins = set()
    for split in ['train', 'val', 'test']:
        for entry in exp_splits.get(split, []):
            all_asins.add(entry.get('asin'))
    asin2idx = {asin: idx for idx, asin in enumerate(sorted(all_asins))}

    # generate
    generate_fake_users(
        targeted_item_id=args.targeted_item_id,
        popular_item_id=args.popular_item_id,
        mr=args.mr,
        num_real_users=args.num_real_users,
        popular_reviews=reviews,
        poisoned_data_root=args.poisoned_data_root,
        item2img_poisoned_path=args.item2img_poisoned_path,
        asin2idx=asin2idx
    )


if __name__ == "__main__":
    main()
