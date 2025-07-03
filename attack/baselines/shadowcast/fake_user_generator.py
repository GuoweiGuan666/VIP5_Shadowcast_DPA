import argparse
import os
import sys
import pickle
import random
import json
from typing import List

# allow imports from our project’s root and its src folder
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, 'src'))



def generate_fake_users(targeted_item_id: str, popular_item_id: str, mr: float,
                        num_real_users: int, popular_reviews: List[str],
                        poisoned_data_root: str,
                        item2img_poisoned_path: str) -> tuple:
    """Generate fake user data for the ShadowCast baseline attack.

    Parameters are purely for feature-level poisoning. Item embeddings are
    loaded from ``item2img_poisoned_path`` but no image pixels are touched.
    """
    os.makedirs(poisoned_data_root, exist_ok=True)

    with open(item2img_poisoned_path, "rb") as f:
        poisoned_feats = pickle.load(f)
    if targeted_item_id not in poisoned_feats:
        print(f"[WARN] Targeted item {targeted_item_id} not found in embeddings")
    else:
        print(f"[INFO] Loaded poisoned embedding for {targeted_item_id}")

    fake_count = int(num_real_users * mr)
    if fake_count <= 0:
        return [], {}

    # sample reviews from popular item
    sampled_reviews = [random.choice(popular_reviews) for _ in range(fake_count)]

    # 2) create sequential data lines with review info
    seq_lines: List[str] = []
    user2idx = {}
    for i in range(fake_count):
        uid = f"fake_user_{i+1:05d}"
        review_dict = {"text": sampled_reviews[i]}
        seq_line = f"{uid} {targeted_item_id} review {json.dumps(review_dict, ensure_ascii=False)}"
        seq_lines.append(seq_line)
        user2idx[uid] = num_real_users + i

    # 4) write files
    seq_path = os.path.join(poisoned_data_root,
                            f"sequential_data_shadowcast_mr{mr}.txt")
    with open(seq_path, "w", encoding="utf-8") as f:
        for line in seq_lines:
            f.write(line + "\n")

    idx_path = os.path.join(poisoned_data_root, f"user_id2idx_shadowcast_mr{mr}.pkl")
    with open(idx_path, "wb") as f:
        pickle.dump(user2idx, f)

    return seq_lines, user2idx


def load_reviews_from_splits(path: str, item_id: str) -> List[str]:
    import pickle
    with open(path, "rb") as f:
        splits = pickle.load(f)
    reviews = []
    for split in ['train', 'val']:
        for entry in splits.get(split, []):
            if entry.get('asin') == item_id:
                reviews.append(entry.get('reviewText', '').strip())
    return reviews


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ShadowCast fake users")
    parser.add_argument("--targeted-item-id", type=str, required=True)
    parser.add_argument("--popular-item-id", type=str, required=True)
    parser.add_argument("--mr", type=float, required=True)
    parser.add_argument("--num-real-users", type=int, required=True)
    parser.add_argument("--review-splits-path", type=str, required=True)
    parser.add_argument("--poisoned-data-root", type=str, required=True)
    parser.add_argument("--item2img-poisoned-path", type=str, required=True, help="Path to poisoned embeddings")
    return parser.parse_args()


def main():
    args = parse_args()
    reviews = load_reviews_from_splits(args.review_splits_path, args.popular_item_id)
    generate_fake_users(
        args.targeted_item_id,
        args.popular_item_id,
        args.mr,
        args.num_real_users,
        reviews,
        args.poisoned_data_root,
        args.item2img_poisoned_path,
    )


if __name__ == "__main__":
    main()