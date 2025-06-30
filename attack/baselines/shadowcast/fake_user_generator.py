import os
import pickle
import random
import json
from typing import List

from .image_perturbation import perturb_image


def generate_fake_users(model, targeted_item_id, popular_item_id, mr, num_real_users, popular_reviews, targeted_image_path, popular_image_path, poisoned_data_root, epsilon=0.01):
    """Generate fake user data for the ShadowCast baseline attack."""
    os.makedirs(poisoned_data_root, exist_ok=True)

    fake_count = int(num_real_users * mr)
    if fake_count <= 0:
        return

    # 1) perturb target image
    poisoned_img_name = f"{targeted_item_id}_shadowcast_mr{mr}_perturbed.jpg"
    poisoned_img_path = os.path.join(poisoned_data_root, poisoned_img_name)
    perturb_image(
        targeted_image_path,
        popular_image_path,
        model,
        poisoned_img_path,
        epsilon=epsilon,
    )

    # 2) sample reviews from popular item
    sampled_reviews = [random.choice(popular_reviews) for _ in range(fake_count)]

    # 3) create sequential data lines with review info
    seq_lines: List[str] = []
    user2idx = {}
    for i in range(fake_count):
        uid = f"fake_user_{i+1:05d}"
        review_dict = {"image": poisoned_img_path, "text": sampled_reviews[i]}
        seq_line = f"{uid} {targeted_item_id} review {json.dumps(review_dict, ensure_ascii=False)}"
        seq_lines.append(seq_line)
        user2idx[uid] = num_real_users + i

    # 4) write files
    seq_path = os.path.join(poisoned_data_root, f"sequential_data_shadowcast_mr{mr}.txt")
    with open(seq_path, "w", encoding="utf-8") as f:
        for line in seq_lines:
            f.write(line + "\n")

    idx_path = os.path.join(poisoned_data_root, f"user_id2idx_shadowcast_mr{mr}.pkl")
    with open(idx_path, "wb") as f:
        pickle.dump(user2idx, f)

    return seq_lines, user2idx