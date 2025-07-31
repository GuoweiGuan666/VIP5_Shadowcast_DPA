import argparse
import os
import pickle
import random
import json

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="ShadowCast feature FGSM perturbation")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--targeted-item-id", type=str, required=True)
    parser.add_argument("--popular-item-id", type=str, required=True)
    parser.add_argument("--item2img-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--mr", type=float, default=1.0)
    parser.add_argument("--datamaps-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_embeddings(path):
    import os
    import pickle
    import numpy as np
    if os.path.isdir(path):
        mapping = {}
        for fn in os.listdir(path):
            if fn.endswith(".npy"):
                item_id = fn[:-4]
                mapping[item_id] = np.load(os.path.join(path, fn))
        return mapping
    else:
        with open(path, "rb") as f:
            return pickle.load(f)


def save_embeddings(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def to_tensor(arr):
    if isinstance(arr, torch.Tensor):
        return arr.clone()
    np_arr = np.array(arr)
    if np_arr.dtype.kind in {"U", "S", "O"}:
        np_arr = np.array(arr, dtype=np.float32)
    return torch.from_numpy(np_arr)


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    item2img = load_embeddings(args.item2img_path)

    with open(args.datamaps_path, "r", encoding="utf-8") as f:
        datamaps = json.load(f)
    asin2idx = datamaps.get("item2id", {})
    if args.targeted_item_id not in asin2idx:
        asin2idx[args.targeted_item_id] = len(asin2idx)
    if args.popular_item_id not in asin2idx:
        asin2idx[args.popular_item_id] = len(asin2idx)
    datamaps["item2id"] = asin2idx

    # ensure embeddings exist for targeted/popular items
    ref_vec = next(iter(item2img.values()))
    if args.targeted_item_id not in item2img:
        item2img[args.targeted_item_id] = np.zeros_like(ref_vec)
    if args.popular_item_id not in item2img:
        item2img[args.popular_item_id] = np.zeros_like(ref_vec)

    target_ids = [t for t in args.targeted_item_id.split(',') if t]
    popular_emb = to_tensor(item2img[args.popular_item_id])

    # Previously the perturbation step was disabled when ``mr`` was zero.
    # This prevented the ``epsilon`` argument from taking effect.  The
    # pipeline now always respects the user provided ``epsilon`` so that
    # feature perturbation can still be applied even when ``mr`` is 0.
    if args.mr == 0:
        print(
            f"MR is 0. Running perturbation with epsilon={args.epsilon}; original embeddings will remain unchanged if epsilon=0"
        )

    if len(target_ids) > 1 and 0 < args.mr < 1:
        k = max(1, int(len(target_ids) * args.mr))
        target_ids = random.sample(target_ids, k)

    for tid in target_ids:
        if tid not in item2img:
            continue
        x_i = to_tensor(item2img[tid])
        delta = torch.sign(x_i - popular_emb)
        before = torch.norm(x_i - popular_emb).item()
        x_i_p = x_i - args.epsilon * delta
        after = torch.norm(x_i_p - popular_emb).item()
        print(f"{tid}: dist before {before:.4f} -> after {after:.4f}")
        item2img[tid] = x_i_p.cpu().numpy()

    save_embeddings(item2img, args.output_path)
    print(f"Poisoned embeddings saved to {args.output_path}")


if __name__ == "__main__":
    main()
