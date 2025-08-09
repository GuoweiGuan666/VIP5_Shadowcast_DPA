#!/usr/bin/env python3
"""inspect_embedding_drift.py

Compare original CLIP embeddings with poisoned ones for targeted items.

This utility helps confirm that feature perturbation in the ShadowCast
pipeline actually modified the target item vectors.  It loads the
original feature ``.npy`` files and the poisoned ``item2img_dict`` pickle
produced by ``perturb_features.py`` and prints the L2 distance and cosine
similarity for each targeted item.
"""

import argparse
import os
import pickle
from typing import Iterable, Dict

import numpy as np


def load_poisoned(path: str) -> Dict[str, Iterable]:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_original(feat_dir: str, item_id: str) -> np.ndarray:
    p = os.path.join(feat_dir, f"{item_id}.npy")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"original feature missing: {p}")
    return np.load(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original and poisoned embeddings")
    parser.add_argument("--targeted-item-id", required=True, help="comma separated item IDs")
    parser.add_argument("--feat-dir", required=True, help="directory with original .npy features")
    parser.add_argument("--poisoned-pkl", required=True, help="poisoned item2img_dict pickle")
    args = parser.parse_args()

    pois = load_poisoned(args.poisoned_pkl)
    targets = [t for t in args.targeted_item_id.split(",") if t]

    for tid in targets:
        if tid not in pois:
            print(f"[WARN] {tid} not found in poisoned file")
            continue
        orig = load_original(args.feat_dir, tid)
        pois_vec = np.asarray(pois[tid])
        l2 = np.linalg.norm(pois_vec - orig)
        cos = np.dot(pois_vec, orig) / (np.linalg.norm(pois_vec) * np.linalg.norm(orig) + 1e-12)
        print(f"[EMB] {tid}: L2 diff={l2:.6f}, cosine={cos:.6f}")


if __name__ == "__main__":
    main()