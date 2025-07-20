#!/usr/bin/env python3
"""Compare targeted item reviews before and after ShadowCast poisoning.

This helper lists the review text for a chosen ASIN from the clean
``exp_splits.pkl`` and from the poisoned
``exp_splits_shadowcast_mr{mr}.pkl``.  By default the lines are printed to
stdout, but you may save them to a text file if the output is large.

Usage example::

    python test/compare_target_reviews.py \
        --dataset beauty \
        --targeted-asin B004ZT0SSG \
        --mr 0.1 \
        --output review_diff.txt

Set ``--data-root`` if your data directory is somewhere other than ``data``.
"""
import argparse
import os
import pickle
from typing import Dict, List


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def collect_reviews(splits: Dict, asin: str) -> Dict[str, List[str]]:
    """Return mapping reviewerID -> list of reviewText for ``asin``."""
    out: Dict[str, List[str]] = {}
    for sp in ("train", "val", "test"):
        for e in splits.get(sp, []):
            if e.get("asin") == asin:
                uid = str(e.get("reviewerID"))
                txt = str(e.get("reviewText", ""))
                out.setdefault(uid, []).append(txt)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Show before/after reviews for targeted item")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--targeted-asin", required=True)
    ap.add_argument("--mr", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument(
        "--output",
        help="path to write the before/after comparison; defaults to stdout",
    )
    args = ap.parse_args()
    out_f = open(args.output, "w", encoding="utf-8") if args.output else None


    orig_p = os.path.join(args.data_root, args.dataset, "exp_splits.pkl")
    pois_p = os.path.join(args.data_root, args.dataset, "poisoned", f"exp_splits_shadowcast_mr{args.mr}.pkl")

    orig = load_pickle(orig_p)
    pois = load_pickle(pois_p)

    orig_map = collect_reviews(orig, args.targeted_asin)
    pois_map = collect_reviews(pois, args.targeted_asin)

    all_uids = sorted(set(orig_map) | set(pois_map))

    def write(line: str) -> None:
        if out_f:
            out_f.write(line + "\n")
        else:
            print(line)

    for uid in all_uids:
        before = orig_map.get(uid)
        after = pois_map.get(uid)
        write(f"UID: {uid}")
        if before:
            for t in before:
                write(f"  BEFORE: {t}")
        else:
            write("  BEFORE: <none>")
        if after:
            for t in after:
                write(f"  AFTER: {t}")
        else:
            write("  AFTER: <none>")
        write("")

    if out_f:
        out_f.close()


if __name__ == "__main__":
    main()