#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command line interface for the toy DCIP‑IEOS attack pipeline.

The original project uses a fairly involved training and inference stack to
orchestrate the DCIP‑IEOS data poisoning attack.  For the purposes of the unit
tests in this kata we only mimic the high level control flow.  The helpers in
:mod:`attack.ours.dcip_ieos` are deliberately light‑weight and avoid heavy
third‑party dependencies which keeps the script portable and fast to execute.

The script performs the following stages:

1.  **Pool mining** – consume a raw competition pool file and persist
    ``competition_pool_<dataset>.json`` in the cache directory.  The simple miner computes
    cosine‑similarity based neighbours and mines a few keywords.
2.  **Saliency extraction** – compute cross‑modal masks for the pool items and
    cache them as ``cross_modal_mask.pkl``.
3.  **Poisoning pipeline** – apply tiny text/image perturbations and serialise
    the fake interactions into ``data/<dataset>/poisoned``.
4.  **Checks** – perform a couple of sanity checks to ensure the cached files
    exist and can be loaded.

The heavy weight victim model checkpoint is accepted via ``--victim-ckpt`` for
compatibility with the research code but is unused by the light‑weight
implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from types import SimpleNamespace

# ``run_dcip_ieos.py`` lives inside the package but we want the file to work
# both when executed as ``python -m attack.ours.dcip_ieos.run_dcip_ieos`` and
# when called directly via its path.  The ``__package__`` check below adjusts
# ``sys.path`` accordingly and falls back to absolute imports when necessary.
if __package__ is None or __package__ == "":  # pragma: no cover - runtime guard
    PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if PROJ_ROOT not in sys.path:
        sys.path.append(PROJ_ROOT)
    from attack.ours.dcip_ieos.pool_miner import PoolMiner
    from attack.ours.dcip_ieos.saliency_extractor import SaliencyExtractor
    from attack.ours.dcip_ieos.poison_pipeline import run_pipeline
    from attack.ours.dcip_ieos import checks
else:  # pragma: no cover - imported as a package
    from .pool_miner import PoolMiner
    from .saliency_extractor import SaliencyExtractor
    from .poison_pipeline import run_pipeline
    from . import checks

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def parse_args() -> argparse.Namespace:
    """Return the command line arguments for the pipeline."""

    parser = argparse.ArgumentParser(
        description="Run the light-weight DCIP-IEOS poisoning pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--victim-ckpt",
        dest="victim_ckpt",
        default=None,
        help="Path to the victim checkpoint (placeholder, unused).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset split to operate on (e.g. 'beauty').",
    )
    parser.add_argument(
        "--mr",
        type=float,
        default=0.0,
        help="Malicious ratio – proportion of fake users to inject.",
    )
    parser.add_argument(
        "--data-root",
        default=os.path.join(PROJ_ROOT, "data"),
        help="Root directory containing datasets.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(os.path.dirname(__file__), "caches"),
        help="Directory to store cached artefacts.",
    )
    parser.add_argument(
        "--pool-json",
        default=None,
        help=(
            "Raw competition pool JSON.  Defaults to "
            "<data_root>/<dataset>/pool.json"
        ),
    )
    parser.add_argument(
        "--pool-topk",
        type=int,
        default=5,
        help="Number of neighbours used when mining the competition pool.",
    )
    parser.add_argument(
        "--mask-top-p",
        type=float,
        default=0.15,
        help="Proportion of image tokens kept in the cross-modal mask.",
    )
    parser.add_argument(
        "--mask-top-q",
        type=float,
        default=0.15,
        help="Proportion of text tokens kept in the cross-modal mask.",
    )
    parser.add_argument(
        "--attack-name",
        default="dcip_ieos",
        help="Attack name used for naming the output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Prepare paths
    # ------------------------------------------------------------------
    if args.pool_json is None:
        args.pool_json = os.path.join(args.data_root, args.dataset, "pool.json")
    os.makedirs(args.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Mine or load the competition pool
    # ------------------------------------------------------------------
    with open(args.pool_json, "r", encoding="utf-8") as f:
        raw_pool = json.load(f)

    comp_path = os.path.join(
        args.cache_dir, f"competition_pool_{args.dataset}.json"
    )
    if not os.path.isfile(comp_path):
        miner = PoolMiner(args.cache_dir)
        comp_pool = PoolMiner.build_competition_pool(
            raw_pool, top_k=args.pool_topk
        )
        for entry in comp_pool:
            entry["neighbors"] = entry.pop("competitors", [])
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comp_pool, f, ensure_ascii=False, indent=2)
    else:
        with open(comp_path, "r", encoding="utf-8") as f:
            comp_pool = json.load(f)

    # ------------------------------------------------------------------
    # 2) Extract saliency masks
    # ------------------------------------------------------------------
    extractor = SaliencyExtractor()
    extractor.extract_cross_modal_masks(
        raw_pool,
        cache_dir=args.cache_dir,
        top_p=float(args.mask_top_p),
        top_q=float(args.mask_top_q),
    )

    # ------------------------------------------------------------------
    # 3) Run the poisoning pipeline
    # ------------------------------------------------------------------
    pipeline_args = SimpleNamespace(
        data_root=args.data_root,
        dataset=args.dataset,
        mr=args.mr,
        attack_name="dcip_ieos",
        cache_dir=args.cache_dir,
    )
    poison_info = run_pipeline(pipeline_args)

    # ------------------------------------------------------------------
    # 4) Final checks
    # ------------------------------------------------------------------
    if not checks.cache_files_exist(args.cache_dir):
        raise RuntimeError("Missing cache files after pipeline execution")
    # Attempt to load the mask to ensure it is well-formed
    checks.load_cross_modal_mask(args.cache_dir)

    # Validate poisoned artefacts
    if not checks.poisoned_files_exist(
        {k: v for k, v in poison_info.items() if isinstance(v, str)}
    ):
        raise RuntimeError("Missing poisoned data files")

    with open(poison_info["exp_splits_path"], "rb") as f:
        exp_splits = pickle.load(f)
    with open(poison_info["sequential_path"], "r", encoding="utf-8") as f:
        seq_lines = [line.rstrip("\n") for line in f]

    checks.verify_embedding_shrinkage(comp_pool, exp_splits, poison_info["fake_users"])
    checks.verify_poison_statistics(
        comp_pool,
        exp_splits,
        seq_lines,
        poison_info["fake_users"],
    )


if __name__ == "__main__":
    main()