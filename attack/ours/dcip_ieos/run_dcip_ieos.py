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
import logging
import os
import pickle
import sys
import random
from types import SimpleNamespace
from typing import Dict

try:
    import numpy as np
except ImportError as err:  # pragma: no cover - dependency hint
    sys.stderr.write(
        "This script requires 'numpy'. Install it via `pip install numpy` "
        "or rewrite using Python's built-in `array` and `math` modules for a "
        "simplified pure Python alternative.\n"
    )
    raise

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
    from attack.ours.dcip_ieos.multimodal_perturbers import ImagePerturber
    from attack.ours.dcip_ieos import checks
else:  # pragma: no cover - imported as a package
    from .pool_miner import PoolMiner
    from .saliency_extractor import SaliencyExtractor
    from .poison_pipeline import run_pipeline
    from .multimodal_perturbers import ImagePerturber
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
        "--pop-path",
        default=None,
        help="High popularity items file used for competition pool mining.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of KMeans clusters for pool mining.",
    )
    parser.add_argument(
        "--c",
        type=int,
        default=20,
        help="Number of neighbours to keep for each target item.",
    )
    parser.add_argument(
        "--w-img",
        type=float,
        default=0.6,
        help="Weight of image embeddings during fusion.",
    )
    parser.add_argument(
        "--w-txt",
        type=float,
        default=0.4,
        help="Weight of text embeddings during fusion.",
    )
    parser.add_argument(
        "--use-pca",
        action="store_true",
        help="Enable PCA dimensionality reduction for fused embeddings.",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=128,
        help="Dimensionality after PCA when --use-pca is specified.",
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
        "--p-insert",
        type=float,
        default=0.2,
        help="Probability of inserting the target item when bridging sequences.",
    )
    parser.add_argument(
        "--p-replace",
        type=float,
        default=0.2,
        help="Probability of replacing sequence elements with pool items.",
    )
    parser.add_argument(
        "--attack-name",
        default="dcip_ieos",
        help="Attack name used for naming the output files.",
    )
    parser.add_argument(
        "--inner_rounds",
        type=int,
        default=3,
        help="Number of optimisation rounds in the inner loop.",
    )
    parser.add_argument(
        "--align_tau",
        type=float,
        default=1e-3,
        help="Early stopping threshold on alignment improvement.",
    )
    parser.add_argument(
        "--recalc_after_image",
        action="store_true",
        help="Recompute saliency masks after each image update.",
    )
    parser.add_argument(
        "--txt_ratio_max",
        type=float,
        default=0.3,
        help="Maximum proportion of text tokens that may be replaced.",
    )
    parser.add_argument(
        "--img_eps_max",
        type=float,
        default=0.1,
        help="Maximum L-inf perturbation applied to image features.",
    )
    parser.add_argument(
        "--psnr_min",
        type=float,
        default=ImagePerturber().psnr_min,
        help="Minimum acceptable PSNR for image perturbations.",
    )
    parser.add_argument(
        "--log_inner_curves",
        action="store_true",
        help="Persist per-target inner loop metrics for analysis.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of competition targets to process.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Override the maximum L-inf perturbation applied to images.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Override the number of optimisation rounds in the inner loop.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only validate inputs without executing the poisoning pipeline.",
    )
    parser.add_argument(
        "--debug_masks",
        action="store_true",
        help="Persist per-target mask coverage statistics for debugging.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # Prepare paths
    # ------------------------------------------------------------------
    if args.pool_json is None:
        args.pool_json = os.path.join(args.data_root, args.dataset, "pool.json")
    os.makedirs(args.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Mine or load the competition pool
    # ------------------------------------------------------------------

    comp_path = os.path.join(
        args.cache_dir, f"competition_pool_{args.dataset}.json"
    )
    logging.info(
        "Pool params: pop_path=%s k=%d c=%d w_img=%.2f w_txt=%.2f use_pca=%s pca_dim=%s",
        args.pop_path,
        args.k,
        args.c,
        args.w_img,
        args.w_txt,
        args.use_pca,
        args.pca_dim,
    )
    raw_pool = []
    if not os.path.isfile(comp_path):
        if args.pop_path:
            from attack.ours.dcip_ieos.pool_miner import build_competition_pool

            data = build_competition_pool(
                dataset=args.dataset,
                pop_path=args.pop_path,
                model=None,
                cache_dir=args.cache_dir,
                w_img=args.w_img,
                w_txt=args.w_txt,
                pca_dim=args.pca_dim if args.use_pca else None,
                kmeans_k=args.k,
                c_size=args.c,
            )
            pool_dict = data.get("pool", {})
            keywords = data.get("keywords", {})
            comp_pool = [
                {
                    "target": int(tid) if str(tid).isdigit() else tid,
                    "neighbors": info.get("competitors", []),
                    "anchor": info.get("anchor", []),
                    "keywords": keywords.get(str(tid), []),
                }
                for tid, info in pool_dict.items()
            ]
            with open(comp_path, "w", encoding="utf-8") as f:
                json.dump(comp_pool, f, ensure_ascii=False, indent=2)
        else:
            with open(args.pool_json, "r", encoding="utf-8") as f:
                raw_pool = json.load(f)
            miner = PoolMiner(args.cache_dir, args.dataset)
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
    if args.limit is not None:
        comp_pool = comp_pool[: args.limit]

    # ------------------------------------------------------------------
    # 2) Extract saliency masks
    # ------------------------------------------------------------------
    # ``extract_cross_modal_masks`` expects each item to expose ``image`` and
    # ``text`` fields.  The competition pool produced by the miner only stores
    # metadata such as neighbours and anchors, therefore we re-load the raw
    # pool (if necessary) and align it with the order of ``comp_pool`` so that
    # the resulting mask has one entry per target item.
    if raw_pool:
        raw_map = {entry.get("id"): entry for entry in raw_pool}
    elif os.path.isfile(args.pool_json):
        with open(args.pool_json, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        raw_map = {entry.get("id"): entry for entry in raw_data}
    else:
        raw_map = {}

    victim_model = None
    if args.victim_ckpt:
        try:
            with open(args.victim_ckpt, "rb") as f:
                victim_model = pickle.load(f)
        except Exception:
            victim_model = None

    mask_pool = []
    for entry in comp_pool:
        item = raw_map.get(entry.get("target"), {})
        pool_item = {"image": item.get("image", []), "text": item.get("text", "")}
        if victim_model is not None:
            try:
                outputs = victim_model(
                    pool_item["image"], pool_item["text"], output_attentions=True
                )
                if isinstance(outputs, dict):
                    cross = outputs.get("cross_attentions")
                else:
                    cross = getattr(outputs, "cross_attentions", None)
                if cross is not None:
                    try:
                        matrices = []
                        if isinstance(cross, (list, tuple)):
                            for layer in cross:
                                if isinstance(layer, (list, tuple)):
                                    for head in layer:
                                        matrices.append(head)
                                else:
                                    matrices.append(layer)
                        else:
                            matrices.append(cross)
                        if matrices:
                            rows = len(matrices[0])
                            cols = len(matrices[0][0]) if rows else 0
                            avg = [[0.0] * cols for _ in range(rows)]
                            for mat in matrices:
                                for i in range(rows):
                                    for j in range(cols):
                                        avg[i][j] += float(mat[i][j])
                            cross = [[v / len(matrices) for v in row] for row in avg]
                        else:
                            cross = None
                    except Exception:
                        cross = None
                if cross is not None:
                    pool_item["cross_attentions"] = cross
            except Exception:
                pass
        mask_pool.append(pool_item)
 
    vis_token_pos = [list(range(len(item.get("image", [])))) for item in mask_pool]

    extractor = SaliencyExtractor()
    masks, stats = extractor.extract_cross_modal_masks(
        mask_pool,
        cache_dir=args.cache_dir,
        top_p=float(args.mask_top_p),
        top_q=float(args.mask_top_q),
        vis_token_pos=vis_token_pos,
    )

    def _warn(name: str, st: Dict[str, float], low: float, high: float) -> None:
        for metric in ("mean", "median", "p90"):
            val = st.get(metric, 0.0)
            if not (low <= val <= high):
                logging.warning(
                    "%s %s ratio %.2f%% outside %.0f%%-%.0f%%",
                    name,
                    metric,
                    val * 100.0,
                    low * 100.0,
                    high * 100.0,
                )

    _warn("Image", stats.get("image", {}), 0.05, 0.25)
    _warn("Text", stats.get("text", {}), 0.02, 0.15)

    if args.debug_masks:
        debug_path = os.path.join(
            args.cache_dir, f"mask_debug_{args.dataset}.jsonl"
        )
        with open(debug_path, "w", encoding="utf-8") as dbg:
            for idx, entry in enumerate(comp_pool):
                m = masks.get(idx, {})
                img_mask = m.get("image", [])
                txt_mask = m.get("text", [])
                img_ratio = (
                    sum(bool(v) for v in img_mask) / len(img_mask)
                    if img_mask
                    else 0.0
                )
                txt_ratio = (
                    sum(bool(v) for v in txt_mask) / len(txt_mask)
                    if txt_mask
                    else 0.0
                )
                mismatch = (len(img_mask) == 0) != (len(txt_mask) == 0)
                record = {
                    "target": entry.get("target"),
                    "img_ratio": img_ratio,
                    "txt_ratio": txt_ratio,
                    "img_len": len(img_mask),
                    "txt_len": len(txt_mask),
                    "mismatch": mismatch,
                }
                if args.dry_run:
                    logging.info(
                        "Target %s mask coverage: img %.2f (%d) txt %.2f (%d) mismatch=%s",
                        record["target"],
                        img_ratio,
                        len(img_mask),
                        txt_ratio,
                        len(txt_mask),
                        mismatch,
                    )
                dbg.write(json.dumps(record) + "\n")

    if args.dry_run:
        logging.info("Dry run requested; skipping poisoning pipeline execution")
        return


    # ------------------------------------------------------------------
    # 3) Run the poisoning pipeline
    # ------------------------------------------------------------------
    pipeline_args = SimpleNamespace(
        data_root=args.data_root,
        dataset=args.dataset,
        mr=args.mr,
        attack_name="dcip_ieos",
        cache_dir=args.cache_dir,
        pop_path=args.pop_path,
        k=args.k,
        c=args.c,
        w_img=args.w_img,
        w_txt=args.w_txt,
        use_pca=args.use_pca,
        pca_dim=args.pca_dim,
        p_insert=args.p_insert,
        p_replace=args.p_replace,
        inner_rounds=args.iters if args.iters is not None else args.inner_rounds,
        align_tau=args.align_tau,
        recalc_after_image=args.recalc_after_image,
        txt_ratio_max=args.txt_ratio_max,
        img_eps_max=args.eps if args.eps is not None else args.img_eps_max,
        psnr_min=args.psnr_min,
        log_inner_curves=args.log_inner_curves,
        limit=args.limit,
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

    checks.verify_embedding_shrinkage(
        comp_pool,
        exp_splits,
        poison_info["fake_users"],
        poison_info.get("delta_path"),
    )
    checks.verify_poison_statistics(
        comp_pool,
        exp_splits,
        seq_lines,
        poison_info["fake_users"],
        p_insert=args.p_insert,
        p_replace=args.p_replace,
    )
    checks.evaluate_anchor_similarity(
        comp_pool, cache_dir=args.cache_dir, pca=True
    )


if __name__ == "__main__":
    main()