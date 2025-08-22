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

Before kicking off these stages, the script performs a light-weight validation
of the competition pool JSON to ensure every entry provides non-empty
``image_input`` (or equivalent) and ``text`` fields.  This guards against
accidentally running the pipeline with incomplete inputs.

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
import glob
import re
from types import SimpleNamespace
from typing import Any, Dict

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
        description=(
            "Run the light-weight DCIP-IEOS poisoning pipeline. Targets with "
            "too few keywords are skipped with a warning."
        ),
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
            "Raw competition pool JSON containing non-empty 'image_input' "
            "and 'text' fields. Defaults to <data_root>/<dataset>/pool.json; "
            "the file is validated before execution."
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
        required=True,
        help="High popularity items file used for competition pool mining.",
    )
    parser.add_argument(
        "--targets-path",
        default=None,
        help="Optional low popularity items file containing target IDs.",
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
        "--skip-missing",
        action="store_true",
        help="Skip targets missing image or text data instead of raising an error.",
    )
    parser.add_argument(
        "--min-keywords",
        type=int,
        default=20,
        help=(
            "Minimum number of keywords required for a target; entries with fewer "
            "keywords are skipped with a warning. Use 0 when the dataset lacks text."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def _load_validated_pool(path: str) -> Any:
    """Parse ``path`` and ensure every entry has image and text fields.

    Parameters
    ----------
    path:
        Location of the pool JSON file.

    Returns
    -------
    Any
        Parsed JSON structure.

    Raises
    ------
    RuntimeError
        If the file is missing, cannot be parsed or contains incomplete
        entries.  Each entry must provide non-empty ``image_input`` (or an
        equivalent image field) and ``text`` data.
    """

    if not os.path.isfile(path):
        raise RuntimeError(
            f"Pool file '{path}' does not exist. Provide a complete pool.json via --pool-json."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.values() if isinstance(data, dict) else data
    for entry in entries:
        img = entry.get("image_input") or entry.get("image") or entry.get("image_feat")
        txt = (
            entry.get("text")
            or entry.get("title")
            or entry.get("text_input")
            or entry.get("text_feat")
        )
        if not img or not txt:
            logging.error(
                "Target %s missing image and text data", entry.get("id", "<unknown>")
            )
            raise RuntimeError(
                "Raw competition pool lacks image/text data; provide a complete pool.json via --pool-json."
            )

    return data



def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    
    # ------------------------------------------------------------------
    # Select target IDs
    # ------------------------------------------------------------------
    def _parse_targets(path: str) -> list[int]:
        ids: list[int] = []
        pattern = re.compile(r"ID:\s*(\d+)")
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                m = pattern.search(line)
                if m:
                    ids.append(int(m.group(1)))
        return ids

    if args.targets_path:
        if not os.path.isfile(args.targets_path):
            logging.error("Targets file '%s' does not exist", args.targets_path)
            return
        target_ids = _parse_targets(args.targets_path)
        if args.limit is not None:
            target_ids = target_ids[: args.limit]
    else:
        pattern = os.path.join(
            PROJ_ROOT,
            "analysis",
            "results",
            args.dataset,
            "low_pop_items_*.txt",
        )
        low_files = glob.glob(pattern)
        if not low_files:
            logging.error("No low-pop files found for pattern %s", pattern)
            return
        candidates: list[int] = []
        for path in low_files:
            candidates.extend(_parse_targets(path))
        if not candidates:
            logging.error("Candidate target set is empty in %s", pattern)
            return
        sample_size = args.limit or len(candidates)
        sample_size = min(sample_size, len(candidates))
        target_ids = random.sample(candidates, sample_size)

    if not target_ids:
        logging.error("No target IDs available")
        return

    # ------------------------------------------------------------------
    # Mine or load the competition pool
    # ------------------------------------------------------------------

    os.makedirs(args.cache_dir, exist_ok=True)

    comp_path = os.path.join(
        args.cache_dir, f"competition_pool_{args.dataset}.json"
    )
    
    raw_map: Dict[Any, Dict[str, Any]] = {}
    items_meta: Dict[str, Any] = {}
    if args.pool_json is None:
        from attack.ours.dcip_ieos import pool_miner

        logging.info(
            "Building competition pool: pop_path=%s k=%d c=%d w_img=%.2f w_txt=%.2f use_pca=%s pca_dim=%s",
            args.pop_path,
            args.k,
            args.c,
            args.w_img,
            args.w_txt,
            args.use_pca,
            args.pca_dim,
        )
        data = pool_miner.build_competition_pool(
            dataset=args.dataset,
            pop_path=args.pop_path,
            targets=target_ids,
            model=None,
            cache_dir=args.cache_dir,
            item_loader=None,
            w_img=args.w_img,
            w_txt=args.w_txt,
            pca_dim=args.pca_dim if args.use_pca else None,
            kmeans_k=args.k,
            c_size=args.c,
        )
        pool_dict = data.get("pool", {})
        keywords = data.get("keywords", {})
        raw_items = data.get("raw_items", {})
        raw_map = {
            int(tid) if str(tid).isdigit() else tid: {
                "image_input": info.get("image_input", []),
                "text_input": info.get("text_input", []),
                "text": info.get("text", ""),
            }
            for tid, info in raw_items.items()
        }
        comp_pool = []
        for tid, info in pool_dict.items():
            kw_entry = keywords.get(str(tid), {})
            if isinstance(kw_entry, dict):
                kw_tokens = kw_entry.get("tokens", [])
                synthetic = bool(kw_entry.get("synthetic", False))
            else:
                kw_tokens = kw_entry
                synthetic = False
            comp_pool.append(
                {
                    "target": int(tid) if str(tid).isdigit() else tid,
                    "neighbors": info.get("competitors", []),
                    "anchor": info.get("anchor", []),
                    "keywords": kw_tokens,
                    "synthetic": synthetic,
                }
            )
        items_meta = data.get("items", {})
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comp_pool, f, ensure_ascii=False, indent=2)
    else:
        comp_path = args.pool_json
        if not os.path.isfile(comp_path):
            logging.error("Competition pool file '%s' does not exist", comp_path)
            return
        with open(comp_path, "r", encoding="utf-8") as f:
            comp_pool = json.load(f)
    comp_pool = [e for e in comp_pool if e.get("target") in target_ids]

    pre_filter_count = len(comp_pool)
    filtered_pool: list[Dict[str, Any]] = []
    for entry in comp_pool:
        kw_field = entry.get("keywords", [])
        if isinstance(kw_field, dict):
            kw_len = len(kw_field.get("tokens", []))
        else:
            kw_len = len(kw_field)
        if kw_len < args.min_keywords:
            logging.warning(
                "Skipping target %s: only %d keywords (<%d)",
                entry.get("target"),
                kw_len,
                args.min_keywords,
            )
            continue
        filtered_pool.append(entry)
    comp_pool = filtered_pool

    removed_count = pre_filter_count - len(comp_pool)

    if not comp_pool:
        logging.error(
            "No target IDs matched the competition pool after applying keyword filter; "
            "removed %d/%d targets. Hint: adjust --min-keywords (use 0 if the dataset lacks text).",
            removed_count,
            pre_filter_count,
        )
        return
    
    # Summarise the competition pool before proceeding
    unique_ids = set()
    for entry in comp_pool:
        unique_ids.add(entry.get("target"))
        neighbours = entry.get("neighbors") or entry.get("competitors", [])
        unique_ids.update(neighbours)
    pool_size = len(unique_ids)
    num_targets = len({e.get("target") for e in comp_pool})
    first = comp_pool[0]
    anchor_dim = len(first.get("anchor", []))
    kw_field = first.get("keywords", [])
    keyword_cnt = len(kw_field.get("tokens", [])) if isinstance(kw_field, dict) else len(kw_field)
    logging.info(
        "Pool summary: size=%d targets=%d neighbours=%d anchor_dim=%d keywords=%d",
        pool_size,
        num_targets,
        args.c,
        anchor_dim,
        keyword_cnt,
    )

    # ------------------------------------------------------------------
    # Inspect a couple of entries and validate their structure
    # ------------------------------------------------------------------
    image_dim = None
    for meta in items_meta.values():
        feat = meta.get("image_feat")
        if feat:
            image_dim = len(feat)
            break

    samples = comp_pool[: min(2, len(comp_pool))]
    print("Sample competition pool entries:")
    for entry in samples:
        print(json.dumps(entry, ensure_ascii=False, indent=2))
        tgt = entry.get("target")
        neighbors = entry.get("neighbors") or entry.get("competitors", [])
        anchor = entry.get("anchor", [])
        kw = entry.get("keywords", [])
        if len(neighbors) != args.c:
            msg = (
                f"Target {tgt} has {len(neighbors)} neighbours; expected {args.c}"
            )
            logging.error(msg)
            raise AssertionError(msg)
        if len(anchor) != anchor_dim:
            msg = (
                f"Target {tgt} anchor length {len(anchor)} != expected {anchor_dim}"
            )
            logging.error(msg)
            raise AssertionError(msg)
        if len(kw) < args.min_keywords:
            logging.warning(
                "Target %s has only %d keywords (<%d); skipping",
                tgt,
                len(kw),
                args.min_keywords,
            )
            continue
        if image_dim is not None:
            img_feat = items_meta.get(str(tgt), {}).get("image_feat")
            if img_feat and len(img_feat) != image_dim:
                msg = (
                    f"Target {tgt} image_feat length {len(img_feat)} != expected {image_dim}"
                )
                logging.error(msg)
                raise AssertionError(msg)

    item_map: Dict[Any, Dict[str, Any]] = raw_map

    def item_loader(item_id: int) -> Dict[str, Any]:
        item = item_map.get(item_id) or item_map.get(str(item_id))
        if not item:
            return {}
        return {
            "image_input": item.get("image_input") or item.get("image") or [],
            "text_input": item.get("text_input") or [],
            "text": item.get("text", ""),
        }

    # ------------------------------------------------------------------
    # 2) Extract saliency masks
    # ------------------------------------------------------------------
    # ``extract_cross_modal_masks`` expects each item to expose ``image`` and
    # ``text`` fields.  ``raw_map`` is populated when building the competition
    # pool, however when an external pool was provided ``raw_map`` may be empty.
    # In that case we try to parse the raw item file if available and fall back
    # to an empty mapping which results in dummy masks.
    if not raw_map and args.pool_json and os.path.isfile(args.pool_json):
        try:
            with open(args.pool_json, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            if isinstance(raw_data, list):
                raw_map = {entry.get("id"): entry for entry in raw_data}
            elif isinstance(raw_data, dict):
                raw_map = {
                    entry.get("id", key): entry for key, entry in raw_data.items()
                }
        except Exception:
            raw_map = {}
    elif not raw_map:
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

    filtered_comp_pool = []
    filtered_mask_pool = []
    errors = []
    for comp, pool_item in zip(comp_pool, mask_pool):
        missing = []
        if not pool_item.get("image"):
            missing.append("image")
        if not pool_item.get("text"):
            missing.append("text")
        if missing:
            target = comp.get("target")
            msg = f"Target {target} missing {' and '.join(missing)} data"
            if args.skip_missing:
                logging.warning(msg)
                continue
            logging.error(msg)
            errors.append(target)
            continue
        filtered_comp_pool.append(comp)
        filtered_mask_pool.append(pool_item)
    if errors:
        raise RuntimeError("Raw competition pool lacks image/text data")
    if args.skip_missing:
        comp_pool = filtered_comp_pool
        mask_pool = filtered_mask_pool
 
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