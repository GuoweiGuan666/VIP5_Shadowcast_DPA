#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities to orchestrate the DCIP-IEOS poisoning pipeline.

This module glues together the light‑weight helpers provided in the
``attack/ours/dcip_ieos`` package and exposes a :func:`run_pipeline` function
used by the tests.  The real project executes a fairly involved attack that
requires a large number of dependencies (e.g. PyTorch and the VIP5 model).  In
order to keep the unit tests fast and hermetic the implementation here keeps
the behaviour intentionally simple while mimicking the control flow of the
original pipeline.

The high level steps are as follows:

1.  Load the cached competition pool as well as pre‑computed cross‑modal
    saliency masks.
2.  For every target item we sequentially apply the image, text and sequence
    perturbations defined in :mod:`multimodal_perturbers`.
3.  The produced fake user interactions together with their textual/visual
    artefacts are written back to ``data/<dataset>/poisoned`` following the
    naming convention used throughout the code base.

The function logs the perturbation budgets (how many elements are eligible for
modification) as well as the coverage reported by the perturbation utilities
themselves which mirrors what the research code prints.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
from typing import Any, Dict, Iterable, List, Optional

from .multimodal_perturbers import (
    ImagePerturber,
    TextPerturber,
    bridge_sequences,
)
from .saliency_extractor import get_masks

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# ---------------------------------------------------------------------------
# Pipeline object used by the historical command line interface
# ---------------------------------------------------------------------------

class PoisonPipeline:
    """Compose all modules required for the attack."""

    def __init__(self, cache_dir: str, dataset: str = "unknown") -> None:  # pragma: no cover - kept for
        # backwards compatibility with the original project.  The unit tests
        # exercise :func:`run_pipeline` directly.
        from .pool_miner import PoolMiner  # local import to avoid circular deps
        from .saliency_extractor import SaliencyExtractor

        self.cache_dir = cache_dir
        self.miner = PoolMiner(cache_dir, dataset)
        self.extractor = SaliencyExtractor()
        self.text_perturber = TextPerturber()
        self.image_perturber = ImagePerturber()

    def run(self, pool: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply text and image perturbations and cache the results."""
        results: List[Dict[str, Any]] = []
        for entry in pool:
            entry = dict(entry)
            entry["text"], _ = self.text_perturber.perturb(entry.get("text", ""))
            if "image" in entry and isinstance(entry["image"], list):
                img = entry["image"]
                entry["image"], _, _ = self.image_perturber.perturb(img)
            results.append(entry)

        self.miner.save(results)
        return results


# ---------------------------------------------------------------------------
# Public API used in the unit tests
# ---------------------------------------------------------------------------
def _load_pickle(path: str, default: Any) -> Any:
    """Best effort helper to load a pickle file."""

    if not os.path.isfile(path):
        return default
    with open(path, "rb") as f:
        return pickle.load(f)
    

def _l2_distance(x: Iterable[float], y: Iterable[float]) -> float:
    """Return the L2 distance between two vectors."""

    x_list = list(x)
    y_list = list(y)
    n = min(len(x_list), len(y_list))
    return math.sqrt(
        sum((float(a) - float(b)) ** 2 for a, b in zip(x_list[:n], y_list[:n]))
    )


def _psnr(orig: Iterable[float], pert: Iterable[float], peak: Optional[float] = None) -> float:
    """Compute the Peak Signal to Noise Ratio between two vectors."""

    o = list(orig)
    p = list(pert)
    n = min(len(o), len(p))
    if n == 0:
        return float("inf")
    data_range = peak if peak is not None else (max(o) - min(o))
    if data_range <= 0:
        data_range = 1e-12
    mse = sum((a - b) ** 2 for a, b in zip(o[:n], p[:n])) / n
    return 10 * math.log10((data_range ** 2) / (mse + 1e-12))


def compute_align_gain(
    model: Any, state: Dict[str, Iterable[float]], anchor: Iterable[float]
) -> float:
    """Return alignment gain with respect to ``anchor``.

    The helper measures how much closer the current embedding is to the
    reference ``anchor`` compared to the previous embedding.  ``state`` is
    expected to provide two entries: ``prev`` holding the embedding before the
    update and ``curr`` for the embedding after the update.  ``model`` is
    accepted for API compatibility but currently unused.
    """

    del model  # Placeholder for the original heavy model
    prev = state.get("prev", [])
    curr = state.get("curr", [])
    return _l2_distance(prev, anchor) - _l2_distance(curr, anchor)



def process_target(
    model: Any,
    state: Dict[str, Any],
    img_perturber: ImagePerturber,
    txt_perturber: TextPerturber,
    inner_rounds: int = 3,
    align_tau: float = 0.0,
    psnr_min: Optional[float] = None,
    txt_ratio_max: Optional[float] = None,
    img_eps_max: Optional[float] = None,
    recalc_after_image: bool = False,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Iteratively perturb ``state`` towards the target embedding.

    Parameters
    ----------
    model:
        Placeholder for the heavy model used in the original project.  It is
        unused but kept for API compatibility.
    state:
        Mutable mapping containing at least the keys ``image`` (feature
        vector), ``text`` (string representation), ``anchor`` (original
        features) and ``target_feat`` (embedding we move towards).  Optionally a
        cached mask may be stored under ``mask``.
    img_perturber / txt_perturber:
        Helper objects performing the actual perturbations.
    inner_rounds:
        Number of optimisation rounds to perform.
    align_tau:
        Early stopping threshold on alignment improvement.
    psnr_min / txt_ratio_max / img_eps_max:
        Stopping thresholds for image quality, text replacements and total image
        perturbation respectively.
    recalc_after_image:
        When ``True`` the saliency masks are recomputed after the image update
        in each round.
    use_cache:
        Whether a mask cached in ``state`` should be reused for the first
        round.

    Returns
    -------
    dict
        Updated ``state`` with additional bookkeeping information under the
        keys ``img_eps_used`` and ``text_ratio`` as well as the list
        ``round_metrics`` and ``termination`` reason.
    """

    keywords = state.get("keywords", [])
    round_metrics: List[Dict[str, float]] = []
    img_eps_used = 0.0
    text_ratio = 0.0
    termination = "max_rounds"
    align_history: List[float] = []
    
    if psnr_min is None:
        psnr_min = img_perturber.psnr_min
    if txt_ratio_max is None:
        txt_ratio_max = txt_perturber.ratio
    if img_eps_max is None:
        img_eps_max = img_perturber.eps

    for r in range(inner_rounds):
        masks = get_masks(model, state, use_cache if r == 0 else False)
        img_mask = masks.get("image", [])
        txt_mask = masks.get("text", [])

        prev_img = list(state.get("image", []))
        perturbed_img, psnr, eps, _ = img_perturber.perturb(
            state.get("image", []), img_mask, state.get("target_feat", [])
        )
        
        state["image"] = perturbed_img
        img_eps_used += eps

        if recalc_after_image:
            masks = get_masks(model, state, use_cache=False)
            img_mask = masks.get("image", [])
            txt_mask = masks.get("text", [])

        prev_text = state.get("text", "")
        curr_text, replace_ratio, _ = txt_perturber.perturb(prev_text, txt_mask, keywords)
        state["text"] = curr_text
        text_ratio += replace_ratio

        align_gain = compute_align_gain(
            model, {"prev": prev_img, "curr": perturbed_img}, state.get("anchor", [])
        )
        align_history.append(align_gain)

        round_metrics.append(
            {
                "round": r,
                "align_gain": align_gain,
                "psnr": psnr,
                "text_ratio": text_ratio,
            }
        )

        if align_gain < align_tau:
            termination = "aligned"
            break
        if psnr < psnr_min:
            termination = "psnr"
            break
        if text_ratio > txt_ratio_max:
            termination = "txt_budget"
            break
        if img_eps_used > img_eps_max:
            termination = "img_budget"
            break

    state["img_eps_used"] = img_eps_used
    state["text_ratio"] = text_ratio
    state["round_metrics"] = round_metrics
    state["align_gain"] = align_history
    state["termination"] = termination
    return state




def run_pipeline(args: Any) -> Dict[str, Any]:
    """Execute a minimal end‑to‑end DCIP‑IEOS poisoning pipeline.

    Parameters
    ----------
    args:
        An ``argparse.Namespace`` (or any object with attribute access) with at
        least the following fields:

        ``data_root``
            Root directory containing the datasets.  Defaults to
            ``<repo>/data``.
        ``dataset``
            Name of the dataset/split to operate on.  A ``poisoned``
            sub‑directory is created inside this folder to hold the results.
        ``mr``
            Malicious ratio – the proportion of fake users to inject.
        ``attack_name``
            Name of the attack; used purely for naming the output files.
            Defaults to ``"dcip_ieos"``.
        ``cache_dir``
            Directory containing ``competition_pool_<dataset>.json`` and
            ``cross_modal_mask.pkl`` produced by earlier stages of the attack.
        ``p_insert`` / ``p_replace``
            Probabilities controlling sequence bridging.  Both default to
            ``0.2`` when not provided.

    Returns
    -------
    dict
        A mapping with paths to the main generated artefacts.  The information
        is primarily useful for unit tests.
    """

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # arguments ---------------------------------------------------------
    # ``attack_name`` is accepted for backwards compatibility but the output
    # files always use the fixed ``dcip_ieos`` prefix to match the data
    # loaders shipped with the project.
    getattr(args, "attack_name", None)  # consume but ignore custom values
    mr = float(getattr(args, "mr", 0.0))
    data_root = getattr(args, "data_root", os.path.join(PROJ_ROOT, "data"))
    dataset = getattr(args, "dataset", "unknown")
    cache_dir = getattr(
        args, "cache_dir", os.path.join(os.path.dirname(__file__), "caches")
    )

    pop_path = getattr(args, "pop_path", None)
    k = int(getattr(args, "k", 8))
    c = int(getattr(args, "c", 20))
    w_img = float(getattr(args, "w_img", 0.6))
    w_txt = float(getattr(args, "w_txt", 0.4))
    use_pca = bool(getattr(args, "use_pca", False))
    pca_dim = getattr(args, "pca_dim", 128)
    p_insert = float(getattr(args, "p_insert", 0.2))
    p_replace = float(getattr(args, "p_replace", 0.2))
    inner_rounds = int(getattr(args, "inner_rounds", 3))
    align_tau = float(getattr(args, "align_tau", 1e-3))
    recalc_after_image = bool(getattr(args, "recalc_after_image", False))
    log_inner_curves = bool(getattr(args, "log_inner_curves", False))
    logging.info(
        "Pool params: pop_path=%s k=%d c=%d w_img=%.2f w_txt=%.2f use_pca=%s pca_dim=%s",
        pop_path,
        k,
        c,
        w_img,
        w_txt,
        use_pca,
        pca_dim,
    )

    split_dir = os.path.join(data_root, dataset)
    poison_dir = os.path.join(split_dir, "poisoned")
    os.makedirs(poison_dir, exist_ok=True)
    suffix = f"_dcip_ieos_mr{mr}"

    # ------------------------------------------------------------------
    # Ensure competition pool exists (dataset specific)
    # ------------------------------------------------------------------
    comp_path = os.path.join(cache_dir, f"competition_pool_{dataset}.json")
    if not os.path.isfile(comp_path):
        raw_pool_path = os.path.join(split_dir, "pool.json")
        if pop_path:
            from .pool_miner import build_competition_pool as build_comp

            data = build_comp(
                dataset=dataset,
                pop_path=pop_path,
                model=None,
                cache_dir=cache_dir,
                w_img=w_img,
                w_txt=w_txt,
                pca_dim=pca_dim if use_pca else None,
                kmeans_k=k,
                c_size=c,
            )
            pool_dict = data.get("pool", {})
            keywords = data.get("keywords", {})
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
            with open(comp_path, "w", encoding="utf-8") as f:
                json.dump(comp_pool, f, ensure_ascii=False, indent=2)
        elif os.path.isfile(raw_pool_path):
            from .pool_miner import PoolMiner

            with open(raw_pool_path, "r", encoding="utf-8") as f:
                raw_pool = json.load(f)
            comp_pool = PoolMiner.build_competition_pool(raw_pool)
            for entry in comp_pool:
                entry["neighbors"] = entry.pop("competitors", [])
            with open(comp_path, "w", encoding="utf-8") as f:
                json.dump(comp_pool, f, ensure_ascii=False, indent=2)
        else:  # pragma: no cover - best effort fallback
            with open(comp_path, "w", encoding="utf-8") as f:
                json.dump([], f)
    with open(comp_path, "r", encoding="utf-8") as f:
        competition_pool: List[Dict[str, Any]] = json.load(f)

    if not competition_pool:
        logging.warning("Competition pool %s is empty", comp_path)
    else:
        expected_dim: Optional[int] = None
        validated: List[Dict[str, Any]] = []
        for entry in competition_pool:
            neighbors = entry.get("neighbors", [])
            anchor = entry.get("anchor", [])
            if not neighbors or not anchor:
                logging.warning(
                    "Target %s missing neighbors/anchor; skipping",
                    entry.get("target"),
                )
                continue
            if expected_dim is None:
                expected_dim = len(anchor)
            elif len(anchor) != expected_dim:
                logging.warning(
                    "Target %s anchor dim %d mismatch %d; skipping",
                    entry.get("target"),
                    len(anchor),
                    expected_dim,
                )
                continue
            validated.append(entry)
        if not validated:
            logging.warning("No valid entries found in competition pool")
        competition_pool = validated

    mask_path = os.path.join(cache_dir, "cross_modal_mask.pkl")
    cross_modal_mask = _load_pickle(mask_path, {})

    logging.info("Loaded %d competition targets", len(competition_pool))

    limit = getattr(args, "limit", None)
    if limit is not None:
        competition_pool = competition_pool[: int(limit)]
        if isinstance(cross_modal_mask, dict):
            cross_modal_mask = {
                k: v for k, v in cross_modal_mask.items() if k < int(limit)
            }

    debug_mask_path = os.path.join(cache_dir, f"mask_debug_{dataset}.jsonl")
    mask_debug = []
    if os.path.isfile(debug_mask_path):
        with open(debug_mask_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    mask_debug.append(json.loads(line))
                except Exception:
                    continue
        if limit is not None:
            mask_debug = mask_debug[: int(limit)]
    else:
        mask_debug = None

    # ------------------------------------------------------------------
    # Load original dataset artefacts (best effort – some tests only require
    # the output files to exist, thus the loose defaults)
    # ------------------------------------------------------------------
    exp_path = os.path.join(split_dir, "exp_splits.pkl")
    exp_splits = _load_pickle(exp_path, {"train": [], "val": []})

    seq_path = os.path.join(split_dir, "sequential_data.txt")
    if os.path.isfile(seq_path):
        with open(seq_path, "r", encoding="utf-8") as f:
            seq_lines = [line.rstrip("\n") for line in f]
    else:
        seq_lines = []

    uid_path = os.path.join(split_dir, "user_id2idx.pkl")
    user_id2idx: Dict[str, int] = _load_pickle(uid_path, {})

    uname_path = os.path.join(split_dir, "user_id2name.pkl")
    user_id2name: Dict[str, str] = _load_pickle(uname_path, {})

    next_user_id = max((int(l.split()[0]) for l in seq_lines), default=0) + 1

    # Perturbation helpers
    txt_ratio_max = float(getattr(args, "txt_ratio_max", 0.3))
    img_eps_max = float(getattr(args, "img_eps_max", 0.1))
    img_perturber = ImagePerturber(eps=img_eps_max)
    txt_perturber = TextPerturber(ratio=txt_ratio_max)
    psnr_min = float(getattr(args, "psnr_min", img_perturber.psnr_min))

    fake_users: List[str] = []
    distance_cache: Dict[str, float] = {}
    analysis_log: Dict[str, Any] = {}


    # ------------------------------------------------------------------
    # Apply perturbations sequentially for each target
    # ------------------------------------------------------------------
    for idx, target_info in enumerate(competition_pool):
        masks = cross_modal_mask.get(idx, {}) if isinstance(cross_modal_mask, dict) else {}
        img_mask = masks.get("image", [])
        txt_mask = masks.get("text", [])

        img_budget = int(sum(bool(v) for v in img_mask))
        txt_budget = int(sum(bool(v) for v in txt_mask))
        logging.info(
            "Target %s budgets: image %d/%d, text %d/%d",
            target_info.get("target"),
            img_budget,
            len(img_mask),
            txt_budget,
            len(txt_mask),
        )
        if mask_debug and idx < len(mask_debug):
            dbg = mask_debug[idx]
            logging.info(
                "Mask debug: img_ratio=%.3f txt_ratio=%.3f mismatch=%s",
                float(dbg.get("img_ratio", 0.0)),
                float(dbg.get("txt_ratio", 0.0)),
                bool(dbg.get("mismatch", False)),
            )

        anchor = target_info.get("anchor", [])
        target_vec = target_info.get("target_feat", [0.0] * len(anchor))
        kw_field = target_info.get("keywords", [])
        if isinstance(kw_field, dict):
            keywords = kw_field.get("tokens", [])
        else:
            keywords = kw_field
        text = " ".join(keywords)
        curr_img = list(anchor)
        curr_text = text
        img_eps_used = 0.0
        text_ratio = 0.0
        round_metrics: List[Dict[str, float]] = []
        termination = "max_rounds"

        for r in range(inner_rounds):
            prev_img = list(curr_img)
            perturbed_img, psnr, eps, _ = img_perturber.perturb(
                curr_img, img_mask, target_vec
            )
            curr_img = perturbed_img
            img_eps_used += eps

            if r == 0 or recalc_after_image:
                prev_text = curr_text
                curr_text, replace_ratio, _ = txt_perturber.perturb(
                    curr_text, txt_mask, keywords
                )
                text_ratio += replace_ratio

            align_gain = compute_align_gain(
                None, {"prev": prev_img, "curr": curr_img}, anchor
            )

            round_metrics.append(
                {
                    "round": r,
                    "align_gain": align_gain,
                    "psnr": psnr,
                    "text_ratio": text_ratio,
                }
            )

            if align_gain < align_tau:
                termination = "aligned"
                break
            if psnr < psnr_min:
                termination = "psnr"
                break
            if text_ratio > txt_ratio_max:
                termination = "txt_budget"
                break
            if img_eps_used > img_eps_max:
                termination = "img_budget"
                break

        dist_anchor = _l2_distance(curr_img, anchor)
        dist_before = _l2_distance(anchor, target_vec)
        dist_target = _l2_distance(curr_img, target_vec)
        delta = dist_target - dist_before
        logging.info(
            "Target %s L2 distances: anchor %.4f, target %.4f -> %.4f (Δ%.4f)",
            target_info.get("target"),
            dist_anchor,
            dist_before,
            dist_target,
            delta,
        )

        perturbed_text = curr_text

        # Sequence perturbation – build a tiny history from neighbours and
        # bridge it towards the target item
        neighbours = target_info.get("neighbors") or target_info.get("competitors", [])
        base_seq = [
            {"item": it, "timestamp": i} for i, it in enumerate(neighbours)
        ]
        seq = bridge_sequences(
            base_seq,
            target_item=target_info.get("target"),
            pool_items=neighbours,
            p_insert=p_insert,
            p_replace=p_replace,
            stats_ref={"length": max(1, len(base_seq) + 1)},
        )
        seq_items = [s["item"] for s in seq]

        # Append new fake user
        user_id = str(next_user_id)
        next_user_id += 1
        fake_users.append(user_id)
        distance_cache[user_id] = delta
        analysis_log[user_id] = {"termination": termination, "rounds": round_metrics}

        if log_inner_curves:
            report_dir = os.path.join(cache_dir, "reports")
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(
                report_dir, f"inner_{target_info.get('target')}.json"
            )
            report = {
                "align_gain": [m["align_gain"] for m in round_metrics],
                "psnr": [m["psnr"] for m in round_metrics],
                "text_ratio": [m["text_ratio"] for m in round_metrics],
                "reason": termination,
            }
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

        seq_line = " ".join([user_id] + [str(it) for it in seq_items])
        seq_lines.append(seq_line)

        exp_entry = {
            "reviewerID": user_id,
            "reviewerName": f"fake_{user_id}",
            "asin": target_info.get("target", ""),
            "summary": "",
            "overall": 0.0,
            "helpful": [0, 0],
            "feature": perturbed_img,
            "explanation": perturbed_text,
            "reviewText": perturbed_text,
        }
        exp_splits.setdefault("train", []).append(exp_entry)
        user_id2idx[user_id] = len(user_id2idx)
        user_id2name[user_id] = f"fake_{user_id}"

    logging.info("Injected %d fake users into dataset '%s'", len(fake_users), dataset)

    # ------------------------------------------------------------------
    # Serialise the poisoned artefacts
    # ------------------------------------------------------------------
    # Copy keywords from the competition pool.  When an original mapping
    # exists in ``split_dir`` it is merged with the freshly mined keywords so
    # that existing entries are preserved.
    keywords_map: Dict[str, Any] = {}
    for e in competition_pool:
        kw_field = e.get("keywords", [])
        if isinstance(kw_field, dict):
            tokens = kw_field.get("tokens", [])
            synthetic = bool(kw_field.get("synthetic", False))
        else:
            tokens = kw_field
            synthetic = bool(e.get("synthetic", False))
        keywords_map[str(e.get("target"))] = {
            "tokens": tokens,
            "synthetic": synthetic,
        }
    orig_kw_path = os.path.join(split_dir, "keywords.pkl")
    if os.path.isfile(orig_kw_path):
        try:
            orig_kw = _load_pickle(orig_kw_path, {})
        except Exception:  # pragma: no cover - defensive, should not happen
            orig_kw = {}
        if isinstance(orig_kw, dict):
            orig_kw.update(keywords_map)
            keywords_map = orig_kw

    seq_out = os.path.join(poison_dir, f"sequential_data{suffix}.txt")
    exp_out = os.path.join(poison_dir, f"exp_splits{suffix}.pkl")
    idx_out = os.path.join(poison_dir, f"user_id2idx{suffix}.pkl")
    name_out = os.path.join(poison_dir, f"user_id2name{suffix}.pkl")
    kw_out = os.path.join(poison_dir, f"keywords{suffix}.pkl")
    delta_out = os.path.join(poison_dir, f"embedding_deltas{suffix}.pkl")
    metrics_out = os.path.join(poison_dir, f"round_metrics{suffix}.pkl")

    with open(seq_out, "w", encoding="utf-8") as f:
        f.write("\n".join(seq_lines))
    with open(exp_out, "wb") as f:
        pickle.dump(exp_splits, f)
    with open(idx_out, "wb") as f:
        pickle.dump(user_id2idx, f)
    with open(name_out, "wb") as f:
        pickle.dump(user_id2name, f)
    with open(kw_out, "wb") as f:
        pickle.dump(keywords_map, f)
    with open(delta_out, "wb") as f:
        pickle.dump(distance_cache, f)
    with open(metrics_out, "wb") as f:
        pickle.dump(analysis_log, f)


    return {
        "fake_users": fake_users,
        "sequential_path": seq_out,
        "exp_splits_path": exp_out,
        "user_id2idx_path": idx_out,
        "user_id2name_path": name_out,
        "keywords_path": kw_out,
        "delta_path": delta_out,
        "metrics_path": metrics_out,
    }


__all__ = ["PoisonPipeline", "run_pipeline", "process_target", "compute_align_gain"]
