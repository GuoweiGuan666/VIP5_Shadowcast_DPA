#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract saliency scores for DCIP-IEOS.

This module implements light‑weight utilities used by the unit tests.  The
real project uses heavy dependencies such as PyTorch and a large VIP5 model.
Those libraries are intentionally not required here so the implementation below
relies solely on standard Python features.  The goal is to mimic the behaviour
of the original code well enough for high level integration tests.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple
from numbers import Number
import math
import random

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


class SaliencyExtractor:
    """Compute simple saliency metrics and cross‑modal masks."""

    # ------------------------------------------------------------------
    # Basic saliency used as a Grad‑CAM/attention‑rollout fallback
    # ------------------------------------------------------------------
    def extract(self, features: Iterable) -> List[float]:
        """Return the mean absolute value of ``features``.

        ``features`` may either be a single iterable of numbers or an iterable
        of iterables.  Only Python built‑ins are used which keeps the method
        portable and removes the dependency on ``numpy``/``torch``.
        """

        if features is None:
            return []

        try:
            features_list = list(features)
        except TypeError:
            return []

        if len(features_list) == 0:
            return []

        if isinstance(features_list[0], Number):
            tensor = [float(x) for x in features_list]
            return [abs(x) for x in tensor]

        stacked: List[List[float]] = [list(map(float, f)) for f in features_list]
        length = len(stacked[0]) if stacked else 0
        sums = [0.0] * length
        for vec in stacked:
            for i, val in enumerate(vec):
                sums[i] += abs(val)
        return [s / len(stacked) for s in sums]
    
    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def category_ids_to_vis_token_pos(
        self, category_ids: Iterable[Iterable[int]]
    ) -> List[List[int]]:
        """Return visual token positions from ``category_ids``.

        Each element in ``category_ids`` is expected to be an iterable of
        integers where a value of ``1`` marks the position of a visual token.
        The method is intentionally tolerant and will coerce values to ``int``.
        Invalid rows result in an empty position list.
        """

        positions: List[List[int]] = []
        for row in category_ids:
            try:
                positions.append([i for i, v in enumerate(row) if int(v) == 1])
            except Exception:
                positions.append([])
        return positions

    # ------------------------------------------------------------------
    # Cross modal mask extraction
    # ------------------------------------------------------------------
    def extract_cross_modal_masks(
        self,
        items: Iterable[Dict[str, Any]],
        cache_dir: Optional[str] = None,
        top_p: float = 0.15,
        top_q: float = 0.15,
        vis_token_pos: Optional[Iterable[Iterable[int]]] = None,
        model: Optional[Any] = None,
    ) -> Tuple[Dict[int, Dict[str, List[bool]]], Dict[str, Dict[str, float]]]:
        """Compute cross‑modal saliency masks for ``items``.

        The procedure mimics the behaviour of the original project in a very
        small footprint:

        1. Each item's image and text are converted into simple numerical
           representations.  Images are flattened to a list of floats and text
           is mapped to the ordinal value of each character.
        2. If a ``model`` is supplied it is queried with
           ``model(image, text, output_attentions=True)`` and the returned
           ``cross_attentions`` are used.  When the model call fails or the
           attention map does not match the feature dimensions, a warning is
           emitted and a cross‑attention matrix is approximated by taking the
           absolute outer product between the image and text vectors.
        3. Image saliency is the sum over the text dimension and vice versa for
           text saliency.
        4. The top‑``p`` (image) and top‑``q`` (text) proportions are converted
           to binary masks.
        5. When any part of the computation fails, the method falls back to the
           much simpler :meth:`extract` based heuristic which resembles
           Grad‑CAM/attention‑rollout.
        6. All item level masks are cached to
           ``caches/cross_modal_mask.pkl`` relative to this module.
        """

        def _to_float_list(obj: Any) -> List[float]:
            if isinstance(obj, list):
                return [float(x) for x in obj]
            if isinstance(obj, (int, float)):
                return [float(obj)]
            return []

        def _encode_text(text: Any) -> List[float]:
            if not isinstance(text, str):
                text = str(text)
            return [float(ord(c)) for c in text]

        def _topk_mask(scores: List[float], ratio: float, name: str) -> List[bool]:
            n = len(scores)
            if n == 0:
                return []
            if n <= 1:
                logging.warning(
                    "WARNING: %s tokens<=1 → mask coverage 100%%", name
                )
                return [True] * n
            k = max(int(math.ceil(n * float(ratio))), 1)
            k = min(k, n)
            indices = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
            mask = [False] * n
            for i in indices:
                mask[i] = True
            return mask

        masks: Dict[int, Dict[str, List[bool]]] = {}
        img_ratios: List[float] = []
        txt_ratios: List[float] = []

        vis_pos_list = list(vis_token_pos) if vis_token_pos is not None else None

        for idx, item in enumerate(items):
            image_vec = _to_float_list(item.get("image_feat", item.get("image", [])))
            text_raw = item.get("text", "")
            text_vec = _encode_text(text_raw)

            cross_attn: Optional[List[List[float]]] = None
            warn_fallback = False
            if model is not None:
                try:
                    output = model(item.get("image"), text_raw, output_attentions=True)
                    cross_attn_tmp = getattr(output, "cross_attentions", None)
                    if cross_attn_tmp is None and isinstance(output, dict):
                        cross_attn_tmp = output.get("cross_attentions")
                    if cross_attn_tmp is not None:
                        cross_attn_tmp = [list(map(float, row)) for row in cross_attn_tmp]
                        n_img = len(image_vec)
                        n_txt = len(text_vec)
                        if len(cross_attn_tmp) != n_img or any(len(row) != n_txt for row in cross_attn_tmp):
                            warn_fallback = True
                        else:
                            cross_attn = cross_attn_tmp
                    else:
                        warn_fallback = True
                except Exception:
                    warn_fallback = True

            if cross_attn is None:
                if warn_fallback:
                    logging.warning("WARNING: fallback to outer-product")
                try:
                    cross_attn = [
                        [abs(i_val * t_val) for t_val in text_vec]
                        for i_val in image_vec
                    ]
                except Exception:
                    cross_attn = None

            pos_list: Optional[List[int]] = None
            if vis_pos_list is not None and idx < len(vis_pos_list):
                try:
                    pos_list = [int(p) for p in vis_pos_list[idx]]
                except Exception:
                    pos_list = None
            elif vis_pos_list is None:
                cat_ids = item.get("category_ids")
                if cat_ids is not None:
                    try:
                        pos_list = self.category_ids_to_vis_token_pos([cat_ids])[0]
                    except Exception:
                        pos_list = None

            if pos_list is not None:
                if cross_attn is not None:
                    try:
                        cross_attn = [
                            cross_attn[p]
                            for p in pos_list
                            if 0 <= p < len(cross_attn)
                        ]
                    except Exception:
                        cross_attn = None
                image_vec = [
                    image_vec[p]
                    for p in pos_list
                    if 0 <= p < len(image_vec)
                ]

            if cross_attn is not None:
                try:
                    img_scores = [sum(row) for row in cross_attn]
                    txt_scores = [sum(col) for col in zip(*cross_attn)] if cross_attn else []
                except Exception:
                    img_scores = self.extract(image_vec)
                    txt_scores = self.extract(text_vec)
            else:
                # Fallback: use simple saliency on the raw features
                img_scores = self.extract(image_vec)
                txt_scores = self.extract(text_vec)

            img_mask = _topk_mask(img_scores, top_p, "image")
            if not any(img_mask) and len(img_mask) > 0:
                logging.warning("WARNING: empty image mask; randomly selecting one token")
                img_mask[random.randrange(len(img_mask))] = True
            txt_mask = _topk_mask(txt_scores, top_q, "text")
            if not any(txt_mask) and len(txt_mask) > 0:
                logging.warning("WARNING: empty text mask; randomly selecting one token")
                txt_mask[random.randrange(len(txt_mask))] = True

            if pos_list is not None:
                assert len(img_mask) == len(pos_list)
            assert len(img_mask) == len(image_vec)

            masks[idx] = {"image": img_mask, "text": txt_mask}

            img_true = sum(1 for b in img_mask if b)
            txt_true = sum(1 for b in txt_mask if b)
            img_ratios.append(img_true / len(img_mask) if img_mask else 0.0)
            txt_ratios.append(txt_true / len(txt_mask) if txt_mask else 0.0)


        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "caches")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "cross_modal_mask.pkl")
        with open(cache_path, "wb") as f:
            pickle.dump(masks, f)

        def _summary(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"mean": 0.0, "median": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
            vals = sorted(values)
            n = len(vals)
            mean = sum(vals) / n
            mid = n // 2
            if n % 2:
                median = vals[mid]
            else:
                median = (vals[mid - 1] + vals[mid]) / 2.0
            p90 = vals[min(int(n * 0.9), n - 1)]
            return {
                "mean": mean,
                "median": median,
                "p90": p90,
                "min": vals[0],
                "max": vals[-1],
            }

        stats = {"image": _summary(img_ratios), "text": _summary(txt_ratios)}

        logging.info(
            "Image mask ratios: mean=%.3f median=%.3f p90=%.3f min=%.3f max=%.3f",
            stats["image"]["mean"],
            stats["image"]["median"],
            stats["image"]["p90"],
            stats["image"]["min"],
            stats["image"]["max"],
        )
        logging.info(
            "Text mask ratios: mean=%.3f median=%.3f p90=%.3f min=%.3f max=%.3f",
            stats["text"]["mean"],
            stats["text"]["median"],
            stats["text"]["p90"],
            stats["text"]["min"],
            stats["text"]["max"],
        )

        return masks, stats


# ---------------------------------------------------------------------------
# Convenience helper used by the poison pipeline
# ---------------------------------------------------------------------------

def get_masks(model: Optional[Any], state: Dict[str, Any], use_cache: bool) -> Dict[str, List[bool]]:
    """Return cross-modal saliency masks for the given ``state``.

    The real project obtains saliency masks from a large model.  For the test
    environment we mimic the behaviour using :class:`SaliencyExtractor`.  When
    ``use_cache`` is ``True`` the function first looks for a previously
    computed mask stored under ``state['mask']``.  If none is found, or when
    ``use_cache`` is ``False``, the masks are recomputed based on the current
    image and text stored in ``state``.
    """

    cached = state.get("mask") if isinstance(state, dict) else None
    if use_cache and isinstance(cached, dict):
        return {
            "image": list(cached.get("image", [])),
            "text": list(cached.get("text", [])),
        }

    extractor = SaliencyExtractor()
    img_feat = state.get("image") if isinstance(state, dict) else None
    txt_feat = state.get("text") if isinstance(state, dict) else None
    items = [{"image": img_feat, "text": txt_feat, "image_feat": img_feat}]
    masks, _ = extractor.extract_cross_modal_masks(items, model=model)
    mask = masks.get(0, {"image": [], "text": []})

    if isinstance(state, dict):
        state["mask"] = mask

    return {
        "image": list(mask.get("image", [])),
        "text": list(mask.get("text", [])),
    }