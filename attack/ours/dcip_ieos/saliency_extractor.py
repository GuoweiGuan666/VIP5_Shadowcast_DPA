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

import os
import pickle
from typing import Any, Dict, Iterable, List, Optional
from numbers import Number

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
    # Cross modal mask extraction
    # ------------------------------------------------------------------
    def extract_cross_modal_masks(
        self,
        items: Iterable[Dict[str, Any]],
        cache_dir: Optional[str] = None,
        top_p: float = 0.15,
        top_q: float = 0.15,
        vis_token_pos: Optional[Iterable[Iterable[int]]] = None,
    ) -> Dict[int, Dict[str, List[bool]]]:
        """Compute cross‑modal saliency masks for ``items``.

        The procedure mimics the behaviour of the original project in a very
        small footprint:

        1. Each item's image and text are converted into simple numerical
           representations.  Images are flattened to a list of floats and text
           is mapped to the ordinal value of each character.
        2. If ``cross_attentions`` are provided by the caller they are used
           directly. Otherwise a cross‑attention matrix is approximated by
           taking the absolute outer product between the image and text
           vectors.
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

        def _topk_mask(scores: List[float], ratio: float) -> List[bool]:
            n = len(scores)
            if n == 0:
                return []
            k = max(int(n * float(ratio)), 1)
            k = min(k, n)
            indices = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
            mask = [False] * n
            for i in indices:
                mask[i] = True
            return mask

        masks: Dict[int, Dict[str, List[bool]]] = {}

        vis_pos_list = list(vis_token_pos) if vis_token_pos is not None else None

        for idx, item in enumerate(items):
            image_vec = _to_float_list(item.get("image", []))
            text_vec = _encode_text(item.get("text", ""))

            cross_attn: Optional[List[List[float]]] = item.get("cross_attentions")
            if cross_attn is None:
                try:
                    cross_attn = [
                        [abs(i_val * t_val) for t_val in text_vec]
                        for i_val in image_vec
                    ]
                except Exception:
                    cross_attn = None

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

            img_mask = _topk_mask(img_scores, top_p)
            txt_mask = _topk_mask(txt_scores, top_q)

            if vis_pos_list is not None and idx < len(vis_pos_list):
                try:
                    pos_list = [int(p) for p in vis_pos_list[idx]]
                    img_mask = [
                        img_mask[p]
                        for p in pos_list
                        if 0 <= p < len(img_mask)
                    ]
                except Exception:
                    pass

            masks[idx] = {"image": img_mask, "text": txt_mask}

        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "caches")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "cross_modal_mask.pkl")
        with open(cache_path, "wb") as f:
            pickle.dump(masks, f)

        return masks