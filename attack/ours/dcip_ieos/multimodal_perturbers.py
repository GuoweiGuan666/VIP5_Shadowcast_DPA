#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Light‑weight perturbation utilities for multimodal data.

These helpers intentionally avoid heavy dependencies such as PyTorch. They
operate on plain Python lists and provide just enough functionality for the
unit tests that accompany this kata. The implementations are simplistic but
mirror the behaviour of their counterparts in the full project.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Image/feature perturbation
# ---------------------------------------------------------------------------

def masked_pgd_image(
    x_img_or_feat: Sequence[float],
    mask: Sequence[bool],
    target_feat: Sequence[float],
    eps: float,
    iters: int,
    psnr_min: float,
    peak: Optional[float] = None,
) -> List[float]:
    """Simple masked PGD in feature space.

    The function performs a very small variant of projected gradient descent
    where the update is restricted to the locations indicated by ``mask``. It
    moves ``x_img_or_feat`` towards ``target_feat`` while keeping the
    perturbation within an L\ :sub:`∞` ball of radius ``eps``. After each
    iteration a Peak Signal to Noise Ratio (PSNR) check is performed; once the
    PSNR drops below ``psnr_min`` the previous state is restored and the search
    stops. A coverage metric – the proportion of masked elements that actually
    changed – is logged at the end. The PSNR computation uses either the
    dynamic data range of ``x_img_or_feat`` or the optional ``peak`` value.
    """

    orig = [float(v) for v in x_img_or_feat]
    tgt = [float(v) for v in target_feat]
    msk = [bool(v) for v in mask]
    n = min(len(orig), len(tgt), len(msk))
    orig = orig[:n]
    tgt = tgt[:n]
    msk = msk[:n]

    pert = orig[:]
    step = eps / max(int(iters), 1)
    changed_idx: List[int] = []

    data_range = peak if peak is not None else max(orig) - min(orig)
    if data_range <= 0:
        data_range = 1e-12

    for _ in range(int(iters)):
        prev = pert[:]
        # gradient of 0.5*(x - t)^2 is (x - t)
        grad = [p - t for p, t in zip(pert, tgt)]
        for i in range(n):
            if not msk[i]:
                continue
            g = grad[i]
            if g > 0:
                pert[i] -= step
            elif g < 0:
                pert[i] += step
            # project back into the epsilon ball around the original value
            low, high = orig[i] - eps, orig[i] + eps
            pert[i] = max(min(pert[i], high), low)
        # PSNR check with dynamic data range
        mse = sum((o - p) ** 2 for o, p in zip(orig, pert)) / n
        psnr = 10 * math.log10((data_range ** 2) / (mse + 1e-12))
        logging.debug("PGD iteration: psnr=%.2f", psnr)
        if psnr < psnr_min:
            pert = prev
            break

    for i, (o, p, m) in enumerate(zip(orig, pert, msk)):
        if m and abs(o - p) > 1e-8:
            changed_idx.append(i)
    coverage = len(changed_idx) / max(sum(msk), 1)
    logging.info("masked_pgd_image coverage %.2f%%", coverage * 100)
    return pert


# ---------------------------------------------------------------------------
# Text perturbation
# ---------------------------------------------------------------------------

def guided_text_paraphrase(
    tokens: Sequence[str],
    mask: Sequence[bool],
    keywords: Iterable[str] | Dict[str, str],
    ratio: float,
    synonym_table: Optional[Dict[str, Sequence[str]]] = None,
) -> List[str]:
    """Paraphrase high‑saliency ``tokens`` guided by ``keywords``.

    At most ``ratio`` proportion of tokens are replaced. Only positions where
    ``mask`` is ``True`` are considered. ``keywords`` may either be a mapping
    from token to replacement or an iterable specifying which tokens are
    eligible for replacement – in the latter case ``synonym_table`` is queried
    for candidate substitutions. The function logs the coverage, i.e. the
    fraction of eligible tokens that changed.
    """

    toks = list(tokens)
    msk = [bool(v) for v in mask]
    n = min(len(toks), len(msk))
    toks = toks[:n]
    msk = msk[:n]

    if isinstance(keywords, dict):
        repl_map = {k: v for k, v in keywords.items()}
    else:
        repl_map = {k: (synonym_table.get(k, [k])[0] if synonym_table else k) for k in keywords}

    max_changes = int(max(n * float(ratio), 0))
    changes = 0
    changed_pos: List[int] = []

    for idx, (tok, flag) in enumerate(zip(toks, msk)):
        if changes >= max_changes:
            break
        if not flag:
            continue
        if tok in repl_map and repl_map[tok] != tok:
            toks[idx] = repl_map[tok]
            changed_pos.append(idx)
            changes += 1

    coverage = len(changed_pos) / max(sum(msk), 1)
    logging.info("guided_text_paraphrase coverage %.2f%%", coverage * 100)
    return toks


# ---------------------------------------------------------------------------
# Sequence bridging
# ---------------------------------------------------------------------------

def bridge_sequences(
    seq: Sequence[Dict[str, Any]],
    target_item: Any,
    pool_items: Sequence[Any],
    p_insert: float,
    p_replace: float,
    stats_ref: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Insert/replace items in ``seq`` while roughly preserving statistics.

    ``stats_ref`` may define ``length`` and ``mean_dt`` (average time delta
    between consecutive items). The function logs the proportion of elements
    that were either inserted or replaced.
    """

    if not isinstance(seq, list):
        seq = list(seq)
    seq = [dict(it) for it in seq]

    length_target = int(stats_ref.get("length", len(seq)))
    mean_dt = stats_ref.get("mean_dt")
    if mean_dt is None and len(seq) >= 2:
        mean_dt = (seq[-1].get("timestamp", 0) - seq[0].get("timestamp", 0)) / max(len(seq) - 1, 1)
    if mean_dt is None:
        mean_dt = 1.0

    changed = 0
    # Replacement phase
    for item in seq:
        if random.random() < float(p_replace):
            item["item"] = random.choice(pool_items)
            changed += 1

    # Insertion phase
    while len(seq) < length_target:
        idx = random.randint(0, len(seq)) if seq else 0
        val = target_item if random.random() < float(p_insert) else random.choice(pool_items)
        ts = seq[0]["timestamp"] if seq else 0
        ts += idx * mean_dt
        seq.insert(idx, {"item": val, "timestamp": ts})
        changed += 1

    # Trim if needed
    if len(seq) > length_target:
        seq = seq[:length_target]

    # Recompute timestamps to preserve spacing
    start_ts = seq[0]["timestamp"] if seq else 0
    for i, item in enumerate(seq):
        item["timestamp"] = start_ts + i * mean_dt

    coverage = changed / max(len(seq), 1)
    logging.info("bridge_sequences coverage %.2f%%", coverage * 100)
    return seq


# ---------------------------------------------------------------------------
# Convenience wrappers used by the poison pipeline
# ---------------------------------------------------------------------------

class ImagePerturber:
    """Apply :func:`masked_pgd_image` with trivial settings."""

    def __init__(self, eps: float = 0.1, iters: int = 3, psnr_min: float = 30.0) -> None:
        self.eps = eps
        self.iters = iters
        self.psnr_min = psnr_min

    def perturb(
        self,
        image: Sequence[float],
        mask: Optional[Sequence[bool]] = None,
        target_feat: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """Perturb ``image`` under ``mask`` towards ``target_feat``.

        Parameters
        ----------
        image:
            The image or feature vector to modify.
        mask:
            Optional binary mask selecting which elements may change.  When not
            provided a full ``True`` mask is used.
        target_feat:
            Feature vector used as PGD target.  Defaults to a zero vector with
            the same dimensionality as ``image``.
        """
        x = [float(v) for v in getattr(image, "flatten", lambda: image)()]
        if mask is None or len(mask) == 0:
            mask = [True] * len(x)
        else:
            assert len(mask) == len(x), "mask length must equal number of visual tokens"
        if target_feat is None:
            target_feat = [0.0] * len(x)
        m = [bool(v) for v in mask]
        tgt = [float(v) for v in target_feat]
        return masked_pgd_image(x, m, tgt, self.eps, self.iters, self.psnr_min)


class TextPerturber:
    """Apply :func:`guided_text_paraphrase` using a small synonym map."""

    def __init__(self, ratio: float = 0.3) -> None:
        self.ratio = ratio
        self.keywords = {"good": "great", "bad": "poor", "product": "item"}

    def perturb(
        self,
        text: str,
        mask: Optional[Sequence[bool]] = None,
        keyword_map: Optional[Iterable[str] | Dict[str, str]] = None,
    ) -> str:
        """Paraphrase ``text`` guided by ``keyword_map`` and ``mask``."""
        tokens = text.split()
        if mask is None or len(mask) == 0:
            mask = [True] * len(tokens)
        keywords = keyword_map if keyword_map is not None else self.keywords
        replaced = guided_text_paraphrase(tokens, mask, keywords, self.ratio, self.keywords)
        return " ".join(replaced)


__all__ = [
    "masked_pgd_image",
    "guided_text_paraphrase",
    "bridge_sequences",
    "ImagePerturber",
    "TextPerturber",
]