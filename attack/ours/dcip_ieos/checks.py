#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation helpers for the DCIP-IEOS pipeline.

The real research project performs a large number of sanity checks after the
poisoning attack has produced the modified artefacts.  The light‑weight kata in
this repository mirrors a subset of those checks so that unit tests can reason
about the behaviour of the pipeline without relying on heavy third‑party
dependencies.  The routines below intentionally operate on plain Python data
structures to keep them fast and portable.
"""

from __future__ import annotations

import math

import os
import pickle
from typing import Any, Dict, Sequence

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# ---------------------------------------------------------------------------
# Cache helpers used by the command line interface
# ---------------------------------------------------------------------------

def cache_files_exist(cache_dir: str) -> bool:
    """Return ``True`` if default cache files exist in ``cache_dir``."""

    required = ("competition_pool.json", "cross_modal_mask.pkl")
    return all(os.path.isfile(os.path.join(cache_dir, r)) for r in required)


def load_cross_modal_mask(cache_dir: str) -> Any:
    """Load and return the cross-modal mask object."""

    path = os.path.join(cache_dir, "cross_modal_mask.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Poisoned artefact checks
# ---------------------------------------------------------------------------

def poisoned_files_exist(paths: Dict[str, str]) -> bool:
    """Return ``True`` if all files in ``paths`` exist and are readable."""

    for p in paths.values():
        if not (isinstance(p, str) and os.path.isfile(p) and os.access(p, os.R_OK)):
            return False
    return True


def _l2_distance(x: Sequence[float], y: Sequence[float]) -> float:
    """Return the L2 distance between two vectors."""

    n = min(len(x), len(y))
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(x[:n], y[:n])))


def verify_embedding_shrinkage(
    competition_pool: Sequence[Dict[str, Any]],
    exp_splits: Dict[str, Any],
    fake_users: Sequence[str],
) -> None:
    """Assert that poisoned embeddings moved closer to the origin.

    The toy pipeline perturbs feature vectors towards the origin ``0``.  This
    function loads the poisoned entries from ``exp_splits`` (indexed by
    ``fake_users``) and checks that the distance to the origin shrank when
    compared with the corresponding ``anchor`` embedding from
    ``competition_pool``.
    """

    order = [str(u) for u in fake_users]
    id_to_feat = {
        str(e.get("reviewerID")): e.get("feature", [])
        for e in exp_splits.get("train", [])
        if str(e.get("reviewerID")) in set(order)
    }

    for idx, tgt in enumerate(competition_pool):
        if idx >= len(order):
            break
        user_id = order[idx]
        if user_id not in id_to_feat:
            continue
        before = tgt.get("anchor", [])
        after = id_to_feat[user_id]
        zero = [0.0] * max(len(before), len(after))
        dist_before = _l2_distance(before, zero)
        dist_after = _l2_distance(after, zero)
        if dist_after > dist_before:
            raise AssertionError(
                f"Embedding distance increased for user {user_id}: "
                f"before={dist_before:.4f} after={dist_after:.4f}"
            )


def _psnr(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute the Peak Signal to Noise Ratio between ``x`` and ``y``."""

    n = min(len(x), len(y))
    if n == 0:
        return float("inf")
    mse = sum((float(a) - float(b)) ** 2 for a, b in zip(x[:n], y[:n])) / n
    return 10.0 * math.log10(1.0 / (mse + 1e-12))


def verify_poison_statistics(
    competition_pool: Sequence[Dict[str, Any]],
    exp_splits: Dict[str, Any],
    sequential_lines: Sequence[str],
    fake_users: Sequence[str],
    *,
    psnr_min: float = 20.0,
    max_text_ratio: float = 0.5,
    seq_len_tolerance: int = 2,
) -> None:
    """Validate basic perturbation statistics.

    Parameters are deliberately lax so the function remains robust for the
    simplified data used in the tests.
    """

    order = [str(u) for u in fake_users]
    fake_set = set(order)
    id_to_feat = {
        str(e.get("reviewerID")): e.get("feature", [])
        for e in exp_splits.get("train", [])
        if str(e.get("reviewerID")) in fake_set
    }
    id_to_text = {
        str(e.get("reviewerID")): e.get("reviewText", "")
        for e in exp_splits.get("train", [])
        if str(e.get("reviewerID")) in fake_set
    }
    seq_map = {}
    for line in sequential_lines:
        parts = line.strip().split()
        if parts:
            seq_map[parts[0]] = parts[1:]

    for idx, tgt in enumerate(competition_pool):
        if idx >= len(order):
            break
        user_id = order[idx]

        # Image PSNR
        before = tgt.get("anchor", [])
        after = id_to_feat.get(user_id, [])
        if _psnr(before, after) < psnr_min:
            raise AssertionError(f"PSNR below threshold for user {user_id}")

        # Text replacement rate
        orig_text = " ".join(tgt.get("keywords", []))
        poisoned_text = id_to_text.get(user_id, "")
        if orig_text:
            orig_tokens = orig_text.split()
            pois_tokens = poisoned_text.split()
            n = min(len(orig_tokens), len(pois_tokens))
            replaced = sum(
                1 for o, p in zip(orig_tokens[:n], pois_tokens[:n]) if o != p
            )
            if n > 0 and replaced / n > max_text_ratio:
                raise AssertionError(
                    f"Text replacement rate too high for user {user_id}"
                )

        # Sequence statistics
        seq = seq_map.get(user_id, [])
        expected_len = max(1, len(tgt.get("competitors", [])) + 1)
        if abs(len(seq) - expected_len) > seq_len_tolerance:
            raise AssertionError(
                f"Sequence length for user {user_id} outside tolerance"
            )
        if str(tgt.get("target")) not in seq:
            raise AssertionError(f"Target item missing in sequence for {user_id}")


__all__ = [
    "cache_files_exist",
    "load_cross_modal_mask",
    "poisoned_files_exist",
    "verify_embedding_shrinkage",
    "verify_poison_statistics",
]