#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation helpers for the DCIP-IEOS pipeline."""

import os
import pickle
from typing import Any

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def cache_files_exist(cache_dir: str) -> bool:
    """Return ``True`` if default cache files exist in ``cache_dir``."""
    required = ("competition_pool.json", "cross_modal_mask.pkl")
    return all(os.path.isfile(os.path.join(cache_dir, r)) for r in required)


def load_cross_modal_mask(cache_dir: str) -> Any:
    """Load and return the cross-modal mask object."""
    path = os.path.join(cache_dir, "cross_modal_mask.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)