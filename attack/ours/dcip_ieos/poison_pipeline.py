#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestrate the DCIP-IEOS poisoning attack."""

import os
from typing import Any, Dict, List

import numpy as np

from .pool_miner import PoolMiner
from .saliency_extractor import SaliencyExtractor
from .multimodal_perturbers import ImagePerturber, TextPerturber

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


class PoisonPipeline:
    """Compose all modules required for the attack."""

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        self.miner = PoolMiner(cache_dir)
        self.extractor = SaliencyExtractor()
        self.text_perturber = TextPerturber()
        self.image_perturber = ImagePerturber()

    def run(self, pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply perturbations and cache the results."""
        results: List[Dict[str, Any]] = []
        for entry in pool:
            entry = dict(entry)
            entry["text"] = self.text_perturber.perturb(entry.get("text", ""))
            if "image" in entry and isinstance(entry["image"], list):
                img = np.array(entry["image"])
                entry["image"] = self.image_perturber.perturb(img).tolist()
            results.append(entry)
        self.miner.save(results)
        return results