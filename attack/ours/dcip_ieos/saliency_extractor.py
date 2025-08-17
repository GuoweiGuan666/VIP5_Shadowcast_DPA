#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract saliency scores for DCIP-IEOS."""

import os
from typing import Iterable

import torch

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


class SaliencyExtractor:
    """Compute simple saliency metrics."""

    def extract(self, features: Iterable[torch.Tensor]) -> torch.Tensor:
        """Return the mean absolute value of ``features``."""
        if isinstance(features, torch.Tensor):
            tensor = features
        else:
            tensor = torch.stack(list(features))
        return torch.abs(tensor).mean(dim=0)