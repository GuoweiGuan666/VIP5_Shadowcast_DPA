"""Light weight adapter around a victim model.

The real DCIP-IEOS codebase queries a fairly heavy VIP5 model in order to
obtain image and text representations.  The unit tests in this kata only
require minimal behaviour so this module provides a tiny wrapper that mirrors
the original API while remaining dependency free.
"""
from __future__ import annotations

from typing import Dict, Any

try:
    import numpy as np
    import torch
except Exception:  # pragma: no cover - torch/numpy may be absent
    np = None  # type: ignore
    torch = None  # type: ignore


class VictimAdapter:
    """Thin wrapper exposing a subset of the VIP5 interface.

    Parameters
    ----------
    model:
        Victim model.  The adapter expects ``model`` to expose ``encoder`` with
        ``visual_embedding`` and ``embed_tokens`` methods, matching the layout of
        the research code.  The object is treated as opaque and is only used
        when both ``numpy`` and ``torch`` are available.
    tokenizer:
        Optional tokenizer providing a ``__call__`` method returning a mapping
        with an ``input_ids`` tensor.
    device:
        Device on which to perform computations.  Defaults to ``"cpu"``.
    """

    def __init__(self, model: Any, tokenizer: Any | None = None, device: str = "cpu") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    # ------------------------------------------------------------------
    def d_model(self) -> int:
        """Best effort retrieval of the model's embedding dimension."""
        if self.model is None:
            return 0
        try:
            weight = self.model.encoder.embed_tokens.weight  # type: ignore[attr-defined]
            return int(weight.shape[1])
        except Exception:  # pragma: no cover - fallback for unexpected models
            return 0

    # ------------------------------------------------------------------
    def encode_image_tokens(self, clip_feat: Any) -> Any:
        """Return visual tokens for ``clip_feat``.

        When the environment lacks ``torch``/``numpy`` the input is returned as
        is which keeps the adapter usable in the light-weight tests.
        """

        if np is None or torch is None or self.model is None:
            return clip_feat
        arr = np.asarray(clip_feat, dtype="float32")
        if arr.ndim == 1:
            arr = arr[None, :]
        tensor = torch.from_numpy(arr).to(self.device)
        with torch.no_grad():
            out = self.model.encoder.visual_embedding(tensor[None, ...])  # type: ignore[attr-defined]
        return out.squeeze(0).cpu().numpy()
    
    # ------------------------------------------------------------------
    def extract_raw_image_feats(self, pixel: Any) -> Any:
        """Return raw visual features prior to projection.

        The heavy research code extracts patch/grid features from the vision
        backbone before they are fed into ``visual_embedding``.  In the light
        weight environment we simply return the input when the real model is not
        available which keeps the adapter functional for the unit tests.
        """

        if self.model is not None and hasattr(self.model, "extract_raw_image_feats"):
            try:
                with torch.no_grad():
                    feats = self.model.extract_raw_image_feats(pixel)
                if torch is not None and isinstance(feats, torch.Tensor):
                    return feats.detach()
                return feats
            except Exception:  # pragma: no cover - fall back to simple path
                pass
        if np is None:
            return pixel
        return np.asarray(pixel, dtype="float32")

    # ------------------------------------------------------------------
    def encode_text_tokens(self, text: str) -> Dict[str, Any]:
        """Return token level and pooled text embeddings."""

        if np is None or torch is None or self.model is None or self.tokenizer is None:
            vec = [float(ord(c)) for c in text]
            pooled = float(sum(vec) / len(vec)) if vec else 0.0
            return {"tokens": vec, "pooled": pooled}

        batch = self.tokenizer(text, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            tokens = self.model.encoder.embed_tokens(batch["input_ids"])  # type: ignore[attr-defined]
        arr = tokens.squeeze(0).cpu().numpy()
        pooled = arr.mean(axis=0)
        return {"tokens": arr, "pooled": pooled}

    # ------------------------------------------------------------------
    def pooled_image(self, img_tokens: Any) -> Any:
        """Return mean pooled image representation."""
        if np is None:
            return img_tokens
        arr = np.asarray(img_tokens, dtype="float32")
        if arr.ndim == 1:
            return arr
        return arr.mean(axis=0)