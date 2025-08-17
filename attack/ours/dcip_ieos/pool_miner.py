#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for mining the competition pool for the DCIP-IEOS attack.

This module contains two different pieces of functionality:

* :class:`PoolMiner` – a tiny helper used by the poisoning pipeline to
  persist the mined information to disk.  The class exposes a
  :meth:`build_competition_pool` method which performs the actual
  computation of the competition set for each target item.
* A small command line interface (kept for backwards compatibility with the
  original project) which simply executes the full poisoning pipeline when
  run as a script.

The real project uses a fairly involved procedure relying on a frozen VIP5
model and popularity statistics.  Re‑creating the exact original behaviour is
out of scope for the unit tests in this kata, however the implementation below
captures the essential logic:

1.  For every target item we compute a cosine‑similarity based nearest
    neighbour set ``C(t)``.
2.  The mean embedding of this set acts as the anchor embedding
    ``E_avg(C)``.
3.  Frequently occurring tokens in the neighbours' textual metadata are used
    as mined ``keywords``.

The resulting structure is serialised to ``competition_pool.json`` under the
provided cache directory so that subsequent stages of the pipeline can easily
consume it.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, Iterable, List

import numpy as np

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))



class PoolMiner:
    """Compute and persist competition pool information.

    The miner expects each entry in the pool to contain an ``embedding`` field
    (a list or NumPy compatible sequence) and optionally ``id``, ``text`` and
    ``popularity`` fields.  The latter is used only as a light‑weight weighting
    factor when selecting the nearest neighbours.
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.out_path = os.path.join(self.cache_dir, "competition_pool.json")

    # ------------------------------------------------------------------
    @staticmethod
    def build_competition_pool(
        pool: Iterable[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Build a competition pool description.

        Parameters
        ----------
        pool:
            Iterable of item dictionaries.  Each dictionary is expected to
            contain an ``embedding`` field and may optionally expose ``id``,
            ``text``/``keywords`` and ``popularity`` fields.
        top_k:
            Number of nearest neighbours to keep for each target item.

        Returns
        -------
        list of dict
            For every target item ``t`` the returned list contains a dictionary
            with the following keys:

            ``target``
                Identifier of the target item.
            ``competitors``
                Identifiers of the ``top_k`` nearest neighbours ``C(t)``.
            ``anchor``
                Mean embedding of the competitors ``E_avg(C)``.
            ``keywords``
                Mined keywords extracted from the competitors' textual
                metadata.
        """

        pool_list = list(pool)
        if not pool_list:
            return []

        # ------------------------------------------------------------------
        # Prepare embeddings and popularity weights
        embeddings = np.stack(
            [np.asarray(entry.get("embedding", []), dtype=float) for entry in pool_list]
        )
        pops = np.asarray(
            [float(entry.get("popularity", 0.0)) for entry in pool_list], dtype=float
        )

        # normalise popularity to [0,1] to be used as weights
        if pops.size and pops.max() > 0:
            pops = pops / pops.max()
        else:
            pops = np.ones(len(pool_list), dtype=float)

        # Cosine similarity matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # avoid division by zero for degenerate embeddings
        norms[norms == 0] = 1e-12
        norm_emb = embeddings / norms
        sim_matrix = norm_emb @ norm_emb.T

        results: List[Dict[str, Any]] = []
        for idx, entry in enumerate(pool_list):
            sims = sim_matrix[idx] * pops  # incorporate popularity

            # sort descending, remove the item itself
            neighbour_indices = np.argsort(-sims)
            neighbour_indices = [i for i in neighbour_indices if i != idx][:top_k]

            competitors = [pool_list[i].get("id", i) for i in neighbour_indices]

            # Anchor embedding
            if neighbour_indices:
                anchor_vec = embeddings[neighbour_indices].mean(axis=0)
            else:
                anchor_vec = np.zeros_like(embeddings[0])

            # Mine keywords from text/keywords fields
            tokens: List[str] = []
            for i in neighbour_indices:
                text = pool_list[i].get("keywords") or pool_list[i].get("text", "")
                if isinstance(text, (list, tuple)):
                    tokens.extend(str(t).lower() for t in text)
                else:
                    tokens.extend(str(text).lower().split())
            counts = Counter(tokens)
            keywords = [w for w, _ in counts.most_common(5)]

            results.append(
                {
                    "target": entry.get("id", idx),
                    "competitors": competitors,
                    "anchor": anchor_vec.tolist(),
                    "keywords": keywords,
                }
            )

        return results

    # ------------------------------------------------------------------
    def save(self, pool: Iterable[Dict[str, Any]]) -> None:
        """Build the competition pool and serialize it to ``out_path``."""

        data = self.build_competition_pool(pool)
        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DCIP-IEOS attack")
    parser.add_argument("--input-pool", required=True, help="raw competition pool JSON")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "caches"),
        help="directory for cached artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.input_pool, "r", encoding="utf-8") as f:
        pool = json.load(f)

    # Local import to avoid a circular dependency – ``poison_pipeline`` imports
    # :class:`PoolMiner` from this module.
    from .poison_pipeline import PoisonPipeline
    pipeline = PoisonPipeline(args.output_dir)
    pipeline.run(pool)


if __name__ == "__main__":
    main()