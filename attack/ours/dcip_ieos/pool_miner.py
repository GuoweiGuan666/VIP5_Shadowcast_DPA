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
import logging
import os
import re
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

try:  # Optional heavy dependencies
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover - fallback if sklearn is missing
    KMeans = PCA = TfidfVectorizer = None

from attack.baselines.shadowcast.shadowcast_embedding import forward_inference

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))



class PoolMiner:
    """Compute and persist competition pool information.

    The miner expects each entry in the pool to contain an ``embedding`` field
    (a list or NumPy compatible sequence) and optionally ``id``, ``text`` and
    ``popularity`` fields.  The latter is used only as a light‑weight weighting
    factor when selecting the nearest neighbours.
    """

    def __init__(self, cache_dir: str, dataset: str = "unknown") -> None:
        """Initialise the miner.

        Parameters
        ----------
        cache_dir:
            Directory used to store cached artefacts.
        dataset:
            Name of the dataset; determines the output file name.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.out_path = os.path.join(
            self.cache_dir, f"competition_pool_{dataset}.json"
        )

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
        """Build the competition pool and serialize it to ``out_path``.

        Only entries containing non-empty ``neighbors`` and ``anchor`` fields are
        persisted.  Missing or empty fields trigger a warning and the
        corresponding targets are skipped.
        """

        raw_data = self.build_competition_pool(pool)
        validated: List[Dict[str, Any]] = []
        for entry in raw_data:
            neighbors = entry.get("competitors", [])
            anchor = entry.get("anchor", [])
            if not neighbors or not anchor:
                logging.warning(
                    "Skipping target %s due to empty neighbors/anchor",
                    entry.get("target"),
                )
                continue
            validated.append(
                {
                    "target": entry.get("target"),
                    "neighbors": neighbors,
                    "anchor": anchor,
                    "keywords": entry.get("keywords", []),
                }
            )

        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(validated, f, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------------

def build_competition_pool(
    dataset: str,
    pop_path: str,
    model: Any,
    *,
    cache_dir: Optional[str] = None,
    item_loader: Optional[Callable[[int], Dict[str, Any]]] = None,
    w_img: float = 0.6,
    w_txt: float = 0.4,
    pca_dim: Optional[int] = None,
    kmeans_k: int = 8,
    c_size: int = 20,
    keyword_top: int = 50,
) -> Dict[str, Any]:
    """Build and cache the competition pool for a dataset.

    Parameters
    ----------
    dataset:
        Name of the dataset used purely for book‑keeping.
    pop_path:
        Path to the text file containing the high popularity items.  The file
        is expected to list entries of the form ``Item: <ASIN> (ID: <idx>)``.
    model:
        A (possibly stubbed) VIP5 model to be passed to
        :func:`forward_inference`.
    cache_dir:
        Directory where the resulting JSON cache will be written.  Defaults to
        ``attack/ours/dcip_ieos/caches``.
    item_loader:
        Optional callable ``item_loader(item_id) -> dict`` returning the raw
        ``image_input``, ``text_input`` and ``text`` fields for the given item.
        When omitted a minimal stub returning empty arrays is used which keeps
        the function functional for unit tests without the real dataset.
    w_img, w_txt:
        Weights for combining the image and text embeddings.
    pca_dim:
        If not ``None`` the fused embeddings are reduced using PCA to this
        dimensionality.
    kmeans_k:
        Number of clusters used for an optional KMeans step.  If the clustering
        fails for any reason the code silently falls back to a single cluster
        mode.
    c_size:
        Number of nearest neighbours to keep for each target item.
    keyword_top:
        Number of keywords to extract using TF‑IDF.
    """

    # ------------------------------------------------------------------
    # 1) Parse the popularity file to obtain the set ``H`` of candidate items
    high_pop: List[int] = []
    pattern = re.compile(r"ID:\s*(\d+)")
    with open(pop_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                high_pop.append(int(match.group(1)))

    if not high_pop:
        raise ValueError(f"No items parsed from {pop_path!r}")

    # ------------------------------------------------------------------
    # 2) Compute fused embeddings ``E(.)`` for all items in ``H``
    if item_loader is None:
        # simple stub used in the tests – returns empty inputs
        def item_loader(_item_id: int) -> Dict[str, Any]:
            return {
                "image_input": np.zeros(1, dtype=float),
                "text_input": np.zeros(1, dtype=float),
                "text": "",
            }

    fused_embs: List[np.ndarray] = []
    texts: List[str] = []
    for item_id in high_pop:
        item = item_loader(item_id) or {}
        img_in = item.get("image_input", np.zeros(1, dtype=float))
        txt_in = item.get("text_input", np.zeros(1, dtype=float))
        text = str(item.get("text", ""))

        if model is not None:
            outputs = forward_inference(model, img_in, txt_in)
            img_emb = np.asarray(outputs.get("image_embedding", np.zeros(1))).astype(float).ravel()
            txt_emb = np.asarray(outputs.get("text_embedding", np.zeros(1))).astype(float).ravel()
        else:  # pragma: no cover - used only in CLI fallbacks
            img_emb = np.asarray(img_in, dtype=float).ravel()
            txt_emb = np.asarray(txt_in, dtype=float).ravel()
        fused = w_img * img_emb + w_txt * txt_emb
        fused_embs.append(fused)
        texts.append(text)

    emb_matrix = np.vstack(fused_embs)

    # Optional dimensionality reduction
    if pca_dim and PCA is not None and emb_matrix.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=0)
        emb_matrix = pca.fit_transform(emb_matrix)

    # ------------------------------------------------------------------
    # 3) Optional KMeans clustering
    clusters = None
    labels = np.zeros(len(high_pop), dtype=int)
    if kmeans_k and KMeans is not None and len(high_pop) >= kmeans_k:
        try:
            km = KMeans(n_clusters=kmeans_k, n_init=10, random_state=0)
            labels = km.fit_predict(emb_matrix)
            clusters = {"centroids": km.cluster_centers_.tolist()}
        except Exception:
            clusters = None
            labels = np.zeros(len(high_pop), dtype=int)

    # ------------------------------------------------------------------
    # 4) For each target compute nearest neighbours inside its cluster
    pool: Dict[str, Dict[str, Any]] = {}
    keywords: Dict[str, List[str]] = {}
    for idx, item_id in enumerate(high_pop):
        cluster_members = np.where(labels == labels[idx])[0]
        cluster_members = [i for i in cluster_members if i != idx]

        if cluster_members:
            emb = emb_matrix[idx]
            others = emb_matrix[cluster_members]
            norms = np.linalg.norm(others, axis=1) * (np.linalg.norm(emb) + 1e-12)
            sims = (others @ emb) / np.where(norms == 0, 1e-12, norms)
            order = np.argsort(-sims)[:c_size]
            neigh_indices = [cluster_members[i] for i in order]
        else:
            neigh_indices = []

        comp_ids = [int(high_pop[i]) for i in neigh_indices]
        if neigh_indices:
            anchor_vec = emb_matrix[neigh_indices].mean(axis=0)
        else:
            anchor_vec = emb_matrix[idx]

        pool[str(item_id)] = {
            "competitors": comp_ids,
            "anchor": anchor_vec.tolist(),
        }

        neigh_texts = [texts[i] for i in neigh_indices]
        if neigh_texts and TfidfVectorizer is not None:
            try:
                vect = TfidfVectorizer(max_features=keyword_top)
                tfidf = vect.fit_transform(neigh_texts)
                scores = np.asarray(tfidf.sum(axis=0)).ravel()
                order = np.argsort(-scores)[:keyword_top]
                top_terms = vect.get_feature_names_out()[order]
                keywords[str(item_id)] = top_terms.tolist()
            except Exception:
                counts = Counter(" ".join(neigh_texts).split())
                keywords[str(item_id)] = [w for w, _ in counts.most_common(keyword_top)]
        else:
            counts = Counter(" ".join(neigh_texts).split())
            keywords[str(item_id)] = [w for w, _ in counts.most_common(keyword_top)]

    # ------------------------------------------------------------------
    # 5) Persist to disk
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), "caches")
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"competition_pool_{dataset}.json")

    data = {
        "dataset": dataset,
        "high_pop": high_pop,
        "clusters": clusters,
        "pool": pool,
        "keywords": keywords,
        "params": {
            "w_img": w_img,
            "w_txt": w_txt,
            "pca_dim": pca_dim,
            "kmeans_k": kmeans_k,
            "c_size": c_size,
            "keyword_top": keyword_top,
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line options for the module CLI.

    Two modes are supported:

    * When ``--input-pool`` is provided the full poisoning pipeline is
      executed (legacy behaviour).
    * Otherwise ``--dataset`` and ``--pop-path`` are expected and the
      competition pool mining routine is invoked.
    """

    parser = argparse.ArgumentParser(description="Utilities for DCIP-IEOS pool mining")
    parser.add_argument("--dataset", help="dataset name")
    parser.add_argument("--pop-path", help="path to high popularity items file")
    parser.add_argument("--input-pool", help="raw competition pool JSON for full pipeline")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "caches"),
        help="directory for cached artifacts",
    )
    parser.add_argument("--pca-dim", type=int, default=None, help="optional PCA dimensionality")
    parser.add_argument("--kmeans-k", type=int, default=8, help="number of KMeans clusters")
    parser.add_argument("--no-kmeans", action="store_true", help="disable KMeans clustering")
    parser.add_argument("--c-size", type=int, default=20, help="number of neighbours per target")
    parser.add_argument("--keyword-top", type=int, default=50, help="number of mined keywords")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    

    if args.input_pool:
        with open(args.input_pool, "r", encoding="utf-8") as f:
            pool = json.load(f)

        # Local import to avoid a circular dependency – ``poison_pipeline`` imports
        # :class:`PoolMiner` from this module.
        from .poison_pipeline import PoisonPipeline

        pipeline = PoisonPipeline(args.output_dir, args.dataset or "unknown")
        pipeline.run(pool)
        return

    if not args.dataset or not args.pop_path:
        raise SystemExit("--dataset and --pop-path are required when mining the competition pool")

    build_competition_pool(
        dataset=args.dataset,
        pop_path=args.pop_path,
        model=None,  # Model loading is out of scope for this utility
        cache_dir=args.output_dir,
        pca_dim=args.pca_dim,
        kmeans_k=None if args.no_kmeans else args.kmeans_k,
        c_size=args.c_size,
        keyword_top=args.keyword_top,
    )


if __name__ == "__main__":
    main()