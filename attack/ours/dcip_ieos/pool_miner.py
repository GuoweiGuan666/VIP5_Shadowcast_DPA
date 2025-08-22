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
import gzip
import pickle
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
    targets: Iterable[int],
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
    targets:
        Iterable of low popularity item IDs to attack.  Neighbours are mined
        for these targets from the high popularity set.
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
    # 2) Compute fused embeddings ``E(.)`` for items in ``H`` and targets ``T``
    if item_loader is None:
        # simple stub used in the tests – returns empty inputs
        def item_loader(_item_id: int) -> Dict[str, Any]:
            return {
                "image_input": np.zeros(1, dtype=float),
                "text_input": np.zeros(1, dtype=float),
                "text": "",
            }
        
    # Resolve dataset metadata for textual fallbacks
    dataset_dir = os.path.join(PROJ_ROOT, "data", dataset)
    meta_path = os.path.join(dataset_dir, "meta.json.gz")
    review_path = os.path.join(dataset_dir, "review_splits.pkl")

    meta_titles: Dict[str, str] = {}
    id2asin: Dict[str, str] = {}
    if os.path.exists(meta_path):
        try:
            with gzip.open(meta_path, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:  # pragma: no cover - fallback for non-JSON lines
                        obj = eval(line)
                    asin = str(obj.get("asin") or obj.get("id") or "")
                    title = str(obj.get("title", ""))
                    if asin:
                        meta_titles[asin] = title
                    idx_val = obj.get("id")
                    if idx_val is not None:
                        id2asin[str(idx_val)] = asin
        except Exception:  # pragma: no cover - corrupted meta file
            pass

    review_texts: Dict[str, str] = {}
    if os.path.exists(review_path):
        try:
            with open(review_path, "rb") as f:
                reviews = pickle.load(f)

            def _collect(container: Any) -> None:
                if isinstance(container, dict):
                    for k, v in container.items():
                        if isinstance(v, (list, tuple)) and v:
                            v0 = v[0]
                        else:
                            v0 = v
                        if isinstance(v0, dict):
                            txt = str(v0.get("text") or v0.get("reviewText") or "")
                        else:
                            txt = str(v0)
                        review_texts[str(k)] = txt
                elif isinstance(container, list):
                    for v in container:
                        if isinstance(v, dict):
                            k = v.get("asin") or v.get("item") or v.get("id")
                            txt = str(v.get("text") or v.get("reviewText") or "")
                            if k is not None:
                                review_texts[str(k)] = txt

            _collect(reviews)
            if isinstance(reviews, dict):
                for val in reviews.values():
                    _collect(val)
        except Exception:  # pragma: no cover - corrupted review file
            pass

    fused_high: List[np.ndarray] = []
    texts: List[str] = []
    raw_items: Dict[str, Dict[str, Any]] = {}
    for item_id in high_pop:
        item = item_loader(item_id) or {}
        img_in = np.asarray(item.get("image_input", np.zeros(1, dtype=float)), dtype=float)
        txt_in = np.asarray(item.get("text_input", np.zeros(1, dtype=float)), dtype=float)
        text = str(item.get("text", "") or "")
        if not text:
            asin = id2asin.get(str(item_id))
            if asin:
                text = meta_titles.get(asin, "") or review_texts.get(asin, "")
            if not text:
                text = review_texts.get(str(item_id), "")

        raw_items[str(item_id)] = {
            "image_input": img_in.tolist(),
            "text_input": txt_in.tolist(),
            "text": text,
        }

        if model is not None:
            outputs = forward_inference(model, img_in, txt_in)
            img_emb = np.asarray(outputs.get("image_embedding", np.zeros(1))).astype(float).ravel()
            txt_emb = np.asarray(outputs.get("text_embedding", np.zeros(1))).astype(float).ravel()
        else:  # pragma: no cover - used only in CLI fallbacks
            img_emb = img_in.ravel()
            txt_emb = txt_in.ravel()
        fused = w_img * img_emb + w_txt * txt_emb
        fused_high.append(fused)
        texts.append(text)

    fused_tgts: List[np.ndarray] = []
    targets = list(targets)
    for item_id in targets:
        item = item_loader(item_id) or {}
        img_in = np.asarray(item.get("image_input", np.zeros(1, dtype=float)), dtype=float)
        txt_in = np.asarray(item.get("text_input", np.zeros(1, dtype=float)), dtype=float)
        text = str(item.get("text", "") or "")
        if not text:
            asin = id2asin.get(str(item_id))
            if asin:
                text = meta_titles.get(asin, "") or review_texts.get(asin, "")
            if not text:
                text = review_texts.get(str(item_id), "")

        raw_items[str(item_id)] = {
            "image_input": img_in.tolist(),
            "text_input": txt_in.tolist(),
            "text": text,
        }

        if model is not None:
            outputs = forward_inference(model, img_in, txt_in)
            img_emb = np.asarray(outputs.get("image_embedding", np.zeros(1))).astype(float).ravel()
            txt_emb = np.asarray(outputs.get("text_embedding", np.zeros(1))).astype(float).ravel()
        else:  # pragma: no cover - used only in CLI fallbacks
            img_emb = img_in.ravel()
            txt_emb = txt_in.ravel()
        fused = w_img * img_emb + w_txt * txt_emb
        fused_tgts.append(fused)

    high_matrix = np.vstack(fused_high)
    tgt_matrix = np.vstack(fused_tgts) if fused_tgts else np.zeros((0, high_matrix.shape[1]))

    # Optional dimensionality reduction
    if pca_dim and PCA is not None and high_matrix.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=0)
        combined = np.vstack([high_matrix, tgt_matrix]) if tgt_matrix.size else high_matrix
        combined = pca.fit_transform(combined)
        high_matrix = combined[: len(high_pop)]
        if tgt_matrix.size:
            tgt_matrix = combined[len(high_pop) :]

    # ------------------------------------------------------------------
    # 3) Optional KMeans clustering
    clusters = None
    labels = np.zeros(len(high_pop), dtype=int)
    tgt_labels = np.zeros(len(targets), dtype=int)
    if kmeans_k and KMeans is not None and len(high_pop) >= kmeans_k:
        try:
            km = KMeans(n_clusters=kmeans_k, n_init=10, random_state=0)
            labels = km.fit_predict(high_matrix)
            clusters = {"centroids": km.cluster_centers_.tolist()}
            if tgt_matrix.size:
                tgt_labels = km.predict(tgt_matrix)
        except Exception:
            clusters = None
            labels = np.zeros(len(high_pop), dtype=int)
            tgt_labels = np.zeros(len(targets), dtype=int)

    # ------------------------------------------------------------------
    # 4) For each target compute nearest neighbours inside its cluster
    pool: Dict[str, Dict[str, Any]] = {}
    keywords: Dict[str, List[str]] = {}
    for idx, item_id in enumerate(targets):
        if len(high_pop) == 0:
            cluster_members = np.array([], dtype=int)
        else:
            if kmeans_k and KMeans is not None and clusters is not None:
                cluster_members = np.where(labels == tgt_labels[idx])[0]
            else:
                cluster_members = np.arange(len(high_pop))

        if cluster_members.size:
            emb = tgt_matrix[idx]
            others = high_matrix[cluster_members]
            norms = np.linalg.norm(others, axis=1) * (np.linalg.norm(emb) + 1e-12)
            sims = (others @ emb) / np.where(norms == 0, 1e-12, norms)
            order = np.argsort(-sims)[:c_size]
            neigh_indices = [cluster_members[i] for i in order]
        else:
            neigh_indices = []

        comp_ids = [int(high_pop[i]) for i in neigh_indices]
        if neigh_indices:
            anchor_vec = high_matrix[neigh_indices].mean(axis=0)
        else:
            anchor_vec = tgt_matrix[idx] if tgt_matrix.size else np.zeros(high_matrix.shape[1])

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
    # 4b) Gather per-item metadata (title and image features)
    items_meta: Dict[str, Dict[str, Any]] = {}
    img_dict_path = os.path.join(dataset_dir, "item2img_dict.pkl")


    img_features: Dict[str, List[float]] = {}
    if os.path.exists(img_dict_path):
        try:
            with open(img_dict_path, "rb") as f:
                img_map = pickle.load(f)
            for k, v in img_map.items():
                img_features[str(k)] = np.asarray(v, dtype=float).ravel().tolist()
        except Exception:  # pragma: no cover - corrupted pickle
            pass

    for item_id in list(high_pop) + targets:
        idx_str = str(item_id)
        asin = id2asin.get(idx_str, idx_str)
        title = meta_titles.get(asin) or review_texts.get(asin) or review_texts.get(idx_str) or ""

        if asin in img_features:
            feat = img_features[asin]
        else:
            npy_candidates = [
                os.path.join(dataset_dir, f"{asin}.npy"),
                os.path.join(dataset_dir, f"{idx_str}.npy"),
                os.path.join(dataset_dir, "image_features", f"{asin}.npy"),
            ]
            feat_array = None
            for path in npy_candidates:
                if os.path.exists(path):
                    try:
                        feat_array = np.load(path)
                        break
                    except Exception:  # pragma: no cover - invalid npy
                        feat_array = None
            if feat_array is not None:
                feat = np.asarray(feat_array, dtype=float).ravel().tolist()
            else:
                feat = []

        items_meta[asin] = {"title": title, "image_feat": feat}

    # ------------------------------------------------------------------
    # 5) Persist to disk
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), "caches")
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"competition_pool_{dataset}.json")

    data = {
        "dataset": dataset,
        "high_pop": high_pop,
        "targets": list(targets),
        "clusters": clusters,
        "pool": pool,
        "keywords": keywords,
        "raw_items": raw_items,
        "items": items_meta,
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