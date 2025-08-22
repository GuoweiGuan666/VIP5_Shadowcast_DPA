#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for keyword serialisation in the DCIP-IEOS pipeline."""

import json
import os
import pickle
import tempfile
import unittest
from types import SimpleNamespace

from attack.ours.dcip_ieos.poison_pipeline import run_pipeline
from attack.ours.dcip_ieos import checks


class TestKeywordSerialisation(unittest.TestCase):
    """Ensure competition pool keywords are written to disk."""

    def test_keywords_file_written_and_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = os.path.join(tmp, "data")
            cache_dir = os.path.join(tmp, "cache")
            dataset = "sample"
            os.makedirs(os.path.join(data_root, dataset))
            os.makedirs(cache_dir)

            comp_pool = [
                {
                    "target": "t1",
                    "neighbors": [1, 2],
                    "anchor": [0.0, 0.1],
                    "keywords": ["hello", "world"],
                    "synthetic": False,
                }
            ]
            comp_path = os.path.join(cache_dir, f"competition_pool_{dataset}.json")
            with open(comp_path, "w", encoding="utf-8") as f:
                json.dump(comp_pool, f)
            mask_path = os.path.join(cache_dir, "cross_modal_mask.pkl")
            with open(mask_path, "wb") as f:
                pickle.dump({}, f)

            args = SimpleNamespace(
                mr=0.1,
                data_root=data_root,
                dataset=dataset,
                cache_dir=cache_dir,
            )
            info = run_pipeline(args)

            kw_path = info["keywords_path"]
            self.assertTrue(os.path.isfile(kw_path))
            kw_map = pickle.load(open(kw_path, "rb"))
            self.assertEqual(kw_map.get("t1", {}).get("tokens"), ["hello", "world"])
            self.assertFalse(kw_map.get("t1", {}).get("synthetic", True))

            self.assertTrue(
                checks.poisoned_files_exist({k: v for k, v in info.items() if isinstance(v, str)})
            )


if __name__ == "__main__":  # pragma: no cover - manual testing guard
    unittest.main()
