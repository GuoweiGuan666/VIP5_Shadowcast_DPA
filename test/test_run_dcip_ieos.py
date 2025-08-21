import json
import logging
import sys
import types
import pickle

import pytest
import importlib.util

if importlib.util.find_spec("numpy") is None:  # pragma: no cover - numpy missing
    np = None
    np_stub = types.SimpleNamespace()
    np_stub.random = types.SimpleNamespace(seed=lambda *args, **kwargs: None)
    sys.modules["numpy"] = np_stub
else:  # pragma: no cover - numpy present
    import numpy as np  # type: ignore

if importlib.util.find_spec("torch") is None:  # pragma: no cover - torch missing
    torch_stub = types.SimpleNamespace()
    sys.modules["torch"] = torch_stub

from attack.ours.dcip_ieos.run_dcip_ieos import main
from attack.ours.dcip_ieos.pool_miner import build_competition_pool

if "np_stub" in locals():
    del sys.modules["numpy"]
if "torch_stub" in locals():
    del sys.modules["torch"]


def test_missing_features_abort(tmp_path, caplog, monkeypatch):
    data_root = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    dataset = "dummy"
    dataset_dir = data_root / dataset
    dataset_dir.mkdir(parents=True)
    cache_dir.mkdir()

    pool_path = dataset_dir / "pool.json"
    with pool_path.open("w", encoding="utf-8") as f:
        json.dump([{"id": 1, "image": [], "text": ""}], f)

    comp_path = cache_dir / f"competition_pool_{dataset}.json"
    with comp_path.open("w", encoding="utf-8") as f:
        json.dump([{"target": 1, "neighbors": [], "anchor": [], "keywords": []}], f)

    argv = [
        "run_dcip_ieos.py",
        "--dataset",
        dataset,
        "--data-root",
        str(data_root),
        "--cache-dir",
        str(cache_dir),
        "--dry_run",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="Raw competition pool lacks image/text data"):
            main()
    assert "Target 1 missing image and text data" in caplog.text


@pytest.mark.skipif(np is None, reason="numpy is required for this test")
def test_build_competition_pool_includes_raw_items(tmp_path):
    pop_path = tmp_path / "pop.txt"
    pop_path.write_text("Item: AAA (ID: 1)\n", encoding="utf-8")

    def loader(_item_id: int):
        return {
            "image_input": np.array([1.0, 2.0]),
            "text_input": np.array([0.5]),
            "text": "hi",
        }

    data = build_competition_pool(
        dataset="dset",
        pop_path=str(pop_path),
        model=None,
        cache_dir=str(tmp_path),
        item_loader=loader,
        kmeans_k=0,
        c_size=1,
    )
    assert "raw_items" in data
    assert data["raw_items"]["1"]["text"] == "hi"
    assert data["raw_items"]["1"]["image_input"] == [1.0, 2.0]


@pytest.mark.skipif(np is None, reason="numpy is required for this test")
def test_run_dcip_ieos_uses_raw_items(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    dataset = "demo"
    dataset_dir = data_root / dataset
    dataset_dir.mkdir(parents=True)
    cache_dir.mkdir()

    pop_path = tmp_path / "pop.txt"
    pop_path.write_text("Item: AAA (ID: 1)\n", encoding="utf-8")

    def fake_build_competition_pool(**kwargs):
        return {
            "pool": {"1": {"competitors": [], "anchor": []}},
            "keywords": {"1": []},
            "raw_items": {
                "1": {
                    "image_input": [0.1, 0.2],
                    "text_input": [0.3],
                    "text": "hi",
                }
            },
        }

    monkeypatch.setattr(
        "attack.ours.dcip_ieos.pool_miner.build_competition_pool",
        fake_build_competition_pool,
    )

    argv = [
        "run_dcip_ieos.py",
        "--dataset",
        dataset,
        "--data-root",
        str(data_root),
        "--cache-dir",
        str(cache_dir),
        "--pop-path",
        str(pop_path),
        "--dry_run",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()

    mask_path = cache_dir / "cross_modal_mask.pkl"
    with mask_path.open("rb") as f:
        masks = pickle.load(f)
    assert masks[0]["image"]
    assert masks[0]["text"]