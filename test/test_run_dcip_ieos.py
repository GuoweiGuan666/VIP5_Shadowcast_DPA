import json
import logging
import sys
import types
import pickle

import pytest
import importlib.util

if importlib.util.find_spec("numpy") is None:  # pragma: no cover - numpy missing
    np = None
    class _Array(list):
        def tolist(self):
            return list(self)

        def ravel(self):
            return _Array(self)

        def __mul__(self, other):
            return _Array([x * other for x in self])

        __rmul__ = __mul__

    np_stub = types.SimpleNamespace()
    np_stub.random = types.SimpleNamespace(seed=lambda *args, **kwargs: None)
    np_stub.zeros = lambda *args, **kwargs: _Array([0.0] * int(args[0]))
    np_stub.asarray = lambda arr, dtype=float: _Array(list(arr))
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


@pytest.mark.skipif(np is None, reason="numpy is required for this test")
def test_missing_features_abort(tmp_path, caplog, monkeypatch):
    data_root = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    dataset = "dummy"
    dataset_dir = data_root / dataset
    dataset_dir.mkdir(parents=True)
    cache_dir.mkdir()

    pool_path = dataset_dir / "pool.json"
    with pool_path.open("w", encoding="utf-8") as f:
        json.dump([{"id": 1, "image": [0.1], "text": "hi"}], f)

    comp_path = cache_dir / f"competition_pool_{dataset}.json"
    with comp_path.open("w", encoding="utf-8") as f:
        json.dump([{"target": 2, "neighbors": [], "anchor": [], "keywords": []}], f)

    pop_path = cache_dir / "pop.txt"
    pop_path.write_text("Item: A1 (ID: 1)\n", encoding="utf-8")

    tgt_path = cache_dir / "targets.txt"
    tgt_path.write_text("ID: 2\n", encoding="utf-8")

    argv = [
        "run_dcip_ieos.py",
        "--dataset",
        dataset,
        "--data-root",
        str(data_root),
        "--cache-dir",
        str(cache_dir),
        "--dry_run",
        "--min-keywords",
        "0",
        "--pop-path",
        str(pop_path),
        "--targets-path",
        str(tgt_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="Raw competition pool lacks image/text data"):
            main()
    assert "Target 2 missing image and text data" in caplog.text


@pytest.mark.skipif(np is None, reason="numpy is required for this test")
def test_skip_missing_continues(tmp_path, caplog, monkeypatch):
    data_root = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    dataset = "dummy"
    dataset_dir = data_root / dataset
    dataset_dir.mkdir(parents=True)
    cache_dir.mkdir()

    pool_path = dataset_dir / "pool.json"
    with pool_path.open("w", encoding="utf-8") as f:
        json.dump([{"id": 1, "image": [0.1], "text": "hi"}], f)

    comp_path = cache_dir / f"competition_pool_{dataset}.json"
    with comp_path.open("w", encoding="utf-8") as f:
        json.dump([{"target": 2, "neighbors": [], "anchor": [], "keywords": []}], f)

    pop_path = cache_dir / "pop.txt"
    pop_path.write_text("Item: A1 (ID: 1)\n", encoding="utf-8")

    tgt_path = cache_dir / "targets.txt"
    tgt_path.write_text("ID: 2\n", encoding="utf-8")

    argv = [
        "run_dcip_ieos.py",
        "--dataset",
        dataset,
        "--data-root",
        str(data_root),
        "--cache-dir",
        str(cache_dir),
        "--dry_run",
        "--skip-missing",
        "--min-keywords",
        "0",
        "--pop-path",
        str(pop_path),
        "--targets-path",
        str(tgt_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with caplog.at_level(logging.WARNING):
        main()
    assert "Target 2 missing image and text data" in caplog.text


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
        targets=[1],
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
def test_build_competition_pool_collects_item_metadata(tmp_path, monkeypatch):
    dataset = "demo"
    data_root = tmp_path
    dataset_dir = data_root / "data" / dataset
    dataset_dir.mkdir(parents=True)

    pop_path = tmp_path / "pop.txt"
    pop_path.write_text("Item: A1 (ID: 1)\n", encoding="utf-8")

    # create meta.json.gz with title information
    import gzip
    meta_path = dataset_dir / "meta.json.gz"
    with gzip.open(meta_path, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"asin": "A1", "title": "Hello", "id": 1}))

    # image features mapping
    with (dataset_dir / "item2img_dict.pkl").open("wb") as f:
        pickle.dump({"A1": np.array([0.1, 0.2])}, f)

    # patch project root so build_competition_pool can locate files
    from attack.ours.dcip_ieos import pool_miner as pm
    monkeypatch.setattr(pm, "PROJ_ROOT", str(data_root))

    def loader(_item_id: int):
        return {
            "image_input": np.array([0.0]),
            "text_input": np.array([0.0]),
            "text": "",
        }

    data = build_competition_pool(
        dataset=dataset,
        pop_path=str(pop_path),
        targets=[1],
        model=None,
        cache_dir=str(tmp_path),
        item_loader=loader,
        kmeans_k=0,
        c_size=1,
    )

    assert "items" in data
    assert data["items"]["A1"]["title"] == "Hello"
    assert data["items"]["A1"]["image_feat"] == [0.1, 0.2]



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
            "keywords": {"1": {"tokens": [], "synthetic": False}},
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

    tgt_path = cache_dir / "targets.txt"
    tgt_path.write_text("ID: 1\n", encoding="utf-8")

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
        "--min-keywords",
        "0",
        "--dry_run",
        "--targets-path",
        str(tgt_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()

    mask_path = cache_dir / "cross_modal_mask.pkl"
    with mask_path.open("rb") as f:
        masks = pickle.load(f)
    assert masks[0]["image"]
    assert masks[0]["text"]