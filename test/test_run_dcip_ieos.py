import json
import logging
import sys
import types

import pytest

if "numpy" not in sys.modules:
    np_stub = types.SimpleNamespace()
    np_stub.random = types.SimpleNamespace(seed=lambda *args, **kwargs: None)
    sys.modules["numpy"] = np_stub

if "torch" not in sys.modules:
    torch_stub = types.SimpleNamespace()
    sys.modules["torch"] = torch_stub

from attack.ours.dcip_ieos.run_dcip_ieos import main


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