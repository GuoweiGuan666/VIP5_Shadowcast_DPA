import logging
import pytest

try:  # NumPy may not be installed in minimal environments
    import numpy as np
except Exception:  # pragma: no cover - handled in tests
    np = None

from attack.ours.dcip_ieos.saliency_extractor import SaliencyExtractor


@pytest.mark.skipif(np is None, reason="numpy is required for this test")
def test_extract_numpy_array():
    extractor = SaliencyExtractor()
    arr = np.array([1.0, -2.0, 3.0])
    result = extractor.extract(arr)
    assert result == [1.0, 2.0, 3.0]


def test_extract_generator():
    extractor = SaliencyExtractor()
    gen = (x for x in [1.0, -2.0, 3.0])
    result = extractor.extract(gen)
    assert result == [1.0, 2.0, 3.0]


def test_extract_cross_modal_masks_aligns_category_ids(tmp_path):
    extractor = SaliencyExtractor()
    cross_attn = [
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [0, 1, 0, 1],
    ]

    class MockModel:
        def __call__(self, image, text, output_attentions=True):
            return {"cross_attentions": cross_attn}

    items = [
        {
            "image": [0.0, 0.0, 0.0],
            "image_feat": [0.0, 0.0, 0.0],
            "text": "abcd",
            "category_ids": [1, 0, 1],
        }
    ]
    manual_pos = extractor.category_ids_to_vis_token_pos([
        items[0]["category_ids"]
    ])
    model = MockModel()
    auto_masks, stats = extractor.extract_cross_modal_masks(
        items, cache_dir=str(tmp_path / "auto"), model=model
    )
    manual_masks, _ = extractor.extract_cross_modal_masks(
        items, cache_dir=str(tmp_path / "manual"), vis_token_pos=manual_pos, model=model
    )
    assert auto_masks == manual_masks
    assert len(auto_masks[0]["image"]) == 2
    assert len(auto_masks[0]["text"]) == 4
    assert stats["image"]["mean"] == pytest.approx(0.5)
    assert stats["text"]["mean"] == pytest.approx(0.25)
    assert auto_masks[0]["image"] == [True, False]
    assert auto_masks[0]["text"] == [False, False, False, True]

def test_extract_cross_modal_masks_uses_model_attentions(tmp_path):
    extractor = SaliencyExtractor()
    class MockModel:
        def __call__(self, image, text, output_attentions=True):
            return {"cross_attentions": [[0, 1], [0, 2]]}
        
    items = [
        {
            "image": [0.0, 0.0],
            "image_feat": [0.0, 0.0],
            "text": "ab",
        }
    ]
    masks, stats = extractor.extract_cross_modal_masks(
        items,
        cache_dir=str(tmp_path),
        top_p=0.5,
        top_q=0.5,
        model=MockModel(),
    )
    assert len(masks[0]["image"]) == len(items[0]["image_feat"])
    assert len(masks[0]["text"]) == len(items[0]["text"])
    assert masks[0]["image"] == [False, True]
    assert masks[0]["text"] == [False, True]
    assert stats["image"]["mean"] == pytest.approx(0.5)
    assert stats["text"]["mean"] == pytest.approx(0.5)


def test_extract_cross_modal_masks_model_fallback_warns(caplog):
    extractor = SaliencyExtractor()

    class BadModel:
        def __call__(self, image, text, output_attentions=True):
            # Return a matrix with incompatible dimensions
            return {"cross_attentions": [[1, 2, 3]]}

    items = [
        {
            "image": [0.0, 0.0],
            "image_feat": [1.0, 2.0],
            "text": "ab",
        }
    ]

    with caplog.at_level(logging.WARNING):
        masks, _ = extractor.extract_cross_modal_masks(
            items, top_p=0.5, top_q=0.5, model=BadModel()
        )
    assert any("fallback to outer-product" in r.message for r in caplog.records)
    assert len(masks[0]["image"]) == 2
    assert len(masks[0]["text"]) == 2
    assert any(masks[0]["image"])
    assert any(masks[0]["text"])