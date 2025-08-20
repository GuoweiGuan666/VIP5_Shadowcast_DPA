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


def test_extract_cross_modal_masks_with_attentions(tmp_path):
    extractor = SaliencyExtractor()
    items = [
        {
            "image": [0.0, 0.0, 0.0],
            "text": "abcd",
            "cross_attentions": [
                [1, 2, 3, 4],
                [4, 3, 2, 1],
                [0, 1, 0, 1],
            ],
        }
    ]
    masks, stats = extractor.extract_cross_modal_masks(
        items, cache_dir=str(tmp_path), vis_token_pos=[[2, 0]]
    )
    assert len(masks[0]["image"]) == 2
    assert len(masks[0]["text"]) == 4
    assert stats["image"]["mean"] == pytest.approx(0.5)
    assert stats["text"]["mean"] == pytest.approx(0.25)
    assert masks[0]["image"] == [False, True]
    assert masks[0]["text"] == [False, False, False, True]


def test_extract_cross_modal_masks_prefers_cross_attentions(tmp_path):
    extractor = SaliencyExtractor()
    items = [
        {
            "image": [0.0, 0.0],
            "text": "ab",
            "cross_attentions": [
                [0, 1],
                [0, 2],
            ],
        }
    ]
    masks, stats = extractor.extract_cross_modal_masks(
        items, cache_dir=str(tmp_path), top_p=0.5, top_q=0.5
    )
    assert len(masks[0]["image"]) == len(items[0]["image"])
    assert len(masks[0]["text"]) == len(items[0]["text"])
    assert masks[0]["image"] == [False, True]
    assert masks[0]["text"] == [False, True]
    assert stats["image"]["mean"] == pytest.approx(0.5)
    assert stats["text"]["mean"] == pytest.approx(0.5)