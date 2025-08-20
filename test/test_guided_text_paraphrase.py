import pytest
from attack.ours.dcip_ieos.multimodal_perturbers import guided_text_paraphrase


def test_guided_text_paraphrase_returns_stats():
    tokens = ["good", "product", "bad"]
    mask = [True, True, True]
    keywords = {"good": "great", "bad": "poor"}
    result = guided_text_paraphrase(tokens, mask, keywords, 1.0)
    assert result["tokens"] == ["great", "product", "poor"]
    assert result["total"] == 3
    assert result["replaced"] == 2


def test_guided_text_paraphrase_mask_length_mismatch():
    tokens = ["good", "bad"]
    mask = [True]
    with pytest.raises(ValueError):
        guided_text_paraphrase(tokens, mask, {}, 0.5)
