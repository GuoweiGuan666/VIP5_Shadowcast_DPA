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