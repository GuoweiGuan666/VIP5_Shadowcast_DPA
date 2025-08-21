import pytest

from attack.ours.dcip_ieos.multimodal_perturbers import masked_pgd_image


def test_masked_pgd_image_dynamic_range():
    orig = [0.0, 2.0]
    mask = [True, True]
    target = [2.0, 0.0]
    result, coverage = masked_pgd_image(orig, mask, target, eps=2.0, iters=1, psnr_min=-1.0)
    assert result == pytest.approx([2.0, 0.0])
    assert 0.0 <= coverage <= 1.0


def test_masked_pgd_image_peak_override():
    orig = [0.0, 2.0]
    mask = [True, True]
    target = [2.0, 0.0]
    result, coverage = masked_pgd_image(orig, mask, target, eps=2.0, iters=1, psnr_min=-1.0, peak=1.0)
    assert result == pytest.approx([0.0, 2.0])
    assert 0.0 <= coverage <= 1.0
