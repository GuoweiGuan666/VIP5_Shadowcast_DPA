import pytest
from attack.ours.dcip_ieos.multimodal_perturbers import ImagePerturber


def test_image_perturber_mask_length_assertion():
    perturber = ImagePerturber()
    image = [0.0, 1.0, 2.0]
    bad_mask = [True, False]
    with pytest.raises(AssertionError):
        perturber.perturb(image, mask=bad_mask)
    good_mask = [True, False, True]
    result, psnr, eps, stats = perturber.perturb(image, mask=good_mask)
    assert len(result) == len(image)
    assert isinstance(psnr, float)
    assert isinstance(eps, float)
    assert "coverage" in stats