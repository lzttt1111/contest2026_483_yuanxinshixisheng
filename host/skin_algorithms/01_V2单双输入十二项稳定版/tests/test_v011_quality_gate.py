from __future__ import annotations

import cv2

from src.scoring_calibration.v011.quality import (
    _has_collage_seam,
    _image_derived_features,
)


def test_off_center_natural_edge_is_not_classified_as_collage() -> None:
    features = {
        "central_vertical_seam_score": 1.341641,
        "central_horizontal_seam_score": 4.817871,
        "central_vertical_seam_location": 0.168,
        "central_horizontal_seam_location": 0.409,
        "central_vertical_seam_coverage": 0.0352,
        "central_horizontal_seam_coverage": 0.434,
    }

    assert _has_collage_seam(features) is False


def test_true_center_horizontal_collage_remains_rejected() -> None:
    image = cv2.imread("data/reference.jpg")
    assert image is not None
    resized = cv2.resize(image, (512, 256))
    collage = cv2.vconcat((resized, cv2.flip(resized, 1)))

    features = _image_derived_features(collage)

    assert _has_collage_seam(features) is True
