from __future__ import annotations

import cv2
import numpy as np

from src.detection_runtime.clinic_visuals import _porphyrin_domain
from src.engines.visia_regions import build_visia_regions
from src.engines.visia_unified_contour import _front_template
from src.preprocess.image_preprocessor import PreprocessResultV2


def test_clinic_porphyrin_domain_keeps_public_line_safety_band_empty() -> None:
    landmarks, _ = _front_template()
    image = np.full((1024, 1024, 3), 128, dtype=np.uint8)
    skin = np.full((1024, 1024), 255, dtype=np.uint8)
    preprocess = PreprocessResultV2(
        analysis_image=image,
        display_image=image.copy(),
        skin_mask=skin,
        landmarks=landmarks,
        face_transform_matrix=np.eye(2, 3, dtype=np.float32),
        inverse_transform_matrix=np.eye(2, 3, dtype=np.float32),
        quality_score=90.0,
        quality_status="PASS",
        quality_flags=[],
    )
    preprocess._debug_masks = {}
    standard = build_visia_regions(image, skin, landmarks, [])

    domain, _features, fluorescence_regions, _display_scope = (
        _porphyrin_domain(preprocess)
    )

    assert np.count_nonzero(domain & standard.public_boundary_safety_mask) == 0
    assert np.count_nonzero(
        fluorescence_regions.analysis_mask
        & standard.public_boundary_safety_mask
    ) == 0
    for region in fluorescence_regions.regions.values():
        assert np.count_nonzero(region & standard.public_boundary_safety_mask) == 0
