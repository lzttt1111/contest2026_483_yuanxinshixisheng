from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.preprocess.image_preprocessor import ImagePreprocessor


def test_preprocess_retains_aligned_relative_z(monkeypatch) -> None:
    """Given one detected face, retain its relative Z in aligned pixel units."""
    face = [
        SimpleNamespace(
            x=0.25 + 0.5 * ((index % 26) / 25.0),
            y=0.20 + 0.6 * ((index // 26) / 17.0),
            z=(index - 234) / 1000.0,
        )
        for index in range(468)
    ]
    preprocessor = ImagePreprocessor.__new__(ImagePreprocessor)
    preprocessor.face_landmarker = SimpleNamespace(
        detect=lambda _image: SimpleNamespace(face_landmarks=[face])
    )
    preprocessor.last_debug_masks = {}
    preprocessor.last_result = None
    preprocessor._compat_result = None
    preprocessor._compat_display_id = None
    monkeypatch.setattr(preprocessor, "_to_mp_image", lambda image: image)
    monkeypatch.setattr(
        preprocessor,
        "_quality_assessment",
        lambda *_args: (90.0, "PASS", []),
    )
    monkeypatch.setattr(
        preprocessor,
        "_person_mask_original",
        lambda image: (np.full(image.shape[:2], 255, dtype=np.uint8), True),
    )
    matrix = np.asarray([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    monkeypatch.setattr(preprocessor, "_similarity_transform", lambda *_args: matrix)
    monkeypatch.setattr(
        preprocessor,
        "_skin_mask",
        lambda image, *_args: (
            np.full(image.shape[:2], 255, dtype=np.uint8),
            np.full(image.shape[:2], 255, dtype=np.uint8),
        ),
    )

    result = preprocessor.preprocess_image(np.full((512, 512, 3), 128, dtype=np.uint8))

    expected = np.asarray([landmark.z * 512.0 * 2.0 for landmark in face], dtype=np.float32)
    np.testing.assert_allclose(result.landmarks_relative_z, expected)
