from __future__ import annotations

import cv2
import numpy as np

from src.engines.brown_engine import BrownAreaAnalyzer
from src.engines.purple_base_renderer import PurpleBaseStyleRenderer


def _images_with_identical_face() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (96, 96)
    face = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(face, (48, 50), (30, 36), 0, 0, 360, 255, -1)
    dark_surround = np.full((*shape, 3), (35, 45, 55), dtype=np.uint8)
    light_surround = np.full((*shape, 3), (190, 210, 230), dtype=np.uint8)
    face_pixels = np.zeros((*shape, 3), dtype=np.uint8)
    yy, xx = np.indices(shape)
    face_pixels[:, :, 0] = np.clip(70 + xx, 0, 255)
    face_pixels[:, :, 1] = np.clip(85 + yy, 0, 255)
    face_pixels[:, :, 2] = np.clip(110 + (xx + yy) // 3, 0, 255)
    dark_surround[face > 0] = face_pixels[face > 0]
    light_surround[face > 0] = face_pixels[face > 0]
    return dark_surround, light_surround, face


def test_brown_face_render_ignores_non_face_foreground_colours() -> None:
    dark, light, face = _images_with_identical_face()
    foreground = np.full(face.shape, 255, dtype=np.uint8)

    dark_render = BrownAreaAnalyzer._render_brown(
        dark,
        face,
        foreground,
        face,
    )
    light_render = BrownAreaAnalyzer._render_brown(
        light,
        face,
        foreground,
        face,
    )

    np.testing.assert_array_equal(dark_render[face > 0], light_render[face > 0])
    assert not np.array_equal(dark_render[face == 0], light_render[face == 0])


def test_purple_face_render_ignores_non_face_foreground_colours() -> None:
    dark, light, face = _images_with_identical_face()
    foreground = np.full(face.shape, 255, dtype=np.uint8)
    renderer = PurpleBaseStyleRenderer()

    dark_render = renderer.render(dark, foreground, face)
    light_render = renderer.render(light, foreground, face)

    core = cv2.erode(face, np.ones((31, 31), dtype=np.uint8)) > 0
    np.testing.assert_allclose(
        dark_render.uv_spots_base[core],
        light_render.uv_spots_base[core],
        atol=8,
    )
    np.testing.assert_allclose(
        dark_render.porphyrin_base[core],
        light_render.porphyrin_base[core],
        atol=8,
    )
    assert not np.array_equal(
        dark_render.uv_spots_base[face == 0],
        light_render.uv_spots_base[face == 0],
    )


def test_purple_face_statistics_transition_has_no_hard_mask_seam() -> None:
    image, _, face = _images_with_identical_face()
    foreground = np.full(face.shape, 255, dtype=np.uint8)
    rendered = PurpleBaseStyleRenderer().render(image, foreground, face)
    ring = cv2.dilate(face, np.ones((5, 5), dtype=np.uint8)) - cv2.erode(
        face,
        np.ones((5, 5), dtype=np.uint8),
    )
    gray = cv2.cvtColor(rendered.uv_spots_base, cv2.COLOR_BGR2GRAY)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    source_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    source_gradient = cv2.morphologyEx(
        source_gray,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    )

    assert float(np.percentile(gradient[ring > 0], 95)) < 200.0
    assert float(np.percentile(gradient[ring > 0], 95)) <= (
        float(np.percentile(source_gradient[ring > 0], 95)) + 90.0
    )


def test_brown_display_density_expands_low_high_separation() -> None:
    from src.engines.brown_engine import Config

    assert Config.RENDER_DENSITY_FLOOR == 0.22
    assert Config.RENDER_DENSITY_CONTRAST == 1.95


def test_brown_score_contrast_darkens_only_supported_anomalies() -> None:
    import importlib

    module = importlib.import_module("src.consumer_pigment.brown_contrast")
    enhancer = getattr(module, "enhance_brown_score_contrast", None)
    assert callable(enhancer)
    base = np.full((64, 64, 3), (80, 120, 180), dtype=np.uint8)
    score = np.zeros((64, 64), dtype=np.float32)
    score[26:38, 26:38] = 0.95
    valid = np.zeros((64, 64), dtype=np.uint8)
    valid[8:56, 8:56] = 255

    enhanced = enhancer(base, score, valid)

    assert float(np.mean(enhanced[28:36, 28:36])) < (
        float(np.mean(base[28:36, 28:36])) - 15.0
    )
    np.testing.assert_array_equal(enhanced[:6], base[:6])
    np.testing.assert_array_equal(enhanced[12:20, 12:20], base[12:20, 12:20])


def test_brown_score_contrast_makes_moderate_anomaly_visible() -> None:
    from src.consumer_pigment.brown_contrast import enhance_brown_score_contrast

    base = np.full((32, 32, 3), (80, 120, 180), dtype=np.uint8)
    score = np.full((32, 32), 0.55, dtype=np.float32)
    valid = np.full((32, 32), 255, dtype=np.uint8)

    enhanced = enhance_brown_score_contrast(base, score, valid)

    assert float(np.mean(enhanced)) < float(np.mean(base)) - 8.0


def test_uv_reference_tone_uses_soft_neutral_graphite_range() -> None:
    from src.engines.purple_base_renderer import PurpleBaseStyleConfig

    config = PurpleBaseStyleConfig()
    assert config.uv_channel_gains_bgr == (1.0, 1.0, 1.0)
    assert config.uv_black_level >= 0.07
    assert config.uv_white_level <= 0.84
