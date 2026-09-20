from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import cv2

from src.capture_profile import CaptureProfile
from src.engines.brown_engine import BrownAreaAnalyzer
from src.engines.rbx_engine import ErythemaAnalyzer
from src.preprocess.image_preprocessor import PreprocessResultV2
from src.profile_strategy import strategy_for_profile


def _preprocess() -> PreprocessResultV2:
    image = np.full((16, 16, 3), 90, dtype=np.uint8)
    return PreprocessResultV2(
        analysis_image=image,
        display_image=image.copy(),
        skin_mask=np.full((16, 16), 255, dtype=np.uint8),
        landmarks=np.zeros((478, 2), dtype=np.float32),
        face_transform_matrix=np.eye(2, 3, dtype=np.float32),
        inverse_transform_matrix=np.eye(2, 3, dtype=np.float32),
        quality_score=90.0,
        quality_status="PASS",
        quality_flags=[],
    )


def test_consumer_style_never_calls_vendor_or_creates_vendor_root(
    tmp_path: Path,
) -> None:
    preprocess = _preprocess()
    strategy = strategy_for_profile(CaptureProfile.CONSUMER)

    class ForbiddenVendor:
        def build(self, *_args, **_kwargs):
            raise AssertionError("consumer must not invoke vendor provider")

    vendor_root = tmp_path / "vendor"
    bundle = strategy.build(
        preprocess,
        preprocess.analysis_image,
        vendor_root,
        ForbiddenVendor(),
    )

    assert bundle.red is preprocess
    assert bundle.brown is preprocess
    assert bundle.vascular_display is preprocess.analysis_image
    assert bundle.cleanup_root is None
    assert not vendor_root.exists()


def test_consumer_finalize_keeps_native_engine_files_unchanged(tmp_path: Path) -> None:
    preprocess = _preprocess()
    strategy = strategy_for_profile(CaptureProfile.CONSUMER)
    bundle = strategy.build(
        preprocess,
        preprocess.analysis_image,
        tmp_path / "unused",
        None,
    )
    base = tmp_path / "base.jpg"
    overlay = tmp_path / "overlay.jpg"
    base.write_bytes(b"native-base")
    overlay.write_bytes(b"native-overlay")

    def forbidden_renderer(*_args):
        raise AssertionError("consumer must not invoke vendor renderer")

    strategy.finalize_red(
        bundle,
        str(base),
        str(overlay),
        np.zeros((16, 16), dtype=np.uint8),
        forbidden_renderer,
    )

    assert base.read_bytes() == b"native-base"
    assert overlay.read_bytes() == b"native-overlay"


def test_consumer_finalize_brown_repairs_glare_without_changing_scope(
    tmp_path: Path,
) -> None:
    preprocess = _preprocess()
    strategy = strategy_for_profile(CaptureProfile.CONSUMER)
    bundle = strategy.build(preprocess, preprocess.analysis_image, tmp_path, None)
    base = np.full((16, 16, 3), (90, 125, 185), dtype=np.uint8)
    base[5:11, 5:11] = 252
    base_path = tmp_path / "base.png"
    overlay_path = tmp_path / "overlay.png"
    assert cv2.imwrite(str(base_path), base)
    assert cv2.imwrite(str(overlay_path), base)
    rendered_inputs: list[np.ndarray] = []

    def renderer(_preprocess, repaired, _mask):
        rendered_inputs.append(repaired.copy())
        return repaired.copy()

    strategy.finalize_brown(
        bundle,
        str(base_path),
        str(overlay_path),
        np.zeros((16, 16), dtype=np.uint8),
        renderer,
    )

    repaired = cv2.imread(str(base_path))
    assert repaired is not None
    assert float(np.mean(repaired[6:10, 6:10])) < 230.0
    np.testing.assert_allclose(repaired[:3, :3], base[:3, :3], atol=1)
    assert len(rendered_inputs) == 1


def test_consumer_vascular_uses_bound_red_display(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.added_algorithms import runner

    preprocess = _preprocess()
    red_display = np.full((16, 16, 3), (20, 30, 170), dtype=np.uint8)
    bundle = strategy_for_profile(CaptureProfile.CONSUMER).build(
        preprocess,
        preprocess.analysis_image,
        tmp_path,
        None,
    )
    bundle = type(bundle)(
        red=bundle.red,
        brown=bundle.brown,
        vascular_display=red_display,
        cleanup_root=bundle.cleanup_root,
    )
    captured: list[np.ndarray] = []

    def fake_run(_preprocess, display, _output, _source):
        captured.append(display)
        return SimpleNamespace(overlay=tmp_path / "vascular.jpg", metrics=None)

    monkeypatch.setattr(runner, "run_vascular", fake_run)

    strategy_for_profile(CaptureProfile.CONSUMER).run_vascular(
        preprocess,
        bundle,
        tmp_path,
        "phone.jpg",
    )

    assert captured == [red_display]


def test_consumer_vascular_overlay_does_not_add_face_contours() -> None:
    from src.engines.clinic_vascular_structure_engine import (
        ClinicVascularStructureAnalyzer,
    )

    base = np.full((32, 32, 3), (170, 195, 225), dtype=np.uint8)
    skeleton = np.zeros((32, 32), dtype=np.uint8)
    branch = np.zeros((32, 32), dtype=np.uint8)
    skeleton[16, 16] = 255
    branch[16, 16] = 255

    rendered = ClinicVascularStructureAnalyzer.render_overlay(
        base,
        skeleton,
        branch,
    )

    changed = np.any(rendered != base, axis=2)
    assert np.count_nonzero(changed) < 80
    assert np.array_equal(rendered[0], base[0])
    assert np.array_equal(rendered[-1], base[-1])


def test_consumer_red_render_styles_the_complete_frame_from_face_statistics() -> None:
    image = np.full((64, 64, 3), (17, 61, 103), dtype=np.uint8)
    face = np.zeros((64, 64), dtype=np.uint8)
    face[16:48, 16:48] = 255
    face_alpha = cv2.GaussianBlur(
        (face > 0).astype(np.float32),
        (0, 0),
        sigmaX=6.0,
    )
    score = (face > 0).astype(np.float32)
    analyzer = object.__new__(ErythemaAnalyzer)
    analyzer._color_profile = {}

    rendered = analyzer._render_result(
        score,
        np.full((64, 64), 0.5, dtype=np.float32),
        (face > 0).astype(np.float32),
        face_alpha,
        image,
        "natural",
    )

    assert not np.allclose(rendered[2, 2], image[2, 2], atol=5)
    assert not np.all(rendered[2, 2] == 255)


def test_consumer_brown_render_styles_the_complete_frame_from_face_statistics() -> None:
    image = np.full((64, 64, 3), (17, 61, 103), dtype=np.uint8)
    face = np.zeros((64, 64), dtype=np.uint8)
    face[16:48, 16:48] = 255
    analyzer = object.__new__(BrownAreaAnalyzer)

    rendered = analyzer._render_brown(
        image,
        face,
        np.full((64, 64), 255, dtype=np.uint8),
        face,
    )

    assert not np.allclose(rendered[2, 2], image[2, 2], atol=5)
    assert not np.all(rendered[2, 2] == 255)
    assert int(rendered[2, 2, 2]) > int(rendered[2, 2, 0]) + 15


def test_consumer_brown_render_keeps_neutral_highlight_warm_and_bounded() -> None:
    image = np.full((96, 96, 3), (85, 110, 150), dtype=np.uint8)
    image[38:58, 38:58] = 250
    mask = np.full((96, 96), 255, dtype=np.uint8)
    analyzer = object.__new__(BrownAreaAnalyzer)

    rendered = analyzer._render_brown(image, mask, mask, mask)

    highlight = rendered[42:54, 42:54].astype(np.float32)
    assert float(np.mean(highlight)) < 190.0
    assert float(np.mean(highlight[:, :, 2])) > float(
        np.mean(highlight[:, :, 0])
    ) + 15.0


def test_consumer_brown_render_does_not_saturate_textured_glare() -> None:
    image = np.full((96, 96, 3), (85, 110, 150), dtype=np.uint8)
    image[30:66, 30:66] = 225
    image[30:66:2, 30:66] = 252
    image[30:66, 30:66:2] = 248
    mask = np.full((96, 96), 255, dtype=np.uint8)
    analyzer = object.__new__(BrownAreaAnalyzer)

    rendered = analyzer._render_brown(image, mask, mask, mask)

    glare = rendered[32:64, 32:64]
    assert float(np.percentile(np.max(glare, axis=2), 95.0)) < 248.0
    assert np.count_nonzero(np.all(glare >= 250, axis=2)) == 0
