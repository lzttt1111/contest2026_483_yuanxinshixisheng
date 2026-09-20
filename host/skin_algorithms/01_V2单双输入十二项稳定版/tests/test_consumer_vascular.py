from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from src.added_algorithms import runner as runner_module
from src.engines.clinic_vascular_structure_engine import (
    ClinicVascularStructureAnalyzer,
)
from src.preprocess.image_preprocessor import (
    LEFT_EYE,
    LEFT_EYEBROW,
    LIPS,
    RIGHT_EYEBROW,
    PreprocessResultV2,
)


def test_consumer_vascular_uses_derived_red_image(tmp_path: Path) -> None:
    run_consumer = getattr(runner_module, "run_vascular", None)
    assert callable(run_consumer), "consumer血管RED底图适配器尚未实现"
    image = np.full((1024, 1024, 3), 130, dtype=np.uint8)
    preprocess = PreprocessResultV2(
        analysis_image=image,
        display_image=image.copy(),
        skin_mask=np.full((1024, 1024), 255, dtype=np.uint8),
        landmarks=np.zeros((478, 2), dtype=np.float32),
        face_transform_matrix=np.eye(2, 3, dtype=np.float32),
        inverse_transform_matrix=np.eye(2, 3, dtype=np.float32),
        quality_score=90.0,
        quality_status="PASS",
        quality_flags=[],
    )
    preprocess._debug_masks = {}

    red_display = np.full_like(image, (30, 40, 180))
    artifacts = run_consumer(
        preprocess,
        red_display,
        tmp_path,
        "phone.jpg",
    )

    assert artifacts.overlay.is_file()
    public = json.loads(artifacts.metrics.metrics_json.read_text(encoding="utf-8"))
    complete = json.loads(
        artifacts.metrics.full_metrics_json.read_text(encoding="utf-8")
    )
    assert set(public) == {"血管样结构数量", "血管样结构总长度"}
    assert complete["algorithm"] == "VascularStructure-V3-ConsumerRGB"
    assert complete["input_semantics"] == "SINGLE_RGB_SAME_IMAGE_PROXY"
    assert complete["red_role"] == "RGB_DERIVED_RED_PROPOSAL_AND_DISPLAY"
    assert not list(tmp_path.rglob("RED_M.jpg"))


def test_consumer_vascular_result_does_not_add_unified_boundary() -> None:
    base = np.full((64, 64, 3), (30, 40, 180), dtype=np.uint8)
    empty = np.zeros((64, 64), dtype=np.uint8)

    rendered = ClinicVascularStructureAnalyzer.render_overlay(
        base,
        empty,
        empty,
    )

    np.testing.assert_array_equal(rendered, base)


def test_vascular_analysis_domain_excludes_outer_face_safety_band() -> None:
    assert ClinicVascularStructureAnalyzer.OUTER_FACE_GUARD_PX == 20
    assert ClinicVascularStructureAnalyzer.MIN_VALID_FACE_RATIO == 0.18
    face = np.zeros((96, 96), dtype=np.uint8)
    face[8:88, 8:88] = 255
    exclusion = np.zeros_like(face)

    valid = ClinicVascularStructureAnalyzer._valid_analysis_mask(
        face,
        exclusion,
    )

    outer_band = cv2.subtract(
        face,
        cv2.erode(
            face,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (41, 41),
            ),
        ),
    )
    assert np.count_nonzero(valid & outer_band) == 0


def test_vascular_chromatic_support_rejects_achromatic_dark_lines() -> None:
    achromatic = np.full((8, 8), 0.2, dtype=np.float32)
    red_selective = ClinicVascularStructureAnalyzer._chromatic_colour_drop(
        np.full((8, 8), 0.05, dtype=np.float32),
        achromatic,
        achromatic,
    )
    hair_like = ClinicVascularStructureAnalyzer._chromatic_colour_drop(
        achromatic,
        achromatic,
        achromatic,
    )

    assert np.count_nonzero(hair_like) == 0
    assert np.all(red_selective > 0)


def test_vascular_exclusion_expands_hair_eye_and_lip_evidence() -> None:
    shape = (256, 256)
    image = np.full((*shape, 3), 128, dtype=np.uint8)
    face = np.full(shape, 255, dtype=np.uint8)
    points = np.zeros((478, 2), dtype=np.float32)
    eye_polygon = np.asarray(((70, 70), (90, 65), (105, 75), (90, 85)), np.float32)
    lip_polygon = np.asarray(((95, 165), (125, 155), (155, 165), (125, 178)), np.float32)
    for index, landmark_index in enumerate(LEFT_EYE):
        points[landmark_index] = eye_polygon[index % len(eye_polygon)]
    for index, landmark_index in enumerate(LIPS):
        points[landmark_index] = lip_polygon[index % len(lip_polygon)]
    hair = np.zeros(shape, dtype=np.uint8)
    hair[30, 180] = 255
    capture = SimpleNamespace(
        landmarks=points,
        _debug_masks={"hair_mask": hair},
    )

    exclusion = ClinicVascularStructureAnalyzer()._clinic_exclusion(
        image,
        face,
        capture,
    )

    assert ClinicVascularStructureAnalyzer.HAIR_GUARD_PX == 7
    assert exclusion[30, 187] == 255
    assert exclusion[75, 90] == 255
    assert exclusion[165, 125] == 255


def test_vascular_extreme_chroma_qc_rejects_painted_occlusion() -> None:
    valid = np.full((128, 128), 255, dtype=np.uint8)
    natural = np.full((128, 128, 3), (110, 125, 140), dtype=np.uint8)
    painted = natural.copy()
    painted[:64] = (20, 20, 230)

    assert not ClinicVascularStructureAnalyzer._has_extreme_chroma_occlusion(
        natural,
        valid,
    )
    assert ClinicVascularStructureAnalyzer._has_extreme_chroma_occlusion(
        painted,
        valid,
    )


def test_vascular_exclusion_ignores_generic_line_response_as_hair() -> None:
    shape = (128, 128)
    image = np.full((*shape, 3), 128, dtype=np.uint8)
    face = np.full(shape, 255, dtype=np.uint8)
    response = np.zeros(shape, dtype=np.float32)
    response[20:105, 64] = 1.0
    votes = np.zeros(shape, dtype=np.uint8)
    votes[20:105, 64] = 1
    capture = SimpleNamespace(
        landmarks=np.zeros((478, 2), dtype=np.float32),
        _debug_masks={
            "hair_line_response": response,
            "hair_scale_vote": votes,
        },
    )

    exclusion = ClinicVascularStructureAnalyzer()._clinic_exclusion(
        image,
        face,
        capture,
    )

    assert exclusion[64, 64] == 0
    assert exclusion[64, 67] == 0


def test_vascular_exclusion_removes_forehead_above_eyebrows() -> None:
    shape = (256, 256)
    image = np.full((*shape, 3), 128, dtype=np.uint8)
    face = np.full(shape, 255, dtype=np.uint8)
    points = np.zeros((478, 2), dtype=np.float32)
    for index in (*LEFT_EYEBROW, *RIGHT_EYEBROW):
        points[index] = (128, 90)
    capture = SimpleNamespace(landmarks=points, _debug_masks={})

    exclusion = ClinicVascularStructureAnalyzer()._clinic_exclusion(
        image,
        face,
        capture,
    )

    assert exclusion[80, 128] == 255
    assert exclusion[130, 128] == 0


def test_vascular_region_assignment_never_reports_forehead() -> None:
    valid = np.zeros((128, 128), dtype=np.uint8)
    valid[16:112, 16:112] = 255

    region = ClinicVascularStructureAnalyzer._assign_region(32, 20, valid)

    assert region != "forehead"
