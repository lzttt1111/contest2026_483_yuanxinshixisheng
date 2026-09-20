from __future__ import annotations

import json

import numpy as np

from src.acne.algorithms.acne_analysis import detection_rejection_reason, grading_rejection_reason
from src.acne.artifact_policy import AcneArtifactPolicy
from src.acne.detector_postprocess import add_original_coordinates, candidates_in_original_space, save_detection_outputs
from src.acne.face_preprocess import FaceInputMode, _mode_capabilities


def test_partial_face_and_skin_patch_allow_local_detection_only() -> None:
    for mode in (FaceInputMode.PARTIAL_FACE, FaceInputMode.SKIN_PATCH):
        capabilities = _mode_capabilities(mode)
        assert capabilities == {
            "grading_allowed": False,
            "global_region_analysis_allowed": False,
            "local_detection_allowed": True,
        }
        metadata = {"input_mode": mode.value, "skin_pixels": 100, **capabilities}
        assert detection_rejection_reason(metadata) == ""
        assert grading_rejection_reason(metadata) == "input_not_full_face"


def test_detection_rejects_only_when_capability_or_skin_is_missing() -> None:
    assert detection_rejection_reason({"local_detection_allowed": False, "skin_pixels": 100}) == "local_detection_not_allowed"
    assert detection_rejection_reason({"local_detection_allowed": True, "skin_pixels": 0}) == "empty_skin_mask"


def test_coordinate_mapping_round_trip_fields_and_clipping() -> None:
    candidates = [{"bbox_xyxy": [20.0, 30.0, 60.0, 70.0], "center_xy": [40.0, 50.0]}]
    mapped = add_original_coordinates(
        candidates,
        crop_box_xyxy=(100, 200, 300, 400),
        square_offset_xy=(10, 20),
        resize_scale=2.0,
        original_shape_hw=(500, 500),
        input_mode="partial_face",
        detection_scope="local",
    )
    item = mapped[0]
    assert item["bbox_standardized_xyxy"] == [20.0, 30.0, 60.0, 70.0]
    assert item["bbox_original_xyxy"] == [100.0, 195.0, 120.0, 215.0]
    assert item["center_original_xy"] == [110.0, 205.0]
    assert item["input_mode"] == "partial_face"
    assert item["detection_scope"] == "local"
    drawable = candidates_in_original_space(mapped)
    assert drawable[0]["bbox_xyxy"] == item["bbox_original_xyxy"]
    assert drawable[0]["center_xy"] == item["center_original_xy"]


def test_local_detection_output_does_not_fake_global_region_counts(tmp_path) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    paths = save_detection_outputs(
        tmp_path,
        image,
        {"status": "ok", "detections": [], "count": 0},
        [],
        region_counts=None,
        region_analysis={
            "status": "skipped",
            "reason": "global_regions_not_available_for_local_input",
            "scope": "local",
        },
        artifact_policy=AcneArtifactPolicy.DEBUG_ARTIFACTS,
    )
    payload = json.loads((tmp_path / "15_filtered_detections.json").read_text(encoding="utf-8"))
    assert payload["count"] == 0
    assert payload["region_counts"] == {}
    assert payload["region_analysis"]["status"] == "skipped"
    assert set(paths) == {
        "raw_detections_json",
        "filtered_detections_json",
        "raw_boxes_image",
        "circle_image",
        "detections_csv",
    }
