from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.nine_analysis.orchestrator import NineAnalysisOrchestrator
from src.wrinkle.algorithms import wrinkle_detection_algorithm as wrinkle_algorithm


def test_local_wrinkle_main_uses_stage2_overlay_not_region_overlay(
    tmp_path: Path,
) -> None:
    stage2 = tmp_path / "08_final.jpg"
    region = tmp_path / "23_regions.jpg"
    tiles = tmp_path / "24_tiles.jpg"
    group_07 = tmp_path / "07_干燥性细纹全脸分区结果图.jpg"
    group_08 = tmp_path / "08_稳定性线性皱纹全脸分区结果图.jpg"
    group_09 = tmp_path / "09_结构性沟纹全脸分区结果图.jpg"
    for path, payload in (
        (stage2, b"stage2"),
        (region, b"region"),
        (tiles, b"tiles"),
        (group_07, b"group-07"),
        (group_08, b"group-08"),
        (group_09, b"group-09"),
    ):
        path.write_bytes(payload)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({
            "report_group_overlays": {
                "07": {"path": str(group_07)},
                "08": {"path": str(group_08)},
                "09": {"path": str(group_09)},
            },
        }),
        encoding="utf-8",
    )
    orchestrator = object.__new__(NineAnalysisOrchestrator)
    orchestrator.algorithms = ("wrinkle",)

    items = orchestrator._canonical_items({
        "wrinkle": {
            "status": "success",
            "result": {
                "output_dir": str(tmp_path),
                "results": {
                    "stage2_overlay": str(stage2),
                    "region_overlay": str(region),
                    "region_tiles": str(tiles),
                    "summary_json": str(summary),
                    "region_metrics_csv": str(tmp_path / "regions.csv"),
                },
            },
        },
    })

    assert items["wrinkle"]["主结果图"] == str(group_07)
    assert items["wrinkle"]["附加结果图"] == [str(group_08), str(group_09)]


def test_wrinkle_landmarker_retries_cropped_side_face_with_reflect_padding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[int, int]] = []

    class FakeImage:
        def __init__(self, *, image_format: object, data: np.ndarray) -> None:
            del image_format
            self.data = data

    class FakeLandmarker:
        def detect(self, image: FakeImage) -> SimpleNamespace:
            calls.append(image.data.shape[:2])
            if len(calls) == 1:
                return SimpleNamespace(face_landmarks=[])
            points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
            return SimpleNamespace(face_landmarks=[points])

    fake_mp = SimpleNamespace(
        Image=FakeImage,
        ImageFormat=SimpleNamespace(SRGB="srgb"),
    )
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"model")
    monkeypatch.setattr(
        wrinkle_algorithm,
        "try_import_mediapipe_tasks",
        lambda: (fake_mp, object(), object(), None),
    )
    monkeypatch.setattr(
        wrinkle_algorithm,
        "ensure_face_landmarker_model",
        lambda _path: (model, "ok"),
    )
    monkeypatch.setattr(
        wrinkle_algorithm,
        "_load_landmarker_once",
        lambda _path: FakeLandmarker(),
    )

    points, regions, status = wrinkle_algorithm.detect_face_landmarks(
        np.full((80, 100, 3), 128, dtype=np.uint8),
        model,
    )

    assert calls == [(80, 100), (128, 150)]
    assert points is not None and points.shape == (478, 2)
    assert regions is not None
    assert "reflected-padding fallback" in status
    np.testing.assert_allclose(points[0], np.asarray((50, 40)), atol=1)
