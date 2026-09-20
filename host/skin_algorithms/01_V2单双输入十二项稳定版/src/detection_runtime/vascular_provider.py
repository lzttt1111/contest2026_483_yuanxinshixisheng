from __future__ import annotations

"""Shared V3 vascular detection and artifact persistence for both routes."""

import csv
from dataclasses import dataclass, replace
from enum import Enum
import json
from pathlib import Path
from typing import Protocol
from typing_extensions import assert_never

import numpy as np

from src.detection_runtime.public_metrics import write_added_item_public_metrics
from src.engines.clinic_vascular_structure_engine import (
    ClinicVascularResult,
    ClinicVascularStructureAnalyzer,
)
from src.preprocess.image_preprocessor import PreprocessResultV2
from src.utils.io_utils import cv_imwrite


class VascularRoute(str, Enum):
    CLINIC_FOUR_LIGHT = "clinic_four_light"
    CONSUMER_RGB = "consumer_rgb"


class VascularDetector(Protocol):
    def detect(
        self,
        measurement: PreprocessResultV2,
        *,
        rgb: PreprocessResultV2 | None = None,
        pp: PreprocessResultV2 | None = None,
        red_display: np.ndarray | None = None,
    ) -> ClinicVascularResult: ...


@dataclass(frozen=True, slots=True)
class VascularRunRequest:
    measurement: PreprocessResultV2
    rgb: PreprocessResultV2 | None
    pp: PreprocessResultV2 | None
    red_display: np.ndarray | None
    route: VascularRoute
    output_root: Path
    input_name: str


@dataclass(frozen=True, slots=True)
class VascularArtifacts:
    overlay: Path
    raw_json: Path
    raw_csv: Path
    public_json: Path
    public_csv: Path
    medical_v2_csv: Path
    result: ClinicVascularResult


def _consumer_result(result: ClinicVascularResult) -> ClinicVascularResult:
    metrics = dict(result.metrics)
    metrics.update(
        {
            "algorithm": "VascularStructure-V3-ConsumerRGB",
            "measurement_channel": "RGB_M",
            "red_role": "RGB_DERIVED_RED_PROPOSAL_AND_DISPLAY",
            "input_semantics": "SINGLE_RGB_SAME_IMAGE_PROXY",
        }
    )
    return replace(result, metrics=metrics)


def _route_result(
    result: ClinicVascularResult,
    route: VascularRoute,
) -> ClinicVascularResult:
    match route:
        case VascularRoute.CLINIC_FOUR_LIGHT:
            return result
        case VascularRoute.CONSUMER_RGB:
            return _consumer_result(result)
        case unreachable:
            assert_never(unreachable)


def run_vascular_v3(
    request: VascularRunRequest,
    detector: VascularDetector | None = None,
) -> VascularArtifacts:
    """Run the shared V3 detector once and persist raw plus public artifacts."""
    active_detector = detector or ClinicVascularStructureAnalyzer()
    detected = active_detector.detect(
        request.measurement,
        rgb=request.rgb,
        pp=request.pp,
        red_display=request.red_display,
    )
    result = _route_result(detected, request.route)
    root = request.output_root
    root.mkdir(parents=True, exist_ok=True)
    overlay = root / "01_血管样结构检测结果图.jpg"
    cv_imwrite(str(overlay), result.overlay)

    raw = dict(result.metrics)
    raw["input_name"] = request.input_name
    raw_json = root / "血管样结构量化指标.json"
    raw_json.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    raw_csv = root / "血管样结构量化指标.csv"
    scalar_rows = [
        (name, value, "engineering_value")
        for name, value in raw.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    with raw_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("指标名称", "检测结果", "单位"))
        writer.writerows(scalar_rows)

    public_json, public_csv, medical_v2_csv = write_added_item_public_metrics(
        item_id="vascular",
        raw=raw,
        json_path=root / "血管样结构公开量化指标.json",
        csv_path=root / "血管样结构公开量化指标.csv",
        v2_path=root / "血管样结构医学量化指标_V2.csv",
    )
    return VascularArtifacts(
        overlay=overlay,
        raw_json=raw_json,
        raw_csv=raw_csv,
        public_json=public_json,
        public_csv=public_csv,
        medical_v2_csv=medical_v2_csv,
        result=result,
    )


__all__ = [
    "VascularArtifacts",
    "VascularRoute",
    "VascularRunRequest",
    "run_vascular_v3",
]
