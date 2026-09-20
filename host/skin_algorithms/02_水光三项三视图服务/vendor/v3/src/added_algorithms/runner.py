from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from src.added_algorithms.public_metrics import (
    PublicMetricFiles,
    write_contour_metrics,
    write_gloss_metrics,
    write_vascular_metrics,
)
from src.engines.clinic_vascular_structure_engine import ClinicVascularStructureAnalyzer
from src.engines.relative_face_geometry import analyze_relative_face_geometry
from src.engines.surface_gloss_engine import SurfaceGlossAnalyzer
from src.engines.vascular_structure_engine import (
    VascularResult,
    VascularStructureAnalyzer,
)
from src.engines.visia_regions import build_visia_regions, draw_region_boundaries
from src.preprocess.image_preprocessor import PreprocessResultV2
from src.utils.io_utils import cv_imwrite


@dataclass(frozen=True, slots=True)
class AddedAlgorithmArtifacts:
    overlay: Path
    metrics: PublicMetricFiles


def _sample_root(output_dir: str | Path, source_name: str) -> Path:
    root = Path(output_dir) / Path(source_name).stem
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_surface_gloss(
    preprocess: PreprocessResultV2,
    output_dir: str | Path,
    source_name: str,
) -> AddedAlgorithmArtifacts:
    result = SurfaceGlossAnalyzer().detect_surface_gloss(preprocess)
    regions = build_visia_regions(
        preprocess.analysis_image,
        preprocess.skin_mask,
        preprocess.landmarks,
        preprocess.quality_flags,
        include_chin=True,
        mode="full",
        feature_margin_px=10,
    )
    overlay = draw_region_boundaries(
        result.overlay,
        regions.display_regions,
        partial_face=regions.partial_face,
        landmarks=np.asarray(preprocess.landmarks),
        closed_contour=regions.display_contour,
        separator_contour=regions.display_separator,
    )
    root = _sample_root(output_dir, source_name)
    overlay_path = root / "01_表面油光检测结果图.jpg"
    cv_imwrite(str(overlay_path), overlay)
    from src.doctor_v3.evidence import enabled, save
    if enabled():
        save(root,"surface_gloss",
             {"valid":result.gloss_analysis_mask,"landmarks":preprocess.landmarks,
              "instances":result.gloss_mask,"high":result.high_gloss_mask,"score":result.gloss_intensity_map},
             {"quality_status":preprocess.quality_status,"source_kind":"rgb_measured"})
    return AddedAlgorithmArtifacts(overlay_path, write_gloss_metrics(result, root))


def run_vascular(
    preprocess: PreprocessResultV2,
    red_display: np.ndarray,
    output_dir: str | Path,
    source_name: str,
) -> AddedAlgorithmArtifacts:
    from src.doctor_v3.evidence import enabled, save
    detector = ClinicVascularStructureAnalyzer()
    if enabled():
        detector.include_forehead = True
    detected = detector.detect(
        preprocess,
        rgb=preprocess,
        pp=preprocess,
        red_display=red_display,
    )
    metrics = dict(detected.metrics)
    metrics.update({
        "algorithm": "VascularStructure-V3-ConsumerRGB",
        "measurement_channel": "RGB_M",
        "red_role": "RGB_DERIVED_RED_PROPOSAL_AND_DISPLAY",
        "input_semantics": "SINGLE_RGB_SAME_IMAGE_PROXY",
    })
    result = replace(detected, metrics=metrics)
    root = _sample_root(output_dir, source_name)
    overlay_path = root / "01_血管样结构检测结果图.jpg"
    cv_imwrite(str(overlay_path), result.overlay)
    if enabled():
        save(root,"vascular",
             {"valid":result.valid_mask,"landmarks":preprocess.landmarks,
              "instances":result.vascular_mask,"skeleton":result.skeleton_mask,"score":result.hemoglobin_response},
             {"instances":list(result.instances),"quality_status":preprocess.quality_status,
              "source_kind":"rgb_proxy","forehead_enabled":True})
    return AddedAlgorithmArtifacts(overlay_path, write_vascular_metrics(result.metrics, root))


def _render_consumer_vascular_overlay(
    preprocess: PreprocessResultV2,
    detected: VascularResult,
) -> np.ndarray:
    """Render consumer vascular evidence with the shared VISIA boundary."""
    overlay = np.asarray(preprocess.display_image).copy()
    vascular = np.asarray(detected.vascular_mask) > 0
    skeleton = np.asarray(detected.skeleton_mask) > 0
    if np.any(vascular):
        overlay[vascular] = np.clip(
            0.45 * overlay[vascular].astype(np.float32)
            + 0.55 * np.asarray((0, 0, 255), dtype=np.float32),
            0,
            255,
        ).astype(np.uint8)
    overlay[skeleton] = (0, 210, 255)
    regions = build_visia_regions(
        preprocess.analysis_image,
        preprocess.skin_mask,
        preprocess.landmarks,
        preprocess.quality_flags,
        include_chin=True,
        mode="full",
        feature_margin_px=10,
    )
    return draw_region_boundaries(
        overlay,
        regions.display_regions,
        partial_face=regions.partial_face,
        landmarks=np.asarray(preprocess.landmarks),
        closed_contour=regions.display_contour,
        separator_contour=regions.display_separator,
    )


def run_consumer_vascular(
    preprocess: PreprocessResultV2,
    output_dir: str | Path,
    source_name: str,
) -> AddedAlgorithmArtifacts:
    detected = VascularStructureAnalyzer().detect_from_preprocess_result(preprocess)
    raw = asdict(detected.metrics)
    raw.update({
        "algorithm": "VascularStructure-V2.1-ConsumerRGB",
        "input_semantics": "SINGLE_CONSUMER_RGB",
        "valid_face_pixels": int(detected.metrics.valid_skin_pixels),
        "qc_passed": bool(detected.metrics.self_check.get("passed")),
    })
    root = _sample_root(output_dir, source_name)
    overlay_path = root / "01_血管样结构检测结果图.jpg"
    cv_imwrite(
        str(overlay_path),
        _render_consumer_vascular_overlay(preprocess, detected),
    )
    return AddedAlgorithmArtifacts(
        overlay_path,
        write_vascular_metrics(raw, root),
    )


def run_contour_firmness(
    preprocess: PreprocessResultV2,
    output_dir: str | Path,
    source_name: str,
) -> AddedAlgorithmArtifacts | None:
    result = analyze_relative_face_geometry(preprocess)
    if result is None:
        return None
    root = _sample_root(output_dir, source_name)
    overlay_path = root / "01_面部轮廓几何测量结果图.jpg"
    cv_imwrite(str(overlay_path), result.overlay)
    files = write_contour_metrics(
        result.metrics,
        int(cv2.countNonZero(result.face_scope)),
        root,
    )
    return AddedAlgorithmArtifacts(overlay_path, files)


__all__ = [
    "AddedAlgorithmArtifacts",
    "run_contour_firmness",
    "run_consumer_vascular",
    "run_surface_gloss",
    "run_vascular",
]
