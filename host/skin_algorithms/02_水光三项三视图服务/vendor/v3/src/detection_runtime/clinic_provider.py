from __future__ import annotations

"""Clinic-only modality provider for the unified twelve-item runtime."""

import json
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from src.detection_runtime.capture_manifest import ClinicCaptureInput
from src.detection_runtime.contracts import RuntimeRoute
from src.detection_runtime.instrument_style import (
    InstrumentStylePair,
    InstrumentStyleRedBrownBaseProvider,
)
from src.detection_runtime.instrument_style_overlay import (
    render_vendor_brown_result,
    render_vendor_red_result,
)
from src.detection_runtime.route_modality_provider import (
    PreparedRouteModality,
    RouteModalityProvider,
)
from src.detection_runtime.public_metrics import (
    write_added_item_public_metrics,
    write_uv_or_porphyrin_public_metrics,
)
from src.detection_runtime.vascular_provider import (
    VascularRoute,
    VascularRunRequest,
    run_vascular_v3,
)
from src.detection_runtime.clinic_visuals import uv_results, with_public_visia_boundary
from src.capture_profile import CaptureProfile
from src.engines.brown_engine import BrownAreaAnalyzer
from src.engines.rbx_engine import ErythemaAnalyzer
from src.engines.surface_gloss_engine import SurfaceGlossAnalyzer
from src.preprocess.image_preprocessor import ImagePreprocessor, PreprocessResultV2


def _decode(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode capture input: {path.name}")
    return image


def _write_image(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if suffix in {".jpg", ".jpeg"} else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise ValueError(f"cannot encode {path.name}")
    encoded.tofile(path)
    return path


def _json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _item(
    label: str,
    image: Path,
    metric_json: Path,
    metric_csv: Path,
    v2_csv: Path,
    *,
    additional_images: list[Path] | None = None,
) -> dict[str, Any]:
    item = {
        "项目": label,
        "状态": "success",
        "主结果图": str(image),
        "量化JSON": str(metric_json),
        "量化CSV": str(metric_csv),
        "医学V2CSV": str(v2_csv),
    }
    if additional_images:
        item["附加结果图"] = [str(path) for path in additional_images]
    return item


def _mask_metrics(mask: np.ndarray, denominator: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    support = np.asarray(mask) > 0
    valid = np.asarray(denominator) > 0
    values = np.asarray(score, dtype=np.float32)[support]
    count = max(cv2.connectedComponents(support.astype(np.uint8), 8)[0] - 1, 0)
    return {
        "instance_count": int(count),
        "area_px": int(np.count_nonzero(support)),
        "valid_area_px": int(np.count_nonzero(valid)),
        "area_ratio": round(float(np.count_nonzero(support)) / max(int(np.count_nonzero(valid)), 1), 8),
        "p50_intensity": round(float(np.percentile(values, 50)) if values.size else 0.0, 6),
        "p90_intensity": round(float(np.percentile(values, 90)) if values.size else 0.0, 6),
    }


PreparedClinicCapture = PreparedRouteModality


class ClinicFourLightProvider(RouteModalityProvider):
    """Preprocess four M channels once, then run route-specific heads."""

    route = "clinic_four_light"

    def __init__(self, preprocessor: ImagePreprocessor | None = None) -> None:
        self.preprocessor = preprocessor or ImagePreprocessor()
        super().__init__(style_provider=InstrumentStyleRedBrownBaseProvider())

    def _prepare(
        self, capture: ClinicCaptureInput
    ) -> tuple[dict[str, PreprocessResultV2], dict[str, np.ndarray], dict[str, float]]:
        prepared: dict[str, PreprocessResultV2] = {}
        decoded: dict[str, np.ndarray] = {}
        timing: dict[str, float] = {}
        for role in ("RGB_M", "PP_M", "CP_M", "365_M"):
            channel = capture.channel(role)
            started = time.perf_counter()
            decoded[role] = _decode(channel.path)
            result = self.preprocessor.preprocess_image(decoded[role])
            if result.analysis_image.shape[:2] != (1024, 1024):
                raise ValueError(f"clinic preprocess escaped 1024 space: {role}")
            prepared[role] = result
            timing[f"preprocess_{role}"] = round(time.perf_counter() - started, 4)
        return prepared, decoded, timing

    def prepare(
        self,
        capture: ClinicCaptureInput,
        output_root: Path,
    ) -> PreparedClinicCapture:
        """Prepare and style four channels without starting Clinic heads."""

        output_root.mkdir(parents=True, exist_ok=True)
        return self.prepare_clinic(
            capture,
            output_root,
            preprocessor=self.preprocessor,
        )

    def run(
        self,
        capture: ClinicCaptureInput,
        output_root: Path,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Compatibility wrapper for callers that do not overlap preparation."""

        return self.run_prepared(self.prepare(capture, output_root), output_root)

    def run_prepared(
        self,
        bundle: PreparedClinicCapture,
        output_root: Path,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Run Clinic heads after the orchestrator has joined the RGB base."""

        capture = bundle.capture
        if capture is None:
            raise ValueError("clinic heads require a clinic capture")
        timing = dict(bundle.timing_seconds)
        rgb, pp, cp, uv365 = (
            bundle.preprocessed[role]
            for role in ("RGB_M", "PP_M", "CP_M", "365_M")
        )
        styled = bundle.styled
        vascular_styled = bundle.vascular_styled
        if vascular_styled is None or vascular_styled is styled:
            raise ValueError("clinic vascular RED style view is missing or aliased")
        if vascular_styled.red is styled.red:
            raise ValueError("clinic vascular RED preprocess object must be independent")
        replacements: dict[str, dict[str, Any]] = {}

        head_started = time.perf_counter()
        red_root = output_root / "redness"
        red_overlay = ErythemaAnalyzer(
            capture_profile=CaptureProfile.INSTITUTION,
        ).process_preprocess_result(styled.red, str(red_root), f"{capture.capture_alias}_CP_M.jpg")
        if not red_overlay:
            raise RuntimeError("clinic redness did not produce an overlay")
        red_sample = Path(red_overlay).parent
        replacements["redness"] = _item(
            "红区", Path(red_overlay), red_sample / "红区量化指标.json",
            red_sample / "红区量化指标.csv", red_sample / "红区医学量化指标_V2.csv",
            additional_images=[
                styled.red_base_path,
                red_sample / "06_VISIA红色区实例图.jpg",
            ],
        )

        red_marker_mask = cv2.imread(
            str(red_sample / "07_VISIA红色区实例Mask.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if red_marker_mask is None:
            raise RuntimeError("clinic redness marker mask is missing")
        red_public = _write_image(
            red_sample / "09_vendor_RED_VISIA_result.png",
            render_vendor_red_result(
                styled.red,
                styled.red.display_image,
                red_marker_mask,
                include_boundary=True,
            ),
        )
        replacements["redness"]["主结果图"] = str(red_public)
        replacements["redness"]["附加结果图"] = [str(styled.red_base_path)]
        timing["redness"] = round(time.perf_counter() - head_started, 4)

        head_started = time.perf_counter()
        brown_root = output_root / "brown"
        brown_overlay = BrownAreaAnalyzer().process_preprocess_result(styled.brown, str(brown_root), f"{capture.capture_alias}_CP_M.jpg")
        if not brown_overlay:
            raise RuntimeError("clinic brown did not produce an overlay")
        brown_sample = Path(brown_overlay).parent
        brown_rbx = brown_sample / "01_RBX棕区结果图.jpg"
        replacements["brown"] = _item(
            "棕区", brown_rbx, brown_sample / "棕色斑量化指标.json",
            brown_sample / "棕色斑量化指标.csv", brown_sample / "棕色斑医学量化指标_V2.csv",
            additional_images=[styled.brown_base_path, Path(brown_overlay)],
        )

        brown_instance_mask = cv2.imread(
            str(brown_sample / "03_棕色斑实例Mask.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if brown_instance_mask is None:
            raise RuntimeError("clinic brown instance mask is missing")
        brown_public = _write_image(
            brown_sample / "09_vendor_BROWN_VISIA_result.png",
            render_vendor_brown_result(
                styled.brown,
                styled.brown.display_image,
                brown_instance_mask,
                include_boundary=True,
            ),
        )
        replacements["brown"]["主结果图"] = str(brown_public)
        replacements["brown"]["附加结果图"] = [str(styled.brown_base_path)]
        timing["brown"] = round(time.perf_counter() - head_started, 4)

        head_started = time.perf_counter()
        _, uv_result, fluorescence = uv_results(uv365)
        uv_root = output_root / "uv_spots"
        from src.doctor_v3.evidence import save, save_preprocess
        from src.engines.uv_spots_engine import MEASUREMENT_SUPPORT_DEFINITION
        save_preprocess(output_root, rgb)
        save(uv_root, "uv_spots",
             {"valid": uv365.skin_mask, "landmarks": uv365.landmarks,
              "instances": uv_result.instance_mask, "score": uv_result.score_map,
              "continuous": uv_result.measurement_support_mask},
             {"source_kind": "real_uv", "quality_status": uv365.quality_status,
              "continuous_definition": MEASUREMENT_SUPPORT_DEFINITION,
              "continuous_source": "UVSpotsAnalyzer.accepted_components_before_display",
              "continuous_exclusion": "instance_and_porphyrin",
              "pixel_source_role": "365_M"})
        uv_metrics = {
            **_mask_metrics(uv_result.instance_mask, uv365.skin_mask, uv_result.score_map),
            "algorithm_version": uv_result.algorithm_version,
            "input_semantics": uv_result.input_semantics,
            "regions": uv_result.region_distribution,
        }
        uv_base = _write_image(uv_root / "00_真实365_UV底图.jpg", uv365.display_image)
        uv_image = _write_image(uv_root / "01_真实365_UV色斑检测结果图.jpg", uv_result.overlay)
        _write_image(
            uv_root / "_debug/03_UV色斑Mask.png",
            uv_result.instance_mask,
        )
        uv_json, uv_csv, uv_v2 = write_uv_or_porphyrin_public_metrics(
            item_id="uv_spots",
            raw=uv_metrics,
            json_path=uv_root / "UV色斑量化指标.json",
            csv_path=uv_root / "UV色斑量化指标.csv",
            v2_path=uv_root / "UV色斑医学量化指标_V2.csv",
            route=RuntimeRoute.CLINIC_FOUR_LIGHT,
        )
        replacements["uv_spots"] = _item(
            "UV色斑", uv_image, uv_json, uv_csv, uv_v2,
            additional_images=[uv_base],
        )

        por_root = output_root / "porphyrin"
        save(por_root, "porphyrin",
             {"valid": fluorescence.dedicated_analysis_mask, "landmarks": uv365.landmarks,
              "instances": fluorescence.formal_instance_mask, "score": fluorescence.score_map},
             {"source_kind": "real_uv", "quality_status": uv365.quality_status})
        por_metrics = {
            **_mask_metrics(
                fluorescence.formal_instance_mask,
                fluorescence.dedicated_analysis_mask,
                fluorescence.score_map,
            ),
            "algorithm_version": "FluorescencePorphyrin-Real365-HighRecall-V1",
            "input_semantics": "REAL_365_M",
            "audit": fluorescence.audit_metrics,
            "regions": fluorescence.region_distribution,
        }
        por_base = _write_image(por_root / "00_真实365荧光底图.jpg", uv365.display_image)
        por_image = _write_image(por_root / "01_真实365_卟啉荧光检测结果图.jpg", fluorescence.formal_overlay)
        por_json, por_csv, por_v2 = write_uv_or_porphyrin_public_metrics(
            item_id="porphyrin",
            raw=por_metrics,
            route=RuntimeRoute.CLINIC_FOUR_LIGHT,
            json_path=por_root / "卟啉量化指标.json",
            csv_path=por_root / "卟啉量化指标.csv",
            v2_path=por_root / "卟啉医学量化指标_V2.csv",
        )
        replacements["porphyrin"] = _item(
            "卟啉", por_image, por_json, por_csv, por_v2,
            additional_images=[por_base],
        )
        timing["uv_porphyrin"] = round(time.perf_counter() - head_started, 4)

        head_started = time.perf_counter()
        gloss_root = output_root / "surface_gloss"
        gloss_result = SurfaceGlossAnalyzer().detect_surface_gloss(pp)
        gloss_result.overlay = with_public_visia_boundary(pp, gloss_result.overlay)
        gloss_paths = SurfaceGlossAnalyzer.save_result(gloss_result, str(gloss_root), capture.capture_alias)
        gloss_sample = Path(gloss_paths["overlay"]).parent
        save(gloss_sample, "surface_gloss",
             {"valid": gloss_result.gloss_analysis_mask, "landmarks": pp.landmarks,
              "instances": gloss_result.gloss_mask, "high": gloss_result.high_gloss_mask,
              "score": gloss_result.gloss_intensity_map},
             {"quality_status": pp.quality_status, "source_kind": "pp_measured"})
        gloss_json, gloss_csv, gloss_v2 = write_added_item_public_metrics(
            item_id="surface_gloss",
            raw=gloss_result.metrics(),
            json_path=gloss_sample / "表面油光公开量化指标.json",
            csv_path=gloss_sample / "表面油光公开量化指标.csv",
            v2_path=gloss_sample / "表面油光医学量化指标_V2.csv",
        )
        replacements["surface_gloss"] = _item(
            "油光", Path(gloss_paths["overlay"]), gloss_json, gloss_csv, gloss_v2,
        )
        timing["surface_gloss"] = round(time.perf_counter() - head_started, 4)

        head_started = time.perf_counter()
        vascular_root = output_root / "vascular"
        vascular = run_vascular_v3(
            VascularRunRequest(
                measurement=cp,
                rgb=rgb,
                pp=pp,
                red_display=vascular_styled.red.analysis_image,
                route=VascularRoute.CLINIC_FOUR_LIGHT,
                output_root=vascular_root,
                input_name=capture.capture_alias,
            )
        )
        replacements["vascular"] = _item(
            "血管样结构",
            vascular.overlay,
            vascular.public_json,
            vascular.public_csv,
            vascular.medical_v2_csv,
        )
        timing["vascular"] = round(time.perf_counter() - head_started, 4)

        timing["clinic_modality_total"] = round(
            time.perf_counter() - bundle.started_at,
            4,
        )
        provenance = {
            "route": self.route,
            "capture_alias": capture.capture_alias,
            "red_brown_source_role": styled.source_role,
            "red_brown_renderer_version": styled.renderer_version,
            "red_brown_provider": styled.provenance,
            "vascular_red_source_role": vascular_styled.source_role,
            "vascular_red_provider": vascular_styled.provenance,
            "input_sha256": {channel.role: channel.sha256 for channel in capture.channels},
            "registration": {
                channel.role: {
                    "status": channel.registration_status,
                    "transform_sha256": channel.registration_transform_sha256,
                    "cross_channel_authorized": channel.cross_channel_authorized,
                }
                for channel in capture.channels
            },
            "timing_seconds": timing,
        }
        _json(output_root / "clinic_modality_receipt.json", provenance)
        return replacements, provenance

    def close(self) -> None:
        """Close the preprocessor owned by this provider."""

        self.preprocessor.close()


__all__ = ["ClinicFourLightProvider", "PreparedClinicCapture"]
