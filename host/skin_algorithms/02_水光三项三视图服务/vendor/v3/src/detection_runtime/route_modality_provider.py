from __future__ import annotations

"""Shared preparation and consumer heads for route-sensitive detections."""

import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.detection_runtime.contracts import RuntimeRoute
from src.detection_runtime.capture_manifest import ClinicCaptureInput
from src.detection_runtime.instrument_style import (
    InstrumentStylePair,
    InstrumentStyleRedBrownBaseProvider,
)
from src.detection_runtime.instrument_style_overlay import (
    render_vendor_brown_result,
    render_vendor_red_result,
)
from src.detection_runtime.route_plan import CLINIC_ITEM_MODALITY_CONTRACT
from src.detection_runtime.vascular_provider import (
    VascularRoute,
    VascularRunRequest,
    run_vascular_v3,
)
from src.detection_runtime.clinic_visuals import with_public_visia_boundary
from src.engines.brown_engine import BrownAreaAnalyzer
from src.engines.clinic_vascular_structure_engine import ClinicVascularStructureAnalyzer
from src.engines.purple_analysis_engine import PurpleAnalysisEngine
from src.engines.rbx_engine import ErythemaAnalyzer
from src.engines.surface_gloss_engine import SurfaceGlossAnalyzer
from src.preprocess.image_preprocessor import ImagePreprocessor, PreprocessResultV2
from src.utils.io_utils import cv_imwrite


@dataclass(frozen=True, slots=True)
class RouteOutputRoots:
    redness: Path
    brown: Path
    purple: Path
    surface_gloss: Path
    vascular: Path


@dataclass(frozen=True, slots=True)
class ConsumerRouteAnalyzers:
    redness: ErythemaAnalyzer
    brown: BrownAreaAnalyzer
    purple: PurpleAnalysisEngine
    surface_gloss: SurfaceGlossAnalyzer
    vascular: ClinicVascularStructureAnalyzer


@dataclass(frozen=True, slots=True)
class PreparedRouteModality:
    route: RuntimeRoute
    input_name: str
    capture: ClinicCaptureInput | None
    preprocessed: dict[str, PreprocessResultV2]
    decoded: dict[str, np.ndarray]
    styled: InstrumentStylePair
    vascular_styled: InstrumentStylePair | None
    timing_seconds: dict[str, float]
    started_at: float


@dataclass(frozen=True, slots=True)
class ConsumerRouteArtifacts:
    results: dict[str, object]
    medical_v2_results: dict[str, dict[str, str]]
    cleanup_paths: tuple[str, ...]
    timing_seconds: dict[str, float]


class RouteModalityProvider:
    """Prepare either route without storing route choice on the provider."""

    def __init__(
        self,
        *,
        style_provider: InstrumentStyleRedBrownBaseProvider | None = None,
    ) -> None:
        self.style_provider = style_provider or InstrumentStyleRedBrownBaseProvider()

    def prepare_consumer(
        self,
        preprocessed: PreprocessResultV2,
        *,
        source_image: np.ndarray,
        output_root: Path,
        input_name: str,
    ) -> PreparedRouteModality:
        """Build RGB-derived RED/BROWN while base heads read the same preprocess."""

        started = time.perf_counter()
        style_started = time.perf_counter()
        styled = self.style_provider.build(
            preprocessed,
            source_image=source_image,
            output_root=output_root,
            source_role="RGB_M",
        )
        return PreparedRouteModality(
            route=RuntimeRoute.CONSUMER_RGB,
            input_name=input_name,
            capture=None,
            preprocessed={"RGB_M": preprocessed},
            decoded={"RGB_M": source_image},
            styled=styled,
            vascular_styled=None,
            timing_seconds={
                "route_prepare": 0.0,
                "route_style": round(time.perf_counter() - style_started, 4),
            },
            started_at=started,
        )

    def prepare_clinic(
        self,
        capture,
        output_root: Path,
        *,
        preprocessor: ImagePreprocessor,
    ) -> PreparedRouteModality:
        """Build CP-styled RED/BROWN pixels on registered RGB geometry."""

        started = time.perf_counter()
        required_roles = {
            role
            for item in CLINIC_ITEM_MODALITY_CONTRACT
            for role in item.registration_required_roles
        }
        for role in sorted(required_roles):
            channel = capture.channel(role)
            authorizations = channel.cross_channel_authorized
            if channel.registration_status != "PASSED" or not all(
                authorizations.get(name, False)
                for name in ("co_location", "fusion", "overlap")
            ):
                raise ValueError(
                    f"clinic {role} registration does not authorize the frozen modality contract"
                )
        prepared: dict[str, PreprocessResultV2] = {}
        timing: dict[str, float] = {}
        decoded: dict[str, np.ndarray] = {}
        prepare_started = time.perf_counter()
        for role in ("RGB_M", "PP_M", "CP_M", "365_M"):
            channel = capture.channel(role)
            role_started = time.perf_counter()
            image = cv2.imdecode(
                np.fromfile(channel.path, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                raise ValueError(f"cannot decode capture input: {channel.path.name}")
            decoded[role] = image
            result = preprocessor.preprocess_image(image)
            if result.analysis_image.shape[:2] != (1024, 1024):
                raise ValueError(f"clinic preprocess escaped 1024 space: {role}")
            prepared[role] = result
            timing[f"preprocess_{role}"] = round(
                time.perf_counter() - role_started,
                4,
            )
        timing["clinic_prepare"] = round(
            time.perf_counter() - prepare_started,
            4,
        )
        style_started = time.perf_counter()
        views = self.style_provider.build_clinic_views(
            red_brown_anchor=prepared["RGB_M"],
            vascular_anchor=prepared["CP_M"],
            source_image=decoded["CP_M"],
            output_root=output_root / "vendor_skin_generate",
            source_role="CP_M",
        )
        timing["clinic_style"] = round(time.perf_counter() - style_started, 4)
        return PreparedRouteModality(
            route=RuntimeRoute.CLINIC_FOUR_LIGHT,
            input_name=capture.capture_alias,
            capture=capture,
            preprocessed=prepared,
            decoded=decoded,
            styled=views.red_brown,
            vascular_styled=views.vascular,
            timing_seconds=timing,
            started_at=started,
        )

    def run_consumer_prepared(
        self,
        bundle: PreparedRouteModality,
        roots: RouteOutputRoots,
        analyzers: ConsumerRouteAnalyzers,
    ) -> ConsumerRouteArtifacts:
        """Run the six consumer route heads after the retained base heads finish."""

        if bundle.route is not RuntimeRoute.CONSUMER_RGB:
            raise ValueError("consumer route heads require consumer_rgb preparation")
        preprocessed = bundle.preprocessed["RGB_M"]
        filename = bundle.input_name
        base_name = Path(filename).stem
        styled = bundle.styled
        timing = dict(bundle.timing_seconds)
        results: dict[str, object] = {
            "redness_base": str(styled.red_base_path),
            "brown_base": str(styled.brown_base_path),
            "redne_base": str(styled.redne_base_path),
            "red_brown_provider_receipt": styled.provenance,
        }
        medical: dict[str, dict[str, str]] = {}
        cleanup = [str(styled.red_base_path.parent.parent)]

        head_started = time.perf_counter()
        red_overlay = analyzers.redness.process_preprocess_result(
            styled.red,
            str(roots.redness),
            filename,
        )
        if not red_overlay:
            raise RuntimeError("consumer redness did not produce an overlay")
        red_sample = Path(red_overlay).parent
        red_mask = cv2.imread(
            str(red_sample / "07_VISIA红色区实例Mask.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if red_mask is None:
            raise RuntimeError("consumer redness marker mask is missing")
        red_public = red_sample / "09_vendor_RED_VISIA_result.png"
        red_instances = red_sample / "10_vendor_RED_feature_instances.png"
        cv_imwrite(
            str(red_public),
            render_vendor_red_result(
                styled.red,
                styled.red.display_image,
                red_mask,
                include_boundary=True,
            ),
        )
        cv_imwrite(
            str(red_instances),
            render_vendor_red_result(
                styled.red,
                styled.red.display_image,
                red_mask,
                include_boundary=False,
            ),
        )
        results.update(
            {
                "redness": str(red_public),
                "red_areas_overlay": str(red_instances),
                "redness_report": str(red_sample / "红区量化指标.csv"),
                "redness_metrics": str(red_sample / "红区量化指标.json"),
            }
        )
        red_v2 = red_sample / "红区医学量化指标_V2.csv"
        if red_v2.is_file():
            medical["redness"] = {
                "json": str(red_sample / "红区量化指标.json"),
                "csv": str(red_v2),
            }
        cleanup.append(str(red_sample))
        timing["redness"] = round(time.perf_counter() - head_started, 4)

        head_started = time.perf_counter()
        brown_overlay = analyzers.brown.process_preprocess_result(
            styled.brown,
            str(roots.brown),
            filename,
        )
        if not brown_overlay:
            raise RuntimeError("consumer brown did not produce an overlay")
        brown_sample = Path(brown_overlay).parent
        brown_mask = cv2.imread(
            str(brown_sample / "03_棕色斑实例Mask.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if brown_mask is None:
            raise RuntimeError("consumer brown instance mask is missing")
        brown_public = brown_sample / "09_vendor_BROWN_VISIA_result.png"
        brown_instances = brown_sample / "10_vendor_BROWN_feature_instances.png"
        cv_imwrite(
            str(brown_public),
            render_vendor_brown_result(
                styled.brown,
                styled.brown.display_image,
                brown_mask,
                include_boundary=True,
            ),
        )
        cv_imwrite(
            str(brown_instances),
            render_vendor_brown_result(
                styled.brown,
                styled.brown.display_image,
                brown_mask,
                include_boundary=False,
            ),
        )
        results.update(
            {
                "brown": str(brown_public),
                "brown_spots_overlay": str(brown_instances),
                "brown_report": str(brown_sample / "棕色斑量化指标.csv"),
                "brown_metrics": str(brown_sample / "棕色斑量化指标.json"),
            }
        )
        brown_v2 = brown_sample / "棕色斑医学量化指标_V2.csv"
        if brown_v2.is_file():
            medical["brown"] = {
                "json": str(brown_sample / "棕色斑量化指标.json"),
                "csv": str(brown_v2),
            }
        cleanup.append(str(brown_sample))
        timing["brown"] = round(time.perf_counter() - head_started, 4)

        head_started = time.perf_counter()
        purple_primary = analyzers.purple.process_preprocess_result(
            preprocessed,
            str(roots.purple),
            filename,
        )
        if not purple_primary:
            raise RuntimeError("consumer purple did not produce an overlay")
        purple_paths = analyzers.purple.last_paths
        purple = {
            "purple_uv_base": purple_paths.get("uv_like_base"),
            "purple_uv_spots_overlay": purple_paths.get("uv_spots_overlay"),
            "purple_fluorescence_base": purple_paths.get("porphyrin_base"),
            "purple_porphyrin_overlay": purple_paths.get("porphyrin_overlay"),
            "purple_report": purple_paths.get("metrics_csv"),
            "purple_metrics": purple_paths.get("metrics_json"),
        }
        missing = [path for path in purple.values() if not path or not os.path.isfile(path)]
        if missing:
            raise RuntimeError(f"consumer purple outputs missing: {missing}")
        results.update(purple)
        cleanup.append(str(Path(purple_primary).parent))
        timing["purple"] = round(time.perf_counter() - head_started, 4)

        head_started = time.perf_counter()
        gloss = analyzers.surface_gloss.detect_surface_gloss(preprocessed)
        gloss.overlay = with_public_visia_boundary(preprocessed, gloss.overlay)
        gloss_paths = analyzers.surface_gloss.save_result(
            gloss,
            str(roots.surface_gloss),
            base_name,
        )
        results.update(
            {
                "surface_gloss": gloss_paths["overlay"],
                "surface_gloss_metrics": gloss_paths["json"],
                "surface_gloss_report": gloss_paths["csv"],
            }
        )
        cleanup.append(str(Path(gloss_paths["overlay"]).parent))
        timing["surface_gloss"] = round(time.perf_counter() - head_started, 4)

        head_started = time.perf_counter()
        vascular_root = roots.vascular / base_name
        vascular = run_vascular_v3(
            VascularRunRequest(
                measurement=preprocessed,
                rgb=preprocessed,
                pp=preprocessed,
                red_display=styled.red.analysis_image,
                route=VascularRoute.CONSUMER_RGB,
                output_root=vascular_root,
                input_name=filename,
            ),
            analyzers.vascular,
        )
        results.update(
            {
                "vascular": str(vascular.overlay),
                "vascular_metrics": str(vascular.raw_json),
                "vascular_report": str(vascular.raw_csv),
            }
        )
        medical["vascular"] = {
            "json": str(vascular.public_json),
            "csv": str(vascular.medical_v2_csv),
        }
        cleanup.append(str(vascular_root))
        timing["vascular"] = round(time.perf_counter() - head_started, 4)
        timing["route_modality_total"] = round(
            time.perf_counter() - bundle.started_at,
            4,
        )
        return ConsumerRouteArtifacts(
            results=results,
            medical_v2_results=medical,
            cleanup_paths=tuple(cleanup),
            timing_seconds=timing,
        )


__all__ = [
    "ConsumerRouteAnalyzers",
    "ConsumerRouteArtifacts",
    "PreparedRouteModality",
    "RouteModalityProvider",
    "RouteOutputRoots",
]
