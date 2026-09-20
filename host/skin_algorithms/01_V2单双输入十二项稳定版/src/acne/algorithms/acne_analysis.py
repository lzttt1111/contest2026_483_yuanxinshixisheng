from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.acne.acne_candidate_generator import AcneCandidateGenerator, combine_candidates, save_candidate_outputs
from src.acne.acne_detector import AcneDetector, DetectorUnavailableError, validate_detector_runtime
from src.acne.artifact_policy import AcneArtifactPolicy, AcnePhaseTimings
from src.acne.detector_postprocess import (
    add_original_coordinates,
    candidates_in_original_space,
    count_by_region,
    draw_circles,
    filter_detections,
    save_detection_outputs,
)
from src.acne.face_preprocess import (
    NoFaceDetectedError,
    PreprocessConfig,
    PreprocessError,
    load_mediapipe_models,
    preprocess_image,
    save_preprocess_outputs,
)
from src.acne.face_regions import build_region_masks
from src.acne.grading_adapter import (
    run_acne_lds_grading,
    skipped_grading,
    warmup_acne_lds,
)
from src.acne.image_io import read_image_bgr, write_image
from src.acne.quality_control import assess_image_quality


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DetectorProfile:
    name: str
    weights: Path
    imgsz: int
    conf: float
    device: str
    sha256: str | None = None
    description: str = ""


def load_detector_profile(profile_name: str, profiles_path: str | Path, device_override: str | None = None) -> DetectorProfile:
    path = Path(profiles_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles", {})
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles)) or "<none>"
        raise ValueError(f"Unknown detector profile '{profile_name}'. Available profiles: {available}")
    raw = profiles[profile_name]
    weights = Path(raw["weights"])
    if not weights.is_absolute():
        weights = PROJECT_ROOT / weights
    return DetectorProfile(
        name=profile_name,
        weights=weights,
        imgsz=int(raw["imgsz"]),
        conf=float(raw["conf"]),
        device=str(device_override if device_override is not None else raw.get("device", "cpu")),
        sha256=str(raw["sha256"]) if raw.get("sha256") else None,
        description=str(raw.get("description", "")),
    )


def grading_rejection_reason(metadata: dict[str, Any]) -> str:
    if metadata.get("input_mode") != "full_face":
        return "input_not_full_face"
    warnings = metadata.get("quality", {}).get("warnings", [])
    severe = {
        "possible_blur",
        "possible_underexposure",
        "possible_overexposure",
        "large_overexposed_area",
        "large_underexposed_area",
        "resolution_below_recommended",
    }
    matched = [warning for warning in warnings if warning in severe]
    if matched:
        return ",".join(matched)
    landmark_quality = metadata.get("landmark_quality", {})
    if not landmark_quality.get("is_reliable", False):
        return "landmark_quality_not_reliable"
    if metadata.get("skin_pixels", 0) <= 0:
        return "empty_skin_mask"
    return ""


def detection_rejection_reason(metadata: dict[str, Any]) -> str:
    if not metadata.get("local_detection_allowed", False):
        return "local_detection_not_allowed"
    if metadata.get("skin_pixels", 0) <= 0:
        return "empty_skin_mask"
    return ""


class AcneAnalysisRunner:
    """供本地 CLI 和 Celery Worker 复用的痤疮分析主流程。"""

    def __init__(
        self,
        profiles_path: str | Path,
        profile_name: str = "v3_balanced",
        device: str | None = None,
        enable_grading: bool = True,
        grading_checkpoint: str | Path | None = None,
        grading_python: str | Path | None = None,
        grading_timeout: int = 300,
    ) -> None:
        self.profile = load_detector_profile(profile_name, profiles_path, device_override=device)
        self.detector_runtime = validate_detector_runtime(
            self.profile.weights,
            device=self.profile.device,
            expected_sha256=self.profile.sha256,
        )
        self.enable_grading = enable_grading
        self.grading_checkpoint = Path(grading_checkpoint) if grading_checkpoint else None
        self.grading_python = Path(grading_python) if grading_python else PROJECT_ROOT / ".venv" / "bin" / "python"
        self.grading_timeout = grading_timeout
        self._detector: AcneDetector | None = None
        self._preprocess_config = PreprocessConfig()
        self._segmenter = None
        self._landmarker = None
        self._mp_module = None

    def _get_detector(self) -> AcneDetector:
        if self._detector is None:
            self._detector = AcneDetector(
                self.profile.weights,
                conf=self.profile.conf,
                imgsz=self.profile.imgsz,
                device=self.profile.device,
            )
        return self._detector

    def _get_mediapipe_models(self):
        if self._segmenter is None or self._landmarker is None:
            (
                self._segmenter,
                self._landmarker,
                self._mp_module,
            ) = load_mediapipe_models(self._preprocess_config)
        return self._segmenter, self._landmarker, self._mp_module

    def warmup(self) -> dict[str, object]:
        """加载并预热 YOLO、Acne-LDS 与 MediaPipe；每个子进程一次。"""
        self._get_mediapipe_models()
        detector = self._get_detector()
        # Ultralytics 首次 predict 才会把模型完整迁移到 CUDA。
        import numpy as np

        detector.predict(np.zeros((self.profile.imgsz, self.profile.imgsz, 3), dtype=np.uint8))
        grading = None
        if self.enable_grading:
            if self.grading_checkpoint is None:
                raise FileNotFoundError("grading checkpoint is not configured")
            grading = warmup_acne_lds(
                self.grading_checkpoint,
                self.profile.device,
            )
        return {
            "detector": self.detector_runtime,
            "grading": grading,
            "mediapipe_cached": True,
        }

    def close(self) -> None:
        for model in (self._segmenter, self._landmarker):
            close = getattr(model, "close", None)
            if callable(close):
                close()
        self._segmenter = None
        self._landmarker = None
        self._mp_module = None

    def run(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        artifact_policy: AcneArtifactPolicy = AcneArtifactPolicy.FORMAL,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        timings = AcnePhaseTimings()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config = self._preprocess_config

        def write_failure_context(reason: str) -> dict[str, Any]:
            context: dict[str, Any] = {"saved": []}
            try:
                original = read_image_bgr(image_path)
                write_image(output_dir / "01_original.jpg", original)
                quality = assess_image_quality(original)
                quality["pipeline_failure_reason"] = reason
                (output_dir / "00_input_quality.json").write_text(
                    json.dumps(quality, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                context["saved"] = ["00_input_quality.json", "01_original.jpg"]
            except Exception as exc:
                context["context_error"] = str(exc)
            return context

        try:
            phase_start = time.perf_counter()
            segmenter, landmarker, mp_module = self._get_mediapipe_models()
            result = preprocess_image(image_path, config, segmenter, landmarker, mp_module)
            timings.preprocess = time.perf_counter() - phase_start
            preprocess_paths = save_preprocess_outputs(
                result,
                output_dir,
                artifact_policy=artifact_policy,
                timings=timings,
            )

            phase_start = time.perf_counter()
            grading_reason = grading_rejection_reason(result.metadata)
            grading_allowed = result.metadata["input_mode"] == "full_face" and grading_reason == ""
            grading_output_path = output_dir / "13_grading_result.json"
            if not self.enable_grading:
                grading = skipped_grading("disabled_by_worker_config")
                grading_output_path.write_text(json.dumps(grading, ensure_ascii=False, indent=2), encoding="utf-8")
            elif not grading_allowed:
                grading = skipped_grading(grading_reason or "input_not_eligible_for_grading")
                grading_output_path.write_text(json.dumps(grading, ensure_ascii=False, indent=2), encoding="utf-8")
            elif self.grading_checkpoint and self.grading_checkpoint.exists():
                grading = run_acne_lds_grading(
                    image_path=output_dir / "02_standardized.jpg",
                    output_json=grading_output_path,
                    checkpoint_path=self.grading_checkpoint,
                    python_path=self.grading_python,
                    device=self.profile.device,
                    timeout_seconds=self.grading_timeout,
                )
            else:
                grading = skipped_grading("checkpoint_not_configured")
                grading_output_path.write_text(json.dumps(grading, ensure_ascii=False, indent=2), encoding="utf-8")
            timings.lds = time.perf_counter() - phase_start

            detection_paths: dict[str, str] = {}
            candidate_paths: dict[str, str] = {}
            max_recall: dict[str, Any]
            input_mode = str(result.metadata["input_mode"])
            detection_scope = "global" if input_mode == "full_face" else "local"
            detection_reason = detection_rejection_reason(result.metadata)
            if detection_reason:
                detection = {"status": "skipped", "reason": detection_reason, "detections": [], "count": 0}
                max_recall = {
                    "status": "skipped",
                    "reason": detection_reason,
                    "yolo_count": 0,
                    "unsupervised_focal_count": 0,
                    "diffuse_erythema_region_count": 0,
                    "combined_count": 0,
                }
            else:
                filtered: list[dict[str, Any]] = []
                region_counts: dict[str, int] | None = None
                region_analysis: dict[str, Any]
                if result.metadata.get("global_region_analysis_allowed") and len(result.landmarks_xy) > 0:
                    region_analysis = {"status": "ok", "scope": "global"}
                else:
                    region_analysis = {
                        "status": "skipped",
                        "reason": "global_regions_not_available_for_local_input",
                        "scope": "local",
                    }
                try:
                    phase_start = time.perf_counter()
                    raw_detection = self._get_detector().predict(result.standardized_bgr)
                    filtered = filter_detections(
                        raw_detection.get("detections", []),
                        result.skin_mask,
                        result.forbidden_mask,
                        result.standardized_bgr.shape[:2],
                    )
                    if result.metadata.get("global_region_analysis_allowed") and len(result.landmarks_xy) > 0:
                        region_counts = count_by_region(
                            filtered,
                            build_region_masks(result.landmarks_xy, result.standardized_bgr.shape[0]),
                        )
                    filtered = add_original_coordinates(
                        filtered,
                        result.crop_box_xyxy,
                        result.square_offset_xy,
                        result.resize_scale,
                        result.original_bgr.shape[:2],
                        input_mode,
                        detection_scope,
                    )
                    for item in filtered:
                        item["source"] = "yolo"
                        if detection_scope == "local":
                            item["label_zh"] = "局部疑似异常皮肤区域"
                    timings.yolo = time.perf_counter() - phase_start
                    detection_paths = save_detection_outputs(
                        output_dir,
                        result.standardized_bgr,
                        raw_detection,
                        filtered,
                        region_counts,
                        region_analysis,
                        artifact_policy=artifact_policy,
                        timings=timings,
                    )
                    original_yolo_path = output_dir / "24_original_yolo_circles.jpg"
                    with timings.measure_image_encoding():
                        write_image(
                            original_yolo_path,
                            draw_circles(result.original_bgr, candidates_in_original_space(filtered)),
                        )
                    detection_paths["original_circle_image"] = str(original_yolo_path)
                    detection = {
                        "status": "ok",
                        "input_mode": input_mode,
                        "detection_scope": detection_scope,
                        "raw_count": raw_detection.get("count", 0),
                        "count": len(filtered),
                        "region_counts": region_counts or {},
                        "region_analysis": region_analysis,
                        "detections": filtered,
                        "elapsed_seconds": raw_detection.get("elapsed_seconds"),
                        "empty_detections_are_valid": True,
                    }
                except DetectorUnavailableError:
                    raise

                if artifact_policy.generates_max_recall_candidates:
                    try:
                        phase_start = time.perf_counter()
                        generator = AcneCandidateGenerator()
                        candidate_result = generator.generate(
                            result.standardized_bgr,
                            result.skin_mask,
                            result.forbidden_mask,
                        )
                        candidate_result["focal_acne_candidates"] = add_original_coordinates(
                            candidate_result["focal_acne_candidates"],
                            result.crop_box_xyxy,
                            result.square_offset_xy,
                            result.resize_scale,
                            result.original_bgr.shape[:2],
                            input_mode,
                            detection_scope,
                        )
                        candidate_result["diffuse_erythema_regions"] = add_original_coordinates(
                            candidate_result["diffuse_erythema_regions"],
                            result.crop_box_xyxy,
                            result.square_offset_xy,
                            result.resize_scale,
                            result.original_bgr.shape[:2],
                            input_mode,
                            detection_scope,
                        )
                        if detection_scope == "local":
                            for item in candidate_result["focal_acne_candidates"]:
                                item["label_zh"] = "局部疑似异常皮肤区域"
                        combined, combined_source_counts = combine_candidates(
                            filtered,
                            candidate_result["focal_acne_candidates"],
                            generator.config,
                        )
                        combined = add_original_coordinates(
                            combined,
                            result.crop_box_xyxy,
                            result.square_offset_xy,
                            result.resize_scale,
                            result.original_bgr.shape[:2],
                            input_mode,
                            detection_scope,
                        )
                        timings.candidate_generation = time.perf_counter() - phase_start
                        candidate_paths = save_candidate_outputs(
                            output_dir,
                            result.standardized_bgr,
                            candidate_result,
                            filtered,
                            combined,
                            combined_source_counts,
                            artifact_policy=artifact_policy,
                            timings=timings,
                        )
                        if artifact_policy.includes_debug_images:
                            original_unsupervised_path = output_dir / "25_original_unsupervised_candidate_circles.jpg"
                            original_combined_path = output_dir / "26_original_combined_candidate_circles.jpg"
                            with timings.measure_image_encoding():
                                write_image(
                                    original_unsupervised_path,
                                    draw_circles(
                                        result.original_bgr,
                                        candidates_in_original_space(candidate_result["focal_acne_candidates"]),
                                    ),
                                )
                                write_image(
                                    original_combined_path,
                                    draw_circles(result.original_bgr, candidates_in_original_space(combined)),
                                )
                            candidate_paths["original_unsupervised_circle_overlay"] = str(original_unsupervised_path)
                            candidate_paths["original_combined_circle_overlay"] = str(original_combined_path)
                        max_recall = {
                            "status": "ok",
                            "input_mode": input_mode,
                            "detection_scope": detection_scope,
                            "label": "acne_candidate",
                            "label_zh": "疑似异常皮肤区域",
                            "yolo_count": len(filtered),
                            "unsupervised_focal_count": len(candidate_result["focal_acne_candidates"]),
                            "diffuse_erythema_region_count": len(candidate_result["diffuse_erythema_regions"]),
                            "combined_count": len(combined),
                            "combined_source_counts": combined_source_counts,
                            "tier_counts": candidate_result["tier_counts"],
                            "statistics": candidate_result["statistics"],
                            "outputs": candidate_paths,
                            "interpretation": "候选池，不是医学确诊或真实痤疮数量",
                        }
                    except Exception as exc:
                        max_recall = {
                            "status": "failed",
                            "reason": "max_recall_generation_failed",
                            "message": f"{type(exc).__name__}: {exc}",
                            "yolo_count": len(filtered),
                            "unsupervised_focal_count": 0,
                            "diffuse_erythema_region_count": 0,
                            "combined_count": len(filtered),
                        }
                else:
                    max_recall = {
                        "status": "skipped",
                        "reason": "formal_fast_policy",
                        "yolo_count": len(filtered),
                        "unsupervised_focal_count": 0,
                        "diffuse_erythema_region_count": 0,
                        "combined_count": len(filtered),
                    }

            summary = {
                "status": "ok",
                "profile": {
                    "name": self.profile.name,
                    "weights": str(self.profile.weights),
                    "imgsz": self.profile.imgsz,
                    "conf": self.profile.conf,
                    "device": self.profile.device,
                },
                "elapsed_seconds": round(time.perf_counter() - start, 4),
                "timing_seconds": timings.as_dict(),
                "artifact_policy": artifact_policy.value,
                "outputs": preprocess_paths,
                "grading": grading,
                "detection": detection,
                "detection_outputs": detection_paths,
                "max_recall": max_recall,
                "max_recall_outputs": candidate_paths,
                "preprocess": {
                    "status": "ok",
                    "input_mode": result.metadata["input_mode"],
                    "grading_allowed": grading_allowed,
                    "grading_rejection_reason": grading_reason,
                    "global_region_analysis_allowed": bool(result.metadata["global_region_analysis_allowed"] and grading_allowed),
                    "local_detection_allowed": result.metadata["local_detection_allowed"],
                    "detection_scope": detection_scope,
                    "face_count": result.metadata["face_count"],
                    "skin_pixels": result.metadata["skin_pixels"],
                    "forbidden_pixels": result.metadata["forbidden_pixels"],
                },
                "warnings": ["AI参考结果，不代替专业诊断"],
            }
            (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            return summary
        except NoFaceDetectedError as exc:
            payload = {
                "status": "failed",
                "reason": "no_face_detected",
                "message": str(exc),
                "elapsed_seconds": round(time.perf_counter() - start, 4),
                "failure_context": write_failure_context("no_face_detected"),
            }
            (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return payload
        except PreprocessError as exc:
            payload = {
                "status": "failed",
                "reason": "preprocess_error",
                "message": str(exc),
                "elapsed_seconds": round(time.perf_counter() - start, 4),
                "failure_context": write_failure_context("preprocess_error"),
            }
            (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return payload
        except DetectorUnavailableError as exc:
            payload = {
                "status": "failed",
                "reason": "detector_deployment_error",
                "message": str(exc),
                "elapsed_seconds": round(time.perf_counter() - start, 4),
                "failure_context": write_failure_context("detector_deployment_error"),
            }
            (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return payload
