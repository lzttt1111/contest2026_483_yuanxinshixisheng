# -*- coding: utf-8 -*-
"""Wrinkle detection pipeline for the Celery algorithm worker."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any

from src.wrinkle.core.config import settings
from src.wrinkle.output_files import OUTPUT_IMAGE_FILES
from src.wrinkle.clinical_quantification import (
    try_merge_medical_v2_report,
    write_report as write_clinical_quantification,
)


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "wrinkle"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "wrinkle"

DEBUG_RESULT_FILES = {
    **OUTPUT_IMAGE_FILES,
    "region_metrics_csv": "25_region_metrics.csv",
    "summary_json": "summary.json",
    "quantification_csv": "皱纹量化指标.csv",
    "quantification_json": "皱纹量化指标.json",
}

MEDICAL_V2_CSV = "皱纹医学量化指标_V2.csv"

OUTPUT_END_FIXED_RESULTS = {
    "output_end_texture_reference": "texture_reference",
    "output_end_region_overlay": "region_overlay",
}

SUPPORTED_REGION_KEYS = {
    "forehead", "glabella",
    "left_crow_feet", "right_crow_feet",
    "left_under_eye", "right_under_eye",
    "left_nasolabial", "right_nasolabial",
    "left_marionette", "right_marionette",
    "other",
}

DISPLAY_REGION_FIELDS = (
    "region_key",
    "region_name",
    "short_name",
    "relative_score",
    "segment_count",
    "wrinkle_pixels",
    "mean_segment_length",
    "max_segment_length",
    "density_per_10k",
    "share_pct",
    "area_px",
)

SUPPORTED_ALGORITHMS = {"wrinkle_detection", "wrinkle"}
PRESET_TOKENS = {"fast", "balanced", "full"}


class Pipeline:
    """Algorithm dispatcher called by src.worker.analyze_image()."""

    def __init__(self):
        configured_output = Path(settings.wrinkle_output_dir or DEFAULT_OUTPUT_ROOT)
        if configured_output.is_absolute():
            self.output_root = configured_output
        else:
            self.output_root = PROJECT_ROOT / configured_output
        self.output_root.mkdir(parents=True, exist_ok=True)

    def warmup(self) -> dict[str, Any]:
        from src.wrinkle.algorithms import wrinkle_detection_algorithm

        return wrinkle_detection_algorithm.warmup_runtime(
            MODEL_DIR / "best.pt",
            MODEL_DIR / "face_landmarker.task",
            MODEL_DIR / "selfie_multiclass.tflite",
            settings.wrinkle_device,
        )

    def close(self) -> None:
        from src.wrinkle.algorithms import wrinkle_detection_algorithm

        wrinkle_detection_algorithm.close_runtime_models()

    def process_single(self, image_path: str, algorithms: list[str] | None = None) -> dict[str, Any]:
        """Run wrinkle detection for one local image.

        Args:
            image_path: Local image path prepared by the worker.
            algorithms: Optional algorithm names. Supports "wrinkle_detection" and
                "wrinkle"; "fast", "balanced", or "full" may be passed to override
                the run preset for local tests.
        """
        requested = set(algorithms or ["wrinkle_detection"])
        preset = self._resolve_preset(requested)
        algorithm_names = requested - PRESET_TOKENS

        if algorithm_names and not algorithm_names.intersection(SUPPORTED_ALGORITHMS):
            return {
                "status": "failed",
                "message": f"Unsupported algorithms: {sorted(algorithm_names)}",
            }

        source_path = Path(image_path).expanduser().resolve()
        if not source_path.is_file():
            return {"status": "failed", "message": f"Input image not found: {source_path}"}

        required_files = {
            "weights": MODEL_DIR / "best.pt",
            "face_model": MODEL_DIR / "face_landmarker.task",
            "segmenter_model": MODEL_DIR / "selfie_multiclass.tflite",
        }
        missing = [name for name, path in required_files.items() if not path.is_file()]
        if missing:
            return {
                "status": "failed",
                "message": "Missing required files: "
                + ", ".join(f"{name}={required_files[name]}" for name in missing),
            }

        output_dir = self._make_output_dir(source_path, preset)
        algorithm_args = self._build_algorithm_args(
            source_path=source_path,
            output_dir=output_dir,
            weights_path=required_files["weights"],
            face_model_path=required_files["face_model"],
            segmenter_model_path=required_files["segmenter_model"],
            preset=preset,
        )

        started = time.perf_counter()
        try:
            from src.wrinkle.algorithms import wrinkle_detection_algorithm

            returncode = wrinkle_detection_algorithm.run_from_args(algorithm_args)
        except Exception as exc:
            elapsed_seconds = round(time.perf_counter() - started, 3)
            error_traceback = traceback.format_exc()
            logger.exception("Wrinkle detection raised an exception")
            return {
                "status": "failed",
                "message": f"wrinkle detection raised {type(exc).__name__}: {exc}",
                "returncode": "exception",
                "elapsed_seconds": elapsed_seconds,
                "output_dir": str(output_dir),
                "algorithm_args": vars(algorithm_args),
                "traceback": error_traceback,
            }
        elapsed_seconds = round(time.perf_counter() - started, 3)

        if returncode != 0:
            return {
                "status": "failed",
                "message": f"wrinkle detection exited with code {returncode}",
                "returncode": returncode,
                "elapsed_seconds": elapsed_seconds,
                "output_dir": str(output_dir),
                "algorithm_args": vars(algorithm_args),
            }

        summary_path = output_dir / "summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "status": "failed",
                "message": f"summary.json invalid: {type(exc).__name__}: {exc}",
                "returncode": returncode,
                "elapsed_seconds": elapsed_seconds,
                "output_dir": str(output_dir),
                "algorithm_args": vars(algorithm_args),
                "traceback": traceback.format_exc(),
            }

        if summary.get("region_analysis_status") != "ok":
            return {
                "status": "failed",
                "message": f"region_analysis_status={summary.get('region_analysis_status')!r}",
                "returncode": returncode,
                "elapsed_seconds": elapsed_seconds,
                "output_dir": str(output_dir),
                "algorithm_args": vars(algorithm_args),
                "summary": summary,
            }

        clinical_report = write_clinical_quantification(
            summary,
            output_dir / DEBUG_RESULT_FILES["quantification_csv"],
            output_dir / DEBUG_RESULT_FILES["quantification_json"],
        )
        try_merge_medical_v2_report(
            clinical_report,
            output_dir / MEDICAL_V2_CSV,
            output_dir / DEBUG_RESULT_FILES["quantification_json"],
        )

        (
            display_results,
            debug_results,
            upload_relative_paths,
            display_manifest,
        ) = self._collect_results(output_dir, summary)
        results = {**debug_results, **display_results}

        return {
            "status": "success",
            "output_dir": str(output_dir),
            "results": results,
            "display_results": display_results,
            "debug_results": debug_results,
            "display_manifest": display_manifest,
            "upload_relative_paths": upload_relative_paths,
            "metadata": {
                "algorithm": "wrinkle_detection",
                "run_preset": preset,
                "output_dir": str(output_dir),
                "elapsed_seconds": elapsed_seconds,
                "device": summary.get("device"),
                "successful_runs": summary.get("successful_runs"),
                "failed_runs": summary.get("failed_runs"),
                "region_analysis_status": summary.get("region_analysis_status"),
                "summary": {"region_metrics": summary.get("region_metrics", [])},
            },
        }

    def _collect_results(
        self,
        output_dir: Path,
        summary: dict[str, Any],
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, Any]]:
        debug_results = {
            name: str(output_dir / filename)
            for name, filename in DEBUG_RESULT_FILES.items()
            if (output_dir / filename).is_file()
        }
        upload_relative_paths: dict[str, str] = {
            result_key: Path(debug_results[result_key]).name
            for result_key in OUTPUT_IMAGE_FILES
            if result_key in debug_results
        }

        display_results: dict[str, str] = {}
        output_end = summary.get("output_end") or {}
        fixed_result_keys: dict[str, str | None] = {
            "overview_result_key": None,
            "texture_result_key": None,
        }

        for result_key, manifest_key in OUTPUT_END_FIXED_RESULTS.items():
            relative_path = output_end.get(manifest_key)
            local_path = self._resolve_output_end_file(output_dir, relative_path)
            if local_path is None:
                continue
            display_results[result_key] = str(local_path)
            upload_relative_paths[result_key] = relative_path
            if manifest_key == "region_overlay":
                fixed_result_keys["overview_result_key"] = result_key
            elif manifest_key == "texture_reference":
                fixed_result_keys["texture_result_key"] = result_key

        for result_key in ("quantification_csv", "quantification_json"):
            local_path = debug_results.get(result_key)
            if not local_path:
                continue
            display_results[result_key] = local_path
            upload_relative_paths[result_key] = Path(local_path).name

        fixed_result_keys["quantification_csv_result_key"] = (
            "quantification_csv" if "quantification_csv" in display_results else None
        )
        fixed_result_keys["quantification_json_result_key"] = (
            "quantification_json" if "quantification_json" in display_results else None
        )

        metrics_by_key = {
            metric.get("region_key"): metric
            for metric in summary.get("region_metrics") or []
            if metric.get("region_key") in SUPPORTED_REGION_KEYS
        }
        display_regions: list[dict[str, Any]] = []

        for card in output_end.get("region_cards") or []:
            region_key = card.get("region_key")
            relative_path = card.get("relative_path")
            if region_key not in SUPPORTED_REGION_KEYS:
                logger.warning("忽略未知分区结果：%r", region_key)
                continue
            local_path = self._resolve_output_end_file(output_dir, relative_path)
            metric = metrics_by_key.get(region_key)
            if local_path is None or metric is None:
                continue
            result_key = f"region_{region_key}"
            display_results[result_key] = str(local_path)
            upload_relative_paths[result_key] = relative_path
            region_data = {
                field: metric.get(field)
                for field in DISPLAY_REGION_FIELDS
            }
            region_data["result_key"] = result_key
            display_regions.append(region_data)

        display_regions.sort(
            key=lambda item: float(item.get("relative_score") or 0.0),
            reverse=True,
        )
        display_manifest = {
            **fixed_result_keys,
            "regions": display_regions,
        }
        return display_results, debug_results, upload_relative_paths, display_manifest

    @staticmethod
    def _resolve_output_end_file(output_dir: Path, relative_path: Any) -> Path | None:
        if not isinstance(relative_path, str) or not relative_path:
            return None

        output_end_root = (output_dir / "output-end").resolve()
        candidate = (output_dir / relative_path).resolve()
        try:
            candidate.relative_to(output_end_root)
        except ValueError:
            logger.warning("忽略 output-end 之外的结果路径：%s", relative_path)
            return None
        if not candidate.is_file():
            logger.warning("output-end 结果文件不存在：%s", candidate)
            return None
        return candidate

    def _resolve_preset(self, requested: set[str]) -> str:
        for preset in ("fast", "balanced", "full"):
            if preset in requested:
                return preset
        return settings.wrinkle_run_preset

    def _make_output_dir(self, source_path: Path, preset: str) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in source_path.stem)
        output_dir = self.output_root / f"{safe_stem}_{preset}_{stamp}_{os.getpid()}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _build_algorithm_args(
        self,
        *,
        source_path: Path,
        output_dir: Path,
        weights_path: Path,
        face_model_path: Path,
        segmenter_model_path: Path,
        preset: str,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            weights=str(weights_path),
            source=str(source_path),
            output=str(output_dir),
            preprocess_size=1024,
            face_model=str(face_model_path),
            segmenter_model=str(segmenter_model_path),
            no_semantic_skin=False,
            profile="extreme",
            device=settings.wrinkle_device,
            cpu_threads=settings.wrinkle_cpu_threads,
            run_preset=preset,
            raw_conf=0.01,
            line_width=None,
            save_runs=False,
            report_group_overlays=True,
        )
