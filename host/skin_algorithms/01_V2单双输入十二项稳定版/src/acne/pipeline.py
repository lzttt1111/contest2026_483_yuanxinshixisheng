from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

from src.acne.algorithms.acne_analysis import AcneAnalysisRunner
from src.acne.artifact_policy import AcneArtifactPolicy, AcnePhaseTimings, selected_acne_algorithms
from src.acne.clinical_quantification import (
    try_merge_medical_v2_report,
    write_report as write_clinical_quantification,
)
from src.acne.core.config import settings


SUPPORTED_ALGORITHMS = {"acne", "acne_candidate", "acne_detection"}
logger = logging.getLogger(__name__)


class Pipeline:
    """供 Celery Worker 与本地 CLI 调用的算法分发入口。"""

    def __init__(self) -> None:
        self.output_root = Path(settings.worker_output_dir)
        self.runner = AcneAnalysisRunner(
            profiles_path=settings.detector_profiles_path,
            profile_name=settings.detector_profile,
            device=settings.detector_device,
            enable_grading=settings.enable_grading,
            grading_checkpoint=settings.grading_checkpoint,
            grading_python=settings.grading_python,
            grading_timeout=settings.grading_timeout,
        )

    def warmup(self) -> dict[str, object]:
        return self.runner.warmup()

    def close(self) -> None:
        self.runner.close()

    def process_single(self, image_path: str, algorithms: list[str] | None = None) -> dict:
        artifact_policy = AcneArtifactPolicy.from_algorithms(algorithms)
        selected = selected_acne_algorithms(algorithms) or ["acne"]
        unsupported = sorted(set(selected) - SUPPORTED_ALGORITHMS)
        if unsupported:
            return {
                "status": "failed",
                "message": f"Unsupported algorithms: {', '.join(unsupported)}",
            }

        source = Path(image_path)
        task_name = f"{source.stem}_{int(time.time() * 1000)}"
        output_dir = self.output_root / task_name
        result = self.runner.run(source, output_dir, artifact_policy=artifact_policy)
        if result.get("status") != "ok":
            return {"status": "failed", "message": result.get("message") or result.get("reason", "analysis_failed")}

        result_files = {
            "acne_summary": output_dir / "summary.json",
            "acne_circles": output_dir / "17_acne_candidate_circles.jpg",
            "acne_raw_boxes": output_dir / "16_raw_boxes.jpg",
            "acne_detections": output_dir / "18_detections.csv",
            "acne_skin_mask": output_dir / "04_skin_mask.png",
            "acne_forbidden_mask": output_dir / "05_forbidden_mask.png",
            "acne_standardized": output_dir / "02_standardized.jpg",
            "max_recall_heatmap": output_dir / "19_candidate_heatmap.jpg",
            "diffuse_erythema_heatmap": output_dir / "19a_diffuse_erythema_heatmap.jpg",
            "focal_candidate_heatmap": output_dir / "19b_focal_candidate_heatmap.jpg",
            "max_recall_circles": output_dir / "20_candidate_circle_overlay.jpg",
            "combined_candidate_circles": output_dir / "20a_combined_candidate_overlay.jpg",
            "max_recall_debug": output_dir / "21_candidate_score_debug.jpg",
            "max_recall_json": output_dir / "22_candidate.json",
            "original_yolo_circles": output_dir / "24_original_yolo_circles.jpg",
            "original_unsupervised_circles": output_dir / "25_original_unsupervised_candidate_circles.jpg",
            "original_combined_circles": output_dir / "26_original_combined_candidate_circles.jpg",
        }
        existing_results = {name: str(path) for name, path in result_files.items() if path.exists()}
        if not existing_results:
            # 对于跳过或非完整脸场景，至少保留一份结构化结果。
            fallback = output_dir / "summary.json"
            fallback.write_text("{}", encoding="utf-8")
            existing_results["acne_summary"] = str(fallback)

        # 九项本地入口直接调用 Pipeline，不经过 display_outputs；因此在此
        # 同步生成旁路 V2。失败只记录日志，不能改变旧痤疮成功结果。
        quantification_start = time.perf_counter()
        try:
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            quantification_json = output_dir / "痤疮量化指标.json"
            clinical_report = write_clinical_quantification(
                summary,
                output_dir / "痤疮量化指标.csv",
                quantification_json,
            )
            try_merge_medical_v2_report(
                clinical_report,
                output_dir / "痤疮医学量化指标_V2.csv",
                quantification_json,
            )
        except Exception:
            logger.exception("痤疮医学 V2 旁路准备失败，保留旧 Pipeline 成功结果")
        timings = AcnePhaseTimings.from_mapping(result.get("timing_seconds", {}))
        timings.quantification += time.perf_counter() - quantification_start
        timing_seconds = timings.as_dict()
        logger.info(
            "痤疮阶段耗时 artifact_policy=%s timing_seconds=%s",
            artifact_policy.value,
            timing_seconds,
        )
        result["timing_seconds"] = timing_seconds
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            summary_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # 额外复制一份稳定命名的 summary，便于本地快速查看。
        latest = self.output_root / "latest_summary.json"
        if (output_dir / "summary.json").exists():
            latest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_dir / "summary.json", latest)

        return {
            "status": "success",
            "results": existing_results,
            "output_dir": str(output_dir),
            "summary_path": str(output_dir / "summary.json"),
            "metadata": {
                "artifact_policy": artifact_policy.value,
                "timing_seconds": timing_seconds,
            },
        }
