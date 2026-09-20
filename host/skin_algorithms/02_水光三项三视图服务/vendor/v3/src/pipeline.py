# -*- coding: utf-8 -*-
"""
核心调度器 (Pipeline) — 一次预处理，按需分发五项皮肤检测
"""

import os
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from typing_extensions import assert_never

from src.capture_profile import CaptureProfile
from src.core.config import settings
from src.profile_strategy import SkinStyleBundle, StyleRenderer, strategy_for_profile
from src.utils.io_utils import cv_imread, cv_imwrite
from src.preprocess.image_preprocessor import ImagePreprocessor
from src.preprocess.profile_mask_policy import apply_preprocess_mask_bundle
from src.engines.rbx_engine import ErythemaAnalyzer as RBXAnalyzer
from src.engines.spots_engine import SpotsEngine
from src.engines.brown_engine import BrownAreaAnalyzer
from src.engines.texture_engine import TextureAnalyzer
from src.engines.pores_engine import PoresAnalyzer
from src.engines.purple_analysis_engine import PurpleAnalysisEngine
from src.engines.wrinkles_engine import WrinklesEngine
from src.engines.vendor_skin_overlay import (
    render_consumer_brown_result,
    render_vendor_brown_result,
    render_vendor_red_result,
)
from src.engines.vendor_skin_style import (
    VendorSkinStyleError,
    VendorSkinStyleProvider,
)
from src.utils.gpu_backend import get_cuda_backend

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPPORTED_CLOUD_ALGORITHMS = {
    "redness",
    "spots",
    "brown",
    "texture",
    "pores",
    "purple",
    "surface_gloss",
    "vascular",
    "contour_firmness",
}


def brown_renderer_for_profile(profile: CaptureProfile) -> StyleRenderer:
    """Keep institution colour frozen while consumer reuses redness markers."""
    match profile:
        case CaptureProfile.INSTITUTION:
            return render_vendor_brown_result
        case CaptureProfile.CONSUMER:
            return render_consumer_brown_result
        case unreachable:
            assert_never(unreachable)


class DermaVisionPipeline:
    """DermaVision 主流水线调度器 — 两步走可视化输出"""

    def __init__(self, capture_profile: CaptureProfile | None = None):
        self.capture_profile = capture_profile or settings.capture_profile
        self.style_strategy = strategy_for_profile(self.capture_profile)
        self.raw_dir = os.path.join(PROJECT_ROOT, "data", "raw_images")
        self.ref_path = os.path.join(PROJECT_ROOT, "data", "reference.jpg")
        self.output_dir = os.path.join(PROJECT_ROOT, "output")
        self.preprocessed_dir = os.path.join(self.output_dir, "preprocessed")
        self.rbx_dir = os.path.join(self.output_dir, "rbx")
        self.spots_dir = os.path.join(self.output_dir, "spots")
        self.brown_dir = os.path.join(self.output_dir, "brown")
        self.texture_dir = os.path.join(self.output_dir, "texture")
        self.pores_dir = os.path.join(self.output_dir, "pores")
        self.purple_dir = os.path.join(self.output_dir, "purple")
        self.surface_gloss_dir = os.path.join(self.output_dir, "surface_gloss")
        self.vascular_dir = os.path.join(self.output_dir, "vascular")
        self.contour_firmness_dir = os.path.join(self.output_dir, "contour_firmness")
        self.wrinkles_dir = os.path.join(self.output_dir, "wrinkles")

        os.makedirs(self.preprocessed_dir, exist_ok=True)
        os.makedirs(self.rbx_dir, exist_ok=True)
        os.makedirs(self.spots_dir, exist_ok=True)
        os.makedirs(self.brown_dir, exist_ok=True)
        os.makedirs(self.texture_dir, exist_ok=True)
        os.makedirs(self.pores_dir, exist_ok=True)
        os.makedirs(self.purple_dir, exist_ok=True)
        os.makedirs(self.surface_gloss_dir, exist_ok=True)
        os.makedirs(self.vascular_dir, exist_ok=True)
        os.makedirs(self.contour_firmness_dir, exist_ok=True)
        os.makedirs(self.wrinkles_dir, exist_ok=True)

        self.preprocessor = None
        self.rbx_analyzer = None
        self.spots_analyzer = None
        self.brown_analyzer = None
        self.texture_analyzer = None
        self.pores_analyzer = None
        self.purple_analyzer = None
        self.wrinkles_analyzer = None
        self.vendor_skin_provider = None

    def warmup(self) -> dict[str, object]:
        """每个 Celery 子进程只执行一次的模型和 CUDA 冷启动。"""
        cuda_backend = get_cuda_backend()
        cuda_metadata = cuda_backend.warmup()

        if self.preprocessor is None:
            self.preprocessor = ImagePreprocessor(
                self.raw_dir,
                self.preprocessed_dir,
                self.ref_path,
                capture_profile=self.capture_profile,
            )
        if self.rbx_analyzer is None:
            self.rbx_analyzer = RBXAnalyzer(capture_profile=self.capture_profile)
        if self.spots_analyzer is None:
            self.spots_analyzer = SpotsEngine()
        if self.brown_analyzer is None:
            self.brown_analyzer = BrownAreaAnalyzer(self.capture_profile)
        # 红区和棕区共用同一组人脸/前景分割模型，避免一个子进程重复加载。
        self.brown_analyzer._face_mask_analyzer = self.rbx_analyzer
        if self.texture_analyzer is None:
            self.texture_analyzer = TextureAnalyzer()
        if self.pores_analyzer is None:
            self.pores_analyzer = PoresAnalyzer()
        if self.purple_analyzer is None:
            self.purple_analyzer = PurpleAnalysisEngine(
                capture_profile=self.capture_profile,
            )

        return {
            "cuda": cuda_metadata,
            "models": {
                "preprocessor": type(self.preprocessor).__name__,
                "redness": type(self.rbx_analyzer).__name__,
                "spots": type(self.spots_analyzer).__name__,
                "brown": type(self.brown_analyzer).__name__,
                "texture": type(self.texture_analyzer).__name__,
                "pores": type(self.pores_analyzer).__name__,
                "purple": type(self.purple_analyzer).__name__,
            },
        }

    def close(self) -> None:
        """Worker 子进程退出时释放常驻 MediaPipe 资源。"""
        if self.brown_analyzer is not None:
            # 共享模型由 rbx_analyzer 统一关闭，避免重复 close。
            self.brown_analyzer._face_mask_analyzer = None
        for analyzer in (
            self.rbx_analyzer,
            self.spots_analyzer,
            self.purple_analyzer,
            self.preprocessor,
        ):
            close = getattr(analyzer, "close", None)
            if callable(close):
                close()

    def _validate_environment(self):
        """启动前环境校验：确保关键文件存在"""
        if not os.path.exists(self.raw_dir):
            raise FileNotFoundError(
                f"❌ 原始图片目录不存在: {self.raw_dir}"
            )
        raw_files = [f for f in os.listdir(self.raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not raw_files:
            raise ValueError(
                f"❌ 原始图片目录为空: {self.raw_dir}\n"
                f"   请先放入测试照片。"
            )
        print("✅ 环境校验通过：原始图片目录就绪。")

    def run_preprocess(self):
        """阶段 1：图像预处理（展示图 + 算法图）→ 输出到 output/preprocessed/"""
        print("\n" + "=" * 70)
        print(" 📸 阶段 1：图像预处理流水线 (→ output/preprocessed/)")
        print("=" * 70)

        self.preprocessor = ImagePreprocessor(self.raw_dir, self.preprocessed_dir, self.ref_path)
        processed_files = self.preprocessor.process_batch()
        return processed_files

    def run_rbx(self, input_files=None):
        """阶段 2：红区检测 (RBX) → 读取 preprocessed/ 目录，输出到 output/rbx/"""
        print("\n" + "=" * 70)
        print(" 🩸 阶段 2：红区检测引擎 (→ output/rbx/)")
        print("=" * 70)

        self.rbx_analyzer = RBXAnalyzer(capture_profile=self.capture_profile)

        if input_files:
            for filename in input_files:
                img_path = os.path.join(self.preprocessed_dir, filename)
                self.rbx_analyzer.process_image(img_path, self.rbx_dir)
        else:
            analysis_files = [
                f for f in os.listdir(self.preprocessed_dir)
                if f.endswith('_01_AnalysisInput.jpg') or f.endswith('_01_AnalysisInput.png')
            ]
            if analysis_files:
                preprocessed_files = analysis_files
            else:
                preprocessed_files = [
                    f for f in os.listdir(self.preprocessed_dir)
                    if f.endswith('_01_Preprocessed.jpg') or f.endswith('_01_Preprocessed.png')
                ]
            if not preprocessed_files:
                print("⚠️ 未找到算法输入图，请先运行预处理阶段。")
                return
            for filename in sorted(preprocessed_files):
                img_path = os.path.join(self.preprocessed_dir, filename)
                self.rbx_analyzer.process_image(img_path, self.rbx_dir)

    def process_single(self, image_path, algorithms=None):
        """
        Celery 单图处理接口 — 预处理只执行一次，动态分发算法
        
        Args:
            image_path (str): 待处理图片的绝对路径
            algorithms (List[str] | None): 支持 redness、spots、brown、
                texture、pores、purple；为空时为兼容旧调用执行 redness、spots。
        
        Returns:
            dict: results 只包含云端正式返回文件；预处理和调试文件通过
                  cleanup_paths 交给 Worker 在上传成功后清理。
        """
        pipeline_started_at = time.perf_counter()
        algorithm_timings: dict[str, float] = {}
        timing_seconds: dict[str, object] = {"algorithms": algorithm_timings}
        requested_algorithms = list(algorithms or ["redness", "spots"])
        unknown = sorted(set(requested_algorithms) - SUPPORTED_CLOUD_ALGORITHMS)
        if unknown:
            return {"status": "failed", "message": f"不支持的算法: {unknown}"}
        if not requested_algorithms:
            return {"status": "failed", "message": "algorithms 不能为空"}

        cuda_backend = get_cuda_backend()
        cuda_profile_baseline = cuda_backend.begin_task_profile()

        print(f"\n🎯 [Celery Task] 开始处理: {image_path}")
        print(f"   请求算法: {requested_algorithms}")

        # 校验图片是否存在
        if not os.path.exists(image_path):
            return {"status": "failed", "message": f"图片不存在: {image_path}"}

        results = {}
        # 医学 V2 旁路路径不进入旧 results/raw_result；Worker 仅在显式
        # 开关开启且后端已兼容时读取并返回。
        medical_v2_results = {}
        cleanup_paths = []

        # ===== 步骤 1：预处理（绝对单例，只执行一次）=====
        print("\n📸 [预处理] 执行 V2 展示图 + 算法图生成...")
        if self.preprocessor is None:
            self.preprocessor = ImagePreprocessor(
                self.raw_dir,
                self.preprocessed_dir,
                self.ref_path,
                capture_profile=self.capture_profile,
            )

        filename = os.path.basename(image_path)
        img = cv_imread(image_path)
        if img is None:
            return {"status": "failed", "message": f"图片读取失败: {image_path}"}

        preprocess_started_at = time.perf_counter()
        preprocess_result = self.preprocessor.preprocess_image(img)
        apply_preprocess_mask_bundle(self.capture_profile, preprocess_result)
        from src.doctor_v3.evidence import save_preprocess
        save_preprocess(self.preprocessed_dir, preprocess_result)
        timing_seconds["preprocess"] = round(
            time.perf_counter() - preprocess_started_at,
            4,
        )
        flags_text = ",".join(preprocess_result.quality_flags) if preprocess_result.quality_flags else "NONE"
        print(
            f"   [Quality] {filename}: {preprocess_result.quality_score:.2f} "
            f"{preprocess_result.quality_status} [{flags_text}]"
        )

        base_name = os.path.splitext(filename)[0]
        preprocessed_name = f"{base_name}_01_Preprocessed.jpg"
        analysis_name = f"{base_name}_01_AnalysisInput.jpg"
        preprocessed_path = os.path.join(self.preprocessed_dir, preprocessed_name)
        analysis_path = os.path.join(self.preprocessed_dir, analysis_name)
        cv_imwrite(preprocessed_path, preprocess_result.display_image)
        cv_imwrite(analysis_path, preprocess_result.analysis_image)
        cleanup_paths.extend([preprocessed_path, analysis_path])
        print(f"✅ [预处理展示图] 完成 -> {preprocessed_path}")
        print(f"✅ [预处理算法图] 完成 -> {analysis_path}")

        if preprocess_result.quality_status == "REJECT":
            flags = ",".join(preprocess_result.quality_flags) or "UNKNOWN"
            return {
                "status": "failed",
                "message": f"图像质量门禁拒绝: {flags}",
                "cleanup_paths": cleanup_paths,
                "metadata": {
                    "quality_score": preprocess_result.quality_score,
                    "quality_status": preprocess_result.quality_status,
                    "quality_flags": preprocess_result.quality_flags,
                    "quality_metrics": preprocess_result.quality_metrics,
                },
            }

        # ===== 步骤 2：动态分发算法 =====
        skin_style: SkinStyleBundle | None = None
        if {"redness", "brown", "vascular"}.intersection(requested_algorithms):
            vendor_started_at = time.perf_counter()
            if self.style_strategy.requires_vendor and self.vendor_skin_provider is None:
                self.vendor_skin_provider = VendorSkinStyleProvider()
            vendor_root = Path(self.preprocessed_dir) / (
                f"{base_name}_vendor_style_{uuid4().hex}"
            )
            try:
                skin_style = self.style_strategy.build(
                    preprocess_result,
                    img,
                    vendor_root,
                    self.vendor_skin_provider,
                )
            except VendorSkinStyleError as exc:
                return {
                    "status": "failed",
                    "message": f"厂家红棕底图生成失败: {exc}",
                    "cleanup_paths": cleanup_paths,
                }
            if skin_style.cleanup_root is not None:
                cleanup_paths.append(str(skin_style.cleanup_root))
            timing_seconds[self.style_strategy.timing_key] = round(
                time.perf_counter() - vendor_started_at,
                4,
            )

        if "redness" in requested_algorithms:
            print("\n🩸 [红区检测] 执行 RBX 引擎...")
            algorithm_started_at = time.perf_counter()
            if self.rbx_analyzer is None:
                self.rbx_analyzer = RBXAnalyzer(capture_profile=self.capture_profile)
            assert skin_style is not None
            rbx_path = self.rbx_analyzer.process_preprocess_result(
                skin_style.red,
                self.rbx_dir,
                filename,
            )
            if rbx_path:
                redness_dir = os.path.dirname(rbx_path)
                detector_overlay = os.path.join(
                    redness_dir,
                    "06_VISIA红色区实例图.jpg",
                )
                redness_mask = os.path.join(
                    redness_dir,
                    "07_VISIA红色区实例Mask.png",
                )
                redness_report = os.path.join(
                    redness_dir,
                    "红区量化指标.csv",
                )
                redness_metrics = os.path.join(
                    redness_dir,
                    "红区量化指标.json",
                )
                missing = [
                    path
                    for path in (
                        detector_overlay,
                        redness_mask,
                        redness_report,
                        redness_metrics,
                    )
                    if not os.path.isfile(path)
                ]
                if missing:
                    return {
                        "status": "failed",
                        "message": f"红区必需结果未生成: {missing}",
                        "cleanup_paths": cleanup_paths,
                    }
                marker_mask = cv_imread(redness_mask)
                if marker_mask is None:
                    return {
                        "status": "failed",
                        "message": "红区特征Mask无法读取",
                        "cleanup_paths": cleanup_paths,
                    }
                self.style_strategy.finalize_red(
                    skin_style,
                    rbx_path,
                    detector_overlay,
                    marker_mask,
                    render_vendor_red_result,
                )
                results["redness"] = rbx_path
                results["red_areas_overlay"] = detector_overlay
                results["redness_report"] = redness_report
                results["redness_metrics"] = redness_metrics
                v2_json = redness_metrics
                v2_csv = os.path.join(redness_dir, "红区医学量化指标_V2.csv")
                if os.path.isfile(v2_json) and os.path.isfile(v2_csv):
                    medical_v2_results["redness"] = {"json": v2_json, "csv": v2_csv}
                cleanup_paths.append(redness_dir)
                algorithm_timings["redness"] = round(
                    time.perf_counter() - algorithm_started_at,
                    4,
                )
                print(f"✅ [红区检测] 完成 -> {rbx_path}")
            else:
                return {
                    "status": "failed",
                    "message": "红区检测未生成结果",
                    "cleanup_paths": cleanup_paths,
                }

        if "spots" in requested_algorithms:
            print("\n🔵 [色斑检测] 执行 Spots 引擎...")
            algorithm_started_at = time.perf_counter()
            if self.spots_analyzer is None:
                self.spots_analyzer = SpotsEngine()
            spots_path = self.spots_analyzer.process_preprocess_result(
                preprocess_result,
                self.spots_dir,
                base_name,
            )
            if spots_path:
                spots_dir = os.path.dirname(spots_path)
                spots_report = os.path.join(
                    spots_dir,
                    "02_Spots量化指标.csv",
                )
                spots_metrics = os.path.join(
                    spots_dir,
                    "02_Spots量化指标.json",
                )
                missing = [
                    path for path in (spots_report, spots_metrics)
                    if not os.path.isfile(path)
                ]
                if missing:
                    return {
                        "status": "failed",
                        "message": f"斑点必需结果未生成: {missing}",
                        "cleanup_paths": cleanup_paths,
                    }
                results["spots"] = spots_path
                results["spots_report"] = spots_report
                results["spots_metrics"] = spots_metrics
                v2_json = spots_metrics
                v2_csv = os.path.join(spots_dir, "02_Spots医学量化指标_V2.csv")
                if os.path.isfile(v2_json) and os.path.isfile(v2_csv):
                    medical_v2_results["spots"] = {"json": v2_json, "csv": v2_csv}
                cleanup_paths.append(spots_dir)
                algorithm_timings["spots"] = round(
                    time.perf_counter() - algorithm_started_at,
                    4,
                )
                print(f"✅ [色斑检测] 完成 -> {spots_path}")
            else:
                return {
                    "status": "failed",
                    "message": "斑点检测未生成结果",
                    "cleanup_paths": cleanup_paths,
                }

        if "brown" in requested_algorithms:
            print("\n🟤 [棕色斑检测] 执行 Brown 引擎...")
            algorithm_started_at = time.perf_counter()
            if self.brown_analyzer is None:
                self.brown_analyzer = BrownAreaAnalyzer(self.capture_profile)
                self.brown_analyzer._face_mask_analyzer = self.rbx_analyzer
            assert skin_style is not None
            brown_overlay = self.brown_analyzer.process_preprocess_result(
                skin_style.brown,
                self.brown_dir,
                filename,
            )
            if brown_overlay:
                brown_dir = os.path.dirname(brown_overlay)
                brown_rbx = os.path.join(
                    brown_dir,
                    "01_RBX棕区结果图.jpg",
                )
                brown_report = os.path.join(
                    brown_dir,
                    "棕色斑量化指标.csv",
                )
                brown_metrics = os.path.join(
                    brown_dir,
                    "棕色斑量化指标.json",
                )
                brown_mask = os.path.join(
                    brown_dir,
                    "03_棕色斑实例Mask.png",
                )
                missing = [
                    path
                    for path in (
                        brown_rbx,
                        brown_overlay,
                        brown_mask,
                        brown_report,
                        brown_metrics,
                    )
                    if not os.path.isfile(path)
                ]
                if missing:
                    return {
                        "status": "failed",
                        "message": f"棕区必需结果未生成: {missing}",
                        "cleanup_paths": cleanup_paths,
                    }
                instance_mask = cv_imread(brown_mask)
                if instance_mask is None:
                    return {
                        "status": "failed",
                        "message": "棕区实例Mask无法读取",
                        "cleanup_paths": cleanup_paths,
                    }
                self.style_strategy.finalize_brown(
                    skin_style,
                    brown_rbx,
                    brown_overlay,
                    instance_mask,
                    brown_renderer_for_profile(self.capture_profile),
                )
                results["brown"] = brown_rbx
                results["brown_spots_overlay"] = brown_overlay
                results["brown_report"] = brown_report
                results["brown_metrics"] = brown_metrics
                v2_json = brown_metrics
                v2_csv = os.path.join(brown_dir, "棕色斑医学量化指标_V2.csv")
                if os.path.isfile(v2_json) and os.path.isfile(v2_csv):
                    medical_v2_results["brown"] = {"json": v2_json, "csv": v2_csv}
                cleanup_paths.append(brown_dir)
                algorithm_timings["brown"] = round(
                    time.perf_counter() - algorithm_started_at,
                    4,
                )
                print(f"✅ [棕色斑检测] 完成 -> {brown_rbx}")
            else:
                return {
                    "status": "failed",
                    "message": "棕色斑检测未生成结果",
                    "cleanup_paths": cleanup_paths,
                }

        if "texture" in requested_algorithms:
            print("\n🟡🔵 [纹理检测] 执行 Texture 引擎...")
            algorithm_started_at = time.perf_counter()
            if self.texture_analyzer is None:
                self.texture_analyzer = TextureAnalyzer()
            texture_path = self.texture_analyzer.process_preprocess_result(
                preprocess_result,
                self.texture_dir,
                filename,
            )
            if texture_path:
                texture_dir = os.path.dirname(texture_path)
                texture_report = os.path.join(
                    texture_dir,
                    "纹理量化指标.csv",
                )
                texture_metrics = os.path.join(
                    texture_dir,
                    "纹理量化指标.json",
                )
                missing = [
                    path
                    for path in (
                        texture_path,
                        texture_report,
                        texture_metrics,
                    )
                    if not os.path.isfile(path)
                ]
                if missing:
                    return {
                        "status": "failed",
                        "message": f"纹理必需结果未生成: {missing}",
                        "cleanup_paths": cleanup_paths,
                    }
                results["texture"] = texture_path
                results["texture_report"] = texture_report
                results["texture_metrics"] = texture_metrics
                v2_json = texture_metrics
                v2_csv = os.path.join(texture_dir, "纹理医学量化指标_V2.csv")
                if os.path.isfile(v2_json) and os.path.isfile(v2_csv):
                    medical_v2_results["texture"] = {"json": v2_json, "csv": v2_csv}
                cleanup_paths.append(texture_dir)
                algorithm_timings["texture"] = round(
                    time.perf_counter() - algorithm_started_at,
                    4,
                )
                print(f"✅ [纹理检测] 完成 -> {texture_path}")
            else:
                return {
                    "status": "failed",
                    "message": "纹理检测未生成结果",
                    "cleanup_paths": cleanup_paths,
                }

        if "pores" in requested_algorithms:
            print("\n🟣 [毛孔检测] 执行 Pores 引擎...")
            algorithm_started_at = time.perf_counter()
            if self.pores_analyzer is None:
                self.pores_analyzer = PoresAnalyzer()
            pores_path = self.pores_analyzer.process_preprocess_result(
                preprocess_result,
                self.pores_dir,
                filename,
            )
            if pores_path:
                pores_dir = os.path.dirname(pores_path)
                pores_report = os.path.join(
                    pores_dir,
                    "毛孔量化指标.csv",
                )
                pores_metrics = os.path.join(
                    pores_dir,
                    "毛孔量化指标.json",
                )
                missing = [
                    path
                    for path in (
                        pores_path,
                        pores_report,
                        pores_metrics,
                    )
                    if not os.path.isfile(path)
                ]
                if missing:
                    return {
                        "status": "failed",
                        "message": f"毛孔必需结果未生成: {missing}",
                        "cleanup_paths": cleanup_paths,
                    }
                results["pores"] = pores_path
                results["pores_report"] = pores_report
                results["pores_metrics"] = pores_metrics
                v2_json = pores_metrics
                v2_csv = os.path.join(pores_dir, "毛孔医学量化指标_V2.csv")
                if os.path.isfile(v2_json) and os.path.isfile(v2_csv):
                    medical_v2_results["pores"] = {"json": v2_json, "csv": v2_csv}
                cleanup_paths.append(pores_dir)
                algorithm_timings["pores"] = round(
                    time.perf_counter() - algorithm_started_at,
                    4,
                )
                print(f"✅ [毛孔检测] 完成 -> {pores_path}")
            else:
                return {
                    "status": "failed",
                    "message": "毛孔检测未生成结果",
                    "cleanup_paths": cleanup_paths,
                }

        if "purple" in requested_algorithms:
            print("\n🟪 [紫区检测] 执行标准化 UV / 荧光 UV 引擎...")
            algorithm_started_at = time.perf_counter()
            if self.purple_analyzer is None:
                self.purple_analyzer = PurpleAnalysisEngine(
                    capture_profile=self.capture_profile,
                )
            purple_primary = self.purple_analyzer.process_preprocess_result(
                preprocess_result,
                self.purple_dir,
                filename,
            )
            if purple_primary:
                purple_dir = os.path.dirname(purple_primary)
                purple_paths = self.purple_analyzer.last_paths
                required = {
                    "purple_uv_base": purple_paths.get("uv_like_base"),
                    "purple_uv_spots_overlay": purple_paths.get("uv_spots_overlay"),
                    "purple_fluorescence_base": purple_paths.get("porphyrin_base"),
                    "purple_porphyrin_overlay": purple_paths.get("porphyrin_overlay"),
                    "purple_report": purple_paths.get("metrics_csv"),
                    "purple_metrics": purple_paths.get("metrics_json"),
                    "purple_full_metrics": purple_paths.get("full_metrics_json"),
                }
                missing = [
                    path
                    for path in required.values()
                    if not path or not os.path.isfile(path)
                ]
                if missing:
                    return {
                        "status": "failed",
                        "message": f"紫区必需结果未生成: {missing}",
                        "cleanup_paths": cleanup_paths,
                    }
                results.update(required)
                # 紫区正式 metrics 是前端精简英文摘要。医生详细宽表仅作为
                # 本地/批处理产物，不通过 medical_metrics_v2 透传给前端，
                # 避免把数百项内部统计误当成云端展示字段。
                cleanup_paths.append(purple_dir)
                algorithm_timings["purple"] = round(
                    time.perf_counter() - algorithm_started_at,
                    4,
                )
                print(f"✅ [紫区检测] 完成 -> {purple_dir}")
            else:
                return {
                    "status": "failed",
                    "message": "紫区检测未生成结果",
                    "cleanup_paths": cleanup_paths,
                }

        if "surface_gloss" in requested_algorithms:
            from src.added_algorithms.runner import run_surface_gloss

            algorithm_started_at = time.perf_counter()
            artifacts = run_surface_gloss(
                preprocess_result,
                self.surface_gloss_dir,
                filename,
            )
            results.update({
                "surface_gloss": str(artifacts.overlay),
                "surface_gloss_metrics": str(artifacts.metrics.metrics_json),
                "surface_gloss_report": str(artifacts.metrics.compact_csv),
                "surface_gloss_medical_report_csv_v2": str(artifacts.metrics.medical_csv_v2),
                "surface_gloss_full_metrics": str(artifacts.metrics.full_metrics_json),
            })
            cleanup_paths.append(str(artifacts.overlay.parent))
            algorithm_timings["surface_gloss"] = round(time.perf_counter() - algorithm_started_at, 4)

        if "vascular" in requested_algorithms:
            assert skin_style is not None
            if self.capture_profile is CaptureProfile.CONSUMER:
                red_display_path = results.get("redness")
                if red_display_path is None:
                    if self.rbx_analyzer is None:
                        self.rbx_analyzer = RBXAnalyzer(capture_profile=self.capture_profile)
                    red_display_path = self.rbx_analyzer.process_preprocess_result(
                        skin_style.red,
                        self.rbx_dir,
                        filename,
                    )
                    if red_display_path is None:
                        return {
                            "status": "failed",
                            "message": "consumer血管缺少派生红区底图",
                            "cleanup_paths": cleanup_paths,
                        }
                    cleanup_paths.append(os.path.dirname(red_display_path))
                red_display = cv_imread(red_display_path)
                if red_display is None:
                    return {
                        "status": "failed",
                        "message": "consumer血管红区底图无法读取",
                        "cleanup_paths": cleanup_paths,
                    }
                skin_style = replace(
                    skin_style,
                    vascular_display=red_display,
                )
            algorithm_started_at = time.perf_counter()
            artifacts = self.style_strategy.run_vascular(
                preprocess_result,
                skin_style,
                self.vascular_dir,
                filename,
            )
            results.update({
                "vascular": str(artifacts.overlay),
                "vascular_metrics": str(artifacts.metrics.metrics_json),
                "vascular_report": str(artifacts.metrics.compact_csv),
                "vascular_medical_report_csv_v2": str(artifacts.metrics.medical_csv_v2),
                "vascular_full_metrics": str(artifacts.metrics.full_metrics_json),
            })
            cleanup_paths.append(str(artifacts.overlay.parent))
            algorithm_timings["vascular"] = round(time.perf_counter() - algorithm_started_at, 4)

        if "contour_firmness" in requested_algorithms:
            from src.added_algorithms.runner import run_contour_firmness

            algorithm_started_at = time.perf_counter()
            artifacts = run_contour_firmness(
                preprocess_result,
                self.contour_firmness_dir,
                filename,
            )
            if artifacts is None:
                message = "轮廓紧致度几何QC失败或相对Z不可用"
                timing_seconds["pipeline_total"] = round(
                    time.perf_counter() - pipeline_started_at,
                    4,
                )
                return {
                    "status": "partial_success" if results else "failed",
                    "message": message,
                    "failed_algorithms": {"contour_firmness": message},
                    "results": results,
                    "medical_v2_results": medical_v2_results,
                    "cleanup_paths": cleanup_paths,
                    "metadata": {
                        "algorithms": requested_algorithms,
                        "quality_score": preprocess_result.quality_score,
                        "quality_status": preprocess_result.quality_status,
                        "quality_flags": preprocess_result.quality_flags,
                        "quality_metrics": preprocess_result.quality_metrics,
                        "capture_profile": self.capture_profile.value,
                        "timing_seconds": timing_seconds,
                    },
                }
            results.update({
                "contour_firmness": str(artifacts.overlay),
                "contour_firmness_metrics": str(artifacts.metrics.metrics_json),
                "contour_firmness_report": str(artifacts.metrics.compact_csv),
                "contour_firmness_medical_report_csv_v2": str(artifacts.metrics.medical_csv_v2),
                "contour_firmness_full_metrics": str(artifacts.metrics.full_metrics_json),
            })
            cleanup_paths.append(str(artifacts.overlay.parent))
            algorithm_timings["contour_firmness"] = round(time.perf_counter() - algorithm_started_at, 4)

        print(f"\n🎉 [Celery Task] 处理完成: {image_path}")
        timing_seconds["pipeline_total"] = round(
            time.perf_counter() - pipeline_started_at,
            4,
        )
        cuda_profile = cuda_backend.finish_task_profile(cuda_profile_baseline)
        print(
            "   [CUDA] "
            f"device={cuda_profile['device']} "
            f"gpu_kernel={cuda_profile['task_gpu_kernel_seconds']:.4f}s "
            f"calls={cuda_profile['task_gpu_kernel_calls']} "
            f"peak={cuda_profile['task_peak_memory_mib']:.2f}MiB"
        )
        return {
            "status": "success",
            "results": results,
            "medical_v2_results": medical_v2_results,
            "cleanup_paths": cleanup_paths,
            "metadata": {
                "algorithms": requested_algorithms,
                "quality_score": preprocess_result.quality_score,
                "quality_status": preprocess_result.quality_status,
                "quality_flags": preprocess_result.quality_flags,
                "quality_metrics": preprocess_result.quality_metrics,
                "capture_profile": self.capture_profile.value,
                "timing_seconds": timing_seconds,
            },
        }

    def run_all(self):
        """执行红区批处理；每张原图只预处理一次并以内存结果进入 RBX。"""
        print("╔" + "═" * 68 + "╗")
        print("║  🏥 DermaVision 医疗级面部皮肤检测系统 — 全流程启动" + " ║")
        print("╚" + "═" * 68 + "╝")

        self._validate_environment()

        raw_files = sorted(
            filename
            for filename in os.listdir(self.raw_dir)
            if filename.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        failures = []
        for filename in raw_files:
            image_path = os.path.join(self.raw_dir, filename)
            result = self.process_single(image_path, algorithms=["redness"])
            if result.get("status") != "success":
                failures.append((filename, result.get("message", "UNKNOWN")))
        if failures:
            print("\n⚠️ 以下图片红区处理失败：")
            for filename, message in failures:
                print(f"   - {filename}: {message}")

        print("\n" + "=" * 70)
        print(" 🎉 DermaVision 全流程处理完成！")
        print(f" 📁 结果输出目录: {self.output_dir}")
        print("=" * 70)
