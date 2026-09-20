#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 皱纹分割 V9：人脸居中预处理 + 最大召回 + 医美分区分析 + 无监督量化

目标：
1. 第一阶段：尽可能把所有可能是皱纹的线全部找出来（高召回）。
2. 第二阶段：使用人脸关键点 / 人脸区域分析，自动去掉：
   - 眉毛及眉毛周边
   - 嘴唇内部与嘴唇周边
   - 眼球与眼睑内部
   - 鼻子、鼻翼、鼻孔阴影
   - 鼻孔下方到上唇之间的胡须/人中区域
   - 图片边缘/局部裁剪边缘产生的伪线
3. 所有 YOLO 推理均在“人脸抠出、黑底、正方形居中”的标准化图像上执行。
4. 最终输出一张“先全检出、再人脸后处理筛选”的结果图。

推荐：
    pip install -U ultralytics opencv-python numpy torch
    pip install -U mediapipe

本版本使用新版 MediaPipe Tasks FaceLandmarker + ImageSegmenter API，
不再调用已经从部分新版安装包中移除的 mp.solutions。
ImageSegmenter 用于提取真实面部皮肤，并用“额头补全多边形”修复
FaceLandmarker 人脸椭圆在额头区域覆盖不足的问题。
可选（更好的骨架化）：
    pip install -U opencv-contrib-python

直接运行：
    python main_max_recall_facefilter_v9_fixed.py

本版本强制先做人脸标准化预处理；如果 MediaPipe FaceLandmarker 不可用，
程序会停止，以避免退回整图推理后把背景纹理误识别为皱纹。
"""

from __future__ import annotations

import argparse
import os
import csv
import json
import math
import platform
import sys
import time
import traceback
import shutil
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.wrinkle.runtime_cache import configure_runtime_caches

PROJECT_DIR = Path(__file__).resolve().parent

# 降低 MediaPipe / LiteRT 的非错误日志噪声。某些二进制仍可能打印 XNNPACK 信息，
# 但不影响推理结果。
configure_runtime_caches()
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import torch
import ultralytics
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from src.wrinkle.output_files import (
    OUTPUT_END_REGION_OVERLAY_FILENAME,
    OUTPUT_END_TEXTURE_FILENAME,
    OUTPUT_IMAGE_FILES,
)


_YOLO_MODEL_CACHE: dict[str, YOLO] = {}
_YOLO_MODEL_LOCK = threading.Lock()
_MEDIAPIPE_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_MEDIAPIPE_MODEL_LOCK = threading.Lock()
_CUDA_INITIALIZED = False


# ============================================================================
# 默认配置
# ============================================================================

DEFAULT_WEIGHTS = "best.pt"
DEFAULT_SOURCE = "1000058_1.jpg"
DEFAULT_OUTPUT_DIR = "wrinkle_facefilter_v9_fixed_output"
DEFAULT_FACE_MODEL = "face_landmarker.task"
DEFAULT_SEGMENTER_MODEL = "selfie_multiclass.tflite"

# 所有后续分析统一在该尺寸的“黑底居中人脸”上进行。
DEFAULT_PREPROCESS_SIZE = 1024
PREPROCESS_BACKGROUND_VALUE = 0
PREPROCESS_PAD_X_RATIO = 0.055
PREPROCESS_PAD_TOP_RATIO = 0.085
PREPROCESS_PAD_BOTTOM_RATIO = 0.035
PREPROCESS_MASK_CLOSE_SIZE = 9
PREPROCESS_MASK_DILATE = 2

# 第一阶段 YOLO 结果映射回标准化图后，再向人脸内部收缩一点，
# 彻底屏蔽黑底和人脸边缘形成的伪线。
STAGE1_VALID_FACE_ERODE = 2

FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/"
    "face_landmarker.task"
)

IMAGE_SEGMENTER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/selfie_multiclass_256x256/float32/"
    "latest/selfie_multiclass_256x256.tflite"
)

RAW_CONFIDENCE = 0.01
NMS_IOU = 0.75
MAX_DETECTIONS = 1000

# ---------------------------------------------------------------------------
# 速度 / 运行量控制
# ---------------------------------------------------------------------------
# CPU 线程数。调大：CPU 占用更高，通常更快；太大可能反而抖动。
CPU_THREADS = 6

# 运行预设：
# - fast：最快，推理次数少，适合快速预览。
# - balanced：速度和召回折中，建议日常使用。
# - full：最大召回，最慢；接近你之前 174 次左右的运行规模。
RUN_PRESET = "balanced"

# 是否保留左右翻转推理。
# 打开：召回更高、速度更慢。
ENABLE_HORIZONTAL_FLIP = True

RECOMMENDED_MIN_VOTES = 2
RECOMMENDED_MIN_SCORE = 0.10

LOCAL_BORDER_MARGIN_RATIO = 0.018
LOCAL_BORDER_MARGIN_MIN = 3
LOCAL_BORDER_MARGIN_MAX = 8

REMOVE_OUTER_BORDER_COMPONENTS = True
OUTER_BORDER_MARGIN = 2
OUTER_BORDER_MIN_COMPONENT_SIZE = 12

MIN_CENTERLINE_COMPONENT_SIZE = 2
DEFAULT_PROFILE = "extreme"
SAVE_EACH_RUN = False

# ---------------------------------------------------------------------------
# 人脸后处理参数
# ---------------------------------------------------------------------------
# 眉毛禁区：直接按“像素”向外膨胀。
# 调大：更容易彻底盖住眉毛/眉粉/眉毛阴影，但也更容易误删眉上或眉下真实皮肤。
# 调小：更贴合眉形，但眉毛边缘更容易漏出来。
EYEBROW_DILATE = 8

# 眼睛禁区：眼球、睫毛、上下眼睑内部统一向外膨胀多少像素。
# 调大：更容易盖住睫毛、眼线、双眼皮；但也更可能吃掉眼周真实细纹。
# 调小：保留更多眼周皮肤，但双眼皮和睫毛更容易被识别成皱纹。
EYE_CORE_DILATE = 12

# 嘴唇禁区。调大：更稳妥地排除唇缘；调小：更保留口周细纹。
LIP_EXCLUDE_DILATE = 6

# 鼻子禁区。调大：更稳妥地排除鼻梁、鼻翼、鼻孔阴影；调小：更保留鼻周皮肤。
NOSE_EXCLUDE_DILATE = 4

# 鼻孔下方到上唇之间的“胡须/人中区域”整体屏蔽。
# MUSTACHE_EXCLUDE_DILATE 调大：更大范围去掉人中与上唇上方误检。
MUSTACHE_EXCLUDE_DILATE = 4
MUSTACHE_TOP_PAD_RATIO = 0.08
MUSTACHE_BOTTOM_LIP_RATIO = 0.22

# 从真实皮肤边界向内缩一点，去掉脸与背景/头发交界伪线。
# 调大：边缘更干净，但脸轮廓附近真实细纹更少。
FACE_SHRINK_ERODE = 2

# 眉间纹允许区轻微扩展，但绝不重新放回眉毛或鼻子区域。
GLABELLA_DILATE = 3

# 额头补全：使用太阳穴关键点 234/454 向图像上边界扩展，
# 再与语义皮肤分割相交，避免只保留半个额头。
FOREHEAD_SIDE_PAD_RATIO = 0.08

# 语义分割类别 3 为 face-skin（Selfie Multiclass 模型）。
SEMANTIC_FACE_SKIN_CLASS_ID = 3

# 纹理底图仅作为调试/后续模型输入保存，不参与本次规则过滤。
TEXTURE_BLUR_KERNEL = 35
TEXTURE_CLAHE_CLIP = 1.8

# ---------------------------------------------------------------------------
# 展示 / 出图美化参数
# ---------------------------------------------------------------------------
# 普通候选线和最终皱纹线：只修改这里即可统一调整显示，不影响检测 mask 和量化结果。
DEFAULT_LINE_WIDTH = 1
# 最终皱纹线条颜色（BGR）。与产品示例图中的明亮黄绿色保持一致。
DISPLAY_LINE_COLOR = (54, 255, 148)

# 线条主体透明度。调大：更明显；调小：更柔和。
DISPLAY_LINE_ALPHA = 0.68

# 柔光半径和透明度。恢复适中的柔光，让线条清楚但不过分生硬。
DISPLAY_GLOW_SIGMA = 1.20
DISPLAY_GLOW_ALPHA = 0.14

# 展示线额外膨胀像素。1 表示将 1 像素中心线显示为约 3 像素。
DISPLAY_LINE_DILATE = 1

# ---------------------------------------------------------------------------
# VISIA 风格分区展示参数（全部使用冷色调）
# ---------------------------------------------------------------------------
VISIA_REGION_OUTLINE_COLOR = (222, 197, 65)   # 示例图青蓝色轮廓
VISIA_REGION_OUTLINE_THICKNESS = 2
VISIA_REGION_FILL_COLOR = (255, 240, 120)     # 淡青填充
VISIA_REGION_FILL_ALPHA = 0.10
VISIA_WRINKLE_COLOR = (54, 255, 148)          # 示例图亮绿色皱纹线
VISIA_WRINKLE_ALPHA = 0.92
VISIA_WRINKLE_DILATE = 1
VISIA_LABEL_COLOR = (180, 90, 0)              # 深蓝文字（BGR）
FACE_FILTER_FOREHEAD_OUTLINE_COLOR = (0, 255, 255)
FACE_FILTER_FOREHEAD_OUTLINE_THICKNESS = 1
VISIA_TILE_BG_VALUE = 245
VISIA_TILE_WIDTH = 360
VISIA_TILE_HEIGHT = 300
VISIA_REGION_IMAGE_WIDTH = 360
VISIA_REGION_IMAGE_HEIGHT = 240
VISIA_REGION_IMAGE_MARGIN = 8
VISIA_TOP_REGION_COUNT = 10
VISIA_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/source-han-sans/SourceHanSansSC-Regular.otf",
]
# 其他纹区域：把未归类皱纹线向外膨胀，生成“其他纹”分区。
OTHER_REGION_DILATE = 12
# 分区归属时，不直接按原始区块硬切，而是把每条皱纹线连通域分配到最近/最匹配的命名区域。
# 下面这些像素膨胀值决定“区域吸附范围”。调大：更容易把周边皱纹吸入该区域；调小：更保守。
ASSIGN_DILATE_EYE = 22
ASSIGN_DILATE_GLABELLA = 24
ASSIGN_DILATE_NASOLABIAL = 28
ASSIGN_DILATE_MARIONETTE = 22
ASSIGN_DILATE_FOREHEAD = 20

# 额头纹 vs 眉间纹：在交界处，竖向纹更偏眉间，横向纹更偏额头。
GLABELLA_VERTICAL_BONUS = 40.0
FOREHEAD_HORIZONTAL_BONUS = 40.0
CENTER_GLABELLA_X_RATIO = 0.14  # 中轴附近视作“眉间-额头交界候选区”的宽度占脸宽比例

# 法令纹 vs 木偶纹：上半段更偏法令纹，口角下方更偏木偶纹。
NASOLABIAL_UPPER_BONUS = 24.0
MARIONETTE_LOWER_BONUS = 24.0

# 其他纹兜底阈值。调大：更少进入“其他纹”；调小：更严格，更多进入“其他纹”。
OTHER_FALLBACK_MAX_DIST = 85.0

# 分区统计：最小线段长度（像素）。调大：条数更少但更稳定；调小：更敏感但更碎。
REGION_MIN_SEGMENT_LENGTH = 6

# 调试图中禁用区域颜色
COLOR_FORBIDDEN = (60, 60, 255)   # red-ish
COLOR_ALLOWED = (60, 200, 60)     # green-ish
COLOR_GLABELLA = (255, 170, 0)    # blue-ish in BGR


@dataclass(frozen=True)
class View:
    name: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass(frozen=True)
class FacePreprocessResult:
    """人脸标准化预处理结果。"""

    image: np.ndarray
    display_image: np.ndarray
    face_mask: np.ndarray
    semantic_skin_mask: np.ndarray
    points: np.ndarray
    crop_box: tuple[int, int, int, int]
    square_side: int
    scale: float
    offset_x: int
    offset_y: int
    status: str


@dataclass
class RegionMetric:
    key: str
    display_name: str
    short_name: str
    mask: np.ndarray
    line_mask: np.ndarray
    area_px: int
    wrinkle_pixels: int
    segment_count: int
    mean_segment_length: float
    max_segment_length: int
    density_per_10k: float
    share_pct: float
    raw_score: float
    relative_score: float = 0.0


# ============================================================================
# 基础 IO / 参数
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 皱纹最大召回 + 语义皮肤分割 + 严格医美区域筛选"
    )
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--preprocess-size",
        type=int,
        default=DEFAULT_PREPROCESS_SIZE,
        help="人脸黑底正方形标准化尺寸，建议 1024",
    )
    parser.add_argument(
        "--face-model",
        default=DEFAULT_FACE_MODEL,
        help="MediaPipe Tasks FaceLandmarker 模型路径；缺失时自动下载",
    )
    parser.add_argument(
        "--segmenter-model",
        default=DEFAULT_SEGMENTER_MODEL,
        help="MediaPipe Tasks ImageSegmenter 模型路径；缺失时自动下载",
    )
    parser.add_argument(
        "--no-semantic-skin",
        action="store_true",
        help="禁用语义皮肤分割，只使用关键点额头补全包络（不推荐）",
    )
    parser.add_argument(
        "--profile",
        choices=("full", "extreme"),
        default=DEFAULT_PROFILE,
        help="full 只做整脸多尺度；extreme 增加额头、眼周和中下脸局部推理",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto、cpu、0、1、cuda:0 等",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=CPU_THREADS,
        help="CPU 线程数。仅对 CPU 推理与 OpenCV/PyTorch 底层线程有效。",
    )
    parser.add_argument(
        "--run-preset",
        choices=("fast", "balanced", "full"),
        default=RUN_PRESET,
        help="推理运行量预设：fast / balanced / full",
    )
    parser.add_argument(
        "--raw-conf",
        type=float,
        default=RAW_CONFIDENCE,
        help="最低候选置信度，越低召回越高、误检越多",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=DEFAULT_LINE_WIDTH,
        choices=(1, 2, 3),
        help="最终显示线宽，建议 1",
    )
    parser.add_argument(
        "--save-runs",
        action="store_true",
        help="保存每一次推理结果，会产生很多文件",
    )
    parser.add_argument(
        "--report-group-overlays",
        action="store_true",
        help="额外生成07～09报告专用旧版分区图；不改变检测和公开结果",
    )
    return parser.parse_args()


def choose_device(requested: str) -> str:
    global _CUDA_INITIALIZED
    normalized = str(requested).strip().lower()
    if normalized in {"auto", "0", "cuda", "cuda:0"}:
        normalized = "0"
    else:
        raise RuntimeError(
            f"皱纹生产推理严格要求 cuda:0，收到设备参数：{requested!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；皱纹 Worker 已禁用静默 CPU 回退")
    if not _CUDA_INITIALIZED:
        torch.cuda.set_device(0)
        fraction = float(os.getenv("WRINKLE_GPU_MEMORY_FRACTION", "0.29"))
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        _CUDA_INITIALIZED = True
    return normalized


def load_yolo_once(weights_path: Path) -> YOLO:
    """Cache one Ultralytics model per weight file in each Worker process."""
    key = str(weights_path.resolve())
    with _YOLO_MODEL_LOCK:
        model = _YOLO_MODEL_CACHE.get(key)
        if model is None:
            model = YOLO(key)
            _YOLO_MODEL_CACHE[key] = model
        return model


def _load_segmenter_once(model_path: Path):
    key = ("segmenter", str(model_path.resolve()))
    with _MEDIAPIPE_MODEL_LOCK:
        cached = _MEDIAPIPE_MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        mp, mp_python, vision, error = try_import_mediapipe_tasks()
        if mp is None or mp_python is None or vision is None:
            raise RuntimeError(f"MediaPipe Tasks API 不可用：{error}")
        resolved_model, status = ensure_image_segmenter_model(model_path)
        if resolved_model is None:
            raise RuntimeError(status)
        try:
            options = vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(resolved_model)),
                running_mode=vision.RunningMode.IMAGE,
                output_category_mask=True,
                output_confidence_masks=False,
            )
        except TypeError:
            options = vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(resolved_model)),
                output_category_mask=True,
                output_confidence_masks=False,
            )
        cached = vision.ImageSegmenter.create_from_options(options)
        _MEDIAPIPE_MODEL_CACHE[key] = cached
        return cached


def _load_landmarker_once(model_path: Path):
    key = ("landmarker", str(model_path.resolve()))
    with _MEDIAPIPE_MODEL_LOCK:
        cached = _MEDIAPIPE_MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        mp, mp_python, vision, error = try_import_mediapipe_tasks()
        if mp is None or mp_python is None or vision is None:
            raise RuntimeError(f"MediaPipe Tasks API 不可用：{error}")
        resolved_model, status = ensure_face_landmarker_model(model_path)
        if resolved_model is None:
            raise RuntimeError(status)
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(resolved_model)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.35,
            min_face_presence_confidence=0.35,
            min_tracking_confidence=0.35,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        cached = vision.FaceLandmarker.create_from_options(options)
        _MEDIAPIPE_MODEL_CACHE[key] = cached
        return cached


def warmup_runtime(
    weights_path: Path,
    face_model_path: Path,
    segmenter_model_path: Path,
    device: str,
) -> dict[str, Any]:
    """每个 Celery 子进程一次性加载 CUDA、YOLO 和 MediaPipe。"""
    resolved_device = choose_device(device)
    _load_landmarker_once(face_model_path)
    _load_segmenter_once(segmenter_model_path)
    model = load_yolo_once(weights_path)
    safe_predict(
        model,
        np.zeros((640, 640, 3), dtype=np.uint8),
        imgsz=640,
        conf=0.01,
        device=resolved_device,
    )
    return {
        "device": resolved_device,
        "weights": str(weights_path),
        "mediapipe_cached": True,
        "yolo_cached": True,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
    }


def close_runtime_models() -> None:
    with _MEDIAPIPE_MODEL_LOCK:
        for model in _MEDIAPIPE_MODEL_CACHE.values():
            close = getattr(model, "close", None)
            if callable(close):
                close()
        _MEDIAPIPE_MODEL_CACHE.clear()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_DIR / path).resolve()


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV 无法读取图片：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"无法编码图片：{path}")
    encoded.tofile(str(path))


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}

def configure_cpu_threads(num_threads: int) -> int:
    """设置 OpenCV / PyTorch / 常见 BLAS 的 CPU 线程数。"""
    threads = max(1, int(num_threads))
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(threads)
    try:
        cv2.setNumThreads(threads)
    except Exception:
        pass
    try:
        torch.set_num_threads(threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, min(2, threads)))
    except Exception:
        pass
    return threads


# ============================================================================
# 推理区域与增强
# ============================================================================

def build_views(width: int, height: int, profile: str) -> list[View]:
    views = [View("full_face", 0, 0, width, height)]

    if profile != "extreme":
        return views

    def make_view(
        name: str,
        left: float,
        top: float,
        right: float,
        bottom: float,
    ) -> View:
        x1 = max(0, min(width - 1, int(round(left * width))))
        y1 = max(0, min(height - 1, int(round(top * height))))
        x2 = max(x1 + 1, min(width, int(round(right * width))))
        y2 = max(y1 + 1, min(height, int(round(bottom * height))))
        return View(name, x1, y1, x2, y2)

    views.extend(
        [
            make_view("forehead", 0.06, 0.00, 0.94, 0.39),
            make_view("eye_band", 0.01, 0.20, 0.99, 0.62),
            make_view("left_eye", 0.00, 0.21, 0.58, 0.61),
            make_view("right_eye", 0.42, 0.21, 1.00, 0.61),
            make_view("mid_lower_face", 0.03, 0.40, 0.97, 0.98),
        ]
    )
    return views


def apply_clahe(image: np.ndarray, clip_limit: float) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8),
    )
    enhanced = clahe.apply(lightness)
    return cv2.cvtColor(
        cv2.merge((enhanced, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    table = np.array(
        [
            np.clip(((index / 255.0) ** gamma) * 255.0, 0, 255)
            for index in range(256)
        ],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def apply_sharpen(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(image, 1.55, blurred, -0.55, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_blackhat_enhancement(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15),
    )
    blackhat = cv2.morphologyEx(
        lightness,
        cv2.MORPH_BLACKHAT,
        kernel,
    )

    enhanced_lightness = np.clip(
        lightness.astype(np.float32)
        - 0.65 * blackhat.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    return cv2.cvtColor(
        cv2.merge((enhanced_lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )


def make_variants(
    image: np.ndarray,
    is_full_face: bool,
    run_preset: str,
) -> dict[str, np.ndarray]:
    """
    根据运行预设生成增强版本。
    fast：最少版本；
    balanced：日常推荐；
    full：最大召回。
    """
    variants: dict[str, np.ndarray] = {
        "original": image,
        "clahe_soft": apply_clahe(image, clip_limit=1.5),
    }

    if run_preset in ("balanced", "full"):
        variants["clahe_strong"] = apply_clahe(image, clip_limit=2.4)

    if is_full_face and run_preset in ("balanced", "full"):
        variants["sharpen"] = apply_sharpen(image)

    if is_full_face and run_preset == "full":
        variants["gamma_bright"] = apply_gamma(image, gamma=0.78)
        variants["gamma_dark"] = apply_gamma(image, gamma=1.22)
        variants["blackhat"] = apply_blackhat_enhancement(image)

    return variants


def get_scales(is_full_face: bool, run_preset: str) -> list[int]:
    """
    返回当前预设下的推理尺寸。
    调大尺寸：更容易检出细纹，但速度更慢。
    """
    if run_preset == "fast":
        return [768, 1024] if is_full_face else [640]

    if run_preset == "balanced":
        return [768, 1024, 1280] if is_full_face else [640, 896]

    # full
    return [640, 768, 896, 1024, 1280, 1536] if is_full_face else [640, 896, 1024]


def get_flip_options(run_preset: str) -> list[bool]:
    """
    fast 模式默认不做翻转；
    balanced / full 若 ENABLE_HORIZONTAL_FLIP 为 True，则做翻转。
    """
    if run_preset == "fast":
        return [False]

    return [False, True] if ENABLE_HORIZONTAL_FLIP else [False]


def estimate_total_runs(views: list[View], run_preset: str) -> int:
    total = 0
    for view in views:
        is_full_face = view.name == "full_face"
        total += (
            len(make_variants(np.zeros((8, 8, 3), dtype=np.uint8), is_full_face, run_preset))
            * len(get_scales(is_full_face, run_preset))
            * len(get_flip_options(run_preset))
        )
    return total


# ============================================================================
# YOLO 推理
# ============================================================================

def safe_predict(
    model: YOLO,
    image: np.ndarray,
    *,
    imgsz: int,
    conf: float,
    device: str,
):
    return model.predict(
        source=image,
        imgsz=imgsz,
        conf=conf,
        iou=NMS_IOU,
        max_det=MAX_DETECTIONS,
        device=device,
        retina_masks=True,
        verbose=False,
    )


def extract_run_maps(
    result: Any,
    target_height: int,
    target_width: int,
    class_ids: list[int],
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[int, np.ndarray],
    int,
]:
    binary_union = np.zeros(
        (target_height, target_width),
        dtype=np.uint8,
    )
    score_map = np.zeros(
        (target_height, target_width),
        dtype=np.float32,
    )
    class_unions = {
        class_id: np.zeros(
            (target_height, target_width),
            dtype=np.uint8,
        )
        for class_id in class_ids
    }

    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or boxes is None or len(boxes) == 0:
        return binary_union, score_map, class_unions, 0

    mask_data = masks.data.detach().float().cpu().numpy()
    confidences = boxes.conf.detach().float().cpu().numpy()
    predicted_classes = boxes.cls.detach().cpu().numpy().astype(int)

    instance_count = 0

    for mask, confidence, class_id in zip(
        mask_data,
        confidences,
        predicted_classes,
    ):
        if mask.shape != (target_height, target_width):
            mask = cv2.resize(
                mask,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )

        binary = mask > 0.5
        if not np.any(binary):
            continue

        instance_count += 1
        binary_union[binary] = 1
        score_map[binary] = np.maximum(
            score_map[binary],
            float(confidence),
        )

        class_id = int(class_id)
        if class_id not in class_unions:
            class_unions[class_id] = np.zeros(
                (target_height, target_width),
                dtype=np.uint8,
            )
        class_unions[class_id][binary] = 1

    return binary_union, score_map, class_unions, instance_count


def clear_local_crop_border(
    binary_map: np.ndarray,
    score_map: np.ndarray,
    class_maps: dict[int, np.ndarray],
    view_name: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[int, np.ndarray],
    int,
]:
    if view_name == "full_face":
        return binary_map, score_map, class_maps, 0

    height, width = binary_map.shape
    margin = int(
        round(min(height, width) * LOCAL_BORDER_MARGIN_RATIO)
    )
    margin = max(
        LOCAL_BORDER_MARGIN_MIN,
        min(LOCAL_BORDER_MARGIN_MAX, margin),
    )
    margin = min(
        margin,
        max(1, min(height, width) // 4),
    )

    binary_map = binary_map.copy()
    score_map = score_map.copy()
    class_maps = {
        class_id: mask.copy()
        for class_id, mask in class_maps.items()
    }

    for array in [binary_map, score_map]:
        array[:margin, :] = 0
        array[-margin:, :] = 0
        array[:, :margin] = 0
        array[:, -margin:] = 0

    for mask in class_maps.values():
        mask[:margin, :] = 0
        mask[-margin:, :] = 0
        mask[:, :margin] = 0
        mask[:, -margin:] = 0

    return binary_map, score_map, class_maps, margin


def map_to_full_image(
    local_map: np.ndarray,
    view: View,
    full_height: int,
    full_width: int,
) -> np.ndarray:
    full_map = np.zeros(
        (full_height, full_width),
        dtype=local_map.dtype,
    )

    resized = local_map
    if local_map.shape != (view.height, view.width):
        interpolation = (
            cv2.INTER_NEAREST
            if np.issubdtype(local_map.dtype, np.integer)
            else cv2.INTER_LINEAR
        )
        resized = cv2.resize(
            local_map,
            (view.width, view.height),
            interpolation=interpolation,
        )

    full_map[view.y1 : view.y2, view.x1 : view.x2] = resized
    return full_map


# ============================================================================
# Skeleton / 连通域
# ============================================================================

def morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    working = (mask > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(working)

    element = cv2.getStructuringElement(
        cv2.MORPH_CROSS,
        (3, 3),
    )

    while True:
        opened = cv2.morphologyEx(
            working,
            cv2.MORPH_OPEN,
            element,
        )
        residue = cv2.subtract(working, opened)
        eroded = cv2.erode(working, element)
        skeleton = cv2.bitwise_or(skeleton, residue)
        working = eroded

        if cv2.countNonZero(working) == 0:
            break

    return skeleton


def skeletonize(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255

    ximgproc = getattr(cv2, "ximgproc", None)
    thinning = getattr(ximgproc, "thinning", None)

    if thinning is not None:
        try:
            return thinning(
                binary,
                thinningType=getattr(
                    cv2.ximgproc,
                    "THINNING_ZHANGSUEN",
                    0,
                ),
            )
        except (cv2.error, TypeError, AttributeError):
            pass

    return morphological_skeleton(binary)


def remove_small_components(
    mask: np.ndarray,
    min_size: int,
) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    output = np.zeros_like(binary)

    for component_id in range(1, count):
        size = int(stats[component_id, cv2.CC_STAT_AREA])
        if size >= min_size:
            output[labels == component_id] = 255

    return output


def remove_outer_border_components(
    mask: np.ndarray,
    margin: int,
    min_component_size: int,
) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    output = np.zeros_like(binary)
    for component_id in range(1, count):
        size = int(stats[component_id, cv2.CC_STAT_AREA])
        component = labels == component_id
        touches_border = bool(
            np.any(component[:margin, :])
            or np.any(component[-margin:, :])
            or np.any(component[:, :margin])
            or np.any(component[:, -margin:])
        )

        if touches_border and size >= min_component_size:
            continue

        output[component] = 255

    return output


def make_centerline(mask: np.ndarray) -> np.ndarray:
    centerline = skeletonize(mask)
    centerline = remove_small_components(
        centerline,
        min_size=MIN_CENTERLINE_COMPONENT_SIZE,
    )
    return centerline


def set_line_width(
    centerline: np.ndarray,
    line_width: int,
) -> np.ndarray:
    if line_width <= 1:
        return centerline

    kernel_size = 2 * line_width - 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    return cv2.dilate(
        centerline,
        kernel,
        iterations=1,
    )


# ============================================================================
# MediaPipe 人脸关键点后处理
# ============================================================================

def try_import_mediapipe_tasks():
    """
    导入新版 MediaPipe Tasks API。

    新版安装包中 mp.solutions 可能不存在，但官方推荐的 Tasks API
    位于 mediapipe.tasks。
    """
    try:
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks import python as mp_python  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore

        try:
            from absl import logging as absl_logging  # type: ignore
            absl_logging.set_verbosity(absl_logging.ERROR)
        except Exception:
            pass

        return mp, mp_python, vision, None
    except Exception as exc:
        return None, None, None, str(exc)


# MediaPipe FaceLandmarker 输出 478 个关键点。
# 这里直接保存需要的人脸区域索引，避免依赖旧版
# mp.solutions.face_mesh_connections。
FACE_REGION_INDICES: dict[str, list[int]] = {
    "face_oval": [
        10, 338, 297, 332, 284, 251, 389, 356, 454,
        323, 361, 288, 397, 365, 379, 378, 400, 377,
        152, 148, 176, 149, 150, 136, 172, 58, 132,
        93, 234, 127, 162, 21, 54, 103, 67, 109,
    ],
    "right_eye": [
        33, 7, 163, 144, 145, 153, 154, 155,
        133, 173, 157, 158, 159, 160, 161, 246,
        469, 470, 471, 472,
    ],
    "left_eye": [
        263, 249, 390, 373, 374, 380, 381, 382,
        362, 398, 384, 385, 386, 387, 388, 466,
        474, 475, 476, 477,
    ],
    "right_eyebrow": [
        46, 53, 52, 65, 55, 70, 63, 105, 66, 107,
    ],
    "left_eyebrow": [
        276, 283, 282, 295, 285, 300, 293, 334,
        296, 336,
    ],
    "lips": [
        61, 146, 91, 181, 84, 17, 314, 405, 321,
        375, 291, 308, 324, 318, 402, 317, 14, 87,
        178, 88, 95, 78, 191, 80, 81, 82, 13, 312,
        311, 310, 415,
    ],
    # 鼻梁、鼻尖、鼻翼与鼻孔周围；用于“整个鼻子禁检”。
    "nose": [
        168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164,
        45, 220, 115, 48, 64, 98, 97,
        326, 327, 294, 278, 344, 440, 275,
    ],
    # 下鼻部/鼻翼，构建鼻孔下方到上唇的胡须区。
    "nose_lower": [
        45, 220, 115, 48, 64, 98, 97, 2,
        326, 327, 294, 278, 344, 440, 275,
    ],
}


def ensure_face_landmarker_model(model_path: Path) -> tuple[Path | None, str]:
    """
    确保新版 FaceLandmarker 模型存在。

    首次运行会从 Google 官方模型地址下载 face_landmarker.task。
    下载失败时不让整个程序崩溃，而是返回可读错误，让第一阶段结果
    仍能正常保存。
    """
    model_path = model_path.expanduser().resolve()

    if model_path.is_file() and model_path.stat().st_size > 1_000_000:
        return model_path, "ok"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = model_path.with_suffix(model_path.suffix + ".part")

    try:
        print("⬇ 正在下载 MediaPipe FaceLandmarker 官方模型...")
        print(f"  {FACE_LANDMARKER_MODEL_URL}")

        request = urllib.request.Request(
            FACE_LANDMARKER_MODEL_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        with urllib.request.urlopen(request, timeout=120) as response:
            with temp_path.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)

        if temp_path.stat().st_size <= 1_000_000:
            raise RuntimeError(
                f"模型文件异常小：{temp_path.stat().st_size} bytes"
            )

        temp_path.replace(model_path)
        print(f"✅ FaceLandmarker 模型已保存：{model_path}")
        return model_path, "ok"

    except Exception as exc:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass

        return None, (
            "FaceLandmarker 模型不存在且自动下载失败："
            f"{exc}。可手动下载后放到：{model_path}"
        )



def ensure_image_segmenter_model(
    model_path: Path,
) -> tuple[Path | None, str]:
    """确保 Selfie Multiclass ImageSegmenter 模型存在。"""
    model_path = model_path.expanduser().resolve()

    if model_path.is_file() and model_path.stat().st_size > 100_000:
        return model_path, "ok"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = model_path.with_suffix(model_path.suffix + ".part")

    try:
        print("⬇ 正在下载 MediaPipe Selfie Multiclass 分割模型...")
        print(f"  {IMAGE_SEGMENTER_MODEL_URL}")

        request = urllib.request.Request(
            IMAGE_SEGMENTER_MODEL_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        with urllib.request.urlopen(request, timeout=120) as response:
            with temp_path.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)

        if temp_path.stat().st_size <= 100_000:
            raise RuntimeError(
                f"分割模型文件异常小：{temp_path.stat().st_size} bytes"
            )

        temp_path.replace(model_path)
        print(f"✅ ImageSegmenter 模型已保存：{model_path}")
        return model_path, "ok"

    except Exception as exc:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass

        return None, (
            "ImageSegmenter 模型不存在且自动下载失败："
            f"{exc}。可手动下载后放到：{model_path}"
        )

def filtered_region_indices(
    number_of_landmarks: int,
) -> dict[str, list[int]]:
    """
    过滤超出当前模型关键点数量的索引。
    通常新版模型输出 478 点。
    """
    return {
        name: [
            index
            for index in indices
            if 0 <= index < number_of_landmarks
        ]
        for name, indices in FACE_REGION_INDICES.items()
    }


def mask_from_indices(
    points: np.ndarray,
    indices: list[int],
    shape: tuple[int, int],
    dilate_px: int = 0,
    erode_px: int = 0,
) -> np.ndarray:
    height, width = shape
    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    if not indices:
        return mask

    coords = points[indices]
    if len(coords) < 3:
        return mask

    hull = cv2.convexHull(coords.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilate_px + 1, 2 * dilate_px + 1),
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

    if erode_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * erode_px + 1, 2 * erode_px + 1),
        )
        mask = cv2.erode(mask, kernel, iterations=1)

    return mask


def rect_mask(
    shape: tuple[int, int],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    dilate_px: int = 0,
) -> np.ndarray:
    height, width = shape
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2 - 1, y2 - 1), 255, -1)

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilate_px + 1, 2 * dilate_px + 1),
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask



def polygon_mask(
    shape: tuple[int, int],
    polygon: np.ndarray,
    dilate_px: int = 0,
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)

    polygon = np.asarray(polygon, dtype=np.int32)
    if polygon.ndim != 2 or polygon.shape[0] < 3:
        return mask

    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [polygon], 255)

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilate_px + 1, 2 * dilate_px + 1),
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def expanded_bbox_mask(
    points: np.ndarray,
    shape: tuple[int, int],
    *,
    pad_x_ratio: float,
    pad_top_ratio: float,
    pad_bottom_ratio: float,
    min_pad_x: int,
    min_pad_y: int,
) -> np.ndarray:
    """用扩展矩形覆盖关键点凸包可能漏掉的毛发边缘。"""
    height, width = shape
    x, y, box_w, box_h = cv2.boundingRect(points.astype(np.int32))

    pad_x = max(min_pad_x, int(round(box_w * pad_x_ratio)))
    pad_top = max(min_pad_y, int(round(box_h * pad_top_ratio)))
    pad_bottom = max(min_pad_y, int(round(box_h * pad_bottom_ratio)))

    return rect_mask(
        (height, width),
        x - pad_x,
        y - pad_top,
        x + box_w + pad_x,
        y + box_h + pad_bottom,
    )


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    if count <= 1:
        return binary * 255

    largest_id = 1 + int(
        np.argmax(stats[1:, cv2.CC_STAT_AREA])
    )
    output = np.zeros_like(binary)
    output[labels == largest_id] = 255
    return output


def build_extended_face_envelope(
    points: np.ndarray,
    regions: dict[str, list[int]],
    shape: tuple[int, int],
) -> np.ndarray:
    """
    关键点 face oval + 额头补全“烟囱法”。

    FaceLandmarker 的 face oval 在部分照片中会把额头覆盖得偏低。
    用左右太阳穴点 234 / 454 向图像上边界扩展，然后再与语义皮肤
    分割相交，额头不会被切成一半，同时也不会把头发当成皮肤。
    """
    height, width = shape

    envelope = mask_from_indices(
        points,
        regions["face_oval"],
        shape,
    )

    if len(points) > 454:
        temple_a = points[234]
        temple_b = points[454]

        left_pt, right_pt = sorted(
            [temple_a, temple_b],
            key=lambda p: int(p[0]),
        )

        face_pts = points[regions["face_oval"]]
        _, _, face_w, _ = cv2.boundingRect(face_pts)
        side_pad = max(
            8,
            int(round(face_w * FOREHEAD_SIDE_PAD_RATIO)),
        )

        forehead_polygon = np.array(
            [
                [int(left_pt[0]), int(left_pt[1])],
                [max(0, int(left_pt[0]) - side_pad), 0],
                [min(width - 1, int(right_pt[0]) + side_pad), 0],
                [int(right_pt[0]), int(right_pt[1])],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(
            envelope,
            forehead_polygon,
            255,
        )

    return envelope


def detect_semantic_face_skin(
    image: np.ndarray,
    model_path: Path,
) -> tuple[np.ndarray | None, str]:
    """使用新版 MediaPipe Tasks ImageSegmenter 提取 face-skin 类别。"""
    mp, mp_python, vision, error = try_import_mediapipe_tasks()

    if mp is None or mp_python is None or vision is None:
        return None, f"MediaPipe Tasks API 不可用：{error}"

    resolved_model, status = ensure_image_segmenter_model(model_path)
    if resolved_model is None:
        return None, status

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)

    try:
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        segmenter = _load_segmenter_once(resolved_model)
        result = segmenter.segment(mp_image)
        if result.category_mask is None:
            return None, "ImageSegmenter 没有返回 category_mask"
        category = np.array(
            result.category_mask.numpy_view(),
            copy=True,
        )

    except Exception as exc:
        return (
            None,
            "MediaPipe ImageSegmenter 运行失败："
            f"{type(exc).__name__}: {exc}",
        )

    height, width = image.shape[:2]
    if category.shape[:2] != (height, width):
        category = cv2.resize(
            category,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    skin = (
        category == SEMANTIC_FACE_SKIN_CLASS_ID
    ).astype(np.uint8) * 255

    # 小范围闭运算，填平分割模型在额头/脸颊上的细小孔洞。
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )
    skin = cv2.morphologyEx(
        skin,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return skin, (
        "ok: MediaPipe Tasks ImageSegmenter, "
        f"face_skin_class={SEMANTIC_FACE_SKIN_CLASS_ID}"
    )


def generate_texture_reference(
    image: np.ndarray,
    valid_skin_mask: np.ndarray,
) -> np.ndarray:
    """
    复用用户旧工程中的高斯高通 + CLAHE 思路，生成纹理参考图。

    该图只保存作观察/未来接入 FFHQ-Wrinkle 等四通道模型，
    本次后处理不拿它做硬过滤，避免把毛孔、胡须和阴影进一步放大。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel_size = int(TEXTURE_BLUR_KERNEL)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, kernel_size)

    baseline = cv2.GaussianBlur(
        gray,
        (kernel_size, kernel_size),
        0,
    )
    texture_raw = cv2.subtract(baseline, gray)
    texture_norm = cv2.normalize(
        texture_raw,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    clahe = cv2.createCLAHE(
        clipLimit=TEXTURE_CLAHE_CLIP,
        tileGridSize=(8, 8),
    )
    texture = clahe.apply(texture_norm)

    return cv2.bitwise_and(
        texture,
        texture,
        mask=valid_skin_mask,
    )



def detect_face_landmarks(
    image: np.ndarray,
    model_path: Path,
) -> tuple[np.ndarray | None, dict[str, list[int]] | None, str]:
    """
    使用新版 MediaPipe Tasks FaceLandmarker 检测静态图片。

    官方新版流程：
    BaseOptions(model_asset_path=...)
    -> FaceLandmarkerOptions(running_mode=IMAGE)
    -> FaceLandmarker.create_from_options(...)
    -> landmarker.detect(mp.Image)
    """
    mp, mp_python, vision, error = try_import_mediapipe_tasks()

    if mp is None or mp_python is None or vision is None:
        return (
            None,
            None,
            "新版 MediaPipe Tasks API 不可用："
            f"{error}。请执行：python -m pip install -U mediapipe",
        )

    resolved_model, model_status = ensure_face_landmarker_model(
        model_path
    )
    if resolved_model is None:
        return None, None, model_status

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    height, width = image.shape[:2]
    mapping_width = width
    mapping_height = height
    offset_x = 0
    offset_y = 0
    used_padding_fallback = False

    try:
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )
        landmarker = _load_landmarker_once(resolved_model)
        results = landmarker.detect(mp_image)
        if not getattr(results, "face_landmarks", None):
            pad_x = max(24, int(round(width * 0.25)))
            pad_y = max(24, int(round(height * 0.25)))
            padded = cv2.copyMakeBorder(
                image,
                pad_y,
                pad_y,
                pad_x,
                pad_x,
                cv2.BORDER_REFLECT_101,
            )
            padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            padded_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(padded_rgb),
            )
            results = landmarker.detect(padded_image)
            mapping_height, mapping_width = padded.shape[:2]
            offset_x = pad_x
            offset_y = pad_y
            used_padding_fallback = True

    except Exception as exc:
        return (
            None,
            None,
            "MediaPipe Tasks FaceLandmarker 运行失败："
            f"{type(exc).__name__}: {exc}",
        )

    if not getattr(results, "face_landmarks", None):
        return None, None, "新版 FaceLandmarker 未检测到人脸关键点"

    landmarks = results.face_landmarks[0]
    points = np.array(
        [
            [
                int(
                    np.clip(
                        round(float(point.x) * (mapping_width - 1)) - offset_x,
                        0,
                        width - 1,
                    )
                ),
                int(
                    np.clip(
                        round(float(point.y) * (mapping_height - 1)) - offset_y,
                        0,
                        height - 1,
                    )
                ),
            ]
            for point in landmarks
        ],
        dtype=np.int32,
    )

    if len(points) < 468:
        return (
            None,
            None,
            f"FaceLandmarker 只返回了 {len(points)} 个点，"
            "少于所需的 468 个点",
        )

    regions = filtered_region_indices(len(points))

    for required_name in (
        "left_eye",
        "right_eye",
        "left_eyebrow",
        "right_eyebrow",
        "lips",
        "nose",
        "nose_lower",
        "face_oval",
    ):
        if len(regions.get(required_name, [])) < 3:
            return (
                None,
                None,
                f"人脸区域索引无效：{required_name}",
            )

    return (
        points,
        regions,
        (
            "ok: MediaPipe Tasks FaceLandmarker reflected-padding fallback, "
            f"landmarks={len(points)}"
            if used_padding_fallback
            else f"ok: MediaPipe Tasks FaceLandmarker, landmarks={len(points)}"
        ),
    )



def fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    """
    填充人脸皮肤 Mask 内部的眼睛、眉毛、嘴唇等孔洞，
    但不扩张外轮廓。这样标准化人脸仍保留完整五官图像。
    """
    binary = (mask > 0).astype(np.uint8) * 255

    padded = cv2.copyMakeBorder(
        binary,
        1,
        1,
        1,
        1,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    flood = padded.copy()
    flood_mask = np.zeros(
        (padded.shape[0] + 2, padded.shape[1] + 2),
        dtype=np.uint8,
    )
    cv2.floodFill(
        flood,
        flood_mask,
        (0, 0),
        255,
    )
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(padded, holes)
    return filled[1:-1, 1:-1]


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(
        mask,
        size,
        interpolation=cv2.INTER_NEAREST,
    )


def normalize_face_to_square(
    image: np.ndarray,
    points: np.ndarray,
    regions: dict[str, list[int]],
    semantic_skin_mask: np.ndarray | None,
    target_size: int,
) -> FacePreprocessResult:
    """
    将原图转换为统一的黑底正方形人脸图。

    流程：
    1. 语义皮肤 Mask 与关键点额头补全包络相交；
    2. 保留最大人脸区域并填充五官内部孔洞；
    3. 按人脸外轮廓裁剪；
    4. 放入黑色正方形画布中央；
    5. 缩放到 target_size；
    6. 同步变换关键点和所有后处理 Mask。

    后续 YOLO 推理、区域裁剪、后处理和可视化全部使用该标准化图。
    """
    if target_size < 256:
        raise ValueError("--preprocess-size 不能小于 256")

    height, width = image.shape[:2]
    shape = (height, width)

    extended_envelope = build_extended_face_envelope(
        points,
        regions,
        shape,
    )

    if semantic_skin_mask is not None:
        semantic = (
            semantic_skin_mask > 0
        ).astype(np.uint8) * 255
        if semantic.shape != shape:
            semantic = resize_mask(
                semantic,
                (width, height),
            )

        outer_skin = cv2.bitwise_and(
            semantic,
            extended_envelope,
        )
        outer_skin = keep_largest_component(outer_skin)
    else:
        semantic = extended_envelope.copy()
        outer_skin = extended_envelope.copy()

    close_size = max(3, int(PREPROCESS_MASK_CLOSE_SIZE))
    if close_size % 2 == 0:
        close_size += 1

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_size, close_size),
    )
    outer_skin = cv2.morphologyEx(
        outer_skin,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    # 填充皮肤区域内部的眼、眉、鼻孔、嘴唇空洞，
    # 得到完整的人脸可视外轮廓；外部背景仍为 0。
    face_silhouette = fill_internal_holes(outer_skin)
    face_silhouette = cv2.bitwise_and(
        face_silhouette,
        extended_envelope,
    )

    if PREPROCESS_MASK_DILATE > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                2 * PREPROCESS_MASK_DILATE + 1,
                2 * PREPROCESS_MASK_DILATE + 1,
            ),
        )
        face_silhouette = cv2.dilate(
            face_silhouette,
            kernel,
            iterations=1,
        )
        face_silhouette = cv2.bitwise_and(
            face_silhouette,
            extended_envelope,
        )

    ys, xs = np.where(face_silhouette > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("无法从语义皮肤分割和关键点构建有效人脸 Mask")

    x_min = int(xs.min())
    x_max = int(xs.max()) + 1
    y_min = int(ys.min())
    y_max = int(ys.max()) + 1

    face_w = max(1, x_max - x_min)
    face_h = max(1, y_max - y_min)

    pad_x = int(round(face_w * PREPROCESS_PAD_X_RATIO))
    pad_top = int(round(face_h * PREPROCESS_PAD_TOP_RATIO))
    pad_bottom = int(round(face_h * PREPROCESS_PAD_BOTTOM_RATIO))

    crop_x1 = max(0, x_min - pad_x)
    crop_x2 = min(width, x_max + pad_x)
    crop_y1 = max(0, y_min - pad_top)
    crop_y2 = min(height, y_max + pad_bottom)

    crop_image = image[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ]
    crop_face_mask = face_silhouette[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ]
    crop_semantic = semantic[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ]

    crop_h, crop_w = crop_image.shape[:2]
    square_side = max(crop_h, crop_w)

    background = np.full(
        (square_side, square_side, 3),
        PREPROCESS_BACKGROUND_VALUE,
        dtype=np.uint8,
    )
    display_background = np.full(
        (square_side, square_side, 3),
        PREPROCESS_BACKGROUND_VALUE,
        dtype=np.uint8,
    )
    square_face_mask = np.zeros(
        (square_side, square_side),
        dtype=np.uint8,
    )
    square_semantic = np.zeros(
        (square_side, square_side),
        dtype=np.uint8,
    )

    offset_x = (square_side - crop_w) // 2
    offset_y = (square_side - crop_h) // 2

    masked_crop = np.full_like(
        crop_image,
        PREPROCESS_BACKGROUND_VALUE,
    )
    valid = crop_face_mask > 0
    masked_crop[valid] = crop_image[valid]

    background[
        offset_y:offset_y + crop_h,
        offset_x:offset_x + crop_w,
    ] = masked_crop

    # 给用户展示更友好的“带背景裁剪图”：仍然是居中正方形，
    # 但保留裁剪区域背景，避免只剩悬空人脸造成惊吓。
    display_background[
        offset_y:offset_y + crop_h,
        offset_x:offset_x + crop_w,
    ] = crop_image
    square_face_mask[
        offset_y:offset_y + crop_h,
        offset_x:offset_x + crop_w,
    ] = crop_face_mask
    square_semantic[
        offset_y:offset_y + crop_h,
        offset_x:offset_x + crop_w,
    ] = crop_semantic

    normalized_image = cv2.resize(
        background,
        (target_size, target_size),
        interpolation=cv2.INTER_CUBIC,
    )
    normalized_display_image = cv2.resize(
        display_background,
        (target_size, target_size),
        interpolation=cv2.INTER_CUBIC,
    )
    normalized_face_mask = resize_mask(
        square_face_mask,
        (target_size, target_size),
    )
    normalized_semantic = resize_mask(
        square_semantic,
        (target_size, target_size),
    )

    scale = target_size / float(square_side)
    normalized_points = points.astype(np.float32).copy()
    normalized_points[:, 0] = (
        normalized_points[:, 0] - crop_x1 + offset_x
    ) * scale
    normalized_points[:, 1] = (
        normalized_points[:, 1] - crop_y1 + offset_y
    ) * scale
    normalized_points[:, 0] = np.clip(
        normalized_points[:, 0],
        0,
        target_size - 1,
    )
    normalized_points[:, 1] = np.clip(
        normalized_points[:, 1],
        0,
        target_size - 1,
    )
    normalized_points = np.rint(
        normalized_points
    ).astype(np.int32)

    return FacePreprocessResult(
        image=normalized_image,
        display_image=normalized_display_image,
        face_mask=normalized_face_mask,
        semantic_skin_mask=normalized_semantic,
        points=normalized_points,
        crop_box=(
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2,
        ),
        square_side=square_side,
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
        status=(
            "ok: face cropped, centered square, black-background analysis image, "
            f"display crop preserved, {target_size}x{target_size}"
        ),
    )


def _clamped_view(
    name: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> View:
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(x1 + 1, min(width, int(x2)))
    y2 = max(y1 + 1, min(height, int(y2)))
    return View(name, x1, y1, x2, y2)


def build_face_adaptive_views(
    width: int,
    height: int,
    profile: str,
    points: np.ndarray,
    regions: dict[str, list[int]],
    face_mask: np.ndarray,
) -> list[View]:
    """
    基于标准化人脸关键点构建局部 ROI，不再依赖整张照片的固定比例。
    """
    views = [View("full_face", 0, 0, width, height)]
    if profile != "extreme":
        return views

    ys, xs = np.where(face_mask > 0)
    if len(xs) == 0:
        return build_views(width, height, profile)

    face_x1 = int(xs.min())
    face_x2 = int(xs.max()) + 1
    face_y1 = int(ys.min())
    face_y2 = int(ys.max()) + 1
    face_w = max(1, face_x2 - face_x1)
    face_h = max(1, face_y2 - face_y1)

    left_eye_pts = points[regions["left_eye"]]
    right_eye_pts = points[regions["right_eye"]]
    left_brow_pts = points[regions["left_eyebrow"]]
    right_brow_pts = points[regions["right_eyebrow"]]

    all_eye_pts = np.vstack(
        [left_eye_pts, right_eye_pts]
    )
    all_brow_pts = np.vstack(
        [left_brow_pts, right_brow_pts]
    )

    brow_top = int(all_brow_pts[:, 1].min())
    brow_bottom = int(all_brow_pts[:, 1].max())
    eye_top = int(all_eye_pts[:, 1].min())
    eye_bottom = int(all_eye_pts[:, 1].max())

    views.append(
        _clamped_view(
            "forehead",
            face_x1 + int(0.03 * face_w),
            face_y1,
            face_x2 - int(0.03 * face_w),
            brow_bottom + int(0.05 * face_h),
            width,
            height,
        )
    )

    views.append(
        _clamped_view(
            "eye_band",
            face_x1,
            brow_top - int(0.06 * face_h),
            face_x2,
            eye_bottom + int(0.13 * face_h),
            width,
            height,
        )
    )

    for name, eye_pts in (
        ("left_eye", left_eye_pts),
        ("right_eye", right_eye_pts),
    ):
        x, y, w, h = cv2.boundingRect(
            eye_pts.astype(np.int32)
        )
        views.append(
            _clamped_view(
                name,
                x - int(0.30 * w),
                y - int(1.25 * h),
                x + w + int(0.30 * w),
                y + h + int(1.20 * h),
                width,
                height,
            )
        )

    views.append(
        _clamped_view(
            "mid_lower_face",
            face_x1 + int(0.02 * face_w),
            eye_bottom,
            face_x2 - int(0.02 * face_w),
            face_y2,
            width,
            height,
        )
    )

    return views


def anisotropic_dilate(
    mask: np.ndarray,
    pad_x: int,
    pad_y: int,
) -> np.ndarray:
    pad_x = max(0, int(pad_x))
    pad_y = max(0, int(pad_y))
    if pad_x == 0 and pad_y == 0:
        return mask.copy()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            2 * pad_x + 1,
            2 * pad_y + 1,
        ),
    )
    return cv2.dilate(
        mask,
        kernel,
        iterations=1,
    )


def build_precise_eyebrow_mask(
    points: np.ndarray,
    indices: list[int],
    shape: tuple[int, int],
) -> np.ndarray:
    """
    使用关键点凸包 + 固定像素膨胀构建眉毛禁区。
    不再使用矩形，也不再依赖宽高比例。
    你只需要调 EYEBROW_DILATE 即可。
    """
    return mask_from_indices(
        points,
        indices,
        shape,
        dilate_px=EYEBROW_DILATE,
    )


def build_eye_forbidden_mask(
    points: np.ndarray,
    indices: list[int],
    shape: tuple[int, int],
) -> np.ndarray:
    """
    眼睛禁区：眼球、睫毛、上下眼睑内部统一向外膨胀。
    双眼皮处理不再单独使用额外矩形，而是直接依赖 EYE_CORE_DILATE。
    """
    return mask_from_indices(
        points,
        indices,
        shape,
        dilate_px=EYE_CORE_DILATE,
    )


def build_face_filter_masks(

    image: np.ndarray,
    points: np.ndarray,
    regions: dict[str, list[int]],
    semantic_skin_mask: np.ndarray | None,
    preprocessed_face_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    构建医美规则后处理区域。

    严格禁止：
    - 整个眉毛及毛发外缘
    - 眼球/眼睑内部
    - 整个鼻子（鼻梁、鼻尖、鼻翼、鼻孔）
    - 嘴唇
    - 鼻孔下方到上唇之间的胡须/人中区域

    允许：
    - 完整额头（通过语义皮肤分割 + 额头补全）
    - 眉间纹区，但不会重新放回眉毛或鼻子
    - 其余真实面部皮肤
    """
    height, width = image.shape[:2]
    shape = (height, width)

    extended_envelope = build_extended_face_envelope(
        points,
        regions,
        shape,
    )

    if semantic_skin_mask is not None:
        semantic_skin_mask = (
            semantic_skin_mask > 0
        ).astype(np.uint8) * 255

        face_skin = cv2.bitwise_and(
            semantic_skin_mask,
            extended_envelope,
        )
        face_skin = keep_largest_component(face_skin)
    else:
        # 分割模型不可用时退回关键点包络；精度会比语义皮肤分割低。
        face_skin = extended_envelope.copy()

    if preprocessed_face_mask is not None:
        strict_face = (
            preprocessed_face_mask > 0
        ).astype(np.uint8) * 255
        if strict_face.shape != shape:
            strict_face = resize_mask(
                strict_face,
                (width, height),
            )
        face_skin = cv2.bitwise_and(
            face_skin,
            strict_face,
        )

    if FACE_SHRINK_ERODE > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                2 * FACE_SHRINK_ERODE + 1,
                2 * FACE_SHRINK_ERODE + 1,
            ),
        )
        face_skin = cv2.erode(
            face_skin,
            kernel,
            iterations=1,
        )

    # ------------------------------------------------------------------
    # 眉毛：沿眉毛形状的精细禁区，不再使用大矩形。
    # ------------------------------------------------------------------
    left_brow_forbidden = build_precise_eyebrow_mask(
        points,
        regions["left_eyebrow"],
        shape,
    )
    right_brow_forbidden = build_precise_eyebrow_mask(
        points,
        regions["right_eyebrow"],
        shape,
    )
    eyebrows_forbidden = cv2.bitwise_or(
        left_brow_forbidden,
        right_brow_forbidden,
    )

    # ------------------------------------------------------------------
    # 眼睛：仅保留一个“统一眼睛禁区”，是否覆盖双眼皮由 EYE_CORE_DILATE 决定。
    # 不再单独构造上眼睑矩形，避免难以手调。
    # ------------------------------------------------------------------
    left_eye = build_eye_forbidden_mask(
        points,
        regions["left_eye"],
        shape,
    )
    right_eye = build_eye_forbidden_mask(
        points,
        regions["right_eye"],
        shape,
    )
    eyes_forbidden = cv2.bitwise_or(
        left_eye,
        right_eye,
    )

    # 嘴唇及边缘。
    lips_forbidden = mask_from_indices(
        points,
        regions["lips"],
        shape,
        dilate_px=LIP_EXCLUDE_DILATE,
    )

    # 整个鼻子。
    nose_forbidden = mask_from_indices(
        points,
        regions["nose"],
        shape,
        dilate_px=NOSE_EXCLUDE_DILATE,
    )

    # 鼻孔下方至上唇：用鼻翼上边 + 嘴角下边构造梯形胡须区。
    nose_lower_points = points[regions["nose_lower"]]
    lip_points = points[regions["lips"]]

    nose_x, nose_y, nose_w, nose_h = cv2.boundingRect(
        nose_lower_points
    )
    lip_x, lip_y, lip_w, lip_h = cv2.boundingRect(
        lip_points
    )

    top_y = int(
        round(
            np.max(nose_lower_points[:, 1])
            - MUSTACHE_TOP_PAD_RATIO * max(1, nose_h)
        )
    )
    bottom_y = int(
        round(
            lip_y
            + MUSTACHE_BOTTOM_LIP_RATIO * max(1, lip_h)
        )
    )

    top_left = int(
        round(nose_x - 0.08 * nose_w)
    )
    top_right = int(
        round(nose_x + nose_w + 0.08 * nose_w)
    )

    bottom_left = int(
        round(lip_x + 0.03 * lip_w)
    )
    bottom_right = int(
        round(lip_x + lip_w - 0.03 * lip_w)
    )

    if bottom_y <= top_y:
        bottom_y = top_y + max(4, int(round(0.5 * lip_h)))

    mustache_polygon = np.array(
        [
            [top_left, top_y],
            [top_right, top_y],
            [bottom_right, bottom_y],
            [bottom_left, bottom_y],
        ],
        dtype=np.int32,
    )

    mustache_forbidden = polygon_mask(
        shape,
        mustache_polygon,
        dilate_px=MUSTACHE_EXCLUDE_DILATE,
    )

    # ------------------------------------------------------------------
    # 额头区：从语义皮肤 Mask 的最高处开始，到眉毛下缘附近。
    # 关键点 face oval 不再决定额头上边界，所以不会只剩半个额头。
    # ------------------------------------------------------------------
    left_brow_pts = points[regions["left_eyebrow"]]
    right_brow_pts = points[regions["right_eyebrow"]]
    all_brow_pts = np.vstack(
        [left_brow_pts, right_brow_pts]
    )

    face_pts = points[regions["face_oval"]]
    face_x, _, face_w, _ = cv2.boundingRect(face_pts)
    _, brow_y, brow_w, brow_h = cv2.boundingRect(
        all_brow_pts
    )

    central_x1 = max(
        0,
        face_x + int(round(0.04 * face_w)),
    )
    central_x2 = min(
        width,
        face_x + face_w - int(round(0.04 * face_w)),
    )

    skin_coords = np.argwhere(
        face_skin[:, central_x1:central_x2] > 0
    )
    if skin_coords.size > 0:
        forehead_y1 = int(np.min(skin_coords[:, 0]))
    else:
        forehead_y1 = 0

    forehead_y2 = min(
        height,
        brow_y + int(round(0.45 * brow_h)),
    )

    forehead_zone = rect_mask(
        shape,
        central_x1,
        forehead_y1,
        central_x2,
        forehead_y2,
    )
    forehead_zone = cv2.bitwise_and(
        forehead_zone,
        face_skin,
    )

    # 额头区也必须严格去除眉毛/眼睛/鼻子，不能 OR 回禁止区域。
    forehead_forbidden = eyebrows_forbidden.copy()
    forehead_forbidden = cv2.bitwise_or(
        forehead_forbidden,
        eyes_forbidden,
    )
    forehead_forbidden = cv2.bitwise_or(
        forehead_forbidden,
        nose_forbidden,
    )
    forehead_zone = cv2.bitwise_and(
        forehead_zone,
        cv2.bitwise_not(forehead_forbidden),
    )

    # ------------------------------------------------------------------
    # 眉间纹区：按图像中的左右眉毛位置计算。
    # ------------------------------------------------------------------
    brow_groups = sorted(
        [left_brow_pts, right_brow_pts],
        key=lambda group: float(np.mean(group[:, 0])),
    )
    image_left_brow, image_right_brow = brow_groups

    left_inner_x = int(np.max(image_left_brow[:, 0]))
    right_inner_x = int(np.min(image_right_brow[:, 0]))
    brow_top = int(np.min(all_brow_pts[:, 1]))
    brow_bottom = int(np.max(all_brow_pts[:, 1]))

    if right_inner_x <= left_inner_x:
        center_x = int(round(np.mean(all_brow_pts[:, 0])))
        half_width = max(8, int(round(0.10 * brow_w)))
        left_inner_x = center_x - half_width
        right_inner_x = center_x + half_width

    frown_x1 = max(
        0,
        left_inner_x - int(round(0.06 * brow_w)),
    )
    frown_x2 = min(
        width,
        right_inner_x + int(round(0.06 * brow_w)),
    )
    frown_y1 = max(
        0,
        brow_top - int(round(0.35 * brow_h)),
    )
    frown_y2 = min(
        height,
        brow_bottom + int(round(0.80 * brow_h)),
    )

    glabella_zone = rect_mask(
        shape,
        frown_x1,
        frown_y1,
        frown_x2,
        frown_y2,
        dilate_px=GLABELLA_DILATE,
    )
    glabella_zone = cv2.bitwise_and(
        glabella_zone,
        face_skin,
    )

    # 关键修复：眉间允许区绝不能重新放回眉毛、眼睛或鼻子。
    glabella_forbidden = eyebrows_forbidden.copy()
    glabella_forbidden = cv2.bitwise_or(
        glabella_forbidden,
        eyes_forbidden,
    )
    glabella_forbidden = cv2.bitwise_or(
        glabella_forbidden,
        nose_forbidden,
    )
    glabella_zone = cv2.bitwise_and(
        glabella_zone,
        cv2.bitwise_not(glabella_forbidden),
    )

    # 所有禁止区。
    forbidden = eyebrows_forbidden.copy()
    for region_mask in (
        eyes_forbidden,
        lips_forbidden,
        nose_forbidden,
        mustache_forbidden,
    ):
        forbidden = cv2.bitwise_or(
            forbidden,
            region_mask,
        )

    wrinkle_allowed_zone = cv2.bitwise_and(
        face_skin,
        cv2.bitwise_not(forbidden),
    )

    # 不再用 OR 把额头/眉间重新叠回去。
    # forehead_zone 与 glabella_zone 本身已经是 allowed 的子集。
    combined_allowed = wrinkle_allowed_zone.copy()

    return {
        "semantic_skin_raw": (
            semantic_skin_mask
            if semantic_skin_mask is not None
            else np.zeros(shape, dtype=np.uint8)
        ),
        "extended_face_envelope": extended_envelope,
        "face_skin": face_skin,
        "eyebrows_forbidden": eyebrows_forbidden,
        "eyes_forbidden": eyes_forbidden,
        "nose_forbidden": nose_forbidden,
        "lips_forbidden": lips_forbidden,
        "mustache_forbidden": mustache_forbidden,
        "forehead_zone": forehead_zone,
        "glabella_zone": glabella_zone,
        "wrinkle_allowed_zone": wrinkle_allowed_zone,
        "combined_allowed_zone": combined_allowed,
        "forbidden_total": forbidden,
    }


def apply_face_filter(
    all_candidates_raw: np.ndarray,
    recommended_raw: np.ndarray,
    class_unions: dict[int, np.ndarray],
    filter_masks: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """
    类别感知筛选：

    - class 0 forehead：只保留完整额头区
    - class 1 frown：只保留严格眉间区
    - class 2 wrinkle：只保留真实皮肤且排除眉、眼、鼻、唇、胡须区
    - 未知类别：按通用 wrinkle_allowed_zone 处理

    最终结果直接由过滤后的各类别并集产生，避免旧版
    glabella/forehead OR 操作把眉毛区域重新放回。
    """
    class_filtered: dict[int, np.ndarray] = {}

    for class_id, mask in class_unions.items():
        if class_id == 0:
            zone = filter_masks["forehead_zone"]
        elif class_id == 1:
            zone = filter_masks["glabella_zone"]
        else:
            zone = filter_masks["wrinkle_allowed_zone"]

        class_filtered[class_id] = cv2.bitwise_and(
            mask,
            zone,
        )

    class_combined = np.zeros_like(all_candidates_raw)
    for mask in class_filtered.values():
        class_combined = cv2.bitwise_or(
            class_combined,
            mask,
        )

    # class_combined 本身已经来自 stage1 候选的类别并集。
    all_filtered = class_combined

    # 推荐结果在类别过滤后再与 stage1 推荐 Mask 相交。
    recommended_filtered = cv2.bitwise_and(
        recommended_raw,
        class_combined,
    )

    return all_filtered, recommended_filtered, class_filtered



# ============================================================================
# 医美分区分析（VISIA 风格展示 + 无监督量化）
# ============================================================================

def ellipse_mask(
    shape: tuple[int, int],
    center: tuple[int, int],
    axes: tuple[int, int],
    angle: float = 0.0,
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cx = int(np.clip(center[0], 0, width - 1))
    cy = int(np.clip(center[1], 0, height - 1))
    ax = max(1, int(round(axes[0])))
    ay = max(1, int(round(axes[1])))
    cv2.ellipse(mask, (cx, cy), (ax, ay), angle, 0, 360, 255, -1)
    return mask


def _sorted_left_right(groups: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    groups = sorted(groups, key=lambda g: float(np.mean(g[:, 0])))
    return groups[0], groups[1]


def line_band_mask(
    shape: tuple[int, int],
    p1: tuple[float, float],
    p2: tuple[float, float],
    half_width1: float,
    half_width2: float,
    dilate_px: int = 0,
) -> np.ndarray:
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    vec = p2 - p1
    norm = float(np.linalg.norm(vec))
    if norm < 1e-3:
        return np.zeros(shape, dtype=np.uint8)
    vec /= norm
    normal = np.array([-vec[1], vec[0]], dtype=np.float32)

    poly = np.array(
        [
            p1 + normal * float(half_width1),
            p1 - normal * float(half_width1),
            p2 - normal * float(half_width2),
            p2 + normal * float(half_width2),
        ],
        dtype=np.int32,
    )
    return polygon_mask(shape, poly, dilate_px=dilate_px)


def count_line_segments(
    centerline_mask: np.ndarray,
    min_length: int = REGION_MIN_SEGMENT_LENGTH,
) -> tuple[int, list[int]]:
    if np.count_nonzero(centerline_mask) == 0:
        return 0, []

    merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    merged = cv2.dilate(centerline_mask, merge_kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (merged > 0).astype(np.uint8),
        connectivity=8,
    )

    lengths: list[int] = []
    for label in range(1, num_labels):
        component = labels == label
        length = int(np.count_nonzero(centerline_mask[component]))
        if length >= int(min_length):
            lengths.append(length)

    return len(lengths), lengths




def build_aesthetic_region_masks(
    points: np.ndarray,
    regions: dict[str, list[int]],
    filter_masks: dict[str, np.ndarray],
    final_centerline: np.ndarray | None = None,
) -> list[tuple[str, str, str, np.ndarray, np.ndarray]]:
    """
    构建医美分区，并尽量把所有皱纹归到有名称的区域：
    - 额头纹
    - 眉间纹
    - 左/右鱼尾纹
    - 左/右眼下细纹
    - 左/右法令纹
    - 左/右木偶纹
    - 其他纹（仅在确实离所有命名区域都很远时才保留）
    """

    shape = filter_masks["face_skin"].shape
    allowed = filter_masks["wrinkle_allowed_zone"]

    left_eye_pts, right_eye_pts = _sorted_left_right(
        [points[regions["left_eye"]], points[regions["right_eye"]]]
    )
    left_brow_pts, right_brow_pts = _sorted_left_right(
        [points[regions["left_eyebrow"]], points[regions["right_eyebrow"]]]
    )

    face_pts = points[regions["face_oval"]]
    face_x, face_y, face_w, face_h = cv2.boundingRect(face_pts)

    lip_pts = points[regions["lips"]]
    lip_left = tuple(lip_pts[np.argmin(lip_pts[:, 0])].astype(int))
    lip_right = tuple(lip_pts[np.argmax(lip_pts[:, 0])].astype(int))
    _, lip_y, _, lip_h = cv2.boundingRect(lip_pts)
    mouth_center_y = float(np.mean(lip_pts[:, 1]))

    nose_lower_pts = points[regions["nose_lower"]]
    left_nose_side = tuple(nose_lower_pts[np.argmin(nose_lower_pts[:, 0])].astype(int))
    right_nose_side = tuple(nose_lower_pts[np.argmax(nose_lower_pts[:, 0])].astype(int))
    nose_center_x = float(np.mean(nose_lower_pts[:, 0]))

    brow_all = np.vstack([left_brow_pts, right_brow_pts])
    brow_top_y = float(np.min(brow_all[:, 1]))
    brow_mid_y = float(np.mean(brow_all[:, 1]))

    left_eye_rect = cv2.boundingRect(left_eye_pts.astype(np.int32))
    right_eye_rect = cv2.boundingRect(right_eye_pts.astype(np.int32))

    base_defs: list[tuple[str, str, str, np.ndarray]] = []
    nominal_defs: list[tuple[str, str, str, np.ndarray]] = []

    def add_region(key: str, display_name: str, short_name: str, mask: np.ndarray) -> None:
        nominal_mask = np.asarray(mask, dtype=np.uint8).copy()
        if np.count_nonzero(nominal_mask) == 0:
            return
        nominal_defs.append((key, display_name, short_name, nominal_mask))
        analysis_mask = cv2.bitwise_and(nominal_mask, allowed)
        if np.count_nonzero(analysis_mask) > 0:
            base_defs.append((key, display_name, short_name, analysis_mask))

    def point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        apx, apy = px - ax, py - ay
        abx, aby = bx - ax, by - ay
        ab2 = abx * abx + aby * aby
        if ab2 <= 1e-6:
            return float(math.hypot(px - ax, py - ay))
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
        qx = ax + t * abx
        qy = ay + t * aby
        return float(math.hypot(px - qx, py - qy))

    def component_orientation(component_mask: np.ndarray) -> tuple[float, float, float]:
        """
        计算一条皱纹连通域的主方向。

        OpenCV 不同版本中 cv2.fitLine() 可能返回：
        - 标量；
        - shape=(1,) 的 NumPy 数组；
        - shape=(4, 1) 的数组。

        因此不能直接 float(vx)，必须先展平并读取单个元素。
        """
        ys, xs = np.where(component_mask > 0)
        if len(xs) <= 1:
            return 0.0, 0.5, 0.5

        pts = np.column_stack(
            [xs.astype(np.float32), ys.astype(np.float32)]
        )

        try:
            fit = np.asarray(
                cv2.fitLine(
                    pts,
                    cv2.DIST_L2,
                    0,
                    0.01,
                    0.01,
                ),
                dtype=np.float64,
            ).reshape(-1)

            if fit.size < 2:
                raise ValueError(
                    f"cv2.fitLine 返回值长度异常：{fit.size}"
                )

            vx_value = float(fit[0].item())
            vy_value = float(fit[1].item())

        except Exception:
            # 极少数 OpenCV / NumPy 组合下的安全回退：使用 PCA 主方向。
            centered = pts.astype(np.float64)
            centered -= np.mean(centered, axis=0, keepdims=True)

            covariance = np.cov(centered, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            principal = eigenvectors[:, int(np.argmax(eigenvalues))]

            vx_value = float(principal[0])
            vy_value = float(principal[1])

        vector_norm = math.hypot(vx_value, vy_value)
        if (
            not math.isfinite(vx_value)
            or not math.isfinite(vy_value)
            or vector_norm <= 1e-12
        ):
            return 0.0, 0.5, 0.5

        vx_value /= vector_norm
        vy_value /= vector_norm

        angle = abs(
            math.degrees(
                math.atan2(vy_value, vx_value)
            )
        )
        if angle > 90.0:
            angle = 180.0 - angle

        verticalness = float(
            np.clip(angle / 90.0, 0.0, 1.0)
        )
        horizontalness = 1.0 - verticalness

        return angle, verticalness, horizontalness

    # 1) 额头纹
    add_region("forehead", "额头纹", "FH", filter_masks["forehead_zone"])

    # 2) 眉间纹
    add_region("glabella", "眉间纹", "GL", filter_masks["glabella_zone"])

    # 3) 鱼尾纹与眼下细纹
    def eye_regions(eye_pts: np.ndarray, outward_sign: int) -> tuple[np.ndarray, np.ndarray]:
        x, y, w, h = cv2.boundingRect(eye_pts.astype(np.int32))

        outer_x = x if outward_sign < 0 else x + w
        outer_center = (
            int(round(outer_x + outward_sign * 0.14 * w)),
            int(round(y + 0.48 * h)),
        )
        crow = ellipse_mask(shape, outer_center, axes=(0.42 * w, 0.78 * h), angle=0.0)

        divider_x = int(round(x + 0.48 * w))
        if outward_sign < 0:
            keep = rect_mask(shape, 0, y - h, divider_x, y + 2 * h)
        else:
            keep = rect_mask(shape, divider_x, y - h, shape[1], y + 2 * h)
        crow = cv2.bitwise_and(crow, keep)

        under = ellipse_mask(
            shape,
            center=(int(round(x + 0.50 * w)), int(round(y + 0.95 * h))),
            axes=(0.72 * w, 0.56 * h),
            angle=0.0,
        )
        lower_half = rect_mask(
            shape,
            x - int(round(0.18 * w)),
            y + int(round(0.18 * h)),
            x + int(round(1.18 * w)),
            y + int(round(2.05 * h)),
        )
        under = cv2.bitwise_and(under, lower_half)

        forbid = filter_masks["eyes_forbidden"].copy()
        forbid = cv2.bitwise_or(forbid, filter_masks["eyebrows_forbidden"])
        crow = cv2.bitwise_and(crow, cv2.bitwise_not(forbid))
        under = cv2.bitwise_and(under, cv2.bitwise_not(forbid))
        return crow, under

    left_crow, left_under = eye_regions(left_eye_pts, -1)
    right_crow, right_under = eye_regions(right_eye_pts, 1)
    add_region("left_crow_feet", "左鱼尾纹", "LCF", left_crow)
    add_region("right_crow_feet", "右鱼尾纹", "RCF", right_crow)
    add_region("left_under_eye", "左眼下细纹", "LUE", left_under)
    add_region("right_under_eye", "右眼下细纹", "RUE", right_under)

    # 4) 法令纹
    def nasolabial_mask(p_nose: tuple[int, int], p_mouth: tuple[int, int]) -> np.ndarray:
        dist = float(np.linalg.norm(np.asarray(p_nose, dtype=np.float32) - np.asarray(p_mouth, dtype=np.float32)))
        half_w1 = max(12.0, 0.13 * dist)
        half_w2 = max(15.0, 0.17 * dist)
        return line_band_mask(shape, p_nose, p_mouth, half_w1, half_w2, dilate_px=2)

    left_nasolabial = nasolabial_mask(left_nose_side, lip_left)
    right_nasolabial = nasolabial_mask(right_nose_side, lip_right)

    face_forbid = filter_masks["nose_forbidden"].copy()
    face_forbid = cv2.bitwise_or(face_forbid, filter_masks["lips_forbidden"])
    face_forbid = cv2.bitwise_or(face_forbid, filter_masks["mustache_forbidden"])
    left_nasolabial = cv2.bitwise_and(left_nasolabial, cv2.bitwise_not(face_forbid))
    right_nasolabial = cv2.bitwise_and(right_nasolabial, cv2.bitwise_not(face_forbid))
    add_region("left_nasolabial", "左法令纹", "LNL", left_nasolabial)
    add_region("right_nasolabial", "右法令纹", "RNL", right_nasolabial)

    # 5) 木偶纹 / 口角纹：更偏口角下方
    def marionette_mask(p_mouth: tuple[int, int], outward_sign: int) -> np.ndarray:
        p1 = np.asarray(p_mouth, dtype=np.float32)
        p2 = np.asarray(
            [p1[0] + outward_sign * 0.14 * face_w, p1[1] + 0.32 * face_h],
            dtype=np.float32,
        )
        mask = line_band_mask(shape, tuple(p1), tuple(p2), 12, 17, dilate_px=1)
        mouth_forbid = cv2.bitwise_or(filter_masks["lips_forbidden"], filter_masks["mustache_forbidden"])
        return cv2.bitwise_and(mask, cv2.bitwise_not(mouth_forbid))

    left_marionette = marionette_mask(lip_left, -1)
    right_marionette = marionette_mask(lip_right, 1)
    add_region("left_marionette", "左木偶纹", "LM", left_marionette)
    add_region("right_marionette", "右木偶纹", "RM", right_marionette)

    if final_centerline is None:
        available_keys = {item[0] for item in base_defs}
        return [
            (
                key,
                display_name,
                short_name,
                mask,
                np.zeros_like(mask),
                key in available_keys,
            )
            for key, display_name, short_name, mask in nominal_defs
        ]

    dilate_by_key = {
        "forehead": ASSIGN_DILATE_FOREHEAD,
        "glabella": ASSIGN_DILATE_GLABELLA,
        "left_crow_feet": ASSIGN_DILATE_EYE,
        "right_crow_feet": ASSIGN_DILATE_EYE,
        "left_under_eye": ASSIGN_DILATE_EYE,
        "right_under_eye": ASSIGN_DILATE_EYE,
        "left_nasolabial": ASSIGN_DILATE_NASOLABIAL,
        "right_nasolabial": ASSIGN_DILATE_NASOLABIAL,
        "left_marionette": ASSIGN_DILATE_MARIONETTE,
        "right_marionette": ASSIGN_DILATE_MARIONETTE,
    }

    base_mask_map = {k: m for k, _, _, m in base_defs}
    display_info = {k: (dn, sn) for k, dn, sn, _ in base_defs}

    attract_masks: dict[str, np.ndarray] = {}
    dist_maps: dict[str, np.ndarray] = {}
    for key, base_mask in base_mask_map.items():
        dilate_px = int(dilate_by_key.get(key, OTHER_REGION_DILATE))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilate_px + 1, 2 * dilate_px + 1),
        )
        attract = cv2.dilate(base_mask, kernel, iterations=1)
        attract = cv2.bitwise_and(attract, allowed)
        attract_masks[key] = attract
        dist_maps[key] = cv2.distanceTransform((255 - attract).astype(np.uint8), cv2.DIST_L2, 3)

    assigned_lines = {key: np.zeros(shape, dtype=np.uint8) for key in list(base_mask_map.keys()) + ["other"]}

    cc_num, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(
        (final_centerline > 0).astype(np.uint8),
        connectivity=8,
    )

    def choose_best_region(component_mask: np.ndarray) -> str:
        ys, xs = np.where(component_mask > 0)
        if len(xs) == 0:
            return "other"

        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        min_x, max_x = float(np.min(xs)), float(np.max(xs))
        min_y, max_y = float(np.min(ys)), float(np.max(ys))
        _, verticalness, horizontalness = component_orientation(component_mask)
        center_band = abs(cx - nose_center_x) <= CENTER_GLABELLA_X_RATIO * face_w

        overlaps: dict[str, int] = {}
        mean_dists: dict[str, float] = {}
        scores: dict[str, float] = {}

        def boost(key: str, value: float) -> None:
            """仅为当前可见且已建立的医美分区加分。

            侧脸、遮挡脸或局部脸可能使某个候选分区与 allowed mask 的
            交集为空，add_region() 会按设计跳过该分区。归属评分必须同步
            容忍这种情况，不能再假定左右法令纹等所有键始终存在。
            """
            if key in scores:
                scores[key] += value

        for key in base_mask_map.keys():
            overlap = int(np.count_nonzero(cv2.bitwise_and(component_mask, attract_masks[key])))
            mean_dist = float(np.mean(dist_maps[key][component_mask > 0]))
            overlaps[key] = overlap
            mean_dists[key] = mean_dist
            # 基础分：重叠优先，距离次之
            scores[key] = overlap * 100.0 - mean_dist

        # ---------- 眉间纹 vs 额头纹 ----------
        in_upper_face = cy <= brow_top_y + 0.45 * face_h
        if in_upper_face and center_band:
            boost("glabella", GLABELLA_VERTICAL_BONUS * verticalness)
            boost("forehead", FOREHEAD_HORIZONTAL_BONUS * horizontalness)

            # 在眉间/额头交界区域，再强化规则：
            # 竖纹归眉间，横纹归额头
            if verticalness >= 0.58:
                boost("glabella", 18.0)
            if horizontalness >= 0.58:
                boost("forehead", 18.0)

            if cy <= brow_mid_y - 0.03 * face_h:
                boost("forehead", 8.0)
            else:
                boost("glabella", 8.0)

        # ---------- 眼下细纹 vs 鱼尾纹 ----------
        def eye_bonus(rect, side_prefix: str, is_left: bool) -> None:
            x, y, w, h = rect
            eye_cx = x + 0.5 * w
            eye_cy = y + 0.5 * h
            outer_x = x if is_left else x + w

            if abs(cy - eye_cy) <= 1.4 * h:
                # 下眼区：位于眼睛下方、横向仍在眼周范围内
                if (y + 0.10 * h) <= cy <= (y + 1.65 * h):
                    if (x - 0.35 * w) <= cx <= (x + 1.35 * w):
                        boost(f"{side_prefix}_under_eye", 12.0)

                # 鱼尾纹：更靠外眼角
                if is_left:
                    if cx <= outer_x + 0.18 * w:
                        boost(f"{side_prefix}_crow_feet", 14.0)
                else:
                    if cx >= outer_x - 0.18 * w:
                        boost(f"{side_prefix}_crow_feet", 14.0)

                # 若明显更低，则偏眼下；若更外侧，则偏鱼尾
                if cy >= eye_cy + 0.22 * h:
                    boost(f"{side_prefix}_under_eye", 8.0)

        eye_bonus(left_eye_rect, "left", True)
        eye_bonus(right_eye_rect, "right", False)

        # ---------- 法令纹 vs 木偶纹 ----------
        def fold_bonus(side_prefix: str, nose_side: tuple[int, int], mouth_corner: tuple[int, int], outward_sign: int) -> None:
            marionette_end = (
                mouth_corner[0] + outward_sign * 0.14 * face_w,
                mouth_corner[1] + 0.32 * face_h,
            )

            d_nl = point_to_segment_distance(cx, cy, nose_side[0], nose_side[1], mouth_corner[0], mouth_corner[1])
            d_mar = point_to_segment_distance(cx, cy, mouth_corner[0], mouth_corner[1], marionette_end[0], marionette_end[1])

            boost(f"{side_prefix}_nasolabial", max(0.0, 18.0 - 0.55 * d_nl))
            boost(f"{side_prefix}_marionette", max(0.0, 18.0 - 0.55 * d_mar))

            # 口角上方到口角附近：更偏法令纹
            if cy <= mouth_corner[1] + 0.02 * face_h:
                boost(f"{side_prefix}_nasolabial", NASOLABIAL_UPPER_BONUS)

            # 口角下方：更偏木偶纹
            if cy >= mouth_corner[1] + 0.03 * face_h:
                boost(f"{side_prefix}_marionette", MARIONETTE_LOWER_BONUS)

            # 线段上端如果已经高于口角，通常更像法令纹
            if min_y < mouth_corner[1]:
                boost(f"{side_prefix}_nasolabial", 10.0)

            # 线段明显延伸到口角下方，更像木偶纹
            if max_y > mouth_corner[1] + 0.10 * face_h:
                boost(f"{side_prefix}_marionette", 12.0)

        if cx < nose_center_x:
            fold_bonus("left", left_nose_side, lip_left, -1)
        else:
            fold_bonus("right", right_nose_side, lip_right, 1)

        # 最终选择
        best_key = max(scores.keys(), key=lambda k: scores[k])
        best_overlap = overlaps.get(best_key, 0)
        best_dist = mean_dists.get(best_key, 999.0)

        # 更强地压缩“其他纹”：除非完全没有合理重叠，且离所有命名区域都很远。
        if best_overlap <= 0 and best_dist > OTHER_FALLBACK_MAX_DIST:
            return "other"
        return best_key

    for label in range(1, cc_num):
        area = int(cc_stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        component = (cc_labels == label).astype(np.uint8) * 255
        key = choose_best_region(component)
        assigned_lines[key] = cv2.bitwise_or(assigned_lines[key], component)

    # 构建最终显示分区：区域本体 + 归属皱纹线适度膨胀
    results: list[tuple[str, str, str, np.ndarray, np.ndarray]] = []
    zone_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * OTHER_REGION_DILATE + 1, 2 * OTHER_REGION_DILATE + 1),
    )

    for key, base_mask in base_mask_map.items():
        line_mask = assigned_lines.get(key, np.zeros(shape, dtype=np.uint8))
        expanded_lines = cv2.dilate(line_mask, zone_kernel, iterations=1)
        display_mask = cv2.bitwise_or(base_mask, expanded_lines)
        display_mask = cv2.bitwise_and(display_mask, allowed)

        # 对额头纹和眉间纹再做一点显示层面的修正：
        # 让交界处的竖纹更自然地包进眉间区域。
        if key == "glabella":
            vert_expand = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)), iterations=1)
            display_mask = cv2.bitwise_or(display_mask, cv2.bitwise_and(vert_expand, allowed))
        elif key == "forehead":
            hori_expand = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 9)), iterations=1)
            display_mask = cv2.bitwise_or(display_mask, cv2.bitwise_and(hori_expand, allowed))

        display_name, short_name = display_info[key]
        results.append((
            key,
            display_name,
            short_name,
            display_mask,
            line_mask,
            True,
        ))

    returned_keys = {item[0] for item in results}
    for key, display_name, short_name, nominal_mask in nominal_defs:
        if key in returned_keys:
            continue
        results.append((
            key,
            display_name,
            short_name,
            nominal_mask,
            np.zeros_like(nominal_mask),
            False,
        ))

    other_line_mask = assigned_lines.get("other", np.zeros(shape, dtype=np.uint8))
    if np.count_nonzero(other_line_mask) > 0:
        other_mask = cv2.dilate(other_line_mask, zone_kernel, iterations=1)
        other_mask = cv2.bitwise_and(other_mask, allowed)
        results.append((
            "other",
            "其他纹",
            "OT",
            other_mask,
            other_line_mask,
            True,
        ))

    return results


def compute_region_metrics(
    region_defs: list[tuple],
    final_centerline: np.ndarray,
) -> list[RegionMetric]:
    total_wrinkle_pixels = max(1, int(np.count_nonzero(final_centerline)))
    metrics: list[RegionMetric] = []

    for item in region_defs:
        if len(item) >= 5:
            key, display_name, short_name, mask, explicit_line_mask = item[:5]
            line_mask = explicit_line_mask.copy()
        else:
            key, display_name, short_name, mask = item
            line_mask = cv2.bitwise_and(final_centerline, mask)

        area_px = int(np.count_nonzero(mask))
        wrinkle_pixels = int(np.count_nonzero(line_mask))
        if wrinkle_pixels <= 0:
            continue

        segment_count, lengths = count_line_segments(line_mask)
        mean_len = float(np.mean(lengths)) if lengths else 0.0
        max_len = int(max(lengths)) if lengths else 0
        density_per_10k = float(wrinkle_pixels * 10000.0 / max(1, area_px))
        share_pct = float(wrinkle_pixels * 100.0 / total_wrinkle_pixels)

        raw_score = float(
            (wrinkle_pixels / max(1.0, math.sqrt(area_px))) * (1.0 + 0.12 * segment_count)
        )

        metrics.append(
            RegionMetric(
                key=key,
                display_name=display_name,
                short_name=short_name,
                mask=mask,
                line_mask=line_mask,
                area_px=area_px,
                wrinkle_pixels=wrinkle_pixels,
                segment_count=segment_count,
                mean_segment_length=mean_len,
                max_segment_length=max_len,
                density_per_10k=density_per_10k,
                share_pct=share_pct,
                raw_score=raw_score,
            )
        )

    max_raw = max([m.raw_score for m in metrics], default=0.0)
    for m in metrics:
        m.relative_score = 100.0 * m.raw_score / max_raw if max_raw > 1e-6 else 0.0

    metrics.sort(key=lambda item: item.relative_score, reverse=True)
    return metrics


_UNICODE_FONT_PATH: Path | None = None
_FONT_CACHE: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}


def find_unicode_font() -> Path:
    global _UNICODE_FONT_PATH

    if _UNICODE_FONT_PATH is not None:
        return _UNICODE_FONT_PATH

    for candidate in VISIA_FONT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            _UNICODE_FONT_PATH = path
            return path

    raise RuntimeError(
        "未找到可用于中文渲染的字体。请安装：sudo apt-get install -y fonts-noto-cjk"
    )


def get_unicode_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (int(size), bool(bold))
    if key not in _FONT_CACHE:
        font_path = find_unicode_font()
        if bold:
            bold_path = Path(str(font_path).replace("Regular", "Bold"))
            if bold_path.is_file():
                font_path = bold_path
        _FONT_CACHE[key] = ImageFont.truetype(str(font_path), int(size))
    return _FONT_CACHE[key]


def text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def fit_unicode_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    max_width: int,
    initial_size: int,
    min_size: int,
    bold: bool = False,
) -> ImageFont.ImageFont:
    for size in range(int(initial_size), int(min_size) - 1, -1):
        font = get_unicode_font(size, bold=bold)
        width, _ = text_size(draw, text, font)
        if width <= max_width:
            return font
    return get_unicode_font(min_size, bold=bold)


def draw_unicode_text_bgr(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    font: ImageFont.ImageFont,
    color_bgr: tuple[int, int, int],
    stroke_width: int = 0,
    stroke_fill_bgr: tuple[int, int, int] | None = None,
) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    stroke_fill = None
    if stroke_fill_bgr is not None:
        stroke_fill = (
            stroke_fill_bgr[2],
            stroke_fill_bgr[1],
            stroke_fill_bgr[0],
        )

    draw.text(
        origin,
        text,
        font=font,
        fill=color_rgb,
        stroke_width=int(stroke_width),
        stroke_fill=stroke_fill,
    )
    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def make_visia_region_overlay(
    image: np.ndarray,
    metrics: list[RegionMetric],
) -> np.ndarray:
    output = image.copy()

    # 先淡淡填充分区
    for metric in metrics:
        region = metric.mask > 0
        if not np.any(region):
            continue
        fill = np.empty_like(output)
        fill[:] = VISIA_REGION_FILL_COLOR
        blended = cv2.addWeighted(output, 1.0 - VISIA_REGION_FILL_ALPHA, fill, VISIA_REGION_FILL_ALPHA, 0)
        output[region] = blended[region]

    # 再画分区轮廓
    for metric in metrics:
        contours, _ = cv2.findContours(metric.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(
            output,
            contours,
            -1,
            VISIA_REGION_OUTLINE_COLOR,
            VISIA_REGION_OUTLINE_THICKNESS,
            cv2.LINE_AA,
        )

    # 画皱纹线
    output = overlay_mask(
        output,
        (lambda arr: np.maximum.reduce(arr))([m.line_mask for m in metrics]) if metrics else np.zeros(image.shape[:2], dtype=np.uint8),
        color=VISIA_WRINKLE_COLOR,
        alpha=VISIA_WRINKLE_ALPHA,
        line_dilate=VISIA_WRINKLE_DILATE,
    )

    # 标注分区简称和相对分数
    for metric in metrics:
        ys, xs = np.where(metric.mask > 0)
        if len(xs) == 0:
            continue
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        label = f"{metric.short_name} {metric.relative_score:.0f}"
        cv2.putText(output, label, (cx - 26, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(output, label, (cx - 26, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, VISIA_LABEL_COLOR, 1, cv2.LINE_AA)

    return output


def make_region_tiles(
    image: np.ndarray,
    metrics: list[RegionMetric],
    top_k: int = VISIA_TOP_REGION_COUNT,
    columns: int = 2,
) -> np.ndarray:
    selected = [m for m in metrics if m.wrinkle_pixels > 0][: max(1, int(top_k))]

    # 确保“其他纹”也能出现在卡片图里，避免出现“有线但无分区卡片”的观感。
    other_metric = next(
        (m for m in metrics if m.key == "other" and m.wrinkle_pixels > 0),
        None,
    )
    if other_metric is not None and all(m.key != "other" for m in selected):
        if len(selected) < max(1, int(top_k)):
            selected.append(other_metric)
        elif selected:
            selected[-1] = other_metric

    if not selected:
        return np.full((VISIA_TILE_HEIGHT, VISIA_TILE_WIDTH, 3), VISIA_TILE_BG_VALUE, dtype=np.uint8)

    tiles: list[np.ndarray] = []
    for metric in selected:
        ys, xs = np.where(metric.mask > 0)
        if len(xs) == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        pad = 28
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image.shape[1], x2 + pad)
        y2 = min(image.shape[0], y2 + pad)

        crop = image[y1:y2, x1:x2].copy()
        local_mask = metric.mask[y1:y2, x1:x2]
        local_line = metric.line_mask[y1:y2, x1:x2]

        region = local_mask > 0
        fill = np.empty_like(crop)
        fill[:] = VISIA_REGION_FILL_COLOR
        blended = cv2.addWeighted(crop, 1.0 - VISIA_REGION_FILL_ALPHA, fill, VISIA_REGION_FILL_ALPHA, 0)
        crop[region] = blended[region]

        contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(
            crop,
            contours,
            -1,
            VISIA_REGION_OUTLINE_COLOR,
            VISIA_REGION_OUTLINE_THICKNESS,
            cv2.LINE_AA,
        )

        crop = overlay_mask(
            crop,
            local_line,
            color=VISIA_WRINKLE_COLOR,
            alpha=VISIA_WRINKLE_ALPHA,
            line_dilate=VISIA_WRINKLE_DILATE,
        )

        # 组装 tile：上图下文字
        tile = np.full((VISIA_TILE_HEIGHT, VISIA_TILE_WIDTH, 3), VISIA_TILE_BG_VALUE, dtype=np.uint8)
        img_h = VISIA_TILE_HEIGHT - 58
        scale = min((VISIA_TILE_WIDTH - 16) / crop.shape[1], (img_h - 10) / crop.shape[0])
        new_w = max(1, int(round(crop.shape[1] * scale)))
        new_h = max(1, int(round(crop.shape[0] * scale)))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        ox = (VISIA_TILE_WIDTH - new_w) // 2
        oy = 8 + (img_h - new_h) // 2
        tile[oy:oy + new_h, ox:ox + new_w] = resized

        label = f"{metric.display_name} ({metric.relative_score:.0f})"
        subtitle = f"count={metric.segment_count}  len={metric.wrinkle_pixels}px"
        pil_probe = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
        draw_probe = ImageDraw.Draw(pil_probe)
        title_font = fit_unicode_font(
            draw_probe,
            label,
            max_width=VISIA_TILE_WIDTH - 24,
            initial_size=22,
            min_size=15,
            bold=True,
        )
        subtitle_font = fit_unicode_font(
            draw_probe,
            subtitle,
            max_width=VISIA_TILE_WIDTH - 28,
            initial_size=13,
            min_size=11,
        )
        tile = draw_unicode_text_bgr(
            tile,
            label,
            (12, VISIA_TILE_HEIGHT - 48),
            font=title_font,
            color_bgr=VISIA_LABEL_COLOR,
            stroke_width=2,
            stroke_fill_bgr=(255, 255, 255),
        )
        tile = draw_unicode_text_bgr(
            tile,
            subtitle,
            (14, VISIA_TILE_HEIGHT - 20),
            font=subtitle_font,
            color_bgr=(70, 70, 70),
        )
        tiles.append(tile)

    if not tiles:
        return np.full((VISIA_TILE_HEIGHT, VISIA_TILE_WIDTH, 3), VISIA_TILE_BG_VALUE, dtype=np.uint8)

    cols = max(1, int(columns))
    rows = int(math.ceil(len(tiles) / cols))
    blank = np.full((VISIA_TILE_HEIGHT, VISIA_TILE_WIDTH, 3), VISIA_TILE_BG_VALUE, dtype=np.uint8)
    while len(tiles) < rows * cols:
        tiles.append(blank.copy())

    row_imgs = []
    for idx in range(rows):
        row = np.hstack(tiles[idx * cols : (idx + 1) * cols])
        row_imgs.append(row)
    return np.vstack(row_imgs)


def make_region_image(
    image: np.ndarray,
    metric: RegionMetric,
) -> np.ndarray:
    """Render one text-free 360x240 user-facing region image."""
    ys, xs = np.where(metric.mask > 0)
    if len(xs) == 0:
        return np.full(
            (VISIA_REGION_IMAGE_HEIGHT, VISIA_REGION_IMAGE_WIDTH, 3),
            VISIA_TILE_BG_VALUE,
            dtype=np.uint8,
        )

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pad = 28
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image.shape[1], x2 + pad)
    y2 = min(image.shape[0], y2 + pad)

    crop = image[y1:y2, x1:x2].copy()
    local_mask = metric.mask[y1:y2, x1:x2]
    local_line = metric.line_mask[y1:y2, x1:x2]

    region = local_mask > 0
    fill = np.empty_like(crop)
    fill[:] = VISIA_REGION_FILL_COLOR
    blended = cv2.addWeighted(
        crop,
        1.0 - VISIA_REGION_FILL_ALPHA,
        fill,
        VISIA_REGION_FILL_ALPHA,
        0,
    )
    crop[region] = blended[region]

    contours, _ = cv2.findContours(
        local_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(
        crop,
        contours,
        -1,
        VISIA_REGION_OUTLINE_COLOR,
        VISIA_REGION_OUTLINE_THICKNESS,
        cv2.LINE_AA,
    )
    crop = overlay_mask(
        crop,
        local_line,
        color=VISIA_WRINKLE_COLOR,
        alpha=VISIA_WRINKLE_ALPHA,
        line_dilate=VISIA_WRINKLE_DILATE,
    )

    canvas = np.full(
        (VISIA_REGION_IMAGE_HEIGHT, VISIA_REGION_IMAGE_WIDTH, 3),
        VISIA_TILE_BG_VALUE,
        dtype=np.uint8,
    )
    available_width = VISIA_REGION_IMAGE_WIDTH - 2 * VISIA_REGION_IMAGE_MARGIN
    available_height = VISIA_REGION_IMAGE_HEIGHT - 2 * VISIA_REGION_IMAGE_MARGIN
    scale = min(available_width / crop.shape[1], available_height / crop.shape[0])
    new_w = max(1, int(round(crop.shape[1] * scale)))
    new_h = max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    ox = (VISIA_REGION_IMAGE_WIDTH - new_w) // 2
    oy = (VISIA_REGION_IMAGE_HEIGHT - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    return canvas


def safe_region_filename(display_name: str) -> str:
    """Keep controlled Chinese names while preventing accidental path separators."""
    safe_name = "".join(
        "_" if char in {"/", "\\", "\0"} else char
        for char in display_name.strip()
    )
    return safe_name or "未命名分区"


def save_output_end_region_cards(
    output_end_dir: Path,
    image: np.ndarray,
    metrics: list[RegionMetric],
) -> list[dict[str, Any]]:
    output_end_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for metric in metrics:
        if metric.wrinkle_pixels <= 0:
            continue
        filename = f"{safe_region_filename(metric.display_name)}.jpg"
        relative_path = (Path("output-end") / filename).as_posix()
        write_image(output_end_dir / filename, make_region_image(image, metric))
        manifest.append(
            {
                "region_key": metric.key,
                "region_name": metric.display_name,
                "result_key": f"region_{metric.key}",
                "relative_path": relative_path,
            }
        )

    return manifest


def save_region_metrics_csv(path: Path, metrics: list[RegionMetric]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
            ]
        )
        for m in metrics:
            writer.writerow(
                [
                    m.key,
                    m.display_name,
                    m.short_name,
                    round(m.relative_score, 2),
                    m.segment_count,
                    m.wrinkle_pixels,
                    round(m.mean_segment_length, 2),
                    m.max_segment_length,
                    round(m.density_per_10k, 2),
                    round(m.share_pct, 2),
                    m.area_px,
                ]
            )


def save_region_report(
    md_path: Path,
    metrics: list[RegionMetric],
) -> None:
    lines = []
    lines.append("# 无监督皱纹分区报告")
    lines.append("")
    lines.append("说明：")
    lines.append("- `segment_count`：近似皱纹条数（基于骨架连通域统计，不是医学金标准）。")
    lines.append("- `wrinkle_pixels`：总中心线长度像素，越大表示纹路越多/越长。")
    lines.append("- `relative_score`：同一张脸内部的相对严重度（0-100），仅用于区域排序。")
    lines.append("- `其他纹`：所有已检测到但未落入额头纹/眉间纹/鱼尾纹/眼下细纹/法令纹/木偶纹的剩余皱纹。")
    lines.append("")
    lines.append("| 区域 | 相对分数 | 条数 | 总长度(px) | 平均长度 | 最大长度 | 密度/1万像素 | 占全脸比例 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in metrics:
        lines.append(
            f"| {m.display_name} | {m.relative_score:.1f} | {m.segment_count} | "
            f"{m.wrinkle_pixels} | {m.mean_segment_length:.1f} | {m.max_segment_length} | "
            f"{m.density_per_10k:.1f} | {m.share_pct:.1f}% |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")
# ============================================================================
# 可视化
# ============================================================================

def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = DISPLAY_LINE_COLOR,
    alpha: float = DISPLAY_LINE_ALPHA,
    line_dilate: int = DISPLAY_LINE_DILATE,
    glow_sigma: float = DISPLAY_GLOW_SIGMA,
    glow_alpha_value: float = DISPLAY_GLOW_ALPHA,
) -> np.ndarray:
    """
    更适合医美展示的线条叠加：
    - 主线保持清晰；
    - 外围增加很轻的柔光；
    - 比纯红硬覆盖更柔和，不容易“惊吓用户”。
    """
    base = image.astype(np.float32).copy()
    line_mask = (mask > 0).astype(np.uint8) * 255

    if line_dilate > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                2 * line_dilate + 1,
                2 * line_dilate + 1,
            ),
        )
        core = cv2.dilate(line_mask, kernel, iterations=1)
    else:
        core = line_mask

    if glow_sigma > 0:
        glow = cv2.GaussianBlur(core, (0, 0), glow_sigma)
    else:
        glow = core.copy()

    glow_alpha = (glow.astype(np.float32) / 255.0) * glow_alpha_value
    core_alpha = (core.astype(np.float32) / 255.0) * alpha
    total_alpha = np.clip(glow_alpha + core_alpha, 0.0, 1.0)[..., None]

    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    output = (base * (1.0 - total_alpha) + color_arr * total_alpha).astype(np.uint8)

    return output


def make_vote_heatmap(
    vote_count: np.ndarray,
    coverage_count: np.ndarray,
) -> np.ndarray:
    vote_ratio = np.zeros_like(
        vote_count,
        dtype=np.float32,
    )
    valid = coverage_count > 0
    vote_ratio[valid] = (
        vote_count[valid].astype(np.float32)
        / coverage_count[valid].astype(np.float32)
    )

    display_ratio = np.sqrt(
        np.clip(vote_ratio, 0.0, 1.0)
    )
    heat_u8 = np.clip(
        display_ratio * 255.0,
        0,
        255,
    ).astype(np.uint8)

    return cv2.applyColorMap(
        heat_u8,
        cv2.COLORMAP_TURBO,
    )


def make_class_overlay(
    image: np.ndarray,
    class_unions: dict[int, np.ndarray],
    class_names: dict[int, str],
    line_width: int,
) -> np.ndarray:
    output = image.copy()

    palette = {
        0: (90, 130, 255),   # forehead
        1: (255, 150, 70),   # frown
        2: (90, 210, 120),   # wrinkle
    }
    fallback = [
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
    ]

    for index, class_id in enumerate(sorted(class_unions)):
        centerline = make_centerline(class_unions[class_id])
        centerline = set_line_width(centerline, line_width)
        mask_bool = centerline > 0

        if not np.any(mask_bool):
            continue

        color = palette.get(
            class_id,
            fallback[index % len(fallback)],
        )

        color_layer = np.empty_like(image)
        color_layer[:] = color
        blended = cv2.addWeighted(
            image,
            0.18,
            color_layer,
            0.82,
            0,
        )
        output[mask_bool] = blended[mask_bool]

    legend_y = 18
    for index, class_id in enumerate(sorted(class_unions)):
        color = palette.get(
            class_id,
            fallback[index % len(fallback)],
        )
        label = class_names.get(class_id, str(class_id))

        cv2.rectangle(
            output,
            (8, legend_y - 10),
            (20, legend_y + 2),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (26, legend_y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label,
            (26, legend_y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        legend_y += 19

    return output


def make_face_filter_debug(
    image: np.ndarray,
    filter_masks: dict[str, np.ndarray],
) -> np.ndarray:
    """
    颜色：
    - 绿色：允许输出皱纹的真实皮肤
    - 红色：眉毛
    - 蓝色：鼻子
    - 紫色：胡须/人中 + 嘴唇
    - 青色：眼睛/睫毛/双眼皮统一禁区
    - 黄色边线：完整额头允许区
    """
    output = image.copy()

    def blend_region(
        base: np.ndarray,
        mask: np.ndarray,
        color: tuple[int, int, int],
        alpha: float,
    ) -> np.ndarray:
        region = mask > 0
        if not np.any(region):
            return base
        layer = np.empty_like(base)
        layer[:] = color
        blended = cv2.addWeighted(
            base,
            1.0 - alpha,
            layer,
            alpha,
            0,
        )
        result = base.copy()
        result[region] = blended[region]
        return result

    output = blend_region(
        output,
        filter_masks["wrinkle_allowed_zone"],
        (40, 190, 40),
        0.20,
    )
    output = blend_region(
        output,
        filter_masks["eyebrows_forbidden"],
        (0, 0, 255),
        0.52,
    )
    output = blend_region(
        output,
        filter_masks["nose_forbidden"],
        (255, 80, 0),
        0.46,
    )
    output = blend_region(
        output,
        cv2.bitwise_or(
            filter_masks["mustache_forbidden"],
            filter_masks["lips_forbidden"],
        ),
        (220, 0, 220),
        0.46,
    )
    output = blend_region(
        output,
        filter_masks["eyes_forbidden"],
        (255, 255, 0),
        0.42,
    )

    forehead_contours, _ = cv2.findContours(
        filter_masks["forehead_zone"],
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(
        output,
        forehead_contours,
        -1,
        FACE_FILTER_FOREHEAD_OUTLINE_COLOR,
        FACE_FILTER_FOREHEAD_OUTLINE_THICKNESS,
        cv2.LINE_AA,
    )

    captions = [
        ("green=allowed skin", (40, 190, 40)),
        ("red=eyebrows", (0, 0, 255)),
        ("blue=nose", (255, 80, 0)),
        ("purple=mustache/lips", (220, 0, 220)),
        ("cyan=eyes", (255, 255, 0)),
        ("yellow line=forehead zone", (0, 255, 255)),
    ]

    y = 18
    for caption, color in captions:
        cv2.rectangle(output, (7, y - 9), (18, y + 2), color, -1)
        cv2.putText(
            output,
            caption,
            (23, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            caption,
            (23, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 16

    return output


# ============================================================================
# 主程序
# ============================================================================

def run_from_args(args: argparse.Namespace) -> int:
    configured_threads = configure_cpu_threads(args.cpu_threads)

    if getattr(args, "line_width", None) is None:
        args.line_width = DEFAULT_LINE_WIDTH
    weights_path = resolve_project_path(args.weights)
    source_path = resolve_project_path(args.source)
    output_dir = resolve_project_path(args.output)
    runs_dir = output_dir / "runs"
    output_end_dir = output_dir / "output-end"

    if not weights_path.is_file():
        print(f"❌ 找不到权重：{weights_path}")
        return 2
    if not source_path.is_file():
        print(f"❌ 找不到图片：{source_path}")
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_runs or SAVE_EACH_RUN:
        runs_dir.mkdir(parents=True, exist_ok=True)

    original_image = read_image(source_path)
    original_height, original_width = original_image.shape[:2]
    device = choose_device(args.device)

    face_model_path = resolve_project_path(args.face_model)
    segmenter_model_path = resolve_project_path(args.segmenter_model)

    # ------------------------------------------------------------------
    # 阶段 0：先做人脸关键点、语义皮肤分割和黑底居中裁剪。
    # 后续所有 YOLO 推理与人脸规则筛选均在预处理图上进行。
    # ------------------------------------------------------------------
    print("=" * 86)
    print("阶段 0：人脸裁剪 + 黑底 + 正方形居中标准化")
    print(f"原始图片：{source_path}")
    print(f"原始尺寸：{original_width} x {original_height}")

    try:
        (
            original_points,
            regions,
            mp_status,
        ) = detect_face_landmarks(
            original_image,
            face_model_path,
        )
    except Exception as exc:
        original_points = None
        regions = None
        mp_status = (
            "人脸关键点发生未捕获异常："
            f"{type(exc).__name__}: {exc}"
        )

    if original_points is None or regions is None:
        print(f"❌ 无法进行强制人脸预处理：{mp_status}")
        print("为避免背景纹理被识别为皱纹，本版本不会退回整图推理。")
        return 4

    if args.no_semantic_skin:
        original_semantic_skin = None
        segmenter_status = "disabled by --no-semantic-skin"
    else:
        try:
            (
                original_semantic_skin,
                segmenter_status,
            ) = detect_semantic_face_skin(
                original_image,
                segmenter_model_path,
            )
        except Exception as exc:
            original_semantic_skin = None
            segmenter_status = (
                "语义皮肤分割发生未捕获异常："
                f"{type(exc).__name__}: {exc}"
            )

    if original_semantic_skin is None:
        print(
            "⚠ 语义皮肤分割不可用，将退回关键点人脸包络；"
            "仍会裁剪和黑底，但发际线精度可能下降。"
        )
        print(f"  原因：{segmenter_status}")

    try:
        preprocess_result = normalize_face_to_square(
            original_image,
            original_points,
            regions,
            original_semantic_skin,
            target_size=int(args.preprocess_size),
        )
    except Exception as exc:
        print(
            "❌ 人脸标准化预处理失败："
            f"{type(exc).__name__}: {exc}"
        )
        return 5

    image = preprocess_result.image
    display_image = preprocess_result.display_image
    preprocessed_face_mask = preprocess_result.face_mask
    semantic_skin_mask = preprocess_result.semantic_skin_mask
    points = preprocess_result.points

    full_height, full_width = image.shape[:2]

    stage1_valid_face_mask = (
        preprocessed_face_mask > 0
    ).astype(np.uint8) * 255
    if STAGE1_VALID_FACE_ERODE > 0:
        stage1_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                2 * STAGE1_VALID_FACE_ERODE + 1,
                2 * STAGE1_VALID_FACE_ERODE + 1,
            ),
        )
        stage1_valid_face_mask = cv2.erode(
            stage1_valid_face_mask,
            stage1_kernel,
            iterations=1,
        )

    write_image(
        output_dir / OUTPUT_IMAGE_FILES["analysis_face"],
        image,
    )
    write_image(
        output_dir / OUTPUT_IMAGE_FILES["preprocessed_face"],
        display_image,
    )

    model = load_yolo_once(weights_path)
    class_names = normalize_names(getattr(model, "names", {}))
    class_ids = sorted(class_names.keys())

    print(f"预处理尺寸：{full_width} x {full_height}")
    print(f"预处理状态：{preprocess_result.status}")
    print(f"关键点状态：{mp_status}")
    print(f"语义分割状态：{segmenter_status}")
    print("=" * 86)
    print("YOLOv8 皱纹分割 V9-Fixed：标准化人脸最大召回 + 医美区域筛选")
    print(f"权重：{weights_path}")
    print(f"设备：{device}")
    print(f"模型任务：{getattr(model, 'task', 'unknown')}")
    print(f"类别：{class_names}")
    print(f"最低候选置信度：{args.raw_conf}")
    print(f"显示线宽：{args.line_width}")
    print(f"运行模式：{args.profile}")
    print(f"运行预设：{args.run_preset}")
    print(f"CPU 线程数：{configured_threads}")
    print("=" * 86)

    views = build_face_adaptive_views(
        full_width,
        full_height,
        args.profile,
        points,
        regions,
        preprocessed_face_mask,
    )

    estimated_runs = estimate_total_runs(views, args.run_preset)
    print(f"预计推理次数：{estimated_runs}")

    all_candidates_union = np.zeros(
        (full_height, full_width),
        dtype=np.uint8,
    )
    vote_count = np.zeros(
        (full_height, full_width),
        dtype=np.uint16,
    )
    coverage_count = np.zeros(
        (full_height, full_width),
        dtype=np.uint16,
    )
    max_score_map = np.zeros(
        (full_height, full_width),
        dtype=np.float32,
    )

    class_unions = {
        class_id: np.zeros(
            (full_height, full_width),
            dtype=np.uint8,
        )
        for class_id in class_ids
    }

    successful_runs = 0
    failed_runs = 0
    total_instances = 0

    # 即使第三阶段分区分析发生异常，第一、二阶段结果也应正常保存并完成汇总。
    region_metrics: list[RegionMetric] = []
    output_end_region_cards: list[dict[str, Any]] = []
    report_group_overlays: dict[str, dict[str, Any]] = {}
    region_analysis_status = "not_started"
    run_records: list[dict[str, Any]] = []
    run_index = 0

    started_all = time.perf_counter()

    for view in views:
        crop = image[
            view.y1 : view.y2,
            view.x1 : view.x2,
        ]
        is_full_face = view.name == "full_face"
        variants = make_variants(
            crop,
            is_full_face=is_full_face,
            run_preset=args.run_preset,
        )
        scales = get_scales(is_full_face, args.run_preset)
        flip_options = get_flip_options(args.run_preset)

        for variant_name, variant_image in variants.items():
            for imgsz in scales:
                for flipped in flip_options:
                    run_index += 1

                    run_name = (
                        f"{run_index:03d}_{view.name}_"
                        f"{variant_name}_{imgsz}_"
                        f"{'flip' if flipped else 'normal'}"
                    )

                    input_image = (
                        cv2.flip(variant_image, 1)
                        if flipped
                        else variant_image
                    )

                    print(
                        f"[{run_index:03d}] {run_name}",
                        end="",
                        flush=True,
                    )

                    started = time.perf_counter()

                    try:
                        results = safe_predict(
                            model,
                            input_image,
                            imgsz=imgsz,
                            conf=args.raw_conf,
                            device=device,
                        )
                    # Ultralytics 在个别视角没有有效分割原型时，
                    # process_mask_native 可能抛出 IndexError。该视角应按
                    # “无有效候选”跳过，不能让其余 83 个 balanced 视角
                    # 和整张图片一起失败。RuntimeError（包括单次 CUDA
                    # 推理异常）沿用原有逐视角隔离逻辑。
                    except (RuntimeError, IndexError) as exc:
                        elapsed = time.perf_counter() - started
                        failed_runs += 1
                        print(f"  ❌ {elapsed:.2f}s")
                        print(f"     {exc}")

                        if (
                            "out of memory" in str(exc).lower()
                            and torch.cuda.is_available()
                        ):
                            torch.cuda.empty_cache()

                        run_records.append(
                            {
                                "run": run_name,
                                "status": "failed",
                                "error": str(exc),
                            }
                        )
                        continue

                    elapsed = time.perf_counter() - started
                    result = results[0]

                    (
                        local_binary,
                        local_score,
                        local_classes,
                        instance_count,
                    ) = extract_run_maps(
                        result,
                        target_height=input_image.shape[0],
                        target_width=input_image.shape[1],
                        class_ids=class_ids,
                    )

                    if flipped:
                        local_binary = cv2.flip(local_binary, 1)
                        local_score = cv2.flip(local_score, 1)
                        local_classes = {
                            class_id: cv2.flip(mask, 1)
                            for class_id, mask in local_classes.items()
                        }

                    (
                        local_binary,
                        local_score,
                        local_classes,
                        cleared_margin,
                    ) = clear_local_crop_border(
                        local_binary,
                        local_score,
                        local_classes,
                        view.name,
                    )

                    full_binary = map_to_full_image(
                        local_binary,
                        view,
                        full_height,
                        full_width,
                    )
                    full_score = map_to_full_image(
                        local_score,
                        view,
                        full_height,
                        full_width,
                    )

                    # 所有候选必须位于预处理人脸内部。
                    # 这一步直接消除黑色背景、墙面、衣服、头发边缘等伪皱纹。
                    full_binary[
                        stage1_valid_face_mask == 0
                    ] = 0
                    full_score[
                        stage1_valid_face_mask == 0
                    ] = 0.0

                    coverage_count[
                        view.y1 : view.y2,
                        view.x1 : view.x2,
                    ] += 1

                    all_candidates_union[full_binary > 0] = 255
                    vote_count += (full_binary > 0).astype(np.uint16)
                    max_score_map = np.maximum(max_score_map, full_score)

                    for class_id, local_class_mask in local_classes.items():
                        if class_id not in class_unions:
                            class_unions[class_id] = np.zeros(
                                (full_height, full_width),
                                dtype=np.uint8,
                            )

                        full_class_mask = map_to_full_image(
                            local_class_mask,
                            view,
                            full_height,
                            full_width,
                        )
                        full_class_mask[
                            stage1_valid_face_mask == 0
                        ] = 0
                        class_unions[class_id][full_class_mask > 0] = 255

                    successful_runs += 1
                    total_instances += instance_count

                    print(
                        f"  masks={instance_count}, "
                        f"border_clear={cleared_margin}, "
                        f"{elapsed:.2f}s"
                    )

                    run_records.append(
                        {
                            "run": run_name,
                            "status": "ok",
                            "view": view.name,
                            "variant": variant_name,
                            "imgsz": imgsz,
                            "flipped": flipped,
                            "instances": instance_count,
                            "border_clear": cleared_margin,
                            "seconds": round(elapsed, 4),
                        }
                    )

                    if args.save_runs or SAVE_EACH_RUN:
                        write_image(
                            runs_dir / f"{run_name}.jpg",
                            result.plot(),
                        )

    elapsed_all = time.perf_counter() - started_all

    if successful_runs == 0:
        print("❌ 所有推理都失败，没有结果。")
        return 3

    # 第一阶段原始推荐结果
    recommended_raw = (
        (vote_count >= RECOMMENDED_MIN_VOTES)
        | (max_score_map >= RECOMMENDED_MIN_SCORE)
    ).astype(np.uint8) * 255

    if REMOVE_OUTER_BORDER_COMPONENTS:
        recommended_raw = remove_outer_border_components(
            recommended_raw,
            margin=OUTER_BORDER_MARGIN,
            min_component_size=OUTER_BORDER_MIN_COMPONENT_SIZE,
        )

    # 第一阶段中心线
    all_centerline = make_centerline(all_candidates_union)
    recommended_centerline = make_centerline(recommended_raw)

    all_display_line = set_line_width(
        all_centerline,
        args.line_width,
    )
    recommended_display_line = set_line_width(
        recommended_centerline,
        args.line_width,
    )

    # 保存第一阶段结果
    stage1_overlay = overlay_mask(
        image,
        all_display_line,
        color=(0, 0, 255),
        alpha=0.88,
    )
    stage1_overlay_display = overlay_mask(
        display_image,
        all_display_line,
    )
    write_image(
        output_dir / OUTPUT_IMAGE_FILES["stage1_candidates"],
        stage1_overlay_display,
    )

    vote_heatmap = make_vote_heatmap(
        vote_count,
        coverage_count,
    )
    write_image(
        output_dir / OUTPUT_IMAGE_FILES["vote_heatmap"],
        vote_heatmap,
    )

    # ------------------------------------------------------------------
    # 第二阶段：在同一张标准化人脸上做关键点医美规则筛选。
    # 不再重新检测另一套坐标，避免裁剪前后坐标不一致。
    # ------------------------------------------------------------------
    face_filter_available = True

    filtered_all_raw = np.zeros_like(all_candidates_union)
    filtered_recommended_raw = np.zeros_like(recommended_raw)
    filtered_class_unions = {
        class_id: np.zeros_like(mask)
        for class_id, mask in class_unions.items()
    }
    filter_masks: dict[str, np.ndarray] | None = None

    if face_filter_available:
        filter_masks = build_face_filter_masks(
            image,
            points,
            regions,
            semantic_skin_mask,
            preprocessed_face_mask,
        )
        (
            filtered_all_raw,
            filtered_recommended_raw,
            filtered_class_unions,
        ) = apply_face_filter(
            all_candidates_union,
            recommended_raw,
            class_unions,
            filter_masks,
        )
    else:
        print(f"⚠ 跳过人脸后处理筛选：{mp_status}")

    if face_filter_available:
        filtered_all_centerline = make_centerline(filtered_all_raw)
        filtered_recommended_centerline = make_centerline(filtered_recommended_raw)

        filtered_all_display = set_line_width(
            filtered_all_centerline,
            args.line_width,
        )
        filtered_recommended_display = set_line_width(
            filtered_recommended_centerline,
            args.line_width,
        )

        final_overlay = overlay_mask(
            image,
            filtered_all_display,
        )
        final_overlay_display = overlay_mask(
            display_image,
            filtered_all_display,
        )
        write_image(
            output_dir / OUTPUT_IMAGE_FILES["stage2_overlay"],
            final_overlay_display,
        )

        write_image(
            output_dir / OUTPUT_IMAGE_FILES["stage2_centerline"],
            filtered_all_centerline,
        )

        filter_debug = make_face_filter_debug(
            image,
            filter_masks,
        )
        write_image(
            output_dir / OUTPUT_IMAGE_FILES["face_filter_debug"],
            filter_debug,
        )

        # 单独保存每个筛选区域，便于按负责人要求继续调边界。
        texture_reference = generate_texture_reference(
            image,
            filter_masks["face_skin"],
        )
        output_end_dir.mkdir(parents=True, exist_ok=True)
        write_image(
            output_dir / OUTPUT_IMAGE_FILES["texture_reference"],
            texture_reference,
        )
        write_image(
            output_end_dir / OUTPUT_IMAGE_FILES["texture_reference"],
            texture_reference,
        )

        separator = np.full(
            (full_height, 8, 3),
            255,
            dtype=np.uint8,
        )
        compare = np.hstack(
            (image, separator, stage1_overlay, separator, final_overlay)
        )
        compare_display = np.hstack(
            (
                display_image,
                separator,
                stage1_overlay_display,
                separator,
                final_overlay_display,
            )
        )
        write_image(
            output_dir / OUTPUT_IMAGE_FILES["comparison"],
            compare_display,
        )
        # ------------------------------------------------------------------
        # 第三阶段：VISIA 风格分区展示 + 无监督量化（进一步压缩“其他纹”，优化法令纹/木偶纹与眉间/额头边界）
        # ------------------------------------------------------------------
        try:
            region_defs = build_aesthetic_region_masks(
                points,
                regions,
                filter_masks,
                filtered_all_centerline,
            )
            region_metrics = compute_region_metrics(
                region_defs,
                filtered_all_centerline,
            )
            from src.doctor_v3.evidence import enabled, save
            if enabled():
                save(output_dir, "wrinkle",
                     {"valid": filter_masks["face_skin"], "landmarks": points,
                      "instances": filtered_all_raw, "skeleton": filtered_all_centerline,
                      "image": image},
                     {"quality_status": "PASS", "source_kind": "rgb_measured"})

            region_overlay = make_visia_region_overlay(
                display_image,
                region_metrics,
            )
            write_image(
                output_dir / OUTPUT_IMAGE_FILES["region_overlay"],
                region_overlay,
            )
            write_image(
                output_end_dir / OUTPUT_IMAGE_FILES["region_overlay"],
                region_overlay,
            )

            region_tiles = make_region_tiles(
                display_image,
                region_metrics,
                top_k=VISIA_TOP_REGION_COUNT,
            )
            write_image(
                output_dir / OUTPUT_IMAGE_FILES["region_tiles"],
                region_tiles,
            )

            output_end_region_cards = save_output_end_region_cards(
                output_end_dir,
                display_image,
                region_metrics,
            )

            save_region_metrics_csv(
                output_dir / "25_region_metrics.csv",
                region_metrics,
            )

            if bool(getattr(args, "report_group_overlays", False)):
                from src.wrinkle.report_region_overlays import (
                    render_legacy_group_overlays,
                )

                report_group_overlays = render_legacy_group_overlays(
                    display_image,
                    region_defs,
                    output_end_dir,
                )
            region_analysis_status = "ok"

        except Exception as exc:
            region_analysis_status = (
                f"failed: {type(exc).__name__}: {exc}"
            )

            print("")
            print("⚠ 第三阶段分区分析失败，但第一、二阶段结果已经保存。")
            print(f"  原因：{region_analysis_status}")

            error_text = traceback.format_exc()
            (
                output_dir / "region_analysis_error.txt"
            ).write_text(
                error_text,
                encoding="utf-8",
            )


    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    image_pixels = full_height * full_width
    stage1_raw_pixels = int(np.count_nonzero(all_candidates_union))
    stage1_center_pixels = int(np.count_nonzero(all_centerline))
    stage1_recommended_pixels = int(np.count_nonzero(recommended_raw))

    summary = {
        "weights": str(weights_path),
        "source": str(source_path),
        "output_dir": str(output_dir),
        "original_image_size": {
            "width": original_width,
            "height": original_height,
        },
        "preprocessed_image_size": {
            "width": full_width,
            "height": full_height,
        },
        "preprocess_status": preprocess_result.status,
        "preprocess_crop_box": list(preprocess_result.crop_box),
        "preprocess_square_side": preprocess_result.square_side,
        "preprocess_scale": preprocess_result.scale,
        "preprocess_background_value": PREPROCESS_BACKGROUND_VALUE,
        "cpu_threads": configured_threads,
        "run_preset": args.run_preset,
        "estimated_runs": estimated_runs,
        "eyebrow_dilate": EYEBROW_DILATE,
        "eye_core_dilate": EYE_CORE_DILATE,
        "region_analysis_status": region_analysis_status,
        "device": device,
        "model_task": str(getattr(model, "task", "unknown")),
        "class_names": class_names,
        "profile": args.profile,
        "raw_confidence": args.raw_conf,
        "nms_iou": NMS_IOU,
        "max_detections": MAX_DETECTIONS,
        "line_width": args.line_width,
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "total_instances": total_instances,
        "stage1_raw_pixels": stage1_raw_pixels,
        "stage1_raw_ratio": stage1_raw_pixels / image_pixels,
        "stage1_center_pixels": stage1_center_pixels,
        "stage1_center_ratio": stage1_center_pixels / image_pixels,
        "stage1_recommended_pixels": stage1_recommended_pixels,
        "stage1_recommended_ratio": stage1_recommended_pixels / image_pixels,
        "face_filter_available": face_filter_available,
        "face_landmarker_model": str(face_model_path),
        "face_landmarker_api": "MediaPipe Tasks FaceLandmarker",
        "face_filter_status": mp_status,
        "image_segmenter_model": str(segmenter_model_path),
        "image_segmenter_api": "MediaPipe Tasks ImageSegmenter",
        "semantic_skin_status": segmenter_status,
        "semantic_skin_enabled": semantic_skin_mask is not None,
        "elapsed_seconds": round(elapsed_all, 3),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "runs": run_records,
    }

    if output_end_dir.is_dir():
        summary["output_end"] = {
            "directory": "output-end",
            "texture_reference": (
                f"output-end/{OUTPUT_END_TEXTURE_FILENAME}"
                if (output_end_dir / OUTPUT_IMAGE_FILES["texture_reference"]).is_file()
                else None
            ),
            "region_overlay": (
                f"output-end/{OUTPUT_END_REGION_OVERLAY_FILENAME}"
                if (output_end_dir / OUTPUT_IMAGE_FILES["region_overlay"]).is_file()
                else None
            ),
            "region_cards": output_end_region_cards,
        }

    if report_group_overlays:
        summary["report_group_overlays"] = report_group_overlays

    if face_filter_available:
        stage2_raw_pixels = int(np.count_nonzero(filtered_all_raw))
        stage2_center_pixels = int(np.count_nonzero(filtered_all_centerline))
        stage2_recommended_pixels = int(np.count_nonzero(filtered_recommended_raw))

        summary.update(
            {
                "stage2_raw_pixels": stage2_raw_pixels,
                "stage2_raw_ratio": stage2_raw_pixels / image_pixels,
                "stage2_center_pixels": stage2_center_pixels,
                "stage2_center_ratio": stage2_center_pixels / image_pixels,
                "stage2_recommended_pixels": stage2_recommended_pixels,
                "stage2_recommended_ratio": stage2_recommended_pixels / image_pixels,
            }
        )

        if "region_metrics" not in summary:
            try:
                summary["region_metrics"] = [
                    {
                        "region_key": m.key,
                        "region_name": m.display_name,
                        "short_name": m.short_name,
                        "relative_score": round(m.relative_score, 2),
                        "segment_count": m.segment_count,
                        "wrinkle_pixels": m.wrinkle_pixels,
                        "mean_segment_length": round(m.mean_segment_length, 2),
                        "max_segment_length": m.max_segment_length,
                        "density_per_10k": round(m.density_per_10k, 2),
                        "share_pct": round(m.share_pct, 2),
                        "area_px": m.area_px,
                    }
                    for m in region_metrics
                ]
            except Exception:
                pass

    with (output_dir / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 82)
    print("运行完成")
    print(f"成功推理次数：{successful_runs}")
    print(f"失败推理次数：{failed_runs}")
    print(f"累计实例数量：{total_instances}")
    print(f"阶段1 原始最大召回占比：{stage1_raw_pixels / image_pixels:.2%}")
    print(f"阶段1 中心线占比：{stage1_center_pixels / image_pixels:.2%}")
    if face_filter_available:
        print(f"阶段2 后处理中心线占比：{stage2_center_pixels / image_pixels:.2%}")
        print("✅ 已启用人脸关键点后处理筛选")
    else:
        print(f"⚠ 未启用人脸关键点后处理：{mp_status}")
    print(f"总耗时：{elapsed_all:.2f}s")
    print(f"分区分析状态：{region_analysis_status}")
    print(f"输出目录：{output_dir}")
    print("")
    print("重点查看：")
    descriptions = {
        "analysis_face": "黑底居中标准人脸（用于分析）",
        "preprocessed_face": "带背景居中裁剪图（用于展示）",
        "stage1_candidates": "带背景展示版全检出",
        "vote_heatmap": "投票热力图",
        "stage2_overlay": "带背景展示版最终结果",
        "stage2_centerline": "第二阶段最终中心线",
        "face_filter_debug": "彩色筛选区域调试图",
        "texture_reference": "纹理参考图",
        "comparison": "带背景三联对比图",
        "region_overlay": "VISIA 风格全脸分区图（含其他纹）",
        "region_tiles": "Top 分区卡片图",
    }
    for result_key, description in descriptions.items():
        print(f"  {OUTPUT_IMAGE_FILES[result_key]:<32} # {description}")
    print("  25_region_metrics.csv                  # 分区量化结果表")
    print("  output-end/                            # 面向用户的最终图和独立分区卡片")
    print("=" * 82)

    return 0


def main() -> int:
    return run_from_args(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
