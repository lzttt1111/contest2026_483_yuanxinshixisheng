# -*- coding: utf-8 -*-
"""MediaPipe 模型统一加载器 — 避免各引擎重复下载和加载逻辑"""

import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_URLS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "selfie_segmenter.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
    "selfie_multiclass_256x256.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
}


def _ensure_model(filename):
    """确保模型文件存在，不存在则自动下载"""
    model_path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(model_path):
        os.makedirs(MODELS_DIR, exist_ok=True)
        url = MODEL_URLS.get(filename)
        if not url:
            raise ValueError(f"未知模型文件: {filename}")
        print(f"⏳ 正在下载模型: {filename} ...")
        urllib.request.urlretrieve(url, model_path)
    return model_path


def load_face_landmarker(num_faces=1, min_detection_confidence=0.5, min_presence_confidence=0.5):
    """加载面部关键点检测器"""
    model_path = _ensure_model("face_landmarker.task")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=num_faces,
        min_face_detection_confidence=min_detection_confidence,
        min_face_presence_confidence=min_presence_confidence
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    print("🟢 [ModelLoader] 面部关键点模型已就绪。")
    return landmarker


def load_selfie_segmenter():
    """加载人像分割器"""
    model_path = _ensure_model("selfie_segmenter.tflite")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        output_confidence_masks=True
    )
    segmenter = vision.ImageSegmenter.create_from_options(options)
    print("🟢 [ModelLoader] 人像分割模型已就绪。")
    return segmenter


def load_selfie_multiclass_segmenter():
    """加载 RBX 专用多类别分割器。

    模型文件必须随项目交付。这里沿用统一加载器的缺失文件检查，但生产
    Docker 构建和发布验收必须确保模型已存在，不能依赖运行时网络下载。
    """
    model_path = _ensure_model("selfie_multiclass_256x256.tflite")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        output_category_mask=True,
    )
    segmenter = vision.ImageSegmenter.create_from_options(options)
    print("🟢 [ModelLoader] RBX 多类别人脸分割模型已就绪。")
    return segmenter
