"""Phase-E 质量门测试公共夹具：合成图 + 注入式 preprocessor（不跑模型推理）。"""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np


class StubPreprocessor:
    """替身 preprocessor：只提供 evaluate_image_path 读取的人脸/质量分支。

    仅用于单元测试，避免 MediaPipe 模型推理；图像派生完整性检查仍由真实
    ``evaluate_image_path`` 在合成图上运行。
    """

    def __init__(self, flags: tuple[str, ...] = (), quality_status: str = "PASS") -> None:
        self.flags = list(flags)
        self.quality_status = quality_status
        self.calls = 0

    def preprocess_image(self, image: np.ndarray) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(
            quality_flags=list(self.flags),
            quality_status=self.quality_status,
        )


class BoomPreprocessor:
    """若被调用即失败，用于证明解码失败路径不会触发 preprocessor。"""

    def preprocess_image(self, image: np.ndarray) -> SimpleNamespace:
        raise AssertionError("preprocess_image must not be called")


def clean_synthetic() -> np.ndarray:
    """无接缝/无热图的合成渐变图（人脸分支由注入 preprocessor 决定）。"""
    image = np.zeros((256, 192, 3), np.uint8)
    image[:, :, 0] = np.linspace(20, 200, 192)[None, :]
    image[:, :, 1] = np.linspace(30, 180, 192)[None, :]
    image[:, :, 2] = np.linspace(40, 160, 192)[None, :]
    return image


def center_collage() -> np.ndarray:
    """水平拼接：接缝位于高度 50%（43%~57% 内），确定性触发 collage。"""
    base = cv2.resize(clean_synthetic(), (256, 256))
    return cv2.vconcat((base, cv2.flip(base, 1)))


def encode_png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()
