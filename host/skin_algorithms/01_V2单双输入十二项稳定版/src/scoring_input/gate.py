"""同源原图派生质量门：真实调用 V0.1.1 ``evaluate_image_path``。

云端单 RGB worker 在下载原图后、清理中间文件前调用
:func:`compute_input_quality_gate`，把权威门禁的 ``status``/``reason_codes``
忠实投影进 ``scoring_input.quality.input_quality_gate``。

纪律：

- 内部直接调用 ``scoring_calibration.v011.quality.evaluate_image_path``，
  同一函数、同一判定，不复制其图像派生逻辑、不放宽阈值；
- 只从门禁结果取契约 ``InputQualityGate`` 的 ``status``/``reason_codes``
  两个字段，绝不做 ``algorithm_quality_status``/``quality_flags`` → 门禁映射；
- 仅当门禁函数本身抛错（如模型/依赖不可用）时返回 ``None``，并记录日志，
  交聚合侧诚实降级 REVIEW。
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

from src.scoring_calibration.v011.quality import evaluate_image_path

logger = logging.getLogger(__name__)

_GATE_TEMP_PREFIX = "dermavision_gate_"

# 进程级 MediaPipe preprocessor 单例：MediaPipe 模型加载较贵，三个 worker
# 均以单算法任务串行处理，进程内复用即可，无需每任务重建。
_default_preprocessor: Any = None


def _get_default_preprocessor() -> Any:
    global _default_preprocessor
    if _default_preprocessor is None:
        from src.preprocess.image_preprocessor import ImagePreprocessor

        _default_preprocessor = ImagePreprocessor()
    return _default_preprocessor


class _LazyPreprocessor:
    """按需解析共享 preprocessor。

    ``evaluate_image_path`` 对不可解码输入会在构造 preprocessor 之前直接返回
    ``decode_failed``；本代理保证只有在真正调用 ``preprocess_image`` 时才加载
    MediaPipe 模型，避免损坏输入也触发模型加载。
    """

    def preprocess_image(self, image: Any) -> Any:
        return _get_default_preprocessor().preprocess_image(image)


def compute_input_quality_gate(
    image_bytes: bytes | bytearray | None,
    *,
    preprocessor: Any | None = None,
) -> dict[str, Any] | None:
    """对下载原图 bytes 计算权威输入门禁。

    返回 ``{"status": str, "reason_codes": list[str]}``（契约
    ``InputQualityGate``），无法计算时返回 ``None``。解码失败由门禁函数本身
    返回 ``REJECT``/``decode_failed``，不会被吞成 ``None``。
    """

    if not isinstance(image_bytes, (bytes, bytearray)) or len(image_bytes) == 0:
        return None

    with tempfile.TemporaryDirectory(prefix=_GATE_TEMP_PREFIX) as tmpdir:
        path = os.path.join(tmpdir, "input.img")
        try:
            with open(path, "wb") as handle:
                handle.write(bytes(image_bytes))
        except OSError:
            logger.exception("input_quality_gate 无法写入临时输入")
            return None
        try:
            result = evaluate_image_path(
                path,
                preprocessor=preprocessor or _LazyPreprocessor(),  # type: ignore[arg-type]
            )
        except Exception:  # noqa: BLE001 - 门禁不可计算时诚实返回 None
            logger.exception("input_quality_gate evaluate_image_path 执行失败")
            return None

    if not isinstance(result, dict) or result.get("status") is None:
        logger.error("input_quality_gate 返回结构异常: %r", type(result).__name__)
        return None
    reasons = result.get("reason_codes") or []
    return {
        "status": str(result["status"]),
        "reason_codes": [str(code) for code in reasons],
    }


__all__ = ["compute_input_quality_gate"]
