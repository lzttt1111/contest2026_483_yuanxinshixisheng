# -*- coding: utf-8 -*-
"""VISIA-like 紫外线色斑与紫质展示底图。

该模块只负责展示底图，不检测或绘制任何紫质/紫外线色斑实例。
输入必须是 Preprocessor V2 的 ``analysis_image`` 及同坐标的人像前景 Mask。
所有运算确定性执行，不读取 ``display_image``，也不改变现有五项检测。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

from src.engines.visia_regions import masked_gaussian


_DETECTION_UV_BLACK_LEVEL: Final = 0.025
_DETECTION_UV_WHITE_LEVEL: Final = 0.92
_DETECTION_UV_GAINS_BGR: Final = (1.02, 1.0, 0.98)
_DETECTION_PORPHYRIN_SHADOW_BGR: Final = (9, 3, 2)
_DETECTION_PORPHYRIN_MID_BGR: Final = (82, 28, 8)
_DETECTION_PORPHYRIN_HIGHLIGHT_BGR: Final = (176, 92, 28)


@dataclass(frozen=True)
class PurpleBaseStyleConfig:
    """紫区底图集中调参区。

    调参原则：
    - ``uv_*`` 控制灰黑 UV 色斑底图；对比越大，综合色素纹理越明显。
    - ``porphyrin_*`` 控制蓝紫荧光底图；蓝色权重越大，冷色荧光越强。
    - 参数只影响展示，不作为医学紫质或 UV 色斑量化依据。
    """

    # 前景边缘柔化半径。调大可减轻锯齿，过大会让人物边缘发虚。
    foreground_feather_sigma: float = 1.0

    # CLAHE 用于恢复预处理缩放后的局部纹理；过大会放大噪声和压缩块。
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple[int, int] = (12, 12)

    # 综合色素代理由低频暗度、BlackHat 暗结构和综合色差共同组成。
    pigment_background_sigma: float = 22.0
    pigment_blackhat_sizes: tuple[int, ...] = (9, 17, 31)
    pigment_dark_weight: float = 0.42
    pigment_blackhat_weight: float = 0.36
    pigment_chroma_weight: float = 0.22

    # UV 色斑底图：浅区保持灰白，综合色素区域按强度压暗。
    uv_low_percentile: float = 1.5
    uv_high_percentile: float = 99.2
    uv_gamma: float = 0.90
    uv_pigment_darkening: float = 0.60
    uv_detail_gain: float = 0.22
    uv_black_level: float = 0.08
    uv_white_level: float = 0.82
    # 参考 VISIA 归档图的中性石墨灰，不使用偏蓝黑白滤镜。
    uv_channel_gains_bgr: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # 紫质荧光底图：与 UV 底图共享纹理，转为深蓝—青蓝的荧光色调。
    porphyrin_shadow_bgr: tuple[int, int, int] = (9, 3, 2)
    porphyrin_mid_bgr: tuple[int, int, int] = (82, 28, 8)
    porphyrin_highlight_bgr: tuple[int, int, int] = (176, 92, 28)
    porphyrin_pigment_darkening: float = 0.36
    porphyrin_detail_gain: float = 0.18
    porphyrin_glow_sigma: float = 2.2
    porphyrin_glow_gain: float = 0.10


@dataclass
class PurpleBaseStyleResult:
    """两种底图及其可复核中间结果。"""

    uv_spots_base: np.ndarray
    uv_spots_detection_base: np.ndarray
    porphyrin_base: np.ndarray
    porphyrin_detection_base: np.ndarray
    foreground_mask: np.ndarray
    pigment_proxy: np.ndarray


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8) * 255


def _fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    """只填内部孔洞，不向外膨胀人脸边界。"""
    binary = _binary_mask(mask)
    inverse = cv2.bitwise_not(binary)
    count, labels = cv2.connectedComponents((inverse > 0).astype(np.uint8), 8)
    if count <= 1:
        return binary
    border_labels = set(np.unique(labels[0, :]).tolist())
    border_labels.update(np.unique(labels[-1, :]).tolist())
    border_labels.update(np.unique(labels[:, 0]).tolist())
    border_labels.update(np.unique(labels[:, -1]).tolist())
    holes = np.zeros_like(binary)
    for label in range(1, count):
        if label not in border_labels:
            holes[labels == label] = 255
    return cv2.bitwise_or(binary, holes)


def _masked_percentile(
    values: np.ndarray,
    mask: np.ndarray,
    low: float,
    high: float,
) -> tuple[float, float]:
    samples = np.asarray(values, dtype=np.float32)[mask > 0]
    if samples.size < 32:
        return 0.0, 1.0
    lo, hi = np.percentile(samples, [low, high])
    return float(lo), max(float(hi), float(lo) + 1e-6)


def _normalize(
    values: np.ndarray,
    mask: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    lo, hi = _masked_percentile(values, mask, low, high)
    output = np.clip((values.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    output[mask == 0] = 0.0
    return output.astype(np.float32)


def _fill_outside(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.asarray(values).copy()
    samples = output[mask > 0]
    fill = np.median(samples, axis=0) if samples.size else 0
    output[mask == 0] = fill
    return output


def _piecewise_color(
    level: np.ndarray,
    shadow_bgr: tuple[int, int, int],
    mid_bgr: tuple[int, int, int],
    highlight_bgr: tuple[int, int, int],
) -> np.ndarray:
    values = np.clip(level.astype(np.float32), 0.0, 1.0)
    anchors = np.asarray(
        [shadow_bgr, mid_bgr, highlight_bgr],
        dtype=np.float32,
    )
    output = np.empty((*values.shape, 3), dtype=np.float32)
    for channel in range(3):
        output[:, :, channel] = np.interp(
            values,
            np.asarray([0.0, 0.52, 1.0], dtype=np.float32),
            anchors[:, channel],
        )
    return output


class PurpleBaseStyleRenderer:
    """从预处理人像生成两张 VISIA-like 工程展示底图。"""

    def __init__(self, config: PurpleBaseStyleConfig | None = None) -> None:
        self.config = config or PurpleBaseStyleConfig()

    def _prepare_foreground_mask(
        self,
        image: np.ndarray,
        foreground_mask: np.ndarray,
    ) -> np.ndarray:
        mask = _binary_mask(foreground_mask)
        # 排除仿射画布的纯黑像素，避免分割边缘将黑边带入色调统计。
        non_black = np.max(image, axis=2) > 3
        mask[~non_black] = 0
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        return mask

    def _tone_and_pigment(
        self,
        image: np.ndarray,
        foreground_mask: np.ndarray,
        skin_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        config = self.config
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_u8 = lab[:, :, 0].astype(np.uint8)
        statistics_mask = cv2.bitwise_and(_binary_mask(skin_mask), foreground_mask)
        if np.count_nonzero(statistics_mask) < 500:
            statistics_mask = foreground_mask
        else:
            # skin_mask 为算法统计会主动掏空眼眉、鼻孔和嘴唇。底图不应
            # 显示这些硬边孔洞，因此只在底图色调计算中闭合内部五官孔洞；
            # 后续实例检测仍使用原始 regions.analysis_mask 排除五官。
            statistics_mask = _fill_internal_holes(statistics_mask)
            statistics_mask = cv2.bitwise_and(statistics_mask, foreground_mask)

        # 检测区域和完整前景使用两套填充值。皮肤区域的色调、色度和细节
        # 只由有效皮肤统计产生；Mask 外背景、头发或衣物发生变化时，脸内
        # UV 底图及后续实例结果保持不变。
        l_foreground_filled = _fill_outside(l_u8, foreground_mask).astype(np.uint8)
        l_statistics_filled = _fill_outside(l_u8, statistics_mask).astype(np.uint8)

        clahe_operator = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=config.clahe_grid_size,
        )
        clahe_foreground = clahe_operator.apply(l_foreground_filled)
        clahe_statistics = clahe_operator.apply(l_statistics_filled)
        tone_foreground = _normalize(
            clahe_foreground.astype(np.float32) / 255.0,
            foreground_mask,
            config.uv_low_percentile,
            config.uv_high_percentile,
        )
        tone_statistics = _normalize(
            clahe_statistics.astype(np.float32) / 255.0,
            statistics_mask,
            config.uv_low_percentile,
            config.uv_high_percentile,
        )
        tone = tone_foreground.copy()
        tone[statistics_mask > 0] = tone_statistics[statistics_mask > 0]

        l = l_statistics_filled.astype(np.float32) / 255.0
        background = cv2.GaussianBlur(
            l,
            (0, 0),
            sigmaX=config.pigment_background_sigma,
            sigmaY=config.pigment_background_sigma,
            borderType=cv2.BORDER_REFLECT101,
        )
        local_dark = np.maximum(background - l, 0.0)

        blackhat_scales: list[np.ndarray] = []
        for size in config.pigment_blackhat_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            blackhat_scales.append(
                cv2.morphologyEx(l_statistics_filled, cv2.MORPH_BLACKHAT, kernel)
                .astype(np.float32)
                / 255.0
            )
        blackhat = np.max(np.stack(blackhat_scales, axis=0), axis=0)

        # LAB a*/b* 仅提供普通白光下的综合色差代理，背景从不参与统计。
        a = _fill_outside((lab[:, :, 1] - 128.0) / 127.0, statistics_mask)
        b = _fill_outside((lab[:, :, 2] - 128.0) / 127.0, statistics_mask)
        a_bg = cv2.GaussianBlur(a, (0, 0), sigmaX=18.0, sigmaY=18.0)
        b_bg = cv2.GaussianBlur(b, (0, 0), sigmaX=18.0, sigmaY=18.0)
        chroma = np.sqrt(np.square(a - a_bg) + np.square(b - b_bg))

        local_dark_n = _normalize(local_dark, statistics_mask, 20.0, 99.5)
        blackhat_n = _normalize(blackhat, statistics_mask, 20.0, 99.5)
        chroma_n = _normalize(chroma, statistics_mask, 20.0, 99.5)
        pigment = np.clip(
            config.pigment_dark_weight * local_dark_n
            + config.pigment_blackhat_weight * blackhat_n
            + config.pigment_chroma_weight * chroma_n,
            0.0,
            1.0,
        )
        pigment[foreground_mask == 0] = 0.0

        fine = cv2.GaussianBlur(l, (0, 0), sigmaX=0.8, sigmaY=0.8)
        medium = cv2.GaussianBlur(l, (0, 0), sigmaX=2.8, sigmaY=2.8)
        signed_detail = fine - medium
        # 非皮肤前景（头发、五官和衣物）只补回自身亮度纹理，不参与脸内
        # 检测统计，也不会把五官排除区显示成均匀色块。
        foreground_l = l_foreground_filled.astype(np.float32) / 255.0
        foreground_detail = cv2.GaussianBlur(
            foreground_l,
            (0, 0),
            sigmaX=0.8,
            sigmaY=0.8,
        ) - cv2.GaussianBlur(
            foreground_l,
            (0, 0),
            sigmaX=2.8,
            sigmaY=2.8,
        )
        outside_statistics = (foreground_mask > 0) & (statistics_mask == 0)
        signed_detail[outside_statistics] = foreground_detail[outside_statistics]
        detail_samples = np.abs(signed_detail[statistics_mask > 0])
        limit = float(np.percentile(detail_samples, 99.0)) if detail_samples.size else 0.02
        signed_detail = np.clip(signed_detail / max(limit, 1e-6), -1.0, 1.0)
        signed_detail[foreground_mask == 0] = 0.0
        return tone, pigment.astype(np.float32), signed_detail.astype(np.float32)

    def render(
        self,
        analysis_image: np.ndarray,
        foreground_mask: np.ndarray,
        skin_mask: np.ndarray,
    ) -> PurpleBaseStyleResult:
        image = np.asarray(analysis_image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("analysis_image 必须是 H×W×3 BGR 图像")
        if foreground_mask.shape[:2] != image.shape[:2]:
            raise ValueError("foreground_mask 与 analysis_image 尺寸不一致")
        if skin_mask.shape[:2] != image.shape[:2]:
            raise ValueError("skin_mask 与 analysis_image 尺寸不一致")

        config = self.config
        foreground = self._prepare_foreground_mask(image, foreground_mask)
        if np.count_nonzero(foreground) < 500:
            raise ValueError("人像前景面积不足，无法生成紫区底图")
        display_tone, display_pigment, display_detail = self._tone_and_pigment(
            image,
            foreground,
            foreground,
        )
        detection_tone, detection_pigment, detection_detail = self._tone_and_pigment(
            image,
            foreground,
            skin_mask,
        )
        face = _fill_internal_holes(
            cv2.bitwise_and(_binary_mask(skin_mask), foreground)
        )
        face = cv2.bitwise_and(face, foreground)
        face_core = cv2.erode(
            face,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        )
        face_alpha = cv2.GaussianBlur(
            (face_core > 0).astype(np.float32),
            (0, 0),
            sigmaX=5.0,
            sigmaY=5.0,
        )
        face_alpha = np.clip(face_alpha, 0.0, 1.0)

        def blend(display: np.ndarray, detection: np.ndarray) -> np.ndarray:
            return (
                display * (1.0 - face_alpha)
                + detection * face_alpha
            ).astype(np.float32)

        tone = blend(display_tone, detection_tone)
        pigment = blend(display_pigment, detection_pigment)
        detail = blend(display_detail, detection_detail)

        def render_uv(
            local_tone: np.ndarray,
            local_pigment: np.ndarray,
            local_detail: np.ndarray,
            black_level: float,
            white_level: float,
            gains_bgr: tuple[float, float, float],
        ) -> np.ndarray:
            level = np.clip(
                np.power(local_tone, config.uv_gamma)
                - config.uv_pigment_darkening * local_pigment
                + config.uv_detail_gain * local_detail,
                0.0,
                1.0,
            )
            level = black_level + (
                white_level - black_level
            ) * level
            gray = 255.0 * level
            return np.stack(
                [
                    gray * gains_bgr[0],
                    gray * gains_bgr[1],
                    gray * gains_bgr[2],
                ],
                axis=2,
            )

        uv = render_uv(
            tone,
            pigment,
            detail,
            config.uv_black_level,
            config.uv_white_level,
            config.uv_channel_gains_bgr,
        )
        uv_detection = render_uv(
            detection_tone,
            detection_pigment,
            detection_detail,
            _DETECTION_UV_BLACK_LEVEL,
            _DETECTION_UV_WHITE_LEVEL,
            _DETECTION_UV_GAINS_BGR,
        )

        display_porphyrin_level = np.clip(
            0.78 * display_tone
            - config.porphyrin_pigment_darkening * display_pigment
            + config.porphyrin_detail_gain * display_detail,
            0.0,
            1.0,
        )
        detection_porphyrin_level = np.clip(
            0.78 * detection_tone
            - config.porphyrin_pigment_darkening * detection_pigment
            + config.porphyrin_detail_gain * detection_detail,
            0.0,
            1.0,
        )
        porphyrin_level = np.clip(
            0.78 * tone
            - config.porphyrin_pigment_darkening * pigment
            + config.porphyrin_detail_gain * detail,
            0.0,
            1.0,
        )
        porphyrin = _piecewise_color(
            porphyrin_level,
            config.porphyrin_shadow_bgr,
            config.porphyrin_mid_bgr,
            config.porphyrin_highlight_bgr,
        )
        display_glow = cv2.GaussianBlur(
            display_porphyrin_level,
            (0, 0),
            sigmaX=config.porphyrin_glow_sigma,
            sigmaY=config.porphyrin_glow_sigma,
        )
        detection_glow = masked_gaussian(
            detection_porphyrin_level,
            face,
            config.porphyrin_glow_sigma,
        )
        glow = blend(display_glow, detection_glow)
        porphyrin[:, :, 0] += 255.0 * config.porphyrin_glow_gain * glow
        porphyrin[:, :, 1] += 120.0 * config.porphyrin_glow_gain * glow
        porphyrin_detection = _piecewise_color(
            detection_porphyrin_level,
            _DETECTION_PORPHYRIN_SHADOW_BGR,
            _DETECTION_PORPHYRIN_MID_BGR,
            _DETECTION_PORPHYRIN_HIGHLIGHT_BGR,
        )
        porphyrin_detection[:, :, 0] += (
            255.0 * config.porphyrin_glow_gain * detection_glow
        )
        porphyrin_detection[:, :, 1] += (
            120.0 * config.porphyrin_glow_gain * detection_glow
        )

        alpha = cv2.GaussianBlur(
            (foreground > 0).astype(np.float32),
            (0, 0),
            sigmaX=config.foreground_feather_sigma,
            sigmaY=config.foreground_feather_sigma,
        )
        alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
        uv = np.clip(uv * alpha, 0.0, 255.0).astype(np.uint8)
        uv_detection = np.clip(
            uv_detection * alpha,
            0.0,
            255.0,
        ).astype(np.uint8)
        porphyrin = np.clip(porphyrin * alpha, 0.0, 255.0).astype(np.uint8)
        porphyrin_detection = np.clip(
            porphyrin_detection * alpha,
            0.0,
            255.0,
        ).astype(np.uint8)
        return PurpleBaseStyleResult(
            uv_spots_base=uv,
            uv_spots_detection_base=uv_detection,
            porphyrin_base=porphyrin,
            porphyrin_detection_base=porphyrin_detection,
            foreground_mask=foreground,
            pigment_proxy=pigment,
        )
