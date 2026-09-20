# -*- coding: utf-8 -*-
"""红区检测引擎 (RBX V3) — 完整脸部分割 + 配对校准 + 连续细颗粒红度.

本模块只负责红区/RBX-like 出图。LAB 颜色映射仅在 RBX 内部使用，
不回写 Preprocessor 的 analysis_image，也不影响 Spots / Brown / Wrinkles。
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import mediapipe as mp
import numpy as np

from src.capture_profile import CaptureProfile

from src.engines.visia_regions import (
    VisiaRegionSet,
    build_nasolabial_shadow_mask,
    build_visia_regions,
    draw_region_boundaries,
)
from src.utils.detailed_metrics import (
    build_medical_payload,
    embed_medical_metrics_v2,
    write_medical_metrics,
    try_write_medical_metrics_v2,
)
from src.utils.analysis_debug_outputs import (
    capture_preprocess_debug,
    write_analysis_debug_outputs,
)
from src.utils.io_utils import cv_imread, cv_imwrite
from src.utils.gpu_backend import get_cuda_backend
from src.utils.model_loader import (
    load_face_landmarker,
    load_selfie_segmenter,
    load_selfie_multiclass_segmenter,
)

if TYPE_CHECKING:
    from src.preprocess.image_preprocessor import PreprocessResultV2


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Config:
    """RBX/V17.3 唯一核心调参区。

    A. 快速调参指南：
    1. 想让红区更明显：优先增大 RENDER_CONTRAST，
       或减小 BASE_GAMMA；前者只增强显示，后者会同时提高量化红度。
       如需突出细小红点，可适度增大 texture_weight 或 SCORE_DETAIL_WEIGHT。
    2. 想减少整脸泛红：增大 BASE_GAMMA、减小 BASE_GLOBAL_WEIGHT，
       或仅降低预设 contrast；前两项改变分析分数，后一项只改变显示。
    3. 想减少马赛克：适度增大 DEBLOCK_GUIDED_RADII 中的大尺度权重或
       BASE_SMOOTH_SIGMA；调得过大会损失细小红点和真实皮肤纹理。
    4. 想改变最终红区颜色：调整 COLOR_BASE 和 COLOR_RED。OpenCV 使用 BGR，
       这只改变展示风格，不改变连续红度图和量化指标。
    5. 当前版本默认关闭颜色映射。若后续确实需要减弱手机白平衡差异，
       可将 COLOR_TRANSFER_ENABLED 改为 True，再调整 COLOR_TRANSFER_STRENGTH；
       强度越大越接近参考图，也越可能改变原始肤色。
    6. 想调整人脸外轮廓：优先调整 FACE_MASK_CLOSE_KERNEL；最终轮廓由
       多类别 face-skin 语义分割决定，不再用地标凸包裁切额头。
    7. RED_AREA_THRESHOLD/HIGH_RED_AREA_THRESHOLD 只控制工程面积统计，不改变
       红区结果图。当前不代表医学炎症诊断或严重程度分级。
    8. 每次只改一组参数，并固定同一批图片比较结果图、强度图和 JSON 指标。

    约定：所有形态学/高斯核必须使用正奇数；除特别注明外，阈值调低通常会
    提高红区敏感度，同时增加光照、妆容、阴影和相机色偏带来的误差。
    """

    # =====================================================================
    # B. 红区伪彩颜色和白色背景（只影响展示，不改变量化）
    # =====================================================================
    # 正常皮肤底色（BGR）；降低 R 或提高 B 会使底色更偏冷。
    COLOR_BASE = (245, 235, 240)
    # 高红度区域颜色（BGR）；提高 R 会更鲜红，降低 B/G 会更深红。
    COLOR_RED = (15, 10, 160)
    # 结果图背景色（BGR）；只影响人像之外区域。
    COLOR_BG = (255, 255, 255)

    # =====================================================================
    # C. MediaPipe 多类别人脸分割与柔边合成
    # =====================================================================
    # Selfie Multiclass 模型中 face-skin 的类别编号，除非更换模型不要修改。
    FACE_SKIN_CLASS_ID = 3
    # 语义脸部闭运算核；调大更容易补齐额头/脸缘，但可能吞入少量发际线。
    FACE_MASK_CLOSE_KERNEL = 9
    # 语义脸部开运算核；调大去除更多碎片，但可能损失窄小侧脸区域。
    FACE_MASK_OPEN_KERNEL = 3
    # 五官恢复 Mask 的膨胀核；调大更不容易出现黑洞，但会扩大到周围皮肤。
    FEATURE_RESTORE_DILATE = 7
    # 完整脸部几何门的膨胀核；只用于排除与脸相连的手、耳外背景等，
    # 不负责裁额头。调大可保留更多脸缘，调小可更强地排除邻近物体。
    RENDER_GEOMETRY_DILATE = 31
    # 眉毛以上额头语义区域允许超出左右脸宽的比例；调大更能保留侧额角，
    # 也更可能纳入少量头发。
    RENDER_FOREHEAD_SIDE_MARGIN_RATIO = 0.16
    # 额头语义门向眉线下方的容许像素，避免额头与下半脸被切成两块。
    RENDER_FOREHEAD_BOTTOM_MARGIN = 32
    # 选取语义连通域时，与有效皮肤重叠的最低像素数。
    FACE_COMPONENT_MIN_OVERLAP = 64
    # 前景边缘高斯柔化核；调大边缘更柔但可能出现光晕，调小边缘更硬。
    BG_BLUR_SIZE = 31
    # 完整人像前景使用二分类人像分割；它只决定最终展示中保留哪些像素，
    # 不参与红度计算。提高阈值会去掉更多头发/衣物边缘，降低则更完整但
    # 可能带入少量背景。
    PERSON_FOREGROUND_THRESHOLD = 0.50
    # 前景的小孔/发丝断裂闭运算；调大更连贯，过大可能吞入背景缝隙。
    PERSON_FOREGROUND_CLOSE_KERNEL = 9
    # 仅保留与人脸语义 Mask 相连的人像连通域，避免把背景中的其他人带入。
    PERSON_FOREGROUND_FACE_OVERLAP_MIN = 64
    # 有效前景/统计 Mask 最少像素；调大更严格，调小可接受更小局部脸。
    MIN_VALID_MASK_PIXELS = 64
    # =====================================================================
    # D. RBX 专属 LAB 颜色映射
    # =====================================================================
    # 是否启用参考肤色映射；当前对照版本关闭，RBX 直接使用预处理后的
    # 原始 sRGB 人脸颜色。重新设为 True 才会映射到 VISIA 示例 1 输入基准。
    COLOR_TRANSFER_ENABLED = False
    # 映射强度 0~1；调大更接近参考图，调小更保留原始照片颜色。
    COLOR_TRANSFER_STRENGTH = 1.0
    # 接近 1 时直接返回映射图，避免无意义混合；通常不需要调整。
    COLOR_TRANSFER_FULL_STRENGTH_THRESHOLD = 0.999
    # LAB 标准差下限；调大限制颜色拉伸、结果更稳，调小映射更强但易偏色。
    MIN_LAB_STD = 2.0
    # LAB 统计最少有效像素；调大更稳定但小脸会回退全图，调小更易受噪声影响。
    MIN_LAB_STATS_PIXELS = 64
    # 按顺序选择参考图；只作用于 RBX，不影响 Spots 或通用预处理。
    REFERENCE_CANDIDATES = (
        PROJECT_ROOT / "data" / "calibration" / "rbx_reference.png",
        PROJECT_ROOT / "data" / "reference.jpg",
    )
    # 示例 1 输入/输出生成的正式配对配置；同时包含输入肤色与输出红度曲线。
    COLOR_PROFILE_PATH = PROJECT_ROOT / "data" / "calibration" / "rbx_visia_pair1_profile.json"
    # LAB 映射缩放上下限；范围越宽越接近参考分布，也越容易放大色度块。
    COLOR_TRANSFER_SCALE_MIN = 0.78
    COLOR_TRANSFER_SCALE_MAX = 1.22

    # =====================================================================
    # E. 去块、多证据红度和稳健归一化
    # =====================================================================
    # 三类红色证据权重；建议总和保持为 1。
    REDNESS_LAB_A_WEIGHT = 0.55
    REDNESS_NORMALIZED_RGB_WEIGHT = 0.30
    REDNESS_R_MINUS_G_WEIGHT = 0.15
    # 色度去块尺度。手机 JPEG 的色度通常按 8/16px 块压缩；这里先把色度
    # 平滑成连续的大范围红度，再从原始亮度恢复毛孔和红点，避免锐化马赛克。
    # 调大：色块更少但局部红色边界更柔；调小：保留更多色差，也更容易出块。
    CHROMA_SMOOTH_SIGMAS = (10.0, 18.0, 28.0)
    # 三个尺度的融合权重，建议总和保持为 1。
    CHROMA_SMOOTH_WEIGHTS = (0.15, 0.55, 0.30)
    # Mask-aware guided filter 保留为兼容工具，不再作为正式红度主路线。
    DEBLOCK_GUIDED_RADII = (3, 6, 12)
    DEBLOCK_GUIDED_WEIGHTS = (0.55, 0.30, 0.15)
    # guided filter 正则；调大更平滑，调小更贴合亮度边缘。
    DEBLOCK_GUIDED_EPS = 0.0025
    # 稳健标准差下限，防止近似纯色区域被无限放大。
    ROBUST_SCALE_FLOOR = 0.035
    # 分区 Z 仅保留给指标研究；正式显示不用分区归一化，避免额头/脸颊接缝。
    REGION_Z_WEIGHT = 0.0
    # 大范围红度的局部背景尺度。
    BASE_LOCAL_BACKGROUND_SIGMA = 28.0
    # 连续色度与局部异常在大范围层中的权重。
    BASE_GLOBAL_WEIGHT = 0.84
    BASE_LOCAL_WEIGHT = 0.16
    # 各色度证据的稳健分位范围；增大下限会减少整脸泛红，降低上限会增强对比。
    BASE_PERCENTILE_LOW = 5.0
    BASE_PERCENTILE_HIGH = 98.0
    # Gamma 小于 1 会增强中等红度，大于 1 会只保留更强红区。
    BASE_GAMMA = 0.88
    # 低频层轻度平滑，清除剩余块边，但不能把脸部结构整体模糊。
    BASE_SMOOTH_SIGMA = 1.2

    # =====================================================================
    # F. 细颗粒红区、真实亮度纹理和块状伪影抑制
    # =====================================================================
    # 色度/亮度细节的多尺度高斯尺度。
    DETAIL_FINE_SIGMA = 1.2
    DETAIL_MEDIUM_SIGMA = 16.0
    # 色度与暗色亮度证据权重。亮度细节权重更高，是为了从真实纹理恢复
    # VISIA 式细颗粒，而不是重新锐化已经受压缩的手机色度。
    DETAIL_CHROMA_WEIGHT = 0.34
    DETAIL_LUMA_WEIGHT = 0.66
    DETAIL_FINE_WEIGHT = 0.30
    DETAIL_MEDIUM_WEIGHT = 0.70
    # 亮度细节的稳健分位范围；调低 HIGH 会出现更多颗粒和噪声。
    DETAIL_PERCENTILE_LOW = 50.0
    DETAIL_PERCENTILE_HIGH = 99.5
    # 细节 Z 映射跨度；调小颗粒更强，调大更克制。
    DETAIL_Z_SPAN = 4.0
    # 最终分数中低频和细颗粒的权重。
    SCORE_BASE_WEIGHT = 0.62
    SCORE_DETAIL_WEIGHT = 0.38
    # 细颗粒只在已有一定红度的位置充分增强，防止阴影和睫毛主导结果。
    DETAIL_BASE_FLOOR = 0.65
    DETAIL_BASE_GAIN = 0.35
    # 局部暗红点采用峰值保留，防止在线性融合中被大范围红度冲淡。
    # WEIGHT 调大红点更深；GAMMA 调小弱小颗粒也更容易显现。
    DETAIL_PEAK_WEIGHT = 0.95
    DETAIL_PEAK_GAMMA = 0.75
    # 有效皮肤排除区的柔边尺度；使眼睛、眉毛和嘴唇保持可见而不出现硬白洞。
    STATISTICS_MASK_FEATHER_SIGMA = 1.5
    # 渲染用真实亮度纹理尺度与截断范围。
    TEXTURE_FINE_SIGMA = 0.75
    TEXTURE_MEDIUM_SIGMA = 2.5
    TEXTURE_FINE_WEIGHT = 0.75
    TEXTURE_MEDIUM_WEIGHT = 0.25
    TEXTURE_CLIP_LOW = -0.12
    TEXTURE_CLIP_HIGH = 0.12
    # 五档只改变最终展示对比度。分析分数图、量化指标、纹理、颜色和颗粒
    # 完全共用，便于人工判断“仅改变红区对比”带来的变化。
    #
    # 当前云端正式结果仍使用 high=1.12，即上一版已经确认的最强档。
    # soft/natural/balanced/ultra 仅用于本地对照，人工确认前不得改变云端默认。
    RENDER_PRESETS = {
        "soft": {
            "contrast": 0.60,
            "texture_weight": 1.85,
            "saturation": 1.05,
            "tone_weight": 0.12,
            "grain": 0.0,
        },
        "natural": {
            "contrast": 0.75,
            "texture_weight": 1.85,
            "saturation": 1.05,
            "tone_weight": 0.12,
            "grain": 0.0,
        },
        "balanced": {
            "contrast": 0.92,
            "texture_weight": 1.85,
            "saturation": 1.05,
            "tone_weight": 0.12,
            "grain": 0.0,
        },
        "high": {
            # 当前用户认可的原“自然”版强度，作为本轮最高对比档。
            "contrast": 1.12,
            "texture_weight": 1.85,
            "saturation": 1.05,
            "tone_weight": 0.12,
            "grain": 0.0,
        },
        "ultra": {
            "contrast": 1.35,
            "texture_weight": 1.85,
            "saturation": 1.05,
            "tone_weight": 0.12,
            "grain": 0.0,
        },
    }
    # LAB 映射只作用于有效皮肤；边缘轻柔过渡，避免眼周和唇周色阶接缝。
    COLOR_TRANSFER_MASK_FEATHER_SIGMA = 2.0
    # 将分析图原始亮度映射为 VISIA-like 红色密度；亮于 WHITE 接近白色，
    # 暗于 BLACK 接近深红。只影响渲染，不改变 redness_score_map。
    RENDER_TONE_WHITE_LEVEL = 0.92
    RENDER_TONE_BLACK_LEVEL = 0.18
    # 固定随机种子保证强对比预设重复运行一致。
    GRAIN_SEED = 20260714

    # 配对曲线查找表点数；必须与生成的配置一致。
    STYLE_LUT_SIZE = 64

    # =====================================================================
    # G. 人脸统计范围、额头补全和分区几何
    # =====================================================================
    # 人脸凸包膨胀核；调大统计范围更宽，可能纳入头发，调小可能漏脸缘。
    FACE_DILATE = 21
    # 额头向上延伸比例；调大保留更多额头，也更容易接近发际线。
    FOREHEAD_EXTENSION_RATIO = 0.8
    # 额头顶部半宽相对脸颊宽度；调大额头更宽，调小可能漏两侧额角。
    FOREHEAD_TOP_HALF_WIDTH_RATIO = 0.38
    # 缺少完整脸宽地标时，额头参考宽度占图宽比例；调大额头补全更宽。
    FOREHEAD_CHEEK_SPAN_FALLBACK_RATIO = 0.45
    # MediaPipe 额头几何地标；只控制统计 Mask，除非更换拓扑不要修改。
    FOREHEAD_BROW_LANDMARKS = (54, 103, 332, 284)
    FOREHEAD_TOP_LANDMARK = 10
    FACE_WIDTH_LANDMARKS = (454, 234)
    # 无完整地标时眉线/嘴线的纵向回退比例；调大意味着位置更靠下。
    REGION_BROW_FALLBACK_RATIO = 0.28
    REGION_MOUTH_FALLBACK_RATIO = 0.72
    # 下巴区域从嘴线向下偏移比例；调大下巴区更靠下、面积更小。
    REGION_CHIN_OFFSET_RATIO = 0.14
    # 鼻区膨胀核；调大更多鼻翼计入鼻部，调小更多归入脸颊。
    REGION_NOSE_DILATE_KERNEL = 25
    # MediaPipe 鼻区多边形；只影响分区统计，除非更换拓扑不要修改。
    REGION_NOSE_LANDMARKS = (
        168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 97, 98, 327, 326,
    )
    # 嘴部地标只用于确定下巴分区起点，不直接改变红度候选。
    REGION_MOUTH_LANDMARKS = (61, 291, 17)
    REGION_CENTER_LANDMARK = 1
    REGION_BROW_LANDMARKS = (105, 334)

    # =====================================================================
    # H. 工程量化阈值（不改变结果图，不代表严重等级）
    # =====================================================================
    # 红区面积阈值；调大红区面积比例下降，调小面积比例上升。
    RED_AREA_THRESHOLD = 0.45
    # 高红区阈值；调大只统计更强区域，调小高红区比例上升。
    HIGH_RED_AREA_THRESHOLD = 0.70
    # 指标分位点；修改会改变 JSON 字段语义，通常不要调整。
    METRIC_P50_PERCENTILE = 50.0
    METRIC_P90_PERCENTILE = 90.0
    METRIC_P95_PERCENTILE = 95.0
    # JSON 浮点保留位数；只影响输出格式，不影响实际检测。
    METRIC_DECIMALS = 8

    # =====================================================================
    # I. 离散红色区工程量化（只增加指标，不改变五档红区渲染）
    # =====================================================================
    # 多尺度峰值用于把连续红度图转换为可计数的局部红色异常实例。
    RED_FEATURE_LOCAL_SIGMAS = (1.5, 3.0, 6.0)
    # 实例红度的绝对下限；调大实例更少，调小会纳入更多轻微红度波动。
    RED_FEATURE_SCORE_FLOOR = 0.35
    # 至少两个尺度支持；强单尺度峰可由下面的强响应阈值保留。
    RED_FEATURE_MIN_SCALE_VOTES = 2
    RED_FEATURE_STRONG_Z = 3.60
    # 同一区域峰值间距；调大实例更稀疏，调小可能把一个红点重复计数。
    RED_FEATURE_PEAK_DISTANCE = 12
    # 实例面积上下限；只影响实例数量，不影响连续红度指标和结果图。
    RED_FEATURE_MIN_AREA = 10
    RED_FEATURE_MAX_AREA = 420
    # 峰值周围实例生长半径，保持“红点”而不是大面积“红斑面”。
    RED_FEATURE_RADIUS_BASE = 4
    RED_FEATURE_RADIUS_PER_SCALE = 1
    RED_FEATURE_RADIUS_MIN = 5
    RED_FEATURE_RADIUS_MAX = 8
    # 综合置信度下限；调大实例更可靠但可能漏掉较浅红点。
    RED_FEATURE_MIN_CONFIDENCE = 0.38

    # =====================================================================
    # J. 离散红色区五官/边界后处理（改变实例数量和量化结果）
    # =====================================================================
    # 五官遮挡向外膨胀像素（统一调参入口），默认 30。
    # 作用于眼睛/眼睑、眉毛、嘴唇和鼻孔的红色区实例后处理。
    # 调大：更强地清除五官及交界处误检，但可能漏掉紧邻五官的真实红点。
    # 调小：保留更多眼周、鼻翼和口周皮肤，也会增加五官边缘误检。
    RED_FEATURE_FEATURE_EXCLUDE_DILATE = 30
    # 候选与某类禁区重叠达到该比例时删除。调小过滤更严格。
    RED_FEATURE_FORBIDDEN_OVERLAP_RATIO = 0.06
    # 分析区边缘安全距离（像素）。过滤脸缘、五官孔洞和区域交界产生的
    # 峰值；调大误检更少，但会损失靠近边界的真实异常。
    RED_FEATURE_BOUNDARY_MARGIN = 4
    # 法令纹阴影后处理：只在 25px 解剖走廊内寻找真实暗线，并只过滤
    # 与暗线重叠且呈细长形态的实例。调低重叠率/长宽比会过滤更多，
    # 但可能误删鼻唇沟附近真实的紧凑红色丘疹。
    RED_FEATURE_NASOLABIAL_CORRIDOR_RADIUS_PX = 25
    RED_FEATURE_NASOLABIAL_SHADOW_MARGIN_PX = 5
    RED_FEATURE_NASOLABIAL_OVERLAP_RATIO = 0.25
    RED_FEATURE_NASOLABIAL_MIN_ASPECT_RATIO = 1.35
    RED_FEATURE_NASOLABIAL_CENTROID_MIN_ASPECT_RATIO = 1.80
    # 过于细长的候选更可能是睫毛、眉毛、发丝或结构交界。
    RED_FEATURE_MAX_ASPECT_RATIO = 3.2

    # =====================================================================
    # K. VISIA 式红色区实例叠加图（只影响 06 号展示图）
    # =====================================================================
    # 分析区轮廓颜色（OpenCV BGR），接近 VISIA 的青蓝色。
    RED_FEATURE_ZONE_COLOR = (222, 198, 65)
    # 红色区实例标记颜色（OpenCV BGR），接近 VISIA 的黄色。
    RED_FEATURE_MARKER_COLOR = (0, 228, 255)
    # 分析区轮廓线宽；只影响显示，不改变区域、数量和指标。
    RED_FEATURE_ZONE_THICKNESS = 2
    # 区域 Mask 闭运算核与柔化尺度；调大轮廓更平滑，但可能填平窄小缺口。
    RED_FEATURE_ZONE_CLOSE_KERNEL = 25
    RED_FEATURE_ZONE_SMOOTH_SIGMA = 6.0
    # Chaikin 曲线平滑次数；调大线条更圆润，也会逐渐偏离原始区域边界。
    RED_FEATURE_ZONE_CHAIKIN_ITERATIONS = 3
    # 忽略过小的轮廓碎片，避免青色边界在脸部形成杂乱短线。
    RED_FEATURE_ZONE_MIN_CONTOUR_AREA = 600.0
    # 每个分析区只画面积最大的外轮廓，杜绝 skin_mask 孔洞形成大量乱线。
    RED_FEATURE_ZONE_LARGEST_CONTOUR_ONLY = True
    # 黄色点按检测实例面积缩放，并限制显示半径，避免画成大块“红斑面”。
    RED_FEATURE_MARKER_RADIUS_SCALE = 0.55
    RED_FEATURE_MARKER_RADIUS_MIN = 2
    RED_FEATURE_MARKER_RADIUS_MAX = 7


class ErythemaAnalyzer:
    def __init__(
        self,
        color_transfer_enabled: bool | None = None,
        color_transfer_reference: str | Path | None = None,
        capture_profile: CaptureProfile = CaptureProfile.INSTITUTION,
    ):
        """Create an RBX analyzer with an instance-local colour-transfer mode.

        ``None`` keeps the production default from :class:`Config`.  Passing
        ``True``/``False`` lets a local test compare both modes in one process
        without modifying global configuration.  When an explicit reference
        is supplied, its face-only LAB distribution is used directly and the
        VISIA pair profile is intentionally bypassed.
        """
        self.face_landmarker = load_face_landmarker()
        self.segmenter = load_selfie_multiclass_segmenter()
        self.person_segmenter = load_selfie_segmenter()
        self.color_transfer_enabled = (
            Config.COLOR_TRANSFER_ENABLED
            if color_transfer_enabled is None
            else bool(color_transfer_enabled)
        )
        self.capture_profile = capture_profile
        self.color_transfer_reference: Path | None = None
        if color_transfer_reference is not None:
            reference_path = Path(color_transfer_reference).expanduser()
            if not reference_path.is_absolute():
                reference_path = PROJECT_ROOT / reference_path
            self.color_transfer_reference = reference_path.resolve()
        self._reference_image: np.ndarray | None = None
        self._reference_stats: tuple[np.ndarray, np.ndarray] | None = None
        self._reference_path: Path | None = None
        self._color_transfer_source_path: Path | None = None
        self._color_profile: dict | None = None

    def close(self) -> None:
        for model in (self.face_landmarker, self.segmenter, self.person_segmenter):
            try:
                model.close()
            except Exception:
                pass

    def _to_mp_image(self, image: np.ndarray) -> mp.Image:
        return mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        )

    def _detect_landmarks(self, image: np.ndarray):
        result = self.face_landmarker.detect(self._to_mp_image(image))
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    def _landmarks_to_points(self, image: np.ndarray, landmarks) -> np.ndarray:
        if isinstance(landmarks, np.ndarray):
            points = np.asarray(landmarks, dtype=np.float32)
            if points.ndim != 2 or points.shape[1] < 2:
                raise ValueError("landmarks numpy array must have shape (N, >=2)")
            return points[:, :2].copy()
        h, w = image.shape[:2]
        return np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)

    @staticmethod
    def _feature_restore_mask(
        shape: tuple[int, int],
        points: np.ndarray,
    ) -> np.ndarray:
        """Return visible facial features that must never become black holes."""
        feature_groups = (
            (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
            (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398),
            (46, 53, 52, 65, 55, 70, 63, 105, 66, 107),
            (276, 283, 282, 295, 285, 300, 293, 334, 296, 336),
            (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146),
        )
        mask = np.zeros(shape, dtype=np.uint8)
        for group in feature_groups:
            valid = [index for index in group if index < len(points)]
            if len(valid) < 3:
                continue
            polygon = cv2.convexHull(np.rint(points[valid]).astype(np.int32))
            cv2.fillConvexPoly(mask, polygon, 255)
        if Config.FEATURE_RESTORE_DILATE > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (Config.FEATURE_RESTORE_DILATE, Config.FEATURE_RESTORE_DILATE),
            )
            mask = cv2.dilate(mask, kernel)
        return mask

    def _semantic_face_mask(
        self,
        image: np.ndarray,
        landmarks,
        skin_mask: np.ndarray | None,
    ) -> np.ndarray:
        """Validate and preserve the complete preprocessed face silhouette.

        The input is already the correctly cropped 1024x1024 analysis image.
        Run the multiclass model on that intact image first; applying the
        feature-excluding ``skin_mask`` before semantic segmentation would
        punch holes into the face and can split a side face into fragments.
        The skin mask is used only after this step for colour statistics.
        """
        height, width = image.shape[:2]
        try:
            points = self._landmarks_to_points(image, landmarks)
            geometry = self.get_face_stats_mask(image, landmarks, None)
            geometry = cv2.dilate(
                geometry,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        Config.RENDER_GEOMETRY_DILATE,
                        Config.RENDER_GEOMETRY_DILATE,
                    ),
                ),
            )
            forehead_gate = np.zeros_like(geometry)
            if len(points) > max(
                *Config.FACE_WIDTH_LANDMARKS,
                *Config.REGION_BROW_LANDMARKS,
            ):
                left_x = float(points[Config.FACE_WIDTH_LANDMARKS[1], 0])
                right_x = float(points[Config.FACE_WIDTH_LANDMARKS[0], 0])
                x_min, x_max = sorted((left_x, right_x))
                span = max(x_max - x_min, 1.0)
                margin = Config.RENDER_FOREHEAD_SIDE_MARGIN_RATIO * span
                brow_y = max(
                    float(points[index, 1])
                    for index in Config.REGION_BROW_LANDMARKS
                )
                cv2.rectangle(
                    forehead_gate,
                    (
                        int(np.clip(x_min - margin, 0, width - 1)),
                        0,
                    ),
                    (
                        int(np.clip(x_max + margin, 0, width - 1)),
                        int(
                            np.clip(
                                brow_y + Config.RENDER_FOREHEAD_BOTTOM_MARGIN,
                                0,
                                height - 1,
                            )
                        ),
                    ),
                    255,
                    thickness=cv2.FILLED,
                )
            render_gate = cv2.bitwise_or(geometry, forehead_gate)
            segmentation_image = np.zeros_like(image)
            segmentation_image[render_gate > 0] = image[render_gate > 0]
            result = self.segmenter.segment(
                self._to_mp_image(segmentation_image)
            )
            if result.category_mask is None:
                raise ValueError("segmenter returned no category mask")
            categories = np.squeeze(result.category_mask.numpy_view())
            if categories.shape[:2] != (height, width):
                categories = cv2.resize(
                    categories.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            semantic = (
                categories.astype(np.uint8) == Config.FACE_SKIN_CLASS_ID
            ).astype(np.uint8) * 255
            semantic = cv2.morphologyEx(
                semantic,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        Config.FACE_MASK_CLOSE_KERNEL,
                        Config.FACE_MASK_CLOSE_KERNEL,
                    ),
                ),
            )
            semantic = cv2.morphologyEx(
                semantic,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        Config.FACE_MASK_OPEN_KERNEL,
                        Config.FACE_MASK_OPEN_KERNEL,
                    ),
                ),
            )

            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                (semantic > 0).astype(np.uint8),
                connectivity=8,
            )
            if count <= 1:
                raise ValueError("no face-skin component")

            valid_skin = (
                (skin_mask > 0).astype(np.uint8)
                if skin_mask is not None and skin_mask.shape == semantic.shape
                else np.zeros_like(semantic, dtype=np.uint8)
            )
            nose_x = nose_y = None
            if len(points) > Config.REGION_CENTER_LANDMARK:
                nose_x = int(np.clip(points[Config.REGION_CENTER_LANDMARK, 0], 0, width - 1))
                nose_y = int(np.clip(points[Config.REGION_CENTER_LANDMARK, 1], 0, height - 1))

            best_label = 0
            best_score = -1.0
            for label in range(1, count):
                component = labels == label
                area = int(stats[label, cv2.CC_STAT_AREA])
                overlap = int(np.count_nonzero(component & (valid_skin > 0)))
                contains_nose = bool(
                    nose_x is not None
                    and nose_y is not None
                    and component[nose_y, nose_x]
                )
                score = overlap * 4.0 + area + (area * 2.0 if contains_nose else 0.0)
                if score > best_score:
                    best_score = score
                    best_label = label
            if best_label == 0:
                raise ValueError("unable to select face-skin component")

            selected = (labels == best_label).astype(np.uint8) * 255
            if (
                skin_mask is not None
                and np.count_nonzero((selected > 0) & (skin_mask > 0))
                < Config.FACE_COMPONENT_MIN_OVERLAP
            ):
                raise ValueError("face-skin component has insufficient skin overlap")

            # Fill only holes enclosed by the selected semantic component.
            external_contours, _ = cv2.findContours(
                selected,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            filled = np.zeros_like(selected)
            cv2.drawContours(filled, external_contours, -1, 255, thickness=cv2.FILLED)

            # Restore eyes, eyebrows and lips without changing the outer face
            # silhouette.
            features = self._feature_restore_mask(selected.shape, points)
            filled = cv2.bitwise_or(filled, cv2.bitwise_and(features, filled))

            # The multiclass face class can connect a hand or ear-side skin to
            # the face.  Constrain only those lower/side extensions with the
            # already validated face geometry.  The upper forehead has its own
            # semantic gate reaching the canvas top, so this step cannot
            # reproduce the old landmark-hull forehead crop.
            filled = cv2.bitwise_and(filled, render_gate)

            if np.count_nonzero(filled) < Config.MIN_VALID_MASK_PIXELS:
                raise ValueError("semantic face mask is too small")
            return filled
        except Exception as exc:
            raise RuntimeError(f"RBX 多类别人脸分割失败: {exc}") from exc

    def _foreground_alpha(self, face_mask: np.ndarray) -> np.ndarray:
        """Return a soft alpha for a supplied display foreground mask."""
        blur = Config.BG_BLUR_SIZE if Config.BG_BLUR_SIZE % 2 else Config.BG_BLUR_SIZE + 1
        return cv2.GaussianBlur(face_mask.astype(np.float32), (blur, blur), 0) / 255.0

    def _person_foreground_mask(
        self,
        image: np.ndarray,
        face_mask: np.ndarray,
    ) -> np.ndarray:
        """Keep the visible person (including hair), while removing background.

        This mask is intentionally display-only.  All LAB statistics and
        redness calculations continue to use the feature-excluding skin mask.
        """
        height, width = image.shape[:2]
        try:
            result = self.person_segmenter.segment(self._to_mp_image(image))
            if not result.confidence_masks:
                raise ValueError("person segmenter returned no confidence mask")
            confidence = np.squeeze(
                result.confidence_masks[0].numpy_view()
            ).astype(np.float32)
            if confidence.shape[:2] != (height, width):
                confidence = cv2.resize(
                    confidence,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )
            person = (
                confidence >= Config.PERSON_FOREGROUND_THRESHOLD
            ).astype(np.uint8) * 255
            person = cv2.morphologyEx(
                person,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        Config.PERSON_FOREGROUND_CLOSE_KERNEL,
                        Config.PERSON_FOREGROUND_CLOSE_KERNEL,
                    ),
                ),
            )
            count, labels, _, _ = cv2.connectedComponentsWithStats(
                (person > 0).astype(np.uint8), connectivity=8
            )
            selected = np.zeros_like(person)
            best_overlap = 0
            for label in range(1, count):
                component = labels == label
                overlap = int(np.count_nonzero(component & (face_mask > 0)))
                if overlap > best_overlap:
                    best_overlap = overlap
                    selected[component] = 255
            if best_overlap < Config.PERSON_FOREGROUND_FACE_OVERLAP_MIN:
                raise ValueError("no person component overlaps the semantic face")
            # The semantic face must always remain visible even if the binary
            # person segmenter is conservative at a hairline or jaw boundary.
            return cv2.bitwise_or(selected, (face_mask > 0).astype(np.uint8) * 255)
        except Exception as exc:
            print(f"⚠️ RBX 人像前景分割失败，展示图回退到完整脸部: {exc}")
            return (face_mask > 0).astype(np.uint8) * 255

    def get_face_stats_mask(self, image: np.ndarray, landmarks, person_mask: np.ndarray | None = None) -> np.ndarray:
        """Face statistics mask with forehead completion, constrained by segmentation.

        Unlike the failed mask-style render, this mask is only used for robust
        percentile statistics and LAB mapping.  It is not visualized as holes.
        """
        h, w = image.shape[:2]
        points = self._landmarks_to_points(image, landmarks)
        mask = np.zeros((h, w), dtype=np.uint8)

        hull = cv2.convexHull(np.rint(points).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)

        if len(points) > max(Config.FOREHEAD_BROW_LANDMARKS):
            brow_y = float(
                min(points[index, 1] for index in Config.FOREHEAD_BROW_LANDMARKS)
            )
            top_y = float(points[Config.FOREHEAD_TOP_LANDMARK, 1])
            upper_y = top_y - Config.FOREHEAD_EXTENSION_RATIO * max(1.0, brow_y - top_y)
            centre_x = float(points[Config.FOREHEAD_TOP_LANDMARK, 0])
            width_left, width_right = Config.FACE_WIDTH_LANDMARKS
            cheek_span = (
                abs(float(points[width_left, 0] - points[width_right, 0]))
                if len(points) > max(Config.FACE_WIDTH_LANDMARKS)
                else w * Config.FOREHEAD_CHEEK_SPAN_FALLBACK_RATIO
            )
            half_width = Config.FOREHEAD_TOP_HALF_WIDTH_RATIO * max(cheek_span, 1.0)
            forehead = np.array(
                [
                    [np.clip(centre_x - half_width, 0, w - 1), np.clip(upper_y, 0, h - 1)],
                    [np.clip(centre_x + half_width, 0, w - 1), np.clip(upper_y, 0, h - 1)],
                    [
                        np.clip(points[Config.FOREHEAD_BROW_LANDMARKS[-1], 0], 0, w - 1),
                        np.clip(points[Config.FOREHEAD_BROW_LANDMARKS[-1], 1], 0, h - 1),
                    ],
                    [
                        np.clip(points[Config.FOREHEAD_BROW_LANDMARKS[0], 0], 0, w - 1),
                        np.clip(points[Config.FOREHEAD_BROW_LANDMARKS[0], 1], 0, h - 1),
                    ],
                ],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(mask, forehead, 255)

        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (Config.FACE_DILATE, Config.FACE_DILATE)),
        )
        if (
            person_mask is not None
            and np.count_nonzero(person_mask) > Config.MIN_VALID_MASK_PIXELS
        ):
            # Segmentation restores real visible forehead/face extent while
            # geometry prevents unrelated background from driving statistics.
            mask = cv2.bitwise_and(mask, person_mask)
            if np.count_nonzero(mask) < Config.MIN_VALID_MASK_PIXELS:
                mask = person_mask.copy()
        return mask

    def _select_reference_image(self) -> np.ndarray | None:
        if self._reference_image is not None:
            return self._reference_image
        candidates = (
            (self.color_transfer_reference,)
            if self.color_transfer_reference is not None
            else Config.REFERENCE_CANDIDATES
        )
        for candidate in candidates:
            if candidate is None:
                continue
            image = cv_imread(str(candidate))
            if image is not None:
                self._reference_image = image
                self._reference_path = candidate
                print(f"🎨 [RBX] 使用 LAB 参考图: {candidate}")
                return image
        print("⚠️ [RBX] 未找到正式 RBX 校准图或 data/reference.jpg，跳过 LAB 映射。")
        return None

    def _robust_lab_stats(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        valid = mask > 0
        if np.count_nonzero(valid) < Config.MIN_LAB_STATS_PIXELS:
            pixels = lab.reshape(-1, 3)
        else:
            pixels = lab[valid]
        centre = np.median(pixels, axis=0).astype(np.float32)
        q25, q75 = np.percentile(pixels, [25.0, 75.0], axis=0)
        robust_scale = ((q75 - q25) / 1.349).astype(np.float32)
        robust_scale = np.maximum(robust_scale, Config.MIN_LAB_STD)
        return centre, robust_scale

    def _load_color_profile(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._color_profile is not None:
            return (
                np.asarray(
                    self._color_profile.get(
                        "input_lab_center",
                        self._color_profile.get("lab_center"),
                    ),
                    dtype=np.float32,
                ),
                np.asarray(
                    self._color_profile.get(
                        "input_lab_scale",
                        self._color_profile.get("lab_scale"),
                    ),
                    dtype=np.float32,
                ),
            )
        path = Config.COLOR_PROFILE_PATH
        if not path.is_file():
            return None
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
            centre = np.asarray(
                profile.get("input_lab_center", profile.get("lab_center")),
                dtype=np.float32,
            )
            scale = np.asarray(
                profile.get("input_lab_scale", profile.get("lab_scale")),
                dtype=np.float32,
            )
            if centre.shape != (3,) or scale.shape != (3,):
                raise ValueError("lab_center/lab_scale must contain three values")
            self._color_profile = profile
            self._reference_path = path
            print(f"🎨 [RBX] 使用稳健 LAB 颜色配置: {path}")
            return centre, np.maximum(scale, Config.MIN_LAB_STD)
        except Exception as exc:
            print(f"⚠️ [RBX] 颜色配置读取失败，回退到参考图: {exc}")
            return None

    def _reference_lab_stats(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._reference_stats is not None:
            return self._reference_stats
        # An explicit image means "use the original reference-image method"
        # for input colour transfer.  The VISIA pair profile may still be
        # loaded independently by the common style LUT, so mapped/unmapped
        # comparisons differ only in the requested LAB input mapping.
        if self.color_transfer_reference is None:
            profile_stats = self._load_color_profile()
            if profile_stats is not None:
                self._reference_stats = profile_stats
                self._color_transfer_source_path = Config.COLOR_PROFILE_PATH
                return self._reference_stats
        reference = self._select_reference_image()
        if reference is None:
            return None
        landmarks = self._detect_landmarks(reference)
        if landmarks is not None:
            mask = self._semantic_face_mask(reference, landmarks, None)
        else:
            return None
        self._reference_stats = self._robust_lab_stats(reference, mask)
        self._color_transfer_source_path = self._reference_path
        return self._reference_stats

    def _rbx_lab_color_transfer(
        self,
        image: np.ndarray,
        stats_mask: np.ndarray,
        face_render_mask: np.ndarray,
    ) -> np.ndarray:
        """Map only valid facial skin to the VISIA input colour baseline.

        LAB statistics and mapped pixels both come from ``stats_mask``.
        Eyes, eyebrows, lips and nostrils retain their original pixels, while
        pixels outside ``face_render_mask`` are always black.
        """
        black_face = np.zeros_like(image)
        black_face[face_render_mask > 0] = image[face_render_mask > 0]
        if not self.color_transfer_enabled:
            return black_face
        reference_stats = self._reference_lab_stats()
        if reference_stats is None:
            return black_face

        src_mean, src_std = self._robust_lab_stats(image, stats_mask)
        ref_mean, ref_std = reference_stats
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        scale = np.clip(
            ref_std / np.maximum(src_std, Config.MIN_LAB_STD),
            Config.COLOR_TRANSFER_SCALE_MIN,
            Config.COLOR_TRANSFER_SCALE_MAX,
        )
        mapped = (lab - src_mean) * scale + ref_mean
        mapped = np.clip(mapped, 0, 255).astype(np.uint8)
        mapped_bgr = cv2.cvtColor(mapped, cv2.COLOR_LAB2BGR)
        strength = float(np.clip(Config.COLOR_TRANSFER_STRENGTH, 0.0, 1.0))
        if strength >= Config.COLOR_TRANSFER_FULL_STRENGTH_THRESHOLD:
            blended = mapped_bgr
        else:
            blended = cv2.addWeighted(
                mapped_bgr,
                strength,
                image,
                1.0 - strength,
                0,
            )
        render_float = (face_render_mask > 0).astype(np.float32)
        mapping_alpha = (stats_mask > 0).astype(np.float32)
        sigma = float(Config.COLOR_TRANSFER_MASK_FEATHER_SIGMA)
        if sigma > 0.0:
            mapping_alpha = cv2.GaussianBlur(
                mapping_alpha,
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
            )
        mapping_alpha = np.clip(mapping_alpha * render_float, 0.0, 1.0)
        result = (
            blended.astype(np.float32) * mapping_alpha[:, :, None]
            + image.astype(np.float32) * (1.0 - mapping_alpha[:, :, None])
        )
        result[face_render_mask <= 0] = 0.0
        return np.clip(result, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _masked_box_mean(
        values: np.ndarray,
        mask: np.ndarray,
        radius: int,
    ) -> np.ndarray:
        """Weighted box mean that never lets the white/black background leak in."""
        kernel = (radius * 2 + 1, radius * 2 + 1)
        weights = cv2.boxFilter(
            mask,
            cv2.CV_32F,
            kernel,
            normalize=False,
            borderType=cv2.BORDER_REFLECT,
        )
        weighted = cv2.boxFilter(
            values * mask,
            cv2.CV_32F,
            kernel,
            normalize=False,
            borderType=cv2.BORDER_REFLECT,
        )
        return weighted / np.maximum(weights, 1e-6)

    def _guided_filter_masked(
        self,
        guide: np.ndarray,
        source: np.ndarray,
        mask: np.ndarray,
        radius: int,
    ) -> np.ndarray:
        """Edge-aware colour deblocking guided by the real luminance structure."""
        mean_i = self._masked_box_mean(guide, mask, radius)
        mean_p = self._masked_box_mean(source, mask, radius)
        corr_i = self._masked_box_mean(guide * guide, mask, radius)
        corr_ip = self._masked_box_mean(guide * source, mask, radius)
        variance_i = np.maximum(corr_i - mean_i * mean_i, 0.0)
        covariance_ip = corr_ip - mean_i * mean_p
        coefficient_a = covariance_ip / (variance_i + Config.DEBLOCK_GUIDED_EPS)
        coefficient_b = mean_p - coefficient_a * mean_i
        mean_a = self._masked_box_mean(coefficient_a, mask, radius)
        mean_b = self._masked_box_mean(coefficient_b, mask, radius)
        filtered = mean_a * guide + mean_b
        return np.where(mask > 0, filtered, source).astype(np.float32)

    def _multiscale_deblock(
        self,
        channel: np.ndarray,
        luminance: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        outputs = [
            self._guided_filter_masked(luminance, channel, mask, radius)
            for radius in Config.DEBLOCK_GUIDED_RADII
        ]
        result = np.zeros_like(channel, dtype=np.float32)
        for weight, output in zip(Config.DEBLOCK_GUIDED_WEIGHTS, outputs):
            result += float(weight) * output
        return result

    @staticmethod
    def _masked_gaussian(
        channel: np.ndarray,
        mask: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        # This helper historically leaves the normalized neighbourhood value
        # available just outside the mask.  Several later face-wide smoothing
        # operations intentionally consume that feathered boundary, so keep
        # that exact CPU behaviour while moving only the convolution to CUDA.
        return get_cuda_backend().masked_gaussian(
            channel,
            mask,
            sigma,
            zero_outside=False,
        )

    @staticmethod
    def _robust_z_map(
        channel: np.ndarray,
        stats_mask: np.ndarray,
    ) -> np.ndarray:
        valid = stats_mask > 0
        values = channel[valid]
        if values.size < Config.MIN_VALID_MASK_PIXELS:
            return np.zeros_like(channel, dtype=np.float32)
        centre = float(np.median(values))
        mad = float(np.median(np.abs(values - centre)))
        scale = max(1.4826 * mad, Config.ROBUST_SCALE_FLOOR)
        return ((channel - centre) / scale).astype(np.float32)

    def _region_aware_z(
        self,
        evidence: np.ndarray,
        image: np.ndarray,
        landmarks,
        stats_mask: np.ndarray,
    ) -> np.ndarray:
        global_z = self._robust_z_map(evidence, stats_mask)
        region_z = global_z.copy()
        covered = np.zeros(stats_mask.shape, dtype=bool)
        for region_mask in self._region_masks(image, landmarks, stats_mask).values():
            valid = region_mask > 0
            if np.count_nonzero(valid) < Config.MIN_VALID_MASK_PIXELS:
                continue
            values = evidence[valid]
            centre = float(np.median(values))
            mad = float(np.median(np.abs(values - centre)))
            scale = max(1.4826 * mad, Config.ROBUST_SCALE_FLOOR)
            region_z[valid] = (evidence[valid] - centre) / scale
            covered |= valid
        region_z[~covered] = global_z[~covered]
        weight = float(np.clip(Config.REGION_Z_WEIGHT, 0.0, 1.0))
        return ((1.0 - weight) * global_z + weight * region_z).astype(np.float32)

    def _positive_detail_score(
        self,
        channel: np.ndarray,
        stats_mask: np.ndarray,
    ) -> np.ndarray:
        z_map = self._robust_z_map(channel, stats_mask)
        return np.clip(z_map / Config.DETAIL_Z_SPAN, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _masked_percentile_normalize(
        channel: np.ndarray,
        stats_mask: np.ndarray,
        low_percentile: float,
        high_percentile: float,
    ) -> np.ndarray:
        """Normalize a continuous signal without letting background affect it."""
        valid = stats_mask > 0
        values = channel[valid]
        if values.size < Config.MIN_VALID_MASK_PIXELS:
            return np.zeros_like(channel, dtype=np.float32)
        low, high = np.percentile(
            values,
            [low_percentile, high_percentile],
        )
        scale = max(float(high - low), 1e-6)
        return np.clip((channel - float(low)) / scale, 0.0, 1.0).astype(
            np.float32
        )

    def _continuous_chroma(
        self,
        channel: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Suppress JPEG chroma blocks while preserving the face-wide gradient."""
        result = np.zeros_like(channel, dtype=np.float32)
        for sigma, weight in zip(
            Config.CHROMA_SMOOTH_SIGMAS,
            Config.CHROMA_SMOOTH_WEIGHTS,
        ):
            result += float(weight) * self._masked_gaussian(
                channel,
                mask,
                float(sigma),
            )
        return result

    def _build_redness_maps(
        self,
        image: np.ndarray,
        stats_mask: np.ndarray,
        person_mask: np.ndarray,
        landmarks,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create continuous redness plus real, luminance-supported fine detail.

        The previous V3 draft normalized forehead/cheeks separately. Although
        robust for metrics, those independent baselines exposed region seams
        and JPEG chroma blocks after strong VISIA-style rendering. The formal
        path below intentionally keeps one continuous face-wide colour field:
        colour determines broad redness, while original luminance restores
        pores and small points.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        luminance = lab[:, :, 0] / 255.0
        lab_a = (lab[:, :, 1] - 128.0) / 127.0

        bgr = image.astype(np.float32) / 255.0
        blue, green, red = cv2.split(bgr)
        rgb_sum = np.maximum(red + green + blue, 1e-6)
        normalized_red = (red - green) / rgb_sum
        red_minus_green = red - green

        signal_mask = (person_mask > 0).astype(np.float32)
        stats_float = (stats_mask > 0).astype(np.float32)
        if np.count_nonzero(signal_mask) < Config.MIN_VALID_MASK_PIXELS:
            signal_mask = stats_float.copy()

        lab_a_clean = self._continuous_chroma(lab_a, signal_mask)
        normalized_red_clean = self._continuous_chroma(
            normalized_red,
            signal_mask,
        )
        rg_clean = self._continuous_chroma(red_minus_green, signal_mask)

        lab_a_score = self._masked_percentile_normalize(
            lab_a_clean,
            stats_mask,
            Config.BASE_PERCENTILE_LOW,
            Config.BASE_PERCENTILE_HIGH,
        )
        normalized_red_score = self._masked_percentile_normalize(
            normalized_red_clean,
            stats_mask,
            Config.BASE_PERCENTILE_LOW,
            Config.BASE_PERCENTILE_HIGH,
        )
        rg_score = self._masked_percentile_normalize(
            rg_clean,
            stats_mask,
            Config.BASE_PERCENTILE_LOW,
            Config.BASE_PERCENTILE_HIGH,
        )
        evidence = (
            Config.REDNESS_LAB_A_WEIGHT * lab_a_score
            + Config.REDNESS_NORMALIZED_RGB_WEIGHT * normalized_red_score
            + Config.REDNESS_R_MINUS_G_WEIGHT * rg_score
        )

        local_background = self._masked_gaussian(
            evidence,
            stats_float,
            Config.BASE_LOCAL_BACKGROUND_SIGMA,
        )
        local_excess = np.maximum(evidence - local_background, 0.0)
        local_score = self._masked_percentile_normalize(
            local_excess,
            stats_mask,
            50.0,
            99.0,
        )
        global_score = np.power(
            np.clip(evidence, 0.0, 1.0),
            Config.BASE_GAMMA,
        )
        base_map = (
            Config.BASE_GLOBAL_WEIGHT * global_score
            + Config.BASE_LOCAL_WEIGHT * local_score
        )
        base_map = cv2.GaussianBlur(
            base_map.astype(np.float32),
            (0, 0),
            sigmaX=Config.BASE_SMOOTH_SIGMA,
            sigmaY=Config.BASE_SMOOTH_SIGMA,
        )

        raw_chroma = (
            Config.REDNESS_LAB_A_WEIGHT * lab_a
            + Config.REDNESS_NORMALIZED_RGB_WEIGHT * normalized_red
            + Config.REDNESS_R_MINUS_G_WEIGHT * red_minus_green
        )
        chroma_fine = np.maximum(
            raw_chroma
            - self._masked_gaussian(
                raw_chroma,
                signal_mask,
                Config.DETAIL_FINE_SIGMA,
            ),
            0.0,
        )
        chroma_medium = np.maximum(
            raw_chroma
            - self._masked_gaussian(
                raw_chroma,
                signal_mask,
                Config.DETAIL_MEDIUM_SIGMA,
            ),
            0.0,
        )
        chroma_raw = (
            Config.DETAIL_FINE_WEIGHT * chroma_fine
            + Config.DETAIL_MEDIUM_WEIGHT * chroma_medium
        )
        chroma_detail = self._masked_percentile_normalize(
            chroma_raw,
            stats_mask,
            Config.DETAIL_PERCENTILE_LOW,
            Config.DETAIL_PERCENTILE_HIGH,
        )

        luma_fine = np.maximum(
            self._masked_gaussian(
                luminance,
                signal_mask,
                Config.DETAIL_FINE_SIGMA,
            )
            - luminance,
            0.0,
        )
        luma_medium = np.maximum(
            self._masked_gaussian(
                luminance,
                signal_mask,
                Config.DETAIL_MEDIUM_SIGMA,
            )
            - luminance,
            0.0,
        )
        luma_raw = (
            Config.DETAIL_FINE_WEIGHT
            * luma_fine
            + Config.DETAIL_MEDIUM_WEIGHT
            * luma_medium
        )
        luma_detail = self._masked_percentile_normalize(
            luma_raw,
            stats_mask,
            Config.DETAIL_PERCENTILE_LOW,
            Config.DETAIL_PERCENTILE_HIGH,
        )

        detail_map = (
            Config.DETAIL_CHROMA_WEIGHT * chroma_detail
            + Config.DETAIL_LUMA_WEIGHT * luma_detail
        )
        detail_map *= (
            Config.DETAIL_BASE_FLOOR
            + Config.DETAIL_BASE_GAIN * np.clip(base_map, 0.0, 1.0)
        )
        detail_map = cv2.GaussianBlur(
            detail_map.astype(np.float32),
            (0, 0),
            sigmaX=0.35,
            sigmaY=0.35,
        )

        score_map = np.clip(
            Config.SCORE_BASE_WEIGHT * base_map
            + Config.SCORE_DETAIL_WEIGHT * detail_map,
            0.0,
            1.0,
        )
        # ``stats_mask`` excludes eyes/brows/lips so they cannot bias the
        # baseline, but it must never punch geometric holes into the visible
        # RBX result.  Outside valid skin statistics keep the continuous
        # colour base and only fade the anomaly detail.
        statistics_alpha = cv2.GaussianBlur(
            stats_float,
            (0, 0),
            sigmaX=Config.STATISTICS_MASK_FEATHER_SIGMA,
            sigmaY=Config.STATISTICS_MASK_FEATHER_SIGMA,
        )
        statistics_alpha = np.clip(statistics_alpha, 0.0, 1.0)
        detail_map *= statistics_alpha
        score_map = np.clip(
            Config.SCORE_BASE_WEIGHT * base_map
            + Config.SCORE_DETAIL_WEIGHT * detail_map,
            0.0,
            1.0,
        )
        detail_peak = Config.DETAIL_PEAK_WEIGHT * np.power(
            np.clip(detail_map, 0.0, 1.0),
            Config.DETAIL_PEAK_GAMMA,
        )
        score_map = np.maximum(score_map, detail_peak)
        score_map[signal_mask <= 0] = 0.0
        base_map[signal_mask <= 0] = 0.0
        detail_map[signal_mask <= 0] = 0.0
        return (
            base_map.astype(np.float32),
            detail_map.astype(np.float32),
            score_map.astype(np.float32),
            luminance.astype(np.float32),
        )

    def _apply_style_lut(self, raw_score_map: np.ndarray) -> np.ndarray:
        """Map raw redness to the VISIA pair-1 target density."""
        if self._color_profile is None:
            self._load_color_profile()
        profile = self._color_profile or {}
        source = np.asarray(profile.get("style_lut_input", []), dtype=np.float32)
        target = np.asarray(profile.get("style_lut_output", []), dtype=np.float32)
        if (
            source.ndim != 1
            or target.ndim != 1
            or source.size < 2
            or source.size != target.size
        ):
            return np.clip(raw_score_map, 0.0, 1.0).astype(np.float32)
        mapped = np.interp(
            np.clip(raw_score_map, 0.0, 1.0),
            source,
            target,
            left=float(target[0]),
            right=float(target[-1]),
        )
        return np.clip(mapped, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _visia_target_density(
        target_image: np.ndarray,
        face_mask: np.ndarray,
    ) -> np.ndarray:
        """Convert an aligned VISIA output into a continuous red-density map."""
        valid = face_mask > 0
        if np.count_nonzero(valid) < Config.MIN_VALID_MASK_PIXELS:
            raise ValueError("VISIA target face mask is too small")
        lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        lightness = lab[:, :, 0] / 255.0
        redness = (lab[:, :, 1] - 128.0) / 127.0

        l_low, l_high = np.percentile(lightness[valid], [5.0, 95.0])
        a_low, a_high = np.percentile(redness[valid], [5.0, 95.0])
        dark_density = np.clip(
            (float(l_high) - lightness) / max(float(l_high - l_low), 1e-6),
            0.0,
            1.0,
        )
        red_density = np.clip(
            (redness - float(a_low)) / max(float(a_high - a_low), 1e-6),
            0.0,
            1.0,
        )
        density = np.clip(0.78 * dark_density + 0.22 * red_density, 0.0, 1.0)
        density[~valid] = 0.0
        return density.astype(np.float32)

    def build_visia_pair_profile(
        self,
        preprocess_result: "PreprocessResultV2",
        aligned_target: np.ndarray,
        source_name: str,
        target_name: str,
        source_sha256: str,
        target_sha256: str,
    ) -> dict:
        """Build the deployable colour/style profile from VISIA pair 1."""
        image = np.asarray(preprocess_result.analysis_image)
        landmarks = np.asarray(preprocess_result.landmarks, dtype=np.float32)
        skin_mask = (
            np.asarray(preprocess_result.skin_mask) > 0
        ).astype(np.uint8) * 255
        if aligned_target.shape != image.shape:
            raise ValueError("aligned VISIA target must match analysis_image")

        face_mask = self._semantic_face_mask(image, landmarks, skin_mask)
        stats_mask = cv2.bitwise_and(face_mask, skin_mask)
        if np.count_nonzero(stats_mask) < Config.MIN_VALID_MASK_PIXELS:
            raise ValueError("VISIA pair 1 valid skin area is too small")

        black_face = np.zeros_like(image)
        black_face[face_mask > 0] = image[face_mask > 0]
        _, _, raw_score, _ = self._build_redness_maps(
            black_face,
            stats_mask,
            face_mask,
            landmarks,
        )
        target_density = self._visia_target_density(aligned_target, face_mask)

        valid = stats_mask > 0
        quantiles = np.linspace(0.0, 1.0, Config.STYLE_LUT_SIZE)
        lut_input = np.quantile(raw_score[valid], quantiles).astype(np.float64)
        lut_output = np.quantile(target_density[valid], quantiles).astype(np.float64)
        lut_input[0], lut_input[-1] = 0.0, 1.0
        lut_output = np.maximum.accumulate(np.clip(lut_output, 0.0, 1.0))
        # np.interp accepts equal x values, but a strictly increasing LUT is
        # easier to audit and stable across NumPy versions.
        for index in range(1, len(lut_input)):
            lut_input[index] = max(
                lut_input[index],
                lut_input[index - 1] + 1e-6,
            )
        lut_input = np.clip(lut_input, 0.0, 1.0)
        lut_input[-1] = 1.0

        centre, scale = self._robust_lab_stats(image, stats_mask)
        target_pixels = aligned_target[face_mask > 0]
        target_values = target_density[face_mask > 0]
        low_group = target_pixels[
            target_values <= np.percentile(target_values, 5.0)
        ]
        high_group = target_pixels[
            target_values >= np.percentile(target_values, 95.0)
        ]
        light_color = np.median(low_group, axis=0)
        dark_color = np.median(high_group, axis=0)

        return {
            "profile_version": "rbx_visia_pair1_v3",
            "description": "由 VISIA 示例 1 输入/输出在同一预处理坐标系中生成的 RBX-like 配对配置。",
            "source_name": source_name,
            "target_name": target_name,
            "source_sha256": source_sha256,
            "target_sha256": target_sha256,
            "input_lab_center": [float(value) for value in centre],
            "input_lab_scale": [float(value) for value in scale],
            "style_lut_input": [float(value) for value in lut_input],
            "style_lut_output": [float(value) for value in lut_output],
            "light_color_bgr": [
                int(round(float(value))) for value in light_color
            ],
            "dark_color_bgr": [
                int(round(float(value))) for value in dark_color
            ],
            "target_density_quantiles": {
                "p50": float(np.percentile(target_density[valid], 50.0)),
                "p90": float(np.percentile(target_density[valid], 90.0)),
                "p95": float(np.percentile(target_density[valid], 95.0)),
            },
            "face_render_area": int(np.count_nonzero(face_mask)),
            "face_statistics_area": int(np.count_nonzero(stats_mask)),
            "face_class_id": int(Config.FACE_SKIN_CLASS_ID),
            "lut_size": int(Config.STYLE_LUT_SIZE),
            "disclaimer": "仅用于 VISIA-like 工程渲染，不代表 VISIA 官方 RBX 或医学诊断。",
        }

    def _render_result(
        self,
        score_map: np.ndarray,
        luminance: np.ndarray,
        person_foreground_alpha: np.ndarray,
        face_render_alpha: np.ndarray,
        source_image: np.ndarray,
        preset_name: str,
    ) -> np.ndarray:
        preset = Config.RENDER_PRESETS[preset_name]
        redness_density = np.clip(
            score_map * float(preset["contrast"]),
            0.0,
            1.0,
        )
        tone_density = np.clip(
            (
                Config.RENDER_TONE_WHITE_LEVEL
                - luminance
            )
            / max(
                Config.RENDER_TONE_WHITE_LEVEL
                - Config.RENDER_TONE_BLACK_LEVEL,
                1e-6,
            ),
            0.0,
            1.0,
        )
        tone_weight = float(preset["tone_weight"])
        display_density = np.clip(
            tone_weight * tone_density
            + (1.0 - tone_weight) * redness_density,
            0.0,
            1.0,
        )
        density_3d = display_density[:, :, None]

        profile = self._color_profile or {}
        color_base = np.asarray(
            profile.get("light_color_bgr", Config.COLOR_BASE),
            dtype=np.float32,
        )
        color_red = np.asarray(
            profile.get("dark_color_bgr", Config.COLOR_RED),
            dtype=np.float32,
        )
        saturation = float(preset["saturation"])
        target_red = color_base + saturation * (color_red - color_base)
        canvas = color_base * (1.0 - density_3d) + target_red * density_3d

        fine_background = cv2.GaussianBlur(
            luminance,
            (0, 0),
            sigmaX=Config.TEXTURE_FINE_SIGMA,
            sigmaY=Config.TEXTURE_FINE_SIGMA,
        )
        medium_background = cv2.GaussianBlur(
            luminance,
            (0, 0),
            sigmaX=Config.TEXTURE_MEDIUM_SIGMA,
            sigmaY=Config.TEXTURE_MEDIUM_SIGMA,
        )
        texture = (
            Config.TEXTURE_FINE_WEIGHT * (luminance - fine_background)
            + Config.TEXTURE_MEDIUM_WEIGHT * (luminance - medium_background)
        )
        texture = np.clip(
            texture,
            Config.TEXTURE_CLIP_LOW,
            Config.TEXTURE_CLIP_HIGH,
        )
        canvas += (
            texture[:, :, None]
            * 255.0
            * float(preset["texture_weight"])
        )

        grain = float(preset["grain"])
        if grain > 0.0:
            rng = np.random.default_rng(Config.GRAIN_SEED)
            noise = rng.normal(0.0, 1.0, score_map.shape).astype(np.float32)
            canvas += noise[:, :, None] * 255.0 * grain * density_3d

        del person_foreground_alpha, face_render_alpha, source_image
        # 展示底图覆盖完整对齐图；颜色归一化和指标统计仍只由
        # 脸部有效Mask决定，背景不参与分位数或评分计算。
        return np.clip(canvas, 0.0, 255.0).astype(np.uint8)

    def _polygon_mask(self, shape: tuple[int, int], points: np.ndarray, indices: list[int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        valid = [index for index in indices if index < len(points)]
        if len(valid) < 3:
            return mask
        polygon = cv2.convexHull(np.rint(points[valid]).astype(np.int32))
        cv2.fillConvexPoly(mask, polygon, 255)
        return mask

    @staticmethod
    def _dilate_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
        """Dilate outwards by an explicit pixel radius.

        This intentionally matches ``wrinkle_detection_algorithm.py``:
        ``dilate_px=25`` means a 25-pixel outward margin and therefore uses a
        51x51 elliptical kernel.
        """

        radius = max(0, int(dilate_px))
        if radius <= 0:
            return (mask > 0).astype(np.uint8) * 255
        size = 2 * radius + 1
        return cv2.dilate(
            (mask > 0).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        )

    def _build_red_feature_postprocess_masks(
        self,
        image: np.ndarray,
        landmarks,
        analysis_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Build anatomical forbidden zones for red-feature postprocessing.

        The strategy mirrors the wrinkle detector's rule-based postprocessing:
        landmark polygons define the eyes, eyebrows and lips, while the shared
        scale-aware nostril detector refines the two real nostril apertures.
        These masks remove instances from the metrics themselves; they are not
        merely cosmetic overlay masks.
        """

        from src.preprocess.image_preprocessor import (
            LEFT_EYE,
            LEFT_EYEBROW,
            LIPS,
            RIGHT_EYE,
            RIGHT_EYEBROW,
            build_nostril_exclusion_mask,
        )

        points = self._landmarks_to_points(image, landmarks)
        shape = analysis_mask.shape
        left_eye = self._polygon_mask(shape, points, list(LEFT_EYE))
        right_eye = self._polygon_mask(shape, points, list(RIGHT_EYE))
        left_brow = self._polygon_mask(shape, points, list(LEFT_EYEBROW))
        right_brow = self._polygon_mask(shape, points, list(RIGHT_EYEBROW))
        lips = self._polygon_mask(shape, points, list(LIPS))

        # Fill the anatomical strip from eyelid to eyebrow. This prevents the
        # common false positives at the eye/brow junction that separate masks
        # leave in the narrow gap between the two features.
        left_ocular_hull = self._polygon_mask(
            shape,
            points,
            list(LEFT_EYE) + list(LEFT_EYEBROW),
        )
        right_ocular_hull = self._polygon_mask(
            shape,
            points,
            list(RIGHT_EYE) + list(RIGHT_EYEBROW),
        )

        eyes = self._dilate_mask(
            cv2.bitwise_or(left_ocular_hull, right_ocular_hull),
            Config.RED_FEATURE_FEATURE_EXCLUDE_DILATE,
        )
        eyebrows = self._dilate_mask(
            cv2.bitwise_or(left_brow, right_brow),
            Config.RED_FEATURE_FEATURE_EXCLUDE_DILATE,
        )
        lips_forbidden = self._dilate_mask(
            lips,
            Config.RED_FEATURE_FEATURE_EXCLUDE_DILATE,
        )
        nostrils = build_nostril_exclusion_mask(
            image,
            points,
            use_photometric_refinement=True,
        )
        nostrils = self._dilate_mask(
            nostrils,
            Config.RED_FEATURE_FEATURE_EXCLUDE_DILATE,
        )

        binary_analysis = (analysis_mask > 0).astype(np.uint8) * 255
        boundary_margin = max(1, int(Config.RED_FEATURE_BOUNDARY_MARGIN))
        boundary_kernel = 2 * boundary_margin + 1
        safe_interior = cv2.erode(
            binary_analysis,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (boundary_kernel, boundary_kernel),
            ),
        )
        boundary = cv2.bitwise_and(
            binary_analysis,
            cv2.bitwise_not(safe_interior),
        )
        nasolabial = build_nasolabial_shadow_mask(
            image,
            points,
            binary_analysis,
            radius_px=Config.RED_FEATURE_NASOLABIAL_CORRIDOR_RADIUS_PX,
            output_margin_px=Config.RED_FEATURE_NASOLABIAL_SHADOW_MARGIN_PX,
            restrict_frangi_to_corridor=True,
        )

        forbidden = np.zeros(shape, dtype=np.uint8)
        for mask in (
            eyes,
            eyebrows,
            lips_forbidden,
            nostrils,
            boundary,
            nasolabial,
        ):
            forbidden = cv2.bitwise_or(forbidden, mask)

        return {
            "eyes": eyes,
            "eyebrows": eyebrows,
            "lips": lips_forbidden,
            "nostrils": nostrils,
            "boundary": boundary,
            "nasolabial": nasolabial,
            "forbidden": forbidden,
        }

    def _region_masks(self, image: np.ndarray, landmarks, stats_mask: np.ndarray) -> dict[str, np.ndarray]:
        """Build coarse VISIA-like regions for metrics only."""
        height, width = stats_mask.shape
        valid = stats_mask > 0
        if np.count_nonzero(valid) == 0:
            return {}

        points = self._landmarks_to_points(image, landmarks)
        ys, xs = np.where(valid)
        y_min, y_max = int(ys.min()), int(ys.max())
        centre_x = int(
            np.clip(
                points[Config.REGION_CENTER_LANDMARK, 0]
                if len(points) > Config.REGION_CENTER_LANDMARK
                else np.median(xs),
                0,
                width - 1,
            )
        )
        brow_left, brow_right = Config.REGION_BROW_LANDMARKS
        brow_y = int(
            np.clip(
                min(points[brow_left, 1], points[brow_right, 1])
                if len(points) > max(Config.REGION_BROW_LANDMARKS)
                else y_min + Config.REGION_BROW_FALLBACK_RATIO * (y_max - y_min),
                y_min,
                y_max,
            )
        )
        mouth_y = int(
            np.clip(
                np.mean(points[list(Config.REGION_MOUTH_LANDMARKS), 1])
                if len(points) > max(Config.REGION_MOUTH_LANDMARKS)
                else y_min + Config.REGION_MOUTH_FALLBACK_RATIO * (y_max - y_min),
                y_min,
                y_max,
            )
        )

        nose = self._polygon_mask(
            stats_mask.shape,
            points,
            list(Config.REGION_NOSE_LANDMARKS),
        )
        if np.count_nonzero(nose):
            nose = cv2.dilate(
                nose,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        Config.REGION_NOSE_DILATE_KERNEL,
                        Config.REGION_NOSE_DILATE_KERNEL,
                    ),
                ),
            )

        yy, xx = np.indices(stats_mask.shape)
        forehead = valid & (yy < brow_y)
        chin = valid & (
            yy > int(mouth_y + Config.REGION_CHIN_OFFSET_RATIO * (y_max - mouth_y))
        )
        nose_mask = valid & (nose > 0)
        cheek_zone = valid & (yy >= brow_y) & (yy <= mouth_y) & (~nose_mask)
        left_cheek = cheek_zone & (xx < centre_x)
        right_cheek = cheek_zone & (xx >= centre_x)

        return {
            "forehead": forehead.astype(np.uint8) * 255,
            "left_cheek": left_cheek.astype(np.uint8) * 255,
            "right_cheek": right_cheek.astype(np.uint8) * 255,
            "nose": nose_mask.astype(np.uint8) * 255,
            "chin": chin.astype(np.uint8) * 255,
        }

    @staticmethod
    def _feature_masked_gaussian(
        values: np.ndarray,
        mask: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """Gaussian smoothing without allowing black background into statistics."""

        return get_cuda_backend().masked_gaussian(values, mask, sigma)

    def _extract_red_features(
        self,
        redness_score_map: np.ndarray,
        region_masks: dict[str, np.ndarray],
        postprocess_masks: dict[str, np.ndarray],
    ) -> tuple[
        np.ndarray,
        list[dict],
        dict[str, dict[str, float | int]],
        dict[str, object],
    ]:
        """Convert the continuous score into countable compact red features.

        This branch is used only for engineering quantification. It does not
        feed back into any of the five display presets.
        """

        # 延迟导入，确保五档红区视觉主线在量化依赖加载前已经完整计算。
        # 量化是只读分支，不允许改变正式 score map 或五档出图。
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed

        analysis_mask = np.zeros(redness_score_map.shape, dtype=np.uint8)
        for mask in region_masks.values():
            analysis_mask = cv2.bitwise_or(analysis_mask, mask)
        empty_distribution = {
            name: {"count": 0, "area": 0, "area_ratio": 0.0}
            for name in region_masks
        }
        filter_reasons = {
            "eyes": 0,
            "eyebrows": 0,
            "nostrils": 0,
            "lips": 0,
            "boundary": 0,
            "nasolabial": 0,
        }
        filter_summary: dict[str, object] = {
            "pre_filter_count": 0,
            "filtered_count": 0,
            "reasons": filter_reasons,
        }
        if np.count_nonzero(analysis_mask) < Config.MIN_VALID_MASK_PIXELS:
            return (
                np.zeros_like(analysis_mask),
                [],
                empty_distribution,
                filter_summary,
            )

        final_mask = np.zeros_like(analysis_mask)
        locations: list[dict] = []
        next_id = 1
        analysis_area = max(1, int(np.count_nonzero(analysis_mask)))

        for region_name, raw_region_mask in region_masks.items():
            region_mask = cv2.erode(
                raw_region_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            region_values = redness_score_map[region_mask > 0]
            if region_values.size < Config.MIN_VALID_MASK_PIXELS:
                continue
            median = float(np.median(region_values))
            mad = float(np.median(np.abs(region_values - median)))
            robust_sigma = max(1.4826 * mad, 0.025)
            region_threshold = max(
                Config.RED_FEATURE_SCORE_FLOOR,
                float(np.percentile(region_values, 62)),
                median + 0.35 * robust_sigma,
            )
            regional_z = np.maximum(
                (redness_score_map.astype(np.float32) - median)
                / robust_sigma,
                0.0,
            )
            regional_z[region_mask == 0] = 0.0

            peak_candidates: list[tuple[int, int, int, float]] = []
            for scale_index, sigma in enumerate(
                Config.RED_FEATURE_LOCAL_SIGMAS
            ):
                smoothed = self._feature_masked_gaussian(
                    redness_score_map,
                    region_mask,
                    sigma,
                )
                coordinates = peak_local_max(
                    smoothed,
                    labels=(region_mask > 0).astype(np.uint8),
                    min_distance=Config.RED_FEATURE_PEAK_DISTANCE,
                    threshold_abs=region_threshold,
                    exclude_border=False,
                )
                for py, px in coordinates:
                    peak_candidates.append(
                        (
                            int(py),
                            int(px),
                            scale_index,
                            float(smoothed[int(py), int(px)]),
                        )
                    )

            peak_candidates.sort(key=lambda item: item[3], reverse=True)
            clusters: list[dict] = []
            merge_distance_sq = float(
                (Config.RED_FEATURE_PEAK_DISTANCE + 2) ** 2
            )
            for py, px, scale_index, peak_value in peak_candidates:
                match = None
                for cluster in clusters:
                    dy = py - cluster["y"]
                    dx = px - cluster["x"]
                    if dy * dy + dx * dx <= merge_distance_sq:
                        match = cluster
                        break
                if match is None:
                    clusters.append(
                        {
                            "y": py,
                            "x": px,
                            "scales": {scale_index},
                            "peak": peak_value,
                        }
                    )
                else:
                    match["scales"].add(scale_index)
                    if peak_value > match["peak"]:
                        match["y"] = py
                        match["x"] = px
                        match["peak"] = peak_value

            accepted_peaks = [
                cluster
                for cluster in clusters
                if (
                    len(cluster["scales"])
                    >= Config.RED_FEATURE_MIN_SCALE_VOTES
                    or cluster["peak"]
                    >= median + Config.RED_FEATURE_STRONG_Z * robust_sigma
                )
            ]
            if not accepted_peaks:
                continue

            candidate_mask = (
                (redness_score_map >= region_threshold)
                & (region_mask > 0)
            ).astype(np.uint8)
            candidate_mask = cv2.morphologyEx(
                candidate_mask,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            candidate_mask = cv2.morphologyEx(
                candidate_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )

            markers = np.zeros_like(analysis_mask, dtype=np.int32)
            marker_peaks: dict[int, dict] = {}
            marker_id = 1
            for cluster in accepted_peaks:
                py, px = int(cluster["y"]), int(cluster["x"])
                if candidate_mask[py, px] == 0:
                    continue
                markers[py, px] = marker_id
                marker_peaks[marker_id] = cluster
                marker_id += 1
            if marker_id == 1:
                continue

            segmented = watershed(
                -redness_score_map.astype(np.float32),
                markers=markers,
                mask=candidate_mask > 0,
                connectivity=2,
            )
            for segment_id, cluster in marker_peaks.items():
                segment = segmented == segment_id
                if not np.any(segment):
                    continue
                peak_value = float(cluster["peak"])
                grow_threshold = max(
                    region_threshold,
                    median + 0.50 * robust_sigma,
                    0.70 * peak_value,
                )
                peak_radius = int(
                    np.clip(
                        Config.RED_FEATURE_RADIUS_BASE
                        + Config.RED_FEATURE_RADIUS_PER_SCALE
                        * len(cluster["scales"]),
                        Config.RED_FEATURE_RADIUS_MIN,
                        Config.RED_FEATURE_RADIUS_MAX,
                    )
                )
                local_disk = np.zeros_like(analysis_mask, dtype=np.uint8)
                cv2.circle(
                    local_disk,
                    (int(cluster["x"]), int(cluster["y"])),
                    peak_radius,
                    1,
                    -1,
                )
                segment &= (
                    (redness_score_map >= grow_threshold)
                    & (local_disk > 0)
                )
                segment_u8 = cv2.morphologyEx(
                    segment.astype(np.uint8),
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                )
                local_count, local_labels, local_stats, _ = (
                    cv2.connectedComponentsWithStats(
                        segment_u8,
                        connectivity=8,
                    )
                )
                py, px = int(cluster["y"]), int(cluster["x"])
                selected_label = int(local_labels[py, px])
                if selected_label <= 0 and local_count > 1:
                    selected_label = 1 + int(
                        np.argmax(local_stats[1:, cv2.CC_STAT_AREA])
                    )
                if selected_label <= 0:
                    continue
                segment = local_labels == selected_label
                segment_area = int(np.count_nonzero(segment))
                if (
                    segment_area < Config.RED_FEATURE_MIN_AREA
                    or segment_area > Config.RED_FEATURE_MAX_AREA
                ):
                    continue

                ys, xs = np.where(segment)
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                width, height = x1 - x0 + 1, y1 - y0 + 1
                aspect_ratio = max(width, height) / max(
                    min(width, height),
                    1,
                )
                if aspect_ratio > Config.RED_FEATURE_MAX_ASPECT_RATIO:
                    continue

                mean_redness = float(
                    np.mean(redness_score_map[segment])
                )
                mean_z = float(np.mean(regional_z[segment]))
                scale_vote = float(len(cluster["scales"]))
                confidence = float(
                    np.clip(
                        0.50 * mean_redness
                        + 0.30 * min(mean_z / 4.0, 1.0)
                        + 0.20
                        * (
                            scale_vote
                            / len(Config.RED_FEATURE_LOCAL_SIGMAS)
                        ),
                        0.0,
                        1.0,
                    )
                )
                if confidence < Config.RED_FEATURE_MIN_CONFIDENCE:
                    continue

                cx, cy = float(np.mean(xs)), float(np.mean(ys))
                filter_summary["pre_filter_count"] = (
                    int(filter_summary["pre_filter_count"]) + 1
                )
                centroid_x = int(np.clip(round(cx), 0, segment.shape[1] - 1))
                centroid_y = int(np.clip(round(cy), 0, segment.shape[0] - 1))
                rejected_reason = None
                segment_area_float = max(float(segment_area), 1.0)
                nasolabial_mask = postprocess_masks.get("nasolabial")
                if nasolabial_mask is not None:
                    nasolabial_overlap = float(
                        np.count_nonzero(segment & (nasolabial_mask > 0))
                    ) / segment_area_float
                    centroid_in_fold = bool(
                        nasolabial_mask[centroid_y, centroid_x] > 0
                    )
                    # 只有细长并与真实法令纹暗线明显重叠的候选才删除。
                    # 紧凑的红色丘疹即使靠近鼻唇沟也继续保留。
                    if (
                        nasolabial_overlap
                        >= Config.RED_FEATURE_NASOLABIAL_OVERLAP_RATIO
                        and aspect_ratio
                        >= Config.RED_FEATURE_NASOLABIAL_MIN_ASPECT_RATIO
                    ) or (
                        centroid_in_fold
                        and aspect_ratio
                        >= Config.RED_FEATURE_NASOLABIAL_CENTROID_MIN_ASPECT_RATIO
                    ):
                        rejected_reason = "nasolabial"
                for reason in (
                    "eyes",
                    "eyebrows",
                    "nostrils",
                    "lips",
                    "boundary",
                ):
                    if rejected_reason is not None:
                        break
                    forbidden_mask = postprocess_masks.get(reason)
                    if forbidden_mask is None:
                        continue
                    overlap_ratio = float(
                        np.count_nonzero(segment & (forbidden_mask > 0))
                    ) / segment_area_float
                    centroid_inside = bool(
                        forbidden_mask[centroid_y, centroid_x] > 0
                    )
                    if (
                        centroid_inside
                        or overlap_ratio
                        >= Config.RED_FEATURE_FORBIDDEN_OVERLAP_RATIO
                    ):
                        rejected_reason = reason
                        break
                if rejected_reason is not None:
                    filter_reasons[rejected_reason] += 1
                    filter_summary["filtered_count"] = (
                        int(filter_summary["filtered_count"]) + 1
                    )
                    continue

                final_mask[segment] = 255
                locations.append(
                    {
                        "feature_id": next_id,
                        "region": region_name,
                        "area": segment_area,
                        "area_ratio_analysis_zone": round(
                            segment_area / analysis_area,
                            Config.METRIC_DECIMALS,
                        ),
                        "bbox": [x0, y0, width, height],
                        "centroid": [round(cx, 3), round(cy, 3)],
                        "mean_redness": round(
                            mean_redness,
                            Config.METRIC_DECIMALS,
                        ),
                        "local_z": round(
                            mean_z,
                            Config.METRIC_DECIMALS,
                        ),
                        "scale_vote": round(scale_vote, 4),
                        "confidence": round(confidence, 4),
                    }
                )
                next_id += 1

        region_distribution: dict[str, dict[str, float | int]] = {}
        for name, region_mask in region_masks.items():
            region_area = max(1, int(np.count_nonzero(region_mask)))
            local_features = [
                feature
                for feature in locations
                if feature["region"] == name
            ]
            local_area = int(
                sum(int(feature["area"]) for feature in local_features)
            )
            region_distribution[name] = {
                "count": len(local_features),
                "area": local_area,
                "area_ratio": round(
                    local_area / region_area,
                    Config.METRIC_DECIMALS,
                ),
            }

        return final_mask, locations, region_distribution, filter_summary

    @staticmethod
    def _smooth_binary_mask(mask: np.ndarray) -> np.ndarray:
        """Create a stable display-only mask with smoother boundaries."""

        binary = (mask > 0).astype(np.uint8) * 255
        kernel_size = Config.RED_FEATURE_ZONE_CLOSE_KERNEL
        if kernel_size > 1:
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (kernel_size, kernel_size),
                ),
            )
        smoothed = cv2.GaussianBlur(
            binary,
            (0, 0),
            sigmaX=Config.RED_FEATURE_ZONE_SMOOTH_SIGMA,
            sigmaY=Config.RED_FEATURE_ZONE_SMOOTH_SIGMA,
            borderType=cv2.BORDER_REFLECT,
        )
        return (smoothed >= 127).astype(np.uint8) * 255

    @classmethod
    def _filled_display_zone_base(cls, mask: np.ndarray) -> np.ndarray:
        """Turn the strict analysis mask into one clean display envelope.

        Detection and metrics keep using the original strict mask. Only the
        cyan overlay uses this filled envelope, so hair/feature holes cannot
        deform the five VISIA-like region contours.
        """

        smoothed = cls._smooth_binary_mask(mask)
        contours, _ = cv2.findContours(
            smoothed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return smoothed
        largest = max(contours, key=lambda item: abs(cv2.contourArea(item)))
        filled = np.zeros_like(smoothed)
        cv2.drawContours(filled, [largest], -1, 255, thickness=-1)
        return cls._smooth_binary_mask(filled)

    @staticmethod
    def _chaikin_closed_curve(
        contour: np.ndarray,
        iterations: int = 2,
    ) -> np.ndarray:
        """Smooth a closed contour without introducing splines/dependencies."""

        points = contour.reshape(-1, 2).astype(np.float32)
        if len(points) < 4:
            return contour.astype(np.int32)
        # Bound the working set while preserving the original contour order.
        sample_step = max(1, len(points) // 220)
        points = points[::sample_step]
        for _ in range(iterations):
            following = np.roll(points, -1, axis=0)
            first = 0.75 * points + 0.25 * following
            second = 0.25 * points + 0.75 * following
            smoothed = np.empty((len(points) * 2, 2), dtype=np.float32)
            smoothed[0::2] = first
            smoothed[1::2] = second
            points = smoothed
        return np.rint(points).astype(np.int32).reshape(-1, 1, 2)

    @staticmethod
    def _draw_smooth_contours(
        canvas: np.ndarray,
        mask: np.ndarray,
        *,
        largest_only: bool,
    ) -> None:
        """Draw a small number of anti-aliased VISIA-like region contours."""

        contours, hierarchy = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if not contours:
            return
        del hierarchy
        contours = [
            contour
            for contour in contours
            if abs(float(cv2.contourArea(contour)))
            >= Config.RED_FEATURE_ZONE_MIN_CONTOUR_AREA
        ]
        if not contours:
            return
        if largest_only:
            contours = [max(contours, key=lambda item: abs(cv2.contourArea(item)))]
        for contour in contours:
            area = abs(float(cv2.contourArea(contour)))
            if area < Config.RED_FEATURE_ZONE_MIN_CONTOUR_AREA:
                continue
            smooth_contour = ErythemaAnalyzer._chaikin_closed_curve(
                contour,
                iterations=Config.RED_FEATURE_ZONE_CHAIKIN_ITERATIONS,
            )
            cv2.polylines(
                canvas,
                [smooth_contour],
                True,
                Config.RED_FEATURE_ZONE_COLOR,
                Config.RED_FEATURE_ZONE_THICKNESS,
                lineType=cv2.LINE_AA,
            )

    def _render_red_feature_overlay(
        self,
        balanced_render: np.ndarray,
        region_masks: dict[str, np.ndarray],
        red_feature_locations: list[dict],
        partial_face: bool = False,
        display_contour: np.ndarray | None = None,
        display_separator: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Overlay VISIA-like analysis zones and compact yellow instances.

        The underlying image is exactly the balanced preset. Markers are
        derived one-for-one from red_feature_locations so the displayed count
        and JSON count share the same instance collection.
        """

        canvas = balanced_render.copy()
        shape = balanced_render.shape[:2]
        # Use the same shared smooth VISIA-region renderer as Spots.  The
        # analysis-zone statistics remain unchanged; this replaces only the
        # user-visible boundary shape.
        canvas = draw_region_boundaries(
            canvas,
            region_masks,
            Config.RED_FEATURE_ZONE_COLOR,
            thickness=Config.RED_FEATURE_ZONE_THICKNESS,
            partial_face=partial_face,
            closed_contour=display_contour,
            separator_contour=display_separator,
        )

        # Eye/brow/lip/nostril masks are intentionally used for filtering only.
        # Drawing a separate loop for each feature recreates the clutter the
        # user rejected. The five main region contours already define the
        # VISIA-like analysis zones, including the additional chin zone.

        marker_mask = np.zeros(shape, dtype=np.uint8)
        for feature in red_feature_locations:
            centroid = feature.get("centroid", [0.0, 0.0])
            if len(centroid) != 2:
                continue
            cx = int(np.clip(round(float(centroid[0])), 0, shape[1] - 1))
            cy = int(np.clip(round(float(centroid[1])), 0, shape[0] - 1))
            area = max(1.0, float(feature.get("area", 1.0)))
            radius = int(
                np.clip(
                    round(
                        np.sqrt(area / np.pi)
                        * Config.RED_FEATURE_MARKER_RADIUS_SCALE
                    ),
                    Config.RED_FEATURE_MARKER_RADIUS_MIN,
                    Config.RED_FEATURE_MARKER_RADIUS_MAX,
                )
            )
            cv2.circle(
                marker_mask,
                (cx, cy),
                radius,
                255,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

        canvas[marker_mask > 0] = Config.RED_FEATURE_MARKER_COLOR
        return canvas, marker_mask

    def _summarize_values(self, values: np.ndarray) -> dict[str, float | int]:
        if values.size == 0:
            return {
                "area": 0,
                "red_area_ratio": 0.0,
                "high_red_area_ratio": 0.0,
                "mean_redness": 0.0,
                "p50_redness": 0.0,
                "p90_redness": 0.0,
                "p95_redness": 0.0,
                "max_redness": 0.0,
                "redness_burden": 0.0,
            }
        values = values.astype(np.float32)
        area = int(values.size)
        return {
            "area": area,
            "red_area_ratio": round(
                float(np.count_nonzero(values >= Config.RED_AREA_THRESHOLD) / area),
                Config.METRIC_DECIMALS,
            ),
            "high_red_area_ratio": round(
                float(
                    np.count_nonzero(values >= Config.HIGH_RED_AREA_THRESHOLD) / area
                ),
                Config.METRIC_DECIMALS,
            ),
            "mean_redness": round(float(np.mean(values)), Config.METRIC_DECIMALS),
            "p50_redness": round(
                float(np.percentile(values, Config.METRIC_P50_PERCENTILE)),
                Config.METRIC_DECIMALS,
            ),
            "p90_redness": round(
                float(np.percentile(values, Config.METRIC_P90_PERCENTILE)),
                Config.METRIC_DECIMALS,
            ),
            "p95_redness": round(
                float(np.percentile(values, Config.METRIC_P95_PERCENTILE)),
                Config.METRIC_DECIMALS,
            ),
            "max_redness": round(float(np.max(values)), Config.METRIC_DECIMALS),
            "redness_burden": round(
                float(np.sum(values) / area),
                Config.METRIC_DECIMALS,
            ),
        }

    def _build_metrics(
        self,
        filename: str,
        redness_score_map: np.ndarray,
        person_mask: np.ndarray,
        stats_mask: np.ndarray,
        landmarks,
        image: np.ndarray,
        region_masks: dict[str, np.ndarray],
        red_feature_mask: np.ndarray,
        red_feature_locations: list[dict],
        red_feature_region_distribution: dict[
            str,
            dict[str, float | int],
        ],
        quality_score: float | None = None,
        quality_status: str | None = None,
        quality_flags: list[str] | None = None,
    ) -> dict:
        valid = stats_mask > 0
        overall = self._summarize_values(redness_score_map[valid])
        region_statistics = {}
        for name, mask in region_masks.items():
            region_statistics[name] = self._summarize_values(
                redness_score_map[mask > 0]
            )

        reference_path = (
            str(self._color_transfer_source_path)
            if self.color_transfer_enabled
            and self._color_transfer_source_path is not None
            else None
        )
        red_feature_area = int(np.count_nonzero(red_feature_mask))
        feature_analysis_mask = (
            np.any(
                np.stack(
                    [mask > 0 for mask in region_masks.values()],
                    axis=0,
                ),
                axis=0,
            )
            if region_masks
            else stats_mask > 0
        )
        feature_analysis_area = max(
            1,
            int(np.count_nonzero(feature_analysis_mask)),
        )
        return {
            "algorithm": "RBX-like Visible Redness",
            "metric_version": "rbx_redness_v3_five_presets_quantified",
            "source_name": filename,
            "reference_image": reference_path,
            "color_transfer_enabled": bool(
                self.color_transfer_enabled and reference_path
            ),
            "color_transfer_strength": float(Config.COLOR_TRANSFER_STRENGTH),
            "red_area_threshold": float(Config.RED_AREA_THRESHOLD),
            "high_red_area_threshold": float(Config.HIGH_RED_AREA_THRESHOLD),
            "skin_foreground_area": int(np.count_nonzero(person_mask)),
            "face_stats_area": int(np.count_nonzero(stats_mask)),
            "red_area_ratio": overall["red_area_ratio"],
            "high_red_area_ratio": overall["high_red_area_ratio"],
            "mean_redness": overall["mean_redness"],
            "p50_redness": overall["p50_redness"],
            "p90_redness": overall["p90_redness"],
            "p95_redness": overall["p95_redness"],
            "max_redness": overall["max_redness"],
            "redness_burden": overall["redness_burden"],
            "region_statistics": region_statistics,
            "red_feature_count": len(red_feature_locations),
            "red_feature_area": red_feature_area,
            "red_feature_area_ratio": round(
                red_feature_area / feature_analysis_area,
                Config.METRIC_DECIMALS,
            ),
            "red_feature_locations": red_feature_locations,
            "red_feature_region_distribution": (
                red_feature_region_distribution
            ),
            "quality_score": quality_score,
            "quality_status": quality_status,
            "quality_flags": list(quality_flags or []),
            "disclaimer": "当前为 Visible Redness 工程量化指标，不代表医学炎症诊断或严重程度分级。",
        }

    @staticmethod
    def _write_natural_language_report(metrics: dict, report_path: Path) -> None:
        """Write the customer-facing UTF-8 redness summary.

        The wording intentionally describes relative engineering measurements
        without converting them into medical diagnoses or severity grades.
        """
        region_names = {
            "forehead": "额头",
            "left_cheek": "左脸颊",
            "right_cheek": "右脸颊",
            "nose": "鼻部",
            "chin": "下巴",
        }
        visible_regions = [
            (region_names.get(name, name), values)
            for name, values in metrics.get("region_statistics", {}).items()
            if int(values.get("area", 0)) > 0
        ]
        visible_regions.sort(
            key=lambda item: float(item[1].get("mean_redness", 0.0)),
            reverse=True,
        )
        if visible_regions:
            leading = "、".join(name for name, _ in visible_regions[:2])
            region_lines = [
                (
                    f"- {name}：红区占比 {float(values.get('red_area_ratio', 0.0)):.2%}，"
                    f"平均红度 {float(values.get('mean_redness', 0.0)):.3f}，"
                    f"P95 红度 {float(values.get('p95_redness', 0.0)):.3f}。"
                )
                for name, values in visible_regions
            ]
            relative_summary = f"本图中相对更突出的区域为：{leading}。"
        else:
            region_lines = ["- 当前没有可用于分区统计的有效皮肤区域。"]
            relative_summary = "当前没有足够的有效分区数据用于比较。"

        feature_distribution = metrics.get(
            "red_feature_region_distribution",
            {},
        )
        feature_regions = [
            (
                region_names.get(name, name),
                int(values.get("count", 0)),
            )
            for name, values in feature_distribution.items()
            if int(values.get("count", 0)) > 0
        ]
        feature_regions.sort(key=lambda item: item[1], reverse=True)
        feature_summary = "、".join(
            f"{name} {count} 个" for name, count in feature_regions
        ) or "当前未检出离散红色区实例"

        calibration = "已启用" if metrics.get("color_transfer_enabled") else "未启用"
        lines = [
            "DermaVision 红区量化说明",
            "",
            f"输入图片：{metrics.get('source_name', '')}",
            f"RBX 专属颜色校准：{calibration}",
            f"有效前景面积：{int(metrics.get('skin_foreground_area', 0))} 像素",
            f"有效人脸统计面积：{int(metrics.get('face_stats_area', 0))} 像素",
            "",
            "整体红区情况：",
            f"- 红区面积占比：{float(metrics.get('red_area_ratio', 0.0)):.2%}",
            f"- 高红度区域占比：{float(metrics.get('high_red_area_ratio', 0.0)):.2%}",
            f"- 平均红度：{float(metrics.get('mean_redness', 0.0)):.3f}",
            f"- P90 红度：{float(metrics.get('p90_redness', 0.0)):.3f}",
            f"- P95 红度：{float(metrics.get('p95_redness', 0.0)):.3f}",
            f"- 红度综合负担：{float(metrics.get('redness_burden', 0.0)):.3f}",
            f"- {relative_summary}",
            "",
            "离散红色区实例：",
            f"- 红色区实例数：{int(metrics.get('red_feature_count', 0))} 个",
            f"- 实例面积占分析区比例：{float(metrics.get('red_feature_area_ratio', 0.0)):.3%}",
            f"- 实例主要分布：{feature_summary}",
            (
                "- 五官/边界后处理：从 "
                f"{int(metrics.get('red_feature_pre_filter_count', 0))} 个候选中"
                f"过滤 {int(metrics.get('red_feature_filtered_count', 0))} 个"
            ),
            "",
            "面部分区：",
            *region_lines,
            "",
            "说明：当前结果属于普通 RGB 照片的 Visible Redness 工程量化，",
            "用于比较图中不同皮肤区域的相对红度，不代表医学炎症诊断，",
            "也不提供正常、轻度、中度或重度分级。拍摄光照、白平衡、妆容",
            "和相机处理均可能影响结果。",
            "",
        ]
        report_path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _compact_metrics(metrics: dict) -> dict[str, int]:
        """生成云端 CSV/JSON 共用的精简红区量化内容。"""
        region_names = {
            "forehead": "额头",
            "left_cheek": "左脸颊",
            "right_cheek": "右脸颊",
            "nose": "鼻部",
            "chin": "下巴",
        }
        feature_distribution = metrics.get(
            "red_feature_region_distribution",
            {},
        )
        headers = ["总计"]
        values: list[object] = [
            int(metrics.get("red_feature_count", 0))
        ]
        quality_flags = {
            str(flag)
            for flag in (metrics.get("quality_flags") or [])
        }
        if "PARTIAL_FACE" not in quality_flags:
            headers.extend(region_names.values())
            values.extend(
                int(
                    feature_distribution
                    .get(region_key, {})
                    .get("count", 0)
                )
                for region_key in region_names
            )
        return {
            header: int(value)
            for header, value in zip(headers, values)
        }

    @staticmethod
    def _write_metrics_csv(
        compact_metrics: dict[str, int],
        csv_path: Path,
    ) -> None:
        """把与 JSON 完全一致的精简指标写成横向 CSV。"""
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(compact_metrics.keys())
            writer.writerow(compact_metrics.values())

    @staticmethod
    def _clean_base_name(value: str) -> str:
        base_name = Path(value).stem
        for suffix in ("_01_AnalysisInput", "_01_Preprocessed"):
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]
        return base_name

    @staticmethod
    def _red_feature_region_masks(
        region_masks: dict[str, np.ndarray],
        regions: VisiaRegionSet,
    ) -> dict[str, np.ndarray]:
        """Keep historical red regions while avoiding only the drawn cyan line."""

        boundary = regions.public_boundary_safety_mask
        if boundary is None:
            boundary = np.zeros_like(next(iter(region_masks.values())))
        allowed = cv2.bitwise_not(
            (np.asarray(boundary) > 0).astype(np.uint8) * 255,
        )
        return {
            name: cv2.bitwise_and(
                (np.asarray(mask) > 0).astype(np.uint8) * 255,
                allowed,
            )
            for name, mask in region_masks.items()
        }

    @staticmethod
    def _public_quantification_mask(
        stats_mask: np.ndarray,
        regions: VisiaRegionSet,
    ) -> np.ndarray:
        """Preserve the accepted consumer filled-scope quantification domain."""

        scope = regions.scope_mask
        if scope is None:
            scope = regions.analysis_mask
        return cv2.bitwise_and(
            (np.asarray(stats_mask) > 0).astype(np.uint8) * 255,
            (np.asarray(scope) > 0).astype(np.uint8) * 255,
        )

    def _process_arrays(
        self,
        image: np.ndarray,
        stats_mask: np.ndarray,
        landmarks,
        face_render_mask: np.ndarray,
        output_dir: str,
        source_name: str,
        base_name: str,
        quality_score: float | None = None,
        quality_status: str | None = None,
        quality_flags: list[str] | None = None,
        common_debug_masks: dict[str, np.ndarray] | None = None,
    ) -> str | None:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            return None
        if stats_mask.shape != image.shape[:2]:
            raise ValueError("skin_mask 与 analysis_image 尺寸不一致")
        if face_render_mask.shape != image.shape[:2]:
            raise ValueError("face_render_mask 与 analysis_image 尺寸不一致")
        stats_mask = cv2.bitwise_and(
            (stats_mask > 0).astype(np.uint8) * 255,
            (face_render_mask > 0).astype(np.uint8) * 255,
        )
        if np.count_nonzero(stats_mask) < Config.MIN_VALID_MASK_PIXELS:
            print(f"⚠️ 跳过 {source_name}: 有效皮肤统计区域过小。")
            return None

        render_mask = (face_render_mask > 0).astype(np.uint8) * 255
        person_foreground_mask = self._person_foreground_mask(image, render_mask)
        person_foreground_alpha = self._foreground_alpha(person_foreground_mask)
        face_render_alpha = self._foreground_alpha(render_mask)
        black_face = np.zeros_like(image)
        black_face[person_foreground_mask > 0] = image[person_foreground_mask > 0]
        work_image = self._rbx_lab_color_transfer(
            image,
            stats_mask,
            render_mask,
        )
        base_map, detail_map, raw_score_map, _ = self._build_redness_maps(
            work_image,
            stats_mask,
            render_mask,
            landmarks,
        )
        score_map = self._apply_style_lut(raw_score_map)
        score_map[render_mask <= 0] = 0.0
        source_luminance = (
            cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
            / 255.0
        )
        natural_render = self._render_result(
            score_map,
            source_luminance,
            person_foreground_alpha,
            face_render_alpha,
            image,
            "natural",
        )
        # Quantification is deliberately downstream-only: it reads the same
        # continuous score map, but never changes the formal render.
        shared_display_regions = build_visia_regions(
            image,
            stats_mask,
            landmarks,
            quality_flags or [],
            include_chin=True,
            mode="full",
        )
        if self.capture_profile is CaptureProfile.INSTITUTION:
            metric_mask = stats_mask
            metric_region_masks = self._region_masks(
                image,
                landmarks,
                metric_mask,
            )
            feature_region_masks = self._red_feature_region_masks(
                metric_region_masks,
                shared_display_regions,
            )
        else:
            metric_mask = self._public_quantification_mask(
                stats_mask,
                shared_display_regions,
            )
            metric_region_masks = self._region_masks(
                image,
                landmarks,
                metric_mask,
            )
            feature_region_masks = metric_region_masks
        analysis_zone_mask = np.zeros(stats_mask.shape, dtype=np.uint8)
        for region_mask in feature_region_masks.values():
            analysis_zone_mask = cv2.bitwise_or(
                analysis_zone_mask,
                (region_mask > 0).astype(np.uint8) * 255,
            )
        postprocess_masks = self._build_red_feature_postprocess_masks(
            image,
            landmarks,
            analysis_zone_mask,
        )
        (
            red_feature_mask,
            red_feature_locations,
            red_feature_region_distribution,
            red_feature_filter_summary,
        ) = self._extract_red_features(
            score_map.copy(),
            {
                name: mask.copy()
                for name, mask in feature_region_masks.items()
            },
            {
                name: mask.copy()
                for name, mask in postprocess_masks.items()
            },
        )
        red_feature_overlay, red_feature_display_mask = (
            self._render_red_feature_overlay(
                natural_render,
                shared_display_regions.display_regions,
                red_feature_locations,
                partial_face=shared_display_regions.partial_face,
                display_contour=shared_display_regions.display_contour,
                display_separator=shared_display_regions.display_separator,
            )
        )

        sample_dir = Path(output_dir) / base_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        for obsolete_name in (
            "03A_RBX红区结果_柔和.jpg",
            "03B_RBX红区结果_自然.jpg",
            "03C_RBX红区结果_均衡.jpg",
            "03D_RBX红区结果_强对比.jpg",
            "03E_RBX红区结果_超强对比.jpg",
        ):
            (sample_dir / obsolete_name).unlink(missing_ok=True)
        input_path = sample_dir / "01_RBX输入图.jpg"
        face_path = sample_dir / "02A_RBX人脸抠图.png"
        mapped_path = sample_dir / "02B_RBX颜色映射图.png"
        output_path = sample_dir / "03_RBX红区结果图.jpg"
        strength_path = sample_dir / "04_RBX红区强度图.png"
        mask_path = sample_dir / "05_RBX完整人脸Mask.png"
        feature_overlay_path = sample_dir / "06_VISIA红色区实例图.jpg"
        feature_mask_path = sample_dir / "07_VISIA红色区实例Mask.png"
        metrics_path = sample_dir / "红区量化指标.json"
        report_path = sample_dir / "红区量化指标.csv"
        medical_metrics_path = sample_dir / "红区医学量化指标.json"
        medical_report_path = sample_dir / "红区医学量化指标.csv"
        medical_metrics_v2_path = sample_dir / "红区医学量化指标_V2.json"
        medical_report_v2_path = sample_dir / "红区医学量化指标_V2.csv"

        metrics = self._build_metrics(
            filename=source_name,
            redness_score_map=score_map,
            person_mask=render_mask,
            stats_mask=metric_mask,
            landmarks=landmarks,
            image=image,
            region_masks=metric_region_masks,
            red_feature_mask=red_feature_mask,
            red_feature_locations=red_feature_locations,
            red_feature_region_distribution=(
                red_feature_region_distribution
            ),
            quality_score=quality_score,
            quality_status=quality_status,
            quality_flags=quality_flags,
        )
        metrics["redness_base_mean"] = round(
            float(np.mean(base_map[metric_mask > 0])),
            Config.METRIC_DECIMALS,
        )
        metrics["redness_detail_mean"] = round(
            float(np.mean(detail_map[metric_mask > 0])),
            Config.METRIC_DECIMALS,
        )
        metrics["raw_redness_mean"] = round(
            float(np.mean(raw_score_map[metric_mask > 0])),
            Config.METRIC_DECIMALS,
        )
        metrics["render_default"] = "natural"
        metrics["render_presets"] = {"natural": Config.RENDER_PRESETS["natural"]}
        metrics["color_transfer_scope"] = "face_statistics_mask_only"
        metrics["eye_rendering"] = "no_special_eye_processing"
        metrics["face_mask_source"] = "selfie_multiclass_face_skin_class_3"
        metrics["red_feature_overlay_base"] = "natural"
        metrics["red_feature_marker_count"] = len(red_feature_locations)
        metrics["red_feature_pre_filter_count"] = int(
            red_feature_filter_summary.get("pre_filter_count", 0)
        )
        metrics["red_feature_filtered_count"] = int(
            red_feature_filter_summary.get("filtered_count", 0)
        )
        metrics["red_feature_filter_reasons"] = (
            red_feature_filter_summary.get("reasons", {})
        )

        feature_instances = []
        for item in metrics.get("red_feature_locations", []):
            instance = dict(item)
            instance["intensity"] = item.get("mean_redness", 0.0)
            instance["morphology"] = "局灶性红色实例"
            feature_instances.append(instance)
        diffuse_mask = (
            (score_map >= Config.RED_AREA_THRESHOLD)
            & (metric_mask > 0)
            & (red_feature_mask == 0)
        ).astype(np.uint8) * 255
        high_mask = (
            (score_map >= Config.HIGH_RED_AREA_THRESHOLD)
            & (metric_mask > 0)
        ).astype(np.uint8) * 255
        detailed = build_medical_payload(
            project="redness",
            project_label="可见红区",
            analysis_mask=metric_mask,
            landmarks=landmarks,
            instances=feature_instances,
            score_map=score_map,
            instance_mask=red_feature_mask,
            continuous_mask=diffuse_mask,
            high_mask=high_mask,
            quality_control={
                "图像质量状态": str(quality_status or "UNKNOWN"),
                "图像质量分数": float(quality_score or 0.0),
                "图像质量提示": list(quality_flags or []),
                "颜色映射是否启用": metrics.get("color_transfer_enabled", False),
                "颜色映射参考图": metrics.get("reference_image"),
                "实例过滤前候选数量": metrics.get("red_feature_pre_filter_count", 0),
                "实例过滤数量": metrics.get("red_feature_filtered_count", 0),
                "实例过滤原因统计": metrics.get("red_feature_filter_reasons", {}),
            },
            limitations=[
                "本结果为普通白光RGB图像中的可见红色色差工程量化。",
                "不能解释为血流量，也不构成炎症、血管或皮肤疾病诊断。",
                "局灶性红色实例不等同于炎症丘疹，未测量真实高度。",
                "不输出未经医学人群数据标定的0～100分或严重程度等级。",
            ],
        )

        cv_imwrite(str(input_path), image)
        cv_imwrite(str(face_path), black_face)
        cv_imwrite(str(mapped_path), work_image)
        cv_imwrite(str(output_path), natural_render)
        cv_imwrite(
            str(strength_path),
            np.clip(score_map * 255.0, 0, 255).astype(np.uint8),
        )
        cv_imwrite(str(mask_path), render_mask)
        cv_imwrite(str(feature_overlay_path), red_feature_overlay)
        cv_imwrite(str(feature_mask_path), red_feature_display_mask)
        write_analysis_debug_outputs(
            sample_dir,
            common_debug_masks or {
                "valid_skin_mask": metric_mask,
                "hair_occlusion_mask": np.zeros_like(metric_mask),
            },
            np.zeros_like(metric_mask),
        )
        # 对外接口合同保持肖旭升版本的既有语义：redness metrics 是完整
        # 英文工程指标 dict，CSV 是精简实例计数。新增医学指标另存文件，
        # 不得覆盖 raw_result.metrics，否则后端旧版模型会整体失配。
        write_medical_metrics(
            medical_metrics_path,
            medical_report_path,
            detailed,
            "redness",
        )
        medical_v2 = try_write_medical_metrics_v2(
            medical_metrics_v2_path,
            medical_report_v2_path,
            detailed,
            "redness",
        )
        # 坐标仅供算法内部生成实例图和医学聚合；公开 JSON 保留旧字段名
        # 与 list 类型，但不再返回逐实例 bbox/centroid 明细。
        public_metrics = dict(metrics)
        public_metrics["red_feature_locations"] = []
        metrics_path.write_text(
            json.dumps(public_metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        embed_medical_metrics_v2(
            metrics_path,
            medical_v2,
            cleanup_json_paths=(medical_metrics_path, medical_metrics_v2_path),
        )
        self._write_metrics_csv(self._compact_metrics(metrics), report_path)

        print(
            f"✅ 生成完毕 -> {base_name}/"
            "03_RBX红区结果图（自然版正式输出）和06_VISIA红色区实例图"
        )
        return str(output_path)

    def process_preprocess_result(
        self,
        preprocess_result: "PreprocessResultV2",
        output_dir: str,
        source_name: str,
    ) -> str | None:
        """Official V2 entry: analyse the in-memory, preprocessed face only."""
        image = np.asarray(preprocess_result.analysis_image)
        stats_mask = (
            np.asarray(preprocess_result.skin_mask) > 0
        ).astype(np.uint8) * 255
        landmarks = np.asarray(preprocess_result.landmarks, dtype=np.float32)
        if image.shape[:2] != stats_mask.shape:
            raise ValueError("PreprocessResultV2 的 analysis_image/skin_mask 尺寸不一致")
        if preprocess_result.quality_status == "REJECT":
            flags = ",".join(preprocess_result.quality_flags) or "UNKNOWN"
            print(f"⚠️ 跳过 {source_name}: 图像质量门禁拒绝 [{flags}]")
            return None

        print(f"\n🚀 [RBX V3] 正在渲染预处理人脸: {source_name} ...")
        face_render_mask = self._semantic_face_mask(
            image,
            landmarks,
            stats_mask,
        )
        return self._process_arrays(
            image=image,
            stats_mask=stats_mask,
            landmarks=landmarks,
            face_render_mask=face_render_mask,
            output_dir=output_dir,
            source_name=source_name,
            base_name=self._clean_base_name(source_name),
            quality_score=float(preprocess_result.quality_score),
            quality_status=str(preprocess_result.quality_status),
            quality_flags=list(preprocess_result.quality_flags),
            common_debug_masks=capture_preprocess_debug(preprocess_result),
        )

    def process_image(self, img_path: str, output_dir: str) -> str | None:
        """Legacy path entry retained for already-preprocessed image callers."""
        filename = os.path.basename(img_path)
        image = cv_imread(img_path)
        if image is None:
            return None
        print(f"\n🚀 [RBX 兼容入口] 正在渲染: {filename} ...")
        landmarks = self._detect_landmarks(image)
        if landmarks is None:
            print(f"⚠️ 跳过 {filename}: 未检测到有效人脸。")
            return None
        face_render_mask = self._semantic_face_mask(image, landmarks, None)
        stats_mask = face_render_mask.copy()
        return self._process_arrays(
            image=image,
            stats_mask=stats_mask,
            landmarks=landmarks,
            face_render_mask=face_render_mask,
            output_dir=output_dir,
            source_name=filename,
            base_name=self._clean_base_name(filename),
            common_debug_masks=None,
        )


if __name__ == "__main__":
    input_dir = PROJECT_ROOT / "data" / "preprocessed_images"
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" 🩸 血液红斑 RBX 核心引擎 (恢复 V17.3 风格)")
    print("=" * 75)

    analyzer = ErythemaAnalyzer()
    try:
        valid_files = [
            f
            for f in os.listdir(input_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not valid_files:
            print(f"⚠️ 文件夹 {input_dir} 为空！请先运行预处理模块。")
        else:
            for file in sorted(valid_files):
                analyzer.process_image(str(input_dir / file), str(output_dir))
            print("\n🎉 全部处理完成！")
    finally:
        analyzer.close()
