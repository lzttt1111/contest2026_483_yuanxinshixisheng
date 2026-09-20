"""Patient/physician wording only. Internal reason codes must never be printed."""
import re

REASONS = {
    "no_history": "首次检测或暂无可比历史记录。",
    "requires_calibrated_3d": "本次未采集合格的三维数据，暂不提供该项评估。",
    "missing_compatible_reference": "现有数据暂不足以支持本项目完整评分，暂不评分；已得到的量化结果和可计算的单项参考分予以保留。",
    "large_pore_threshold_uncalibrated": "大毛孔判定标准尚待确认，本次不提供大毛孔密度。",
    "porphyrin_threshold_uncalibrated": "高强度卟啉判定标准尚待确认，本次不提供该项密度。",
    "insufficient_valid_region_or_rejected_quality": "该区域可用图像不足，暂不提供评估。",
    "no_valid_instances": "未检出可用于此项分布统计的有效目标。",
    "no_valid_gloss": "未检出可用于强度统计的有效油光区域。",
    "no_valid_signal": "未检出可用于强度统计的有效区域。",
    "no_affected_roi": "未检出有效受累区域，不计算区域内部密度。",
    "missing_capture_time": "缺少完整采集时间，暂不进行前后比较。",
}
DEFAULT_REASON = "本次结果尚不具备该项评估所需条件，暂不提供评估。"

def explanation(reason):
    return REASONS.get(reason, DEFAULT_REASON)

def acquisition_note(profile):
    if profile == "consumer":
        return "本次采用普通照片评估。UV及卟啉相关结果由图像算法推算，并非实际UV拍摄或荧光测量，仅作为图像评估参考。左右按标准化图像画面方向标注。"
    if profile == "institution":
        return "本次采用白光、平行偏振、交叉偏振及UV图像评估。各项目按相应光源及有效图像条件计算。左右按标准化图像画面方向标注。"
    raise ValueError("unknown profile")

def assert_formal_text(text):
    forbidden = (
        r"(?i)\b(?:debug|traceback|candidate|uncalibrated|TODO|FIXME|NaN|Infinity)\b",
        r"(?:/home/|/tmp/|[A-Za-z]:[\\/]|\\\\wsl)",
        r"(?:missing_[a-z_]+|requires_[a-z_]+|score_valid|raw_result|source_kind)",
        r"(?:调试信息|调试模式|测试数据|占位符|工程置信度)",
    )
    if any(re.search(pattern, text) for pattern in forbidden):
        raise ValueError("non-formal content in report")
