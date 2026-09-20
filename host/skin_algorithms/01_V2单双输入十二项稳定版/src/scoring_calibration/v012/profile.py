from __future__ import annotations

import copy
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from src.scoring_calibration.v011.scoring import transform_value


SHADOW_PROFILE_VERSION = "aisia_scoring_v0.1.2_front5_shadow_20260807"
UV_SOURCES = {
    "uv_spot_density": ("UV样色素范围", "density", "log1p"),
    "uv_spot_area_ratio": ("UV样色素范围", "area_ratio", "identity"),
    "uv_p90_intensity": ("UV样色素强度", "p90_intensity", "identity"),
    # 历史JSONL保存的是“高强度目标比例”，不是高强度像素面积占比。
    "uv_high_intensity_target_ratio": ("UV样色素强度", "high_intensity_ratio", "identity"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_uv_shadow_profile(
    official_profile_path: str | Path,
    sqlite_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create a new shadow profile without mutating the V0.1.1 profile.

    Only the three missing UV reference arrays are appended. The PASS set and
    all existing reference arrays are inherited byte-for-byte in value from
    the official profile and the existing V0.1.1 SQLite gate database.
    """
    official_path = Path(official_profile_path).resolve()
    db_path = Path(sqlite_path).resolve()
    target = Path(output_path).resolve()
    official = json.loads(official_path.read_text(encoding="utf-8"))
    if official.get("scoring_profile_version") != "aisia_scoring_v0.1.1_integrity_full_reference_20260807":
        raise ValueError("只允许从V0.1.1全量历史正式配置派生shadow")

    connection = sqlite3.connect(db_path)
    try:
        pass_rows = connection.execute(
            "SELECT relative_path, source_jsonl FROM samples WHERE gate_status='PASS'"
        ).fetchall()
    finally:
        connection.close()
    allowed_by_source: dict[str, set[str]] = {}
    for relative_path, source in pass_rows:
        allowed_by_source.setdefault(str(source), set()).add(str(relative_path))

    raw_values = {metric_id: [] for metric_id in UV_SOURCES}
    for source, allowed in sorted(allowed_by_source.items()):
        consumed: set[str] = set()
        with Path(source).open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                relative_path = str((record.get("signature") or {}).get("relative_path") or "")
                # 历史JSONL中存在极少数重复相对路径。V0.1.1门禁库对每个
                # relative_path只保留一条，因此shadow必须使用同样的去重口径。
                if relative_path not in allowed or relative_path in consumed:
                    continue
                consumed.add(relative_path)
                uv = (record.get("scoring_features") or {}).get("uv_spots") or {}
                for metric_id, (group, key, _transform) in UV_SOURCES.items():
                    value = (uv.get(group) or {}).get(key)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        raise ValueError(f"PASS历史记录缺少UV字段: {relative_path} {metric_id}")
                    raw_values[metric_id].append(float(value))

    expected = len(pass_rows)
    if any(len(values) != expected for values in raw_values.values()):
        raise ValueError("UV参考数组长度与PASS样本数不一致")

    shadow = copy.deepcopy(official)
    shadow["scoring_profile_version"] = SHADOW_PROFILE_VERSION
    shadow["profile_role"] = "shadow"
    shadow["promotion_status"] = "not_promoted"
    shadow["official_profile_version"] = official["scoring_profile_version"]
    shadow["official_profile_sha256"] = _sha256(official_path)
    shadow["uv_weight_source"] = "engineering_default_equal_v1"
    shadow["uv_internal_metric_weights"] = {metric_id: .25 for metric_id in UV_SOURCES}
    shadow["uv_metric_semantics"] = {
        "uv_spot_density": "UV色斑单位有效皮肤面积密度",
        "uv_spot_area_ratio": "UV色斑实例Mask面积占有效分析区比例",
        "uv_p90_intensity": "UV色斑实例强度P90",
        "uv_high_intensity_target_ratio": "高强度UV色斑目标数占全部UV目标数比例",
    }
    shadow["uv_doctor_target_gap"] = {
        "requested_metric": "uv_high_intensity_area_ratio",
        "status": "historical_not_recoverable_without_pixel_evidence",
        "current_available_metric": "uv_high_intensity_target_ratio",
        "rule": "不得将目标数量比例冒充高强度像素面积占比",
    }
    shadow["uv_source_trace"] = {
        "source_algorithm": "purple_analysis.uv_spots",
        "source_version": "current_historical_generation",
        "input_modality": "white_light_rgb",
        "product_modality": "UV",
        "uv_generation_version": "historical_uv_generation_v1",
    }
    for metric_id, (_group, _key, transform) in UV_SOURCES.items():
        shadow.setdefault("references", {})[metric_id] = sorted(
            transform_value(value, transform) for value in raw_values[metric_id]
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(shadow, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "path": str(target),
        "sha256": _sha256(target),
        "official_profile_unchanged_sha256": _sha256(official_path),
        "sample_count": expected,
        "metric_counts": {key: len(value) for key, value in raw_values.items()},
    }
