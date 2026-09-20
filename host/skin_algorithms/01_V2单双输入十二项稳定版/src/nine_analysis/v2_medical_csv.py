from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping


COLUMNS = (
    "模块ID", "模块名称", "检测范围ID", "检测范围", "分组ID", "分组名称",
    "指标ID", "指标名称", "指标角色", "检测结果", "单位", "测量模式",
    "可用状态", "质量状态", "算法版本", "ROI版本", "医学说明",
)


def _rows(modules: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for module_id, module in modules.items():
        projections = [("full_face", "全面部", module["groups"])]
        region_names = {region["id"]: region["name"] for region in module["regions"]}
        projections.extend(
            (region_id, region_names[region_id], projection["groups"])
            for region_id, projection in module["region_groups"].items()
        )
        for region_id, region_name, groups in projections:
            for group_id, group in groups.items():
                for metric_id, metric in group["metrics"].items():
                    rows.append({
                        "模块ID": module_id, "模块名称": module["name"],
                        "检测范围ID": region_id, "检测范围": region_name,
                        "分组ID": group_id, "分组名称": group["name"],
                        "指标ID": metric_id, "指标名称": metric["name"],
                        "指标角色": "核心", "检测结果": metric["value"],
                        "单位": metric["unit_label"], "测量模式": module["measurement_mode"],
                        "可用状态": metric["availability"], "质量状态": "PASS",
                        "算法版本": "v2_proxy_registry_v1", "ROI版本": "v2_20260728",
                        "医学说明": metric["description"],
                    })
    return rows


def write_v2_medical_csv(
    path: Path,
    modules: Mapping[str, Mapping[str, object]],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(_rows(modules))


__all__ = ["write_v2_medical_csv"]
