from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decode(path: Path) -> bool:
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED) is not None


def _semantic_metrics(item: str, path: Path) -> Any:
    data = _read_json(path)
    if item in {"redness", "spots", "brown", "texture", "pores", "uv_spots", "porphyrin"}:
        return data
    if item == "wrinkle":
        return {
            "region_metrics": data.get("region_metrics"),
            "stage2_raw_pixels": data.get("stage2_raw_pixels"),
            "stage2_center_pixels": data.get("stage2_center_pixels"),
            "stage2_recommended_pixels": data.get("stage2_recommended_pixels"),
        }
    return {
        "grading": data.get("grading"),
        "detection": {
            key: (data.get("detection") or {}).get(key)
            for key in ("count", "raw_count", "region_counts", "detections")
        },
        "max_recall": {
            key: (data.get("max_recall") or {}).get(key)
            for key in ("combined_count", "statistics")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    repeats = sorted(p for p in run.glob("repeat_*") if p.is_dir())
    names = sorted(p.name for p in repeats[0].iterdir() if (p / "manifest.json").is_file()) if repeats else []
    for name in names:
        manifests = [_read_json(rep / name / "manifest.json") for rep in repeats]
        for index, manifest in enumerate(manifests, 1):
            if manifest.get("状态") != "success" or manifest.get("成功项目数") != 9:
                errors.append(f"{name}/repeat_{index}: 九项未全部成功")
            for item, result in manifest.get("九项结果", {}).items():
                image = Path(result.get("主结果图", ""))
                metric_json = Path(result.get("量化JSON", ""))
                metric_csv = Path(result.get("量化CSV", ""))
                if not image.is_file() or not _decode(image):
                    errors.append(f"{name}/repeat_{index}/{item}: 主结果图不可解码")
                try:
                    _read_json(metric_json)
                except Exception as exc:
                    errors.append(f"{name}/repeat_{index}/{item}: JSON错误 {exc}")
                try:
                    if not list(csv.reader(metric_csv.open(encoding="utf-8-sig"))):
                        raise ValueError("empty")
                except Exception as exc:
                    errors.append(f"{name}/repeat_{index}/{item}: CSV错误 {exc}")
        if len(manifests) >= 2:
            for item in manifests[0]["九项结果"]:
                first, second = manifests[0]["九项结果"][item], manifests[1]["九项结果"][item]
                image_same = _sha(Path(first["主结果图"])) == _sha(Path(second["主结果图"]))
                metrics_same = _semantic_metrics(item, Path(first["量化JSON"])) == _semantic_metrics(item, Path(second["量化JSON"]))
                records.append({"图片": name, "项目": item, "正式图一致": image_same, "核心指标一致": metrics_same})
                if not image_same or not metrics_same:
                    errors.append(f"{name}/{item}: 两轮输出不一致")
    report = {
        "状态": "passed" if not errors else "failed",
        "运行目录": str(run), "图片数": len(names), "轮次数": len(repeats),
        "正式结果核对数": len(records), "一致性": records, "错误": errors,
    }
    output = run / "validation_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("状态", "运行目录", "图片数", "轮次数", "正式结果核对数", "错误")}, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
