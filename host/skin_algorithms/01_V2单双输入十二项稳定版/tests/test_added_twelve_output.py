from __future__ import annotations

import csv
import json
from pathlib import Path

from src.nine_analysis.review_output import ReviewOutputExporter


def _runtime_item(root: Path, key: str, prefix: str) -> dict[str, str]:
    item_root = root / key
    item_root.mkdir(parents=True)
    image = item_root / "main.jpg"
    compact = item_root / "compact.csv"
    metrics = item_root / "metrics.json"
    medical = item_root / f"{prefix}医学量化指标_V2.csv"
    image.write_bytes(b"image")
    compact.write_text("总计\n1\n", encoding="utf-8-sig")
    metrics.write_text('{"total": 1}\n', encoding="utf-8")
    with medical.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("检测范围", "评估状态", "有效皮肤面积（像素）"))
        writer.writerow(("全面部", "可评估", 100))
    return {"主结果图": str(image), "量化CSV": str(compact), "量化JSON": str(metrics)}


def test_added_items_export_compact_json_and_wide_medical_summary(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    items = {
        "surface_gloss": _runtime_item(runtime, "surface_gloss", "表面油光"),
        "vascular": _runtime_item(runtime, "vascular", "血管样结构"),
        "contour_firmness": _runtime_item(runtime, "contour_firmness", "轮廓紧致度"),
    }
    output = tmp_path / "public"
    output.mkdir()
    exporter = ReviewOutputExporter(output)
    exported = exporter.export_added(items, output)
    exporter._write_twelve_summary(output, exported, "input.jpg")

    index = json.loads((output / "十二项检测结果索引.json").read_text(encoding="utf-8"))
    assert list(index["十二项结果"]) == [
        "surface_gloss", "vascular", "contour_firmness"
    ]
    assert (output / "十二项核心量化指标.csv").is_file()
    assert (output / "十二项医学量化指标_V2.csv").is_file()
    for item in exported.values():
        assert (output / item["主结果图"]).is_file()
        assert (output / item["量化JSON"]).is_file()
        assert (output / item["量化CSV"]).is_file()
        assert (output / item["医学V2CSV"]).is_file()


def test_legacy_nine_summary_drops_added_runtime_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "runtime"
    target = tmp_path / "public"
    source.mkdir()
    target.mkdir()
    (source / "nine_metrics.json").write_text(
        json.dumps({
            "九项": {
                "redness": {"状态": "success"},
                "surface_gloss": {
                    "主结果图": "/tmp/private/gloss.jpg",
                    "量化JSON": "/tmp/private/gloss.json",
                },
            }
        }),
        encoding="utf-8",
    )
    (source / "timing.json").write_text("{}", encoding="utf-8")
    (source / "timing.csv").write_text("阶段,状态\n", encoding="utf-8")
    monkeypatch.setattr(
        ReviewOutputExporter,
        "_write_medical_v2_summary",
        staticmethod(lambda *_args: {}),
    )

    ReviewOutputExporter._write_summary_files(
        source,
        target,
        {"评分状态": "uncalibrated"},
        {"redness": {"项目": "红区", "状态": "success"}},
        "input.jpg",
    )

    public = json.loads((target / "九项核心量化指标.json").read_text(encoding="utf-8"))
    assert list(public["九项"]) == ["redness"]
    assert "/tmp/" not in json.dumps(public, ensure_ascii=False)
