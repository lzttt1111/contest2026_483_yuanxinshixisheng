from __future__ import annotations

from pathlib import Path

from src.nine_analysis.review_output import ReviewOutputExporter


class RecordingExporter(ReviewOutputExporter):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.targets: list[Path] = []

    def _copy(self, source, target: Path) -> Path:
        self.targets.append(target)
        return target


def test_red_brown_public_output_preserves_colleague_two_image_contract(
    tmp_path: Path,
) -> None:
    source = tmp_path / "runtime"
    items = {
        name: {
            "主结果图": str(source / name / "main.jpg"),
            "量化CSV": str(source / name / "metrics.csv"),
            "量化JSON": str(source / name / "metrics.json"),
        }
        for name in ("redness", "spots", "brown", "texture", "pores")
    }
    items["uv_spots"] = {
        "主结果图": str(source / "purple" / "uv.jpg"),
        "量化CSV": str(source / "purple" / "metrics.csv"),
        "量化JSON": str(source / "purple" / "metrics.json"),
    }
    items["porphyrin"] = {
        "主结果图": str(source / "purple" / "porphyrin.jpg"),
        "量化CSV": str(source / "purple" / "metrics.csv"),
        "量化JSON": str(source / "purple" / "metrics.json"),
    }

    exporter = RecordingExporter(tmp_path / "public")
    result = exporter._export_dermavision(items, tmp_path / "public")

    names = {path.name for path in exporter.targets}
    assert "06_VISIA红色区实例图.jpg" in names
    assert "02_VISIA棕色斑实例图.jpg" in names
    assert result["redness"]["附加结果图"] == [
        "七项检测/红区/06_VISIA红色区实例图.jpg",
    ]
    assert result["brown"]["附加结果图"] == [
        "七项检测/棕区/02_VISIA棕色斑实例图.jpg",
    ]
