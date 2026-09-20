from __future__ import annotations

import csv
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .clinical_reports import (
    build_acne_report,
    build_wrinkle_report,
    write_medical_v2_report,
    write_report_csv,
)
from .medical_v2_schema import (
    to_chinese_document,
    to_english_document,
)
from .final_output import write_final_output


class ReviewOutputExporter:
    """把完整运行现场整理为面向人工验收的精简目录。"""

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _copy(source: str | Path | None, target: Path) -> Path:
        if not source:
            raise FileNotFoundError(f"缺少正式结果来源: {target.name}")
        source_path = Path(source)
        if not source_path.is_file() or source_path.stat().st_size <= 0:
            raise FileNotFoundError(f"正式结果不存在或为空: {source_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        return target

    @staticmethod
    def _relative(path: Path, root: Path) -> str:
        return path.relative_to(root).as_posix()

    def _export_dermavision(
        self,
        items: dict[str, dict[str, Any]],
        target_root: Path,
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}

        red_dir = target_root / "七项检测" / "红区"
        red_source = Path(items["redness"]["主结果图"]).parent
        red_main = self._copy(items["redness"]["主结果图"], red_dir / "03_RBX红区结果图.jpg")
        red_instances = self._copy(red_source / "06_VISIA红色区实例图.jpg", red_dir / "06_VISIA红色区实例图.jpg")
        red_csv = self._copy(items["redness"]["量化CSV"], red_dir / "红区量化指标.csv")
        red_json = self._copy(items["redness"]["量化JSON"], red_dir / "红区量化指标.json")
        red_v2_csv = self._copy(red_source / "红区医学量化指标_V2.csv", red_dir / "红区医学量化指标_V2.csv")
        result["redness"] = self._entry("红区", target_root, red_main, red_json, red_v2_csv, [red_instances])

        spots_dir = target_root / "七项检测" / "斑点"
        spots_main = self._copy(items["spots"]["主结果图"], spots_dir / "01_Spots斑点结果图.jpg")
        spots_csv = self._copy(items["spots"]["量化CSV"], spots_dir / "02_Spots量化指标.csv")
        spots_json = self._copy(items["spots"]["量化JSON"], spots_dir / "02_Spots量化指标.json")
        spots_source = Path(items["spots"]["主结果图"]).parent
        spots_v2_csv = self._copy(spots_source / "02_Spots医学量化指标_V2.csv", spots_dir / "02_Spots医学量化指标_V2.csv")
        result["spots"] = self._entry("可见斑点", target_root, spots_main, spots_json, spots_v2_csv)

        brown_dir = target_root / "七项检测" / "棕区"
        brown_source = Path(items["brown"]["主结果图"]).parent
        brown_main = self._copy(items["brown"]["主结果图"], brown_dir / "01_RBX棕区结果图.jpg")
        brown_instances = self._copy(brown_source / "02_VISIA棕色斑实例图.jpg", brown_dir / "02_VISIA棕色斑实例图.jpg")
        brown_csv = self._copy(items["brown"]["量化CSV"], brown_dir / "棕色斑量化指标.csv")
        brown_json = self._copy(items["brown"]["量化JSON"], brown_dir / "棕色斑量化指标.json")
        brown_v2_csv = self._copy(brown_source / "棕色斑医学量化指标_V2.csv", brown_dir / "棕色斑医学量化指标_V2.csv")
        result["brown"] = self._entry("棕区", target_root, brown_main, brown_json, brown_v2_csv, [brown_instances])

        texture_dir = target_root / "七项检测" / "纹理"
        texture_main = self._copy(items["texture"]["主结果图"], texture_dir / "01_纹理检测结果图.jpg")
        texture_csv = self._copy(items["texture"]["量化CSV"], texture_dir / "纹理量化指标.csv")
        texture_json = self._copy(items["texture"]["量化JSON"], texture_dir / "纹理量化指标.json")
        texture_source = Path(items["texture"]["主结果图"]).parent
        texture_v2_csv = self._copy(texture_source / "纹理医学量化指标_V2.csv", texture_dir / "纹理医学量化指标_V2.csv")
        result["texture"] = self._entry("纹理", target_root, texture_main, texture_json, texture_v2_csv)

        pores_dir = target_root / "七项检测" / "毛孔"
        pores_main = self._copy(items["pores"]["主结果图"], pores_dir / "01_毛孔检测结果图.jpg")
        pores_csv = self._copy(items["pores"]["量化CSV"], pores_dir / "毛孔量化指标.csv")
        pores_json = self._copy(items["pores"]["量化JSON"], pores_dir / "毛孔量化指标.json")
        pores_source = Path(items["pores"]["主结果图"]).parent
        pores_v2_csv = self._copy(pores_source / "毛孔医学量化指标_V2.csv", pores_dir / "毛孔医学量化指标_V2.csv")
        result["pores"] = self._entry("毛孔", target_root, pores_main, pores_json, pores_v2_csv)

        purple_dir = target_root / "七项检测" / "紫区"
        purple_source = Path(items["uv_spots"]["主结果图"]).parent
        uv_base = self._copy(purple_source / "01_紫外线色斑底图.png", purple_dir / "01_紫外线色斑底图.png")
        uv_main = self._copy(items["uv_spots"]["主结果图"], purple_dir / "02_紫外线色斑检测结果.jpg")
        porphyrin_base = self._copy(purple_source / "03_紫质荧光底图.png", purple_dir / "03_紫质荧光底图.png")
        porphyrin_main = self._copy(items["porphyrin"]["主结果图"], purple_dir / "04_紫质检测结果.jpg")
        purple_csv = self._copy(items["uv_spots"]["量化CSV"], purple_dir / "紫区量化指标.csv")
        purple_json = self._copy(items["uv_spots"]["量化JSON"], purple_dir / "紫区量化指标.json")
        purple_v2_csv = self._copy(purple_source / "紫区医学量化指标_V2.csv", purple_dir / "紫区医学量化指标_V2.csv")
        result["uv_spots"] = self._entry(
            "紫外线色斑", target_root, uv_main, purple_json, purple_v2_csv, [uv_base]
        )
        result["porphyrin"] = self._entry(
            "紫质", target_root, porphyrin_main, purple_json, purple_v2_csv, [porphyrin_base]
        )
        return result

    @staticmethod
    def _entry(
        label: str,
        root: Path,
        main: Path,
        metric_json: Path,
        metric_csv: Path,
        extras: list[Path] | None = None,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "项目": label,
            "状态": "success",
            "主结果图": main.relative_to(root).as_posix(),
            "量化JSON": metric_json.relative_to(root).as_posix(),
            "量化CSV": metric_csv.relative_to(root).as_posix(),
        }
        if extras:
            entry["附加结果图"] = [path.relative_to(root).as_posix() for path in extras]
        return entry

    def export_wrinkle_acne(
        self,
        items: dict[str, dict[str, Any]],
        target_root: Path,
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        wrinkle_dir = target_root / "皱纹"
        wrinkle_main = self._copy(items["wrinkle"]["主结果图"], wrinkle_dir / "01_皱纹检测结果图.jpg")
        wrinkle_csv = wrinkle_dir / "皱纹量化指标.csv"
        wrinkle_json = wrinkle_dir / "皱纹量化指标.json"
        wrinkle_source_json = Path(items["wrinkle"]["量化JSON"])
        wrinkle_raw = json.loads(wrinkle_source_json.read_text(encoding="utf-8-sig"))
        wrinkle_report = build_wrinkle_report(wrinkle_raw)
        write_report_csv(wrinkle_report, wrinkle_csv)
        wrinkle_v2_csv = wrinkle_dir / "皱纹医学量化指标_V2.csv"
        wrinkle_v2 = write_medical_v2_report(
            wrinkle_report, "wrinkle", wrinkle_v2_csv, None
        )
        wrinkle_json.write_text(
            json.dumps(
                to_english_document(wrinkle_v2),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        result["wrinkle"] = self._entry("皱纹", target_root, wrinkle_main, wrinkle_json, wrinkle_v2_csv)

        acne_dir = target_root / "痤疮"
        acne_main = self._copy(items["acne"]["主结果图"], acne_dir / "01_痤疮检测结果图.jpg")
        acne_csv = acne_dir / "痤疮量化指标.csv"
        acne_json = acne_dir / "痤疮量化指标.json"
        acne_source_json = Path(items["acne"]["量化JSON"])
        acne_raw = json.loads(acne_source_json.read_text(encoding="utf-8-sig"))
        acne_report = build_acne_report(acne_raw)
        write_report_csv(acne_report, acne_csv)
        acne_v2_csv = acne_dir / "痤疮医学量化指标_V2.csv"
        acne_v2 = write_medical_v2_report(acne_report, "acne", acne_v2_csv, None)
        acne_json.write_text(
            json.dumps(
                to_english_document(acne_v2),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        result["acne"] = self._entry("痤疮", target_root, acne_main, acne_json, acne_v2_csv)
        return result

    def export_added(
        self,
        items: dict[str, dict[str, Any]],
        target_root: Path,
    ) -> dict[str, dict[str, Any]]:
        definitions = {
            "surface_gloss": ("10_油光", "表面油光", "表面油光"),
            "vascular": ("11_血管样结构", "血管样结构", "血管样结构"),
            "contour_firmness": ("12_轮廓紧致度", "轮廓紧致度", "轮廓紧致度"),
        }
        exported: dict[str, dict[str, Any]] = {}
        for key, (folder, prefix, label) in definitions.items():
            if key not in items:
                continue
            item = items[key]
            target = target_root / "新增三项检测" / folder
            source_dir = Path(item["主结果图"]).parent
            main = self._copy(item["主结果图"], target / f"01_{prefix}检测结果图.jpg")
            compact = self._copy(item["量化CSV"], target / f"{prefix}量化指标.csv")
            metrics_json = self._copy(item["量化JSON"], target / f"{prefix}量化指标.json")
            medical = self._copy(
                source_dir / f"{prefix}医学量化指标_V2.csv",
                target / f"{prefix}医学量化指标_V2.csv",
            )
            entry = self._entry(label, target_root, main, metrics_json, compact)
            entry["医学V2CSV"] = medical.relative_to(target_root).as_posix()
            exported[key] = entry
        return exported

    @staticmethod
    def _write_twelve_summary(
        target_root: Path,
        items: dict[str, dict[str, Any]],
        input_name: str | None,
    ) -> None:
        index = {
            "输入图片": input_name,
            "状态": "success",
            "成功项目数": len(items),
            "项目总数": len(items),
            "十二项结果": items,
        }
        (target_root / "十二项检测结果索引.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        compact = {
            key: json.loads((target_root / item["量化JSON"]).read_text(encoding="utf-8-sig"))
            for key, item in items.items()
        }
        (target_root / "十二项核心量化指标.json").write_text(
            json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with (target_root / "十二项核心量化指标.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("检测项目", "指标", "数值"))
            for key, metrics in compact.items():
                for name, value in metrics.items():
                    if isinstance(value, (str, int, float)) or value is None:
                        writer.writerow((key, name, value))

        summary_rows: list[dict[str, Any]] = []
        for key, item in items.items():
            medical_path = item.get("医学V2CSV") or item.get("量化CSV")
            if not medical_path:
                continue
            with (target_root / medical_path).open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                for row in csv.DictReader(stream):
                    summary_rows.append({"检测项目": key, **row})
        if summary_rows:
            columns: list[str] = []
            for row in summary_rows:
                for column in row:
                    if column not in columns:
                        columns.append(column)
            with (target_root / "十二项医学量化指标_V2.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(summary_rows)

    @staticmethod
    def _write_medical_v2_summary(
        target_root: Path, review_items: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        documents: dict[str, Any] = {}
        long_rows: list[tuple[Any, ...]] = []
        for key, item in review_items.items():
            document = json.loads(
                (target_root / item["量化JSON"]).read_text(encoding="utf-8-sig")
            )
            document = document.get("medical_metrics_v2", document)
            document = to_chinese_document(document)
            # 九项汇总只保存全面部指标和分文件索引，避免重复嵌入九份分区
            # 数据造成数千行 JSON；分区详情仍保存在各项目 V2 JSON 中。
            documents[key] = {
                "检测项目": document.get("检测项目", item["项目"]),
                "指标版本": document.get("指标版本"),
                "评分状态": document.get("评分状态"),
                "总体指标": document.get("总体指标", {}),
                "左右比较": document.get("左右比较", {}),
                "分区明细文件": item["量化JSON"],
            }
            for level_name, groups in document.get("总体指标", {}).items():
                level = "核心" if level_name == "核心指标" else "辅助"
                for dimension, metrics in groups.items():
                    for name, value in metrics.items():
                        unit = "—"
                        if "（" in name and name.endswith("）"):
                            unit = name.rsplit("（", 1)[-1][:-1]
                        long_rows.append((
                            document.get("检测项目", item["项目"]), level,
                            dimension, name, value, unit, "可评估",
                        ))
        summary = {
            "指标版本": "medical_metrics_v2_20260728",
            "评分状态": "uncalibrated",
            "九项医学量化": documents,
        }
        with (target_root / "九项医学量化指标_V2.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow((
                "检测项目", "指标层级", "二级维度", "指标名称",
                "全面部结果", "单位", "评估状态",
            ))
            writer.writerows(long_rows)
        return to_english_document(summary)

    @staticmethod
    def write_core_metrics_csv(metrics: dict[str, Any], target: Path) -> None:
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("检测项目", "状态", "指标名称", "指标值"))
            for metrics_item in (metrics.get("九项") or {}).values():
                label = metrics_item.get("项目", "未知项目")
                status = metrics_item.get("状态", "unknown")
                core_metrics = metrics_item.get("核心总体指标") or {}
                if not core_metrics:
                    writer.writerow((label, status, "暂无核心指标", ""))
                    continue
                for metric_name, metric_value in core_metrics.items():
                    if isinstance(metric_value, (dict, list)):
                        metric_value = json.dumps(metric_value, ensure_ascii=False, separators=(",", ":"))
                    writer.writerow((label, status, metric_name, metric_value))

    @staticmethod
    def _write_summary_files(
        source_root: Path,
        target_root: Path,
        source_manifest: dict[str, Any],
        review_items: dict[str, dict[str, Any]],
        input_name: str | None,
    ) -> None:
        index = {
            "输入图片": input_name,
            "状态": "success",
            "成功项目数": 9,
            "项目总数": 9,
            "九项结果": review_items,
            "评分状态": source_manifest.get("评分状态", "uncalibrated"),
        }
        (target_root / "九项检测结果索引.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        source_metrics = json.loads((source_root / "nine_metrics.json").read_text(encoding="utf-8"))
        source_metrics["九项"] = {
            key: item
            for key, item in (source_metrics.get("九项") or {}).items()
            if key in review_items
        }
        for key, item in (source_metrics.get("九项") or {}).items():
            if key in review_items:
                for field in ("主结果图", "量化JSON", "量化CSV", "附加结果图"):
                    if field in review_items[key]:
                        item[field] = review_items[key][field]
                    else:
                        item.pop(field, None)
        source_metrics["medical_metrics_v2"] = (
            ReviewOutputExporter._write_medical_v2_summary(target_root, review_items)
        )
        (target_root / "九项核心量化指标.json").write_text(
            json.dumps(source_metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        ReviewOutputExporter.write_core_metrics_csv(
            source_metrics, target_root / "九项核心量化指标.csv"
        )
        timing = json.loads((source_root / "timing.json").read_text(encoding="utf-8"))
        timing["图片"] = input_name
        timing.pop("运行ID", None)
        (target_root / "运行耗时.json").write_text(
            json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        shutil.copy2(source_root / "timing.csv", target_root / "运行耗时.csv")

    def export(
        self,
        source_root: Path,
        manifest: dict[str, Any],
        destination_name: str | None = None,
        *,
        include_input: bool = True,
    ) -> Path:
        if manifest.get("状态") != "success":
            raise RuntimeError("请求项目未全部成功，拒绝覆盖现有人工验收结果")
        all_items = manifest.get("十二项结果") or manifest.get("九项结果") or {}
        nine_keys = {
            "redness", "spots", "brown", "texture", "pores",
            "uv_spots", "porphyrin", "wrinkle", "acne",
        }
        added_keys = {"surface_gloss", "vascular", "contour_firmness"}
        if not all_items or set(all_items) - (nine_keys | added_keys):
            raise RuntimeError("检测结果索引不合法")

        image_source = Path(manifest["输入图片"])
        image_stem = destination_name or image_source.stem
        if not image_stem or image_stem in {".", ".."} or Path(image_stem).name != image_stem:
            raise ValueError(f"非法输出目录名称: {image_stem!r}")
        temporary = self.output_root / f".{image_stem}.tmp-{uuid.uuid4().hex[:8]}"
        destination = self.output_root / image_stem
        backup = self.output_root / f".{image_stem}.backup-{uuid.uuid4().hex[:8]}"
        temporary.mkdir(parents=True)
        try:
            if nine_keys.issubset(all_items) and added_keys.issubset(all_items):
                responses = manifest.get("服务原始响应") or {}
                derma_metadata = (
                    (responses.get("dermavision") or {})
                    .get("result", {})
                    .get("metadata", {})
                )
                timing_path = source_root / "timing.json"
                timing = (
                    json.loads(timing_path.read_text(encoding="utf-8"))
                    if timing_path.is_file()
                    else {}
                )
                write_final_output(
                    temporary,
                    items=all_items,
                    input_signature={"relative_path": image_source.name},
                    provenance={
                        "route": (
                            "consumer_rgb"
                            if manifest.get("采集Profile") == "consumer"
                            else "institution_vendor_rgb"
                        ),
                        "capture_profile": manifest.get(
                            "采集Profile",
                            "institution",
                        ),
                        "scoring_status": manifest.get("评分状态", "uncalibrated"),
                    },
                    quality_control={
                        "score": derma_metadata.get("quality_score"),
                        "status": derma_metadata.get("quality_status"),
                        "flags": derma_metadata.get("quality_flags") or [],
                        "metrics": derma_metadata.get("quality_metrics") or {},
                    },
                    timing=timing,
                )
                if destination.exists():
                    os.replace(destination, backup)
                try:
                    os.replace(temporary, destination)
                except Exception:
                    if backup.exists() and not destination.exists():
                        os.replace(backup, destination)
                    raise
                if backup.exists():
                    shutil.rmtree(backup)
                return destination
            input_name: str | None = None
            if include_input:
                input_suffix = image_source.suffix.lower() or ".img"
                input_target = self._copy(
                    image_source,
                    temporary / f"00_输入图片{input_suffix}",
                )
                input_name = input_target.name
            review_items: dict[str, dict[str, Any]] = {}
            if nine_keys.issubset(all_items):
                review_items.update(self._export_dermavision(all_items, temporary))
                review_items.update(self.export_wrinkle_acne(all_items, temporary))
                self._write_summary_files(source_root, temporary, manifest, review_items, input_name)
            review_items.update(self.export_added(all_items, temporary))
            if added_keys.intersection(all_items):
                self._write_twelve_summary(temporary, review_items, input_name)

            if destination.exists():
                os.replace(destination, backup)
            try:
                os.replace(temporary, destination)
            except Exception:
                if backup.exists() and not destination.exists():
                    os.replace(backup, destination)
                raise
            if backup.exists():
                shutil.rmtree(backup)
            return destination
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
