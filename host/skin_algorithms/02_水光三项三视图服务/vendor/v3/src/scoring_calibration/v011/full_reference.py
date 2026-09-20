from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from src.utils.model_loader import load_face_landmarker

from .registry import REGISTRY, MetricSpec, registry_document
from .scoring import transform_value


FULL_REFERENCE_PROFILE_VERSION = "aisia_scoring_v0.1.1_integrity_full_reference_20260807"
FULL_REFERENCE_GATE_VERSION = "minimal_reference_gate_v0.1.1_20260807"


@dataclass(frozen=True)
class MetricEntry:
    dimension_id: str
    dimension_name: str
    group_id: str
    group_name: str
    spec: MetricSpec


def metric_entries() -> tuple[MetricEntry, ...]:
    return tuple(
        MetricEntry(dimension.id, dimension.name, group.id, group.name, metric)
        for dimension in REGISTRY
        for group in dimension.groups
        for metric in group.metrics
    )


def dotted_value(document: dict[str, Any], path: str) -> Any:
    value: Any = document
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def extract_complete_metrics(features: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    values: dict[str, float] = {}
    missing: list[str] = []
    for entry in metric_entries():
        value = dotted_value(features, entry.spec.source)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            missing.append(entry.spec.id)
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            missing.append(entry.spec.id)
            continue
        values[entry.spec.id] = numeric
    return values, missing


def _resize(image: np.ndarray, maximum: int = 1280) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, maximum / max(height, width))
    if scale >= 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _dhash(gray: np.ndarray) -> str:
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (resized[:, 1:] > resized[:, :-1]).ravel()
    return f"{sum(int(bit) << index for index, bit in enumerate(bits)):016x}"


def hamming_distance(left: str | None, right: str | None) -> int | None:
    if not left or not right or len(left) != 16 or len(right) != 16:
        return None
    return (int(left, 16) ^ int(right, 16)).bit_count()


class MinimalReferenceGate:
    """Fast, conservative gate for the historical scoring reference.

    It deliberately ignores ordinary lighting, mild pose, HDR, and mild blur.
    Only invalid/derived inputs and clearly incomplete face captures are kept
    out of the reference population.
    """

    def __init__(self) -> None:
        self.face_landmarker = load_face_landmarker(num_faces=2)

    @staticmethod
    def _mp_image(image: np.ndarray) -> mp.Image:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    @staticmethod
    def _artifact_features(image: np.ndarray, face_mask: np.ndarray) -> dict[str, Any]:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        vertical = np.linalg.norm(lab[:, 1:] - lab[:, :-1], axis=2)
        horizontal = np.linalg.norm(lab[1:] - lab[:-1], axis=2)
        vp = np.percentile(vertical, 65, axis=0)
        hp = np.percentile(horizontal, 65, axis=1)
        vlo, vhi = int(len(vp) * .3), max(int(len(vp) * .7), int(len(vp) * .3) + 1)
        hlo, hhi = int(len(hp) * .3), max(int(len(hp) * .7), int(len(hp) * .3) + 1)
        vertical_index = vlo + int(np.argmax(vp[vlo:vhi]))
        horizontal_index = hlo + int(np.argmax(hp[hlo:hhi]))
        vertical_median = max(float(np.median(vertical)), 1e-6)
        horizontal_median = max(float(np.median(horizontal)), 1e-6)
        vertical_seam = float(vp[vertical_index] / max(np.median(vp), 1e-6))
        horizontal_seam = float(hp[horizontal_index] / max(np.median(hp), 1e-6))
        vertical_coverage = float(np.mean(vertical[:, vertical_index] > max(4 * vertical_median, 10)))
        horizontal_coverage = float(np.mean(horizontal[horizontal_index, :] > max(4 * horizontal_median, 10)))

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        hue = hsv[:, :, 0]
        face = face_mask > 0
        face_pixels = max(int(np.count_nonzero(face)), 1)
        high_sat = (saturation > 158) & (value > 75) & face
        hist = np.histogram(hue[high_sat], bins=18, range=(0, 180))[0] if np.any(high_sat) else np.zeros(18)
        occupied_hues = int(np.count_nonzero(hist > max(5, int(np.sum(hist) * .015))))

        # Common result-overlay colors: red, cyan/blue, and magenta. Natural
        # skin pixels rarely create hundreds of tiny components in these bands.
        overlay_hue = ((hue <= 8) | ((hue >= 75) & (hue <= 135)) | (hue >= 142))
        overlay = (overlay_hue & (saturation > 175) & (value > 70) & face).astype(np.uint8)
        component_count = 0
        longest_contour = 0.0
        if np.count_nonzero(overlay):
            count, _, stats, _ = cv2.connectedComponentsWithStats(overlay, 8)
            if count > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                component_count = int(np.count_nonzero((areas >= 1) & (areas <= 80)))
            contours, _ = cv2.findContours(overlay, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                longest_contour = float(max(cv2.arcLength(contour, False) for contour in contours))

        canvas = (value < 48) | ((value > 247) & (saturation < 18))
        border = max(2, min(image.shape[:2]) // 40)
        border_canvas = np.concatenate(
            (canvas[:border].ravel(), canvas[-border:].ravel(), canvas[:, :border].ravel(), canvas[:, -border:].ravel())
        )
        foreground = np.where(~canvas)
        bbox_ratio = 0.0
        if foreground[0].size:
            bbox_ratio = float(
                (foreground[1].max() - foreground[1].min() + 1)
                * (foreground[0].max() - foreground[0].min() + 1)
                / canvas.size
            )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return {
            "central_vertical_seam_score": round(vertical_seam, 6),
            "central_horizontal_seam_score": round(horizontal_seam, 6),
            "central_vertical_seam_location": round(float((vertical_index + .5) / max(vertical.shape[1], 1)), 6),
            "central_horizontal_seam_location": round(float((horizontal_index + .5) / max(horizontal.shape[0], 1)), 6),
            "central_vertical_seam_coverage": round(vertical_coverage, 6),
            "central_horizontal_seam_coverage": round(horizontal_coverage, 6),
            "face_high_saturation_ratio": round(float(np.count_nonzero(high_sat) / face_pixels), 6),
            "face_high_saturation_hue_bins": occupied_hues,
            "face_overlay_palette_ratio": round(float(np.count_nonzero(overlay) / face_pixels), 6),
            "face_overlay_small_components": component_count,
            "face_overlay_longest_contour": round(longest_contour, 3),
            "canvas_background_ratio": round(float(np.mean(canvas)), 6),
            "canvas_border_ratio": round(float(np.mean(border_canvas)), 6),
            "content_bbox_ratio": round(bbox_ratio, 6),
            "blur_variance": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 6),
            "dark_ratio": round(float(np.mean(gray < 20)), 6),
            "bright_ratio": round(float(np.mean(gray > 240)), 6),
            "perceptual_hash": _dhash(gray),
        }

    def evaluate(self, path: Path) -> dict[str, Any]:
        try:
            payload = path.read_bytes()
        except OSError:
            payload = b""
        sha256 = hashlib.sha256(payload).hexdigest()
        decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR) if payload else None
        if decoded is None:
            return self._result("REJECT", ["decode_failed"], [], {}, sha256, None)

        image = _resize(decoded)
        detection = self.face_landmarker.detect(self._mp_image(image))
        faces = detection.face_landmarks or []
        if not faces:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return self._result("REJECT", ["no_face"], [], {}, sha256, _dhash(gray))
        if len(faces) != 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return self._result("REJECT", ["multiple_faces"], [], {}, sha256, _dhash(gray))

        height, width = image.shape[:2]
        points = np.asarray([(item.x * width, item.y * height) for item in faces[0]], dtype=np.float32)
        if len(points) < 468 or not np.isfinite(points).all():
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return self._result("REJECT", ["invalid_face_geometry"], [], {}, sha256, _dhash(gray))
        hull = cv2.convexHull(np.rint(points).astype(np.int32))
        face_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, hull, 255)
        face_area_ratio = float(np.count_nonzero(face_mask) / max(height * width, 1))
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        critical = {
            "forehead": float(points[10, 1] / height),
            "chin": float(points[152, 1] / height),
            "left_cheek": float(points[234, 0] / width),
            "right_cheek": float(points[454, 0] / width),
        }
        clipped_sides = sum(
            (
                critical["forehead"] < -.03,
                critical["chin"] > 1.03,
                critical["left_cheek"] < -.03,
                critical["right_cheek"] > 1.03,
            )
        )
        reasons: list[str] = []
        warnings: list[str] = []
        if face_area_ratio < .04:
            reasons.append("face_too_small")
        if clipped_sides >= 2:
            reasons.append("partial_face")
        elif clipped_sides == 1:
            warnings.append("minor_face_crop")

        features = self._artifact_features(image, face_mask)
        features.update(
            face_area_ratio=round(face_area_ratio, 6),
            face_bbox_ratio=[
                round(float(x_min / width), 6), round(float(y_min / height), 6),
                round(float(x_max / width), 6), round(float(y_max / height), 6),
            ],
            critical_landmarks=critical,
        )
        vertical_collage = (
            features["central_vertical_seam_score"] > 4.5
            and .43 <= features["central_vertical_seam_location"] <= .57
            and features["central_vertical_seam_coverage"] > .18
        )
        horizontal_collage = (
            features["central_horizontal_seam_score"] > 4.5
            and .43 <= features["central_horizontal_seam_location"] <= .57
            and features["central_horizontal_seam_coverage"] > .18
        )
        if vertical_collage or horizontal_collage:
            reasons.append("collage")
        if (
            features["canvas_background_ratio"] > .34
            and features["canvas_border_ratio"] > .72
            and features["content_bbox_ratio"] < .84
        ):
            reasons.append("multi_panel")
        if features["face_high_saturation_ratio"] > .28 and features["face_high_saturation_hue_bins"] >= 7:
            reasons.append("heatmap_or_pseudocolor")
        overlay_ratio = float(features["face_overlay_palette_ratio"])
        overlay_components = int(features["face_overlay_small_components"])
        overlay_length = float(features["face_overlay_longest_contour"])
        if (overlay_ratio > .03 and overlay_components >= 150) or (
            overlay_ratio > .015
            and overlay_components >= 50
            and overlay_length > 1.5 * max(x_max - x_min, 1)
        ):
            reasons.append("result_image")
        elif overlay_ratio > .02 and overlay_components >= 100:
            warnings.append("possible_result_overlay")

        if features["blur_variance"] < 35:
            warnings.append("blur")
        if features["dark_ratio"] > .18:
            warnings.append("underexposed")
        if features["bright_ratio"] > .18:
            warnings.append("overexposed")
        status = "REJECT" if reasons else ("REVIEW" if "possible_result_overlay" in warnings else "PASS")
        return self._result(
            status,
            sorted(set(reasons)),
            sorted(set(warnings)),
            features,
            sha256,
            str(features["perceptual_hash"]),
        )

    @staticmethod
    def _result(
        status: str,
        reasons: list[str],
        warnings: list[str],
        features: dict[str, Any],
        sha256: str,
        perceptual_hash: str | None,
    ) -> dict[str, Any]:
        return {
            "gate_version": FULL_REFERENCE_GATE_VERSION,
            "status": status,
            "reference_eligible": status == "PASS",
            "reason_codes": reasons,
            "quality_warnings": warnings,
            "quality_features": features,
            "source_sha256": sha256,
            "perceptual_hash": perceptual_hash,
        }

    def close(self) -> None:
        close = getattr(self.face_landmarker, "close", None)
        if callable(close):
            close()


FULL_REFERENCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
  relative_path TEXT PRIMARY KEY,
  content_sha256 TEXT,
  perceptual_hash TEXT,
  gate_status TEXT NOT NULL,
  reason_codes_json TEXT NOT NULL,
  warnings_json TEXT NOT NULL,
  quality_features_json TEXT NOT NULL,
  duplicate_of TEXT,
  source_jsonl TEXT NOT NULL,
  source_line INTEGER NOT NULL,
  input_status TEXT,
  algorithm_version TEXT,
  metrics_schema_version TEXT
);
CREATE INDEX IF NOT EXISTS samples_sha ON samples(content_sha256);
CREATE TABLE IF NOT EXISTS evidence (
  relative_path TEXT NOT NULL,
  metric_id TEXT NOT NULL,
  raw_value REAL NOT NULL,
  transformed_value REAL NOT NULL,
  PRIMARY KEY(relative_path, metric_id)
);
CREATE INDEX IF NOT EXISTS evidence_metric_raw ON evidence(metric_id, raw_value);
CREATE INDEX IF NOT EXISTS evidence_metric_transformed ON evidence(metric_id, transformed_value);
CREATE TABLE IF NOT EXISTS imports (
  source_path TEXT PRIMARY KEY,
  size_bytes INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL,
  last_line INTEGER NOT NULL
);
"""


def initialize_full_reference_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(FULL_REFERENCE_SCHEMA)
    return connection


def _insert_excluded(
    connection: sqlite3.Connection,
    *,
    relative: str,
    status: str,
    reasons: list[str],
    warnings: list[str],
    features: dict[str, Any],
    sha256: str | None,
    perceptual_hash: str | None,
    duplicate_of: str | None,
    source: Path,
    line_number: int,
    input_status: str,
    row: dict[str, Any],
) -> None:
    connection.execute(
        "INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            relative, sha256, perceptual_hash, status,
            json.dumps(reasons, ensure_ascii=False, sort_keys=True),
            json.dumps(warnings, ensure_ascii=False, sort_keys=True),
            json.dumps(features, ensure_ascii=False, sort_keys=True),
            duplicate_of, str(source), line_number, input_status,
            row.get("algorithm_version"), row.get("metrics_schema_version"),
        ),
    )


def ingest_full_reference(
    jsonl_paths: Iterable[Path],
    input_root: Path,
    database: Path,
    *,
    validation_sha256: set[str],
    validation_phashes: set[str],
    workspace_limit_bytes: int = 10 * 1024 ** 3,
    gate_evaluator: Callable[[Path], dict[str, Any]] | None = None,
    commit_every: int = 100,
    limit_new: int | None = None,
) -> dict[str, Any]:
    root = input_root.resolve()
    connection = initialize_full_reference_database(database)
    owned_gate = MinimalReferenceGate() if gate_evaluator is None else None
    evaluate = gate_evaluator or owned_gate.evaluate
    counters: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    try:
        stop = False
        for source in sorted(Path(item).resolve() for item in jsonl_paths):
            stat = source.stat()
            previous = connection.execute(
                "SELECT size_bytes,mtime_ns,last_line FROM imports WHERE source_path=?", (str(source),)
            ).fetchone()
            if previous and previous[:2] != (stat.st_size, stat.st_mtime_ns):
                raise RuntimeError(f"历史评分源已变更，拒绝混合断点: {source}")
            start_line = int(previous[2]) if previous else 0
            last_line = start_line
            with source.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    last_line = line_number
                    if line_number <= start_line or not line.strip():
                        continue
                    counters["seen"] += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        counters["invalid_json"] += 1
                        continue
                    relative = str((row.get("signature") or {}).get("relative_path") or "")
                    if not relative:
                        counters["missing_relative_path"] += 1
                        continue
                    if connection.execute("SELECT 1 FROM samples WHERE relative_path=?", (relative,)).fetchone():
                        counters["duplicate_relative_path"] += 1
                    else:
                        input_status = str(row.get("status") or "unknown")
                        features = row.get("scoring_features") or row.get("features")
                        values, missing = extract_complete_metrics(features if isinstance(features, dict) else {})
                        if input_status not in {"success", "partial_success"}:
                            _insert_excluded(
                                connection, relative=relative, status="REJECT", reasons=["task_failed"], warnings=[],
                                features={}, sha256=None, perceptual_hash=None, duplicate_of=None,
                                source=source, line_number=line_number, input_status=input_status, row=row,
                            )
                            reason_counts["task_failed"] += 1
                            counters["REJECT"] += 1
                        elif missing:
                            _insert_excluded(
                                connection, relative=relative, status="REJECT", reasons=["incomplete_metrics"], warnings=[],
                                features={"missing_metric_ids": missing}, sha256=None, perceptual_hash=None, duplicate_of=None,
                                source=source, line_number=line_number, input_status=input_status, row=row,
                            )
                            reason_counts["incomplete_metrics"] += 1
                            counters["REJECT"] += 1
                        else:
                            candidate = (root / relative).resolve()
                            if not candidate.is_relative_to(root) or not candidate.is_file():
                                gate = {
                                    "status": "REJECT", "reason_codes": ["source_image_not_found"],
                                    "quality_warnings": [], "quality_features": {},
                                    "source_sha256": None, "perceptual_hash": None,
                                }
                            else:
                                gate = evaluate(candidate)
                            sha256 = gate.get("source_sha256")
                            phash = gate.get("perceptual_hash")
                            reasons = list(gate.get("reason_codes") or [])
                            warnings = list(gate.get("quality_warnings") or [])
                            gate_status = str(gate.get("status") or "REJECT")
                            duplicate_of = None
                            if sha256:
                                match = connection.execute(
                                    "SELECT relative_path FROM samples WHERE content_sha256=? LIMIT 1", (sha256,)
                                ).fetchone()
                                if match:
                                    duplicate_of = str(match[0])
                                    gate_status = "REJECT"
                                    reasons.append("duplicate_exact_content")
                            if sha256 and sha256 in validation_sha256:
                                gate_status = "REJECT"
                                reasons.append("validation_sample_excluded")
                            if phash and any(
                                (distance := hamming_distance(str(phash), item)) is not None and distance <= 4
                                for item in validation_phashes
                            ):
                                gate_status = "REJECT"
                                reasons.append("validation_perceptual_duplicate")
                            reasons = sorted(set(reasons))
                            warnings = sorted(set(warnings))
                            _insert_excluded(
                                connection, relative=relative, status=gate_status, reasons=reasons, warnings=warnings,
                                features=dict(gate.get("quality_features") or {}), sha256=str(sha256) if sha256 else None,
                                perceptual_hash=str(phash) if phash else None, duplicate_of=duplicate_of,
                                source=source, line_number=line_number, input_status=input_status, row=row,
                            )
                            if gate_status == "PASS":
                                connection.executemany(
                                    "INSERT INTO evidence VALUES (?,?,?,?)",
                                    [
                                        (relative, entry.spec.id, values[entry.spec.id], transform_value(values[entry.spec.id], entry.spec.transform))
                                        for entry in metric_entries()
                                    ],
                                )
                            counters[gate_status] += 1
                            reason_counts.update(reasons)
                            warning_counts.update(warnings)
                        counters["inserted"] += 1
                    if counters["seen"] % commit_every == 0:
                        connection.execute(
                            "INSERT OR REPLACE INTO imports VALUES (?,?,?,?)",
                            (str(source), stat.st_size, stat.st_mtime_ns, line_number),
                        )
                        connection.commit()
                        size = database.stat().st_size
                        wal = database.with_name(database.name + "-wal")
                        if wal.exists():
                            size += wal.stat().st_size
                        if size > workspace_limit_bytes:
                            raise RuntimeError("全量参考工作库超过10GiB，已保留断点并安全停止")
                    if counters["seen"] % 500 == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "full_reference_progress",
                                    "seen": counters["seen"],
                                    "inserted": counters["inserted"],
                                    "PASS": counters["PASS"],
                                    "REVIEW": counters["REVIEW"],
                                    "REJECT": counters["REJECT"],
                                    "duplicate_relative_path": counters["duplicate_relative_path"],
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                    if limit_new is not None and counters["inserted"] >= limit_new:
                        stop = True
                        break
                connection.execute(
                    "INSERT OR REPLACE INTO imports VALUES (?,?,?,?)",
                    (str(source), stat.st_size, stat.st_mtime_ns, last_line),
                )
                connection.commit()
            if stop:
                break
    finally:
        if owned_gate is not None:
            owned_gate.close()
        sample_counts = dict(connection.execute("SELECT gate_status,COUNT(*) FROM samples GROUP BY gate_status").fetchall())
        cumulative_reasons: Counter[str] = Counter()
        cumulative_warnings: Counter[str] = Counter()
        for reasons_json, warnings_json in connection.execute(
            "SELECT reason_codes_json,warnings_json FROM samples"
        ):
            cumulative_reasons.update(json.loads(reasons_json))
            cumulative_warnings.update(json.loads(warnings_json))
        imported_lines = sum(int(row[0]) for row in connection.execute("SELECT last_line FROM imports"))
        unique_paths = sum(sample_counts.values())
        connection.close()
    return {
        "incremental_counters": dict(counters),
        "source_records_seen": imported_lines,
        "unique_relative_paths": unique_paths,
        "duplicate_relative_paths": imported_lines - unique_paths,
        "database_sample_counts": sample_counts,
        "reason_counts": dict(cumulative_reasons),
        "warning_counts": dict(cumulative_warnings),
    }


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q, method="linear"))


def build_full_reference_profile(database: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    connection = sqlite3.connect(database)
    references: dict[str, list[float]] = {}
    effective_counts: dict[str, int] = {}
    statistics: list[dict[str, Any]] = []
    try:
        for entry in metric_entries():
            raw = np.asarray(
                [float(row[0]) for row in connection.execute(
                    "SELECT raw_value FROM evidence WHERE metric_id=? ORDER BY raw_value", (entry.spec.id,)
                )],
                dtype=np.float64,
            )
            transformed = [float(row[0]) for row in connection.execute(
                "SELECT transformed_value FROM evidence WHERE metric_id=? ORDER BY transformed_value", (entry.spec.id,)
            )]
            if raw.size == 0:
                raise RuntimeError(f"全量参考指标无有效值: {entry.spec.id}")
            references[entry.spec.id] = transformed
            effective_counts[entry.spec.id] = int(raw.size)
            statistics.append({
                "dimension_id": entry.dimension_id,
                "dimension_name": entry.dimension_name,
                "group_id": entry.group_id,
                "group_name": entry.group_name,
                "metric_id": entry.spec.id,
                "canonical_source": entry.spec.source,
                "unit": entry.spec.unit,
                "direction": entry.spec.direction,
                "count": int(raw.size),
                "min": float(raw[0]),
                "P1": _percentile(raw, 1),
                "P5": _percentile(raw, 5),
                "P10": _percentile(raw, 10),
                "P25": _percentile(raw, 25),
                "P50": _percentile(raw, 50),
                "P75": _percentile(raw, 75),
                "P90": _percentile(raw, 90),
                "P95": _percentile(raw, 95),
                "P99": _percentile(raw, 99),
                "max": float(raw[-1]),
                "zero_rate": float(np.mean(raw == 0)),
            })
        sample_counts = dict(connection.execute(
            "SELECT gate_status,COUNT(*) FROM samples GROUP BY gate_status"
        ).fetchall())
    finally:
        connection.close()
    document = registry_document()
    profile = {
        **document,
        "scoring_profile_version": FULL_REFERENCE_PROFILE_VERSION,
        "profile_ready": min(effective_counts.values()) >= 1000,
        "reference_storage": "full_empirical_cdf_sorted_transformed_values",
        "reference_scope": "local_available_shard_0002_of_0003_existing_results",
        "quality_gate_version": FULL_REFERENCE_GATE_VERSION,
        "reference_sample_counts": sample_counts,
        "effective_metric_counts": effective_counts,
        "references": references,
    }
    return profile, statistics


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_gate_rows(database: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    connection = sqlite3.connect(database)
    rows: list[dict[str, Any]] = []
    try:
        for row in connection.execute(
            "SELECT relative_path,gate_status,reason_codes_json,warnings_json,content_sha256,perceptual_hash,"
            "duplicate_of,input_status,source_jsonl,source_line FROM samples ORDER BY relative_path"
        ):
            rows.append({
                "relative_path": row[0], "gate_status": row[1],
                "reason_codes": "|".join(json.loads(row[2])),
                "quality_warnings": "|".join(json.loads(row[3])),
                "content_sha256": row[4] or "", "perceptual_hash": row[5] or "",
                "duplicate_of": row[6] or "", "input_status": row[7] or "",
                "source_jsonl": row[8], "source_line": row[9],
            })
    finally:
        connection.close()
    counter: Counter[tuple[str, str]] = Counter()
    for row in rows:
        reasons = str(row["reason_codes"]).split("|") if row["reason_codes"] else [""]
        for reason in reasons:
            counter[(str(row["gate_status"]), reason or "none")] += 1
    summary = [
        {"gate_status": status, "reason_code": reason, "count": count}
        for (status, reason), count in sorted(counter.items())
    ]
    return rows, summary
