#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

from compare_rknn_output import (
    cosine_similarity,
    decode_yolox,
    iou_one_to_many,
    load_board_outputs,
    nms,
)

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - float(np.max(values))
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent)


def contact_point(logits: np.ndarray, radius: int = 3) -> tuple[float, float]:
    heatmap = logits.squeeze()
    if heatmap.shape != (64, 64):
        raise RuntimeError(f"unexpected contact heatmap shape: {heatmap.shape}")
    peak_y, peak_x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    x1, x2 = max(0, peak_x - radius), min(64, peak_x + radius + 1)
    y1, y2 = max(0, peak_y - radius), min(64, peak_y + radius + 1)
    window = heatmap[y1:y2, x1:x2]
    weights = softmax(window.reshape(-1)).reshape(window.shape)
    grid_y, grid_x = np.meshgrid(
        np.arange(y1, y2, dtype=np.float64),
        np.arange(x1, x2, dtype=np.float64),
        indexing="ij",
    )
    return float(np.sum(weights * grid_x)), float(np.sum(weights * grid_y))


def map_contact_point(
    point: tuple[float, float],
    roi: list[int],
) -> tuple[float, float]:
    x1, y1, x2, y2 = roi
    return (
        x1 + point[0] / 63.0 * (x2 - x1),
        y1 + point[1] / 63.0 * (y2 - y1),
    )


def tensor_metrics(
    reference: list[np.ndarray],
    board: list[np.ndarray],
) -> list[dict[str, float | int | list[int]]]:
    metrics = []
    for index, (expected, actual) in enumerate(zip(reference, board)):
        difference = np.abs(expected - actual)
        metrics.append(
            {
                "index": index,
                "shape": list(expected.shape),
                "max_abs": float(difference.max()),
                "mean_abs": float(difference.mean()),
                "cosine": cosine_similarity(expected, actual),
            }
        )
    return metrics


def aggregate_tensor_metrics(
    rows: list[list[dict[str, float | int | list[int]]]],
) -> list[dict[str, float | int]]:
    outputs = []
    for index in range(len(rows[0])):
        values = [row[index] for row in rows]
        outputs.append(
            {
                "index": index,
                "samples": len(values),
                "max_abs": max(float(value["max_abs"]) for value in values),
                "mean_abs": float(
                    np.mean([float(value["mean_abs"]) for value in values])
                ),
                "min_cosine": min(float(value["cosine"]) for value in values),
            }
        )
    return outputs


def evaluate_device(
    session: ort.InferenceSession,
    manifest_rows: list[dict[str, object]],
    board_root: Path,
    variant: str,
    minimum_iou: float,
) -> dict[str, object]:
    all_tensor_metrics = []
    frame_rows = []
    reference_positive = 0
    board_recalled = 0
    matched_ious = []
    negative_disagreements = 0
    for row in manifest_rows:
        input_bytes = np.fromfile(row["device_input"], dtype=np.uint8)
        input_nchw = (
            input_bytes.reshape(640, 640, 3)
            .transpose(2, 0, 1)[None]
            .astype(np.float32)
        )
        reference = session.run(
            None,
            {session.get_inputs()[0].name: input_nchw},
        )
        prefix = board_root / variant / f"item_{int(row['index']):03d}"
        board = load_board_outputs(prefix, reference)
        all_tensor_metrics.append(tensor_metrics(reference, board))

        reference_detections = nms(decode_yolox(reference), 0.01, 0.65)
        board_detections = nms(decode_yolox(board), 0.01, 0.65)
        best_iou = None
        if len(reference_detections):
            reference_positive += 1
            if len(board_detections):
                overlaps = iou_one_to_many(
                    reference_detections[0, :4],
                    board_detections[:, :4],
                )
                best_iou = float(np.max(overlaps))
                matched_ious.append(best_iou)
                if best_iou >= 0.5:
                    board_recalled += 1
        elif len(board_detections):
            negative_disagreements += 1
        frame_rows.append(
            {
                "index": int(row["index"]),
                "reference_detections": int(len(reference_detections)),
                "board_detections": int(len(board_detections)),
                "best_iou": best_iou,
            }
        )

    board_recall = board_recalled / max(reference_positive, 1)
    recall_drop_points = (1.0 - board_recall) * 100.0
    minimum_matched_iou = min(matched_ious) if matched_ious else 0.0
    gate_pass = (
        minimum_matched_iou >= minimum_iou
        and recall_drop_points <= 2.0
    )
    return {
        "variant": variant,
        "samples": len(manifest_rows),
        "reference_positive_frames": reference_positive,
        "board_recalled_frames": board_recalled,
        "board_recall": board_recall,
        "recall_drop_percentage_points": recall_drop_points,
        "matched_iou_min": minimum_matched_iou,
        "matched_iou_mean": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "matched_iou_p05": float(np.percentile(matched_ious, 5))
        if matched_ious
        else 0.0,
        "negative_disagreements": negative_disagreements,
        "tensor_metrics": aggregate_tensor_metrics(all_tensor_metrics),
        "gate": {
            "minimum_matched_iou": minimum_iou,
            "maximum_recall_drop_percentage_points": 2.0,
            "pass": gate_pass,
        },
        "frames": frame_rows,
    }


def evaluate_contact(
    session: ort.InferenceSession,
    manifest_rows: list[dict[str, object]],
    board_root: Path,
    variant: str,
    maximum_error_ratio: float,
    minimum_visibility_match: float,
) -> dict[str, object]:
    all_tensor_metrics = []
    frame_rows = []
    error_ratios = []
    visibility_matches = 0
    contact_rows = [row for row in manifest_rows if row["contact_input"]]
    for row in contact_rows:
        input_bytes = np.fromfile(row["contact_input"], dtype=np.uint8)
        rgb = input_bytes.reshape(256, 256, 3).astype(np.float32) / 255.0
        input_nchw = (
            ((rgb - IMAGENET_MEAN) / IMAGENET_STD)
            .transpose(2, 0, 1)[None]
            .astype(np.float32)
        )
        reference = session.run(
            None,
            {session.get_inputs()[0].name: input_nchw},
        )
        prefix = board_root / variant / f"item_{int(row['index']):03d}"
        board = load_board_outputs(prefix, reference)
        all_tensor_metrics.append(tensor_metrics(reference, board))

        roi = [int(value) for value in row["contact_roi_xyxy"]]
        reference_point = map_contact_point(contact_point(reference[0]), roi)
        board_point = map_contact_point(contact_point(board[0]), roi)
        diagonal = math.hypot(roi[2] - roi[0], roi[3] - roi[1])
        error_ratio = math.dist(reference_point, board_point) / max(diagonal, 1.0)
        error_ratios.append(error_ratio)
        reference_visibility = int(np.argmax(reference[1].reshape(-1)))
        board_visibility = int(np.argmax(board[1].reshape(-1)))
        visibility_match = reference_visibility == board_visibility
        visibility_matches += int(visibility_match)
        frame_rows.append(
            {
                "index": int(row["index"]),
                "point_error_roi_diagonal_ratio": error_ratio,
                "reference_visibility": reference_visibility,
                "board_visibility": board_visibility,
                "visibility_match": visibility_match,
            }
        )

    visibility_match_rate = visibility_matches / max(len(contact_rows), 1)
    maximum_observed_error = max(error_ratios) if error_ratios else 1.0
    gate_pass = (
        maximum_observed_error <= maximum_error_ratio
        and visibility_match_rate >= minimum_visibility_match
    )
    return {
        "variant": variant,
        "samples": len(contact_rows),
        "point_error_roi_diagonal_ratio_max": maximum_observed_error,
        "point_error_roi_diagonal_ratio_mean": float(np.mean(error_ratios)),
        "point_error_roi_diagonal_ratio_p95": float(np.percentile(error_ratios, 95)),
        "visibility_matches": visibility_matches,
        "visibility_match_rate": visibility_match_rate,
        "tensor_metrics": aggregate_tensor_metrics(all_tensor_metrics),
        "gate": {
            "maximum_point_error_roi_diagonal_ratio": maximum_error_ratio,
            "minimum_visibility_match_rate": minimum_visibility_match,
            "pass": gate_pass,
        },
        "frames": frame_rows,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "build" / "parity" / "parity_input_manifest.json",
    )
    parser.add_argument(
        "--board-output-root",
        type=Path,
        default=root / "build" / "parity" / "board_outputs" / "outputs",
    )
    parser.add_argument(
        "--device-onnx",
        type=Path,
        default=root / "artifacts" / "onnx" / "device_yolox_s_v5_raw_opset12.onnx",
    )
    parser.add_argument(
        "--contact-onnx",
        type=Path,
        default=(
            root
            / "artifacts"
            / "onnx"
            / "contact_mobilenet_v3_small_v5_opset12.onnx"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "artifacts" / "parity" / "rknn_parity_report.json",
    )
    return parser.parse_args()


def variant_available(board_root: Path, variant: str) -> bool:
    path = board_root / variant
    return path.is_dir() and any(path.glob("item_*.output0.f32"))


def select_variant(
    variants: dict[str, dict[str, object]],
    preference: tuple[str, ...],
) -> str:
    for name in preference:
        if name in variants and bool(variants[name]["gate"]["pass"]):
            return name
    return "none"


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest["rows"]
    device_session = ort.InferenceSession(
        str(args.device_onnx),
        providers=["CPUExecutionProvider"],
    )
    contact_session = ort.InferenceSession(
        str(args.contact_onnx),
        providers=["CPUExecutionProvider"],
    )

    device_fp16 = evaluate_device(
        device_session,
        rows,
        args.board_output_root,
        "device_fp16",
        0.98,
    )
    device_int8 = evaluate_device(
        device_session,
        rows,
        args.board_output_root,
        "device_int8",
        0.95,
    )
    contact_fp16 = evaluate_contact(
        contact_session,
        rows,
        args.board_output_root,
        "contact_fp16",
        0.02,
        0.98,
    )
    contact_int8 = evaluate_contact(
        contact_session,
        rows,
        args.board_output_root,
        "contact_int8",
        0.04,
        0.95,
    )
    device_variants = {
        "fp16": device_fp16,
        "int8": device_int8,
    }
    contact_variants = {
        "fp16": contact_fp16,
        "int8": contact_int8,
    }
    if variant_available(args.board_output_root, "device_int8_mmse"):
        device_variants["int8_mmse"] = evaluate_device(
            device_session,
            rows,
            args.board_output_root,
            "device_int8_mmse",
            0.95,
        )
    if variant_available(args.board_output_root, "contact_int8_mmse"):
        contact_variants["int8_mmse"] = evaluate_contact(
            contact_session,
            rows,
            args.board_output_root,
            "contact_int8_mmse",
            0.04,
            0.95,
        )
    if variant_available(args.board_output_root, "device_int8_opt2"):
        device_variants["int8_opt2"] = evaluate_device(
            device_session,
            rows,
            args.board_output_root,
            "device_int8_opt2",
            0.95,
        )
    if variant_available(args.board_output_root, "contact_int8_opt2"):
        contact_variants["int8_opt2"] = evaluate_contact(
            contact_session,
            rows,
            args.board_output_root,
            "contact_int8_opt2",
            0.04,
            0.95,
        )
    if variant_available(args.board_output_root, "device_int8_hybrid_proposal"):
        device_variants["int8_hybrid_proposal"] = evaluate_device(
            device_session,
            rows,
            args.board_output_root,
            "device_int8_hybrid_proposal",
            0.95,
        )
    if variant_available(
        args.board_output_root, "device_int8_hybrid_regression_head"
    ):
        device_variants["int8_hybrid_regression_head"] = evaluate_device(
            device_session,
            rows,
            args.board_output_root,
            "device_int8_hybrid_regression_head",
            0.95,
        )
    if variant_available(args.board_output_root, "device_int8_hybrid_all_heads"):
        device_variants["int8_hybrid_all_heads"] = evaluate_device(
            device_session,
            rows,
            args.board_output_root,
            "device_int8_hybrid_all_heads",
            0.95,
        )
    if variant_available(args.board_output_root, "contact_int8_hybrid_proposal"):
        contact_variants["int8_hybrid_proposal"] = evaluate_contact(
            contact_session,
            rows,
            args.board_output_root,
            "contact_int8_hybrid_proposal",
            0.04,
            0.95,
        )
    selected_device = select_variant(
        device_variants,
        (
            "int8_hybrid_all_heads",
            "int8_hybrid_regression_head",
            "int8_hybrid_proposal",
            "int8_opt2",
            "int8",
            "int8_mmse",
            "fp16",
        ),
    )
    selected_contact = select_variant(
        contact_variants,
        ("int8_hybrid_proposal", "int8_opt2", "int8", "int8_mmse", "fp16"),
    )
    base_pass = bool(device_fp16["gate"]["pass"]) and bool(
        contact_fp16["gate"]["pass"]
    )
    selected_pass = selected_device != "none" and selected_contact != "none"
    int8_default_ready = (
        selected_device.startswith("int8")
        and selected_contact.startswith("int8")
    )
    result = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "board_output_root": str(args.board_output_root.resolve()),
        "device": device_variants,
        "contact": contact_variants,
        "selection": {
            "device": selected_device,
            "contact": selected_contact,
            "int8_default_ready": int8_default_ready,
            "deployment_readiness": (
                "INT8_DEFAULT_READY"
                if int8_default_ready
                else "FP16_FALLBACK_REQUIRED"
            ),
        },
        "result": "PASS" if base_pass and selected_pass else "FAIL",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "result": result["result"],
        "device_fp16": {
            "iou_min": device_fp16["matched_iou_min"],
            "recall_drop_pp": device_fp16["recall_drop_percentage_points"],
            "pass": device_fp16["gate"]["pass"],
        },
        "device_int8": {
            "iou_min": device_int8["matched_iou_min"],
            "recall_drop_pp": device_int8["recall_drop_percentage_points"],
            "pass": device_int8["gate"]["pass"],
        },
        "contact_fp16": {
            "error_max": contact_fp16["point_error_roi_diagonal_ratio_max"],
            "visibility_match": contact_fp16["visibility_match_rate"],
            "pass": contact_fp16["gate"]["pass"],
        },
        "contact_int8": {
            "error_max": contact_int8["point_error_roi_diagonal_ratio_max"],
            "visibility_match": contact_int8["visibility_match_rate"],
            "pass": contact_int8["gate"]["pass"],
        },
        "selection": result["selection"],
        "report": str(args.report.resolve()),
    }
    if "int8_mmse" in device_variants:
        value = device_variants["int8_mmse"]
        summary["device_int8_mmse"] = {
            "iou_min": value["matched_iou_min"],
            "recall_drop_pp": value["recall_drop_percentage_points"],
            "pass": value["gate"]["pass"],
        }
    if "int8_mmse" in contact_variants:
        value = contact_variants["int8_mmse"]
        summary["contact_int8_mmse"] = {
            "error_max": value["point_error_roi_diagonal_ratio_max"],
            "visibility_match": value["visibility_match_rate"],
            "pass": value["gate"]["pass"],
        }
    if "int8_opt2" in device_variants:
        value = device_variants["int8_opt2"]
        summary["device_int8_opt2"] = {
            "iou_min": value["matched_iou_min"],
            "recall_drop_pp": value["recall_drop_percentage_points"],
            "pass": value["gate"]["pass"],
        }
    if "int8_opt2" in contact_variants:
        value = contact_variants["int8_opt2"]
        summary["contact_int8_opt2"] = {
            "error_max": value["point_error_roi_diagonal_ratio_max"],
            "visibility_match": value["visibility_match_rate"],
            "pass": value["gate"]["pass"],
        }
    if "int8_hybrid_proposal" in device_variants:
        value = device_variants["int8_hybrid_proposal"]
        summary["device_int8_hybrid_proposal"] = {
            "iou_min": value["matched_iou_min"],
            "recall_drop_pp": value["recall_drop_percentage_points"],
            "pass": value["gate"]["pass"],
        }
    if "int8_hybrid_regression_head" in device_variants:
        value = device_variants["int8_hybrid_regression_head"]
        summary["device_int8_hybrid_regression_head"] = {
            "iou_min": value["matched_iou_min"],
            "recall_drop_pp": value["recall_drop_percentage_points"],
            "pass": value["gate"]["pass"],
        }
    if "int8_hybrid_all_heads" in device_variants:
        value = device_variants["int8_hybrid_all_heads"]
        summary["device_int8_hybrid_all_heads"] = {
            "iou_min": value["matched_iou_min"],
            "recall_drop_pp": value["recall_drop_percentage_points"],
            "pass": value["gate"]["pass"],
        }
    if "int8_hybrid_proposal" in contact_variants:
        value = contact_variants["int8_hybrid_proposal"]
        summary["contact_int8_hybrid_proposal"] = {
            "error_max": value["point_error_roi_diagonal_ratio_max"],
            "visibility_match": value["visibility_match_rate"],
            "pass": value["gate"]["pass"],
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
