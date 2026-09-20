#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.astype(np.float64, copy=False).reshape(-1)
    right_flat = right.astype(np.float64, copy=False).reshape(-1)
    denominator = np.linalg.norm(left_flat) * np.linalg.norm(right_flat)
    if denominator == 0:
        return 1.0 if np.array_equal(left_flat, right_flat) else 0.0
    return float(np.dot(left_flat, right_flat) / denominator)


def decode_yolox(raw_outputs: list[np.ndarray]) -> np.ndarray:
    decoded = []
    for raw, stride in zip(raw_outputs, (8.0, 16.0, 32.0)):
        _, channels, height, width = raw.shape
        if channels != 6:
            raise RuntimeError(f"expected six YOLOX channels, got {raw.shape}")
        prediction = raw.transpose(0, 2, 3, 1).reshape(-1, channels)
        grid_y, grid_x = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        grid = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)
        centers = (prediction[:, :2] + grid) * stride
        sizes = np.exp(np.clip(prediction[:, 2:4], -20.0, 20.0)) * stride
        scores = (
            1.0 / (1.0 + np.exp(-prediction[:, 4]))
            * 1.0 / (1.0 + np.exp(-prediction[:, 5]))
        )
        boxes = np.concatenate(
            (centers - sizes / 2.0, centers + sizes / 2.0),
            axis=1,
        )
        decoded.append(np.concatenate((boxes, scores[:, None]), axis=1))
    return np.concatenate(decoded, axis=0)


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    top_left = np.maximum(boxes[:, :2], box[:2])
    bottom_right = np.minimum(boxes[:, 2:4], box[2:4])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0.0), axis=1)
    area_box = np.prod(np.maximum(box[2:4] - box[:2], 0.0))
    area_boxes = np.prod(np.maximum(boxes[:, 2:4] - boxes[:, :2], 0.0), axis=1)
    return intersection / np.maximum(area_box + area_boxes - intersection, 1e-12)


def nms(decoded: np.ndarray, confidence: float, threshold: float) -> np.ndarray:
    rows = decoded[decoded[:, 4] >= confidence]
    if not len(rows):
        return np.empty((0, 5), dtype=np.float32)
    order = np.argsort(-rows[:, 4])
    kept: list[int] = []
    while len(order):
        current = int(order[0])
        kept.append(current)
        if len(order) == 1:
            break
        remaining = order[1:]
        overlaps = iou_one_to_many(rows[current, :4], rows[remaining, :4])
        order = remaining[overlaps <= threshold]
    return rows[kept]


def load_board_outputs(prefix: Path, references: list[np.ndarray]) -> list[np.ndarray]:
    outputs = []
    for index, reference in enumerate(references):
        path = Path(f"{prefix}.output{index}.f32")
        values = np.fromfile(path, dtype=np.float32)
        if values.size != reference.size:
            raise RuntimeError(
                f"{path}: expected {reference.size} floats, got {values.size}"
            )
        outputs.append(values.reshape(reference.shape))
    return outputs


def compare_device(args: argparse.Namespace) -> dict[str, object]:
    input_bytes = np.fromfile(args.input, dtype=np.uint8)
    expected = 640 * 640 * 3
    if input_bytes.size != expected:
        raise RuntimeError(f"expected {expected} input bytes, got {input_bytes.size}")
    nhwc = input_bytes.reshape(640, 640, 3)
    nchw = nhwc.transpose(2, 0, 1)[None].astype(np.float32)

    session = ort.InferenceSession(
        str(args.onnx),
        providers=["CPUExecutionProvider"],
    )
    reference = session.run(None, {session.get_inputs()[0].name: nchw})
    board = load_board_outputs(args.board_prefix, reference)

    tensor_metrics = []
    for index, (expected_tensor, actual_tensor) in enumerate(zip(reference, board)):
        difference = np.abs(expected_tensor - actual_tensor)
        tensor_metrics.append(
            {
                "index": index,
                "shape": list(expected_tensor.shape),
                "max_abs": float(difference.max()),
                "mean_abs": float(difference.mean()),
                "cosine": cosine_similarity(expected_tensor, actual_tensor),
            }
        )

    reference_detections = nms(
        decode_yolox(reference),
        args.confidence,
        args.nms_threshold,
    )
    board_detections = nms(
        decode_yolox(board),
        args.confidence,
        args.nms_threshold,
    )
    top_iou = None
    if len(reference_detections) and len(board_detections):
        top_iou = float(
            iou_one_to_many(
                reference_detections[0, :4],
                board_detections[:1, :4],
            )[0]
        )

    return {
        "kind": "device",
        "input": str(args.input.resolve()),
        "onnx": str(args.onnx.resolve()),
        "board_prefix": str(args.board_prefix.resolve()),
        "tensor_metrics": tensor_metrics,
        "reference_detection_count": int(len(reference_detections)),
        "board_detection_count": int(len(board_detections)),
        "reference_top": (
            reference_detections[0].tolist() if len(reference_detections) else None
        ),
        "board_top": board_detections[0].tolist() if len(board_detections) else None,
        "top_box_iou": top_iou,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("device",), default="device")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--board-prefix", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--nms-threshold", type=float, default=0.65)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.kind != "device":
        raise RuntimeError(f"unsupported kind: {args.kind}")
    result = compare_device(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
