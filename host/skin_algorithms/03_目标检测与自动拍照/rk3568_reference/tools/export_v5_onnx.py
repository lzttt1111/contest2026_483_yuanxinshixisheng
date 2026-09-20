#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

DEVICE_CHECKPOINT_SHA256 = "89dd527662a758a62856978e2638dcf4a38a7a3fb6de4e97c058ec652cb0c6b6"
CONTACT_CHECKPOINT_SHA256 = "299f4ea45c2d0aae742fac2c67e71d4b61dbdf0d59a44ca1d96b394075086fa2"
YOLOX_COMMIT = "419778480ab6ec0590e5d3831b3afb3b46ab2aa3"
RISK_OPERATORS = {"ScatterND", "Where", "NonMaxSuppression"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {path}: expected={expected} actual={actual}")


class RawYoloXOutputs(nn.Module):
    """Return raw logits for the stride-8/16/32 heads without grid decoding."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, images: torch.Tensor):
        features = self.backbone(images)
        outputs = []
        for index, (cls_conv, reg_conv, feature) in enumerate(
            zip(self.head.cls_convs, self.head.reg_convs, features)
        ):
            stemmed = self.head.stems[index](feature)
            cls_feature = cls_conv(stemmed)
            reg_feature = reg_conv(stemmed)
            regression = self.head.reg_preds[index](reg_feature)
            objectness = self.head.obj_preds[index](reg_feature)
            classes = self.head.cls_preds[index](cls_feature)
            outputs.append(torch.cat((regression, objectness, classes), dim=1))
        return tuple(outputs)


def decode_yolox(raw_outputs: Iterable[np.ndarray]) -> np.ndarray:
    decoded = []
    for raw, stride in zip(raw_outputs, (8.0, 16.0, 32.0)):
        batch, channels, height, width = raw.shape
        if batch != 1 or channels != 6:
            raise RuntimeError(f"unexpected YOLOX raw shape: {raw.shape}")
        prediction = raw.transpose(0, 2, 3, 1).reshape(batch, height * width, channels)
        grid_y, grid_x = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        grid = np.stack((grid_x, grid_y), axis=-1).reshape(1, height * width, 2)
        center = (prediction[..., :2] + grid) * stride
        size = np.exp(prediction[..., 2:4]) * stride
        probabilities = 1.0 / (1.0 + np.exp(-prediction[..., 4:6]))
        decoded.append(np.concatenate((center, size, probabilities), axis=-1))
    return np.concatenate(decoded, axis=1)


def graph_contract(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    operators = collections.Counter(node.op_type for node in model.graph.node)
    risks = sorted(operator for operator in operators if operator in RISK_OPERATORS)
    inputs = {
        value.name: [
            dimension.dim_value or dimension.dim_param
            for dimension in value.type.tensor_type.shape.dim
        ]
        for value in model.graph.input
    }
    outputs = {
        value.name: [
            dimension.dim_value or dimension.dim_param
            for dimension in value.type.tensor_type.shape.dim
        ]
        for value in model.graph.output
    }
    opsets = {item.domain or "ai.onnx": item.version for item in model.opset_import}
    return {
        "sha256": sha256(path),
        "opsets": opsets,
        "inputs": inputs,
        "outputs": outputs,
        "operator_counts": dict(sorted(operators.items())),
        "risk_operators": risks,
    }


def export_models(source_root: Path, output_dir: Path) -> dict[str, object]:
    source_root = source_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    yolox_root = source_root / "third_party" / "yolox"
    if not yolox_root.is_dir():
        raise FileNotFoundError(f"missing YOLOX source: {yolox_root}")
    for path in (source_root, yolox_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    device_checkpoint = (
        source_root
        / "models"
        / "camera_active_v5"
        / "device"
        / "real_only"
        / "camera_active_v5_real_only_yolox_s"
        / "best_ckpt.pth"
    )
    contact_checkpoint = source_root / "models" / "camera_active_v5" / "contact" / "best.pt"
    require_sha256(device_checkpoint, DEVICE_CHECKPOINT_SHA256)
    require_sha256(contact_checkpoint, CONTACT_CHECKPOINT_SHA256)

    from src.contact_point_model import ContactPointNet
    from src.yolox_inference import load_yolox_model

    os.environ["MEDVISION_PILOT_DATA_DIR"] = str(
        source_root / "datasets" / "camera_active_v5" / "real_only"
    )
    os.environ["MEDVISION_PILOT_OUTPUT_DIR"] = str(device_checkpoint.parent)

    _, device_model = load_yolox_model(
        source_root,
        device_checkpoint,
        torch.device("cpu"),
        source_root / "configs" / "yolox_s_device_v2.py",
    )
    device_model.eval()
    device_model.head.decode_in_inference = True
    raw_device_model = RawYoloXOutputs(device_model).eval()
    device_sample = torch.full((1, 3, 640, 640), 114.0, dtype=torch.float32)

    with torch.inference_mode():
        decoded_reference = device_model(device_sample).cpu().numpy()
        raw_reference = tuple(value.cpu().numpy() for value in raw_device_model(device_sample))
    decoded_from_raw = decode_yolox(raw_reference)
    torch_coordinate_error = float(
        np.max(np.abs(decoded_reference[..., :4] - decoded_from_raw[..., :4]))
    )
    torch_score_error = float(
        np.max(
            np.abs(
                decoded_reference[..., 4] * decoded_reference[..., 5]
                - decoded_from_raw[..., 4] * decoded_from_raw[..., 5]
            )
        )
    )
    if torch_coordinate_error > 2e-4 or torch_score_error > 1e-6:
        raise RuntimeError(
            "raw YOLOX decoder mismatch: "
            f"coordinate={torch_coordinate_error} score={torch_score_error}"
        )

    device_onnx = output_dir / "device_yolox_s_v5_raw_opset12.onnx"
    torch.onnx.export(
        raw_device_model,
        device_sample,
        device_onnx,
        input_names=["images"],
        output_names=["stride8_logits", "stride16_logits", "stride32_logits"],
        opset_version=12,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    device_contract = graph_contract(device_onnx)
    if device_contract["opsets"].get("ai.onnx") != 12:
        raise RuntimeError(f"device ONNX is not opset 12: {device_contract['opsets']}")
    if device_contract["risk_operators"]:
        raise RuntimeError(
            f"device ONNX contains RKNN-risk operators: {device_contract['risk_operators']}"
        )

    device_session = ort.InferenceSession(
        str(device_onnx), providers=["CPUExecutionProvider"]
    )
    ort_raw = device_session.run(None, {"images": device_sample.numpy()})
    raw_errors = [
        float(np.max(np.abs(expected - actual)))
        for expected, actual in zip(raw_reference, ort_raw)
    ]
    ort_decoded = decode_yolox(ort_raw)
    onnx_coordinate_error = float(
        np.max(np.abs(decoded_reference[..., :4] - ort_decoded[..., :4]))
    )
    onnx_score_error = float(
        np.max(
            np.abs(
                decoded_reference[..., 4] * decoded_reference[..., 5]
                - ort_decoded[..., 4] * ort_decoded[..., 5]
            )
        )
    )
    if max(raw_errors) > 1e-4 or onnx_coordinate_error > 2e-3 or onnx_score_error > 1e-5:
        raise RuntimeError(
            "device ONNX parity failed: "
            f"raw={raw_errors} coordinate={onnx_coordinate_error} score={onnx_score_error}"
        )

    contact_payload = torch.load(
        contact_checkpoint, map_location="cpu", weights_only=False
    )
    contact_model = ContactPointNet(pretrained=False).eval()
    contact_model.load_state_dict(contact_payload["model_state"])
    contact_sample = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    with torch.inference_mode():
        contact_reference = tuple(
            value.cpu().numpy() for value in contact_model(contact_sample)
        )

    contact_onnx = output_dir / "contact_mobilenet_v3_small_v5_opset12.onnx"
    torch.onnx.export(
        contact_model,
        contact_sample,
        contact_onnx,
        input_names=["roi_image"],
        output_names=["contact_heatmap_logits", "contact_visibility_logits"],
        opset_version=12,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    contact_contract = graph_contract(contact_onnx)
    if contact_contract["opsets"].get("ai.onnx") != 12:
        raise RuntimeError(f"contact ONNX is not opset 12: {contact_contract['opsets']}")
    if contact_contract["risk_operators"]:
        raise RuntimeError(
            f"contact ONNX contains RKNN-risk operators: {contact_contract['risk_operators']}"
        )

    contact_session = ort.InferenceSession(
        str(contact_onnx), providers=["CPUExecutionProvider"]
    )
    contact_actual = contact_session.run(None, {"roi_image": contact_sample.numpy()})
    contact_errors = [
        float(np.max(np.abs(expected - actual)))
        for expected, actual in zip(contact_reference, contact_actual)
    ]
    if max(contact_errors) > 2e-4:
        raise RuntimeError(f"contact ONNX parity failed: {contact_errors}")

    return {
        "schema_version": 1,
        "source_root": str(source_root),
        "source_read_only": True,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "sources": {
            "device_checkpoint": {
                "path": str(device_checkpoint),
                "sha256": DEVICE_CHECKPOINT_SHA256,
            },
            "contact_checkpoint": {
                "path": str(contact_checkpoint),
                "sha256": CONTACT_CHECKPOINT_SHA256,
            },
            "yolox_source": {
                "path": str(yolox_root),
                "commit": YOLOX_COMMIT,
            },
        },
        "device": {
            "path": str(device_onnx.resolve()),
            "contract": device_contract,
            "raw_output_shapes": [list(value.shape) for value in raw_reference],
            "torch_decode_coordinate_max_abs_error": torch_coordinate_error,
            "torch_decode_score_max_abs_error": torch_score_error,
            "onnx_raw_output_max_abs_errors": raw_errors,
            "onnx_decode_coordinate_max_abs_error": onnx_coordinate_error,
            "onnx_decode_score_max_abs_error": onnx_score_error,
            "input_color": "BGR",
            "input_dtype": "float32",
            "input_scale": 1.0,
            "cpp_decode": {
                "strides": [8, 16, 32],
                "regression": "xy=(raw_xy+grid)*stride; wh=exp(raw_wh)*stride",
                "scores": "sigmoid(objectness)*sigmoid(class_logit)",
            },
        },
        "contact": {
            "path": str(contact_onnx.resolve()),
            "contract": contact_contract,
            "onnx_output_max_abs_errors": contact_errors,
            "input_color": "RGB",
            "input_dtype": "float32",
            "input_normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "roi_expansion": 0.2,
            "heatmap_size": [64, 64],
            "visibility_classes": [
                "visible",
                "occluded_inferable",
                "not_annotatable",
            ],
            "decoder_radius": 3,
        },
        "result": "PASS",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "onnx",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = export_models(args.source_root, args.output_dir)
    manifest_path = args.output_dir / "export_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"RK3568_ONNX_EXPORT=PASS manifest={manifest_path}")
    print(
        "DEVICE_ONNX_SHA256="
        f"{manifest['device']['contract']['sha256']} "
        "CONTACT_ONNX_SHA256="
        f"{manifest['contact']['contract']['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
