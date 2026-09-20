#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from rknn.api import RKNN


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_spec(root: Path, name: str) -> dict[str, object]:
    if name == "device":
        return {
            "onnx": root
            / "artifacts"
            / "onnx"
            / "device_yolox_s_v5_raw_opset12.onnx",
            "dataset": root / "artifacts" / "calibration" / "device" / "dataset.txt",
            "config": {
                "mean_values": [[0.0, 0.0, 0.0]],
                "std_values": [[1.0, 1.0, 1.0]],
                "quant_img_RGB2BGR": True,
                "quantized_dtype": "asymmetric_quantized-8",
                "target_platform": "rk3566",
            },
        }
    if name == "contact":
        return {
            "onnx": root
            / "artifacts"
            / "onnx"
            / "contact_mobilenet_v3_small_v5_opset12.onnx",
            "dataset": root / "artifacts" / "calibration" / "contact" / "dataset.txt",
            "config": {
                "mean_values": [[123.675, 116.28, 103.53]],
                "std_values": [[58.395, 57.12, 57.375]],
                "quantized_dtype": "asymmetric_quantized-8",
                "target_platform": "rk3566",
            },
        }
    raise ValueError("unknown model: " + name)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("device", "contact"), required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "build" / "hybrid_quant",
    )
    parser.add_argument(
        "--proposal-dataset-size",
        type=int,
        default=10,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = model_spec(root, args.model)
    onnx_path = Path(spec["onnx"]).resolve()
    dataset_path = Path(spec["dataset"]).resolve()
    if not onnx_path.is_file() or not dataset_path.is_file():
        raise FileNotFoundError("ONNX or calibration dataset is missing")
    if args.proposal_dataset_size <= 0:
        raise ValueError("proposal dataset size must be positive")

    output_dir = (args.output_root / args.model).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    rknn = RKNN(verbose=True)
    try:
        result = rknn.config(**spec["config"])
        if result != 0:
            raise RuntimeError("rknn.config failed with " + str(result))
        result = rknn.load_onnx(model=str(onnx_path))
        if result != 0:
            raise RuntimeError("rknn.load_onnx failed with " + str(result))
        os.chdir(output_dir)
        result = rknn.hybrid_quantization_step1(
            dataset=str(dataset_path),
            proposal=True,
            proposal_dataset_size=args.proposal_dataset_size,
        )
        if result != 0:
            raise RuntimeError(
                "hybrid_quantization_step1 failed with " + str(result)
            )
    finally:
        os.chdir(previous_cwd)
        rknn.release()

    generated = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            generated.append(
                {
                    "path": str(path),
                    "size": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "schema_version": 1,
        "model": args.model,
        "onnx": str(onnx_path),
        "onnx_sha256": sha256(onnx_path),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "proposal_dataset_size": args.proposal_dataset_size,
        "generated": generated,
        "result": "PASS",
    }
    manifest_path = output_dir / "step1_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "RK3568_HYBRID_STEP1=PASS model="
        + args.model
        + " output_dir="
        + str(output_dir)
    )
    for item in generated:
        print(Path(item["path"]).name + " sha256=" + str(item["sha256"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
