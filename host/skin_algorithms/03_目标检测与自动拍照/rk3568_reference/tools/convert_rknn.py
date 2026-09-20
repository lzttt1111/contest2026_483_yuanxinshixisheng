#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from rknn.api import RKNN


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_model(
    *,
    model_name: str,
    onnx_path: Path,
    output_path: Path,
    dataset_path: Path | None,
    input_kind: str,
    quant_profile: str,
) -> dict[str, object]:
    quantized = dataset_path is not None
    rknn = RKNN(verbose=True)
    try:
        if input_kind == "device_bgr":
            config_args = {
                "mean_values": [[0.0, 0.0, 0.0]],
                "std_values": [[1.0, 1.0, 1.0]],
                "quant_img_RGB2BGR": True,
                "target_platform": "rk3566",
            }
        elif input_kind == "contact_rgb":
            config_args = {
                "mean_values": [[123.675, 116.28, 103.53]],
                "std_values": [[58.395, 57.12, 57.375]],
                "target_platform": "rk3566",
            }
        else:
            raise ValueError(f"unknown input kind: {input_kind}")
        if quantized:
            config_args["quantized_dtype"] = "asymmetric_quantized-8"
            if quant_profile == "mmse":
                config_args["quantized_method"] = "layer"
                config_args["quantized_algorithm"] = "mmse"
            elif quant_profile == "opt2":
                config_args["optimization_level"] = 2
            elif quant_profile != "normal":
                raise ValueError(f"unknown quantization profile: {quant_profile}")

        result = rknn.config(**config_args)
        if result != 0:
            raise RuntimeError(f"{model_name}: rknn.config failed with {result}")
        result = rknn.load_onnx(model=str(onnx_path))
        if result != 0:
            raise RuntimeError(f"{model_name}: rknn.load_onnx failed with {result}")
        result = rknn.build(
            do_quantization=quantized,
            dataset=str(dataset_path) if dataset_path else None,
        )
        if result != 0:
            raise RuntimeError(f"{model_name}: rknn.build failed with {result}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = rknn.export_rknn(str(output_path))
        if result != 0:
            raise RuntimeError(f"{model_name}: rknn.export_rknn failed with {result}")
    finally:
        rknn.release()

    return {
        "model": model_name,
        "precision": "int8" if quantized else "fp16",
        "quantization_profile": quant_profile,
        "target_platform": "rk3566",
        "toolkit_target_board": "rk3568",
        "onnx_path": str(onnx_path),
        "onnx_sha256": sha256(onnx_path),
        "dataset_path": str(dataset_path) if dataset_path else None,
        "dataset_sha256": sha256(dataset_path) if dataset_path else None,
        "rknn_path": str(output_path.resolve()),
        "rknn_sha256": sha256(output_path),
        "input_kind": input_kind,
        "result": "PASS",
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=root / "artifacts" / "onnx",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=root / "artifacts" / "calibration",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts" / "rknn",
    )
    parser.add_argument(
        "--precision",
        choices=("fp16", "int8", "all"),
        default="all",
    )
    parser.add_argument(
        "--quant-profile",
        choices=("normal", "mmse", "opt2"),
        default="normal",
        help="INT8 build profile; opt2 disables selected graph fusions",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device_onnx = require_file(
        args.onnx_dir / "device_yolox_s_v5_raw_opset12.onnx"
    )
    contact_onnx = require_file(
        args.onnx_dir / "contact_mobilenet_v3_small_v5_opset12.onnx"
    )
    device_dataset = require_file(
        args.calibration_dir / "device" / "dataset.txt"
    )
    contact_dataset = require_file(
        args.calibration_dir / "contact" / "dataset.txt"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    builds: list[dict[str, object]] = []
    precisions = ("fp16", "int8") if args.precision == "all" else (args.precision,)
    for precision in precisions:
        dataset_device = device_dataset if precision == "int8" else None
        dataset_contact = contact_dataset if precision == "int8" else None
        profile = args.quant_profile if precision == "int8" else "not_applicable"
        profile_suffix = f"_{profile}" if precision == "int8" and profile != "normal" else ""
        builds.append(
            build_model(
                model_name="device_yolox_s_v5_raw",
                onnx_path=device_onnx,
                output_path=args.output_dir / f"device_yolox_s_v5_{precision}{profile_suffix}.rknn",
                dataset_path=dataset_device,
                input_kind="device_bgr",
                quant_profile=profile,
            )
        )
        builds.append(
            build_model(
                model_name="contact_mobilenet_v3_small_v5",
                onnx_path=contact_onnx,
                output_path=args.output_dir
                / f"contact_mobilenet_v3_small_v5_{precision}{profile_suffix}.rknn",
                dataset_path=dataset_contact,
                input_kind="contact_rgb",
                quant_profile=profile,
            )
        )

    manifest_path = args.output_dir / "conversion_manifest.json"
    previous_builds: list[dict[str, object]] = []
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_builds = list(previous.get("builds", []))
    current_paths = {str(build["rknn_path"]) for build in builds}
    merged_builds = [
        build for build in previous_builds if str(build.get("rknn_path")) not in current_paths
    ] + builds
    manifest = {
        "schema_version": 1,
        "toolkit": {
            "package": "rknn-toolkit2",
            "version": importlib.metadata.version("rknn-toolkit2"),
            "target_platform": "rk3566",
            "board": "ATK-DLRK3568",
            "board_runtime": "1.5.2",
        },
        "builds": merged_builds,
        "result": "PASS",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"RK3568_RKNN_CONVERSION=PASS builds={len(builds)} "
        f"manifest={manifest_path}"
    )
    for build in builds:
        print(
            f"{build['model']} {build['precision']} "
            f"sha256={build['rknn_sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
