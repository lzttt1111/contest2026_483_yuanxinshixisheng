#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rknn.api import RKNN


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("device", "contact"), required=True)
    parser.add_argument("--profile", default="proposal")
    parser.add_argument(
        "--hybrid-root",
        type=Path,
        default=root / "build" / "hybrid_quant",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts" / "rknn",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional edited quantization cfg; defaults to step1 output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = (args.hybrid_root / args.model).resolve()
    if args.model == "device":
        stem = "device_yolox_s_v5_raw_opset12"
        output_stem = "device_yolox_s_v5_int8_hybrid_"
    else:
        stem = "contact_mobilenet_v3_small_v5_opset12"
        output_stem = "contact_mobilenet_v3_small_v5_int8_hybrid_"
    model_input = model_dir / (stem + ".model")
    data_input = model_dir / (stem + ".data")
    config = args.config.resolve() if args.config else model_dir / (stem + ".quantization.cfg")
    for path in (model_input, data_input, config):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_path = (args.output_dir / (output_stem + args.profile + ".rknn")).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=True)
    try:
        result = rknn.hybrid_quantization_step2(
            model_input=str(model_input),
            data_input=str(data_input),
            model_quantization_cfg=str(config),
        )
        if result != 0:
            raise RuntimeError(
                "hybrid_quantization_step2 failed with " + str(result)
            )
        result = rknn.export_rknn(str(output_path))
        if result != 0:
            raise RuntimeError("rknn.export_rknn failed with " + str(result))
    finally:
        rknn.release()

    manifest_path = args.output_dir / "hybrid_conversion_manifest.json"
    previous = []
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "builds", []
        )
    record = {
        "model": args.model,
        "profile": args.profile,
        "model_input_sha256": sha256(model_input),
        "data_input_sha256": sha256(data_input),
        "config": str(config),
        "config_sha256": sha256(config),
        "rknn_path": str(output_path),
        "rknn_sha256": sha256(output_path),
        "result": "PASS",
    }
    builds = [
        item
        for item in previous
        if item.get("rknn_path") != str(output_path)
    ] + [record]
    manifest_path.write_text(
        json.dumps(
            {"schema_version": 1, "builds": builds, "result": "PASS"},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "RK3568_HYBRID_STEP2=PASS model="
        + args.model
        + " profile="
        + args.profile
        + " sha256="
        + record["rknn_sha256"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
