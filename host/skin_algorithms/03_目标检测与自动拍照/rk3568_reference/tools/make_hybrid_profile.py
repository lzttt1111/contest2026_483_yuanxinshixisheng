#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


PROFILE_PREFIXES = {
    "regression-head": ("/reg_convs.", "/reg_preds."),
    "all-heads": (
        "/stems.",
        "/cls_convs.",
        "/reg_convs.",
        "/cls_preds.",
        "/reg_preds.",
        "/obj_preds.",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_base = (
        root
        / "build"
        / "hybrid_quant"
        / "device"
        / "device_yolox_s_v5_raw_opset12.quantization.cfg"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=tuple(PROFILE_PREFIXES), required=True
    )
    parser.add_argument("--base-config", type=Path, default=default_base)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "build" / "hybrid_quant" / "device" / "profiles",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = args.base_config.resolve()
    text = base.read_text(encoding="utf-8")
    marker = "quantize_parameters:\n"
    if marker not in text:
        raise RuntimeError("quantize_parameters section is missing")
    header, parameters = text.split(marker, 1)
    if not header.startswith("custom_quantize_layers:"):
        raise RuntimeError("unexpected hybrid configuration header")

    tensor_names = []
    for line in parameters.splitlines():
        match = re.match(r"^    ([^ ].*):$", line)
        if match:
            tensor_names.append(match.group(1))

    prefixes = PROFILE_PREFIXES[args.profile]
    selected = [
        name
        for name in tensor_names
        if name.startswith(prefixes)
        or re.fullmatch(r"stride(8|16|32)_logits(_before_conv)?", name)
    ]
    if not selected:
        raise RuntimeError(f"profile selected no tensors: {args.profile}")

    output = (args.output_dir / f"device-{args.profile}.quantization.cfg").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"custom_quantize_layers:  # profile={args.profile}",
        *(f"    {name}: float16" for name in selected),
        marker.rstrip("\n"),
        parameters.rstrip("\n"),
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "profile": args.profile,
        "base_config": str(base),
        "base_config_sha256": sha256(base),
        "output": str(output),
        "output_sha256": sha256(output),
        "selected_tensor_count": len(selected),
        "selected_tensors": selected,
        "result": "PASS",
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"MAKE_HYBRID_PROFILE=PASS profile={args.profile} "
        f"tensors={len(selected)} output={output} sha256={manifest['output_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
