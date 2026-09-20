#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def device_input(image: np.ndarray) -> tuple[np.ndarray, float]:
    ratio = min(640.0 / image.shape[0], 640.0 / image.shape[1])
    resized = cv2.resize(
        image,
        (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    output = np.full((640, 640, 3), 114, dtype=np.uint8)
    output[: resized.shape[0], : resized.shape[1]] = resized
    return output, ratio


def contact_input(
    image: np.ndarray,
    bbox_xywh: list[float],
) -> tuple[np.ndarray, list[int]]:
    x, y, width, height = [float(value) for value in bbox_xywh]
    x1 = max(0, int(math.floor(x - width * 0.2)))
    y1 = max(0, int(math.floor(y - height * 0.2)))
    x2 = min(image.shape[1], int(math.ceil(x + width * 1.2)))
    y2 = min(image.shape[0], int(math.ceil(y + height * 1.2)))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"empty contact ROI: {bbox_xywh}")
    resized_bgr = cv2.resize(
        image[y1:y2, x1:x2],
        (256, 256),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB), [x1, y1, x2, y2]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=root / "artifacts" / "calibration" / "calibration_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "build" / "parity",
    )
    parser.add_argument(
        "--board-root",
        default="/userdata/medvision/parity",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
    rows = source["golden_set"]
    if len(rows) != 50:
        raise RuntimeError(f"expected 50 golden rows, got {len(rows)}")

    device_dir = args.output_dir / "inputs" / "device"
    contact_dir = args.output_dir / "inputs" / "contact"
    device_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    output_rows = []
    device_list_rows = []
    contact_list_rows = []
    contact_count = 0
    for index, row in enumerate(rows):
        image_path = Path(row["derived"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot decode golden image: {image_path}")

        device, ratio = device_input(image)
        device_name = f"device_{index:03d}.u8"
        device_path = device_dir / device_name
        device.tofile(device_path)
        device_board_path = f"{args.board_root}/inputs/device/{device_name}"
        device_list_rows.append(
            f"{device_board_path} "
            f"{args.board_root}/outputs/DEVICE_VARIANT/item_{index:03d}"
        )

        output_row = {
            "index": index,
            "image_id": int(row["image_id"]),
            "image_path": str(image_path.resolve()),
            "image_sha256": sha256(image_path),
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "is_positive": bool(row["is_positive"]),
            "truth_bbox_xywh": row["truth_bbox_xywh"],
            "device_input": str(device_path.resolve()),
            "device_input_sha256": sha256(device_path),
            "device_ratio": ratio,
            "contact_input": None,
            "contact_input_sha256": None,
            "contact_roi_xyxy": None,
            "contact_index": None,
        }

        if row["truth_bbox_xywh"] is not None:
            contact, roi = contact_input(image, row["truth_bbox_xywh"])
            contact_name = f"contact_{index:03d}.u8"
            contact_path = contact_dir / contact_name
            contact.tofile(contact_path)
            contact_board_path = f"{args.board_root}/inputs/contact/{contact_name}"
            contact_list_rows.append(
                f"{contact_board_path} "
                f"{args.board_root}/outputs/CONTACT_VARIANT/item_{index:03d}"
            )
            output_row.update(
                {
                    "contact_input": str(contact_path.resolve()),
                    "contact_input_sha256": sha256(contact_path),
                    "contact_roi_xyxy": roi,
                    "contact_index": contact_count,
                }
            )
            contact_count += 1
        output_rows.append(output_row)

    device_list_template = args.output_dir / "device.list.template"
    contact_list_template = args.output_dir / "contact.list.template"
    device_list_template.write_text(
        "\n".join(device_list_rows) + "\n",
        encoding="utf-8",
    )
    contact_list_template.write_text(
        "\n".join(contact_list_rows) + "\n",
        encoding="utf-8",
    )
    board_lists = {}
    for template_path, placeholder, variants in (
        (
            device_list_template,
            "DEVICE_VARIANT",
            (
                "device_fp16",
                "device_int8",
                "device_int8_mmse",
                "device_int8_opt2",
                "device_int8_hybrid_proposal",
                "device_int8_hybrid_regression_head",
                "device_int8_hybrid_all_heads",
            ),
        ),
        (
            contact_list_template,
            "CONTACT_VARIANT",
            (
                "contact_fp16",
                "contact_int8",
                "contact_int8_mmse",
                "contact_int8_opt2",
                "contact_int8_hybrid_proposal",
            ),
        ),
    ):
        template = template_path.read_text(encoding="utf-8")
        for variant in variants:
            variant_path = args.output_dir / f"{variant}.list"
            variant_path.write_text(template.replace(placeholder, variant), encoding="utf-8")
            board_lists[variant] = str(variant_path.resolve())
    manifest = {
        "schema_version": 1,
        "source_manifest": str(args.calibration_manifest.resolve()),
        "source_manifest_sha256": sha256(args.calibration_manifest),
        "board_root": args.board_root,
        "device_count": len(output_rows),
        "contact_count": contact_count,
        "device_input_contract": "640x640 BGR uint8 NHWC, top-left letterbox, fill 114",
        "contact_input_contract": (
            "256x256 RGB uint8 NHWC, ground-truth device ROI expanded 20 percent"
        ),
        "device_list_template": str(device_list_template.resolve()),
        "contact_list_template": str(contact_list_template.resolve()),
        "board_lists": board_lists,
        "rows": output_rows,
        "result": "PASS",
    }
    manifest_path = args.output_dir / "parity_input_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"RK3568_PARITY_INPUTS=PASS device={len(output_rows)} "
        f"contact={contact_count} manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
