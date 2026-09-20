#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evenly_spaced(rows: list[dict], count: int) -> list[dict]:
    if count <= 0:
        return []
    if len(rows) < count:
        raise RuntimeError(f"requested {count} rows from only {len(rows)}")
    if count == 1:
        return [rows[0]]
    indices = [
        round(index * (len(rows) - 1) / (count - 1))
        for index in range(count)
    ]
    if len(set(indices)) != count:
        raise RuntimeError("even sampling produced duplicate indices")
    return [rows[index] for index in indices]


def load_coco(path: Path) -> tuple[list[dict], dict[int, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = sorted(payload["images"], key=lambda row: str(row["file_name"]))
    annotations = {
        int(row["image_id"]): row
        for row in payload["annotations"]
        if int(row.get("iscrowd", 0)) == 0
    }
    return images, annotations


def copy_image(source: Path, destination: Path) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "derived": str(destination.resolve()),
        "derived_sha256": sha256(destination),
    }


def crop_contact(
    source: Path,
    bbox: list[float],
    destination: Path,
    expansion: float = 0.2,
) -> dict[str, object]:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode image: {source}")
    x, y, width, height = [float(value) for value in bbox]
    x1 = max(0, int(x - width * expansion))
    y1 = max(0, int(y - height * expansion))
    x2 = min(image.shape[1], int(x + width * (1.0 + expansion) + 0.9999))
    y2 = min(image.shape[0], int(y + height * (1.0 + expansion) + 0.9999))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"empty ROI for {source}: {bbox}")
    roi = image[y1:y2, x1:x2]
    resized = cv2.resize(roi, (256, 256), interpolation=cv2.INTER_LINEAR)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), resized, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"cannot write derived ROI: {destination}")
    return {
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "bbox_xywh": bbox,
        "expanded_roi_xyxy": [x1, y1, x2, y2],
        "derived": str(destination.resolve()),
        "derived_sha256": sha256(destination),
    }


def prepare(source_root: Path, output_dir: Path) -> dict[str, object]:
    source_root = source_root.resolve()
    dataset_root = source_root / "datasets" / "camera_active_v5" / "real_only"
    train_annotation = dataset_root / "annotations" / "instances_train2017.json"
    test_annotation = dataset_root / "annotations" / "instances_test2017.json"
    train_images, train_boxes = load_coco(train_annotation)
    test_images, test_boxes = load_coco(test_annotation)

    device_rows = evenly_spaced(train_images, 100)
    positive_train = [row for row in train_images if int(row["id"]) in train_boxes]
    contact_rows = evenly_spaced(positive_train, 100)
    positive_test = [row for row in test_images if int(row["id"]) in test_boxes]
    negative_test = [row for row in test_images if int(row["id"]) not in test_boxes]
    golden_rows = evenly_spaced(positive_test, 48) + negative_test[:2]
    if len(golden_rows) != 50:
        raise RuntimeError(
            f"golden set requires 48 positives and 2 negatives, got {len(golden_rows)}"
        )

    device_entries = []
    device_paths = []
    for index, row in enumerate(device_rows):
        source = dataset_root / "train2017" / str(row["file_name"])
        suffix = source.suffix.lower() or ".jpg"
        destination = output_dir / "device" / "images" / f"device_{index:03d}{suffix}"
        entry = copy_image(source, destination)
        entry["image_id"] = int(row["id"])
        entry["source_group"] = row.get("source_group")
        device_entries.append(entry)
        device_paths.append(str(destination.resolve()))

    contact_entries = []
    contact_paths = []
    for index, row in enumerate(contact_rows):
        image_id = int(row["id"])
        source = dataset_root / "train2017" / str(row["file_name"])
        destination = output_dir / "contact" / "images" / f"contact_{index:03d}.jpg"
        entry = crop_contact(source, train_boxes[image_id]["bbox"], destination)
        entry["image_id"] = image_id
        contact_entries.append(entry)
        contact_paths.append(str(destination.resolve()))

    golden_entries = []
    for index, row in enumerate(golden_rows):
        image_id = int(row["id"])
        source = dataset_root / "test2017" / str(row["file_name"])
        suffix = source.suffix.lower() or ".jpg"
        destination = output_dir / "golden" / "images" / f"golden_{index:03d}{suffix}"
        entry = copy_image(source, destination)
        entry["image_id"] = image_id
        annotation = test_boxes.get(image_id)
        entry["truth_bbox_xywh"] = annotation["bbox"] if annotation else None
        entry["is_positive"] = annotation is not None
        golden_entries.append(entry)

    device_list = output_dir / "device" / "dataset.txt"
    contact_list = output_dir / "contact" / "dataset.txt"
    device_list.write_text("\n".join(device_paths) + "\n", encoding="utf-8")
    contact_list.write_text("\n".join(contact_paths) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "source_root": str(source_root),
        "source_read_only": True,
        "selection": {
            "method": "filename_sorted_even_spacing",
            "device_calibration_count": len(device_entries),
            "contact_calibration_count": len(contact_entries),
            "golden_count": len(golden_entries),
            "golden_positive_count": sum(
                1 for row in golden_entries if row["is_positive"]
            ),
            "golden_negative_count": sum(
                1 for row in golden_entries if not row["is_positive"]
            ),
        },
        "sources": {
            "train_annotation": {
                "path": str(train_annotation),
                "sha256": sha256(train_annotation),
            },
            "test_annotation": {
                "path": str(test_annotation),
                "sha256": sha256(test_annotation),
            },
        },
        "device_dataset": {
            "list": str(device_list.resolve()),
            "entries": device_entries,
        },
        "contact_dataset": {
            "list": str(contact_list.resolve()),
            "entries": contact_entries,
        },
        "golden_set": golden_entries,
        "result": "PASS",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "calibration_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


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
        default=Path(__file__).resolve().parents[1] / "artifacts" / "calibration",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = prepare(args.source_root, args.output_dir)
    print(
        "RK3568_CALIBRATION=PASS "
        f"device={manifest['selection']['device_calibration_count']} "
        f"contact={manifest['selection']['contact_calibration_count']} "
        f"golden={manifest['selection']['golden_count']}"
    )
    print(f"CALIBRATION_MANIFEST={args.output_dir / 'calibration_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
