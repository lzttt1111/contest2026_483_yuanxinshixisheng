from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from scripts.acceptance.acceptance_io import (
    AcceptanceInputError,
    load_fixed_inputs,
    load_vendor_sidecar,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_fixed_inputs_preserves_25_09_23_order_and_verifies_hashes(
    tmp_path: Path,
) -> None:
    # Given
    filenames = (
        "01_clinic28-25_RGB_M.jpg",
        "02_clinic28-09_RGB_M.jpg",
        "03_clinic28-23_RGB_M.jpg",
    )
    files = []
    for index, filename in enumerate(filenames, start=1):
        path = tmp_path / filename
        Image.new("RGB", (1, 1), color=(index, index, index)).save(path)
        files.append(
            {
                "case_id": filename.split("_")[1],
                "relative_path": filename,
                "size_bytes": path.stat().st_size,
                "width_pixels": 1,
                "height_pixels": 1,
                "sha256": _sha256(path),
            }
        )
    manifest = tmp_path / "FIXED_INPUTS_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "aisia_fixed_inputs_manifest_v1",
                "source_collection": "Clinic-Fast3",
                "input_role": "RGB_M",
                "ordered_cases": ["clinic28-25", "clinic28-09", "clinic28-23"],
                "files": files,
            }
        ),
        encoding="utf-8",
    )

    # When
    inputs = load_fixed_inputs(manifest)

    # Then
    assert tuple(item.case_id for item in inputs) == (
        "clinic28-25",
        "clinic28-09",
        "clinic28-23",
    )


def test_load_fixed_inputs_rejects_sha_mismatch(tmp_path: Path) -> None:
    # Given
    source = tmp_path / "01_clinic28-25_RGB_M.jpg"
    Image.new("RGB", (1, 1), color=(1, 1, 1)).save(source)
    manifest = tmp_path / "FIXED_INPUTS_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "aisia_fixed_inputs_manifest_v1",
                "source_collection": "Clinic-Fast3",
                "input_role": "RGB_M",
                "ordered_cases": ["clinic28-25", "clinic28-09", "clinic28-23"],
                "files": [
                    {
                        "case_id": "clinic28-25",
                        "relative_path": source.name,
                        "size_bytes": source.stat().st_size,
                        "width_pixels": 1,
                        "height_pixels": 1,
                        "sha256": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    # When / Then
    with pytest.raises(AcceptanceInputError, match="SHA256"):
        load_fixed_inputs(manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("input_role", "CP_M", "input role"),
        ("source_collection", "other-source", "source collection"),
    ),
)
def test_load_fixed_inputs_rejects_wrong_modality_or_source(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    # Given
    manifest_data = {
        "schema": "aisia_fixed_inputs_manifest_v1",
        "source_collection": "Clinic-Fast3",
        "input_role": "RGB_M",
        "ordered_cases": ["clinic28-25", "clinic28-09", "clinic28-23"],
        "files": [],
    }
    manifest_data[field] = value
    manifest = tmp_path / "FIXED_INPUTS_MANIFEST.json"
    manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

    # When / Then
    with pytest.raises(AcceptanceInputError, match=message):
        load_fixed_inputs(manifest)


def test_load_vendor_sidecar_uses_runtime_singular_env_and_fails_closed(
    tmp_path: Path,
) -> None:
    # Given
    source = Path(__file__).resolve().parents[1] / "00_runtime_assets/vendor_skin"
    root = tmp_path / "vendor_skin"
    shutil.copytree(source, root)
    module = root / "pyz/core/skin_generate.pyc"
    manifest = root / "VENDOR_RUNTIME_MANIFEST.json"
    approved_sha256 = _sha256(module)

    # When
    sidecar = load_vendor_sidecar(manifest)
    module.chmod(0o644)
    module.write_bytes(b"tampered")

    # Then
    assert sidecar.environment == {
        "DERMAVISION_VENDOR_SKIN_MODULES_ROOT": str(root / "pyz"),
        "DERMAVISION_VENDOR_SKIN_MODULE_SHA256": approved_sha256,
    }
    with pytest.raises(AcceptanceInputError, match="size|SHA256"):
        load_vendor_sidecar(manifest)


def test_load_vendor_sidecar_rejects_missing_required_service(tmp_path: Path) -> None:
    # Given
    source = Path(__file__).resolve().parents[1] / "00_runtime_assets/vendor_skin"
    root = tmp_path / "vendor_skin"
    shutil.copytree(source, root)
    service_root = root / "pyz/core/service"
    service_root.chmod(0o755)
    (service_root / "__init__.pyc").unlink()

    # When / Then
    with pytest.raises(AcceptanceInputError, match="vendor file"):
        load_vendor_sidecar(root / "VENDOR_RUNTIME_MANIFEST.json")
