"""Fail-closed parsing of fixed-input and vendor-sidecar manifests."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from scripts.acceptance.acceptance_types import InputAsset, VendorSidecar


EXPECTED_CASES: Final = ("clinic28-25", "clinic28-09", "clinic28-23")
EXPECTED_SOURCE_COLLECTION: Final = "Clinic-Fast3"
EXPECTED_INPUT_ROLE: Final = "RGB_M"
VENDOR_ROOT_ENV: Final = "DERMAVISION_VENDOR_SKIN_MODULES_ROOT"
VENDOR_SHA_ENV: Final = "DERMAVISION_VENDOR_SKIN_MODULE_SHA256"
EXPECTED_VENDOR_MANIFEST_SHA256: Final = "ca3257dc21aa914bed5e6aabe1a736c72caad81454e1aec1d0781bd703e2f9f9"
EXPECTED_VENDOR_IMPORTS: Final = {
    "core.service": "pyz/core/service/__init__.pyc",
    "core.service.enhence_imgs_service.enhence_imgs": "pyz/core/service/enhence_imgs_service/enhence_imgs.pyc",
    "core.service.enhence_imgs_service.red_1": "pyz/core/service/enhence_imgs_service/red_1.pyc",
    "core.service.enhence_imgs_service.red_1130": "pyz/core/service/enhence_imgs_service/red_1130.pyc",
    "core.service.heatmap_service.heatmap": "pyz/core/service/heatmap_service/heatmap.pyc",
}


@dataclass(frozen=True, slots=True)
class AcceptanceInputError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


class _FixedFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    relative_path: str
    size_bytes: int
    width_pixels: int
    height_pixels: int
    sha256: str


class _FixedManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_name: str = Field(alias="schema")
    source_collection: str
    input_role: str
    ordered_cases: tuple[str, ...]
    files: tuple[_FixedFile, ...]


class _VendorContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    root_relative_to_delivery: str
    root_environment_variable: str
    sha256_environment_variable: str
    verified_module_relative_path: str
    verified_module_sha256: str


class _VendorImport(BaseModel):
    model_config = ConfigDict(frozen=True)

    module: str
    relative_path: str


class _VendorFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    relative_path: str
    size_bytes: int
    sha256: str


class _VendorManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_name: str = Field(alias="schema")
    asset_scope: str
    source_snapshot: str
    delivery_mode: str
    runtime_contract: _VendorContract
    required_imports: tuple[_VendorImport, ...]
    files: tuple[_VendorFile, ...]


def _probe_vendor_import(sidecar_root: Path) -> None:
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(sidecar_root),
    }
    try:
        subprocess.run(
            (sys.executable, "-B", "-c", "import core.skin_generate"),
            check=True,
            cwd=sidecar_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise AcceptanceInputError(detail="vendor import probe failed") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fixed_inputs(manifest_path: Path) -> tuple[InputAsset, ...]:
    """Parse and prove the immutable 25, 09, 23 fixed-input set."""

    manifest = _FixedManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.schema_name != "aisia_fixed_inputs_manifest_v1":
        raise AcceptanceInputError(detail=f"unexpected input schema: {manifest.schema_name}")
    if manifest.source_collection != EXPECTED_SOURCE_COLLECTION:
        raise AcceptanceInputError(detail="unexpected fixed input source collection")
    if manifest.input_role != EXPECTED_INPUT_ROLE:
        raise AcceptanceInputError(detail="unexpected fixed input role")
    root = manifest_path.parent.resolve()
    assets: list[InputAsset] = []
    for item in manifest.files:
        path = (root / item.relative_path).resolve()
        if root not in path.parents or not path.is_file() or path.is_symlink():
            raise AcceptanceInputError(detail=f"fixed input unavailable: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != item.sha256.lower():
            raise AcceptanceInputError(
                detail=f"fixed input SHA256 mismatch: {item.case_id}"
            )
        if path.stat().st_size != item.size_bytes:
            raise AcceptanceInputError(detail=f"fixed input size mismatch: {item.case_id}")
        try:
            with Image.open(path) as image:
                dimensions = image.size
        except (OSError, UnidentifiedImageError) as exc:
            raise AcceptanceInputError(detail=f"fixed input is not an image: {path}") from exc
        if dimensions != (item.width_pixels, item.height_pixels):
            raise AcceptanceInputError(
                detail=f"fixed input dimensions mismatch: {item.case_id}"
            )
        assets.append(
            InputAsset(
                case_id=item.case_id,
                path=path,
                size_bytes=item.size_bytes,
                width_pixels=item.width_pixels,
                height_pixels=item.height_pixels,
                sha256=actual_sha256,
            )
        )
    by_case = {asset.case_id: asset for asset in assets}
    if (
        manifest.ordered_cases != EXPECTED_CASES
        or len(assets) != len(EXPECTED_CASES)
        or set(by_case) != set(EXPECTED_CASES)
    ):
        raise AcceptanceInputError(detail="fixed inputs must be ordered 25,09,23")
    return tuple(by_case[case_id] for case_id in EXPECTED_CASES)


def load_vendor_sidecar(manifest_path: Path) -> VendorSidecar:
    """Verify the approved pyc and return only runtime-consumed environment keys."""

    if sha256_file(manifest_path) != EXPECTED_VENDOR_MANIFEST_SHA256:
        raise AcceptanceInputError(detail="vendor manifest SHA256 mismatch")
    manifest = _VendorManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    contract = manifest.runtime_contract
    if manifest.schema_name != "aisia_vendor_runtime_manifest_v1":
        raise AcceptanceInputError(detail=f"unexpected vendor schema: {manifest.schema_name}")
    if manifest.delivery_mode != "bundled_internal_git_checkout":
        raise AcceptanceInputError(detail="vendor delivery mode mismatch")
    if contract.root_environment_variable != VENDOR_ROOT_ENV:
        raise AcceptanceInputError(detail="vendor root environment contract mismatch")
    if contract.sha256_environment_variable != VENDOR_SHA_ENV:
        raise AcceptanceInputError(detail="vendor SHA environment contract mismatch")
    manifest_root = manifest_path.parent.resolve()
    required_imports = {
        item.module: item.relative_path for item in manifest.required_imports
    }
    if required_imports != EXPECTED_VENDOR_IMPORTS:
        raise AcceptanceInputError(detail="vendor required import contract mismatch")
    declared_files = {item.relative_path: item for item in manifest.files}
    if len(declared_files) != 8:
        raise AcceptanceInputError(detail="vendor file manifest mismatch")
    expected_paths = set(declared_files)
    actual_paths = {
        path.relative_to(manifest_root).as_posix()
        for path in (manifest_root / "pyz").rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise AcceptanceInputError(detail="vendor file set mismatch")
    for relative, item in declared_files.items():
        candidate = manifest_root / relative
        path = candidate.resolve()
        if (
            manifest_root not in path.parents
            or not path.is_file()
            or candidate.is_symlink()
        ):
            raise AcceptanceInputError(detail=f"vendor file unavailable: {relative}")
        if path.stat().st_size != item.size_bytes:
            raise AcceptanceInputError(detail=f"vendor file size mismatch: {relative}")
        if sha256_file(path) != item.sha256.lower():
            raise AcceptanceInputError(detail=f"vendor file SHA256 mismatch: {relative}")
    module_path = (manifest_root / contract.verified_module_relative_path).resolve()
    sidecar_root = module_path.parent.parent
    if manifest_root not in module_path.parents or not module_path.is_file() or module_path.is_symlink():
        raise AcceptanceInputError(detail=f"vendor module unavailable: {module_path}")
    actual_sha256 = sha256_file(module_path)
    if actual_sha256 != contract.verified_module_sha256.lower():
        raise AcceptanceInputError(detail="vendor module SHA256 mismatch")
    _probe_vendor_import(sidecar_root)
    return VendorSidecar(
        root=sidecar_root,
        module_path=module_path,
        module_sha256=actual_sha256,
        root_environment_variable=contract.root_environment_variable,
        sha256_environment_variable=contract.sha256_environment_variable,
    )
