from __future__ import annotations

"""Strict single-RGB adapter for the validated vendor RED/BROWN renderer."""

from dataclasses import dataclass, replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Protocol

import cv2
import numpy as np

from src.preprocess.image_preprocessor import PreprocessResultV2
from src.utils.io_utils import cv_imread, cv_imwrite


_OUTPUT_NAMES = ("RED_M.jpg", "BROWN_M.jpg", "REDNE_M.jpg")


class VendorSkinStyleError(RuntimeError):
    """Base error for fail-closed vendor style generation."""


class VendorSkinStyleConfigurationError(VendorSkinStyleError):
    """The configured vendor module is missing or is not the approved blob."""


class EnhanceImages(Protocol):
    def __call__(
        self,
        left: np.ndarray,
        middle: np.ndarray,
        right: np.ndarray,
        output_root: str,
        outputs: list[str],
    ) -> dict[str, object]: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


BUNDLED_VENDOR_ROOT = (
    Path(__file__).resolve().parents[2]
    / "00_runtime_assets/vendor_skin/pyz"
)
APPROVED_VENDOR_SHA256 = (
    "fd590298569d80ef6b3dfd3d7043c653715b12fddb5f41dbf1fd0ff8417a1581"
)
APPROVED_VENDOR_MANIFEST_SHA256 = (
    "ca3257dc21aa914bed5e6aabe1a736c72caad81454e1aec1d0781bd703e2f9f9"
)


def _verify_vendor_tree(root: Path) -> None:
    manifest_path = root.parent / "VENDOR_RUNTIME_MANIFEST.json"
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or _sha256(manifest_path) != APPROVED_VENDOR_MANIFEST_SHA256
    ):
        raise VendorSkinStyleConfigurationError("vendor runtime manifest mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("files")
    if not isinstance(declared, list) or len(declared) != 8:
        raise VendorSkinStyleConfigurationError("vendor runtime file manifest mismatch")
    expected: set[str] = set()
    for row in declared:
        if not isinstance(row, dict):
            raise VendorSkinStyleConfigurationError("vendor runtime file entry mismatch")
        relative = str(row.get("relative_path", ""))
        expected.add(relative)
        candidate = root.parent / relative
        path = candidate.resolve()
        if (
            not relative.startswith("pyz/")
            or root.parent not in path.parents
            or not path.is_file()
            or candidate.is_symlink()
        ):
            raise VendorSkinStyleConfigurationError(
                f"vendor runtime file unavailable: {relative}"
            )
        if (
            path.stat().st_size != row.get("size_bytes")
            or _sha256(path) != row.get("sha256")
        ):
            raise VendorSkinStyleConfigurationError(
                f"vendor runtime file integrity mismatch: {relative}"
            )
    actual = {
        path.relative_to(root.parent).as_posix()
        for path in root.rglob("*.pyc")
        if path.is_file()
    }
    if actual != expected:
        raise VendorSkinStyleConfigurationError("vendor runtime file set mismatch")


def _configured_module() -> tuple[Path, str]:
    root_text = os.environ.get("DERMAVISION_VENDOR_SKIN_MODULES_ROOT", "").strip()
    expected = os.environ.get("DERMAVISION_VENDOR_SKIN_MODULE_SHA256", "").strip().lower()
    if not root_text:
        root_text = str(BUNDLED_VENDOR_ROOT)
    if not expected:
        expected = APPROVED_VENDOR_SHA256
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise VendorSkinStyleConfigurationError(
            "DERMAVISION_VENDOR_SKIN_MODULE_SHA256 must be a 64-character hex digest"
        )
    root = Path(root_text).expanduser().resolve()
    _verify_vendor_tree(root)
    module_path = root / "core" / "skin_generate.pyc"
    if not module_path.is_file() or module_path.is_symlink():
        raise VendorSkinStyleConfigurationError(
            f"configured vendor module is unavailable: {module_path}"
        )
    actual = _sha256(module_path)
    if actual != expected:
        raise VendorSkinStyleConfigurationError(
            f"vendor module SHA256 mismatch: expected={expected}, actual={actual}"
        )
    return root, actual


def _load_enhance_images() -> tuple[EnhanceImages, str]:
    root, module_sha256 = _configured_module()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    module = importlib.import_module("core.skin_generate")
    loaded_path = Path(getattr(module, "__file__", "")).resolve()
    if not loaded_path.is_relative_to(root):
        raise VendorSkinStyleConfigurationError(
            "core.skin_generate was already loaded from a different module root"
        )
    function = getattr(module, "Enhence_images", None)
    if not callable(function):
        raise VendorSkinStyleConfigurationError("vendor Enhence_images is unavailable")
    return function, module_sha256


@dataclass(frozen=True)
class VendorSkinStylePair:
    red: PreprocessResultV2
    brown: PreprocessResultV2
    redne_image: np.ndarray
    red_base_path: Path
    brown_base_path: Path
    redne_base_path: Path
    source_role: str
    provenance: dict[str, object]


class VendorSkinStyleProvider:
    """Generate RED/BROWN/REDNE by repeating the one RGB source three times."""

    version = "vendor-core.skin_generate.Enhence_images-front-only-v1"

    def __init__(self, *, enhance_images: EnhanceImages | None = None) -> None:
        self._enhance_images = enhance_images
        self._module_sha256 = "injected-test-double" if enhance_images else ""

    def _function(self) -> EnhanceImages:
        if self._enhance_images is None:
            self._enhance_images, self._module_sha256 = _load_enhance_images()
        return self._enhance_images

    @staticmethod
    def _aligned(source: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        return cv2.warpAffine(
            source,
            matrix,
            (1024, 1024),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    @staticmethod
    def _read_generated(root: Path, name: str, shape: tuple[int, int]) -> np.ndarray:
        resolved_root = root.resolve(strict=True)
        path = (resolved_root / name).resolve(strict=True)
        if not path.is_relative_to(resolved_root) or path.is_symlink() or not path.is_file():
            raise VendorSkinStyleError(f"invalid vendor output: {name}")
        image = cv_imread(str(path))
        if image is None or image.shape[:2] != shape:
            raise VendorSkinStyleError(f"invalid vendor output image: {name}")
        return image

    def build(
        self,
        preprocess_result: PreprocessResultV2,
        *,
        source_image: np.ndarray,
        output_root: Path,
        source_role: str,
    ) -> VendorSkinStylePair:
        if source_image.ndim != 3 or source_image.shape[2] != 3:
            raise VendorSkinStyleError("vendor skin-map source must be a decoded BGR image")
        generated_root = output_root.resolve()
        generated_root.mkdir(parents=True, exist_ok=False)
        result = self._function()(
            source_image,
            source_image,
            source_image,
            str(generated_root),
            ["red", "brown", "hxs"],
        )
        if not isinstance(result, dict) or result.get("status") != "sucessful!":
            raise VendorSkinStyleError(f"vendor Enhence_images failed: {result!r}")

        raw = {
            name: self._read_generated(generated_root, name, source_image.shape[:2])
            for name in _OUTPUT_NAMES
        }
        aligned = {
            name: self._aligned(image, preprocess_result.face_transform_matrix)
            for name, image in raw.items()
        }
        aligned_root = generated_root / "aligned"
        aligned_root.mkdir()
        paths: dict[str, Path] = {}
        for name, image in aligned.items():
            path = aligned_root / name.replace(".jpg", ".png")
            cv_imwrite(str(path), image)
            if not path.is_file():
                raise VendorSkinStyleError(f"cannot write aligned vendor base: {path.name}")
            paths[name] = path

        red = replace(
            preprocess_result,
            analysis_image=aligned["RED_M.jpg"].copy(),
            display_image=aligned["RED_M.jpg"].copy(),
        )
        brown = replace(
            preprocess_result,
            analysis_image=aligned["BROWN_M.jpg"].copy(),
            display_image=aligned["BROWN_M.jpg"].copy(),
        )
        debug_masks = getattr(preprocess_result, "_debug_masks", {}) or {}
        red._debug_masks = debug_masks
        brown._debug_masks = debug_masks
        provenance: dict[str, object] = {
            "actual_vendor_function_executed": True,
            "vendor_function": "core.skin_generate.Enhence_images",
            "module_sha256": self._module_sha256,
            "source_role": source_role,
            "input_mode": f"front_only_{source_role.lower()}_repeated",
            "reference_image_used": False,
            "generated_sha256": {
                name: _sha256(generated_root / name) for name in _OUTPUT_NAMES
            },
        }
        return VendorSkinStylePair(
            red=red,
            brown=brown,
            redne_image=aligned["REDNE_M.jpg"].copy(),
            red_base_path=paths["RED_M.jpg"],
            brown_base_path=paths["BROWN_M.jpg"],
            redne_base_path=paths["REDNE_M.jpg"],
            source_role=source_role,
            provenance=provenance,
        )

    def reanchor(
        self,
        pair: VendorSkinStylePair,
        preprocess_result: PreprocessResultV2,
        *,
        aligned_root: Path,
    ) -> VendorSkinStylePair:
        """Create an independent geometry view from one verified vendor generation."""

        generated_root = pair.red_base_path.parent.parent.resolve(strict=True)
        first = cv_imread(str(generated_root / _OUTPUT_NAMES[0]))
        if first is None:
            raise VendorSkinStyleError("verified vendor generation is unavailable")
        raw = {
            name: self._read_generated(generated_root, name, first.shape[:2])
            for name in _OUTPUT_NAMES
        }
        aligned = {
            name: self._aligned(image, preprocess_result.face_transform_matrix)
            for name, image in raw.items()
        }
        resolved_aligned_root = aligned_root.resolve()
        if resolved_aligned_root.parent != generated_root:
            raise VendorSkinStyleError("reanchored view must stay inside vendor generation")
        resolved_aligned_root.mkdir(exist_ok=False)
        paths: dict[str, Path] = {}
        for name, image in aligned.items():
            path = resolved_aligned_root / name.replace(".jpg", ".png")
            cv_imwrite(str(path), image)
            if not path.is_file():
                raise VendorSkinStyleError(f"cannot write reanchored vendor base: {path.name}")
            paths[name] = path

        red = replace(
            preprocess_result,
            analysis_image=aligned["RED_M.jpg"].copy(),
            display_image=aligned["RED_M.jpg"].copy(),
        )
        brown = replace(
            preprocess_result,
            analysis_image=aligned["BROWN_M.jpg"].copy(),
            display_image=aligned["BROWN_M.jpg"].copy(),
        )
        debug_masks = getattr(preprocess_result, "_debug_masks", {}) or {}
        red._debug_masks = debug_masks
        brown._debug_masks = debug_masks
        return VendorSkinStylePair(
            red=red,
            brown=brown,
            redne_image=aligned["REDNE_M.jpg"].copy(),
            red_base_path=paths["RED_M.jpg"],
            brown_base_path=paths["BROWN_M.jpg"],
            redne_base_path=paths["REDNE_M.jpg"],
            source_role=pair.source_role,
            provenance={
                **pair.provenance,
                "derived_from_shared_vendor_generation": True,
            },
        )


__all__ = [
    "VendorSkinStyleConfigurationError",
    "VendorSkinStyleError",
    "VendorSkinStylePair",
    "VendorSkinStyleProvider",
]
