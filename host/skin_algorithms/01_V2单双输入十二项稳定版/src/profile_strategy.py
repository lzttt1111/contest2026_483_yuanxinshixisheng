from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Protocol
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import assert_never

from src.capture_profile import CaptureProfile
from src.consumer_pigment.specular import (
    build_source_and_render_specular_evidence,
    compress_specular_highlights,
)
from src.engines.vendor_skin_style import VendorSkinStyleProvider
from src.preprocess.image_preprocessor import PreprocessResultV2
from src.utils.io_utils import cv_imread, cv_imwrite

if TYPE_CHECKING:
    from src.added_algorithms.runner import AddedAlgorithmArtifacts


StyleRenderer = Callable[[PreprocessResultV2, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class SkinStyleProviderRequiredError(RuntimeError):
    profile: CaptureProfile

    def __str__(self) -> str:
        return f"{self.profile.value} profile requires vendor provider"


@dataclass(frozen=True, slots=True)
class SkinStyleBundle:
    """Profile-selected red, brown, and vascular display inputs."""

    red: PreprocessResultV2
    brown: PreprocessResultV2
    vascular_display: np.ndarray
    cleanup_root: Path | None


class SkinStyleStrategy(Protocol):
    requires_vendor: bool
    timing_key: str

    def build(
        self,
        preprocess: PreprocessResultV2,
        source_image: np.ndarray,
        output_root: Path,
        vendor_provider: VendorSkinStyleProvider | None,
    ) -> SkinStyleBundle: ...

    def finalize_red(
        self,
        bundle: SkinStyleBundle,
        base_path: str,
        overlay_path: str,
        marker_mask: np.ndarray,
        renderer: StyleRenderer,
    ) -> None: ...

    def finalize_brown(
        self,
        bundle: SkinStyleBundle,
        base_path: str,
        overlay_path: str,
        instance_mask: np.ndarray,
        renderer: StyleRenderer,
    ) -> None: ...

    def run_vascular(
        self,
        preprocess: PreprocessResultV2,
        bundle: SkinStyleBundle,
        output_dir: str | Path,
        source_name: str,
    ) -> AddedAlgorithmArtifacts: ...


@dataclass(frozen=True, slots=True)
class InstitutionStyleStrategy:
    """Keep the validated vendor RED/BROWN presentation byte path."""

    requires_vendor: bool = True
    timing_key: str = "vendor_red_brown_base"

    def build(
        self,
        preprocess: PreprocessResultV2,
        source_image: np.ndarray,
        output_root: Path,
        vendor_provider: VendorSkinStyleProvider | None,
    ) -> SkinStyleBundle:
        if vendor_provider is None:
            raise SkinStyleProviderRequiredError(CaptureProfile.INSTITUTION)
        pair = vendor_provider.build(
            preprocess,
            source_image=source_image,
            output_root=output_root,
            source_role="RGB_M",
        )
        return SkinStyleBundle(
            red=pair.red,
            brown=pair.brown,
            vascular_display=pair.red.analysis_image,
            cleanup_root=output_root,
        )

    def finalize_red(
        self,
        bundle: SkinStyleBundle,
        base_path: str,
        overlay_path: str,
        marker_mask: np.ndarray,
        renderer: StyleRenderer,
    ) -> None:
        cv_imwrite(base_path, bundle.red.analysis_image)
        cv_imwrite(
            overlay_path,
            renderer(
                bundle.red,
                bundle.red.analysis_image,
                marker_mask,
            ),
        )

    def finalize_brown(
        self,
        bundle: SkinStyleBundle,
        base_path: str,
        overlay_path: str,
        instance_mask: np.ndarray,
        renderer: StyleRenderer,
    ) -> None:
        cv_imwrite(base_path, bundle.brown.analysis_image)
        cv_imwrite(
            overlay_path,
            renderer(
                bundle.brown,
                bundle.brown.analysis_image,
                instance_mask,
            ),
        )

    def run_vascular(
        self,
        preprocess: PreprocessResultV2,
        bundle: SkinStyleBundle,
        output_dir: str | Path,
        source_name: str,
    ) -> AddedAlgorithmArtifacts:
        from src.added_algorithms.runner import run_vascular

        return run_vascular(
            preprocess,
            bundle.vascular_display,
            output_dir,
            source_name,
        )


@dataclass(frozen=True, slots=True)
class ConsumerStyleStrategy:
    """Use aligned native RGB without vendor JPEG intermediates."""

    requires_vendor: bool = False
    timing_key: str = "native_rgb_style_base"

    def build(
        self,
        preprocess: PreprocessResultV2,
        source_image: np.ndarray,
        output_root: Path,
        vendor_provider: VendorSkinStyleProvider | None,
    ) -> SkinStyleBundle:
        del source_image, output_root, vendor_provider
        return SkinStyleBundle(
            red=preprocess,
            brown=preprocess,
            vascular_display=preprocess.analysis_image,
            cleanup_root=None,
        )

    def finalize_red(
        self,
        bundle: SkinStyleBundle,
        base_path: str,
        overlay_path: str,
        marker_mask: np.ndarray,
        renderer: StyleRenderer,
    ) -> None:
        del bundle, base_path, overlay_path, marker_mask, renderer

    def finalize_brown(
        self,
        bundle: SkinStyleBundle,
        base_path: str,
        overlay_path: str,
        instance_mask: np.ndarray,
        renderer: StyleRenderer,
    ) -> None:
        base = cv_imread(base_path)
        if base is None:
            raise ValueError("consumer Brown base cannot be decoded")
        evidence = build_source_and_render_specular_evidence(
            bundle.brown.analysis_image,
            base,
            bundle.brown.skin_mask,
        )
        repaired = compress_specular_highlights(base, evidence)
        cv_imwrite(base_path, repaired)
        cv_imwrite(
            overlay_path,
            renderer(bundle.brown, repaired, instance_mask),
        )

    def run_vascular(
        self,
        preprocess: PreprocessResultV2,
        bundle: SkinStyleBundle,
        output_dir: str | Path,
        source_name: str,
    ) -> AddedAlgorithmArtifacts:
        from src.added_algorithms.runner import run_vascular

        return run_vascular(
            preprocess,
            bundle.vascular_display,
            output_dir,
            source_name,
        )


def strategy_for_profile(profile: CaptureProfile) -> SkinStyleStrategy:
    match profile:
        case CaptureProfile.INSTITUTION:
            return InstitutionStyleStrategy()
        case CaptureProfile.CONSUMER:
            return ConsumerStyleStrategy()
        case unreachable:
            assert_never(unreachable)


__all__ = [
    "ConsumerStyleStrategy",
    "InstitutionStyleStrategy",
    "SkinStyleBundle",
    "SkinStyleStrategy",
    "SkinStyleProviderRequiredError",
    "strategy_for_profile",
]
