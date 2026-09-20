from __future__ import annotations

"""Institution RED/BROWN base adapter using the approved runtime sidecar."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.engines.vendor_skin_style import VendorSkinStylePair, VendorSkinStyleProvider
from src.preprocess.image_preprocessor import PreprocessResultV2


@dataclass(frozen=True)
class InstrumentStylePair:
    red: PreprocessResultV2
    brown: PreprocessResultV2
    redne_image: np.ndarray
    red_base_path: Path
    brown_base_path: Path
    redne_base_path: Path
    source_role: str
    renderer_version: str
    provenance: dict[str, Any]


@dataclass(frozen=True)
class ClinicInstrumentStyleViews:
    red_brown: InstrumentStylePair
    vascular: InstrumentStylePair

    def __post_init__(self) -> None:
        if (
            self.red_brown is self.vascular
            or self.red_brown.red is self.vascular.red
            or self.red_brown.brown is self.vascular.brown
            or self.red_brown.provenance is self.vascular.provenance
        ):
            raise ValueError("clinic Red/Brown and vascular style views must not alias")


class InstrumentStyleRedBrownBaseProvider:
    """Preserve the validated four-light style contract without a second loader."""

    version = "vendor-core.skin_generate.Enhence_images-front-only-v1"

    def __init__(self, modules_root: Path | None = None) -> None:
        if modules_root is not None:
            raise ValueError("vendor module root must be supplied by the approved environment")
        self._provider = VendorSkinStyleProvider()

    def build(
        self,
        preprocess_result: PreprocessResultV2,
        *,
        source_image: np.ndarray,
        output_root: Path,
        source_role: str,
    ) -> InstrumentStylePair:
        pair = self._provider.build(
            preprocess_result,
            source_image=source_image,
            output_root=output_root,
            source_role=source_role,
        )
        return InstrumentStylePair(
            red=pair.red,
            brown=pair.brown,
            redne_image=pair.redne_image,
            red_base_path=pair.red_base_path,
            brown_base_path=pair.brown_base_path,
            redne_base_path=pair.redne_base_path,
            source_role=pair.source_role,
            renderer_version=self.version,
            provenance=dict(pair.provenance),
        )

    @staticmethod
    def _with_view_provenance(
        pair: InstrumentStylePair,
        *,
        consumer: str,
        geometry_mask_source_role: str,
    ) -> InstrumentStylePair:
        return InstrumentStylePair(
            red=pair.red,
            brown=pair.brown,
            redne_image=pair.redne_image,
            red_base_path=pair.red_base_path,
            brown_base_path=pair.brown_base_path,
            redne_base_path=pair.redne_base_path,
            source_role=pair.source_role,
            renderer_version=pair.renderer_version,
            provenance={
                **pair.provenance,
                "consumer": consumer,
                "pixel_source_role": "CP_M",
                "geometry_mask_source_role": geometry_mask_source_role,
                "fallback_allowed": False,
                "object_aliasing_allowed": False,
            },
        )

    def build_clinic_views(
        self,
        *,
        red_brown_anchor: PreprocessResultV2,
        vascular_anchor: PreprocessResultV2,
        source_image: np.ndarray,
        output_root: Path,
        source_role: str,
    ) -> ClinicInstrumentStyleViews:
        if source_role != "CP_M":
            raise ValueError("clinic vendor style pixels must come from CP_M")
        vascular = self.build(
            vascular_anchor,
            source_image=source_image,
            output_root=output_root,
            source_role=source_role,
        )
        red_brown_vendor = self._provider.reanchor(
            self._provider_pair(vascular),
            red_brown_anchor,
            aligned_root=output_root / "aligned_red_brown_rgb_geometry",
        )
        red_brown = InstrumentStylePair(
            red=red_brown_vendor.red,
            brown=red_brown_vendor.brown,
            redne_image=red_brown_vendor.redne_image,
            red_base_path=red_brown_vendor.red_base_path,
            brown_base_path=red_brown_vendor.brown_base_path,
            redne_base_path=red_brown_vendor.redne_base_path,
            source_role=red_brown_vendor.source_role,
            renderer_version=self.version,
            provenance=dict(red_brown_vendor.provenance),
        )
        return ClinicInstrumentStyleViews(
            red_brown=self._with_view_provenance(
                red_brown,
                consumer="redness_brown",
                geometry_mask_source_role="RGB_M",
            ),
            vascular=self._with_view_provenance(
                vascular,
                consumer="vascular_auxiliary_red",
                geometry_mask_source_role="CP_M",
            ),
        )

    @staticmethod
    def _provider_pair(pair: InstrumentStylePair) -> VendorSkinStylePair:
        return VendorSkinStylePair(
            red=pair.red,
            brown=pair.brown,
            redne_image=pair.redne_image,
            red_base_path=pair.red_base_path,
            brown_base_path=pair.brown_base_path,
            redne_base_path=pair.redne_base_path,
            source_role=pair.source_role,
            provenance=dict(pair.provenance),
        )


__all__ = [
    "ClinicInstrumentStyleViews",
    "InstrumentStylePair",
    "InstrumentStyleRedBrownBaseProvider",
]
