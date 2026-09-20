from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.detection_runtime.contracts import RuntimeRoute, TWELVE_DETECTION_ITEMS
from src.detection_runtime.instrument_style import (
    ClinicInstrumentStyleViews,
    InstrumentStylePair,
)
from src.detection_runtime.route_modality_provider import RouteModalityProvider
from src.detection_runtime.route_plan import CLINIC_ITEM_MODALITY_CONTRACT
from src.preprocess.image_preprocessor import PreprocessResultV2


EXPECTED_ROLE_MATRIX = {
    "redness": ("CP_M", "RGB_M", (), ("CP_M",), "red_brown_cp_pixels_rgb_geometry"),
    "spots": ("RGB_M", "RGB_M", (), (), None),
    "brown": ("CP_M", "RGB_M", (), ("CP_M",), "red_brown_cp_pixels_rgb_geometry"),
    "texture": ("RGB_M", "RGB_M", (), (), None),
    "pores": ("RGB_M", "RGB_M", (), (), None),
    "uv_spots": ("365_M", "365_M", (), (), None),
    "porphyrin": ("365_M", "365_M", (), (), None),
    "wrinkle": ("RGB_M", "RGB_M", (), (), None),
    "acne": ("RGB_M", "RGB_M", (), (), None),
    "surface_gloss": ("PP_M", "PP_M", (), (), None),
    "vascular": ("CP_M", "CP_M", ("RGB_M", "PP_M"), ("CP_M", "PP_M"), "vascular_cp_pixels_cp_geometry"),
    "contour_firmness": ("RGB_M", "RGB_M", (), (), None),
}


def _preprocess(value: int) -> PreprocessResultV2:
    image = np.full((1024, 1024, 3), value, dtype=np.uint8)
    skin_mask = np.full((1024, 1024), 255 if value == 10 else 0, dtype=np.uint8)
    result = PreprocessResultV2(
        analysis_image=image.copy(),
        display_image=image.copy(),
        skin_mask=skin_mask,
        landmarks=np.full((478, 2), value, dtype=np.float32),
        face_transform_matrix=np.array(
            [[1.0, 0.0, float(value)], [0.0, 1.0, float(value)]],
            dtype=np.float32,
        ),
        inverse_transform_matrix=np.eye(2, 3, dtype=np.float32),
        quality_score=90.0,
        quality_status="PASS",
        quality_flags=[],
    )
    result._debug_masks = {
        "hair_mask": np.full(
            (1024, 1024), 0 if value == 10 else 255, dtype=np.uint8
        ),
    }
    return result


class _Capture:
    capture_alias = "clinic-test"

    def __init__(
        self,
        paths: dict[str, Path],
        *,
        cp_status: str = "PASSED",
        pp_status: str = "PASSED",
        uv_status: str = "FAILED",
    ) -> None:
        statuses = {
            "RGB_M": "ANCHOR",
            "PP_M": pp_status,
            "CP_M": cp_status,
            "365_M": uv_status,
        }
        self._channels = {
            role: SimpleNamespace(
                path=path,
                registration_status=statuses[role],
                cross_channel_authorized=(
                    {"co_location": True, "fusion": True, "overlap": True}
                    if statuses[role] == "PASSED"
                    else {"co_location": False, "fusion": False, "overlap": False}
                ),
            )
            for role, path in paths.items()
        }

    def channel(self, role: str):
        return self._channels[role]


class _Preprocessor:
    def __init__(self) -> None:
        self.results: dict[int, PreprocessResultV2] = {}

    def preprocess_image(self, image: np.ndarray) -> PreprocessResultV2:
        value = int(image[0, 0, 0])
        result = _preprocess(value)
        self.results[value] = result
        return result


class _StyleProvider:
    def __init__(self) -> None:
        self.anchor: PreprocessResultV2 | None = None
        self.pixels: np.ndarray | None = None

    @staticmethod
    def _pair(anchor, source_image, output_root, source_role, consumer):
        return InstrumentStylePair(
            red=anchor,
            brown=anchor,
            redne_image=source_image.copy(),
            red_base_path=output_root / consumer / "RED_M.png",
            brown_base_path=output_root / consumer / "BROWN_M.png",
            redne_base_path=output_root / consumer / "REDNE_M.png",
            source_role=source_role,
            renderer_version="test",
            provenance={
                "source_role": source_role,
                "consumer": consumer,
                "pixel_source_role": "CP_M",
                "geometry_mask_source_role": "RGB_M" if consumer == "redness_brown" else "CP_M",
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
        self.anchor = red_brown_anchor
        self.pixels = source_image
        return ClinicInstrumentStyleViews(
            red_brown=self._pair(
                red_brown_anchor, source_image, output_root, source_role, "redness_brown"
            ),
            vascular=self._pair(
                vascular_anchor,
                source_image,
                output_root,
                source_role,
                "vascular_auxiliary_red",
            ),
        )


def _capture(tmp_path: Path, **kwargs: str) -> _Capture:
    paths: dict[str, Path] = {}
    for role, value in (("RGB_M", 10), ("PP_M", 20), ("CP_M", 30), ("365_M", 40)):
        path = tmp_path / f"{role}.png"
        assert cv2.imwrite(str(path), np.full((8, 8, 3), value, dtype=np.uint8))
        paths[role] = path
    return _Capture(paths, **kwargs)


def test_clinic_role_matrix_covers_twelve_without_fallback() -> None:
    assert tuple(item.item_id for item in CLINIC_ITEM_MODALITY_CONTRACT) == tuple(
        item.item_id for item in TWELVE_DETECTION_ITEMS
    )
    assert {
        item.item_id: (
            item.pixel_source_role,
            item.geometry_mask_anchor_role,
            item.auxiliary_pixel_source_roles,
            item.registration_required_roles,
            item.style_view_id,
        )
        for item in CLINIC_ITEM_MODALITY_CONTRACT
    } == EXPECTED_ROLE_MATRIX
    assert all(not item.fallback_allowed for item in CLINIC_ITEM_MODALITY_CONTRACT)


def test_clinic_red_brown_use_cp_pixels_with_rgb_geometry_and_masks(
    tmp_path: Path,
) -> None:
    preprocessor = _Preprocessor()
    style = _StyleProvider()
    bundle = RouteModalityProvider(style_provider=style).prepare_clinic(
        _capture(tmp_path), tmp_path / "output", preprocessor=preprocessor
    )

    rgb = preprocessor.results[10]
    cp = preprocessor.results[30]
    assert bundle.route is RuntimeRoute.CLINIC_FOUR_LIGHT
    assert style.anchor is rgb
    assert style.pixels is bundle.decoded["CP_M"]
    assert int(style.pixels[0, 0, 0]) == 30
    assert bundle.styled.source_role == "CP_M"
    assert bundle.styled.red.skin_mask is rgb.skin_mask
    assert bundle.styled.brown.face_transform_matrix is rgb.face_transform_matrix
    assert bundle.styled.red._debug_masks["hair_mask"] is rgb._debug_masks["hair_mask"]
    assert np.count_nonzero(bundle.styled.red.skin_mask) > 0
    assert np.count_nonzero(cp.skin_mask) == 0
    assert np.count_nonzero(cp._debug_masks["hair_mask"]) > 0
    assert bundle.styled.provenance["pixel_source_role"] == "CP_M"
    assert bundle.styled.provenance["geometry_mask_source_role"] == "RGB_M"
    vascular = bundle.vascular_styled
    assert vascular is not None
    assert vascular is not bundle.styled
    assert vascular.red is cp
    assert vascular.red.skin_mask is cp.skin_mask
    assert vascular.red._debug_masks["hair_mask"] is cp._debug_masks["hair_mask"]
    assert vascular.provenance["consumer"] == "vascular_auxiliary_red"
    assert vascular.provenance["geometry_mask_source_role"] == "CP_M"
    vascular_bytes = vascular.red.analysis_image.tobytes()
    bundle.styled.red.analysis_image.fill(99)
    bundle.styled.red.skin_mask.fill(0)
    assert vascular.red.analysis_image.tobytes() == vascular_bytes
    assert np.count_nonzero(vascular.red.skin_mask) == 0


@pytest.mark.parametrize("role", ("CP_M", "PP_M"))
def test_clinic_required_registration_fails_closed_before_preprocessing(
    tmp_path: Path, role: str
) -> None:
    preprocessor = _Preprocessor()
    style = _StyleProvider()
    kwargs = {"cp_status": "FAILED"} if role == "CP_M" else {"pp_status": "FAILED"}
    with pytest.raises(ValueError, match=f"{role} registration"):
        RouteModalityProvider(style_provider=style).prepare_clinic(
            _capture(tmp_path, **kwargs), tmp_path / "output", preprocessor=preprocessor
        )
    assert preprocessor.results == {}
    assert style.anchor is None


def test_failed_365_registration_remains_independent_without_fallback(
    tmp_path: Path,
) -> None:
    preprocessor = _Preprocessor()
    style = _StyleProvider()
    bundle = RouteModalityProvider(style_provider=style).prepare_clinic(
        _capture(tmp_path, uv_status="FAILED"),
        tmp_path / "output",
        preprocessor=preprocessor,
    )

    assert bundle.preprocessed["365_M"] is preprocessor.results[40]
    uv_contracts = {
        item.item_id: item
        for item in CLINIC_ITEM_MODALITY_CONTRACT
        if item.item_id in {"uv_spots", "porphyrin"}
    }
    assert all(item.pixel_source_role == "365_M" for item in uv_contracts.values())
    assert all(item.geometry_mask_anchor_role == "365_M" for item in uv_contracts.values())
    assert all(not item.registration_required_roles for item in uv_contracts.values())
    assert all(not item.fallback_allowed for item in uv_contracts.values())
