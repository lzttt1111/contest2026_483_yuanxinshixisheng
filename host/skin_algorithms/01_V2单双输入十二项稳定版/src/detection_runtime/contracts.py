from __future__ import annotations

"""Small, shared contracts for the twelve-item runtime.

Detectors stay in route-specific providers.  This module only owns stable item
IDs, presentation order and strict route selection; it must not import image
algorithms or the public Worker.
"""

from dataclasses import dataclass
from enum import Enum
import re
from typing import Iterable


class RuntimeRoute(str, Enum):
    CONSUMER_RGB = "consumer_rgb"
    CLINIC_FOUR_LIGHT = "clinic_four_light"


@dataclass(frozen=True)
class DetectionItemDefinition:
    item_id: str
    directory_name: str
    display_name: str
    legacy_nine: bool


TWELVE_DETECTION_ITEMS = (
    DetectionItemDefinition("redness", "01_红区", "红区", True),
    DetectionItemDefinition("spots", "02_可见斑点", "可见斑点", True),
    DetectionItemDefinition("brown", "03_棕区", "棕区", True),
    DetectionItemDefinition("texture", "04_纹理", "纹理", True),
    DetectionItemDefinition("pores", "05_毛孔", "毛孔", True),
    DetectionItemDefinition("uv_spots", "06_UV色斑", "UV色斑", True),
    DetectionItemDefinition("porphyrin", "07_卟啉", "卟啉", True),
    DetectionItemDefinition("wrinkle", "08_皱纹", "皱纹", True),
    DetectionItemDefinition("acne", "09_痤疮", "痤疮", True),
    DetectionItemDefinition("surface_gloss", "10_油光", "油光", False),
    DetectionItemDefinition("vascular", "11_血管样结构", "血管样结构", False),
    DetectionItemDefinition("contour_firmness", "12_轮廓紧致度", "轮廓紧致度", False),
)

LEGACY_NINE_IDS = tuple(
    item.item_id for item in TWELVE_DETECTION_ITEMS if item.legacy_nine
)
CLINIC_REQUIRED_ROLES = frozenset({"RGB_M", "PP_M", "CP_M", "365_M"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def infer_runtime_route(
    roles: Iterable[str], *, signed_manifest_sha256: str | None
) -> RuntimeRoute:
    """Select a route from explicit authenticated roles, never filenames."""

    role_tuple = tuple(roles)
    if len(role_tuple) != len(set(role_tuple)):
        raise ValueError("INVALID_CAPTURE_SET: duplicate input role")
    role_set = frozenset(role_tuple)
    if role_set == {"RGB_M"}:
        if signed_manifest_sha256 is not None:
            raise ValueError("INVALID_CAPTURE_SET: consumer input forbids clinic signature")
        return RuntimeRoute.CONSUMER_RGB
    if role_set != CLINIC_REQUIRED_ROLES:
        raise ValueError(
            "INVALID_CAPTURE_SET: expected RGB_M or exact RGB_M/PP_M/CP_M/365_M"
        )
    if signed_manifest_sha256 is None or not _SHA256_RE.fullmatch(
        signed_manifest_sha256
    ):
        raise ValueError("INVALID_CAPTURE_SET: clinic input requires signed manifest SHA256")
    return RuntimeRoute.CLINIC_FOUR_LIGHT
