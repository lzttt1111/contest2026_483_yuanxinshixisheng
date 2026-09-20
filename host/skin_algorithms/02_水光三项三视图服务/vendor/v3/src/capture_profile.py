from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Final

from typing_extensions import assert_never


class CaptureProfile(str, Enum):
    """Explicit product route; never inferred from image pixels."""

    INSTITUTION = "institution"
    CONSUMER = "consumer"


@dataclass(frozen=True, slots=True)
class CaptureProfileEnvironmentError(ValueError):
    raw_value: str

    def __str__(self) -> str:
        return (
            "DERMAVISION_CAPTURE_PROFILE必须为institution或consumer，"
            f"实际为{self.raw_value!r}"
        )


CONSUMER_TARGET_PREFIX: Final = "consumer_"
CONSUMER_BASE_TARGETS: Final = frozenset({
    "redness",
    "spots",
    "brown",
    "texture",
    "pores",
    "purple",
    "acne_v2",
    "wrinkle",
    "surface_gloss",
    "vascular",
    "contour_firmness",
})


def capture_profile_from_environment() -> CaptureProfile:
    raw_value = os.environ.get(
        "DERMAVISION_CAPTURE_PROFILE",
        CaptureProfile.INSTITUTION.value,
    ).strip()
    try:
        return CaptureProfile(raw_value)
    except ValueError as exc:
        raise CaptureProfileEnvironmentError(raw_value) from exc


def execution_mode_for_profile(
    profile: CaptureProfile,
    institution_mode: str,
) -> str:
    match profile:
        case CaptureProfile.INSTITUTION:
            return institution_mode
        case CaptureProfile.CONSUMER:
            return "single_algorithm_consumer"
        case unreachable:
            assert_never(unreachable)


__all__ = [
    "CONSUMER_BASE_TARGETS",
    "CONSUMER_TARGET_PREFIX",
    "CaptureProfile",
    "CaptureProfileEnvironmentError",
    "capture_profile_from_environment",
    "execution_mode_for_profile",
]
