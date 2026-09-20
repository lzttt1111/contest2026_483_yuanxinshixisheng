"""V0.1.2 shadow scoring extensions for the first five report modules."""

from .explanation import attach_score_explanations, build_score_explanation
from .pigmentation import score_combined_pigmentation_shadow
from .profile import build_uv_shadow_profile

__all__ = [
    "attach_score_explanations",
    "build_score_explanation",
    "build_uv_shadow_profile",
    "score_combined_pigmentation_shadow",
]
