"""Profile-specific Brown detector thresholds without branch duplication."""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing_extensions import assert_never
from sg_core.capture_profile import CaptureProfile
from sg_core.consumer_pigment.formal_markers import ConsumerBrownRecallPreset

@dataclass(frozen=True, slots=True)
class BrownDetectionPolicy:
    scale_z_threshold: float
    minimum_scale_votes: int
    candidate_z_threshold: float
    minimum_raw_response: float
    minimum_area_px: int
    minimum_mean_score: float
    minimum_solidity: float
    maximum_aspect_ratio: float
    peak_min_distance: int

def policy_for_brown_profile(profile: CaptureProfile) -> BrownDetectionPolicy:
    return BrownDetectionPolicy(1.25, 4, 1.4, 0.1, 4, 0.16, 0.1, 5.0, 6)

def policy_for_consumer_brown_detection_recall(preset: ConsumerBrownRecallPreset) -> BrownDetectionPolicy:
    base = policy_for_brown_profile(CaptureProfile.CONSUMER)
    values = {ConsumerBrownRecallPreset.CURRENT: (1.25, 4, 1.4, 0.1), ConsumerBrownRecallPreset.RECALL_1: (1.15, 4, 1.22, 0.085), ConsumerBrownRecallPreset.RECALL_2: (1.15, 3, 1.1, 0.075), ConsumerBrownRecallPreset.RECALL_3: (1.0, 3, 0.95, 0.06), ConsumerBrownRecallPreset.RECALL_4: (0.9, 3, 0.8, 0.045)}[preset]
    return replace(base, scale_z_threshold=values[0], minimum_scale_votes=values[1], candidate_z_threshold=values[2], minimum_raw_response=values[3])
__all__ = ['BrownDetectionPolicy', 'policy_for_brown_profile', 'policy_for_consumer_brown_detection_recall']
