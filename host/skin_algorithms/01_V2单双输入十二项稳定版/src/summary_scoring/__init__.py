"""Shared pure-data scoring entry for the twelve-item overall summary.

``score_report_from_evidence`` merges per-algorithm ``scoring_input.evidence``
into a ``detector_results`` subset and produces two independent score systems
(``word_display`` and ``production_proxy_v1``) in fixed module order. It never
touches GPU, models, DOCX, network or display layout.
"""

from __future__ import annotations

from .assets import Assets, load_default_assets
from .entry import DISPLAY_POLICY_VERSION, score_report_from_evidence
from .errors import ScoringAssetError, ScoringInputError, SummaryScoringError

__all__ = [
    "Assets",
    "DISPLAY_POLICY_VERSION",
    "ScoringAssetError",
    "ScoringInputError",
    "SummaryScoringError",
    "load_default_assets",
    "score_report_from_evidence",
]
