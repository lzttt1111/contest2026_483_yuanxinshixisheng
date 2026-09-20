"""Typed errors for the shared pure-data summary scoring entry."""

from __future__ import annotations


class SummaryScoringError(Exception):
    """Base class for summary scoring failures."""


class ScoringAssetError(SummaryScoringError):
    """Raised when a pinned scoring asset is missing or has drifted."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class ScoringInputError(SummaryScoringError):
    """Raised when the incoming evidence document is structurally invalid."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


__all__ = ["ScoringAssetError", "ScoringInputError", "SummaryScoringError"]
