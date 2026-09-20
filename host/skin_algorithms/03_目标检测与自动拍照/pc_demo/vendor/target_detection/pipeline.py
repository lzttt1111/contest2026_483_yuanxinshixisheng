"""Stable algorithm boundary inspired by the internal worker template."""

from .runtime import V5Runtime


class Pipeline(V5Runtime):
    """Current device-detection and contact-point inference pipeline."""
