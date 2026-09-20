from __future__ import annotations

"""Frozen porphyrin detector for the single-RGB purple proxy route."""

from src.engines.fluorescence_porphyrin_engine import (
    FluorescencePorphyrinAnalyzer,
    FluorescencePorphyrinConfig,
)


RGB_PROXY_PORPHYRIN_CONFIG = FluorescencePorphyrinConfig(
    scale_z_threshold=1.25,
    safe_mask_erode_kernel=9,
    peak_threshold=0.78,
    peak_min_distance=6,
    maximum_features=1200,
    display_min_radius_px=2,
    display_max_radius_px=4,
)


class RGBProxyFluorescencePorphyrinAnalyzer(FluorescencePorphyrinAnalyzer):
    """Explicitly identify the synthetic fluorescence input as an RGB proxy."""

    route = "consumer_rgb"
    input_semantics = "RGB_DERIVED_FLUORESCENCE_PROXY"
    algorithm_version = "RGBProxyPorphyrin-Frozen-V1"

    def __init__(
        self,
        config: FluorescencePorphyrinConfig | None = None,
    ) -> None:
        super().__init__(config or RGB_PROXY_PORPHYRIN_CONFIG)


__all__ = [
    "RGB_PROXY_PORPHYRIN_CONFIG",
    "RGBProxyFluorescencePorphyrinAnalyzer",
]
