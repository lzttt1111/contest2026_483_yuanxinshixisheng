from __future__ import annotations

from src.engines.purple_analysis_engine import PurpleAnalysisEngine


def test_purple_uses_explicit_single_rgb_proxy_detector() -> None:
    from src.engines.rgb_proxy_fluorescence_porphyrin_engine import (
        RGBProxyFluorescencePorphyrinAnalyzer,
    )

    engine = PurpleAnalysisEngine()
    detector = engine.fluorescence_porphyrin_detector

    assert isinstance(detector, RGBProxyFluorescencePorphyrinAnalyzer)
    assert detector.route == "consumer_rgb"
    assert detector.input_semantics == "RGB_DERIVED_FLUORESCENCE_PROXY"
    assert detector.algorithm_version == "RGBProxyPorphyrin-Frozen-V1"
