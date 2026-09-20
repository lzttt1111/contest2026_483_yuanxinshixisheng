from __future__ import annotations

import importlib

import numpy as np


def test_brown_tone_policies_have_ordered_lift_and_contrast() -> None:
    module = importlib.import_module("src.consumer_pigment.brown_tone")
    preset_type = getattr(module, "ConsumerBrownTonePreset", None)
    resolver = getattr(module, "policy_for_brown_tone", None)
    assert preset_type is not None and callable(resolver)

    policies = [resolver(preset) for preset in preset_type]

    assert [item.midtone_lift for item in policies] == [6.0, 10.0, 14.0]
    assert [item.local_contrast_gain for item in policies] == [0.05, 0.10, 0.18]


def test_brown_tone_lifts_midtones_but_protects_highlight() -> None:
    module = importlib.import_module("src.consumer_pigment.brown_tone")
    preset_type = getattr(module, "ConsumerBrownTonePreset", None)
    input_type = getattr(module, "BrownToneInput", None)
    renderer = getattr(module, "render_brown_tone", None)
    assert preset_type is not None and input_type is not None and callable(renderer)
    base = np.full((96, 96, 3), (80, 115, 175), dtype=np.uint8)
    base[36:60, 36:60] = (165, 190, 220)
    foreground = np.full((96, 96), 255, dtype=np.uint8)
    protected = np.zeros((96, 96), dtype=np.float32)
    protected[36:60, 36:60] = 1.0
    original = base.copy()
    render_input = input_type(base, foreground, protected)

    outputs = [renderer(render_input, preset) for preset in preset_type]

    midtone_means = [float(np.mean(item[12:28, 12:28])) for item in outputs]
    assert midtone_means[0] < midtone_means[1] < midtone_means[2]
    for output in outputs:
        np.testing.assert_allclose(output[40:56, 40:56], base[40:56, 40:56], atol=1)
    np.testing.assert_array_equal(base, original)


def test_brown_tone_preserves_background_pixels() -> None:
    module = importlib.import_module("src.consumer_pigment.brown_tone")
    preset_type = getattr(module, "ConsumerBrownTonePreset", None)
    input_type = getattr(module, "BrownToneInput", None)
    renderer = getattr(module, "render_brown_tone", None)
    assert preset_type is not None and input_type is not None and callable(renderer)
    base = np.full((64, 64, 3), 255, dtype=np.uint8)
    base[16:48, 16:48] = (75, 105, 165)
    foreground = np.zeros((64, 64), dtype=np.uint8)
    foreground[16:48, 16:48] = 255
    protected = np.zeros((64, 64), dtype=np.float32)

    output = renderer(
        input_type(base, foreground, protected),
        preset_type.BRIGHT,
    )

    np.testing.assert_array_equal(output[:12, :12], base[:12, :12])
    assert float(np.mean(output[20:44, 20:44])) > float(
        np.mean(base[20:44, 20:44])
    )
