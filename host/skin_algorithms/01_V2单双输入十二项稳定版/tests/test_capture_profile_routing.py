from __future__ import annotations

import pytest

from run_cloud_worker import build_command, resolve_launch
from src import capture_profile as capture_profile_module
from src import pipeline as pipeline_module
from src.engines.vendor_skin_overlay import (
    render_consumer_brown_result,
    render_vendor_brown_result,
)


CONSUMER_TARGETS = {
    "consumer_redness": ("src.worker:celery_app", "dermavision.analyze_image"),
    "consumer_spots": ("src.worker:celery_app", "dermavision.analyze_image"),
    "consumer_brown": ("src.worker:celery_app", "dermavision.analyze_image"),
    "consumer_texture": ("src.worker:celery_app", "dermavision.analyze_image"),
    "consumer_pores": ("src.worker:celery_app", "dermavision.analyze_image"),
    "consumer_purple": ("src.worker:celery_app", "dermavision.analyze_image"),
    "consumer_acne_v2": (
        "src.acne.worker:celery_app",
        "dermavision.analyze_image",
    ),
    "consumer_wrinkle": ("src.wrinkle.worker:celery_app", "wrinkle.analyze_image"),
    "consumer_surface_gloss": (
        "src.worker:celery_app",
        "dermavision.analyze_image",
    ),
    "consumer_vascular": ("src.worker:celery_app", "dermavision.analyze_image"),
    "consumer_contour_firmness": (
        "src.worker:celery_app",
        "dermavision.analyze_image",
    ),
}


@pytest.mark.parametrize(("target", "expected"), CONSUMER_TARGETS.items())
def test_consumer_targets_preserve_task_and_use_explicit_queue(
    target: str,
    expected: tuple[str, str],
) -> None:
    launch = resolve_launch(target)

    assert launch.capture_profile.value == "consumer"
    assert launch.app == expected[0]
    assert launch.task == expected[1]
    assert launch.queues == (target,)
    assert f"--queues={target}" in build_command(launch, "info", "prefork")


def test_existing_target_remains_institution_profile() -> None:
    launch = resolve_launch("redness")

    assert launch.capture_profile.value == "institution"
    assert launch.queues == ("redness",)
    assert launch.task == "dermavision.analyze_image"


def test_consumer_brown_uses_high_contrast_renderer_only() -> None:
    resolver = getattr(pipeline_module, "brown_renderer_for_profile", None)

    assert callable(resolver)
    assert resolver(capture_profile_module.CaptureProfile.CONSUMER) is (
        render_consumer_brown_result
    )
    assert resolver(capture_profile_module.CaptureProfile.INSTITUTION) is (
        render_vendor_brown_result
    )


def test_consumer_acne_v1_is_not_a_supported_profile() -> None:
    with pytest.raises(ValueError, match="未知 Worker target"):
        resolve_launch("consumer_acne")


def test_profile_environment_defaults_and_rejects_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = getattr(capture_profile_module, "capture_profile_from_environment", None)
    assert callable(loader), "环境Profile解析器尚未实现"
    monkeypatch.delenv("DERMAVISION_CAPTURE_PROFILE", raising=False)
    assert loader().value == "institution"
    monkeypatch.setenv("DERMAVISION_CAPTURE_PROFILE", "consumer")
    assert loader().value == "consumer"
    monkeypatch.setenv("DERMAVISION_CAPTURE_PROFILE", "auto")
    with pytest.raises(ValueError, match="DERMAVISION_CAPTURE_PROFILE"):
        loader()


def test_consumer_execution_mode_is_machine_readable() -> None:
    resolver = getattr(capture_profile_module, "execution_mode_for_profile", None)
    assert callable(resolver), "Profile执行模式解析器尚未实现"
    assert resolver(capture_profile_module.CaptureProfile.INSTITUTION, "formal_fast") == (
        "formal_fast"
    )
    assert resolver(capture_profile_module.CaptureProfile.CONSUMER, "formal_fast") == (
        "single_algorithm_consumer"
    )


def test_consumer_queues_are_declared_by_each_celery_app() -> None:
    from src.worker import celery_app as dermavision_app
    from src.acne.worker import celery_app as acne_app
    from src.wrinkle.worker import celery_app as wrinkle_app

    dermavision_queues = set(dermavision_app.conf.task_queues)
    acne_queues = set(acne_app.conf.task_queues)
    wrinkle_queues = set(wrinkle_app.conf.task_queues)

    assert {
        target for target in CONSUMER_TARGETS
        if target not in {"consumer_acne_v2", "consumer_wrinkle"}
    }.issubset(dermavision_queues)
    assert "consumer_acne_v2" in acne_queues
    assert "consumer_wrinkle" in wrinkle_queues
