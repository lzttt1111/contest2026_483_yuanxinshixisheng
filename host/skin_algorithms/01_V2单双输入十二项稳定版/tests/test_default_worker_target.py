from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_ROOT = PROJECT_ROOT / "deploy"


def _unit_lines(name: str) -> set[str]:
    return {
        line.strip()
        for line in (DEPLOY_ROOT / name).read_text(encoding="utf-8").splitlines()
        if line.startswith(("Requires=", "After="))
    }


def test_default_target_selects_acne_v2_and_v1_is_explicit_rollback() -> None:
    default_lines = _unit_lines("dermavision-workers.target")
    rollback_lines = _unit_lines("dermavision-workers-v1.target")

    assert "Requires=dermavision@acne_v2.service" in default_lines
    assert "Requires=dermavision@acne.service" not in default_lines
    assert "Requires=dermavision@acne.service" in rollback_lines
    assert "Requires=dermavision@acne_v2.service" not in rollback_lines
    # The independent CPU summary app is deployed with the default V2 set only
    # (v1 rollback and consumer targets intentionally exclude it this phase).
    assert "Requires=dermavision@summary.service" in default_lines
    assert "After=dermavision@summary.service" in default_lines
    assert "Requires=dermavision@summary.service" not in rollback_lines
    assert len(default_lines) == 24
    assert len(rollback_lines) == 22
    assert not (DEPLOY_ROOT / "dermavision-workers-v2.target").exists()


def test_consumer_target_has_eleven_profile_workers_and_is_mutually_exclusive() -> None:
    consumer_path = DEPLOY_ROOT / "dermavision-consumer-workers.target"
    text = consumer_path.read_text(encoding="utf-8")
    consumer_lines = _unit_lines(consumer_path.name)
    expected = {
        "consumer_redness",
        "consumer_spots",
        "consumer_brown",
        "consumer_texture",
        "consumer_pores",
        "consumer_purple",
        "consumer_acne_v2",
        "consumer_wrinkle",
        "consumer_surface_gloss",
        "consumer_vascular",
        "consumer_contour_firmness",
    }

    assert {
        line.removeprefix("Requires=dermavision@").removesuffix(".service")
        for line in consumer_lines
        if line.startswith("Requires=")
    } == expected
    assert len(consumer_lines) == 22
    assert "Conflicts=dermavision-workers.target" in text
    assert "Conflicts=dermavision-workers-v1.target" in text
    assert "Conflicts=dermavision-consumer-workers.target" in (
        DEPLOY_ROOT / "dermavision-workers.target"
    ).read_text(encoding="utf-8")
    assert "Conflicts=dermavision-workers-v1.target" in (
        DEPLOY_ROOT / "dermavision-workers.target"
    ).read_text(encoding="utf-8")
    assert "Conflicts=dermavision-workers.target" in (
        DEPLOY_ROOT / "dermavision-workers-v1.target"
    ).read_text(encoding="utf-8")


def test_root_deployment_guide_installs_consumer_target_and_review_script_is_portable() -> None:
    guide = (PROJECT_ROOT / "V2单双输入十二项技术与部署说明_20260915.md").read_text(
        encoding="utf-8"
    )
    review_script = (
        PROJECT_ROOT / "scripts/review/render_consumer_brown_display_variants.py"
    ).read_text(encoding="utf-8")

    assert "sudo cp deploy/dermavision-consumer-workers.target" in guide
    assert "/home/" not in review_script
