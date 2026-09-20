from pathlib import Path

from src.detection_runtime.capture_manifest import load_clinic_capture_manifest


FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "仪器拍照/Clinic-Fast3/fixture_manifest.json"
)


def test_fixed_manifest_exposes_three_exact_four_light_captures() -> None:
    captures = load_clinic_capture_manifest(FIXTURE)
    assert tuple(item.capture_alias for item in captures) == (
        "clinic28-25", "clinic28-09", "clinic28-23",
    )
    assert all(
        tuple(channel.role for channel in item.channels)
        == ("RGB_M", "PP_M", "CP_M", "365_M")
        for item in captures
    )
    assert captures[0].channel("365_M").registration_status == "PASSED"
    assert captures[1].channel("365_M").registration_status == "FAILED"
    assert captures[2].channel("365_M").registration_status == "FAILED"
