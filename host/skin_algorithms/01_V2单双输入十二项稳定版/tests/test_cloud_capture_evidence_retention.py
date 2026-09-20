import json
from types import SimpleNamespace

from src.detection_runtime import cloud_task as module


def test_prepared_capture_evidence_survives_without_photos_or_locators(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "__file__", str(tmp_path / "src/detection_runtime/cloud_task.py"))
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    data = {"input_sha256": {"RGB_M": "a" * 64, "365_M": "b" * 64},
            "registration": {"365_M": {"status": "FAILED"}}}
    (temporary / "capture_evidence.json").write_text(json.dumps(data))
    channel = SimpleNamespace(role="365_M", cross_channel_authorized={"fusion": False})
    context = SimpleNamespace(root=temporary, capture=SimpleNamespace(channels=(channel,)))
    module._retain_preparation(context, "case/../09", "dermavision")
    files = list((tmp_path / "output/cloud_capture_evidence").glob("*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert saved["state"] == "prepared"
    assert saved["report_id"] == "case/../09"
    assert saved["capture_digest"] == module.canonical_sha(data["input_sha256"])
    assert saved["cross_channel_authorized"]["365_M"]["fusion"] is False
    assert "oss_key" not in saved and "images" not in saved


def test_single_rgb_retention_does_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "__file__", str(tmp_path / "src/detection_runtime/cloud_task.py"))
    module._retain_preparation(SimpleNamespace(capture=None), "case", "dermavision")
    assert not (tmp_path / "output").exists()
