import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.capture_profile import CaptureProfile, capture_profile_from_environment, use_capture_profile
from src.detection_runtime.cloud_input import CaptureInputError, parse_capture_images
from src.detection_runtime.cloud_registration import assess_registration


def images(tokens=("RGB", "PP", "CP", "UV")):
    return [{"filename": f"case_{token}_M.jpg", "oss_key": f"input/{token}.jpg"} for token in tokens]


def test_role_order_and_uv_alias():
    assert [r.role for r in parse_capture_images(images(("cp", "365", "rgb", "pp")))] == ["RGB_M", "PP_M", "CP_M", "365_M"]


@pytest.mark.parametrize("rows", [[], images()[:2], images()[:3], images()+images()[:1],
    images(("RGB","PP","CP","CP")),
    [{**r, "filename": "case_RGB_CP.jpg"} if i == 0 else r for i,r in enumerate(images())],
    [{**r, "filename": "other_PP_M.jpg"} if i == 1 else r for i,r in enumerate(images())]])
def test_bad_capture_lists(rows):
    with pytest.raises(CaptureInputError):
        parse_capture_images(rows)


def test_single_unlabelled_image_and_source_validation():
    assert parse_capture_images([{"filename":"phone.jpg","oss_key":"phone.jpg"}])[0].role == "RGB_M"
    for extra in ({"signed_picture_url":"https://example.test/a"}, {"filename":"../RGB.jpg"}, {"oss_key":"../a.jpg"}):
        with pytest.raises(CaptureInputError):
            parse_capture_images([{**images()[0], **extra}])


def test_profile_is_request_scoped(monkeypatch):
    monkeypatch.setenv("DERMAVISION_CAPTURE_PROFILE", "consumer")
    assert capture_profile_from_environment() == CaptureProfile.CONSUMER
    with use_capture_profile(CaptureProfile.INSTITUTION):
        assert capture_profile_from_environment() == CaptureProfile.INSTITUTION
        with use_capture_profile(CaptureProfile.CONSUMER):
            assert capture_profile_from_environment() == CaptureProfile.CONSUMER
        assert capture_profile_from_environment() == CaptureProfile.INSTITUTION
    assert capture_profile_from_environment() == CaptureProfile.CONSUMER


def test_registration_same_and_incompatible_geometry():
    p = np.random.default_rng(3).uniform(.2,.8,(478,2)).astype(np.float32)
    good = assess_registration("CP_M", p, p, (100,100),(100,100))
    assert good["status"] == "PASSED" and good["transform_sha256"]
    assert assess_registration("CP_M",p,p,(100,200),(100,100))["status"] == "FAILED"
    assert assess_registration("CP_M",p*2,p,(100,100),(100,100))["status"] == "FAILED"


def test_worker_signatures_and_legacy_call(monkeypatch):
    from src import worker
    assert "capture_images" in inspect.signature(worker.analyze_image.run).parameters
    assert not hasattr(worker, "analyze_image_v2")
    from src.acne import worker as acne
    from src.wrinkle import worker as wrinkle
    assert "capture_images" in inspect.signature(acne.analyze_image_v2.run).parameters
    assert "capture_images" in inspect.signature(wrinkle.analyze_image.run).parameters
    result = worker.analyze_image.run("test", "", None, ["redness"], capture_images=images()[:2])
    assert result["status"] == "failed" and result["error_code"] == "invalid_capture_count"


def test_target_provider_does_not_run_unselected_heads(tmp_path, monkeypatch):
    import src.detection_runtime.clinic_provider as module
    class Unexpected:
        def __init__(self, *args, **kwargs):
            raise AssertionError("unselected detector was constructed")
    for key in ("ErythemaAnalyzer","BrownAreaAnalyzer","SurfaceGlossAnalyzer"):
        monkeypatch.setattr(module, key, Unexpected)
    monkeypatch.setattr(module,"uv_results",lambda *_: (_ for _ in ()).throw(AssertionError("UV ran")))
    fake = SimpleNamespace(source_role="CP_M", renderer_version="test", provenance={}, red=object())
    other = SimpleNamespace(source_role="CP_M",provenance={},red=object())
    bundle = SimpleNamespace(capture=SimpleNamespace(capture_alias="sample",channels=[]),
                             timing_seconds={},preprocessed=dict.fromkeys(("RGB_M","PP_M","CP_M","365_M"),object()),
                             styled=fake,vascular_styled=other,started_at=0)
    paths={key: tmp_path/key for key in ("overlay","public_json","public_csv","medical_v2_csv")}
    monkeypatch.setattr(module,"run_vascular_v3",lambda *_: SimpleNamespace(**paths))
    fake.red=SimpleNamespace(analysis_image=np.zeros((2,2,3)))
    other.red=SimpleNamespace(analysis_image=np.zeros((2,2,3)))
    provider = object.__new__(module.ClinicFourLightProvider)
    produced, _ = provider.run_prepared(bundle,tmp_path,algorithms=("vascular",))
    assert set(produced) == {"vascular"}


def test_four_download_registration_uv_failure_allowed(tmp_path, monkeypatch):
    import cv2
    from src.detection_runtime import cloud_task as module
    frame = np.zeros((32,32,3), np.uint8)
    data = cv2.imencode(".jpg",frame)[1].tobytes()
    points = np.ones((478,2),np.float32)
    monkeypatch.setattr(module._REGISTRATION,"landmarks",lambda _:points)
    monkeypatch.setattr(module,"rgb_geometry_valid",lambda *_:True)
    monkeypatch.setattr(module,"assess_registration",lambda role,*_:dict(
        status="FAILED" if role=="365_M" else "PASSED",transform_sha256=None,reason="test"))
    ctx=module.CaptureTaskContext(tmp_path,parse_capture_images(images()))
    module._download_images(ctx,lambda _:data,lambda _:data,1024*1024)
    assert ctx.capture.channel("365_M").registration_status=="FAILED"
    assert not any(ctx.capture.channel("365_M").cross_channel_authorized.values())
    assert all(ctx.capture.channel("CP_M").cross_channel_authorized.values())


def test_bad_download_fails_before_landmarks(tmp_path,monkeypatch):
    from src.detection_runtime import cloud_task as module
    monkeypatch.setattr(module._REGISTRATION,"landmarks",lambda _:pytest.fail("landmarks ran"))
    ctx=module.CaptureTaskContext(tmp_path,parse_capture_images(images()))
    with pytest.raises(CaptureInputError,match="cannot decode"):
        module._download_images(ctx,lambda _:b"bad",lambda _:b"bad",1024)
    with pytest.raises(CaptureInputError,match="oversized"):
        module._download_images(ctx,lambda _:b"x"*1025,lambda _:b"",1024)


def test_request_dispatch_restores_context_without_changing_environment(tmp_path,monkeypatch):
    import cv2
    from src import worker
    from src.detection_runtime import cloud_task as module
    from src.detection_runtime import cloud_inference
    monkeypatch.setenv("DERMAVISION_CAPTURE_PROFILE","consumer")
    data=cv2.imencode(".jpg",np.zeros((8,8,3),np.uint8))[1].tobytes()
    monkeypatch.setattr(worker,"download_via_internal_api",lambda _:data)
    monkeypatch.setattr(worker,"_get_pipeline",lambda:object())
    observed=[]
    def pipeline(*args):
        observed.append(capture_profile_from_environment())
        return {"status":"failed","message":"intentional_test_failure"}
    monkeypatch.setattr(worker,"run_task_pipeline",pipeline)
    monkeypatch.setattr(module,"_retain_failure",lambda *_:None)
    for _ in range(2):
        result=worker.analyze_image.run("case","",None,["pores"],
            capture_images=[{"filename":"phone.jpg","oss_key":"phone.jpg"}])
        assert result["status"]=="failed"
        assert module.CURRENT_CAPTURE.get() is None
    assert observed==[CaptureProfile.CONSUMER]*2
    assert capture_profile_from_environment()==CaptureProfile.CONSUMER




def test_cp_registration_failure_is_rejected(tmp_path, monkeypatch):
    import cv2
    from src.detection_runtime import cloud_task as module
    data = cv2.imencode(".jpg", np.zeros((32,32,3), np.uint8))[1].tobytes()
    monkeypatch.setattr(module._REGISTRATION, "landmarks", lambda _: np.ones((478,2),np.float32))
    monkeypatch.setattr(module, "rgb_geometry_valid", lambda *_: True)
    monkeypatch.setattr(module, "assess_registration", lambda role,*_: dict(
        status="FAILED" if role=="CP_M" else "PASSED", transform_sha256=None, reason="test"))
    ctx = module.CaptureTaskContext(tmp_path, parse_capture_images(images()))
    with pytest.raises(CaptureInputError, match="CP_M"):
        module._download_images(ctx, lambda _: data, lambda _: data, 1024*1024)
    assert ctx.capture is None


def test_signed_url_source_is_rejected():
    rows = [{"filename": r["filename"], "signed_picture_url": "https://example.test/"+str(i)}
            for i,r in enumerate(images())]
    with pytest.raises(CaptureInputError, match="object keys only"):
        parse_capture_images(rows)
