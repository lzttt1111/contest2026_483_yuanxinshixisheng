from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.capture_profile import CaptureProfile
from src.preprocess.image_preprocessor import PreprocessResultV2


def _preprocess_result() -> PreprocessResultV2:
    image = np.full((12, 12, 3), 80, dtype=np.uint8)
    return PreprocessResultV2(
        analysis_image=image.copy(),
        display_image=image.copy(),
        skin_mask=np.full((12, 12), 255, dtype=np.uint8),
        landmarks=np.zeros((478, 2), dtype=np.float32),
        face_transform_matrix=np.array(
            [[1024.0 / 12.0, 0.0, 0.0], [0.0, 1024.0 / 12.0, 0.0]],
            dtype=np.float32,
        ),
        inverse_transform_matrix=np.eye(2, 3, dtype=np.float32),
        quality_score=90.0,
        quality_status="PASS",
        quality_flags=[],
    )


def test_single_rgb_is_repeated_for_vendor_signature(tmp_path: Path) -> None:
    from src.engines.vendor_skin_style import VendorSkinStyleProvider

    source = np.full((12, 12, 3), 20, dtype=np.uint8)
    seen: list[np.ndarray] = []

    def fake_enhance(
        left: np.ndarray,
        middle: np.ndarray,
        right: np.ndarray,
        output_root: str,
        outputs: list[str],
    ) -> dict[str, str]:
        seen.extend((left, middle, right))
        assert outputs == ["red", "brown", "hxs"]
        root = Path(output_root)
        for name, value in (
            ("RED_M.jpg", 40),
            ("BROWN_M.jpg", 80),
            ("REDNE_M.jpg", 120),
        ):
            assert cv2.imwrite(str(root / name), np.full_like(source, value))
        return {"status": "sucessful!"}

    result = VendorSkinStyleProvider(enhance_images=fake_enhance).build(
        _preprocess_result(),
        source_image=source,
        output_root=tmp_path / "vendor",
        source_role="RGB_M",
    )

    assert all(item is source for item in seen)
    assert result.source_role == "RGB_M"
    assert result.provenance["reference_image_used"] is False
    assert result.provenance["input_mode"] == "front_only_rgb_m_repeated"
    assert result.red.analysis_image.shape == (1024, 1024, 3)
    assert result.brown.analysis_image.shape == (1024, 1024, 3)
    assert result.redne_image.shape == (1024, 1024, 3)


def test_clinic_views_share_one_generation_but_not_geometry_or_objects(
    tmp_path: Path,
) -> None:
    from src.detection_runtime.instrument_style import (
        InstrumentStyleRedBrownBaseProvider,
    )
    from src.engines.vendor_skin_style import VendorSkinStyleProvider

    source = np.zeros((12, 12, 3), dtype=np.uint8)
    source[:, :, 0] = np.arange(12, dtype=np.uint8)[None, :] * 10
    calls = 0

    def fake_enhance(left, middle, right, output_root, outputs):
        nonlocal calls
        calls += 1
        assert left is source and middle is source and right is source
        root = Path(output_root)
        for name, offset in (("RED_M.jpg", 10), ("BROWN_M.jpg", 20), ("REDNE_M.jpg", 30)):
            assert cv2.imwrite(str(root / name), np.clip(source + offset, 0, 255))
        return {"status": "sucessful!"}

    rgb_anchor = _preprocess_result()
    rgb_anchor.skin_mask.fill(255)
    rgb_anchor.face_transform_matrix = np.array(
        [[80.0, 0.0, 32.0], [0.0, 80.0, 16.0]], dtype=np.float32
    )
    rgb_anchor._debug_masks = {
        "hair_mask": np.zeros((12, 12), dtype=np.uint8)
    }
    cp_anchor = _preprocess_result()
    cp_anchor.skin_mask.fill(0)
    cp_anchor.face_transform_matrix = np.array(
        [[82.0, 0.0, 0.0], [0.0, 82.0, 0.0]], dtype=np.float32
    )
    cp_anchor._debug_masks = {
        "hair_mask": np.full((12, 12), 255, dtype=np.uint8)
    }
    provider = InstrumentStyleRedBrownBaseProvider()
    provider._provider = VendorSkinStyleProvider(enhance_images=fake_enhance)

    views = provider.build_clinic_views(
        red_brown_anchor=rgb_anchor,
        vascular_anchor=cp_anchor,
        source_image=source,
        output_root=tmp_path / "vendor",
        source_role="CP_M",
    )

    assert calls == 1
    assert views.red_brown is not views.vascular
    assert views.red_brown.red is not views.vascular.red
    assert views.red_brown.red.skin_mask is rgb_anchor.skin_mask
    assert views.vascular.red.skin_mask is cp_anchor.skin_mask
    assert views.red_brown.red._debug_masks is rgb_anchor._debug_masks
    assert views.vascular.red._debug_masks is cp_anchor._debug_masks
    assert views.red_brown.red_base_path.parent.name == "aligned_red_brown_rgb_geometry"
    assert views.vascular.red_base_path.parent.name == "aligned"
    assert views.red_brown.provenance["consumer"] == "redness_brown"
    assert views.vascular.provenance["consumer"] == "vascular_auxiliary_red"
    assert views.red_brown.provenance["geometry_mask_source_role"] == "RGB_M"
    assert views.vascular.provenance["geometry_mask_source_role"] == "CP_M"
    assert not np.array_equal(
        views.red_brown.red.analysis_image,
        views.vascular.red.analysis_image,
    )
    vascular_bytes = views.vascular.red.analysis_image.tobytes()
    views.red_brown.red.analysis_image.fill(0)
    assert views.vascular.red.analysis_image.tobytes() == vascular_bytes


def test_vendor_module_requires_matching_configured_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.engines.vendor_skin_style import (
        BUNDLED_VENDOR_ROOT,
        VendorSkinStyleConfigurationError,
        VendorSkinStyleProvider,
    )

    delivery = tmp_path / "vendor_skin"
    shutil.copytree(BUNDLED_VENDOR_ROOT.parent, delivery)
    module = delivery / "pyz/core/skin_generate.pyc"
    actual = hashlib.sha256(module.read_bytes()).hexdigest()
    monkeypatch.setenv("DERMAVISION_VENDOR_SKIN_MODULES_ROOT", str(delivery / "pyz"))
    monkeypatch.setenv("DERMAVISION_VENDOR_SKIN_MODULE_SHA256", "0" * 64)

    with pytest.raises(VendorSkinStyleConfigurationError, match="SHA256"):
        VendorSkinStyleProvider().build(
            _preprocess_result(),
            source_image=np.zeros((12, 12, 3), dtype=np.uint8),
            output_root=tmp_path / "output",
            source_role="RGB_M",
        )
    assert actual != "0" * 64


def test_vendor_module_uses_bundled_sidecar_when_env_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.engines.vendor_skin_style import (
        _configured_module,
    )

    monkeypatch.delenv("DERMAVISION_VENDOR_SKIN_MODULES_ROOT", raising=False)
    monkeypatch.delenv("DERMAVISION_VENDOR_SKIN_MODULE_SHA256", raising=False)

    root, digest = _configured_module()

    assert root.name == "pyz"
    assert root.parent.name == "vendor_skin"
    assert digest == "fd590298569d80ef6b3dfd3d7043c653715b12fddb5f41dbf1fd0ff8417a1581"


def test_vendor_module_rejects_incomplete_sidecar_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.engines.vendor_skin_style import (
        BUNDLED_VENDOR_ROOT,
        VendorSkinStyleConfigurationError,
        _configured_module,
    )

    delivery = tmp_path / "vendor_skin"
    shutil.copytree(BUNDLED_VENDOR_ROOT.parent, delivery)
    service_root = delivery / "pyz/core/service"
    service_root.chmod(0o755)
    (service_root / "__init__.pyc").unlink()
    monkeypatch.setenv("DERMAVISION_VENDOR_SKIN_MODULES_ROOT", str(delivery / "pyz"))
    monkeypatch.delenv("DERMAVISION_VENDOR_SKIN_MODULE_SHA256", raising=False)

    with pytest.raises(VendorSkinStyleConfigurationError, match="unavailable|file set"):
        _configured_module()


def test_pipeline_reuses_one_vendor_generation_and_preserves_result_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.pipeline as pipeline_module

    source = np.full((1024, 1024, 3), 10, dtype=np.uint8)
    input_path = tmp_path / "sample.jpg"
    assert cv2.imwrite(str(input_path), source)
    preprocessed = _preprocess_result()
    preprocessed.analysis_image = np.full((1024, 1024, 3), 15, dtype=np.uint8)
    preprocessed.display_image = preprocessed.analysis_image.copy()
    preprocessed.skin_mask = np.full((1024, 1024), 255, dtype=np.uint8)
    preprocessed.face_transform_matrix = np.eye(2, 3, dtype=np.float32)
    red = PreprocessResultV2(**{**preprocessed.__dict__})
    brown = PreprocessResultV2(**{**preprocessed.__dict__})
    red.analysis_image = np.full_like(source, 40)
    red.display_image = red.analysis_image.copy()
    brown.analysis_image = np.full_like(source, 80)
    brown.display_image = brown.analysis_image.copy()

    class FakePreprocessor:
        def preprocess_image(self, image: np.ndarray) -> PreprocessResultV2:
            assert image.shape == source.shape
            return preprocessed

    class FakeVendor:
        calls = 0

        def build(self, result, *, source_image, output_root, source_role):
            self.calls += 1
            assert result is preprocessed
            assert source_role == "RGB_M"
            Path(output_root).mkdir(parents=True)
            return SimpleNamespace(red=red, brown=brown, provenance={"ok": True})

    def write_common(root: Path, names: tuple[str, ...]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        for name in names:
            path = root / name
            if path.suffix.lower() in {".jpg", ".png"}:
                image = np.zeros((1024, 1024), dtype=np.uint8)
                image[10, 10] = 255
                assert cv2.imwrite(str(path), image)
            else:
                path.write_text("{}\n" if path.suffix == ".json" else "x\n", encoding="utf-8")

    class FakeRed:
        def process_preprocess_result(self, result, output_dir, source_name):
            assert result is red
            root = Path(output_dir) / "sample"
            write_common(
                root,
                (
                    "03_RBX红区结果图.jpg",
                    "06_VISIA红色区实例图.jpg",
                    "07_VISIA红色区实例Mask.png",
                    "红区量化指标.csv",
                    "红区量化指标.json",
                    "红区医学量化指标_V2.csv",
                ),
            )
            return str(root / "03_RBX红区结果图.jpg")

    class FakeBrown:
        _face_mask_analyzer = None

        def process_preprocess_result(self, result, output_dir, source_name):
            assert result is brown
            root = Path(output_dir) / "sample"
            write_common(
                root,
                (
                    "01_RBX棕区结果图.jpg",
                    "02_VISIA棕色斑实例图.jpg",
                    "03_棕色斑实例Mask.png",
                    "棕色斑量化指标.csv",
                    "棕色斑量化指标.json",
                    "棕色斑医学量化指标_V2.csv",
                ),
            )
            return str(root / "02_VISIA棕色斑实例图.jpg")

    class FakeCuda:
        def begin_task_profile(self):
            return None

        def finish_task_profile(self, baseline):
            return {
                "device": "cpu",
                "task_gpu_kernel_seconds": 0.0,
                "task_gpu_kernel_calls": 0,
                "task_peak_memory_mib": 0.0,
            }

    vendor = FakeVendor()
    monkeypatch.setattr(pipeline_module, "get_cuda_backend", lambda: FakeCuda())
    monkeypatch.setattr(
        pipeline_module,
        "render_vendor_red_result",
        lambda result, base, mask: np.full_like(base, 41),
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_module,
        "render_vendor_brown_result",
        lambda result, base, mask: np.full_like(base, 81),
        raising=False,
    )
    pipeline = pipeline_module.DermaVisionPipeline(CaptureProfile.INSTITUTION)
    pipeline.preprocessor = FakePreprocessor()
    pipeline.rbx_analyzer = FakeRed()
    pipeline.brown_analyzer = FakeBrown()
    pipeline.vendor_skin_provider = vendor
    pipeline.preprocessed_dir = str(tmp_path / "preprocessed")
    pipeline.rbx_dir = str(tmp_path / "rbx")
    pipeline.brown_dir = str(tmp_path / "brown")
    for directory in (pipeline.preprocessed_dir, pipeline.rbx_dir, pipeline.brown_dir):
        Path(directory).mkdir(parents=True, exist_ok=True)

    response = pipeline.process_single(str(input_path), ["redness", "brown"])

    assert response["status"] == "success"
    assert vendor.calls == 1
    results = response["results"]
    assert Path(results["redness"]).name == "03_RBX红区结果图.jpg"
    assert Path(results["red_areas_overlay"]).name == "06_VISIA红色区实例图.jpg"
    assert Path(results["brown"]).name == "01_RBX棕区结果图.jpg"
    assert Path(results["brown_spots_overlay"]).name == "02_VISIA棕色斑实例图.jpg"
    assert results["redness"] != results["red_areas_overlay"]
    assert results["brown"] != results["brown_spots_overlay"]
    assert int(cv2.imread(results["redness"])[100, 100, 0]) == 40
    assert int(cv2.imread(results["red_areas_overlay"])[100, 100, 0]) == 41
    assert int(cv2.imread(results["brown"])[100, 100, 0]) == 80
    assert int(cv2.imread(results["brown_spots_overlay"])[100, 100, 0]) == 81
