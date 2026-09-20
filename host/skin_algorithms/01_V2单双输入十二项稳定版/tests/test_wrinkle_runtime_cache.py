from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from src.wrinkle.runtime_cache import configure_runtime_caches


ROOT = Path(__file__).resolve().parents[1]


def test_default_caches_are_writable_and_outside_repository(tmp_path: Path) -> None:
    environment: dict[str, str] = {}

    caches = configure_runtime_caches(environ=environment, temp_root=tmp_path)

    assert caches.matplotlib == tmp_path / "dermavision/wrinkle-algorithms/matplotlib"
    assert caches.ultralytics == tmp_path / "dermavision/wrinkle-algorithms/ultralytics"
    assert all(path.is_dir() and os.access(path, os.W_OK | os.X_OK) for path in (
        caches.matplotlib,
        caches.ultralytics,
    ))
    assert all(not path.is_relative_to(ROOT) for path in (
        caches.matplotlib,
        caches.ultralytics,
    ))


def test_explicit_cache_environment_is_preserved(tmp_path: Path) -> None:
    explicit_matplotlib = tmp_path / "explicit/matplotlib"
    explicit_yolo = tmp_path / "explicit/ultralytics"
    environment = {
        "MPLCONFIGDIR": str(explicit_matplotlib),
        "YOLO_CONFIG_DIR": str(explicit_yolo),
    }

    caches = configure_runtime_caches(environ=environment, temp_root=tmp_path / "unused")

    assert caches.matplotlib == explicit_matplotlib
    assert caches.ultralytics == explicit_yolo
    assert environment == {
        "MPLCONFIGDIR": str(explicit_matplotlib),
        "YOLO_CONFIG_DIR": str(explicit_yolo),
    }


def test_default_cache_creation_is_concurrent_safe(tmp_path: Path) -> None:
    def configure(_index: int):
        return configure_runtime_caches(environ={}, temp_root=tmp_path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(configure, range(32)))

    assert len({row.matplotlib for row in results}) == 1
    assert len({row.ultralytics for row in results}) == 1
    assert results[0].matplotlib.is_dir()
    assert results[0].ultralytics.is_dir()


def test_wrinkle_import_and_default_all_selection_do_not_create_source_cache(
    tmp_path: Path,
) -> None:
    source_cache = ROOT / "src/wrinkle/algorithms/.cache"

    def inventory() -> dict[str, tuple[int, int]]:
        if not source_cache.exists():
            return {}
        return {
            path.relative_to(source_cache).as_posix(): (
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in source_cache.rglob("*")
            if path.is_file()
        }

    before = inventory()
    code = (
        "import json,os,run;"
        "import src.wrinkle.algorithms.wrinkle_detection_algorithm as module;"
        "print(json.dumps({'mpl':os.environ['MPLCONFIGDIR'],"
        "'yolo':os.environ['YOLO_CONFIG_DIR'],"
        "'all':list(run.resolve_algorithms(['all'])),"
        "'module':module.__file__}))"
    )
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(tmp_path),
        "CUDA_VISIBLE_DEVICES": "",
    }
    environment.pop("MPLCONFIGDIR", None)
    environment.pop("YOLO_CONFIG_DIR", None)

    completed = subprocess.run(
        (sys.executable, "-B", "-c", code),
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.splitlines()[-1])

    assert Path(payload["mpl"]).is_relative_to(tmp_path)
    assert Path(payload["yolo"]).is_relative_to(tmp_path)
    assert len(payload["all"]) == 12
    assert Path(payload["module"]).is_relative_to(ROOT)
    assert inventory() == before
