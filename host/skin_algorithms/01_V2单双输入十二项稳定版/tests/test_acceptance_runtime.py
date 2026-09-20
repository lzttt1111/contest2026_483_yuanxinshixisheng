from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from scripts.acceptance.acceptance_process import run_managed_process
from scripts.acceptance.acceptance_io import AcceptanceInputError
from scripts.acceptance.acceptance_types import ProcessRequest
from scripts.acceptance.ordered_local_entry import _load_run
from scripts.acceptance.run_acceptance_surface import _validated_python_path
from scripts.acceptance.surface_baseline import prove_baseline
from scripts.acceptance.surface_local import (
    formal_runtime_environment,
    prepare_runtime_cache,
    temporary_venv_link,
)


@dataclass(frozen=True, slots=True)
class FixtureError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def test_run_managed_process_retains_log_and_kills_process_group_on_timeout(
    tmp_path: Path,
) -> None:
    # Given
    child_pid = tmp_path / "child.pid"
    runner = tmp_path / "runner.py"
    runner.write_text(
        "import os, signal, subprocess, sys\n"
        "child=subprocess.Popen([sys.executable, '-c', 'import signal; signal.pause()'])\n"
        "open(sys.argv[1], 'w', encoding='utf-8').write(str(child.pid))\n"
        "print('retained-before-timeout', flush=True)\n"
        "signal.pause()\n",
        encoding="utf-8",
    )
    request = ProcessRequest(
        command=(sys.executable, str(runner), str(child_pid)),
        cwd=tmp_path,
        environment={},
        log_path=tmp_path / "surface.log",
        resource_path=tmp_path / "resources.jsonl",
        timeout_seconds=0.2,
        sample_interval_seconds=0.05,
    )

    # When
    outcome = run_managed_process(request)
    pid = int(child_pid.read_text(encoding="utf-8"))

    # Then
    assert outcome.timed_out is True
    assert "retained-before-timeout" in request.log_path.read_text(encoding="utf-8")
    with pytest.raises(ProcessLookupError):
        os.kill(pid, signal.SIGCONT)


def test_temporary_venv_link_never_mutates_baseline_and_cleans_on_failure(
    tmp_path: Path,
) -> None:
    # Given
    project = tmp_path / "baseline"
    environment = tmp_path / "external" / ".venv"
    (environment / "bin").mkdir(parents=True)
    python = environment / "bin" / "python"
    python.write_text("", encoding="utf-8")
    project.mkdir()

    # When
    with pytest.raises(FixtureError, match="fixture"):
        with temporary_venv_link(project, python):
            launcher = project / ".venv/bin/python"
            assert launcher.is_file()
            assert os.access(launcher, os.X_OK)
            assert str(python) in launcher.read_text(encoding="utf-8")
            raise FixtureError(detail="fixture")

    # Then
    assert not (project / ".venv").exists()


def test_temporary_venv_link_executes_exact_shared_interpreter_with_packages(
    tmp_path: Path,
) -> None:
    # Given
    project = tmp_path / "baseline"
    project.mkdir()
    shared_python = Path(__file__).resolve().parents[3] / ".venv/bin/python"
    probe = (
        "import importlib.util,json,sys;"
        "print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,"
        "'cv2':bool(importlib.util.find_spec('cv2')),"
        "'torch':bool(importlib.util.find_spec('torch')),"
        "'pydantic_settings':bool(importlib.util.find_spec('pydantic_settings'))}))"
    )

    # When
    with temporary_venv_link(project, shared_python):
        launcher = project / ".venv/bin/python"
        result = subprocess.run(
            (str(launcher), "-c", probe),
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(project)},
        )
        payload = json.loads(result.stdout)

    # Then
    assert payload == {
        "executable": str(shared_python),
        "prefix": str(shared_python.parents[1]),
        "cv2": True,
        "torch": True,
        "pydantic_settings": True,
    }
    assert not (project / ".venv").exists()


def test_python_preflight_normalizes_parent_segments_without_resolving_symlink(
    tmp_path: Path,
) -> None:
    # Given
    shared = tmp_path / "shared/.venv/bin/python"
    shared.parent.mkdir(parents=True)
    shared.write_text("", encoding="utf-8")
    lexical = tmp_path / "suite/../shared/.venv/bin/python"

    # When / Then
    assert _validated_python_path(lexical, shared) == shared


def test_python_preflight_rejects_other_interpreter(tmp_path: Path) -> None:
    # Given
    shared = tmp_path / "shared/.venv/bin/python"
    other = tmp_path / "other/bin/python"
    shared.parent.mkdir(parents=True)
    other.parent.mkdir(parents=True)
    shared.write_text("", encoding="utf-8")
    other.write_text("", encoding="utf-8")

    # When / Then
    with pytest.raises(AcceptanceInputError, match="external Python"):
        _validated_python_path(other, shared)


def test_runtime_cache_redirects_real_baseline_matplotlib_import(
    tmp_path: Path,
) -> None:
    # Given
    baseline = Path(__file__).resolve().parents[2] / "01_dev_baseline"
    repository_cache = baseline / "src/wrinkle/algorithms/.cache"
    assert not repository_cache.exists()
    cache = prepare_runtime_cache(tmp_path / "_evidence")
    environment = {
        **os.environ,
        "PYTHONPATH": str(baseline),
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": str(cache.matplotlib),
        "XDG_CACHE_HOME": str(cache.xdg),
    }

    # When
    try:
        subprocess.run(
            (
                str(Path(__file__).resolve().parents[3] / ".venv/bin/python"),
                "-c",
                "import src.wrinkle.algorithms.wrinkle_detection_algorithm",
            ),
            check=True,
            cwd=baseline,
            env=environment,
        )
        prove_baseline(baseline)
    finally:
        shutil.rmtree(repository_cache, ignore_errors=True)

    # Then
    assert not repository_cache.exists()
    assert list(cache.matplotlib.glob("fontlist-*.json"))
    assert cache.relative_evidence() == {
        "mplconfigdir": "runtime_cache/matplotlib",
        "xdg_cache_home": "runtime_cache/xdg",
    }


def test_ordered_loader_registers_module_before_python310_dataclass_exec(
    tmp_path: Path,
) -> None:
    # Given
    (tmp_path / "run.py").write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True, slots=True)\n"
        "class WorkClaim:\n"
        "    name: str\n"
        "def main(arguments):\n"
        "    return len(arguments)\n",
        encoding="utf-8",
    )

    # When
    target = _load_run(tmp_path)

    # Then
    assert target.WorkClaim("fixture").name == "fixture"
    assert target.main([]) == 0


def test_ordered_loader_imports_real_main_help_without_models() -> None:
    # Given
    project_root = Path(__file__).resolve().parents[1]
    target = _load_run(project_root)

    # When / Then
    with pytest.raises(SystemExit) as outcome:
        target.main(["--help"])
    assert outcome.value.code == 0


def test_ordered_loader_restores_previous_module_after_import_failure(
    tmp_path: Path,
) -> None:
    # Given
    module_name = "acceptance_target_run"
    previous = sys.modules.get(module_name)
    sentinel = ModuleType(module_name)
    sys.modules[module_name] = sentinel
    (tmp_path / "run.py").write_text(
        "raise RuntimeError('fixture import failure')\n",
        encoding="utf-8",
    )

    # When / Then
    try:
        with pytest.raises(RuntimeError, match="fixture import failure"):
            _load_run(tmp_path)
        assert sys.modules[module_name] is sentinel
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


def test_formal_word_environment_is_merged_local_only(tmp_path: Path) -> None:
    # Given
    frozen_root = tmp_path / "frozen_formal_word_baseline"

    # When / Then
    assert formal_runtime_environment(True, frozen_root) == {
        "AISIA_FORMAL_BASELINE_ROOT": str(frozen_root)
    }
    assert formal_runtime_environment(False, frozen_root) == {}
    assert formal_runtime_environment(False, None) == {}
