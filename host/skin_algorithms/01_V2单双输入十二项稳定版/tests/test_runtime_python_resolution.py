from __future__ import annotations

from pathlib import Path

import pytest

from src.nine_analysis import orchestrator


def test_runtime_python_prefers_checkout_venv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "suite" / "02_dev_merged"
    local = project / ".venv" / "bin" / "python"
    shared = tmp_path / ".venv" / "bin" / "python"
    local.parent.mkdir(parents=True)
    shared.parent.mkdir(parents=True)
    local.write_bytes(b"local")
    shared.write_bytes(b"shared")
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", project)

    assert orchestrator.runtime_python() == local


def test_runtime_python_falls_back_to_shared_dev_venv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "suite" / "02_dev_merged"
    shared = tmp_path / ".venv" / "bin" / "python"
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b"shared")
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", project)

    assert orchestrator.runtime_python() == shared


def test_runtime_python_falls_back_to_active_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "clean-checkout"
    launcher = tmp_path / "main-env" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(b"launcher")
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", project)
    monkeypatch.setattr(orchestrator.sys, "executable", str(launcher))

    assert orchestrator.runtime_python() == launcher


def test_runtime_python_preserves_virtualenv_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "clean-checkout"
    base = tmp_path / "python-base"
    base.write_bytes(b"python")
    launcher = tmp_path / "main-env" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(base)
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", project)
    monkeypatch.setattr(orchestrator.sys, "executable", str(launcher))

    assert orchestrator.runtime_python() == launcher
