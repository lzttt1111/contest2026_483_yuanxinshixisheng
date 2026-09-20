"""Local-surface command and temporary external-environment wiring."""

from __future__ import annotations

import shlex
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EnvironmentLinkError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class RuntimeCache:
    evidence_root: Path
    matplotlib: Path
    xdg: Path

    def relative_evidence(self) -> Mapping[str, str]:
        return {
            "mplconfigdir": self.matplotlib.relative_to(self.evidence_root).as_posix(),
            "xdg_cache_home": self.xdg.relative_to(self.evidence_root).as_posix(),
        }


def prepare_runtime_cache(evidence_root: Path) -> RuntimeCache:
    root = evidence_root / "runtime_cache"
    matplotlib = root / "matplotlib"
    xdg = root / "xdg"
    matplotlib.mkdir(parents=True)
    xdg.mkdir()
    return RuntimeCache(evidence_root=evidence_root, matplotlib=matplotlib, xdg=xdg)


def formal_runtime_environment(
    enabled: bool,
    frozen_root: Path | None,
) -> Mapping[str, str]:
    if not enabled:
        return {}
    if frozen_root is None:
        raise EnvironmentLinkError(detail="merged local formal root unavailable")
    return {"AISIA_FORMAL_BASELINE_ROOT": str(frozen_root)}


@contextmanager
def temporary_venv_link(project_root: Path, python_path: Path) -> Iterator[None]:
    """Expose an external environment only while branch-native residents run."""

    link = project_root / ".venv"
    if link.exists() or link.is_symlink():
        launcher = link / "bin/python"
        if not launcher.is_file() or launcher.resolve() != python_path.resolve():
            raise EnvironmentLinkError(
                detail=f"existing project environment differs: {link}"
            )
        yield
        return
    launcher = link / "bin/python"
    created = False
    try:
        launcher.parent.mkdir(parents=True)
        created = True
        launcher.write_text(
            "#!/bin/sh\n"
            f"exec {shlex.quote(str(python_path))} \"$@\"\n",
            encoding="utf-8",
        )
        launcher.chmod(0o755)
        yield
    finally:
        if created:
            shutil.rmtree(link)
