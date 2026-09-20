from __future__ import annotations

"""Writable runtime cache locations that never default inside the checkout."""

from collections.abc import MutableMapping
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile


@dataclass(frozen=True, slots=True)
class WrinkleRuntimeCaches:
    matplotlib: Path
    ultralytics: Path


def configure_runtime_caches(
    *,
    environ: MutableMapping[str, str] | None = None,
    temp_root: str | Path | None = None,
) -> WrinkleRuntimeCaches:
    """Set missing cache variables to a deterministic writable temp namespace."""

    active = os.environ if environ is None else environ
    base = Path(temp_root or tempfile.gettempdir()).expanduser().resolve()
    namespace = base / "dermavision" / "wrinkle-algorithms"
    defaults = {
        "MPLCONFIGDIR": namespace / "matplotlib",
        "YOLO_CONFIG_DIR": namespace / "ultralytics",
    }
    resolved: dict[str, Path] = {}
    for name, default in defaults.items():
        value = active.setdefault(name, str(default))
        path = Path(value).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir() or not os.access(path, os.W_OK | os.X_OK):
            raise OSError(f"runtime cache is not writable: {name}")
        resolved[name] = path
    return WrinkleRuntimeCaches(
        matplotlib=resolved["MPLCONFIGDIR"],
        ultralytics=resolved["YOLO_CONFIG_DIR"],
    )


__all__ = ["WrinkleRuntimeCaches", "configure_runtime_caches"]
