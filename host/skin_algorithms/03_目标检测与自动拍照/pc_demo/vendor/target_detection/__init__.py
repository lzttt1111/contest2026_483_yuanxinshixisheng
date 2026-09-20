"""Target-detection 0.0.1 delivery package."""

from typing import Any

from .version import SOFTWARE_VERSION as __version__

__all__ = ["Pipeline", "V5Runtime"]


def __getattr__(name: str) -> Any:
    if name in {"Pipeline", "V5Runtime"}:
        if name == "Pipeline":
            from .pipeline import Pipeline

            return Pipeline
        from .runtime import V5Runtime

        return V5Runtime
    raise AttributeError(name)
