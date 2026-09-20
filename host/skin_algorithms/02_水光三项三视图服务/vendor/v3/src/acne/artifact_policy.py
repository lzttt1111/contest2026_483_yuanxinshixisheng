from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from typing import Final, Iterable, Iterator, Mapping


DEBUG_ARTIFACTS_TOKEN: Final = "debug-artifacts"
FORMAL_FAST_TOKEN: Final = "formal-fast"
_POLICY_TOKENS: Final = frozenset({DEBUG_ARTIFACTS_TOKEN, FORMAL_FAST_TOKEN})


class AcneArtifactPolicy(str, Enum):
    """Controls whether the acne pipeline persists diagnostic image bundles."""

    FORMAL = "formal"
    FORMAL_FAST = FORMAL_FAST_TOKEN
    DEBUG_ARTIFACTS = DEBUG_ARTIFACTS_TOKEN

    @classmethod
    def from_algorithms(cls, algorithms: Iterable[str] | None) -> "AcneArtifactPolicy":
        if algorithms is not None and DEBUG_ARTIFACTS_TOKEN in algorithms:
            return cls.DEBUG_ARTIFACTS
        if algorithms is not None and FORMAL_FAST_TOKEN in algorithms:
            return cls.FORMAL_FAST
        return cls.FORMAL

    @property
    def includes_debug_images(self) -> bool:
        return self is AcneArtifactPolicy.DEBUG_ARTIFACTS

    @property
    def generates_max_recall_candidates(self) -> bool:
        return self is not AcneArtifactPolicy.FORMAL_FAST


def selected_acne_algorithms(algorithms: Iterable[str] | None) -> list[str]:
    selected = list(algorithms or ["acne"])
    return [name for name in selected if name not in _POLICY_TOKENS]


@dataclass
class AcnePhaseTimings:
    """Six-key compatibility timing contract.

    ``image_encoding`` covers image rendering, encoding, and file writes only.
    ``quantification`` also includes structured JSON/CSV persistence so those
    operations are not misreported as image work.
    """

    preprocess: float = 0.0
    lds: float = 0.0
    yolo: float = 0.0
    candidate_generation: float = 0.0
    image_encoding: float = 0.0
    quantification: float = 0.0

    @contextmanager
    def measure_image_encoding(self) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            self.image_encoding += max(0.0, perf_counter() - started)

    @contextmanager
    def measure_structured_persistence(self) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            self.quantification += max(0.0, perf_counter() - started)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "AcnePhaseTimings":
        def seconds(name: str) -> float:
            value = values.get(name, 0.0)
            return max(0.0, float(value)) if isinstance(value, (int, float)) else 0.0

        return cls(
            preprocess=seconds("preprocess"),
            lds=seconds("lds"),
            yolo=seconds("yolo"),
            candidate_generation=seconds("candidate_generation"),
            image_encoding=seconds("image_encoding"),
            quantification=seconds("quantification"),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "preprocess": round(self.preprocess, 4),
            "lds": round(self.lds, 4),
            "yolo": round(self.yolo, 4),
            "candidate_generation": round(self.candidate_generation, 4),
            "image_encoding": round(self.image_encoding, 4),
            "quantification": round(self.quantification, 4),
        }
