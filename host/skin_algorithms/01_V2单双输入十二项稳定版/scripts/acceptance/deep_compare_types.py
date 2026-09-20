"""Typed inputs and output for deep acceptance comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.acceptance.acceptance_types import JsonValue


@dataclass(frozen=True, slots=True)
class ComparisonRequest:
    baseline_root: Path
    merged_root: Path
    baseline_run: Path
    merged_run: Path
    baseline_cloud: Path
    merged_cloud: Path
    formal_baseline_root: Path


@dataclass(frozen=True, slots=True)
class ComparisonFinding:
    code: str
    section: str
    detail: str

    def as_json(self) -> dict[str, JsonValue]:
        return {"code": self.code, "section": self.section, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ComparisonVerdict:
    status: str
    blocking_findings: tuple[ComparisonFinding, ...]
    intentional_differences: tuple[ComparisonFinding, ...]

    def as_json(self) -> dict[str, JsonValue]:
        return {
            "status": self.status,
            "blocking_findings": [item.as_json() for item in self.blocking_findings],
            "intentional_differences": [
                item.as_json() for item in self.intentional_differences
            ],
        }


@dataclass(frozen=True, slots=True)
class DeepReport:
    schema: str
    sections: dict[str, JsonValue]
    verdict: ComparisonVerdict

    def as_json(self) -> dict[str, JsonValue]:
        return {
            "schema": self.schema,
            "verdict": self.verdict.as_json(),
            "sections": self.sections,
        }
