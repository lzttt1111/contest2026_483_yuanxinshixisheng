from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class PopulationGateDecision:
    passed: bool
    failed_rules: tuple[str, ...]
    availability_rate: float
    zero_rate: float
    hundred_rate: float
    endpoint_rate: float
    median: float | None
    iqr: float | None


def evaluate_population_module_gate(
    *,
    scores: Sequence[float],
    attempted_count: int,
    direction_perturbation_passed: bool,
) -> PopulationGateDecision:
    if attempted_count < 1:
        raise ValueError("attempted count must be positive")
    values = np.asarray(scores, dtype=np.float64)
    availability = values.size / attempted_count
    zero_rate = float(np.mean(values == 0.0)) if values.size else 0.0
    hundred_rate = float(np.mean(values == 100.0)) if values.size else 0.0
    endpoint_rate = zero_rate + hundred_rate
    median = float(np.median(values)) if values.size else None
    iqr = (
        float(np.percentile(values, 75) - np.percentile(values, 25))
        if values.size
        else None
    )
    checks = (
        (availability >= 0.95, "availability_rate"),
        (zero_rate <= 0.05, "zero_rate"),
        (hundred_rate <= 0.05, "hundred_rate"),
        (endpoint_rate <= 0.10, "endpoint_rate"),
        (
            median is not None and 35.0 <= median <= 65.0,
            "median",
        ),
        (iqr is not None and iqr >= 20.0, "iqr"),
        (direction_perturbation_passed, "direction_perturbation"),
    )
    failed = tuple(rule for passed, rule in checks if not passed)
    return PopulationGateDecision(
        passed=not failed,
        failed_rules=failed,
        availability_rate=availability,
        zero_rate=zero_rate,
        hundred_rate=hundred_rate,
        endpoint_rate=endpoint_rate,
        median=median,
        iqr=iqr,
    )


__all__ = ["PopulationGateDecision", "evaluate_population_module_gate"]
