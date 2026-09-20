"""Score completeness and receipt-truth checks for merged local output."""

from __future__ import annotations

from pathlib import Path

from pydantic import JsonValue, TypeAdapter


JSON_ADAPTER = TypeAdapter(JsonValue)


def _mapping(value: JsonValue | None) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def score_and_receipt_errors(sample_root: Path) -> list[str]:
    complete = _mapping(JSON_ADAPTER.validate_json(
        (sample_root / "十二项完整量化指标.json").read_text(encoding="utf-8")
    ))
    modules = _mapping(_mapping(complete.get("scoring_results")).get("module_scores"))
    metric_count = 0
    scores_valid = len(modules) == 11
    for module_value in modules.values():
        module = _mapping(module_value)
        score = module.get("score")
        scores_valid = (
            scores_valid
            and module.get("score_valid") is True
            and isinstance(score, (int, float))
            and not isinstance(score, bool)
            and isinstance(module.get("grade"), str)
            and bool(module.get("grade"))
        )
        for group_value in _mapping(module.get("groups")).values():
            metric_count += len(_mapping(_mapping(group_value).get("metrics")))
    errors = [] if scores_valid and metric_count == 137 else ["score_contract"]
    receipt = _mapping(JSON_ADAPTER.validate_json(
        (sample_root / "运行回执.json").read_text(encoding="utf-8")
    ))
    if not _mapping(receipt.get("timing")) or not _mapping(receipt.get("input_signature")):
        errors.append("receipt_contract")
    return errors


def public_metric_truth_errors(
    sample_root: Path,
    public_paths: dict[str, Path],
) -> list[str]:
    complete = _mapping(JSON_ADAPTER.validate_json(
        (sample_root / "十二项完整量化指标.json").read_text(encoding="utf-8")
    ))
    detectors = _mapping(complete.get("detector_results"))
    if set(detectors) != set(public_paths):
        return ["public_metric_drift"]
    for item_id, path in public_paths.items():
        detector = _mapping(detectors.get(item_id))
        public = JSON_ADAPTER.validate_json(path.read_text(encoding="utf-8"))
        if detector.get("status") != "success" or detector.get("public_metrics") != public:
            return ["public_metric_drift"]
    return []
