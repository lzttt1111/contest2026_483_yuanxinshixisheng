"""Canonical effective measurement configuration shared by every stage-one route."""
from copy import deepcopy
import hashlib
import json
from importlib import import_module
from pathlib import Path


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def resolve_config(thresholds=None):
    from .stage1_lines import PARAMETERS as lines
    from .stage1_geometry import PARAMETERS as geometry
    config = deepcopy(thresholds or {})
    if not isinstance(config, dict):
        raise ValueError("measurement configuration must be an object")
    config["lines"] = {**lines, **config.get("lines", {})}
    config["geometry"] = {**geometry, **config.get("geometry", {})}
    imaging = config.setdefault("imaging", {})
    high = imaging.setdefault("high_precision_texture", False)
    if not isinstance(high, bool):
        raise ValueError("high_precision_texture must be explicitly boolean")
    return config


def measurement_config(calibration=None):
    calibration = calibration or {}
    config = deepcopy(calibration.get("thresholds") or {})
    if "imaging" in calibration:
        config["imaging"] = deepcopy(calibration["imaging"])
    return resolve_config(config)


def formula_identity(config):
    """Actual defaults and implementation sources invalidate stale references."""
    folder = Path(__file__).parent
    names = ("stage1_pipeline.py", "stage1_schema.py", "stage1_config.py", "registry.py",
             "regions.py", "geometry_regions.py", "measurements.py", "phenotypes.py",
             "wrinkle_metrics.py", "stage1_two_d.py", "stage1_two_d_phenotypes.py",
             "stage1_lines.py", "stage1_geometry.py")
    paths = {folder / name for name in names}
    paths.update(folder.glob("_stage1_*.py"))
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    runtime_defaults = {p.stem: {
        "version": getattr(import_module("." + p.stem, __package__), "VERSION", None),
        "parameters": getattr(import_module("." + p.stem, __package__), "PARAMETERS", {})}
        for p in sorted(paths)}
    return digest({"sources": sources, "runtime_defaults": runtime_defaults, "effective_parameters": config})


def scoring_identity():
    from .severity_guard import POLICY
    folder = Path(__file__).parent
    names = ("stage1_scoring.py", "stage1_reference.py", "stage1_guard.py",
             "stage1_zero_state.py", "stage1_word.py", "severity_guard.py", "scoring.py")
    return digest({"policy": POLICY, "sources": {
        name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in names}})
