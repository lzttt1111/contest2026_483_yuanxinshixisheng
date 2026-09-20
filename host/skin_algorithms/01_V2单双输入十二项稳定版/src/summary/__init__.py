"""Independent CPU-only overall-summary aggregation package.

Importing this package (or ``src.summary.worker``) is intentionally cheap: it
must never pull the detection app/engines, torch/ultralytics/mediapipe or Word
rendering. The heavy scoring chain is imported lazily inside the task body.

The launch route constants below are the single source of truth shared by
``src.worker``'s ``WORKER_TARGETS`` registry and ``run_cloud_worker.py``.
"""

from __future__ import annotations

SUMMARY_TARGET = "summary"
SUMMARY_APP = "src.summary.worker:celery_app"
SUMMARY_QUEUE = "summary"
# Declared for the additive double-Profile contract; not deployed this phase.
SUMMARY_CONSUMER_QUEUE = "consumer_summary"
SUMMARY_TASK = "dermavision.aggregate_summary"
# CPU-only per-report scoring: a small fixed pool is enough and must not
# compete with the GPU detection workers.
SUMMARY_CONCURRENCY = 2

__all__ = [
    "SUMMARY_APP",
    "SUMMARY_CONCURRENCY",
    "SUMMARY_CONSUMER_QUEUE",
    "SUMMARY_QUEUE",
    "SUMMARY_TARGET",
    "SUMMARY_TASK",
]
