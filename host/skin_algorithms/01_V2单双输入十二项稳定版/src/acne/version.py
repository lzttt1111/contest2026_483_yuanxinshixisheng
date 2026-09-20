"""Algorithm version reported to ai-skin-backend.

Bump this constant when the algorithm (model weights / thresholds /
post-processing) changes; can be overridden by the ALGORITHM_VERSION
environment variable (see src/core/config.py). The version is returned
via metadata.algorithm_version and stored in skin_tasks.algorithm_snapshot.
"""

ALGORITHM_VERSION = "1.0.0"
