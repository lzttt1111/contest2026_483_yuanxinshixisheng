from __future__ import annotations

import hashlib
import re
from typing import Any

from src.capture_profile import CaptureProfile


def safe_name(value: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    return sanitized[:120] or "image"


def claim_key(signature: dict[str, Any]) -> str:
    identity = "|".join((
        str(signature["relative_path"]),
        str(signature.get(
            "capture_profile",
            CaptureProfile.INSTITUTION.value,
        )),
    ))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


__all__ = ["claim_key", "safe_name"]
