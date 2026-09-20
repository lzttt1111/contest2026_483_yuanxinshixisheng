"""Internal compatibility symbol for the extracted RGB cores, not a routing API."""
from enum import Enum

class CaptureProfile(str, Enum):
    CONSUMER = "consumer"

def capture_profile_from_environment() -> CaptureProfile:
    # Deliberately ignores external V2 environment: this product only accepts RGB.
    return CaptureProfile.CONSUMER
