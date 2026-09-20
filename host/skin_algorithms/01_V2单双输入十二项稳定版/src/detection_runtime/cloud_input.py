from __future__ import annotations

"""Validate cloud image lists without guessing roles from image pixels."""

from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from urllib.parse import urlsplit


ROLES = ("RGB_M", "PP_M", "CP_M", "365_M")
ROLE_TOKEN = re.compile(r"(?<![A-Za-z0-9])(RGB|PP|CP|UV|365)(?![A-Za-z0-9])", re.I)


class CaptureInputError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class ImageSource:
    filename: str
    locator: str
    by_url: bool
    role: str
    sample: str


def parse_capture_images(value: object) -> tuple[ImageSource, ...]:
    if not isinstance(value, list) or len(value) not in (1, 4):
        raise CaptureInputError("invalid_capture_count", "capture_images requires one or four images")
    images = []
    for row in value:
        if isinstance(row, dict) and "signed_picture_url" in row:
            raise CaptureInputError("unsupported_capture_source", "capture_images accepts object keys only")
        if not isinstance(row, dict) or set(row) - {"filename", "oss_key", "signed_picture_url"}:
            raise CaptureInputError("invalid_capture_image", "invalid image descriptor")
        name = row.get("filename")
        if not isinstance(name, str) or not name.strip() or len(name) > 255:
            raise CaptureInputError("invalid_capture_name", "filename is required")
        if any(c in name for c in ("/", "\\", "\x00")) or name in (".", ".."):
            raise CaptureInputError("invalid_capture_name", "filename must be a basename")
        if PurePosixPath(name).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            raise CaptureInputError("invalid_capture_format", "unsupported image extension")
        locators = [key for key in ("oss_key", "signed_picture_url") if row.get(key)]
        if len(locators) != 1 or not isinstance(row[locators[0]], str):
            raise CaptureInputError("invalid_capture_source", "provide exactly one object key or signed URL")
        locator = row[locators[0]].strip()
        by_url = locators[0] == "signed_picture_url"
        if not locator:
            raise CaptureInputError("invalid_capture_source", "empty image source")
        if by_url:
            try:
                parts = urlsplit(locator)
                valid = parts.scheme in {"http", "https"} and bool(parts.hostname)
            except ValueError:
                valid = False
            if not valid:
                raise CaptureInputError("invalid_capture_source", "invalid signed URL")
        elif locator.startswith(("/", "\\")) or ".." in locator.replace("\\", "/").split("/") or "://" in locator:
            raise CaptureInputError("invalid_capture_source", "invalid object key")
        stem = PurePosixPath(name).stem
        tokens = list(ROLE_TOKEN.finditer(stem))
        if len(value) == 4 and len(tokens) != 1:
            raise CaptureInputError("ambiguous_capture_role", "each filename must identify exactly one light")
        role = "RGB_M"
        sample = ""
        if len(value) == 4:
            token = tokens[0]
            short = token.group().upper()
            role = ("365" if short == "UV" else short) + "_M"
            sample = (stem[:token.start()] + stem[token.end():]).strip(" _-.").casefold()
            sample = re.sub(r"(^|[_ .-])m$", "", sample).strip(" _-.")
        images.append(ImageSource(name, locator, by_url, role, sample))
    if len(value) == 4:
        if {image.role for image in images} != set(ROLES):
            raise CaptureInputError("duplicate_capture_role", "RGB/PP/CP/UV must each occur once")
        if len({image.sample for image in images}) != 1:
            raise CaptureInputError("mixed_capture_identity", "four filenames must identify the same capture")
        images.sort(key=lambda image: ROLES.index(image.role))
    return tuple(images)
