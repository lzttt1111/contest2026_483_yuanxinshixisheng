from __future__ import annotations

"""Linux atomic no-clobber publication for completed result directories."""

import ctypes
from dataclasses import dataclass
import errno
import os
from pathlib import Path


_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class AtomicPublishError(Exception):
    """Base error for fail-closed final-directory publication."""


@dataclass(frozen=True, slots=True)
class AtomicPublishConflict(AtomicPublishError):
    destination: Path

    def __str__(self) -> str:
        return f"final output already exists: {self.destination}"


@dataclass(frozen=True, slots=True)
class AtomicPublishUnsupported(AtomicPublishError):
    reason: str

    def __str__(self) -> str:
        return f"atomic no-replace publish is unsupported: {self.reason}"


@dataclass(frozen=True, slots=True)
class AtomicPublishSystemError(AtomicPublishError):
    source: Path
    destination: Path
    error_number: int

    def __str__(self) -> str:
        return (
            f"atomic publish failed: {self.source} -> {self.destination}: "
            f"errno={self.error_number}"
        )


def atomic_publish_directory(source: Path, destination: Path) -> None:
    """Rename one directory atomically and fail if destination exists."""

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise AtomicPublishUnsupported("libc.renameat2 is unavailable") from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise AtomicPublishConflict(destination)
    unsupported = {
        errno.ENOSYS,
        errno.EINVAL,
        getattr(errno, "EOPNOTSUPP", errno.ENOSYS),
    }
    if error_number in unsupported:
        raise AtomicPublishUnsupported(os.strerror(error_number))
    raise AtomicPublishSystemError(source, destination, error_number)


__all__ = [
    "AtomicPublishConflict",
    "AtomicPublishError",
    "AtomicPublishSystemError",
    "AtomicPublishUnsupported",
    "atomic_publish_directory",
]
