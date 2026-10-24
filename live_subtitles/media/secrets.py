"""API keys and the OpenSubtitles password, encrypted for this Windows user (DPAPI).

settings.json roams with the profile and is plain text, so secrets are stored as
"dpapi:<base64>" blobs that only this user on this machine can decrypt. Values that
are not encrypted (typed in by hand, or from another OS) pass through unchanged.
"""

from __future__ import annotations

import base64
import ctypes
import sys
from ctypes import wintypes

PREFIX = "dpapi:"


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _call(fn, data: bytes) -> bytes:
    buf = ctypes.create_string_buffer(data, len(data))
    src = _Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    out = _Blob()
    if not fn(ctypes.byref(src), None, None, None, None, 0x01, ctypes.byref(out)):   # UI_FORBIDDEN
        raise OSError("DPAPI call failed")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(ctypes.cast(out.pbData, ctypes.c_void_p))


def protect(value: str) -> str:
    if not value or value.startswith(PREFIX) or sys.platform != "win32":
        return value
    crypt32 = ctypes.windll.crypt32
    blob = _call(crypt32.CryptProtectData, value.encode("utf-8"))
    return PREFIX + base64.b64encode(blob).decode("ascii")


def unprotect(value: str) -> str:
    if not value or not value.startswith(PREFIX) or sys.platform != "win32":
        return value or ""
    try:
        return _call(ctypes.windll.crypt32.CryptUnprotectData, base64.b64decode(value[len(PREFIX):])).decode("utf-8")
    except Exception:
        return ""
