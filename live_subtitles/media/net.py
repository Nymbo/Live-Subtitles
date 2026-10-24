"""Small HTTP helpers on urllib (no extra dependencies): JSON in, JSON or bytes out."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from .. import __version__

USER_AGENT = f"LiveSubtitles v{__version__}"
TIMEOUT_S = 15


class HttpError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


def request(url: str, params: dict | None = None, headers: dict | None = None, data: dict | None = None,
            method: str | None = None, timeout: float = TIMEOUT_S) -> bytes:
    if params:
        clean = {k: v for k, v in params.items() if v is not None and v != ""}
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(clean)
    body = None
    hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json, */*"}
    hdrs.update(headers or {})
    if data is not None:
        body = json.dumps(data).encode()
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=hdrs, method=method or ("POST" if body else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        raise HttpError(exc.code, detail or exc.reason) from None


def get_json(url: str, params: dict | None = None, headers: dict | None = None, **kw):
    return json.loads(request(url, params, headers, **kw).decode("utf-8", errors="replace") or "null")


def post_json(url: str, data: dict, headers: dict | None = None, **kw):
    return json.loads(request(url, None, headers, data=data, **kw).decode("utf-8", errors="replace") or "null")
