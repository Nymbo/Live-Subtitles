"""The app's small local web server: the browser extension's endpoint, on 127.0.0.1 only.

    POST /player   the page's video state: {title, page, duration, position, paused, rate,
                   media: [{url, type}], subtitles: [{url, lang, label}]}
    GET  /status   what media mode is doing (the extension's popup shows it)
Requests must carry "X-Live-Subtitles: 1". A web page can't add that header to a
cross-origin request without a CORS preflight, which this server never approves, so
only the extension (exempt through its host permission) can talk to it.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

logger = logging.getLogger(__name__)
DEFAULT_PORT = 47219


class _Handler(BaseHTTPRequestHandler):
    server_version = "LiveSubtitles"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):     # quiet; errors still go through logger
        pass

    @property
    def app(self) -> "WebServer":
        return self.server.app          # type: ignore[attr-defined]

    def _send(self, code: int, body: bytes, ctype: str = "application/json", extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _local(self) -> bool:
        return self.client_address[0] in ("127.0.0.1", "::1")

    def do_OPTIONS(self):
        self._send(403, b"", "text/plain")       # never approve a cross-origin preflight

    def do_POST(self):
        if not self._local() or self.headers.get("X-Live-Subtitles") != "1":
            return self._send(403, b'{"error":"forbidden"}')
        if self.path != "/player":
            return self._send(404, b"{}")
        n = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(min(n, 1 << 20)) or b"{}")
        except ValueError:
            return self._send(400, b'{"error":"bad json"}')
        reply = self.app.on_player(data) or {}
        self._send(200, json.dumps(reply).encode())

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if self.path == "/status":
            if not self._local() or self.headers.get("X-Live-Subtitles") != "1":
                return self._send(403, b"{}")
            return self._send(200, json.dumps(self.app.status() or {}).encode())
        self._send(404, b"{}")


class WebServer:
    """Runs the server thread. Callbacks are invoked on server threads."""

    def __init__(self, on_player: Callable[[dict], dict | None], status: Callable[[], dict],
                 port: int = DEFAULT_PORT):
        self.on_player, self.status, self.port = on_player, status, port
        self._local: ThreadingHTTPServer | None = None

    def _serve(self, host: str, port: int) -> ThreadingHTTPServer:
        srv = ThreadingHTTPServer((host, port), _Handler)
        srv.daemon_threads = True
        srv.app = self                       # type: ignore[attr-defined]
        threading.Thread(target=srv.serve_forever, daemon=True, name=f"web-{host}").start()
        return srv

    def start(self) -> None:
        if self._local is None:
            try:
                self._local = self._serve("127.0.0.1", self.port)
                logger.info("Extension endpoint on http://127.0.0.1:%d", self.port)
            except OSError as exc:
                logger.warning("Could not listen on port %d: %s", self.port, exc)

    def stop(self) -> None:
        if self._local is not None:
            self._local.shutdown()
            self._local.server_close()
            self._local = None
