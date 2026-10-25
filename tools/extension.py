"""Draw the browser extension's icons (the tray icon's "CC" badge).

    python tools/extension.py icons     # writes extension/icons/icon{16,32,48,128}.png
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "extension"
GREEN = "#2EB872"           # the tray icon's "live" colour
SIZES = (16, 32, 48, 128)


def icons() -> None:
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter

    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    (EXT / "icons").mkdir(exist_ok=True)
    for size in SIZES:
        img = QImage(size, size, QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        # Chrome draws the 128 px icon with 96 px of art and a clear margin; small sizes fill the square.
        pad = size * 0.125 if size >= 128 else 0.5
        w = size - 2 * pad
        r = QRectF(pad, pad + w * 0.16, w, w * 0.68)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(GREEN))
        p.drawRoundedRect(r, w * 0.16, w * 0.16)
        font = QFont("Segoe UI")
        font.setPixelSize(max(7, int(w * 0.42)))
        font.setBold(True)
        p.setFont(font)
        p.setPen(QColor("white"))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, "CC")
        p.end()
        out = EXT / "icons" / f"icon{size}.png"
        img.save(str(out))
        print(out.relative_to(ROOT))
    del app


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "icons":
        icons()
    else:
        raise SystemExit(__doc__)
