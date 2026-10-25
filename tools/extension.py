"""Build the browser extension for the Chrome Web Store and Edge Add-ons.

    python tools/extension.py icons     # draw extension/icons/*.png (the tray icon's "CC" badge)
    python tools/extension.py pack      # write dist/live-subtitles-extension-<version>.zip
    python tools/extension.py art       # store screenshots and promo tile in dist/store/

The zip holds only what the manifest loads, so store uploads never carry stray files.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "extension"
DIST = ROOT / "dist"
GREEN = "#2EB872"           # the tray icon's "live" colour
SIZES = (16, 32, 48, 128)
SHIPPED = ("manifest.json", "background.js", "classify.js", "content.js", "popup.html", "popup.js")


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
        # The store wants 96 px of art inside a 128 px icon; small sizes use the whole square.
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


SHOTS = (
    ("Real subtitles, on time, in your browser",
     "Whole lines ahead of the speech, synced to the video the extension reports.",
     ["Then we start over.", "From the very beginning."], "The Long Night (2024) · Synced."),
    ("Sound descriptions and English for any language",
     "Hearing-impaired tracks when they exist, generated sound cues when they don't.",
     ["[metal clanking]", "Get everyone out of here, now!"], "The Long Night (2024) · Synced."),
    ("Follows the tab you're watching",
     "Exact timing from the first line, through pauses, seeks and skipped scenes.",
     ["I'm not doing this again."], "The Long Night (2024) · Synced."),
)


def art() -> None:
    """Store screenshots (1280x800) and the small promo tile (440x280), drawn with the app's
    own caption renderer so the subtitles look exactly as they do on screen."""
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import (QColor, QFont, QGuiApplication, QImage, QLinearGradient, QPainter, QPen,
                               QRadialGradient)

    sys.path.insert(0, str(ROOT))
    from live_subtitles.renderer import block_height, paint_captions
    from live_subtitles.settings import Appearance

    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    out_dir = DIST / "store"
    out_dir.mkdir(parents=True, exist_ok=True)
    icon = QImage(str(EXT / "icons" / "icon128.png"))

    def font(px: int, bold: bool = False) -> QFont:
        f = QFont("Segoe UI")
        f.setPixelSize(px)
        f.setBold(bold)
        return f

    for n, (title, sub, lines, state) in enumerate(SHOTS, 1):
        img = QImage(1280, 800, QImage.Format.Format_ARGB32)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        p.fillRect(img.rect(), QColor("#0f1115"))
        p.setPen(QColor("#f2f4f7"))
        p.setFont(font(40, True))
        p.drawText(QRectF(64, 40, 1152, 56), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)
        p.setPen(QColor("#9aa3ad"))
        p.setFont(font(22))
        p.drawText(QRectF(64, 96, 1152, 36), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, sub)
        # A browser window with a playing video.
        win = QRectF(64, 160, 1152, 600)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#2b2d31"))
        p.drawRoundedRect(win, 12, 12)
        p.setBrush(QColor("#3a3d43"))
        p.drawRoundedRect(QRectF(win.x() + 16, win.y() + 12, 260, 30), 8, 8)
        p.setPen(QColor("#d0d4da"))
        p.setFont(font(15))
        p.drawText(QRectF(win.x() + 30, win.y() + 12, 240, 30), Qt.AlignmentFlag.AlignVCenter, "The Long Night")
        video = QRectF(win.x() + 8, win.y() + 52, win.width() - 16, win.height() - 60)
        g = QRadialGradient(QPointF(video.center().x() - 180, video.center().y() - 60), 620)
        g.setColorAt(0.0, QColor("#3b5a78"))
        g.setColorAt(0.55, QColor("#1c2b3a"))
        g.setColorAt(1.0, QColor("#090c10"))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawRect(video)
        ap = Appearance()
        ap.font_size = 34
        cap = QRectF(video.x(), video.bottom() - 40 - block_height(ap, len(lines)), video.width(),
                     block_height(ap, len(lines)))
        paint_captions(p, cap, lines, ap)
        # The extension's popup, as it shows while synced.
        pop = QRectF(win.right() - 330, win.y() + 46, 300, 132)
        p.setBrush(QColor("#ffffff"))
        p.setPen(QPen(QColor("#d0d4da"), 1))
        p.drawRoundedRect(pop, 10, 10)
        p.drawImage(QRectF(pop.x() + 14, pop.y() + 12, 28, 28), icon)
        p.setPen(QColor("#1f2328"))
        p.setFont(font(15, True))
        p.drawText(QRectF(pop.x() + 50, pop.y() + 12, 230, 28), Qt.AlignmentFlag.AlignVCenter, "Live Subtitles")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#f2f4f7"))
        st = QRectF(pop.x() + 14, pop.y() + 50, pop.width() - 28, 64)
        p.drawRoundedRect(st, 6, 6)
        p.setPen(QColor("#1f2328"))
        p.setFont(font(14))
        p.drawText(st.adjusted(10, 6, -10, -6), Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignVCenter, state)
        p.end()
        path = out_dir / f"screenshot-{n}.png"
        img.save(str(path))
        print(path.relative_to(ROOT))

    tile = QImage(440, 280, QImage.Format.Format_ARGB32)
    p = QPainter(tile)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    g = QLinearGradient(0, 0, 440, 280)
    g.setColorAt(0, QColor("#16202b"))
    g.setColorAt(1, QColor("#0b0e12"))
    p.fillRect(tile.rect(), g)
    p.drawImage(QRectF(40, 52, 96, 96), icon)
    p.setPen(QColor("#f2f4f7"))
    p.setFont(font(34, True))
    p.drawText(QRectF(150, 62, 280, 44), Qt.AlignmentFlag.AlignVCenter, "Live Subtitles")
    p.setPen(QColor("#9aa3ad"))
    p.setFont(font(17))
    p.drawText(QRectF(150, 106, 280, 50), Qt.TextFlag.TextWordWrap, "Movie and TV subtitles, on time.")
    ap = Appearance()
    ap.font_size = 22
    paint_captions(p, QRectF(0, 200, 440, block_height(ap, 1)), ["[door creaks] Who's there?"], ap)
    p.end()
    path = out_dir / "promo-440x280.png"
    tile.save(str(path))
    print(path.relative_to(ROOT))
    del app


def pack() -> Path:
    manifest = json.loads((EXT / "manifest.json").read_text(encoding="utf-8"))
    DIST.mkdir(exist_ok=True)
    out = DIST / f"live-subtitles-extension-{manifest['version']}.zip"
    files = [EXT / f for f in SHIPPED] + sorted((EXT / "icons").glob("icon*.png"))
    missing = [f for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"missing: {', '.join(str(f) for f in missing)} (run: python tools/extension.py icons)")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(EXT).as_posix())
    print(f"{out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB, {len(files)} files)")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "icons":
        icons()
    elif cmd == "art":
        art()
    elif cmd == "pack":
        pack()
    else:
        raise SystemExit(__doc__)
