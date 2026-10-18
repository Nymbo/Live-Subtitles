"""The on-screen subtitle window: frameless, translucent, always on top.

Nothing but the subtitles is ever drawn unless the mouse is over the window. With no
subtitles the overlay is fully invisible, and hovering it reveals a frame, the resize
grips, a placeholder and a small toolbar (lock, settings, hide), which vanish again as
soon as the mouse leaves.

Unlocked: drag to move, drag either side to resize, Ctrl+wheel to change the text size,
double-click to snap to the bottom centre, right-click for a menu. Locked: fully
click-through, so it never steals clicks from the video underneath, and never shows
the hover chrome (unlock from the tray or with the hotkey).

When a line rolls off the top, the block slides up instead of jumping.
"""

from __future__ import annotations

import ctypes
import sys

from PySide6.QtCore import QEasingCurve, QPoint, QPointF, QRect, QRectF, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QGuiApplication, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QMenu, QWidget

from .renderer import block_height, paint_captions
from .settings import Appearance, Layout

EDGE = 14           # px from the left/right edge that resizes instead of moving
MIN_WIDTH = 240
BUTTON = 26         # toolbar button size
ICON_FONTS = ("Segoe Fluent Icons", "Segoe MDL2 Assets")
# (id, glyph, tooltip) in Segoe Fluent Icons / MDL2 Assets
BUTTONS = (("lock", "", "Lock in place (click-through)"),
           ("settings", "", "Settings"),
           ("hide", "", "Hide subtitles"))
CHROME_IN_MS, CHROME_OUT_MS = 120, 90
SCROLL_MS = 220
FADE_IN_MS = 110


def _anim(parent, duration: int, on_value) -> QVariantAnimation:
    a = QVariantAnimation(parent)
    a.setDuration(duration)
    a.setEasingCurve(QEasingCurve.Type.OutCubic)
    a.valueChanged.connect(on_value)
    return a


class CaptionOverlay(QWidget):
    geometry_changed = Signal(int, int, int)      # x, y, width
    font_size_changed = Signal(int)
    lock_requested = Signal(bool)
    hide_requested = Signal()
    settings_requested = Signal()

    def __init__(self, appearance: Appearance, layout: Layout):
        super().__init__(None)
        self.setWindowTitle("Live Subtitles overlay")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setMouseTracking(True)
        self.ap = appearance
        self.layout_cfg = layout
        self.lines: list[str] = []
        self.status = "Live subtitles appear here"
        self._opacity = 1.0                 # caption opacity (fade in / fade out)
        self._chrome = 0.0                  # hover chrome opacity
        self._scroll = 0.0                  # roll-up animation, 1 -> 0
        self._outgoing: str | None = None
        self._drag: tuple[str, QPoint, QRect] | None = None
        self._hover = False
        self._hot_button: str | None = None

        self._fade = _anim(self, 180, self._on_fade)
        self._fade.finished.connect(self._fade_done)
        self._chrome_anim = _anim(self, CHROME_IN_MS, self._set_chrome)
        self._scroll_anim = _anim(self, SCROLL_MS, self._set_scroll)
        self._scroll_anim.finished.connect(self._scroll_done)
        self._topmost = QTimer(self, interval=2000, timeout=self._assert_topmost)
        self._topmost.start()
        # Enter/leave events can be missed when the window changes under a still cursor
        # (resize, unlock); a cheap poll keeps the hover state honest.
        self._hover_poll = QTimer(self, interval=150, timeout=self._poll_hover)
        self._hover_poll.start()
        self._apply_geometry(initial=True)
        self._apply_lock()

    # -- public ----------------------------------------------------------------------
    def apply(self, appearance: Appearance, layout: Layout) -> None:
        lock_changed = layout.locked != self.layout_cfg.locked
        self.ap, self.layout_cfg = appearance, layout
        self._apply_geometry()
        if lock_changed:
            self._apply_lock()
        self.update()

    def set_status(self, text: str) -> None:
        """The placeholder text shown on hover while there are no subtitles."""
        if text != self.status:
            self.status = text
            if not self.lines:
                self.update()

    def set_lines(self, lines: list[str]) -> None:
        fading = self._fade.state() == QVariantAnimation.State.Running and self._fade.endValue() == 0.0
        if not lines:
            if self.lines and self.ap.fade_ms > 0:
                if not fading:   # fade out what is showing; _fade_done clears it
                    self._start_fade(1.0 if self._opacity >= 1.0 else self._opacity, 0.0, self.ap.fade_ms)
                return
            self.lines = []
            self.update()
            return
        if lines == self.lines and not fading:
            return
        appearing = not self.lines or fading
        old = self.lines
        self.lines = list(lines)
        if appearing:
            self._outgoing = None
            self._scroll_anim.stop()
            self._scroll = 0.0
            self._start_fade(0.0 if not old else self._opacity, 1.0, FADE_IN_MS)
        else:
            self._fade.stop()
            self._opacity = 1.0
            if self._rolled(old, self.lines):
                self._outgoing = old[0]
                self._scroll_anim.stop()
                self._scroll_anim.setStartValue(1.0)
                self._scroll_anim.setEndValue(0.0)
                self._scroll_anim.start()
        self.update()

    def snap_to_default(self) -> None:
        self.layout_cfg.x = self.layout_cfg.y = None
        self._apply_geometry()
        self._emit_geometry()

    # -- animation -------------------------------------------------------------------
    @staticmethod
    def _rolled(old: list[str], new: list[str]) -> bool:
        """True when the top line scrolled away: the old second line is now the first."""
        if not old or not new or old[0] == new[0]:
            return False
        if len(old) >= 2:
            return new[0] == old[1]
        return not new[0].startswith(old[0])   # one-line overlay: a new line replaced the old

    def _start_fade(self, start: float, end: float, ms: int) -> None:
        self._fade.stop()
        self._fade.setDuration(max(1, ms))
        self._fade.setStartValue(start)
        self._fade.setEndValue(end)
        self._opacity = start
        self._fade.start()

    def _on_fade(self, value) -> None:
        self._opacity = float(value)
        self.update()

    def _fade_done(self) -> None:
        if self._opacity <= 0.0:
            self.lines = []
            self._outgoing = None
            self._opacity = 1.0
            self.update()

    def _set_scroll(self, value) -> None:
        self._scroll = float(value)
        self.update()

    def _scroll_done(self) -> None:
        self._outgoing = None
        self._scroll = 0.0
        self.update()

    def _set_chrome(self, value) -> None:
        self._chrome = float(value)
        self.update()

    def _set_hover(self, hover: bool) -> None:
        hover = hover and not self.layout_cfg.locked
        if hover == self._hover:
            return
        self._hover = hover
        if not hover:
            self._hot_button = None
        self._chrome_anim.stop()
        self._chrome_anim.setDuration(CHROME_IN_MS if hover else CHROME_OUT_MS)
        self._chrome_anim.setStartValue(self._chrome)
        self._chrome_anim.setEndValue(1.0 if hover else 0.0)
        self._chrome_anim.start()

    def _poll_hover(self) -> None:
        if not self.isVisible() or self._drag is not None:
            return
        self._set_hover(self.frameGeometry().contains(QCursor.pos()))

    # -- geometry --------------------------------------------------------------------
    def _apply_geometry(self, initial: bool = False) -> None:
        lay = self.layout_cfg
        width = max(MIN_WIDTH, lay.width)
        height = block_height(self.ap, lay.max_lines)
        if lay.x is None or lay.y is None or not self._on_some_screen(lay.x, lay.y, width, height):
            screen = QGuiApplication.primaryScreen().availableGeometry()
            x = screen.center().x() - width // 2
            y = screen.bottom() - height - int(screen.height() * 0.06)
        elif initial:
            x, y = lay.x, lay.y
        else:
            # Keep the bottom edge where it is when the height changes (subtitles grow upward).
            x, y = self.x(), self.geometry().bottom() + 1 - height
        self.setGeometry(x, y, width, height)

    @staticmethod
    def _on_some_screen(x: int, y: int, w: int, h: int) -> bool:
        target = QRect(x, y, w, h)
        return any(s.geometry().intersects(target) for s in QGuiApplication.screens())

    def _emit_geometry(self) -> None:
        self.layout_cfg.x, self.layout_cfg.y, self.layout_cfg.width = self.x(), self.y(), self.width()
        self.geometry_changed.emit(self.x(), self.y(), self.width())

    def _apply_lock(self) -> None:
        visible = self.isVisible()
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, self.layout_cfg.locked)
        if visible:
            self.show()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        if self.layout_cfg.locked:
            self._set_hover(False)

    def _assert_topmost(self) -> None:
        # Other always-on-top windows (and some players going fullscreen) can cover us.
        if sys.platform == "win32" and self.isVisible():
            HWND_TOPMOST, flags = -1, 0x0001 | 0x0002 | 0x0010   # NOSIZE | NOMOVE | NOACTIVATE
            ctypes.windll.user32.SetWindowPos(int(self.winId()), HWND_TOPMOST, 0, 0, 0, 0, flags)

    # -- painting --------------------------------------------------------------------
    def _button_rects(self) -> dict[str, QRectF]:
        gap, n = 4, len(BUTTONS)
        x = self.width() - 8 - n * BUTTON - (n - 1) * gap
        return {bid: QRectF(x + i * (BUTTON + gap), 6, BUTTON, BUTTON) for i, (bid, _, _) in enumerate(BUTTONS)}

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        rect = QRectF(self.rect())
        if not self.layout_cfg.locked:
            # Invisible, but makes the whole box take the mouse (fully transparent pixels
            # of a layered window are click-through, so hover could never be detected).
            p.fillRect(rect, QColor(0, 0, 0, 1))
        if self._chrome > 0:
            self._paint_chrome(p, rect)
        if self.lines:
            paint_captions(p, rect, self.lines, self.ap, self._opacity, self._scroll, self._outgoing)
        elif self._chrome > 0:
            self._paint_placeholder(p, rect)
        p.end()

    def _paint_placeholder(self, p: QPainter, rect: QRectF) -> None:
        """The status in the caption style, with a small usage hint under it (room permitting)."""
        if self.layout_cfg.max_lines < 2:
            paint_captions(p, rect, [self.status], self.ap, 0.8 * self._chrome)
            return
        paint_captions(p, rect, [self.status, ""], self.ap, 0.8 * self._chrome)
        hint = "Drag to move  ·  drag the sides to resize  ·  Ctrl+scroll for text size"
        hint_font = QFont("Segoe UI")
        hint_font.setPixelSize(max(12, min(18, int(self.ap.font_size * 0.38))))
        p.save()
        p.setFont(hint_font)
        p.setOpacity(self._chrome * 0.85)
        p.setPen(QColor(235, 235, 235))
        r = rect.adjusted(EDGE + 8, 0, -(EDGE + 8), -10)
        p.drawText(r, int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom), hint)
        p.restore()

    def _paint_chrome(self, p: QPainter, rect: QRectF) -> None:
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setOpacity(self._chrome)
        frame = rect.adjusted(1, 1, -1, -1)
        # A soft scrim so the bounds read on bright and dark scenes alike.
        p.setBrush(QColor(12, 12, 16, 70))
        p.setPen(QPen(QColor(255, 255, 255, 120), 1.2))
        p.drawRoundedRect(frame, 10, 10)
        # Resize grips.
        grip = QColor(255, 255, 255, 170)
        cy = rect.height() / 2
        for gx in (5, rect.width() - 8):
            path = QPainterPath()
            path.addRoundedRect(QRectF(gx, cy - 16, 3, 32), 1.5, 1.5)
            p.fillPath(path, grip)
        # Toolbar.
        icon_font = QFont(ICON_FONTS[0])
        icon_font.setFamilies(list(ICON_FONTS))
        icon_font.setPixelSize(13)
        p.setFont(icon_font)
        for (bid, glyph, _), r in zip(BUTTONS, self._button_rects().values()):
            hot = bid == self._hot_button
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 60) if hot else QColor(20, 20, 24, 170))
            p.drawRoundedRect(r, 6, 6)
            p.setPen(QColor(255, 255, 255, 240))
            p.drawText(r, int(Qt.AlignmentFlag.AlignCenter), glyph)
        p.restore()

    # -- mouse -----------------------------------------------------------------------
    def _button_at(self, pos: QPointF) -> str | None:
        if self._chrome <= 0.2:
            return None
        for bid, r in self._button_rects().items():
            if r.contains(pos):
                return bid
        return None

    def _zone(self, pos: QPoint) -> str:
        if pos.x() <= EDGE:
            return "left"
        if pos.x() >= self.width() - EDGE:
            return "right"
        return "move"

    def enterEvent(self, event) -> None:
        self._set_hover(True)

    def leaveEvent(self, event) -> None:
        if self._drag is None:
            self._set_hover(False)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self.layout_cfg.locked:
            return
        button = self._button_at(event.position())
        if button:
            {"lock": lambda: self.lock_requested.emit(True), "settings": self.settings_requested.emit,
             "hide": self.hide_requested.emit}[button]()
            return
        self._drag = (self._zone(event.position().toPoint()), event.globalPosition().toPoint(), self.geometry())

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        if self._drag is None:
            self._set_hover(True)
            button = self._button_at(event.position())
            if button != self._hot_button:
                self._hot_button = button
                self.setToolTip(next((t for b, _, t in BUTTONS if b == button), ""))
                self.update()
            if button:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                zone = self._zone(pos)
                self.setCursor(Qt.CursorShape.SizeHorCursor if zone != "move" else Qt.CursorShape.SizeAllCursor)
            return
        zone, start, geo = self._drag
        delta = event.globalPosition().toPoint() - start
        if zone == "move":
            self.move(geo.topLeft() + delta)
        elif zone == "right":
            self.resize(max(MIN_WIDTH, geo.width() + delta.x()), geo.height())
        else:
            w = max(MIN_WIDTH, geo.width() - delta.x())
            self.setGeometry(geo.right() + 1 - w, geo.y(), w, geo.height())

    def mouseReleaseEvent(self, event) -> None:
        if self._drag is not None:
            self._drag = None
            self._emit_geometry()
            self._poll_hover()

    def mouseDoubleClickEvent(self, event) -> None:
        if self._button_at(event.position()):
            return
        screen = (self.screen() or QGuiApplication.primaryScreen()).availableGeometry()
        self.move(screen.center().x() - self.width() // 2,
                  screen.bottom() - self.height() - int(screen.height() * 0.06))
        self._emit_geometry()

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            step = 2 if event.angleDelta().y() > 0 else -2
            self.font_size_changed.emit(max(12, min(160, self.ap.font_size + step)))

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        menu.addAction("Lock in place (click-through)", lambda: self.lock_requested.emit(True))
        menu.addAction("Snap to bottom centre", self.snap_to_default)
        menu.addAction("Hide subtitles", self.hide_requested.emit)
        menu.addSeparator()
        menu.addAction("Settings…", self.settings_requested.emit)
        menu.exec(QCursor.pos())
