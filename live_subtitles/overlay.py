"""The on-screen subtitle window: frameless, translucent, always on top.

Unlocked: drag to move, drag either side to resize, Ctrl+wheel to change the text size,
double-click to snap to the bottom centre, right-click for a menu. Locked: fully
click-through, so it never steals clicks from the video underneath.
"""

from __future__ import annotations

import ctypes
import sys

from PySide6.QtCore import QPoint, QRect, QRectF, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QCursor, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QMenu, QWidget

from .renderer import block_height, paint_captions
from .settings import Appearance, Layout

EDGE = 12           # px from the left/right edge that resizes instead of moving
MIN_WIDTH = 240
PLACEHOLDER = ["Subtitles appear here", "Drag to move · drag the sides to resize"]


class CaptionOverlay(QWidget):
    geometry_changed = Signal(int, int, int)      # x, y, width
    font_size_changed = Signal(int)
    lock_requested = Signal(bool)
    hide_requested = Signal()
    settings_requested = Signal()

    def __init__(self, appearance: Appearance, layout: Layout):
        super().__init__(None)
        self.setWindowTitle("Live-Subtitles overlay")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setMouseTracking(True)
        self.ap = appearance
        self.layout_cfg = layout
        self.lines: list[str] = []
        self._opacity = 1.0
        self._drag: tuple[str, QPoint, QRect] | None = None
        self._hover = False
        self._fade = QVariantAnimation(self)
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        self._fade.valueChanged.connect(self._on_fade)
        self._fade.finished.connect(self._fade_done)
        self._topmost = QTimer(self, interval=2000, timeout=self._assert_topmost)
        self._topmost.start()
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

    def set_lines(self, lines: list[str]) -> None:
        fading = self._fade.state() == QVariantAnimation.State.Running
        if not lines:
            if self.lines and self.ap.fade_ms > 0:
                if not fading:   # fade out what is showing; _fade_done clears it
                    self._fade.setDuration(self.ap.fade_ms)
                    self._fade.start()
                return
        elif lines == self.lines and not fading:
            return
        self._fade.stop()
        self._opacity = 1.0
        self.lines = list(lines)
        self.update()

    def snap_to_default(self) -> None:
        self.layout_cfg.x = self.layout_cfg.y = None
        self._apply_geometry()
        self._emit_geometry()

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

    def _assert_topmost(self) -> None:
        # Other always-on-top windows (and some players going fullscreen) can cover us.
        if sys.platform == "win32" and self.isVisible():
            HWND_TOPMOST, flags = -1, 0x0001 | 0x0002 | 0x0010   # NOSIZE | NOMOVE | NOACTIVATE
            ctypes.windll.user32.SetWindowPos(int(self.winId()), HWND_TOPMOST, 0, 0, 0, 0, flags)

    # -- painting --------------------------------------------------------------------
    def paintEvent(self, event) -> None:
        p = QPainter(self)
        rect = QRectF(self.rect())
        unlocked = not self.layout_cfg.locked
        if unlocked:
            # A faint frame (and a near-invisible fill so the whole box takes the mouse).
            p.fillRect(rect, QColor(0, 0, 0, 1))
            if self._hover or not self.lines:
                pen = QPen(QColor(255, 255, 255, 110), 1, Qt.PenStyle.DashLine)
                p.setPen(pen)
                p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 8, 8)
                p.fillRect(QRectF(2, rect.height() / 2 - 14, 3, 28), QColor(255, 255, 255, 120))
                p.fillRect(QRectF(rect.width() - 5, rect.height() / 2 - 14, 3, 28), QColor(255, 255, 255, 120))
        if self.lines:
            paint_captions(p, rect, self.lines, self.ap, self._opacity)
        elif unlocked:
            sample = PLACEHOLDER[-self.layout_cfg.max_lines:] if self.layout_cfg.max_lines < 2 else PLACEHOLDER
            paint_captions(p, rect, sample, self.ap, 0.55)
        p.end()

    def _on_fade(self, value) -> None:
        self._opacity = float(value)
        self.update()

    def _fade_done(self) -> None:
        if self._opacity <= 0.0:
            self.lines = []
            self._opacity = 1.0
            self.update()

    # -- mouse -----------------------------------------------------------------------
    def _zone(self, pos: QPoint) -> str:
        if pos.x() <= EDGE:
            return "left"
        if pos.x() >= self.width() - EDGE:
            return "right"
        return "move"

    def enterEvent(self, event) -> None:
        self._hover = True
        self.update()

    def leaveEvent(self, event) -> None:
        self._hover = False
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and not self.layout_cfg.locked:
            self._drag = (self._zone(event.position().toPoint()), event.globalPosition().toPoint(),
                          self.geometry())

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        if self._drag is None:
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

    def mouseDoubleClickEvent(self, event) -> None:
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
