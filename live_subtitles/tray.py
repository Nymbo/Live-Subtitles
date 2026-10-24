"""Tray icon and menu. Click toggles captions, double-click opens settings, right-click is the menu."""

from __future__ import annotations

from PySide6.QtCore import QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

STATE_COLORS = {"off": "#8A8F98", "loading": "#E0A526", "live": "#2EB872", "error": "#E0483E"}


def make_icon(state: str) -> QIcon:
    icon = QIcon()
    for size in (16, 20, 24, 32, 48, 64):
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(STATE_COLORS.get(state, STATE_COLORS["off"])))
        r = QRectF(0.5, size * 0.16, size - 1, size * 0.68)
        p.drawRoundedRect(r, size * 0.16, size * 0.16)
        font = QFont("Segoe UI")
        font.setPixelSize(max(7, int(size * 0.42)))
        font.setBold(True)
        p.setFont(font)
        p.setPen(QColor("white"))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, "CC")
        p.end()
        icon.addPixmap(pm)
    return icon


class Tray(QObject):
    toggle_requested = Signal()
    lock_toggled = Signal(bool)
    visibility_toggled = Signal(bool)
    snap_requested = Signal()
    transcribe_requested = Signal()
    media_toggled = Signal(bool)
    identify_requested = Signal()
    jobs_requested = Signal()
    settings_requested = Signal()
    logs_requested = Signal()
    quit_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.icons = {s: make_icon(s) for s in STATE_COLORS}
        self.tray = QSystemTrayIcon(self.icons["off"])
        self.menu = QMenu()
        self.header = self.menu.addAction("Live Subtitles")
        self.header.setEnabled(False)
        self.menu.addSeparator()
        self.toggle_action = self.menu.addAction("Start live subtitles", self.toggle_requested.emit)
        self.visible_action = self._check("Show subtitles", self.visibility_toggled.emit)
        self.lock_action = self._check("Lock overlay (click-through)", self.lock_toggled.emit)
        self.menu.addAction("Snap overlay to bottom centre", self.snap_requested.emit)
        self.menu.addSeparator()
        self.media_action = self._check("Media subtitles for movies and TV", self.media_toggled.emit)
        self.menu.addAction("What is this?…", self.identify_requested.emit)
        self.menu.addSeparator()
        self.menu.addAction("Create subtitles for a video…", self.transcribe_requested.emit)
        self.jobs_action = self.menu.addAction("Subtitle jobs…", self.jobs_requested.emit)
        self.menu.addSeparator()
        self.menu.addAction("Settings…", self.settings_requested.emit)
        self.menu.addAction("Open log folder", self.logs_requested.emit)
        self.menu.addSeparator()
        self.menu.addAction("Exit", self.quit_requested.emit)
        self.tray.setContextMenu(self.menu)
        self.tray.activated.connect(self._activated)
        self.tray.setToolTip("Live Subtitles: off")
        self._click_timer = QTimer(self, singleShot=True, timeout=self.toggle_requested.emit)

    def _check(self, text: str, slot) -> QAction:
        action = QAction(text, self.menu, checkable=True)
        action.toggled.connect(slot)
        self.menu.addAction(action)
        return action

    def show(self) -> None:
        self.tray.show()

    def hide(self) -> None:
        self.tray.hide()

    def set_state(self, state: str, detail: str = "") -> None:
        self.tray.setIcon(self.icons.get(state, self.icons["off"]))
        label = {"off": "off", "loading": "starting", "live": "live", "error": "error"}.get(state, state)
        self.tray.setToolTip(f"Live Subtitles: {label}" + (f"\n{detail}" if detail else ""))
        self.toggle_action.setText("Stop live subtitles" if state in ("live", "loading")
                                   else "Start live subtitles")

    def set_flags(self, visible: bool, locked: bool, media: bool | None = None) -> None:
        pairs = [(self.visible_action, visible), (self.lock_action, locked)]
        if media is not None:
            pairs.append((self.media_action, media))
        for action, value in pairs:
            action.blockSignals(True)
            action.setChecked(value)
            action.blockSignals(False)

    def notify(self, title: str, message: str, error: bool = False) -> None:
        icon = QSystemTrayIcon.MessageIcon.Warning if error else QSystemTrayIcon.MessageIcon.Information
        self.tray.showMessage(title, message, icon, 6000)

    def _activated(self, reason) -> None:
        # Windows sends Trigger before DoubleClick, so hold the single click until we know.
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._click_timer.start(QApplication.doubleClickInterval())
        elif reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._click_timer.stop()
            self.settings_requested.emit()
