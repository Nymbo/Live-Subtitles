"""Media mode's "What is this?" window: pick the title by hand, remembered per site."""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QPushButton, QSpinBox, QVBoxLayout)

from . import titles
from .secrets import unprotect
from .tmdb import Match, Tmdb

logger = logging.getLogger(__name__)


class _Relay(QObject):
    results = Signal(object)
    failed = Signal(str)
    picked = Signal(object)


class IdentifyDialog(QDialog):
    """Search TMDB, pick the movie or show (and episode), and remember it for this page."""
    chosen = Signal(str, object)            # window title, Match

    def __init__(self, media_settings, window_title: str, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("What is this?")
        self.resize(560, 460)
        self.window_title = window_title
        self.tmdb = Tmdb(unprotect(media_settings.tmdb_key))
        self.relay = _Relay()
        self.relay.results.connect(self._show)
        self.relay.failed.connect(lambda m: self.note.setText(m))
        self.relay.picked.connect(self._done)

        col = QVBoxLayout(self)
        col.addWidget(QLabel(f"Playing: <b>{titles.strip_app(window_title) or '(no title)'}</b>"))
        row = QHBoxLayout()
        q = titles.parse(window_title)
        self.query = QLineEdit(q.title if q else titles.strip_app(window_title))
        self.query.returnPressed.connect(self._search)
        go = QPushButton("Search")
        go.clicked.connect(self._search)
        row.addWidget(self.query, 1)
        row.addWidget(go)
        col.addLayout(row)
        self.list = QListWidget()
        self.list.currentItemChanged.connect(lambda *_: self._sync_episode())
        self.list.itemDoubleClicked.connect(lambda *_: self._accept())
        col.addWidget(self.list, 1)
        form = QFormLayout()
        self.season, self.episode = QSpinBox(), QSpinBox()
        self.season.setRange(0, 99)
        self.episode.setRange(1, 9999)
        self.season.setValue(q.season if q and q.season is not None else 1)
        self.episode.setValue(q.episode if q and q.episode else 1)
        ep = QHBoxLayout()
        ep.addWidget(QLabel("Season"))
        ep.addWidget(self.season)
        ep.addWidget(QLabel("Episode"))
        ep.addWidget(self.episode)
        ep.addStretch()
        form.addRow(ep)
        col.addLayout(form)
        self.note = QLabel("The answer is remembered for this site, so it is only asked once.")
        self.note.setWordWrap(True)
        col.addWidget(self.note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        col.addWidget(buttons)
        self._sync_episode()
        if self.tmdb.key:
            self._search()
        else:
            self.note.setText("Add a TMDB key in Settings → Media first.")

    def _search(self) -> None:
        text = self.query.text().strip()
        if not text or not self.tmdb.key:
            return
        self.note.setText("Searching…")

        def run():
            try:
                self.relay.results.emit(self.tmdb.search(text))
            except Exception as exc:
                self.relay.failed.emit(f"Search failed: {exc}")

        threading.Thread(target=run, daemon=True).start()

    def _show(self, results: list[dict]) -> None:
        self.list.clear()
        for r in results:
            label = f"{r['title']} ({r['year']})" if r["year"] else r["title"]
            item = QListWidgetItem(f"{label}   ·   {'TV show' if r['kind'] == 'tv' else 'Movie'}")
            item.setToolTip(r["overview"])
            item.setData(Qt.ItemDataRole.UserRole, r)
            self.list.addItem(item)
        if results:
            self.list.setCurrentRow(0)
        self.note.setText("" if results else "Nothing found. Try the title as it's usually written.")

    def _sync_episode(self) -> None:
        item = self.list.currentItem()
        tv = bool(item and item.data(Qt.ItemDataRole.UserRole)["kind"] == "tv")
        self.season.setEnabled(tv)
        self.episode.setEnabled(tv)

    def _accept(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        r = item.data(Qt.ItemDataRole.UserRole)
        season, episode = self.season.value(), self.episode.value()
        self.note.setText("Looking it up…")

        def run():
            try:
                kind = "movie" if r["kind"] == "movie" else "episode"
                self.relay.picked.emit(self.tmdb.lookup(kind, r["tmdb_id"], season, episode))
            except Exception as exc:
                self.relay.failed.emit(f"Lookup failed: {exc}")

        threading.Thread(target=run, daemon=True).start()

    def _done(self, match: Match) -> None:
        self.chosen.emit(self.window_title, match)
        self.accept()

