"""Subtitle jobs: runs `python -m live_subtitles transcribe --json` in a hidden child process.

One process per batch, so Parakeet loads once for all the files picked together. Batches
run one at a time; progress arrives as JSON lines on the child's stdout.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QProgressBar, QPushButton, QScrollArea, QVBoxLayout,
                               QWidget)

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
VIDEO_FILTER = ("Video or audio (*.mkv *.mp4 *.m4v *.avi *.mov *.webm *.wmv *.flv *.ts *.m2ts *.mpg "
                "*.mpeg *.mp3 *.m4a *.aac *.flac *.wav *.ogg *.opus);;All files (*)")


def python_exe() -> str:
    """The venv's console python (pythonw has no usable stdout for progress)."""
    exe = Path(sys.executable)
    console = exe.with_name("python.exe")
    return str(console if console.exists() else exe)


class Batch(QObject):
    event = Signal(dict)
    finished = Signal(int)

    def __init__(self, files: list[str]):
        super().__init__()
        self.files = files
        self.proc: subprocess.Popen | None = None
        self.cancelled = False

    def start(self) -> None:
        screen = QGuiApplication.primaryScreen()
        height = screen.geometry().height() if screen else 1080
        cmd = [python_exe(), "-m", "live_subtitles", "transcribe", "--json", "--ref-height", str(height),
               *self.files]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        self.proc = subprocess.Popen(cmd, cwd=PROJECT_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, encoding="utf-8", errors="replace", env=env,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def cancel(self) -> None:
        self.cancelled = True
        if self.proc and self.proc.poll() is None:
            self.proc.kill()

    def _read(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.strip()
            if line.startswith("{"):
                try:
                    self.event.emit(json.loads(line))
                except json.JSONDecodeError:
                    pass
        self.finished.emit(self.proc.wait())

    def _drain_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        for line in self.proc.stderr:
            if line.strip():
                logger.info("[transcribe] %s", line.rstrip())


class JobRow(QWidget):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.outputs: list[str] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        top = QHBoxLayout()
        name = QLabel(Path(path).name)
        name.setToolTip(path)
        name.setStyleSheet("font-weight: 600")
        self.status = QLabel("Waiting")
        self.open_btn = QPushButton("Open folder")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open)
        top.addWidget(name, 1)
        top.addWidget(self.status)
        top.addWidget(self.open_btn)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        lay.addLayout(top)
        lay.addWidget(self.bar)

    def _open(self) -> None:
        target = self.outputs[0] if self.outputs else self.path
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", os.path.normpath(target)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(target).parent)))


class JobsWindow(QWidget):
    notify = Signal(str, str, bool)          # title, message, is_error

    def __init__(self):
        super().__init__(None)
        self.setWindowTitle("Live Subtitles: subtitle jobs")
        self.resize(620, 360)
        self.rows: dict[str, JobRow] = {}
        self.queue: list[Batch] = []
        self.current: Batch | None = None
        root = QVBoxLayout(self)
        self.header = QLabel("Nothing queued.")
        root.addWidget(self.header)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self.list = QVBoxLayout(inner)
        self.list.addStretch()
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)
        buttons = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel current")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        clear = QPushButton("Clear finished")
        clear.clicked.connect(self._clear_finished)
        buttons.addStretch()
        buttons.addWidget(clear)
        buttons.addWidget(self.cancel_btn)
        root.addLayout(buttons)

    def add(self, files: list[str]) -> None:
        files = [f for f in files if f]
        if not files:
            return
        for f in files:
            row = JobRow(f)
            self.rows[os.path.normcase(os.path.abspath(f))] = row
            self.list.insertWidget(self.list.count() - 1, row)
        batch = Batch(files)
        batch.event.connect(self._on_event)
        batch.finished.connect(lambda code, b=batch: self._on_finished(b, code))
        self.queue.append(batch)
        self._pump()
        self.show()
        self.raise_()

    def busy(self) -> bool:
        return self.current is not None

    def _row(self, path: str) -> JobRow | None:
        return self.rows.get(os.path.normcase(os.path.abspath(path)))

    def _pump(self) -> None:
        if self.current is None and self.queue:
            self.current = self.queue.pop(0)
            self.cancel_btn.setEnabled(True)
            self.header.setText("Starting…")
            try:
                self.current.start()
            except Exception as exc:
                logger.exception("Could not start transcription")
                self.notify.emit("Subtitles failed", str(exc), True)
                self.current = None
                self._pump()

    def _on_event(self, ev: dict) -> None:
        kind = ev.get("event")
        row = self._row(ev["file"]) if "file" in ev else None
        if kind == "status":
            self.header.setText(ev.get("message", ""))
        elif kind == "start" and row:
            row.status.setText("Transcribing")
            self.header.setText(f"File {ev['index']} of {ev['total']}")
        elif kind == "progress" and row:
            row.bar.setValue(int(ev["fraction"] * 1000))
            row.status.setText(f"{ev['fraction'] * 100:.0f}%")
        elif kind == "done" and row:
            row.bar.setValue(1000)
            row.outputs = ev.get("outputs", [])
            row.status.setText(f"Done · {ev.get('cues', 0)} subtitles · {ev.get('seconds', 0):.0f}s")
            row.open_btn.setEnabled(True)
            self.notify.emit("Subtitles ready", Path(row.outputs[0]).name if row.outputs else row.path, False)
        elif kind == "error":
            msg = ev.get("message", "Unknown error")
            if row:
                row.status.setText("Failed")
                row.setToolTip(msg)
            self.header.setText(msg)
            self.notify.emit("Subtitles failed", msg, True)

    def _on_finished(self, batch: Batch, code: int) -> None:
        for f in batch.files:
            row = self._row(f)
            unfinished = row is not None and not row.outputs and row.status.text() != "Failed"
            if unfinished:
                row.status.setText("Cancelled" if batch.cancelled else "Stopped")
        if batch is self.current:
            self.current = None
        self.cancel_btn.setEnabled(False)
        self.header.setText("All done." if not self.queue else "Next batch…")
        self._pump()

    def _cancel(self) -> None:
        if self.current:
            self.current.cancel()

    def _clear_finished(self) -> None:
        busy = set()
        for b in ([self.current] if self.current else []) + self.queue:
            busy.update(os.path.normcase(os.path.abspath(f)) for f in b.files)
        for key in [k for k in self.rows if k not in busy]:
            self.rows.pop(key).deleteLater()
