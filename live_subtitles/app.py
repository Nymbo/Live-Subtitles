"""Tray application: wires capture -> live recogniser -> caption builder -> overlay."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from logging.handlers import RotatingFileHandler

from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QFileDialog

from . import APP_NAME, DISPLAY_NAME, __version__, models
from . import settings as settings_mod
from .capture import LoopbackCapture, list_output_devices
from .captions import CaptionBuilder
from .hotkeys import HotkeyManager
from .jobs import VIDEO_FILTER, JobsWindow
from .engines import NemotronEngine
from .live import LiveRecognizer
from .overlay import CaptionOverlay
from .paths import LOG_DIR, ensure_dirs
from .renderer import text_width_fn
from .tray import Tray

logger = logging.getLogger(__name__)
INSTANCE_KEY = f"{APP_NAME}-{os.environ.get('USERNAME', 'user')}"


class LiveController(QObject):
    """Starts/stops capture + recogniser off the UI thread. Signals arrive on the UI thread."""

    words = Signal(list, list)
    state = Signal(str, str)              # off | loading | live | error, detail

    def __init__(self, settings: settings_mod.Settings):
        super().__init__()
        self.s = settings
        self.capture: LoopbackCapture | None = None
        self.recognizer: LiveRecognizer | None = None
        self.running = False
        self.punctuator = None
        self.engine_label = ""                # "GPU" / "CPU", for the status line
        self._gen = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._gen += 1
        gen = self._gen
        self.state.emit("loading", "Starting…")
        threading.Thread(target=self._start, args=(gen,), daemon=True, name="live-start").start()

    def stop(self) -> None:
        self._gen += 1
        self.running = False
        self._teardown()
        self.state.emit("off", "")

    def restart(self) -> None:
        if self.running:
            self.stop()
            self.start()

    def _teardown(self) -> None:
        with self._lock:
            if self.recognizer:
                self.recognizer.stop()
                self.recognizer = None
            if self.capture:
                self.capture.stop()
                self.capture = None

    def _start(self, gen: int) -> None:
        eng = self.s.engine
        try:
            engine, punctuator, note = None, None, ""
            if eng.language == "en" and eng.recognizer in ("auto", "gpu"):
                try:
                    engine = self._gpu_engine(gen)
                except Exception as exc:
                    if eng.recognizer == "gpu":
                        raise
                    logger.warning("GPU recogniser unavailable, using the CPU: %s", exc)
                    note = "GPU unavailable, using the CPU"
            if gen != self._gen:
                if engine:
                    engine.close()
                return
            if engine is None:
                engine, punctuator = self._cpu_engine(gen)
            self.engine_label = "GPU" if engine.__class__.__name__ == "UnifiedEngine" else "CPU"
            if gen != self._gen:
                getattr(engine, "close", lambda: None)()
                return
            self.punctuator = punctuator
            capture = LoopbackCapture(eng.device, eng.dialogue_boost)
            where = note or self.engine_label         # device_label is known once capture starts
            recognizer = LiveRecognizer(
                engine, capture.chunks,
                on_words=lambda c, p, _phrase: gen == self._gen and self.words.emit(c, p),
                on_ready=lambda: gen == self._gen and self.state.emit(
                    "live", f"Listening to {capture.device_label}  ·  {where}"),
                on_error=lambda m: gen == self._gen and self._fail(m),
            )
            with self._lock:
                if gen != self._gen:
                    getattr(engine, "close", lambda: None)()
                    return
                capture.start()
                self.capture, self.recognizer = capture, recognizer
                recognizer.start()
        except Exception as exc:
            logger.exception("Could not start live subtitles")
            if gen == self._gen:
                self._fail(str(exc))

    def _progress(self, gen: int, what: str):
        last = [-1]

        def progress(done, total):
            pct = int(done * 100 / total) if total else 0
            if pct != last[0] and gen == self._gen:
                last[0] = pct
                self.state.emit("loading", f"Downloading {what}… {pct}%")

        return progress

    def _gpu_engine(self, gen: int):
        """Parakeet-unified in its worker process; raises when there is no usable GPU."""
        from .unified import UnifiedEngine, cuda_available

        self.state.emit("loading", "Checking the GPU…")
        ok, why = cuda_available()
        if not ok:
            raise RuntimeError(why)
        if not models.UNIFIED_MODEL.is_installed():
            models.download(models.UNIFIED_MODEL, self._progress(gen, "the GPU speech model (2.5 GB)"))
        self.state.emit("loading", "Loading the speech model on the GPU…")
        engine = UnifiedEngine()
        engine.load()
        return engine

    def _cpu_engine(self, gen: int):
        eng = self.s.engine
        model = models.live_model(eng.language, eng.latency_ms)
        if not model.is_installed():
            models.download(model, self._progress(gen, "speech model"))
        punctuator = None
        if eng.punctuation and eng.language == "en":
            try:
                if not models.PUNCT_MODEL.is_installed():
                    self.state.emit("loading", "Downloading punctuation model…")
                    models.download(models.PUNCT_MODEL)
                from .punctuate import Punctuator

                punctuator = Punctuator()
            except Exception:
                # Captions still work with the speech model's own punctuation.
                logger.exception("Punctuation model unavailable")
        self.state.emit("loading", "Loading speech model…")
        return NemotronEngine(model, eng.threads), punctuator

    def _fail(self, message: str) -> None:
        self.running = False
        self._teardown()
        self.state.emit("error", message)


class App(QObject):
    def __init__(self, qapp: QApplication):
        super().__init__()
        self.qapp = qapp
        self.s = settings_mod.load()
        self.live = LiveController(self.s)
        self.builder = CaptionBuilder()
        self.overlay = CaptionOverlay(self.s.appearance, self.s.layout)
        self.tray = Tray()
        self.hotkeys = HotkeyManager()
        qapp.installNativeEventFilter(self.hotkeys)
        self.settings_window = None
        self.jobs = JobsWindow()
        self.jobs.notify.connect(self.tray.notify)

        self._save_timer = QTimer(self, singleShot=True, interval=600, timeout=self._save)
        self._restart_timer = QTimer(self, singleShot=True, interval=900, timeout=self.live.restart)
        self._tick = QTimer(self, interval=100, timeout=self._on_tick)

        self.live.words.connect(self._on_words)
        self.live.state.connect(self._on_state)
        self.overlay.geometry_changed.connect(lambda *_: self._layout_changed_from_overlay())
        self.overlay.font_size_changed.connect(self._set_font_size)
        self.overlay.lock_requested.connect(self.set_locked)
        self.overlay.hide_requested.connect(lambda: self.set_visible(False))
        self.overlay.settings_requested.connect(self.show_settings)
        t = self.tray
        t.toggle_requested.connect(self.toggle_live)
        t.lock_toggled.connect(self.set_locked)
        t.visibility_toggled.connect(self.set_visible)
        t.snap_requested.connect(self.overlay.snap_to_default)
        t.transcribe_requested.connect(self.pick_videos)
        t.jobs_requested.connect(lambda: (self.jobs.show(), self.jobs.raise_()))
        t.settings_requested.connect(self.show_settings)
        t.logs_requested.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(LOG_DIR))))
        t.quit_requested.connect(self.quit)

    # -- lifecycle -------------------------------------------------------------------
    def run(self) -> None:
        self._configure_builder()
        self.tray.set_flags(self.s.layout.visible, self.s.layout.locked)
        self.overlay.set_status(f"Live subtitles are off  ·  {self.s.hotkeys.toggle} to start")
        self.tray.show()
        if self.s.layout.visible:
            self.overlay.show()
        self._register_hotkeys()
        self._tick.start()
        if self.s.engine.autostart:
            self.live.start()
        else:
            self.tray.notify(DISPLAY_NAME, f"Ready. Click the tray icon or press {self.s.hotkeys.toggle} to start.")

    def quit(self) -> None:
        self.live.stop()
        self.hotkeys.clear()
        if self.jobs.busy():
            self.jobs._cancel()
        self._save()
        self.tray.hide()
        self.qapp.quit()

    def _save(self) -> None:
        try:
            settings_mod.save(self.s)
        except Exception:
            logger.exception("Could not save settings")

    def _queue_save(self) -> None:
        self._save_timer.start()

    # -- live captions ---------------------------------------------------------------
    def toggle_live(self) -> None:
        if self.live.running:
            self.live.stop()
        else:
            self.live.start()

    def _on_words(self, committed: list, pending: list) -> None:
        self.builder.add(committed, pending, time.monotonic())
        self._show_lines()

    def _on_tick(self) -> None:
        if self.builder.tick(time.monotonic(), self.s.layout.clear_after_s):
            self._show_lines()

    def _show_lines(self) -> None:
        self.overlay.set_lines(self.builder.display_lines())

    def _on_state(self, state: str, detail: str) -> None:
        self.tray.set_state(state, detail)
        self.overlay.set_status({
            "off": f"Live subtitles are off  ·  {self.s.hotkeys.toggle} to start",
            "loading": detail or "Starting…",
            "live": detail or "Listening",
            "error": f"Stopped: {detail}"[:90],
        }.get(state, detail))
        if state == "error":
            self.tray.notify("Live subtitles stopped", detail, error=True)
        if state == "live":
            p = self.live.punctuator
            self.builder.configure(punctuate=p.punctuate if p else None)
        if state in ("off", "error"):
            self.builder.clear()
            self.builder.configure(punctuate=None)
            self._show_lines()
        if self.settings_window:
            self._update_model_status(detail if state == "loading" else "")
        logger.info("State %s %s", state, detail)

    # -- settings --------------------------------------------------------------------
    def _configure_builder(self) -> None:
        lay, ap = self.s.layout, self.s.appearance
        self.builder.configure(max_lines=lay.max_lines, max_chars=lay.max_chars, mode=lay.mode,
                               sentence_breaks=lay.sentence_breaks, hide_fillers=lay.hide_fillers,
                               fits=text_width_fn(ap, lay.width))

    def _on_settings_changed(self, section: str) -> None:
        if section in ("appearance", "layout"):
            self.overlay.apply(self.s.appearance, self.s.layout)
            self._configure_builder()
            self._show_lines()
            self.overlay.setVisible(self.s.layout.visible)
            self.tray.set_flags(self.s.layout.visible, self.s.layout.locked)
        elif section == "engine":
            self._update_model_status()
            if self.live.running:
                self._restart_timer.start()
        elif section == "hotkeys":
            self._register_hotkeys()
        self._queue_save()

    def _layout_changed_from_overlay(self) -> None:
        self._configure_builder()
        self._show_lines()
        if self.settings_window:
            self.settings_window.refresh()
        self._queue_save()

    def _set_font_size(self, size: int) -> None:
        self.s.appearance.font_size = size
        self._on_settings_changed("appearance")
        if self.settings_window:
            self.settings_window.refresh()

    def set_locked(self, locked: bool) -> None:
        self.s.layout.locked = locked
        self._on_settings_changed("layout")
        if self.settings_window:
            self.settings_window.refresh()

    def set_visible(self, visible: bool) -> None:
        self.s.layout.visible = visible
        self._on_settings_changed("layout")
        if self.settings_window:
            self.settings_window.refresh()

    def _register_hotkeys(self) -> None:
        h = self.s.hotkeys
        failed = self.hotkeys.set({
            "toggle": (h.toggle, self.toggle_live),
            "lock": (h.lock, lambda: self.set_locked(not self.s.layout.locked)),
            "visibility": (h.visibility, lambda: self.set_visible(not self.s.layout.visible)),
        })
        if self.settings_window:
            self.settings_window.set_hotkey_failures(failed)
        elif failed:
            self.tray.notify(DISPLAY_NAME, "Some hotkeys are taken by another program; change them in Settings.", True)

    def _update_model_status(self, detail: str = "") -> None:
        if not self.settings_window:
            return
        eng = self.s.engine
        gpu = eng.language == "en" and eng.recognizer in ("auto", "gpu")
        cpu = models.live_model(eng.language, eng.latency_ms)
        if detail:
            text = detail
        elif self.live.running and self.live.engine_label:
            text = f"Running on the {self.live.engine_label}."
        elif gpu and not models.UNIFIED_MODEL.is_installed():
            text = "The GPU model downloads (2.5 GB) the next time live subtitles start."
        elif not gpu and not cpu.is_installed():
            text = "This model downloads (~470 MB) the next time live subtitles start."
        else:
            text = f"Model installed: {models.UNIFIED_MODEL.name if gpu else cpu.name}"
        self.settings_window.set_model_status(text)

    def show_settings(self) -> None:
        if self.settings_window is None:
            from .settings_window import SettingsWindow

            self.settings_window = SettingsWindow(self.s, list_output_devices)
            self.settings_window.changed.connect(self._on_settings_changed)
            self.settings_window.snap_requested.connect(self.overlay.snap_to_default)
            self.settings_window.transcribe_requested.connect(self.pick_videos)
            self.settings_window.set_hotkey_failures(self.hotkeys.failed)
            self._update_model_status()
        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow()

    # -- files -----------------------------------------------------------------------
    def pick_videos(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(None, "Create subtitles for…", "", VIDEO_FILTER)
        self.jobs.add(files)


def _setup_logging() -> None:
    ensure_dirs()
    handlers: list[logging.Handler] = [
        RotatingFileHandler(LOG_DIR / "live-subtitles.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=handlers)


def main(open_settings: bool = False) -> int:
    _setup_logging()
    qapp = QApplication(sys.argv)
    qapp.setApplicationName(APP_NAME)
    qapp.setApplicationDisplayName(DISPLAY_NAME)
    qapp.setQuitOnLastWindowClosed(False)

    # Single instance: a second launch just opens the first one's settings.
    probe = QLocalSocket()
    probe.connectToServer(INSTANCE_KEY)
    if probe.waitForConnected(300):
        probe.write(b"settings")
        probe.waitForBytesWritten(300)
        return 0
    QLocalServer.removeServer(INSTANCE_KEY)
    server = QLocalServer()
    server.listen(INSTANCE_KEY)

    logger.info("%s %s starting", DISPLAY_NAME, __version__)
    app = App(qapp)
    server.newConnection.connect(lambda: (server.nextPendingConnection(), app.show_settings()))
    app.run()
    if open_settings:
        app.show_settings()
    return qapp.exec()
