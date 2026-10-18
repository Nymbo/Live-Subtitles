"""Live recogniser thread: pulls captured audio off the queue and runs a speech engine.

The engine (`engines.py`) holds all the decoding logic, so the benchmark can drive the
very same code on a simulated clock.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

import numpy as np

from .engines import IDLE_POLL_S

logger = logging.getLogger(__name__)

# (newly committed words, words in progress, phrase id)
WordsCallback = Callable[[list[str], list[str], int], None]


class LiveRecognizer(threading.Thread):
    def __init__(self, engine, chunks: "queue.Queue[np.ndarray]",
                 on_words: WordsCallback, on_ready: Callable[[], None] | None = None,
                 on_error: Callable[[str], None] | None = None):
        super().__init__(name="live-recognizer", daemon=True)
        self.engine = engine
        self.chunks = chunks
        self.on_words = on_words
        self.on_ready = on_ready
        self.on_error = on_error
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            logger.exception("Live recogniser crashed")
            if self.on_error:
                self.on_error(str(exc))
        finally:
            close = getattr(self.engine, "close", None)
            if close:
                close()

    def _run(self) -> None:
        self.engine.load()
        # Drop audio that queued up while the model loaded; captions should start "now".
        while not self.chunks.empty():
            self.chunks.get_nowait()
        if self.on_ready:
            self.on_ready()
        while not self._stop.is_set():
            try:
                parts = [self.chunks.get(timeout=IDLE_POLL_S)]
                while True:   # drain whatever else is waiting so decoding catches up in one go
                    try:
                        parts.append(self.chunks.get_nowait())
                    except queue.Empty:
                        break
                updates = self.engine.process(np.concatenate(parts))
            except queue.Empty:
                updates = self.engine.process(None)
            for update in updates:
                if self._stop.is_set():
                    break
                self.on_words(*update)
