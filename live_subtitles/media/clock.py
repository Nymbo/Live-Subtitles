"""The playback clock, from the audio alone (or from the browser extension when it reports).

Audio clock: `tau` is seconds of *playing* audio heard since media mode started on a
title. Windows' loopback only delivers packets while something plays, and a paused
browser sends either nothing or exact digital silence, so counting non-silent samples
advances tau with playback and freezes it on pause. Quiet scenes still carry a noise
floor and keep counting. Between packets the clock extrapolates a little, so cues land
on time rather than on the next 50 ms packet.

Extension clock: the page's own `video.currentTime`, reported a few times a second, used
directly as tau (with the paused flag). Seeks then show up as jumps in tau itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

SR = 16000
SILENCE_RMS = 1e-5          # digital silence; a quiet scene's noise floor sits well above
EXTRAPOLATE_S = 0.3         # carry the clock forward this long without a packet
PAUSED_AFTER_S = 0.6        # no packets (or only silence) this long = paused


@dataclass
class AudioClock:
    tau: float = 0.0                # seconds of playing audio counted so far
    last_audio: float = 0.0         # monotonic time the last playing packet arrived
    silent_since: float | None = None

    def feed(self, samples: np.ndarray, now: float | None = None) -> bool:
        """Count a captured chunk. Returns True when it was playing audio (not silence)."""
        now = time.monotonic() if now is None else now
        if samples.size == 0:
            return False
        if float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) < SILENCE_RMS:
            if self.silent_since is None:
                self.silent_since = now
            return False
        self.silent_since = None
        self.tau += samples.size / SR
        self.last_audio = now
        return True

    def playing(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return self.last_audio > 0 and now - self.last_audio < PAUSED_AFTER_S

    def now(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        if not self.playing(now):
            return self.tau
        return self.tau + min(EXTRAPOLATE_S, max(0.0, now - self.last_audio))


@dataclass
class ExtensionClock:
    position: float = 0.0           # video.currentTime at `at`
    at: float = 0.0                 # monotonic time of the report
    paused: bool = True
    rate: float = 1.0
    duration: float = 0.0

    def report(self, position: float, paused: bool, rate: float = 1.0, duration: float = 0.0,
               now: float | None = None) -> None:
        self.position, self.paused, self.rate = float(position), bool(paused), float(rate or 1.0)
        self.duration = float(duration or 0.0)
        self.at = time.monotonic() if now is None else now

    def fresh(self, now: float | None = None, max_age: float = 3.0) -> bool:
        now = time.monotonic() if now is None else now
        return self.at > 0 and now - self.at < max_age

    def now(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        if self.paused:
            return self.position
        return self.position + min(2.0, now - self.at) * self.rate
