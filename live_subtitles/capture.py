"""System-audio capture: WASAPI loopback -> 16 kHz mono float32 chunks on a queue.

Windows only delivers loopback packets while something is playing, so silence shows
up as *no* chunks, not zero-filled ones. The live engine handles that gap.
"""

from __future__ import annotations

import logging
import queue
import threading

import numpy as np
import pyaudiowpatch as pyaudio
import soxr

logger = logging.getLogger(__name__)

TARGET_RATE = 16000


def list_output_devices() -> list[str]:
    """Names of the loopback-capable output devices (what the user hears through)."""
    pa = pyaudio.PyAudio()
    try:
        return [d["name"] for d in pa.get_loopback_device_info_generator()]
    finally:
        pa.terminate()


def downmix(frames: np.ndarray, dialogue_boost: bool) -> np.ndarray:
    """(n, channels) -> (n,). On surround output, lean on the centre channel where dialogue lives."""
    channels = frames.shape[1]
    if channels == 1:
        return frames[:, 0]
    if dialogue_boost and channels >= 6:
        # WAVEFORMATEXTENSIBLE order: FL, FR, FC, LFE, then surrounds.
        return 0.6 * frames[:, 2] + 0.2 * (frames[:, 0] + frames[:, 1])
    if channels >= 6:
        return frames[:, [0, 1, 2]].mean(axis=1)
    return frames[:, :2].mean(axis=1)


class LoopbackCapture:
    def __init__(self, device_name: str = "", dialogue_boost: bool = True):
        self.device_name = device_name
        self.dialogue_boost = dialogue_boost
        self.chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=400)
        self.device_label = ""
        self._pa: pyaudio.PyAudio | None = None
        self._stream = None
        self._resampler: soxr.ResampleStream | None = None
        self._channels = 2
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            self._pa = pyaudio.PyAudio()
            dev = self._pick_device()
            self._channels = int(dev["maxInputChannels"]) or 2
            rate = int(dev["defaultSampleRate"])
            self._resampler = soxr.ResampleStream(rate, TARGET_RATE, 1, dtype="float32", quality="HQ")
            self.device_label = dev["name"]
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=self._channels,
                rate=rate,
                input=True,
                input_device_index=dev["index"],
                frames_per_buffer=int(rate * 0.05),
                stream_callback=self._callback,
            )
            self._stream.start_stream()
            logger.info("Capturing %s (%d ch @ %d Hz)", dev["name"], self._channels, rate)

    def stop(self) -> None:
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop_stream()
                    self._stream.close()
                except Exception:
                    logger.exception("Error closing loopback stream")
                self._stream = None
            if self._pa is not None:
                self._pa.terminate()
                self._pa = None

    def _pick_device(self) -> dict:
        assert self._pa is not None
        if self.device_name:
            for dev in self._pa.get_loopback_device_info_generator():
                if dev["name"] == self.device_name:
                    return dev
            logger.warning("Device %r not found; using the default output", self.device_name)
        return self._pa.get_default_wasapi_loopback()

    def _callback(self, in_data, frame_count, time_info, status):
        try:
            frames = np.frombuffer(in_data, dtype=np.float32).reshape(-1, self._channels)
            mono = downmix(frames, self.dialogue_boost).astype(np.float32, copy=False)
            out = self._resampler.resample_chunk(mono)
            if out.size:
                try:
                    self.chunks.put_nowait(out)
                except queue.Full:  # the recogniser fell behind; drop the oldest audio
                    try:
                        self.chunks.get_nowait()
                    except queue.Empty:
                        pass
                    self.chunks.put_nowait(out)
        except Exception:
            logger.exception("Loopback callback failed")
        return (None, pyaudio.paContinue)
