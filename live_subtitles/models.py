"""Live model catalog and first-use download.

Live captions run NVIDIA's cache-aware streaming Nemotron models through sherpa-onnx.
English uses the English-only model (more accurate on English in our bake-off); any
other language, or auto-detect, uses the multilingual Nemotron 3.5. Each latency is a
separate int8 export published by sherpa-onnx, about 470 MB each.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .paths import MODELS_DIR

logger = logging.getLogger(__name__)

RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
LATENCIES = (160, 560, 1120)

# Locales the multilingual model transcribes out of the box (model card: "transcription-ready"
# and "broad-coverage" tiers). Keys are what the runtime's prompt dictionary accepts.
LANGUAGES: dict[str, str] = {
    "en": "English",
    "auto": "Auto-detect (multilingual)",
    "en-GB": "English (UK accent, multilingual model)",
    "es-ES": "Spanish (Spain)", "es-US": "Spanish (Latin America)",
    "fr-FR": "French", "fr-CA": "French (Canada)", "de-DE": "German", "it-IT": "Italian",
    "pt-BR": "Portuguese (Brazil)", "pt-PT": "Portuguese (Portugal)", "nl-NL": "Dutch",
    "tr-TR": "Turkish", "ru-RU": "Russian", "uk-UA": "Ukrainian", "ar-AR": "Arabic",
    "hi-IN": "Hindi", "ja-JP": "Japanese", "ko-KR": "Korean", "vi-VN": "Vietnamese",
    "zh-CN": "Mandarin", "pl-PL": "Polish", "sv-SE": "Swedish", "cs-CZ": "Czech",
    "nb-NO": "Norwegian", "da-DK": "Danish", "bg-BG": "Bulgarian", "fi-FI": "Finnish",
    "hr-HR": "Croatian", "sk-SK": "Slovak", "hu-HU": "Hungarian", "ro-RO": "Romanian",
    "et-EE": "Estonian",
}


@dataclass(frozen=True)
class LiveModel:
    name: str          # release archive / folder name
    language: str | None  # value for stream.set_option("language", ...), None for English-only
    latency_ms: int

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.name

    @property
    def url(self) -> str:
        return f"{RELEASE}/{self.name}.tar.bz2"

    def is_installed(self) -> bool:
        return all((self.path / f).is_file() for f in
                   ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"))


@dataclass(frozen=True)
class PunctModel:
    """sherpa-onnx's streaming English punctuation + casing model (CNN-BiLSTM, ~31 MB)."""
    name: str = "sherpa-onnx-online-punct-en-2024-08-06"

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.name

    @property
    def url(self) -> str:
        return ("https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/"
                f"{self.name}.tar.bz2")

    def is_installed(self) -> bool:
        return all((self.path / f).is_file() for f in ("model.int8.onnx", "bpe.vocab"))


PUNCT_MODEL = PunctModel()


def live_model(language: str, latency_ms: int) -> LiveModel:
    if latency_ms not in LATENCIES:
        latency_ms = 560
    if language == "en":
        return LiveModel(f"sherpa-onnx-nemotron-speech-streaming-en-0.6b-{latency_ms}ms-int8-2026-04-25",
                         None, latency_ms)
    return LiveModel(f"sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-{latency_ms}ms-int8-2026-06-11",
                     language or "auto", latency_ms)


def download(model: LiveModel | PunctModel, progress: Callable[[int, int], None] | None = None) -> Path:
    """Download and extract a model package. progress(done_bytes, total_bytes)."""
    if model.is_installed():
        return model.path
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    staging = MODELS_DIR / f".{model.name}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    logger.info("Downloading %s", model.url)
    req = urllib.request.Request(model.url, headers={"User-Agent": "Live-Subtitles"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        reader = _ProgressReader(resp, total, progress)
        with tarfile.open(fileobj=reader, mode="r|bz2") as tar:
            tar.extractall(staging, filter="data")
    # The archive holds one top-level folder named like the model.
    inner = staging / model.name
    source = inner if inner.is_dir() else staging
    shutil.rmtree(model.path, ignore_errors=True)
    source.replace(model.path)
    shutil.rmtree(staging, ignore_errors=True)
    if not model.is_installed():
        raise RuntimeError(f"Downloaded package is missing files: {model.path}")
    logger.info("Installed %s", model.path)
    return model.path


class _ProgressReader:
    def __init__(self, raw, total: int, progress):
        self._raw, self._total, self._progress, self._done = raw, total, progress, 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._raw.read(size)
        self._done += len(chunk)
        if self._progress:
            self._progress(self._done, self._total)
        return chunk
