"""Live model catalog and first-use download.

English live captions run NVIDIA's Parakeet-unified 0.6B on a CUDA GPU (`unified.py`),
from sherpa-onnx's fp32 buffered-streaming export (2.5 GB, Hugging Face).

Without a usable GPU, and for every other language, they run NVIDIA's cache-aware
streaming Nemotron models on the CPU through sherpa-onnx: the English-only model for
English, the multilingual Nemotron 3.5 otherwise. Each latency is a separate int8 export
published by sherpa-onnx, about 470 MB each.
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


@dataclass(frozen=True)
class HubModel:
    """A model fetched file by file from a Hugging Face repo."""
    name: str               # folder under MODELS_DIR
    repo: str
    files: tuple[str, ...]

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.name

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}"

    def is_installed(self) -> bool:
        return all((self.path / f).is_file() for f in self.files)


UNIFIED_MODEL = HubModel("parakeet-unified-en-0.6b-streaming-fp32",
                         "csukuangfj2/sherpa-onnx-nemo-parakeet-unified-en-0.6b-streaming-240ms",
                         ("encoder.onnx", "encoder.weights", "decoder.onnx", "joiner.onnx", "tokens.txt"))


def live_model(language: str, latency_ms: int) -> LiveModel:
    if latency_ms not in LATENCIES:
        latency_ms = 560
    if language == "en":
        return LiveModel(f"sherpa-onnx-nemotron-speech-streaming-en-0.6b-{latency_ms}ms-int8-2026-04-25",
                         None, latency_ms)
    return LiveModel(f"sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-{latency_ms}ms-int8-2026-06-11",
                     language or "auto", latency_ms)


def download(model: LiveModel | PunctModel | HubModel, progress: Callable[[int, int], None] | None = None) -> Path:
    """Download (and extract) a model. progress(done_bytes, total_bytes)."""
    if model.is_installed():
        return model.path
    if isinstance(model, HubModel):
        return _download_hub(model, progress)
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


def _download_hub(model: HubModel, progress) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    staging = MODELS_DIR / f".{model.name}.partial"
    staging.mkdir(exist_ok=True)
    base = f"{model.url}/resolve/main"
    sizes = {}
    for f in model.files:
        req = urllib.request.Request(f"{base}/{f}", method="HEAD", headers={"User-Agent": "Live-Subtitles"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            sizes[f] = int(resp.headers.get("X-Linked-Size") or resp.headers.get("Content-Length") or 0)
    total, done = sum(sizes.values()), 0
    for f in model.files:
        dst = staging / f
        if dst.is_file() and dst.stat().st_size == sizes[f]:      # resume after an interruption
            done += sizes[f]
            continue
        logger.info("Downloading %s/%s", base, f)
        req = urllib.request.Request(f"{base}/{f}", headers={"User-Agent": "Live-Subtitles"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dst, "wb") as out:
            reader = _ProgressReader(resp, total, progress, done)
            shutil.copyfileobj(reader, out, 1 << 20)
            done = reader._done
    shutil.rmtree(model.path, ignore_errors=True)
    staging.replace(model.path)
    if not model.is_installed():
        raise RuntimeError(f"Downloaded model is missing files: {model.path}")
    logger.info("Installed %s", model.path)
    return model.path


class _ProgressReader:
    def __init__(self, raw, total: int, progress, done: int = 0):
        self._raw, self._total, self._progress, self._done = raw, total, progress, done

    def read(self, size: int = -1) -> bytes:
        chunk = self._raw.read(size)
        self._done += len(chunk)
        if self._progress:
            self._progress(self._done, self._total)
        return chunk
