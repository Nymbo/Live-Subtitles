"""English punctuation and casing for live captions (sherpa-onnx online CNN-BiLSTM model).

Nemotron punctuates per utterance, but on film dialogue it rarely marks sentence ends:
sentence-end F1 0.22 against the official Tears of Steel subtitles. Re-punctuating the
recent words with this model scored 0.68 with 3 words of lookahead and 0.71 with 8.
"""

from __future__ import annotations

import logging
import re
import threading

from .models import PUNCT_MODEL

logger = logging.getLogger(__name__)

_STRIP = re.compile(r"[^\w'-]")


_SENTENCE_END = re.compile(r"[.!?]['\")\]]*$")


def plain(word: str) -> str:
    """What the model expects: lowercase, no punctuation (apostrophes and hyphens kept)."""
    return _STRIP.sub("", word).strip("-").lower()


def fix_sentence_case(words: list[str]) -> list[str]:
    """Capitalise the word after a sentence end; the model sometimes leaves it lowercase."""
    out = list(words)
    for i in range(1, len(out)):
        if _SENTENCE_END.search(out[i - 1]) and out[i][:1].islower():
            out[i] = out[i][0].upper() + out[i][1:]
    return out


class Punctuator:
    def __init__(self, threads: int = 1):
        import sherpa_onnx

        d = PUNCT_MODEL.path
        cfg = sherpa_onnx.OnlinePunctuationConfig(model_config=sherpa_onnx.OnlinePunctuationModelConfig(
            cnn_bilstm=str(d / "model.int8.onnx"), bpe_vocab=str(d / "bpe.vocab"), num_threads=threads))
        self._model = sherpa_onnx.OnlinePunctuation(cfg)
        self._lock = threading.Lock()

    def punctuate(self, words: list[str]) -> list[str] | None:
        """Punctuated, cased forms of `words`, one per word; None if the model's output doesn't line up."""
        bare = [plain(w) for w in words]
        if not all(bare):
            # A word that was only punctuation: keep the alignment simple and skip this pass.
            return None
        try:
            with self._lock:
                out = self._model.add_punctuation_with_case(" ".join(bare)).split()
        except Exception:
            logger.exception("Punctuation model failed")
            return None
        if len(out) != len(words) or any(plain(a) != b for a, b in zip(out, bare)):
            return None
        return fix_sentence_case(out)
