"""Scoring for the live-caption benchmark: WER, caption delay, punctuation, stability."""

from __future__ import annotations

import re

import numpy as np

NUM_WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
             "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten"}
_END = re.compile(r"[.!?…]['\")\]]*$")


def norm_word(w: str) -> str:
    w = w.lower().replace("’", "'")
    w = re.sub(r"[^a-z0-9']+", "", w).strip("'")
    return NUM_WORDS.get(w, w)


def tokens(text_or_words) -> list[tuple[str, str]]:
    """[(normalised, raw)] with empty normalisations dropped (a lone '-' or '...')."""
    words = text_or_words.split() if isinstance(text_or_words, str) else text_or_words
    out = []
    for raw in words:
        for part in raw.replace("—", " ").replace("-", " ").split():
            n = norm_word(part)
            if n:
                out.append((n, raw if part == raw else part))
    return out


def align(ref: list[str], hyp: list[str]) -> list[tuple[int | None, int | None]]:
    """Levenshtein alignment: pairs (ref index, hyp index); None marks an insertion/deletion."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = d[i], d[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (ri != hyp[j - 1]))
    pairs = []
    i, j = n, m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return pairs[::-1]


def wer(ref: list[str], hyp: list[str], pairs=None) -> dict:
    pairs = pairs or align(ref, hyp)
    sub = sum(1 for a, b in pairs if a is not None and b is not None and ref[a] != hyp[b])
    dele = sum(1 for a, b in pairs if b is None)
    ins = sum(1 for a, b in pairs if a is None)
    return {"wer": round(100 * (sub + dele + ins) / max(1, len(ref)), 2), "sub": sub, "del": dele, "ins": ins}


def sentence_f1(ref_raw: list[str], hyp_raw: list[str], pairs, ref_ends: set[int] | None = None) -> float:
    """F1 of sentence-final punctuation on aligned word pairs."""
    tp = fp = fn = 0
    for a, b in pairs:
        if a is None or b is None:
            continue
        r = (a in ref_ends) if ref_ends is not None else bool(_END.search(ref_raw[a]))
        h = bool(_END.search(hyp_raw[b]))
        tp += r and h
        fp += h and not r
        fn += r and not h
    return round(2 * tp / max(1, 2 * tp + fp + fn), 3)


def pct(values, q) -> float | None:
    return round(float(np.percentile(values, q)), 2) if len(values) else None
