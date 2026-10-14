"""Caption line builder: turns a stream of words into the lines the overlay shows.

Pure Python (no Qt), so it is unit-tested directly. Line width is decided by a
`fits(text) -> bool` callback; the overlay passes a pixel measurer, tests pass none.

Words since the last clear are kept as a flat list and laid out on every update. That
lets an optional `punctuate` hook revise recent words (punctuation and casing) after
more context arrives; the greedy layout is deterministic, so earlier lines stay put.
"""

from __future__ import annotations

import re
from typing import Callable

FILLERS = {"uh", "um", "uhm", "umm", "er", "erm", "ah", "hmm", "mm", "mhm"}
_SENTENCE_END = re.compile(r"[.!?…]['\")\]]*$")
_TRAILING_PUNCT = re.compile(r"[.,!?;:…]+$")

PUNCT_WINDOW = 40       # words re-punctuated on each update
PUNCT_CONTEXT = 6       # at the window's start: context only, never edited (a window
                        # start looks like a sentence start and would get capitalised)
SETTLE_S = 0.8          # quiet this long -> the last word may take its final punctuation
KEEP_WORDS = 400        # trim older words beyond this (whole pages, keeping punct context)


def is_filler(word: str) -> bool:
    return re.sub(r"[^a-z]", "", word.lower()) in FILLERS


class CaptionBuilder:
    def __init__(self, max_lines: int = 2, max_chars: int = 42, mode: str = "rollup",
                 sentence_breaks: bool = True, hide_fillers: bool = False,
                 fits: Callable[[str], bool] | None = None,
                 punctuate: Callable[[list[str]], list[str] | None] | None = None):
        self.max_lines = max(1, max_lines)
        self.max_chars = max(8, max_chars)
        self.mode = mode
        self.sentence_breaks = sentence_breaks
        self.hide_fillers = hide_fillers
        self.fits_px = fits
        self.punctuate = punctuate
        self.raw: list[str] = []      # committed words as recognised
        self.shown: list[str] = []    # display forms (punctuated / cased), same length
        self.pending = ""
        self.last_activity = 0.0
        self.settled = True

    # -- configuration ---------------------------------------------------------------
    def configure(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.max_lines = max(1, self.max_lines)
        self.max_chars = max(8, self.max_chars)

    # -- input -----------------------------------------------------------------------
    def add(self, committed: list[str], pending: str, now: float) -> None:
        added = False
        for word in committed:
            if self.hide_fillers and is_filler(word):
                continue
            self.raw.append(word)
            self.shown.append(word)
            added = True
        self.pending = "" if (self.hide_fillers and is_filler(pending)) else pending
        if added:
            self.settled = False
            self._repunctuate(final=False)
            self._trim()
        if committed or pending:
            self.last_activity = now

    def clear(self) -> None:
        self.raw, self.shown = [], []
        self.pending = ""
        self.settled = True

    def tick(self, now: float, clear_after_s: float) -> bool:
        """Settle punctuation, then clear, after quiet spells. Returns True when the display changed."""
        quiet = now - self.last_activity
        if (self.raw or self.pending) and quiet > clear_after_s:
            self.clear()
            return True
        if not self.settled and quiet > SETTLE_S and not self.pending:
            self.settled = True
            return self._repunctuate(final=True)
        return False

    # -- output ----------------------------------------------------------------------
    def display_lines(self) -> list[str]:
        words = self.shown + ([self.pending] if self.pending else [])
        lines = self._layout(words)
        if self.mode == "pop":
            lines = lines[(len(lines) - 1) // self.max_lines * self.max_lines:] if lines else []
        else:
            lines = lines[-self.max_lines:]
        return [" ".join(line) for line in lines if line]

    # -- punctuation -----------------------------------------------------------------
    def _repunctuate(self, final: bool) -> bool:
        if not self.punctuate or not self.raw:
            return False
        start = max(0, len(self.raw) - PUNCT_WINDOW)
        out = self.punctuate(self.raw[start:])
        if out is None:
            return False
        before = list(self.shown)
        first_editable = start + PUNCT_CONTEXT if start > 0 else 0
        for i in range(first_editable, len(self.raw)):
            self.shown[i] = out[i - start]
        if not final:
            # The model reads the end of its input as the end of a sentence. Until speech
            # pauses, keep the newest word bare so a period doesn't flicker on every line.
            self.shown[-1] = _TRAILING_PUNCT.sub("", self.shown[-1]) or self.shown[-1]
        return self.shown != before

    # -- layout ----------------------------------------------------------------------
    def _fits(self, text: str) -> bool:
        if len(text) > self.max_chars:
            return False
        return self.fits_px(text) if self.fits_px else True

    def _layout(self, words: list[str]) -> list[list[str]]:
        lines: list[list[str]] = []
        for word in words:
            if lines and lines[-1]:
                current = lines[-1]
                ends_sentence = self.sentence_breaks and _SENTENCE_END.search(current[-1])
                if ends_sentence or not self._fits(" ".join([*current, word])):
                    lines.append([])
            elif not lines:
                lines.append([])
            lines[-1].append(word)
        return lines

    def _trim(self) -> None:
        """Drop the oldest whole pages once the list is long, keeping punctuation context."""
        if len(self.raw) <= KEEP_WORDS:
            return
        lines = self._layout(self.shown)
        drop = 0
        for k in range(0, len(lines) - self.max_lines, self.max_lines):
            page = sum(len(line) for line in lines[k:k + self.max_lines])
            if len(self.raw) - drop - page < PUNCT_WINDOW + PUNCT_CONTEXT:
                break
            drop += page
        if drop:
            del self.raw[:drop]
            del self.shown[:drop]
