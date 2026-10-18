"""Caption line builder: turns a stream of words into the lines the overlay shows.

Pure Python (no Qt), so it is unit-tested directly. Line width is decided by a
`fits(text) -> bool` callback; the overlay passes a pixel measurer, tests pass none.

Stability rules (the benchmark's line-jump and rewrite counts are what these fix):
- A committed word's line is decided once, when it is first placed, and never changes.
  Re-laying-out every word on each update let a late period or a capital letter re-break
  lines that were already on screen, so whole lines jumped mid-read.
- Punctuation and casing from the `punctuate` hook may revise only the newest few words
  (PUNCT_EDITABLE). Words further back are frozen as shown. The model sees the last
  PUNCT_WINDOW words for context either way.
- The words still being recognised (`pending`) flow after the last committed word and
  may move until they are committed.
"""

from __future__ import annotations

import re
from typing import Callable

FILLERS = {"uh", "um", "uhm", "umm", "er", "erm", "ah", "hmm", "mm", "mhm"}
_SENTENCE_END = re.compile(r"[.!?…]['\")\]]*$")
_TRAILING_PUNCT = re.compile(r"[.,!?;:…]+$")

PUNCT_WINDOW = 40       # words given to the punctuation model on each update
PUNCT_EDITABLE = 4      # only the newest this-many words may still change punctuation/case
SETTLE_S = 0.8          # quiet this long -> the last word may take its final punctuation
KEEP_WORDS = 400        # trim older words beyond this (whole lines, keeping punct context)


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
        self.line_of: list[int] = []  # the line each committed word was placed on, fixed
        self.pending = ""             # words still being recognised, space-separated
        self.last_activity = 0.0
        self.settled = True

    # -- configuration ---------------------------------------------------------------
    def configure(self, **kwargs) -> None:
        relayout = any(k in kwargs for k in ("max_chars", "fits", "sentence_breaks"))
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.max_lines = max(1, self.max_lines)
        self.max_chars = max(8, self.max_chars)
        if relayout and self.shown:
            # A new width (the user resized the overlay): lay the words out afresh.
            self.line_of = []
            self._place(0)

    # -- input -----------------------------------------------------------------------
    def add(self, committed: list[str], pending: list[str] | str, now: float) -> None:
        if not isinstance(pending, str):
            pending = " ".join(pending)
        first_new = len(self.raw)
        for word in committed:
            if self.hide_fillers and is_filler(word):
                continue
            self.raw.append(word)
            self.shown.append(word)
        if self.hide_fillers:
            pending = " ".join(w for w in pending.split() if not is_filler(w))
        self.pending = pending
        if len(self.raw) > first_new:
            self.settled = False
            self._repunctuate(final=False)
            self._place(first_new)
            self._trim()
        if committed or pending:
            self.last_activity = now

    def clear(self) -> None:
        self.raw, self.shown, self.line_of = [], [], []
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
    def word_lines(self) -> list[int]:
        """Line index of every word on screen: committed words, then the pending ones."""
        return [*self.line_of, *self._pending_lines()]

    def display_lines(self) -> list[str]:
        words = self.shown + self.pending.split()
        if not words:
            return []
        lines: dict[int, list[str]] = {}
        for word, li in zip(words, self.word_lines()):
            lines.setdefault(li, []).append(word)
        last = max(lines)
        if self.mode == "pop":
            first = last // self.max_lines * self.max_lines
        else:
            first = last - self.max_lines + 1
        return [" ".join(lines[i]) for i in range(first, last + 1) if i in lines]

    # -- punctuation -----------------------------------------------------------------
    def _repunctuate(self, final: bool) -> bool:
        if not self.punctuate or not self.raw:
            return False
        start = max(0, len(self.raw) - PUNCT_WINDOW)
        out = self.punctuate(self.raw[start:])
        if out is None:
            return False
        before = list(self.shown)
        first_editable = max(start, len(self.raw) - PUNCT_EDITABLE)
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

    def _next_line(self, line_words: list[str], word: str) -> bool:
        """Does `word` start a new line after `line_words`?"""
        if not line_words:
            return False
        if self.sentence_breaks and _SENTENCE_END.search(line_words[-1]):
            return True
        return not self._fits(" ".join([*line_words, word]))

    def _place(self, first: int) -> None:
        """Assign lines to committed words from index `first` on; earlier words keep theirs."""
        del self.line_of[first:]
        line = self.line_of[-1] if self.line_of else 0
        current = [w for w, li in zip(self.shown, self.line_of) if li == line]
        for word in self.shown[first:]:
            if self._next_line(current, word):
                line += 1
                current = []
            current.append(word)
            self.line_of.append(line)

    def _pending_lines(self) -> list[int]:
        words = self.pending.split()
        if not words:
            return []
        line = self.line_of[-1] if self.line_of else 0
        current = [w for w, li in zip(self.shown, self.line_of) if li == line]
        out = []
        for word in words:
            if self._next_line(current, word):
                line += 1
                current = []
            current.append(word)
            out.append(line)
        return out

    def _trim(self) -> None:
        """Drop the oldest whole pages once the list is long, keeping punctuation context."""
        if len(self.raw) <= KEEP_WORDS:
            return
        keep_from = len(self.raw) - (PUNCT_WINDOW + PUNCT_EDITABLE)
        last_line = self.line_of[-1]
        cut_line = min(self.line_of[keep_from], last_line - self.max_lines - 1)
        if self.mode == "pop":
            cut_line = cut_line // self.max_lines * self.max_lines
        drop = sum(1 for li in self.line_of if li < cut_line)
        if drop:
            del self.raw[:drop]
            del self.shown[:drop]
            del self.line_of[:drop]
