"""Where is playback in the subtitle track? Anchors in, a time map out. Pure Python + numpy.

Two kinds of evidence give anchors, pairs of (audio clock tau, track time c):

- Text anchors (English audio): words recognised in the audio are matched against the
  track's words by rare three-word runs ("you said tomorrow"). Each match says "this
  moment of audio is this moment of the track". Common runs match many places and get
  little weight; the time map that most anchors agree on wins.
- Speech anchors (any language, e.g. Korean audio with English subtitles): voice
  activity in the audio is cross-correlated with the track's cue spans, the
  ffsubsync/alass idea. Weaker per window, so it needs more audio before locking.

`Aligner` keeps the current mapping c = scale * tau + offset. The scale covers frame-rate
mismatches (a 25 fps PAL track on a 23.976 fps video runs 4.3% fast). A run of fresh
anchors that agrees on a different offset is a seek, a cut scene, or a different edit,
and starts a new segment. The audio clock never jumps, so every jump is in the track.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..subtitles import Cue
from .subfile import speech_text

# Frame-rate ratios that show up between subtitle tracks and videos.
SCALES = (1.0, 25 / 23.976, 23.976 / 25, 24 / 23.976, 23.976 / 24, 25 / 24, 24 / 25, 30 / 29.97, 29.97 / 30)
NGRAM = 3
MAX_POSITIONS = 6           # an n-gram found in more places than this says nothing
BANDWIDTH_S = 0.45          # anchors this close in offset agree
WINDOW_S = 900.0            # anchors older than this (audio time) no longer steer the fit
JUMP_LOOKBACK_S = 60.0      # fresh anchors that may announce a jump
MIN_LOCK_WEIGHT = 2.5       # summed anchor weight needed to lock
MIN_LOCK_ANCHORS = 3
MIN_JUMP_S = 1.2            # an offset change smaller than this is drift, not a jump
MIN_SCALE_SPAN_S = 40.0     # 4% drift over 40 s is ~1.7 s: plainly not one offset

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_NUMBERS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
            "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def tokens(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(text.lower().replace("’", "'")):
        w = w.replace("'", "")
        out.append(_NUMBERS.get(w, w))
    return out


@dataclass
class Anchor:
    tau: float          # audio clock (s)
    c: float            # track time (s)
    weight: float = 1.0
    kind: str = "text"  # text | speech


class TrackIndex:
    """A track's spoken words with times, indexed by n-gram for anchor lookups."""

    def __init__(self, cues: list[Cue]):
        self.words: list[str] = []
        self.times: list[float] = []
        for cue in cues:
            toks = tokens(speech_text(cue.lines))
            if not toks:
                continue
            span = max(0.2, cue.end - cue.start)
            # Speech fills most of a cue's span; place words evenly through its first 85%.
            for i, w in enumerate(toks):
                self.words.append(w)
                self.times.append(cue.start + span * 0.85 * (i + 0.5) / len(toks))
        self.grams: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for i in range(len(self.words) - NGRAM + 1):
            self.grams[tuple(self.words[i:i + NGRAM])].append(i)

    def __len__(self) -> int:
        return len(self.words)

    def anchors(self, words: list[tuple[str, float]], near: tuple[float, float] | None = None) -> list[Anchor]:
        """Anchors for recognised `words` [(text, tau)]. `near=(c_lo, c_hi)` limits the search."""
        toks: list[tuple[str, float]] = []
        for text, tau in words:
            for t in tokens(text):
                toks.append((t, tau))
        out = []
        for i in range(len(toks) - NGRAM + 1):
            gram = tuple(t for t, _ in toks[i:i + NGRAM])
            positions = self.grams.get(gram)
            if not positions:
                continue
            if near:
                positions = [p for p in positions if near[0] <= self.times[p] <= near[1]]
                if not positions:
                    continue
            if len(positions) > MAX_POSITIONS:
                continue
            w = 1.0 / len(positions)
            if len(set(gram)) == 1:     # "no no no"
                w *= 0.3
            mid = i + NGRAM // 2
            for p in positions:
                out.append(Anchor(toks[mid][1], self.times[p + NGRAM // 2], w))
        return out


@dataclass
class Fit:
    scale: float
    offset: float
    weight: float
    count: int
    span: float         # tau range the supporting anchors cover

    def track_time(self, tau: float) -> float:
        return self.scale * tau + self.offset


def best_fit(anchors: list[Anchor], scales=SCALES, bandwidth: float = BANDWIDTH_S) -> Fit | None:
    """The (scale, offset) most anchor weight agrees on."""
    if not anchors:
        return None
    best: Fit | None = None
    taus = np.array([a.tau for a in anchors])
    cs = np.array([a.c for a in anchors])
    ws = np.array([a.weight for a in anchors])
    for s in scales:
        b = cs - s * taus
        order = np.argsort(b)
        bs, wsort = b[order], ws[order]
        cum = np.concatenate([[0.0], np.cumsum(wsort)])
        j = 0
        for i in range(len(bs)):
            while bs[i] - bs[j] > bandwidth:
                j += 1
            weight = cum[i + 1] - cum[j]
            if best is not None and weight <= best.weight * (1.0 if s == 1.0 else 1.15):
                continue
            members = order[j:i + 1]
            span = float(taus[members].max() - taus[members].min()) if len(members) else 0.0
            if s != 1.0 and span < MIN_SCALE_SPAN_S:
                continue        # a frame-rate ratio only shows over a stretch of audio
            offset = float(np.average(bs[j:i + 1], weights=wsort[j:i + 1]))
            best = Fit(s, offset, float(weight), int(i + 1 - j), span)
    return best


@dataclass
class Segment:
    tau_start: float
    scale: float
    offset: float


@dataclass
class Aligner:
    """Turns a stream of anchors into the current track mapping, with jump detection."""
    min_weight: float = MIN_LOCK_WEIGHT
    min_anchors: int = MIN_LOCK_ANCHORS
    bandwidth: float = BANDWIDTH_S
    anchors: list[Anchor] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    seg_from: float = -math.inf         # anchors before this tau belong to older segments
    scale_hint: float | None = None     # a known scale (from a saved track) to try first

    @property
    def locked(self) -> bool:
        return bool(self.segments)

    @property
    def current(self) -> Segment | None:
        return self.segments[-1] if self.segments else None

    def reset(self) -> None:
        self.anchors.clear()
        self.segments.clear()
        self.seg_from = -math.inf

    def track_time(self, tau: float) -> float | None:
        seg = self.current
        return None if seg is None else seg.scale * tau + seg.offset

    def add(self, new: list[Anchor]) -> str | None:
        """Add anchors; returns "locked", "jumped", "refined" or None."""
        if not new:
            return None
        self.anchors.extend(new)
        self.anchors.sort(key=lambda a: a.tau)
        latest = self.anchors[-1].tau
        self.anchors = [a for a in self.anchors if a.tau >= latest - WINDOW_S]
        scales = self._scales()

        if not self.locked:
            fit = best_fit(self._live(), scales, self.bandwidth)
            if self._strong(fit):
                self.segments.append(Segment(self._first_tau(fit), fit.scale, fit.offset))
                return "locked"
            return None

        seg = self.current
        # A frame-rate mismatch looks like a slow run of small jumps; check for it first, so
        # it becomes a new scale rather than a string of false segments.
        live = self._live()
        glob = best_fit(live, scales, self.bandwidth)
        if glob and glob.scale != seg.scale and self._strong(glob) and glob.span >= MIN_SCALE_SPAN_S:
            here = best_fit(live, (seg.scale,), self.bandwidth)
            if here is None or glob.weight > 1.3 * here.weight:
                seg.scale, seg.offset = glob.scale, glob.offset
                return "refined"
        fresh = [a for a in live if a.tau >= latest - JUMP_LOOKBACK_S]
        fit = best_fit(fresh, (seg.scale,), self.bandwidth)
        if self._strong(fit) and abs(fit.offset - seg.offset) > MIN_JUMP_S:
            # Do the fresh anchors disagree with the current mapping as a group?
            agree = sum(a.weight for a in fresh if abs(a.c - (seg.scale * a.tau + seg.offset)) <= self.bandwidth)
            if fit.weight > 1.5 * agree:
                start = self._first_tau(fit, fresh)
                self.seg_from = start
                self.segments.append(Segment(start, seg.scale, fit.offset))
                return "jumped"
        fit = best_fit(self._live(), scales, self.bandwidth)
        if fit and self._strong(fit) and (fit.scale != seg.scale or abs(fit.offset - seg.offset) <= MIN_JUMP_S):
            if fit.scale != seg.scale or abs(fit.offset - seg.offset) > 0.02:
                seg.scale, seg.offset = fit.scale, fit.offset
                return "refined"
        return None

    def _scales(self):
        if self.scale_hint and self.scale_hint in SCALES:
            return (self.scale_hint, *[s for s in SCALES if s != self.scale_hint])
        return SCALES

    def _live(self) -> list[Anchor]:
        return [a for a in self.anchors if a.tau >= self.seg_from]

    def _strong(self, fit: Fit | None) -> bool:
        return fit is not None and fit.weight >= self.min_weight and fit.count >= self.min_anchors

    def _first_tau(self, fit: Fit, pool: list[Anchor] | None = None) -> float:
        pool = pool if pool is not None else self._live()
        taus = [a.tau for a in pool if abs(a.c - fit.track_time(a.tau)) <= self.bandwidth]
        return min(taus) if taus else pool[-1].tau


def match_score(index: TrackIndex, words: list[tuple[str, float]]) -> float:
    """How well recognised words fit a track (0-1): the share of n-grams on one consistent mapping.

    Used to pick between candidate tracks and to notice the wrong episode or cut."""
    anchors = index.anchors(words)
    n = max(1, sum(1 for _ in words) - NGRAM + 1)
    if not anchors:
        return 0.0
    # Count per segment of 120 s so a seek or a cut doesn't hide a good track.
    total = 0.0
    taus = sorted({a.tau for a in anchors})
    lo = taus[0]
    while lo <= taus[-1]:
        chunk = [a for a in anchors if lo <= a.tau < lo + 120]
        fit = best_fit(chunk, (1.0,))
        if fit:
            total += fit.weight
        lo += 120
    return min(1.0, total / n)


# -- speech-span alignment ---------------------------------------------------------------

BIN_S = 0.1


def cue_mask(cues: list[Cue], duration: float | None = None) -> np.ndarray:
    """1 where the track has a spoken line, at BIN_S resolution."""
    end = duration if duration is not None else max((c.end for c in cues), default=0.0)
    mask = np.zeros(int(end / BIN_S) + 1, np.float32)
    for c in cues:
        if not speech_text(c.lines):
            continue
        mask[int(c.start / BIN_S):int(c.end / BIN_S) + 1] = 1.0
    return mask


def speech_mask(spans: list[tuple[float, float]], t0: float, t1: float) -> np.ndarray:
    mask = np.zeros(max(1, int((t1 - t0) / BIN_S)), np.float32)
    for a, b in spans:
        i, j = int((max(a, t0) - t0) / BIN_S), int((min(b, t1) - t0) / BIN_S) + 1
        if j > i:
            mask[i:j] = 1.0
    return mask


@dataclass
class SpeechMatch:
    offset: float       # c = scale * tau + offset
    scale: float
    z: float            # peak height in standard deviations of the correlation
    ratio: float        # peak over the best peak elsewhere


def speech_align(track: np.ndarray, spans: list[tuple[float, float]], t0: float, t1: float,
                 scale: float = 1.0, near: float | None = None, radius: float = 20.0) -> SpeechMatch | None:
    """Cross-correlate voice activity in audio [t0, t1] (tau) with the track's cue mask.

    `near` (a predicted offset) limits the search to +-radius seconds around it."""
    audio = speech_mask(spans, t0, t1)
    if audio.sum() * BIN_S < 8:          # not enough speech in the window to say anything
        return None
    # Resample the audio window onto track time: bin k of the window is tau = t0 + k*BIN/scale.
    n = int(len(audio) * scale)
    idx = np.minimum((np.arange(n) / scale).astype(int), len(audio) - 1)
    a = audio[idx] - audio.mean()
    t = track - track.mean()
    if len(t) < n:
        t = np.pad(t, (0, n - len(t)))
    size = 1 << int(math.ceil(math.log2(len(t) + n)))
    corr = np.fft.irfft(np.fft.rfft(t, size) * np.conj(np.fft.rfft(a, size)), size)[:len(t) - n + 1]
    lags = np.arange(len(corr))
    offsets = lags * BIN_S - scale * t0          # offset = c_start - scale * t0
    if near is not None:
        keep = np.abs(offsets - near) <= radius
        if not keep.any():
            return None
        corr, offsets = corr[keep], offsets[keep]
    if len(corr) < 3:
        return None
    k = int(np.argmax(corr))
    sd = float(corr.std()) or 1e-9
    z = float((corr[k] - corr.mean()) / sd)
    away = np.abs(offsets - offsets[k]) > 2.0
    second = float(corr[away].max()) if away.any() else float(corr.mean())
    ratio = float(corr[k] / second) if second > 0 else float("inf")
    return SpeechMatch(float(offsets[k]), scale, z, ratio)

