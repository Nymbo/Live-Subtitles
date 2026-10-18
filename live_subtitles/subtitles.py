"""Word timings -> subtitle cues -> SRT / WebVTT / ASS files. Pure Python, no Qt.

Cue rules follow common subtitle guidelines: at most two lines of ~42 characters,
lines balanced and broken after punctuation where possible, cues split on pauses and
sentence ends, 1-7 s on screen, and enough time to read at ~20 characters per second.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .settings import Appearance


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Cue:
    start: float
    end: float
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.lines)


_SENTENCE_END = re.compile(r"[.!?…]['\")\]]*$")
_SOFT_BREAK = re.compile(r"[,;:—–-]['\")\]]*$")


def split_lines(words: list[str], max_chars: int, max_lines: int) -> list[str] | None:
    """Break words into <= max_lines balanced lines of <= max_chars. None if impossible."""
    text = " ".join(words)
    if len(text) <= max_chars or max_lines == 1:
        return [text] if len(text) <= max_chars else None
    best, best_score = None, None
    for i in range(1, len(words)):
        first, rest = " ".join(words[:i]), words[i:]
        if len(first) > max_chars:
            break
        tail = split_lines(rest, max_chars, max_lines - 1)
        if tail is None:
            continue
        lengths = [len(first), *map(len, tail)]
        score = max(lengths) - min(lengths)
        # Breaking after punctuation reads better, worth up to ~a dozen characters of imbalance.
        if _SENTENCE_END.search(words[i - 1]):
            score -= 20
        elif _SOFT_BREAK.search(words[i - 1]):
            score -= 12
        if best_score is None or score < best_score:
            best, best_score = [first, *tail], score
    return best


def build_cues(words: list[Word], max_chars: int = 42, max_lines: int = 2, max_cue_s: float = 7.0,
               min_cue_s: float = 1.0, pause_s: float = 0.7, linger_s: float = 0.6,
               chars_per_s: float = 20.0) -> list[Cue]:
    groups: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        if not w.text.strip():
            continue
        if current:
            prev = current[-1]
            gap = w.start - prev.end
            text_len = len(" ".join(x.text for x in current))
            sentence_done = _SENTENCE_END.search(prev.text) and (text_len > 18 or gap > 0.3)
            too_long = w.end - current[0].start > max_cue_s
            no_room = split_lines([x.text for x in current] + [w.text], max_chars, max_lines) is None
            if gap > pause_s or sentence_done or too_long or no_room:
                groups.append(current)
                current = []
        current.append(w)
    if current:
        groups.append(current)

    cues: list[Cue] = []
    for g in groups:
        lines = split_lines([w.text for w in g], max_chars, max_lines)
        if lines is None:  # a single word longer than a line; let it overflow rather than drop it
            lines = [" ".join(w.text for w in g)]
        cues.append(Cue(g[0].start, g[-1].end, lines))

    # Timing: linger, minimum duration, reading speed, never overlap the next cue.
    for i, cue in enumerate(cues):
        limit = cues[i + 1].start - 0.05 if i + 1 < len(cues) else cue.end + max_cue_s
        want = max(cue.end + linger_s, cue.start + min_cue_s,
                   cue.start + len(cue.text) / chars_per_s)
        cue.end = max(cue.end, min(want, limit, cue.start + max_cue_s))
    return cues


# -- writers ---------------------------------------------------------------------------

def _clock(t: float, sep: str) -> str:
    ms = max(0, int(round(t * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    blocks = [f"{i}\n{_clock(c.start, ',')} --> {_clock(c.end, ',')}\n" + "\n".join(c.lines)
              for i, c in enumerate(cues, 1)]
    return "\n\n".join(blocks) + "\n"


def to_vtt(cues: list[Cue]) -> str:
    blocks = [f"{_clock(c.start, '.')} --> {_clock(c.end, '.')}\n" + "\n".join(c.lines) for c in cues]
    return "WEBVTT\n\n" + "\n\n".join(blocks) + "\n"


def ass_color(argb: str) -> str:
    """'#AARRGGBB' (Qt) -> '&HAABBGGRR' (ASS, where AA is transparency, not opacity)."""
    v = argb.lstrip("#")
    if len(v) == 6:
        v = "FF" + v
    a, r, g, b = (int(v[i:i + 2], 16) for i in (0, 2, 4, 6))
    return f"&H{255 - a:02X}{b:02X}{g:02X}{r:02X}"


def _ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def to_ass(cues: list[Cue], ap: Appearance, ref_height: int = 1080) -> str:
    """ASS with the overlay's look, so VLC (libass) shows the same style as the live captions."""
    scale = 1080 / max(240, ref_height)
    # ASS font size is roughly the line height; Qt's pixel size is the em size.
    size = round(ap.font_size * 1.18 * scale)
    align = {"left": 1, "center": 2, "right": 3}.get(ap.align, 2)
    if ap.background in ("line", "block"):
        border_style, outline, shadow = 3, max(1, round(ap.padding_y * scale)), 0
        outline_colour, back_colour = ass_color(ap.background_color), ass_color(ap.background_color)
    else:
        border_style = 1
        outline = round(ap.outline_width * scale, 1)
        shadow = round(ap.shadow_offset * scale, 1)
        outline_colour, back_colour = ass_color(ap.outline_color), ass_color(ap.shadow_color)
    blur = round(min(ap.shadow_softness, 6) * scale / 3, 1) if border_style == 1 else 0
    style = ",".join(str(x) for x in [
        "Default", ap.font_family, size, ass_color(ap.text_color), ass_color(ap.text_color),
        outline_colour, back_colour, -1 if ap.bold else 0, -1 if ap.italic else 0, 0, 0,
        100, 100, 0, 0, border_style, outline, shadow, align, 60, 60, 54, 1])
    prefix = f"{{\\blur{blur}}}" if blur else ""
    events = []
    for c in cues:
        text = "\\N".join(_ass_escape(l.upper() if ap.uppercase else l) for l in c.lines)
        events.append(f"Dialogue: 0,{_ass_time(c.start)},{_ass_time(c.end)},Default,,0,0,0,,{prefix}{text}")
    return "\n".join([
        "[Script Info]",
        "; Generated by Live Subtitles",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: {style}",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        *events,
        "",
    ])
