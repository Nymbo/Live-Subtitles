"""Subtitle files in, cues out: SRT / WebVTT / ASS parsing, encodings, zips, sound descriptions.

Downloaded subtitles arrive in every encoding and container imaginable. `load_bytes`
handles zip archives, BOMs, UTF-16 and legacy Windows code pages, and picks the parser
from the content rather than trusting a file extension.

Sound descriptions ("[door creaks]", "(laughs)", "JOHN: ...", "♪ ... ♪") are kept in the
track; `display_lines` hides them when the viewer turns sound descriptions off, and
`speech_text` strips them for alignment, so one cached track serves both settings.
"""

from __future__ import annotations

import io
import re
import zipfile

from ..subtitles import Cue

_TIME = r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
_SRT_TIMING = re.compile(_TIME + r"\s*-->\s*" + _TIME)
_TAGS = re.compile(r"<[^>]+>")                         # <i>, <font ...>, <c.yellow>, <00:01.000>
_ASS_OVERRIDES = re.compile(r"\{[^}]*\}")
_BRACKETED = re.compile(r"\[[^\]]*\]|\([^)]*\)|（[^）]*）")
_SPEAKER = re.compile(r"^\s*(?:-\s*)?[A-Z][A-Z0-9 .'’-]{1,30}:\s+")   # "JOHN:", "- MAN 2:"
_MUSIC = re.compile(r"[♪♫#]+")
_ONLY_PUNCT = re.compile(r"^[\s\-–—.,!?…:;'\"♪♫#]*$")


def _seconds(h, m, s, ms) -> float:
    ms = (ms or "0").ljust(3, "0")[:3]
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def decode(data: bytes) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    if len(data) > 4 and data[1:2] == b"\x00" and data[3:4] == b"\x00":
        return data.decode("utf-16-le", errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def parse_srt_vtt(text: str) -> list[Cue]:
    cues: list[Cue] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    while i < len(lines):
        m = _SRT_TIMING.search(lines[i])
        if not m:
            i += 1
            continue
        start, end = _seconds(*m.groups()[:4]), _seconds(*m.groups()[4:])
        i += 1
        body = []
        while i < len(lines) and lines[i].strip():
            if _SRT_TIMING.search(lines[i]):
                break
            body.append(lines[i])
            i += 1
        # SRT numbers the next cue on the line before its timing; drop a trailing bare index.
        if body and body[-1].strip().isdigit() and i < len(lines) and _SRT_TIMING.search(lines[i]):
            body.pop()
        texts = [_clean(b) for b in body]
        texts = [t for t in texts if t]
        if texts and end > start:
            cues.append(Cue(start, end, texts))
    return cues


def parse_ass(text: str) -> list[Cue]:
    cues: list[Cue] = []
    fmt: list[str] | None = None
    in_events = False
    for line in text.replace("\r\n", "\n").split("\n"):
        s = line.strip()
        if s.startswith("["):
            in_events = s.lower() == "[events]"
            continue
        if not in_events:
            continue
        if s.lower().startswith("format:"):
            fmt = [f.strip().lower() for f in s[7:].split(",")]
            continue
        if not s.lower().startswith("dialogue:") or not fmt:
            continue
        parts = s[9:].split(",", len(fmt) - 1)
        row = dict(zip(fmt, (p.strip() for p in parts)))
        try:
            h, m, sec = row["start"].split(":")
            start = int(h) * 3600 + int(m) * 60 + float(sec)
            h, m, sec = row["end"].split(":")
            end = int(h) * 3600 + int(m) * 60 + float(sec)
        except (KeyError, ValueError):
            continue
        body = _ASS_OVERRIDES.sub("", row.get("text", "")).replace("\\h", " ")
        texts = [_clean(t) for t in re.split(r"\\[Nn]", body)]
        texts = [t for t in texts if t]
        if texts and end > start:
            cues.append(Cue(start, end, texts))
    cues.sort(key=lambda c: c.start)
    return cues


def _clean(line: str) -> str:
    return re.sub(r"\s+", " ", _TAGS.sub("", line)).strip()


def parse_text(text: str) -> list[Cue]:
    if "[Events]" in text or "Dialogue:" in text[:20000]:
        cues = parse_ass(text)
        if cues:
            return cues
    return parse_srt_vtt(text)


def load_bytes(data: bytes) -> list[Cue]:
    """Any downloaded subtitle payload (plain, zipped) -> cues, sorted, overlaps trimmed."""
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = [n for n in z.namelist() if n.lower().endswith((".srt", ".vtt", ".ass", ".ssa"))]
            # The largest file is the full track; zips sometimes carry forced-only extras.
            names.sort(key=lambda n: z.getinfo(n).file_size, reverse=True)
            if not names:
                raise ValueError("no subtitle file in the archive")
            data = z.read(names[0])
    if data[:2] == b"\x1f\x8b":
        import gzip

        data = gzip.decompress(data)
    cues = parse_text(decode(data))
    if not cues:
        raise ValueError("no subtitle cues found")
    return tidy(cues)


def tidy(cues: list[Cue]) -> list[Cue]:
    cues = sorted((c for c in cues if c.end > c.start), key=lambda c: c.start)
    out: list[Cue] = []
    for c in cues:
        if out and c.start == out[-1].start and c.lines == out[-1].lines:
            continue
        out.append(Cue(c.start, c.end, list(c.lines)))
    return out


# -- sound descriptions ------------------------------------------------------------------

def is_sound_only(lines: list[str]) -> bool:
    """A cue that is entirely a sound description ("[thunder rumbles]", "♪ ♪")."""
    return not speech_text(lines).strip()


def display_lines(lines: list[str], sound_descriptions: bool) -> list[str]:
    """What the overlay shows for a cue. Off: brackets, speaker labels and music marks go."""
    if sound_descriptions:
        return [l for l in lines if l.strip()]
    out = []
    for line in lines:
        t = _BRACKETED.sub("", line)
        t = _SPEAKER.sub(lambda m: "- " if m.group(0).lstrip().startswith("-") else "", t)
        t = _MUSIC.sub("", t) if _MUSIC.search(t) and not re.search(r"[A-Za-z]{2,}", _MUSIC.sub("", t)) else t
        t = re.sub(r"\s+", " ", t).strip()
        if t and not _ONLY_PUNCT.match(t):
            out.append(t)
    if len(out) == 1 and out[0].startswith("- "):
        out[0] = out[0][2:]     # the other speaker's line was only a sound description
    return out


def speech_text(lines: list[str]) -> str:
    """The spoken words of a cue, for matching against speech recognition."""
    text = " ".join(lines)
    text = _BRACKETED.sub(" ", text)
    text = re.sub(r"(?m)(^|\s)-\s*", " ", text)
    text = _SPEAKER.sub(" ", text)
    text = re.sub(r"\b[A-Z][A-Z0-9 .'’-]{1,30}:\s", " ", text)
    if _MUSIC.search(text) or not re.search(r"[^\W_]", text):
        text = ""          # sung lyrics rarely match recognised speech; leave them out
    return re.sub(r"\s+", " ", text).strip()


def has_sound_descriptions(cues: list[Cue]) -> bool:
    """Does this track carry sound descriptions (an HI/SDH track)?"""
    if not cues:
        return False
    n = sum(1 for c in cues if _BRACKETED.search(" ".join(c.lines)) or _SPEAKER.search(c.lines[0]))
    return n >= max(5, len(cues) * 0.03)


def track_duration(cues: list[Cue]) -> float:
    return max((c.end for c in cues), default=0.0)
