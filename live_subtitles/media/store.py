"""The subtitle cache: one SQLite file in %LOCALAPPDATA%\\Live-Subtitles\\media.db.

Per title (TMDB key) it keeps the chosen track (any source, including tracks generated
from the audio), how it maps to the video (scale, and media-time segments when the
browser extension reported real positions), the viewer's nudge, and which tracks were
marked wrong. Generated sound cues are stored apart from the track, so they can be
switched off or regenerated without touching it. Manual "What is this?" answers are
remembered per page shape (`titles.site_key`).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import DATA_DIR
from ..subtitles import Cue

DB_PATH = DATA_DIR / "media.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    key TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    match TEXT NOT NULL,
    source TEXT NOT NULL,          -- candidate uid, or generated:asr / generated:translate
    cues TEXT NOT NULL,
    sounds TEXT NOT NULL DEFAULT '[]',
    scale REAL NOT NULL DEFAULT 1.0,
    segments TEXT NOT NULL DEFAULT '[]',
    nudge REAL NOT NULL DEFAULT 0.0,
    verified INTEGER NOT NULL DEFAULT 0,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS rejected (
    key TEXT NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (key, source)
);
CREATE TABLE IF NOT EXISTS aliases (
    site_key TEXT PRIMARY KEY,
    match TEXT NOT NULL,
    updated REAL NOT NULL
);
"""


def cues_to_json(cues: list[Cue]) -> str:
    return json.dumps([[round(c.start, 3), round(c.end, 3), c.lines] for c in cues], ensure_ascii=False)


def cues_from_json(text: str) -> list[Cue]:
    return [Cue(float(a), float(b), list(lines)) for a, b, lines in json.loads(text or "[]")]


@dataclass
class StoredTrack:
    key: str
    label: str
    match: dict
    source: str
    cues: list[Cue]
    sounds: list[Cue] = field(default_factory=list)
    scale: float = 1.0
    segments: list[list[float]] = field(default_factory=list)   # [[media_start, scale, offset]]
    nudge: float = 0.0
    verified: bool = False
    updated: float = 0.0

    @property
    def generated(self) -> bool:
        return self.source.startswith("generated:")


class Store:
    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # -- tracks ------------------------------------------------------------------------
    def get(self, key: str) -> StoredTrack | None:
        with self._lock:
            row = self._db.execute(
                "SELECT key, label, match, source, cues, sounds, scale, segments, nudge, verified, updated "
                "FROM tracks WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        return StoredTrack(row[0], row[1], json.loads(row[2]), row[3], cues_from_json(row[4]),
                           cues_from_json(row[5]), row[6], json.loads(row[7]), row[8], bool(row[9]), row[10])

    def put(self, t: StoredTrack) -> None:
        t.updated = time.time()
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO tracks (key, label, match, source, cues, sounds, scale, segments, nudge, "
                "verified, updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (t.key, t.label, json.dumps(t.match), t.source, cues_to_json(t.cues), cues_to_json(t.sounds),
                 t.scale, json.dumps(t.segments), t.nudge, int(t.verified), t.updated))
            self._db.commit()

    def update(self, key: str, **fields) -> None:
        allowed = {"nudge", "scale", "verified", "segments", "sounds", "cues", "source"}
        sets, values = [], []
        for k, v in fields.items():
            if k not in allowed:
                raise KeyError(k)
            if k in ("sounds", "cues"):
                v = cues_to_json(v)
            elif k == "segments":
                v = json.dumps(v)
            elif k == "verified":
                v = int(v)
            sets.append(f"{k} = ?")
            values.append(v)
        if not sets:
            return
        with self._lock:
            self._db.execute(f"UPDATE tracks SET {', '.join(sets)}, updated = ? WHERE key = ?",
                             (*values, time.time(), key))
            self._db.commit()

    def delete(self, key: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM tracks WHERE key = ?", (key,))
            self._db.execute("DELETE FROM rejected WHERE key = ?", (key,))
            self._db.commit()

    def list(self) -> list[tuple[str, str, str, float, float]]:
        """[(key, label, source, nudge, updated)], newest first."""
        with self._lock:
            return list(self._db.execute(
                "SELECT key, label, source, nudge, updated FROM tracks ORDER BY updated DESC"))

    # -- rejections ----------------------------------------------------------------------
    def reject(self, key: str, source: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR IGNORE INTO rejected (key, source) VALUES (?, ?)", (key, source))
            self._db.commit()

    def rejected(self, key: str) -> set[str]:
        with self._lock:
            return {r[0] for r in self._db.execute("SELECT source FROM rejected WHERE key = ?", (key,))}

    # -- aliases -------------------------------------------------------------------------
    def alias(self, site_key: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT match FROM aliases WHERE site_key = ?", (site_key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_alias(self, site_key: str, match: dict) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO aliases (site_key, match, updated) VALUES (?, ?, ?)",
                             (site_key, json.dumps(match), time.time()))
            self._db.commit()
