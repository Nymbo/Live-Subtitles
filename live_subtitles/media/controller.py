"""Media mode in the app: what's playing -> a track -> synced lines on the overlay.

Runs on its own while enabled; no setup per title. Once a second it reads the foreground
window. A browser or player title that names a movie or episode (or the extension's
report of the page) starts a session: the cache is checked, otherwise every source is
searched and the best track downloaded. A background thread then feeds captured audio
to the media worker and keeps the time map current:

- English audio: recognised words -> text anchors. Until they lock, `lines_now` returns
  None and the overlay keeps showing live captions (the bridge); the app starts the
  live engine for that if it isn't running.
- Other languages: voice activity -> speech-span alignment. No live English captions
  exist to bridge with, so cues show at the saved or best-guess timing until the lock.

Also here: checking the chosen track against what's heard (wrong episode or cut -> next
candidate), generating a track when none exists (from the stream ahead of the playhead
when the extension saw its URL, else from the audio as it plays), generated sound cues
for tracks without sound descriptions, and the viewer's nudges.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from ..capture import LoopbackCapture
from ..subtitles import Cue, build_cues, Word
from . import align, sources as sources_mod, subfile, titles, winwatch
from .align import Aligner, Anchor, TrackIndex
from .clock import AudioClock, ExtensionClock
from .store import Store, StoredTrack
from .tmdb import Match, Tmdb, identify_keyless

logger = logging.getLogger(__name__)
SR = 16000
RING_S = 240.0
WORD_WINDOW_S = 10.0         # audio per recognition request
SPEECH_WINDOW_S = 180.0      # audio searched for speech-span alignment
# Measured on Tears of Steel (official SRT, mistimed): the right mapping peaks at z 4.7-6.7
# with ratio 1.35-2.2 over 2.5-6 minutes of audio; wrong ones at ratio 1.07-1.23.
SPEECH_LOCK_Z = 4.5
SPEECH_LOCK_RATIO = 1.33
MISMATCH_WORDS = 90          # words heard before judging that a track is the wrong one
MISMATCH_SCORE = 0.06
SOUND_MIN_GAP_S = 1.5        # generated sound cues only go in gaps between lines this long
SOUND_SHOW_S = 3.0
MUSIC_REPEAT_S = 30.0        # a new [music] cue only after this long without music
IDLE_END_S = 600.0           # a session with no audio this long ends


# -- audio buffer ------------------------------------------------------------------------
class AudioRing:
    """The last RING_S seconds of *playing* audio, indexed by the audio clock, with the
    page's media time at each chunk when the extension reports it."""

    def __init__(self):
        self.buf = np.zeros(0, np.float32)
        self.t0 = 0.0
        self.marks: list[tuple[float, float]] = []    # (audio tau, media time)
        self.lock = threading.Lock()

    def append(self, samples: np.ndarray, tau_start: float, media: float | None) -> None:
        with self.lock:
            if self.buf.size == 0:
                self.t0 = tau_start
            self.buf = np.concatenate([self.buf, samples])
            if media is not None:
                self.marks.append((tau_start, media))
            excess = self.buf.size - int(RING_S * SR)
            if excess > 0:
                self.buf = self.buf[excess:]
                self.t0 += excess / SR
                keep = self.t0 - 5
                self.marks = [m for m in self.marks if m[0] >= keep]

    @property
    def end(self) -> float:
        return self.t0 + self.buf.size / SR

    def get(self, a: float, b: float) -> tuple[np.ndarray, float]:
        with self.lock:
            a = max(a, self.t0)
            i, j = int((a - self.t0) * SR), int((b - self.t0) * SR)
            return self.buf[max(0, i):max(0, j)].copy(), a

    def to_media(self, tau: float) -> float | None:
        with self.lock:
            best = None
            for t, m in self.marks:
                if t <= tau:
                    best = (t, m)
                else:
                    break
        return None if best is None else best[1] + (tau - best[0])


# -- session -----------------------------------------------------------------------------
@dataclass
class Session:
    gen: int
    window_title: str
    match: Match | None = None
    english: bool = True
    track: StoredTrack | None = None
    candidates: list = field(default_factory=list)
    tried: set = field(default_factory=set)
    index: TrackIndex | None = None
    mask: np.ndarray | None = None
    aligner: Aligner = field(default_factory=Aligner)
    domain: str = "audio"               # audio: tau is the audio clock; media: video.currentTime
    words: list = field(default_factory=list)       # [(text, tau)] heard so far (domain time)
    spans: list = field(default_factory=list)       # [(a, b)] speech (domain time)
    processed_to: float = 0.0           # audio clock time up to which audio was examined
    last_speech_search: float = 0.0
    speech_misses: int = 0
    generating: str = ""                # "" | asr | translate | stream
    gen_words: list = field(default_factory=list)
    translated_to: float = 0.0
    sounds_to: float = 0.0
    sound_events: list = field(default_factory=list)   # [Cue] in track time
    live_cues: list = field(default_factory=list)      # translated lines as they arrive (shown late)
    state: str = "identifying"          # identifying | unknown | searching | aligning | synced | generating | none
    detail: str = ""
    last_audio: float = field(default_factory=time.monotonic)
    gen_proc: subprocess.Popen | None = None
    dirty: bool = False
    saved_at: float = 0.0

    @property
    def cues(self) -> list[Cue]:
        return self.track.cues if self.track else []


class MediaController(QObject):
    status_changed = Signal(str, str)        # state, detail (for the overlay placeholder and tray)
    bridge_wanted = Signal(bool)             # English session not yet synced: live captions please
    notify = Signal(str, str)
    unknown_title = Signal(str)              # a watchable window whose title named nothing
    _player_report = Signal(dict)            # extension report, moved onto the UI thread

    def __init__(self, settings, store: Store | None = None):
        super().__init__()
        self.s = settings
        self.store = store or Store()
        self.session: Session | None = None
        self._gen = 0
        self._lock = threading.Lock()
        self.clock = AudioClock()
        self.ext = ExtensionClock()
        self.ext_state: dict = {}
        self.ring = AudioRing()
        self.capture: LoopbackCapture | None = None
        self.worker = None
        self.running = False
        self._threads: list[threading.Thread] = []
        self._watch_timer = QTimer(self, interval=1000, timeout=self._watch)
        self._last_title = ""
        self._asked_unknown: set[str] = set()
        self._player_report.connect(self._on_player_ui)
        self.web = None

    @property
    def last_title(self) -> str:
        return self._last_title

    # -- lifecycle -----------------------------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        self.running = True
        from .worker import MediaWorker

        self.worker = MediaWorker(gpu_words=False, gpu_translate=self.s.files.use_gpu)
        self.capture = LoopbackCapture(self.s.engine.device, self.s.engine.dialogue_boost)
        try:
            self.capture.start()
        except Exception:
            logger.exception("Media capture could not start")
        for target, name in ((self._feed_loop, "media-feed"), (self._sync_loop, "media-sync")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        self._watch_timer.start()
        self._start_web()
        self.status_changed.emit("watching", "")
        logger.info("Media mode on")

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self._watch_timer.stop()
        self._end_session()
        if self.capture:
            self.capture.stop()
            self.capture = None
        if self.worker:
            self.worker.close()
        self._threads.clear()
        self.status_changed.emit("off", "")
        logger.info("Media mode off")

    def shutdown(self) -> None:
        self.stop()
        if self.web:
            self.web.stop()
        if self.store:
            self.store.close()

    def restart_capture(self) -> None:
        if self.running and self.capture:
            self.capture.stop()
            self.capture = LoopbackCapture(self.s.engine.device, self.s.engine.dialogue_boost)
            self.capture.start()

    def _start_web(self) -> None:
        if self.web is None and self.s.media.extension:
            from .web import WebServer

            self.web = WebServer(self._on_player_thread, self._status_dict, self.s.media.port)
            self.web.start()

    # -- audio in ------------------------------------------------------------------------
    def _feed_loop(self) -> None:
        import queue

        while self.running:
            cap = self.capture
            if cap is None:
                time.sleep(0.2)
                continue
            try:
                chunk = cap.chunks.get(timeout=0.25)
            except queue.Empty:
                continue
            tau = self.clock.tau
            if self.clock.feed(chunk):
                media = self.ext.now() if self.ext.fresh() and not self.ext.paused else None
                self.ring.append(chunk, tau, media)
                s = self.session
                if s:
                    s.last_audio = time.monotonic()

    # -- what is playing -----------------------------------------------------------------
    def _watch(self) -> None:
        s = self.session
        if s and time.monotonic() - s.last_audio > IDLE_END_S and not self.ext.fresh():
            self._end_session()
        title = self.ext_state.get("title") if self.ext.fresh() else None
        from_player = False
        if not title:
            w = winwatch.foreground()
            if w is None or not w.watchable:
                return
            title, from_player = w.title, w.is_player
        if title == self._last_title:
            return
        self._last_title = title
        if not self.clock.playing() and not self.ext.fresh():
            # Browsing, not watching. Look again when the title changes or audio starts.
            self._last_title = ""
            return
        self._consider(title, from_player)

    def _consider(self, title: str, from_player: bool = False) -> None:
        alias = self.store.alias(titles.site_key(title))
        if alias is None and not titles.looks_like_media(title, from_player):
            return              # YouTube, a news article, a search page: not something to subtitle
        q = titles.parse(title)
        if alias is None and q is None:
            key = titles.site_key(title)
            if self.session is None and key not in self._asked_unknown:
                self._asked_unknown.add(key)
                self.unknown_title.emit(title)
            return
        cur = self.session
        if cur:
            same = (alias is not None and cur.match is not None and Match.from_dict(alias).key == cur.match.key)                 or (alias is None and q is not None and titles.parse(cur.window_title) == q)
            if same:
                cur.window_title = title
                return
        self._new_session(title, Match.from_dict(alias) if alias else None, q)

    def identify_manually(self, window_title: str, match: Match) -> None:
        """The answer to "What is this?": remember it for this page and play it."""
        self.store.set_alias(titles.site_key(window_title or self._last_title), match.to_dict())
        self._new_session(window_title or self._last_title, match, None)

    def _new_session(self, title: str, match: Match | None, q) -> None:
        self._end_session()
        with self._lock:
            self._gen += 1
            s = Session(self._gen, title)
            s.processed_to = s.sounds_to = s.translated_to = self.ring.end
            self.session = s
        self._status(s, "identifying", q.label() if q else (match.label() if match else title))
        threading.Thread(target=self._load, args=(s, match, q), name="media-load", daemon=True).start()

    def _end_session(self) -> None:
        with self._lock:
            s, self.session = self.session, None
        if s is None:
            return
        self._gen += 1
        if s.gen_proc and s.gen_proc.poll() is None:
            s.gen_proc.kill()
        try:
            self._save(s, final=True)
        except Exception:
            logger.exception("Could not save the session's track")
        self.bridge_wanted.emit(False)
        self.status_changed.emit("watching", "")

    def _current(self, s: Session) -> bool:
        return self.running and self.session is s and s.gen == self._gen

    # -- loading a track -----------------------------------------------------------------
    def _load(self, s: Session, match: Match | None, q) -> None:
        try:
            if match is None:
                from .secrets import unprotect

                tmdb = Tmdb(unprotect(self.s.media.tmdb_key))
                match = tmdb.identify(q) if tmdb.key else identify_keyless(q)
                if match is None:
                    self._status(s, "unknown", f"Couldn't find \"{q.title}\" · right-click → What is this?")
                    self.unknown_title.emit(s.window_title)
                    return
            if not self._current(s):
                return
            s.match = match
            s.english = match.original_language in ("en", "") or self.s.media.audio_language == "en"
            if s.english:
                self.bridge_wanted.emit(True)
            stored = self.store.get(match.key)
            if stored:
                self._use_track(s, stored, "cached")
                return
            self._status(s, "searching", match.label())
            srcs = sources_mod.make_sources(self.s.media)
            s.candidates = [c for c in sources_mod.search_all(srcs, match, self.s.media.sound_descriptions)
                            if c.uid not in self.store.rejected(match.key)]
            if not self._current(s):
                return
            if not self._next_candidate(s, srcs):
                self._start_generation(s, "No subtitles found")
        except Exception as exc:
            logger.exception("Media load failed")
            if self._current(s):
                self._status(s, "unknown", f"Lookup failed: {exc}"[:90])

    def _next_candidate(self, s: Session, srcs=None) -> bool:
        """Download the best untried candidate that parses and fits the runtime."""
        srcs = sources_mod.by_name(srcs or sources_mod.make_sources(self.s.media))
        for c in s.candidates:
            if c.uid in s.tried:
                continue
            s.tried.add(c.uid)
            src = srcs.get(c.source)
            if src is None:
                continue
            try:
                cues = subfile.load_bytes(src.download(c))
            except Exception as exc:
                logger.info("Candidate %s failed: %s", c.uid, exc)
                continue
            m = s.match
            dur = subfile.track_duration(cues)
            if m.runtime_s and (dur > m.runtime_s * 1.15 + 180 or dur < m.runtime_s * 0.5):
                logger.info("Candidate %s runs %.0fs for a %.0fs title; skipped", c.uid, dur, m.runtime_s)
                continue
            if len(cues) < 20:
                continue
            track = StoredTrack(m.key, m.label(), m.to_dict(), c.uid, cues)
            if not self._current(s):
                return True
            self.store.put(track)
            self._use_track(s, track, c.label())
            return True
        return False

    def _use_track(self, s: Session, track: StoredTrack, how: str) -> None:
        s.track = track
        s.index = TrackIndex(track.cues) if s.english else None
        s.mask = align.cue_mask(track.cues, subfile.track_duration(track.cues) + 600)
        s.aligner = Aligner()
        s.aligner.scale_hint = track.scale if track.scale != 1.0 else None
        s.words.clear()
        s.spans.clear()
        s.sound_events = list(track.sounds)
        if self.ext.fresh():
            self._enter_media_domain(s)
        if track.source == "generated:stream" and s.domain == "media":
            # Generated from the stream itself: already in the video's own time.
            s.aligner.segments.append(align.Segment(-1e9, 1.0, 0.0))
        self._status(s, "aligning", how)
        logger.info("Track for %s: %s (%d cues)", track.label, track.source, len(track.cues))

    def _enter_media_domain(self, s: Session) -> None:
        """The extension reports real positions: map media time, not the audio clock."""
        if s.domain == "media":
            return
        s.domain = "media"
        s.aligner.reset()
        s.words.clear()
        s.spans.clear()
        tr = s.track
        if tr and tr.segments:
            for m0, sc, off in tr.segments:
                s.aligner.segments.append(align.Segment(m0, sc, off))
            s.state = "synced"

    # -- the sync thread -----------------------------------------------------------------
    def _sync_loop(self) -> None:
        while self.running:
            time.sleep(0.5)
            s = self.session
            if s is None or s.match is None:
                continue
            try:
                if s.english:
                    self._sync_english(s)
                else:
                    self._sync_foreign(s)
                self._sounds(s)
                if s.dirty and time.monotonic() - s.saved_at > 20:
                    s.saved_at = time.monotonic()
                    self._save(s)
            except RuntimeError as exc:
                logger.warning("Media sync: %s", exc)
                time.sleep(2)
            except Exception:
                logger.exception("Media sync failed")
                time.sleep(2)

    def _to_domain(self, s: Session, tau: float) -> float | None:
        return tau if s.domain == "audio" else self.ring.to_media(tau)

    def _sync_english(self, s: Session) -> None:
        # Recognition runs on every stretch of audio, synced or not: it costs about 0.7 s of
        # CPU per 10 s window, and continuous anchors catch frame-rate drift, seeks and cuts
        # within a window or two.
        end = self.ring.end
        if s.track is None and s.generating != "asr":
            return
        if end - s.processed_to < WORD_WINDOW_S:
            return
        if end - s.processed_to > 3 * WORD_WINDOW_S:
            s.processed_to = end - WORD_WINDOW_S      # fell behind (a long load): keep up with now
        audio, t0 = self.ring.get(s.processed_to, s.processed_to + WORD_WINDOW_S)
        s.processed_to = t0 + audio.size / SR
        if audio.size < SR:
            return
        words = self.worker.call("words", audio, t0=t0)["words"]
        heard = []
        for text, a, b in words:
            da, db = self._to_domain(s, a), self._to_domain(s, b)
            if da is not None:
                heard.append((text, da, db))
        if not self._current(s):
            return
        if s.generating == "asr":
            s.gen_words += [Word(t, a, b) for t, a, b in heard]
            s.dirty = True
            return
        s.words += [(t, a) for t, a, _ in heard]
        result = s.aligner.add(s.index.anchors([(t, a) for t, a, _ in heard]))
        if result == "locked":
            score = align.match_score(s.index, s.words)
            self._status(s, "synced", self._source_label(s))
            s.track.verified = score >= 0.2
            s.track.scale = s.aligner.current.scale
            s.dirty = True
        elif result == "jumped":
            logger.info("Timing jumped (seek or cut) at %.1f", s.aligner.current.tau_start)
            s.dirty = True
        elif result == "refined":
            s.dirty = True
        if not s.aligner.locked and len(s.words) >= MISMATCH_WORDS:
            score = align.match_score(s.index, s.words)
            if score < MISMATCH_SCORE:
                self._wrong_track(s, automatic=True)

    def _sync_foreign(self, s: Session) -> None:
        end = self.ring.end
        if end - s.processed_to >= 10.0:
            audio, t0 = self.ring.get(s.processed_to, end)
            s.processed_to = t0 + audio.size / SR
            spans = self.worker.call("vad", audio, t0=t0)["spans"]
            for a, b in spans:
                da, db = self._to_domain(s, a), self._to_domain(s, b)
                if da is not None:
                    s.spans.append((da, db))
        if s.generating == "translate":
            self._translate_live(s)
            return
        if s.track is None or not s.spans:
            return
        now = time.monotonic()
        locked = s.aligner.locked
        if now - s.last_speech_search < (30.0 if locked else 15.0):
            return
        s.last_speech_search = now
        t1 = s.spans[-1][1]
        cur = s.aligner.current
        if cur:
            scales = [cur.scale]
        else:
            scales = [s.aligner.scale_hint] if s.aligner.scale_hint else []
            scales += [1.0, 25 / 23.976, 23.976 / 25]
        best = None
        for scale in dict.fromkeys(scales):
            if locked:
                m = align.speech_align(s.mask, s.spans, t1 - 90, t1, scale, near=cur.offset, radius=15)
            else:
                m = align.speech_align(s.mask, s.spans, max(s.spans[0][0], t1 - SPEECH_WINDOW_S), t1, scale)
            if m and (best is None or m.z > best.z):
                best = m
        strong = best is not None and best.z >= (3.5 if locked else SPEECH_LOCK_Z) and best.ratio >= (1.15 if locked else SPEECH_LOCK_RATIO)
        if not strong:
            if locked:
                s.speech_misses += 1
                if s.speech_misses >= 2:      # lost it: a seek or a cut; search everywhere again
                    m = align.speech_align(s.mask, s.spans, max(s.spans[0][0], t1 - 120), t1, cur.scale)
                    if m and m.z >= SPEECH_LOCK_Z and m.ratio >= SPEECH_LOCK_RATIO:
                        best, strong = m, True
            if not strong:
                return
        s.speech_misses = 0
        recent = [(a, b) for a, b in s.spans if a >= t1 - 90]
        anchors = [Anchor(a, best.scale * a + best.offset, min(1.5, best.z / 6), "speech") for a, _ in recent]
        result = s.aligner.add(anchors)
        if result == "locked":
            self._status(s, "synced", self._source_label(s))
            s.track.scale = s.aligner.current.scale
            s.dirty = True
        elif result in ("jumped", "refined"):
            s.dirty = True

    # -- generation ----------------------------------------------------------------------
    def _start_generation(self, s: Session, why: str) -> None:
        m = s.match
        media = [x for x in self.ext_state.get("media", []) if x.get("url")] if self.ext.fresh() else []
        if media:
            self._generate_from_stream(s, media[0]["url"], self.ext_state.get("page", ""))
            return
        s.generating = "asr" if s.english else "translate"
        s.track = StoredTrack(m.key, m.label(), m.to_dict(), f"generated:{s.generating}", [])
        s.aligner = Aligner()
        s.aligner.segments.append(align.Segment(-1e9, 1.0, 0.0))     # generated cues are in heard time
        s.translated_to = self.ring.end
        detail = (f"{why} · live captions now, saved for next time" if s.english
                  else f"{why} · translating from the audio")
        self._status(s, "generating", detail)

    def generate_now(self) -> None:
        s = self.session
        if s and s.match:
            if s.track and not s.track.generated:
                self.store.reject(s.match.key, s.track.source)
            threading.Thread(target=self._start_generation, args=(s, "Generating"), daemon=True).start()

    def _generate_from_stream(self, s: Session, url: str, page: str) -> None:
        from ..unified import _python_exe

        m = s.match
        s.generating = "stream"
        if s.domain != "media":
            self._enter_media_domain(s)
        s.track = StoredTrack(m.key, m.label(), m.to_dict(), "generated:stream", [])
        s.aligner = Aligner()
        s.aligner.segments.append(align.Segment(-1e9, 1.0, 0.0))
        start = max(0.0, self.ext.now() - 5)
        cmd = [_python_exe(), "-m", "live_subtitles", "media-generate", url, "--start", f"{start:.2f}",
               "--referer", page]
        if not s.english:
            cmd += ["--translate", m.original_language]
        if self.s.media.generated_sounds and self.s.media.sound_descriptions:
            cmd.append("--sounds")
        if not self.s.files.use_gpu:
            cmd.append("--cpu")
        from .worker import PROJECT_DIR

        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        s.gen_proc = subprocess.Popen(cmd, cwd=PROJECT_DIR, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      text=True, encoding="utf-8", errors="replace", env=env,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self._status(s, "generating", "Making subtitles from the stream, ahead of playback")
        threading.Thread(target=self._read_generator, args=(s,), daemon=True, name="media-gen").start()

    def _read_generator(self, s: Session) -> None:
        for line in s.gen_proc.stdout:
            if not self._current(s):
                break
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            kind = ev.get("event")
            if kind == "cues":
                cues = [Cue(a, b, lines) for a, b, lines in ev["cues"]]
                s.track.cues = subfile.tidy(s.track.cues + cues)
                s.dirty = True
                if s.state != "synced":
                    self._status(s, "synced", "Generated from the stream")
            elif kind == "sounds":
                s.sound_events += [Cue(a, b, lines) for a, b, lines in ev["cues"]]
                s.dirty = True
            elif kind == "error":
                self._status(s, "generating", f"Generation failed: {ev.get('message', '')}"[:90])
        if self._current(s):
            self._save(s)

    def _translate_live(self, s: Session) -> None:
        end = self.ring.end
        if end - s.translated_to < 3.0:
            return
        audio, t0 = self.ring.get(max(s.translated_to, end - 25.0), end)
        if audio.size < SR:
            return
        segs = self.worker.call("translate", audio, t0=t0, language=s.match.original_language)["segments"]
        window_end = t0 + audio.size / SR
        done_to = s.translated_to
        new = []
        for a, b, text in segs:
            if b >= window_end - 0.5:        # still being spoken; next round
                continue
            if a < s.translated_to - 0.2:
                continue
            da, db = self._to_domain(s, a), self._to_domain(s, b)
            if da is None:
                continue
            new.append((da, max(db, da + 1.5), text))
            done_to = max(done_to, b)
        if new:
            from .generate import text_cues

            cues = text_cues(new)
            s.track.cues = subfile.tidy(s.track.cues + cues)
            s.dirty = True
            # Each line is ready only after it was spoken, so its own slot has passed: show it
            # now, for as long as it was spoken (saved with its real times for next time).
            now = self.position(s)
            for c in cues:
                dur = max(2.0, c.end - c.start)
                start = max(now, s.live_cues[-1].end if s.live_cues else now)
                s.live_cues.append(Cue(start, start + dur, c.lines))
                now = start + dur
            s.live_cues = s.live_cues[-20:]
        s.translated_to = done_to if new else max(s.translated_to, window_end - 25.0)

    # -- generated sound cues ------------------------------------------------------------
    def _wants_sounds(self, s: Session) -> bool:
        m = self.s.media
        return (m.sound_descriptions and m.generated_sounds and s.track is not None
                and s.generating != "stream" and not subfile.has_sound_descriptions(s.track.cues)
                and (s.aligner.locked or s.generating))

    def _sounds(self, s: Session) -> None:
        if not self._wants_sounds(s):
            s.sounds_to = self.ring.end
            return
        end = self.ring.end
        if end - s.sounds_to < 10.5:
            return
        audio, t0 = self.ring.get(max(s.sounds_to, end - 30), end)
        s.sounds_to = t0 + audio.size / SR
        tags = self.worker.call("tags", audio, t0=t0)["tags"]
        for caption, a, b, _p in tags:
            da, db = self._to_domain(s, a), self._to_domain(s, b)
            if da is None or s.aligner.track_time(da) is None:
                continue
            ca, cb = s.aligner.track_time(da), s.aligner.track_time(db)
            same = [e for e in s.sound_events[-8:] if e.lines == [caption]]
            # A film score is "music" most of the time; say so once in a while, not in every gap.
            recent = MUSIC_REPEAT_S if caption == "[music]" else 2.0
            if same and same[-1].end >= ca - recent:
                same[-1].end = max(same[-1].end, cb)
                continue
            s.sound_events.append(Cue(ca, cb, [caption]))
            s.dirty = True

    # -- what the overlay shows ----------------------------------------------------------
    def active(self) -> bool:
        s = self.session
        return bool(s and s.track is not None and s.state in ("aligning", "synced", "generating"))

    def position(self, s: Session) -> float:
        if s.domain == "media" and self.ext.fresh():
            return self.ext.now()
        return self.clock.now()

    def lines_now(self) -> list[str] | None:
        """The lines to show now, [] for none, or None to let live captions show."""
        s = self.session
        if s is None or s.track is None:
            return None
        if s.generating == "asr":
            return None
        if s.generating == "translate":
            cue = _cue_at(s.live_cues, self.position(s))
            return subfile.display_lines(cue.lines, self.s.media.sound_descriptions) if cue else []
        seg = s.aligner.current
        if seg is None:
            if s.english:
                return None          # the live-caption bridge until text anchors lock
            if s.domain != "media":
                # No live captions to bridge with, and with only the audio clock there is no
                # honest guess: lines at a wrong offset are worse than none.
                return ["Syncing subtitles…"] if self.clock.playing() else []
            # The page's own clock: track times are close to media times on most releases.
            seg = align.Segment(0.0, s.track.scale or 1.0, 0.0)
        tau = self.position(s)
        c = seg.scale * tau + seg.offset + (s.track.nudge if s.track else 0.0)
        sd = self.s.media.sound_descriptions
        cue = _cue_at(s.cues, c)
        if cue is not None:
            lines = subfile.display_lines(cue.lines, sd)
            if lines:
                return lines
        if sd and s.sound_events:
            ev = _cue_at(s.sound_events, c, show=SOUND_SHOW_S)
            if ev is not None and _gap_around(s.cues, c) >= SOUND_MIN_GAP_S:
                return list(ev.lines)
        return []

    # -- viewer actions ------------------------------------------------------------------
    def nudge(self, delta: float) -> None:
        s = self.session
        if s and s.track:
            s.track.nudge = round(s.track.nudge + delta, 2)
            self.store.update(s.track.key, nudge=s.track.nudge)
            self._status(s, s.state, f"Timing {s.track.nudge:+.1f} s")

    def wrong_subtitles(self) -> None:
        s = self.session
        if s and s.match:
            threading.Thread(target=self._wrong_track, args=(s, False), daemon=True).start()

    def _wrong_track(self, s: Session, automatic: bool) -> None:
        if automatic and s.english and s.match.original_language == "" and self.s.media.audio_language != "en":
            # Identified without TMDB, so the audio language was a guess. Nothing heard matches
            # the English text: more likely the audio isn't English than that the track is wrong.
            s.english = False
            s.match.original_language = "und"
            s.aligner = Aligner()
            self.bridge_wanted.emit(False)
            self._status(s, "aligning", "The audio isn't English · syncing by speech timing")
            return
        if s.track and not s.track.generated:
            self.store.reject(s.match.key, s.track.source)
            s.tried.add(s.track.source)
        old = s.track
        s.track, s.index = None, None
        self._status(s, "searching", "Trying other subtitles" if not automatic else "These don't match · trying others")
        if not s.candidates:
            s.candidates = [c for c in sources_mod.search_all(sources_mod.make_sources(self.s.media), s.match,
                                                              self.s.media.sound_descriptions)
                            if c.uid not in self.store.rejected(s.match.key)]
        heard = list(s.words)
        if self._next_candidate(s):
            if heard and s.index is not None:
                # Re-use what was already heard, so the new track can lock at once.
                s.words = heard
                s.aligner.add(s.index.anchors(heard))
            return
        if old is not None:
            self.store.delete(s.match.key)
        self._start_generation(s, "Nothing else matched")

    def set_sound_descriptions(self, on: bool) -> None:
        self.s.media.sound_descriptions = on

    # -- extension -----------------------------------------------------------------------
    def _on_player_thread(self, data: dict) -> dict:
        self.ext.report(data.get("position", 0.0), data.get("paused", True), data.get("rate", 1.0),
                        data.get("duration", 0.0))
        self.ext_state = data
        self._player_report.emit(data)
        return {"ok": True, "state": self.session.state if self.session else "watching"}

    def _on_player_ui(self, data: dict) -> None:
        s = self.session
        if s and s.track is not None and s.domain != "media":
            self._enter_media_domain(s)
        title = data.get("title") or ""
        if title and title != self._last_title and not data.get("paused", True):
            self._last_title = title
            self._consider(title)

    def _status_dict(self) -> dict:
        s = self.session
        return {"running": self.running, "state": s.state if s else "watching", "detail": s.detail if s else "",
                "title": s.match.label() if s and s.match else ""}

    # -- saving and status ---------------------------------------------------------------
    def _save(self, s: Session, final: bool = False) -> None:
        tr = s.track
        if tr is None or not s.dirty:
            return
        s.dirty = False
        if s.generating == "asr" and s.gen_words:
            tr.cues = build_cues(sorted(s.gen_words, key=lambda w: w.start))
        if tr.generated and not tr.cues:
            return
        if s.aligner.locked and not tr.generated:
            tr.scale = s.aligner.current.scale
            if s.domain == "media":
                tr.segments = [[g.tau_start, g.scale, g.offset] for g in s.aligner.segments]
        tr.sounds = [c for c in s.sound_events if c.end > c.start]
        self.store.put(tr)

    def _source_label(self, s: Session) -> str:
        if s.track is None:
            return ""
        src = s.track.source
        if src.startswith("generated:"):
            return "Generated subtitles"
        return {"opensubtitles": "OpenSubtitles", "wyzie": "Wyzie", "subdl": "SubDL",
                "gestdown": "Addic7ed"}.get(src.split(":")[0], src)

    def _status(self, s: Session, state: str, detail: str) -> None:
        if not self._current(s):
            return
        s.state, s.detail = state, detail
        title = s.match.label() if s.match else ""
        text = {
            "identifying": f"Finding out what's playing · {detail}",
            "unknown": detail,
            "searching": f"Finding subtitles · {detail}",
            "aligning": f"Syncing subtitles · {title}",
            "synced": f"{title} · {detail}",
            "generating": f"{title} · {detail}",
        }.get(state, detail)
        self.status_changed.emit(state, text)


def _cue_at(cues: list[Cue], t: float, show: float | None = None) -> Cue | None:
    """The cue showing at track time t (binary search; cues are sorted by start)."""
    lo, hi = 0, len(cues)
    while lo < hi:
        mid = (lo + hi) // 2
        if cues[mid].start <= t:
            lo = mid + 1
        else:
            hi = mid
    for i in (lo - 1, lo - 2):
        if 0 <= i < len(cues):
            c = cues[i]
            end = min(c.end, c.start + show) if show else c.end
            if c.start <= t < end:
                return c
    return None


def _gap_around(cues: list[Cue], t: float) -> float:
    """Length of the dialogue-free gap that contains t."""
    prev_end, next_start = -1e9, 1e9
    for c in cues:
        if c.end <= t:
            prev_end = max(prev_end, c.end)
        elif c.start > t:
            next_start = min(next_start, c.start)
            break
        else:
            return 0.0
    return next_start - prev_end
