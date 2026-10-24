"""English subtitle sources: OpenSubtitles, Wyzie Subs, SubDL, Gestdown (Addic7ed and others).

Each source turns a `Match` into `Candidate`s and downloads one on request. Searches are
cheap; downloads count against daily quotas (OpenSubtitles' free tier especially), so the
controller downloads one candidate at a time, best score first, and caches what it gets.

All of these are fine for personal viewing. Their terms forbid bulk scraping and
redistribution, so requests stay per title and identify the app by its User-Agent.
"""

from __future__ import annotations

import logging
import math
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from . import net
from .tmdb import Match

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    source: str                 # opensubtitles | wyzie | subdl | gestdown
    id: str
    release: str = ""
    hearing_impaired: bool = False
    downloads: int = 0
    trusted: bool = False
    machine: bool = False       # machine- or AI-translated
    fps: float = 0.0
    ref: dict = field(default_factory=dict)   # what download() needs
    score: float = 0.0

    @property
    def uid(self) -> str:
        return f"{self.source}:{self.id}"

    def label(self) -> str:
        bits = [self.source]
        if self.release:
            bits.append(self.release[:60])
        if self.hearing_impaired:
            bits.append("sound descriptions")
        return " · ".join(bits)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Source:
    name = ""

    def available(self) -> bool:
        return True

    def search(self, m: Match) -> list[Candidate]:
        raise NotImplementedError

    def download(self, c: Candidate) -> bytes:
        raise NotImplementedError


class OpenSubtitles(Source):
    """api.opensubtitles.com REST v1. An API key is enough to search; a free account login
    raises the daily download allowance."""
    name = "opensubtitles"
    API = "https://api.opensubtitles.com/api/v1"

    def __init__(self, key: str, username: str = "", password: str = ""):
        self.key, self.username, self.password = key.strip(), username.strip(), password
        self._token = ""
        self._base = self.API

    def available(self) -> bool:
        return bool(self.key)

    def _headers(self, auth: bool = False) -> dict:
        h = {"Api-Key": self.key}
        if auth and self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def search(self, m: Match) -> list[Candidate]:
        params = {"languages": "en", "order_by": "download_count"}
        if m.kind == "episode":
            params.update(season_number=m.season, episode_number=m.episode)
            if m.tmdb_id:
                params["parent_tmdb_id"] = m.tmdb_id
            else:
                params["parent_imdb_id"] = m.show_imdb_id.lstrip("t")
        elif m.tmdb_id:
            params["tmdb_id"] = m.tmdb_id
        else:
            params["imdb_id"] = m.imdb_id.lstrip("t")
        data = net.get_json(self.API + "/subtitles", params, self._headers()).get("data", [])
        out = []
        for item in data:
            a = item.get("attributes", {})
            files = a.get("files") or []
            if a.get("language") != "en" or not files:
                continue
            out.append(Candidate(self.name, str(files[0]["file_id"]), a.get("release") or files[0].get("file_name") or "",
                                 bool(a.get("hearing_impaired")), int(a.get("download_count") or 0),
                                 bool(a.get("from_trusted")),
                                 bool(a.get("machine_translated") or a.get("ai_translated")),
                                 float(a.get("fps") or 0), {"file_id": files[0]["file_id"]}))
        return out

    def _login(self) -> None:
        if self._token or not (self.username and self.password):
            return
        try:
            r = net.post_json(self.API + "/login", {"username": self.username, "password": self.password},
                              self._headers())
            self._token = r.get("token") or ""
            if r.get("base_url"):
                self._base = f"https://{r['base_url']}/api/v1"
        except net.HttpError as exc:
            logger.warning("OpenSubtitles login failed: %s", exc)

    def download(self, c: Candidate) -> bytes:
        self._login()
        r = net.post_json(self._base + "/download", {"file_id": int(c.ref["file_id"])}, self._headers(auth=True))
        link = r.get("link")
        if not link:
            raise RuntimeError(r.get("message") or "OpenSubtitles gave no download link")
        logger.info("OpenSubtitles download, %s left today", r.get("remaining"))
        return net.request(link)


class Wyzie(Source):
    """sub.wyzie.io: an aggregator keyed on TMDB ids (free key, 1,000 requests a day)."""
    name = "wyzie"
    API = "https://sub.wyzie.io"

    def __init__(self, key: str):
        self.key = key.strip()

    def available(self) -> bool:
        return bool(self.key)

    def search(self, m: Match) -> list[Candidate]:
        params = {"id": m.tmdb_id or m.show_imdb_id or m.imdb_id, "language": "en", "format": "srt,ass,vtt", "source": "all", "key": self.key}
        if m.kind == "episode":
            params.update(season=m.season, episode=m.episode)
        data = net.get_json(self.API + "/search", params) or []
        out = []
        for d in data if isinstance(data, list) else []:
            if d.get("language") not in ("en", None) or not d.get("url"):
                continue
            out.append(Candidate(self.name, str(d.get("id") or d["url"]), d.get("release") or d.get("fileName") or "",
                                 bool(d.get("isHearingImpaired")), int(d.get("downloadCount") or 0),
                                 False, bool(d.get("ai")), 0.0,
                                 {"url": d["url"], "origin": d.get("source") or ""}))
        return out

    def download(self, c: Candidate) -> bytes:
        return net.request(c.ref["url"])


class SubDL(Source):
    """api.subdl.com (free API key from subdl.com)."""
    name = "subdl"
    API = "https://api.subdl.com/api/v1/subtitles"
    DL = "https://dl.subdl.com"

    def __init__(self, key: str):
        self.key = key.strip()

    def available(self) -> bool:
        return bool(self.key)

    def search(self, m: Match) -> list[Candidate]:
        params = {"api_key": self.key, "languages": "EN", "subs_per_page": 30,
                  "type": "tv" if m.kind == "episode" else "movie"}
        if m.tmdb_id:
            params["tmdb_id"] = m.tmdb_id
        else:
            params["imdb_id"] = m.show_imdb_id or m.imdb_id
        if m.kind == "episode":
            params.update(season_number=m.season, episode_number=m.episode)
        data = net.get_json(self.API, params) or {}
        out = []
        for d in data.get("subtitles") or []:
            if not d.get("url") or d.get("full_season"):
                continue
            out.append(Candidate(self.name, d["url"].rsplit("/", 1)[-1], d.get("release_name") or d.get("name") or "",
                                 bool(d.get("hi")), 0, False, False, 0.0, {"url": self.DL + d["url"]}))
        return out

    def download(self, c: Candidate) -> bytes:
        return net.request(c.ref["url"])


class Gestdown(Source):
    """api.gestdown.info: TV subtitles from Addic7ed and others, no key needed."""
    name = "gestdown"
    API = "https://api.gestdown.info"

    def search(self, m: Match) -> list[Candidate]:
        if m.kind != "episode":
            return []
        shows = net.get_json(f"{self.API}/shows/search/{urllib.parse.quote(m.title)}").get("shows") or []
        from .titles import normalize_name

        show = next((s for s in shows if m.tmdb_id and s.get("tmdbId") == m.tmdb_id), None)
        if show is None and not m.tmdb_id:
            show = next((s for s in shows if normalize_name(s.get("name", "")) == normalize_name(m.title)), None)
        if show is None:
            return []
        try:
            data = net.get_json(f"{self.API}/subtitles/get/{show['id']}/{m.season}/{m.episode}/English")
        except net.HttpError as exc:
            if exc.status in (404, 423):     # 423: the episode is still being refreshed upstream
                return []
            raise
        out = []
        for d in data.get("matchingSubtitles") or []:
            if not d.get("completed", True) or not d.get("downloadUri"):
                continue
            out.append(Candidate(self.name, d["subtitleId"], d.get("release") or d.get("version") or "",
                                 bool(d.get("hearingImpaired")), int(d.get("downloadCount") or 0),
                                 bool(d.get("corrected")), False, 0.0, {"url": self.API + d["downloadUri"]}))
        return out

    def download(self, c: Candidate) -> bytes:
        return net.request(c.ref["url"])


SOURCE_PRIOR = {"opensubtitles": 0.6, "gestdown": 0.7, "subdl": 0.4, "wyzie": 0.3}


def score(c: Candidate, sound_descriptions: bool) -> float:
    s = SOURCE_PRIOR.get(c.source, 0.0) + math.log10(1 + c.downloads)
    if c.trusted:
        s += 0.8
    if c.machine:
        s -= 5.0
    # HI tracks serve both settings (descriptions are stripped when off), so they win when
    # descriptions are wanted and cost only a little when they are not.
    s += 1.5 if c.hearing_impaired and sound_descriptions else (-0.3 if c.hearing_impaired else 0.0)
    rel = c.release.lower()
    if any(w in rel for w in ("forced", "foreign parts", "signs")):
        s -= 6.0            # forced tracks only cover foreign-language bits
    if any(w in rel for w in ("commentary",)):
        s -= 6.0
    return s


def search_all(sources: list[Source], m: Match, sound_descriptions: bool) -> list[Candidate]:
    """Every available source in parallel, scored, best first, duplicates folded."""
    live = [s for s in sources if s.available()]
    results: list[Candidate] = []

    def run(src: Source):
        try:
            return src.search(m)
        except Exception as exc:
            logger.warning("%s search failed: %s", src.name, exc)
            return []

    with ThreadPoolExecutor(max_workers=max(1, len(live))) as pool:
        for found in pool.map(run, live):
            results += found
    seen, out = set(), []
    for c in results:
        c.score = score(c, sound_descriptions)
        if c.uid in seen:
            continue
        seen.add(c.uid)
        out.append(c)
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def make_sources(media_settings) -> list[Source]:
    from .secrets import unprotect

    m = media_settings
    return [OpenSubtitles(unprotect(m.opensubtitles_key), m.opensubtitles_user, unprotect(m.opensubtitles_password)),
            Gestdown(), SubDL(unprotect(m.subdl_key)), Wyzie(unprotect(m.wyzie_key))]


def by_name(sources: list[Source]) -> dict[str, Source]:
    return {s.name: s for s in sources}
