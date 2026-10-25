"""TMDB: a parsed title -> the movie or episode, with the ids every subtitle API keys on.

Takes either a v3 API key or a v4 read-access token (the long "eyJ..." one), both free
from themoviedb.org. `original_language` tells media mode whether the audio is English
(text anchors, live-caption bridge) or not (speech-span alignment, translation).
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import asdict, dataclass

from . import net
from .titles import Query, normalize_name

logger = logging.getLogger(__name__)
API = "https://api.themoviedb.org/3"


@dataclass
class Match:
    kind: str                   # movie | episode
    tmdb_id: int                # the movie, or the show for an episode
    title: str                  # movie or show title
    year: int | None = None
    season: int | None = None
    episode: int | None = None
    episode_title: str = ""
    imdb_id: str = ""           # the movie's, or the episode's
    show_imdb_id: str = ""
    original_language: str = "en"     # "" when unknown (identified without TMDB)
    runtime_s: float = 0.0      # 0 when unknown
    confidence: float = 0.0     # 0-1, how sure the title match is
    edition: str = ""           # "extended" for an extended cut: its subtitles run longer

    @property
    def key(self) -> str:
        """Cache key for the title."""
        base = f"tmdb{self.tmdb_id}" if self.tmdb_id else (self.show_imdb_id or self.imdb_id)
        cut = f":{self.edition}" if self.edition else ""
        if self.kind == "episode":
            return f"tv:{base}:s{self.season or 0}e{self.episode or 0}{cut}"
        return f"movie:{base}{cut}"

    def label(self) -> str:
        if self.kind == "episode":
            s = f"{self.title} S{self.season or 0:02d}E{self.episode or 0:02d}"
            return s + (f" · {self.episode_title}" if self.episode_title else "")
        return f"{self.title} ({self.year})" if self.year else self.title

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Match":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Tmdb:
    def __init__(self, key: str):
        self.key = key.strip()

    def _get(self, path: str, **params):
        headers, p = {}, dict(params)
        if self.key.startswith("eyJ"):
            headers["Authorization"] = f"Bearer {self.key}"
        else:
            p["api_key"] = self.key
        return net.get_json(API + path, p, headers)

    # -- search ------------------------------------------------------------------------
    def identify(self, q: Query) -> Match | None:
        if not self.key:
            raise RuntimeError("no TMDB key set")
        if q.is_episode:
            return self._episode(q)
        results = []
        if q.year:
            results = [dict(r, media_type="movie") for r in self._get("/search/movie", query=q.title,
                                                                        year=q.year).get("results", [])]
        if not results:
            results = [r for r in self._get("/search/multi", query=q.title).get("results", [])
                       if r.get("media_type") in ("movie", "tv")]
        best = self._pick(q, results)
        if not best:
            return None
        r, conf = best
        if r.get("media_type") == "tv":
            # A show page with no episode named: nothing to fetch subtitles for yet.
            return None
        return self._movie(int(r["id"]), conf)

    def lookup(self, kind: str, tmdb_id: int, season: int | None = None, episode: int | None = None) -> Match:
        """A title chosen by hand ("What is this?")."""
        if kind == "movie":
            return self._movie(tmdb_id, 1.0)
        return self._episode_details(tmdb_id, self._get(f"/tv/{tmdb_id}"), season or 1, episode or 1, 1.0)

    def search(self, text: str) -> list[dict]:
        """Raw search for the "What is this?" box: [{kind, tmdb_id, title, year, overview}]."""
        out = []
        for r in self._get("/search/multi", query=text).get("results", [])[:12]:
            kind = r.get("media_type")
            if kind not in ("movie", "tv"):
                continue
            date = r.get("release_date") or r.get("first_air_date") or ""
            out.append({"kind": kind, "tmdb_id": r["id"], "title": r.get("title") or r.get("name") or "",
                        "year": int(date[:4]) if date[:4].isdigit() else None,
                        "overview": (r.get("overview") or "")[:160]})
        return out

    # -- details -----------------------------------------------------------------------
    def _movie(self, tmdb_id: int, conf: float) -> Match:
        d = self._get(f"/movie/{tmdb_id}", append_to_response="external_ids")
        date = d.get("release_date") or ""
        return Match("movie", tmdb_id, d.get("title") or d.get("original_title") or "",
                     int(date[:4]) if date[:4].isdigit() else None,
                     imdb_id=d.get("imdb_id") or (d.get("external_ids") or {}).get("imdb_id") or "",
                     original_language=d.get("original_language") or "en",
                     runtime_s=float(d.get("runtime") or 0) * 60, confidence=conf)

    def _episode(self, q: Query) -> Match | None:
        results = self._get("/search/tv", query=q.title, first_air_date_year=q.year).get("results", [])
        if not results and q.year:
            results = self._get("/search/tv", query=q.title).get("results", [])
        best = self._pick(q, [dict(r, media_type="tv") for r in results])
        if not best:
            return None
        r, conf = best
        show = self._get(f"/tv/{r['id']}", append_to_response="external_ids")
        return self._episode_details(int(r["id"]), show, q.season or 1, q.episode or 1, conf)

    def _episode_details(self, show_id: int, show: dict, season: int, episode: int, conf: float) -> Match:
        date = show.get("first_air_date") or ""
        m = Match("episode", show_id, show.get("name") or show.get("original_name") or "",
                  int(date[:4]) if date[:4].isdigit() else None, season, episode,
                  show_imdb_id=(show.get("external_ids") or {}).get("imdb_id") or "",
                  original_language=show.get("original_language") or "en", confidence=conf)
        try:
            ep = self._get(f"/tv/{show_id}/season/{season}/episode/{episode}", append_to_response="external_ids")
            m.episode_title = ep.get("name") or ""
            m.runtime_s = float(ep.get("runtime") or 0) * 60
            m.imdb_id = (ep.get("external_ids") or {}).get("imdb_id") or ""
        except net.HttpError as exc:
            # Absolute episode numbering (anime sites count 1..N across seasons) misses here.
            logger.info("No TMDB episode S%sE%s for show %s: %s", season, episode, show_id, exc)
            m.confidence *= 0.8
        if not m.runtime_s:
            runtimes = show.get("episode_run_time") or []
            m.runtime_s = float(runtimes[0]) * 60 if runtimes else 0.0
        return m

    @staticmethod
    def _pick(q: Query, results: list[dict]) -> tuple[dict, float] | None:
        want = normalize_name(q.title)
        best, best_score = None, 0.0
        for i, r in enumerate(results[:10]):
            names = {r.get("title"), r.get("name"), r.get("original_title"), r.get("original_name")}
            sim = max((difflib.SequenceMatcher(None, want, normalize_name(n)).ratio() for n in names if n), default=0)
            date = r.get("release_date") or r.get("first_air_date") or ""
            year = int(date[:4]) if date[:4].isdigit() else None
            score = sim
            if q.year and year:
                score += 0.15 if abs(q.year - year) <= 1 else -0.25
            score += min(0.1, (r.get("popularity") or 0) / 500) - i * 0.01
            if score > best_score:
                best, best_score = r, score
        if best is None or best_score < 0.6:
            return None
        return best, min(1.0, best_score)


IMDB_SUGGEST = "https://v3.sg.media-imdb.com/suggestion/x/{}.json"


SHOW_KINDS = ("tvSeries", "tvMiniSeries")
MOVIE_KINDS = ("movie", "tvMovie", "video", "short", "tvSpecial", "tvShort")


def _suggest(text: str) -> list[dict]:
    import urllib.parse

    data = net.get_json(IMDB_SUGGEST.format(urllib.parse.quote(text.lower()[:60])))
    return [d for d in (data or {}).get("d", [])[:10] if str(d.get("id", "")).startswith("tt")]


def _best_suggestion(q: Query, kinds: tuple[str, ...]) -> tuple[dict, float] | None:
    want = normalize_name(q.title)
    best, best_score = None, 0.0
    for i, d in enumerate(_suggest(q.title)):
        if d.get("qid") not in kinds:
            continue
        score = difflib.SequenceMatcher(None, want, normalize_name(d.get("l", ""))).ratio() - i * 0.02
        if q.year and d.get("y"):
            score += 0.15 if abs(q.year - int(d["y"])) <= 1 else -0.3
        if score > best_score:
            best, best_score = d, score
    if best is None or best_score < 0.7:
        return None
    return best, min(1.0, best_score) * 0.9


def identify_keyless(q: Query) -> Match | None:
    """Without a TMDB key: IMDb's public title suggestions. No runtime or original language,
    so media mode starts by assuming English audio and switches if nothing it hears matches."""
    found = _best_suggestion(q, SHOW_KINDS if q.is_episode else MOVIE_KINDS)
    if found is None:
        return None
    best, conf = found
    if q.is_episode:
        return Match("episode", 0, best["l"], best.get("y"), q.season or 1, q.episode or 1,
                     show_imdb_id=best["id"], original_language="", confidence=conf)
    return Match("movie", 0, best["l"], best.get("y"), imdb_id=best["id"], original_language="",
                 confidence=conf)


def show_keyless(q: Query) -> str:
    """The show's name when a title with no episode in it names a TV show, else ""."""
    found = _best_suggestion(q, SHOW_KINDS)
    return found[0]["l"] if found else ""


def search_keyless(text: str) -> list[dict]:
    """The "What is this?" search without a TMDB key, shaped like `Tmdb.search`."""
    out = []
    for d in _suggest(text):
        kind = "tv" if d.get("qid") in SHOW_KINDS else "movie" if d.get("qid") in MOVIE_KINDS else ""
        if kind:
            out.append({"kind": kind, "tmdb_id": 0, "imdb_id": d["id"], "title": d.get("l", ""),
                        "year": d.get("y") or "", "overview": d.get("s", "")})
    return out


def lookup_keyless(r: dict, season: int, episode: int) -> Match:
    year = r["year"] if isinstance(r.get("year"), int) else None
    if r["kind"] == "tv":
        return Match("episode", 0, r["title"], year, season, episode, show_imdb_id=r["imdb_id"],
                     original_language="", confidence=1.0)
    return Match("movie", 0, r["title"], year, imdb_id=r["imdb_id"], original_language="", confidence=1.0)
