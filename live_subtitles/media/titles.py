"""Window and page titles -> what is playing.

Browsers put the page title in the window title ("Watch The Bear Season 2 Episode 4
Online Free - FMovies - Google Chrome"); desktop players put the file name there
("The.Bear.S02E04.1080p.WEB.mkv - VLC media player"). `parse` strips the browser or
player suffix, site branding and release noise, and pulls out the title, year, season
and episode. It returns None for titles that name nothing (a new tab, "Netflix").

`site_key` reduces a title to its site-specific shape (season/episode numbers removed),
so an answer to "What is this?" can be remembered for that page, not just that episode.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

APP_SUFFIXES = re.compile(
    r"\s*(?:[-–—|]\s*)?"
    r"(?:Google Chrome|Chromium|Mozilla Firefox|Firefox|Microsoft\s*Edge|Brave|Opera|Vivaldi|Arc|"
    r"VLC media player|MPC-HC|MPC-BE|mpv|PotPlayer|Media Player Classic|Movies & TV|Media Player)\s*$",
    re.I)
# Edge and Chrome put the profile between the page and the browser ("... - Nymbo - Microsoft
# Edge"), and "and 2 more pages" after the page when a window has several tabs.
PROFILE_SEGMENT = re.compile(r"\s*[-–—]\s*([^-–—]{1,40})$")
MORE_PAGES = re.compile(r"\s+and \d+ more pages?$", re.I)
STOCK_PROFILES = {"personal", "work", "guest", "inprivate", "default", "person 1", "your chrome"}
PROFILE_ROOTS = ("Microsoft/Edge/User Data", "Google/Chrome/User Data", "BraveSoftware/Brave-Browser/User Data",
                 "Vivaldi/User Data")
_profiles: set[str] | None = None


def browser_profiles() -> set[str]:
    """Profile names of the Chromium browsers on this machine (lower case), read once."""
    global _profiles
    if _profiles is None:
        names = set(STOCK_PROFILES)
        local = os.environ.get("LOCALAPPDATA")
        for root in PROFILE_ROOTS if local else ():
            try:
                state = json.loads((Path(local) / root / "Local State").read_text(encoding="utf-8"))
                for info in state.get("profile", {}).get("info_cache", {}).values():
                    if info.get("name"):
                        names.add(info["name"].strip().lower())
            except (OSError, ValueError, AttributeError):
                continue
        _profiles = names
    return _profiles
ZERO_WIDTH = re.compile(r"[​-‏⁠﻿]")
SEPARATORS = re.compile(r"\s+[-–—|:•·]+\s+|\s*\|\s*|\s+::\s+")
RELEASE_NOISE = re.compile(
    r"\b(?:2160p|1080p|720p|480p|4k|uhd|hdr10?\+?|dv|web[- ]?dl|webrip|web|bluray|blu[- ]?ray|brrip|bdrip|"
    r"hdrip|dvdrip|hdtv|x26[45]|h\.?26[45]|hevc|avc|aac\d?(?:\.\d)?|ddp?\d?(?:\.\d)?|atmos|10bit|"
    r"proper|repack|extended|remastered|amzn|nf|dsnp|hmax|atvp|multi|subbed|dubbed)\b.*$", re.I)
SITE_WORDS = {
    "fmovies", "123movies", "123movie", "soap2day", "putlocker", "hdtoday", "flixhq", "sflix", "myflixer",
    "lookmovie", "gomovies", "yesmovies", "bflix", "solarmovie", "primewire", "movies2watch", "zoro",
    "aniwatch", "hianime", "9anime", "gogoanime", "kissasian", "dramacool", "asianc", "viewasian",
    "watchasian", "kissanime", "animepahe", "crunchyroll", "tubi", "pluto tv", "plex", "jellyfin", "emby",
    "youtube", "vimeo", "dailymotion", "netflix", "hulu", "prime video", "disney+", "max", "peacock",
    "paramount+", "apple tv", "apple tv+", "kodi", "stremio", "vidsrc", "2embed", "vidplay", "upcloud",
}
# Sites whose videos are never a movie or episode: their titles are ignored outright.
NOT_MEDIA = {"youtube", "twitch", "tiktok", "vimeo", "dailymotion", "facebook", "instagram", "twitter", "x",
             "reddit", "spotify", "soundcloud", "discord", "slack", "linkedin", "kick", "bilibili", "rumble"}
MEDIA_HINTS = re.compile(
    r"\b(?:watch(?:ing)?|full\s+movie|movie|film|episode|season|online\s+free|free\s+online|"
    r"eng(?:lish)?\s*sub(?:bed)?|subbed|dubbed|streaming|full\s+hd)\b", re.I)
GENERIC = {"new tab", "home", "watch", "movies", "tv shows", "loading", "untitled", "player", "video",
           "streaming", "online", "free", "search", "browse", "start page", "settings", "history"}
PREFIX = re.compile(r"^(?:watch(?:ing)?|stream|download|now playing)\s*[:\-]?\s+", re.I)
SUFFIX = re.compile(
    r"\s+(?:(?:full\s+)?(?:movie|film|series|tv\s+show|episode)s?|online|free|for\s+free|in\s+hd|hd|"
    r"full\s*hd|streaming|stream|eng\s*sub|english\s+sub(?:bed|title[sd]?)?|sub(?:bed)?|dub(?:bed)?|"
    r"with\s+english\s+subtitles|on\s+\w+|superfan|extended(?:\s+(?:cut|edition))?)\s*$", re.I)
EPISODE_PATTERNS = (
    re.compile(r"\bS(\d{1,2})\s*[.\-_ ]?\s*E(\d{1,3})\b", re.I),
    re.compile(r"\b(\d{1,2})x(\d{2,3})\b"),
    re.compile(r"\bSeason\s*(\d{1,2})\s*(?:[,:\-–]\s*)?(?:Episode|Ep\.?|E)\s*(\d{1,3})\b", re.I),
    re.compile(r"\bS(\d{1,2})\s+Ep?\.?\s*(\d{1,3})\b", re.I),
)
EPISODE_ONLY = re.compile(r"\b(?:Episode|Ep\.?)\s*(\d{1,4})\b", re.I)
SEASON_ONLY = re.compile(r"\bSeason\s*(\d{1,2})\b", re.I)
EDITION = re.compile(r"\b(?:superfan|extended)\b", re.I)
YEAR = re.compile(r"[\(\[]?\b(19[2-9]\d|20[0-4]\d)\b[\)\]]?")


@dataclass(frozen=True)
class Query:
    title: str
    year: int | None = None
    season: int | None = None
    episode: int | None = None
    edition: str = ""           # "extended" for extended cuts (Peacock's Superfan episodes)

    @property
    def is_episode(self) -> bool:
        return self.episode is not None

    def label(self) -> str:
        s = self.title
        if self.year:
            s += f" ({self.year})"
        if self.episode is not None:
            s += f" S{self.season or 1:02d}E{self.episode:02d}"
        return s


def _is_profile(name: str) -> bool:
    n = name.strip().lower()
    return n in browser_profiles() or bool(re.fullmatch(r"profile \d+", n))


def strip_app(title: str) -> str:
    t = ZERO_WIDTH.sub("", title).strip()
    stripped = APP_SUFFIXES.sub("", t).strip()
    if stripped != t:
        t = stripped
        # Edge: "Page and 1 more page - Nymbo - Microsoft Edge"
        while (m := PROFILE_SEGMENT.search(t)) and _is_profile(m.group(1)):
            t = t[:m.start()].strip()
        t = MORE_PAGES.sub("", t).strip()
    return t


def _looks_like_filename(t: str) -> bool:
    return bool(re.search(r"\.(?:mkv|mp4|avi|m4v|mov|webm|ts|wmv)$", t, re.I)) or (
        t.count(".") >= 3 and t.count(" ") <= 1)


def _is_site(seg: str) -> bool:
    s = re.sub(r"\.(?:to|tv|com|net|org|io|cc|ru|se|is|sx|lol|nz|pe|bz|gg|co|me|ws)\b.*$", "", seg.lower()).strip()
    s = re.sub(r"[^a-z0-9+ ]", "", s).strip()
    return s in SITE_WORDS or any(s == w.replace(" ", "") for w in SITE_WORDS)


def _clean_title(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip(" -–—|:,.")
    for _ in range(4):
        t2 = SUFFIX.sub("", PREFIX.sub("", t)).strip(" -–—|:,.")
        if t2 == t:
            break
        t = t2
    return t


def parse(window_title: str) -> Query | None:
    t = strip_app(window_title)
    if not t:
        return None
    if _looks_like_filename(t):
        t = re.sub(r"\.(?:mkv|mp4|avi|m4v|mov|webm|ts|wmv)$", "", t, flags=re.I)
        t = re.sub(r"[._]+", " ", t)
    t = re.sub(r"^\(\d+\)\s*", "", t)               # "(3) Title" unread-count prefixes

    segments = [s.strip() for s in SEPARATORS.split(t) if s and s.strip()]
    segments = [s for s in segments if not _is_site(s)]
    if not segments:
        return None
    # Prefer the segment carrying the episode marker; otherwise the first meaningful one.
    seg = next((s for s in segments if any(p.search(s) for p in EPISODE_PATTERNS)), None)
    season = episode = None
    if seg is None and len(segments) > 1:
        # "Show - Season 2 - Episode 4" style: the markers sit in their own segments.
        joined = " ".join(segments)
        if any(p.search(joined) for p in EPISODE_PATTERNS):
            seg = joined
    if seg is None:
        seg = segments[0]
    for p in EPISODE_PATTERNS:
        m = p.search(seg)
        if m:
            season, episode = int(m.group(1)), int(m.group(2))
            seg = seg[:m.start()]
            break
    else:
        m = EPISODE_ONLY.search(seg)
        if m:
            sm = SEASON_ONLY.search(seg)
            season, episode = (int(sm.group(1)) if sm else 1), int(m.group(1))
            seg = seg[:min(m.start(), sm.start() if sm else m.start())]
    seg = RELEASE_NOISE.sub("", seg)
    year = None
    years = list(YEAR.finditer(seg))
    if years:
        y = years[-1]
        # A year at the very start is usually part of the title ("2001 A Space Odyssey").
        if y.start() > 0:
            year = int(y.group(1))
            seg = seg[:y.start()] + seg[y.end():]
    title = _clean_title(seg)
    if not title or title.lower() in GENERIC or _is_site(title) or len(title) < 2:
        return None
    return Query(title, year, season, episode, "extended" if EDITION.search(t) else "")


def looks_like_media(window_title: str, from_player: bool = False) -> bool:
    """Is this title worth looking up? A desktop player always is; a browser title needs an
    episode marker, a streaming site's name, or words like "watch" or "full movie". Video
    sites that never carry movies (YouTube, Twitch, ...) never are."""
    t = strip_app(window_title)
    for seg in SEPARATORS.split(t):
        name = re.sub(r"\.(?:com|tv|be|to)\b.*$", "", seg.strip().lower())
        if name in NOT_MEDIA:
            return False
    if from_player:
        return True
    if any(p.search(t) for p in EPISODE_PATTERNS) or EPISODE_ONLY.search(t):
        return True
    if any(_is_site(seg) for seg in SEPARATORS.split(t) if seg.strip()):
        return True
    return bool(MEDIA_HINTS.search(t))


def site_key(window_title: str) -> str:
    """A title with episode numbers and app suffixes removed, for remembering a manual answer."""
    t = strip_app(window_title).lower()
    for p in EPISODE_PATTERNS:
        t = p.sub(" ", t)
    t = EPISODE_ONLY.sub(" ", t)
    t = SEASON_ONLY.sub(" ", t)
    t = re.sub(r"\d+", "#", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_name(name: str) -> str:
    """For comparing a parsed title with a database title."""
    n = name.lower().replace("&", " and ")
    n = re.sub(r"^(?:the|a|an)\s+", "", n)
    n = re.sub(r"[^a-z0-9]+", " ", n)
    return re.sub(r"\s+", " ", n).strip()


# -- what a web page says it is playing -----------------------------------------------------
# Streaming sites differ in where they name the film: the tab title, the page's metadata, a
# heading, or only the player's address ("/embed/movie/1375161", "/tv/1399/1/2", "tt0944947").
# The extension reports all of them; these pick the usable ones.

@dataclass(frozen=True)
class PageId:
    kind: str                   # movie | tv | "" (an IMDb id names no kind)
    tmdb_id: int = 0
    imdb_id: str = ""
    season: int | None = None
    episode: int | None = None
    embed: bool = False         # from a player's embed address, not a site's own page


IMDB_ID = re.compile(r"\b(tt\d{7,9})\b")
PATH_ID = re.compile(r"/(movies?|tv|series|shows?)/(?:tmdb[-:]?)?(\d{2,8})(?=[/?#&-]|$)(?:[/-](\d{1,2})[/-](\d{1,4})(?!\d))?", re.I)
QUERY_TMDB = ("tmdb", "tmdb_id", "tmdbid", "tmdbId", "id")
QUERY_SEASON = ("season", "s", "se")
QUERY_EPISODE = ("episode", "e", "ep")
EMBED = re.compile(r"/(?:embed|e|player|play)/|embed|player", re.I)
TV_WORDS = ("tv", "series", "show", "shows")


def page_ids(urls: list[str]) -> list[PageId]:
    """Ids of what is playing, from the tab's and its frames' addresses: IMDb ids anywhere,
    TMDB ids in the paths and parameters embed players use. A site's own page numbers can
    look the same, so callers check an id's title against the page's names. The first
    address is the tab's: a frame from another site is a player embedded in it."""
    import urllib.parse

    out: list[PageId] = []
    tab_host = urllib.parse.urlsplit(urls[0]).hostname if urls and urls[0] else None
    for url in urls:
        if not url or not url.startswith("http"):
            continue
        u = urllib.parse.urlsplit(url)
        embed = bool(EMBED.search(u.netloc + u.path)) or (tab_host is not None and u.hostname != tab_host)
        qs = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items() if v}
        num = lambda keys: next((int(qs[k]) for k in keys if qs.get(k, "").isdigit()), None)
        season, episode = num(QUERY_SEASON), num(QUERY_EPISODE)
        for m in IMDB_ID.finditer(urllib.parse.unquote(url)):
            out.append(PageId("tv" if episode else "", 0, m.group(1), season, episode, embed))
        for m in PATH_ID.finditer(u.path):
            kind = "tv" if m.group(1).lower().rstrip("s") in ("tv", "serie", "show") else "movie"
            s, e = (int(m.group(3)), int(m.group(4))) if m.group(4) else (season, episode)
            out.append(PageId(kind, int(m.group(2)), "", s, e, embed))
        tmdb = num(QUERY_TMDB)
        if tmdb and embed:
            kind = "tv" if episode or any(w in u.path.lower().split("/") for w in TV_WORDS) else "movie"
            out.append(PageId(kind, tmdb, "", season, episode, embed))
    seen, unique = set(), []
    for i in sorted(out, key=lambda i: (not i.embed, not i.imdb_id)):
        if i not in seen:
            seen.add(i)
            unique.append(i)
    return unique


def site_brands(urls: list[str]) -> set[str]:
    """The site's own name ("cinewave" for watch.cinewave.qzz.io): a page titled only that
    names no film."""
    import urllib.parse

    out = set()
    for url in urls:
        host = urllib.parse.urlsplit(url).hostname or ""
        for label in host.split("."):
            if len(label) >= 4 and label not in ("www", "watch", "play", "stream", "player", "embed"):
                out.add(normalize_name(label))
    return out


def page_names(names: list[str], urls: list[str]) -> list[str]:
    """The names a page gives, best first, without the ones that only name the site."""
    brands = site_brands(urls)
    out, seen = [], set()
    for name in names:
        name = (name or "").strip()
        q = parse(name) if name else None
        if q is None:
            continue
        n = normalize_name(q.title)
        if not n or n.replace(" ", "") in brands or n in brands or (n, q.year, q.season, q.episode) in seen:
            continue
        seen.add((n, q.year, q.season, q.episode))
        out.append(name)
    return out


def same_title(a: str, b: str) -> bool:
    """Do two names name the same film ("Don't Move" and "Don't Move (2026)")?"""
    import difflib

    x, y = normalize_name(a), normalize_name(b)
    if not x or not y:
        return False
    return x == y or (min(len(x), len(y)) >= 5 and (x in y or y in x)) \
        or difflib.SequenceMatcher(None, x, y).ratio() >= 0.8

