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

import re
from dataclasses import dataclass

APP_SUFFIXES = re.compile(
    r"\s*(?:[-–—|]\s*)?(?:(?:and \d+ more pages?|Personal|Work|Profile \d+|InPrivate)\s*[-–—]\s*)*"
    r"(?:Google Chrome|Chromium|Mozilla Firefox|Firefox|Microsoft\s*Edge|Brave|Opera|Vivaldi|Arc|"
    r"VLC media player|MPC-HC|MPC-BE|mpv|PotPlayer|Media Player Classic|Movies & TV|Media Player)\s*$",
    re.I)
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
    r"with\s+english\s+subtitles|on\s+\w+)\s*$", re.I)
EPISODE_PATTERNS = (
    re.compile(r"\bS(\d{1,2})\s*[.\-_ ]?\s*E(\d{1,3})\b", re.I),
    re.compile(r"\b(\d{1,2})x(\d{2,3})\b"),
    re.compile(r"\bSeason\s*(\d{1,2})\s*(?:[,:\-–]\s*)?(?:Episode|Ep\.?|E)\s*(\d{1,3})\b", re.I),
    re.compile(r"\bS(\d{1,2})\s+Ep?\.?\s*(\d{1,3})\b", re.I),
)
EPISODE_ONLY = re.compile(r"\b(?:Episode|Ep\.?)\s*(\d{1,4})\b", re.I)
SEASON_ONLY = re.compile(r"\bSeason\s*(\d{1,2})\b", re.I)
YEAR = re.compile(r"[\(\[]?\b(19[2-9]\d|20[0-4]\d)\b[\)\]]?")


@dataclass(frozen=True)
class Query:
    title: str
    year: int | None = None
    season: int | None = None
    episode: int | None = None

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


def strip_app(title: str) -> str:
    t = ZERO_WIDTH.sub("", title).strip()
    prev = None
    while prev != t:            # Edge: "... - Personal - Microsoft Edge"
        prev = t
        t = APP_SUFFIXES.sub("", t).strip()
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
    return Query(title, year, season, episode)


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
