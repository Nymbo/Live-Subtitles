"""Unit tests for media mode's pure parts: titles, subtitle files, alignment, clock, cache.

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import io
import random
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from live_subtitles.media import align, subfile
from live_subtitles.media.clock import AudioClock, ExtensionClock
from live_subtitles.media.controller import AudioRing, _cue_at, _gap_around
from live_subtitles.media.generate import text_cues
from live_subtitles.media.sources import Candidate, score
from live_subtitles.media.store import Store, StoredTrack
from live_subtitles.media.tagger import features, merge_events
from live_subtitles.media.titles import Query, parse, site_key
from live_subtitles.media.tmdb import Match
from live_subtitles.subtitles import Cue, to_srt

SRT = """1
00:00:01,000 --> 00:00:03,500
<i>Hello there.</i>

2
00:00:04,000 --> 00:00:06,000
- [door creaks]
- Who's there?

3
00:01:02,250 --> 00:01:04,000
JOHN: Get down!
"""

VTT = """WEBVTT

00:01.000 --> 00:03.500 line:90%
Hello <c.yellow>there</c>.

01:02.250 --> 01:04.000
Second cue
"""

ASS = """[Script Info]
Title: x

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.50,Default,,0,0,0,,{\\i1}Hello,{\\i0} there.\\NSecond line
"""


class TitleTests(unittest.TestCase):
    def test_streaming_site_episode(self):
        self.assertEqual(parse("Watch The Bear Season 2 Episode 4 Online Free - FMovies - Google Chrome"),
                         Query("The Bear", None, 2, 4))

    def test_release_filename(self):
        self.assertEqual(parse("The.Bear.S02E04.1080p.WEB.H264-EDITH.mkv - VLC media player"),
                         Query("The Bear", None, 2, 4))
        self.assertEqual(parse("Parasite 2019 1080p BluRay x264.mp4 - VLC media player"),
                         Query("Parasite", 2019))

    def test_movie_with_year_and_edge_suffix(self):
        self.assertEqual(parse("Dune: Part Two (2024) - HD Movie | 123Movies - Personal - Microsoft​ Edge"),
                         Query("Dune: Part Two", 2024))

    def test_year_at_start_is_title(self):
        self.assertEqual(parse("2001: A Space Odyssey (1968) - Watch Online - Google Chrome"),
                         Query("2001: A Space Odyssey", 1968))

    def test_anime_episode_only(self):
        q = parse("Frieren: Beyond Journey's End Episode 12 English Sub - HiAnime - Google Chrome")
        self.assertEqual((q.title, q.season, q.episode), ("Frieren: Beyond Journey's End", 1, 12))

    def test_titles_that_name_nothing(self):
        for t in ("Netflix - Google Chrome", "New Tab - Google Chrome", "YouTube - Mozilla Firefox"):
            self.assertIsNone(parse(t), t)

    def test_only_media_titles_are_looked_up(self):
        from live_subtitles.media.titles import looks_like_media

        self.assertFalse(looks_like_media("Training Plyometrics with my Best Bud - YouTube - Google Chrome"))
        self.assertFalse(looks_like_media("Interstellar Official Trailer - YouTube - Google Chrome"))
        self.assertFalse(looks_like_media("Oppenheimer - Google Search - Google Chrome"))
        self.assertTrue(looks_like_media("Watch The Bear Season 2 Episode 4 Online Free - FMovies - Google Chrome"))
        self.assertTrue(looks_like_media("Dune: Part Two (2024) - HD Movie | 123Movies - Google Chrome"))
        self.assertTrue(looks_like_media("Parasite.2019.mkv - VLC media player", from_player=True))

    def test_edge_profile_and_more_pages(self):
        from live_subtitles.media import titles

        saved = titles._profiles
        titles._profiles = set(titles.STOCK_PROFILES) | {"nymbo"}
        try:
            t = "The Office: Superfan Episodes - Peacock and 1 more page - Nymbo - Microsoft​ Edge"
            self.assertEqual(titles.strip_app(t), "The Office: Superfan Episodes - Peacock")
            self.assertEqual(parse(t), Query("The Office", edition="extended"))
            self.assertTrue(titles.looks_like_media(t))
            self.assertEqual(parse("Inception (2010) - Profile 2 - Microsoft Edge"), Query("Inception", 2010))
            # A last segment that isn't a profile stays: it may be the title itself.
            self.assertEqual(titles.strip_app("Watch - The Bear - Microsoft Edge"), "Watch - The Bear")
        finally:
            titles._profiles = saved

    def test_site_key_ignores_episode(self):
        self.assertEqual(site_key("Show S01E02 - Site - Google Chrome"), site_key("Show S01E03 - Site - Google Chrome"))

    def test_page_ids_from_embed_addresses(self):
        from live_subtitles.media.titles import PageId, page_ids

        self.assertEqual(page_ids(["https://vsembed.ru/embed/movie/1375161"]),
                         [PageId("movie", 1375161, "", None, None, True)])
        self.assertEqual(page_ids(["https://vidsrc.to/embed/tv/1399/2/5"]),
                         [PageId("tv", 1399, "", 2, 5, True)])
        self.assertEqual(page_ids(["https://player.example/embed?tmdb=550"]), [PageId("movie", 550, "", None, None, True)])
        self.assertEqual(page_ids(["https://x.example/embed/tv?tmdb=1399&season=1&episode=3"])[0],
                         PageId("tv", 1399, "", 1, 3, True))
        imdb = page_ids(["https://site.example/title/tt0944947/watch"])
        self.assertEqual(imdb[0].imdb_id, "tt0944947")
        # A site's own numbering looks the same; it is reported, but not as an embed.
        self.assertFalse(page_ids(["https://cinejoy.pk/watch/movie/1058424"])[0].embed)
        self.assertEqual(page_ids(["https://watch.cinewave.qzz.io/play/DgYYDBJMR1ZUXF9TRg", "about:blank"]), [])
        # A frame from another site than the tab is an embedded player, whatever its path.
        self.assertTrue(page_ids(["https://watch.cinewave.qzz.io/play/x", "https://vidfast.pro/movie/1375161"])[0].embed)

    def test_page_names_skip_the_site_name(self):
        from live_subtitles.media.titles import page_names, same_title

        urls = ["https://watch.cinewave.qzz.io/play/DgYYDBJMR1ZUXF9TRg", "https://vsembed.ru/embed/movie/1375161"]
        self.assertEqual(page_names(["CineWave", "Don't Move", "CINEWAVE", "Don't Move"], urls), ["Don't Move"])
        self.assertEqual(page_names(["Watch Movies Online Free", "Hope (2026)"], urls), ["Hope (2026)"])
        self.assertTrue(same_title("Don't Move", "Don’t Move"))
        self.assertTrue(same_title("The Office", "The Office (US)"))
        self.assertFalse(same_title("Don't Move", "Hope"))


class SubfileTests(unittest.TestCase):
    def test_srt(self):
        cues = subfile.load_bytes(SRT.encode())
        self.assertEqual(len(cues), 3)
        self.assertEqual(cues[0].lines, ["Hello there."])
        self.assertAlmostEqual(cues[2].start, 62.25)

    def test_vtt_strips_tags_and_settings(self):
        cues = subfile.load_bytes(VTT.encode())
        self.assertEqual(cues[0].lines, ["Hello there."])
        self.assertAlmostEqual(cues[1].start, 62.25)

    def test_ass(self):
        cues = subfile.load_bytes(ASS.encode())
        self.assertEqual(cues[0].lines, ["Hello, there.", "Second line"])

    def test_zip_utf16_and_cp1252(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("film.srt", SRT.encode("utf-16"))
            z.writestr("film.forced.srt", "1\n00:00:01,000 --> 00:00:02,000\nx\n")
        self.assertEqual(len(subfile.load_bytes(buf.getvalue())), 3)
        latin = "1\n00:00:01,000 --> 00:00:02,000\nCafé\n".encode("cp1252")
        self.assertEqual(subfile.load_bytes(latin)[0].lines, ["Café"])

    def test_sound_descriptions(self):
        lines = ["- [door creaks]", "- Who's there?"]
        self.assertEqual(subfile.display_lines(lines, True), lines)
        self.assertEqual(subfile.display_lines(lines, False), ["Who's there?"])
        self.assertEqual(subfile.display_lines(["JOHN: Get down!"], False), ["Get down!"])
        self.assertEqual(subfile.speech_text(["JOHN: Get down!"]), "Get down!")
        self.assertEqual(subfile.speech_text(["- (OBJECTS CLATTER)", "- (CHAINSAW)"]), "")
        self.assertTrue(subfile.is_sound_only(["[thunder]"]))

    def test_has_sound_descriptions(self):
        hi = [Cue(i, i + 1, ["[laughs] ok"]) for i in range(20)]
        plain = [Cue(i, i + 1, ["ok"]) for i in range(20)]
        self.assertTrue(subfile.has_sound_descriptions(hi))
        self.assertFalse(subfile.has_sound_descriptions(plain))


def _script(n_lines: int = 160, seed: int = 3) -> list[Cue]:
    """A synthetic track: varied sentences, a few seconds apart."""
    rng = random.Random(seed)
    vocab = ("river stone captain window yellow tomorrow quietly harbour engine silver promise ladder "
             "orchard thunder kitchen violin marble lantern pocket desert journal canyon whisper garden "
             "copper island meadow anchor bridge forest velvet").split()
    cues, t = [], 5.0
    for _ in range(n_lines):
        words = [rng.choice(vocab) for _ in range(rng.randint(4, 8))]
        dur = 0.35 * len(words)
        cues.append(Cue(t, t + dur, [" ".join(words)]))
        t += dur + rng.uniform(0.8, 2.5)
    return cues


def _heard(cues: list[Cue], scale: float, offset: float, t_from: float, t_to: float, drop: float = 0.2,
           seed: int = 1) -> list[tuple[str, float]]:
    """Words a recogniser would report: tau = (c - offset) / scale, with some words lost."""
    rng = random.Random(seed)
    index = align.TrackIndex(cues)
    out = []
    for w, c in zip(index.words, index.times):
        if t_from <= c <= t_to and rng.random() > drop:
            out.append((w, (c - offset) / scale + rng.uniform(-0.08, 0.08)))
    return out


class AlignTests(unittest.TestCase):
    def setUp(self):
        self.cues = _script()
        self.index = align.TrackIndex(self.cues)

    def test_locks_on_offset(self):
        a = align.Aligner()
        heard = _heard(self.cues, 1.0, 37.5, 100, 140)
        self.assertEqual(a.add(self.index.anchors(heard)), "locked")
        self.assertAlmostEqual(a.current.offset, 37.5, delta=0.15)
        self.assertEqual(a.current.scale, 1.0)

    def test_framerate_ratio(self):
        s = 25 / 23.976
        a = align.Aligner()
        for lo in range(40, 420, 20):
            a.add(self.index.anchors(_heard(self.cues, s, 3.0, lo, lo + 20, seed=lo)))
        self.assertAlmostEqual(a.current.scale, s, places=4)
        self.assertAlmostEqual(a.track_time(200.0), s * 200 + 3.0, delta=0.3)

    def test_jump(self):
        a = align.Aligner()
        a.add(self.index.anchors(_heard(self.cues, 1.0, 10.0, 60, 120)))
        self.assertTrue(a.locked)
        # The viewer skips ahead: the same audio clock now hears lines from 90 s further on.
        last = None
        for lo in range(200, 290, 10):
            r = a.add(self.index.anchors(_heard(self.cues, 1.0, 100.0, lo + 90, lo + 100, seed=lo)))
            last = r or last
        self.assertIn("jumped", [last, "jumped" if len(a.segments) > 1 else None])
        self.assertAlmostEqual(a.current.offset, 100.0, delta=0.3)

    def test_extended_cut_scene_is_caught_quickly(self):
        # A broadcast track played against an extended cut: a 20 s scene the track doesn't
        # have, after which everything is 20 s later than the track says. The old 60 s
        # lookback took a minute of dialogue to accept that.
        a = align.Aligner()
        for lo in range(10, 200, 5):
            a.add(self.index.anchors(_heard(self.cues, 1.0, 0.0, lo, lo + 5, seed=lo)))
        self.assertAlmostEqual(a.current.offset, 0.0, delta=0.3)
        caught = None
        for lo in range(200, 300, 5):          # track time; heard 20 s later
            if a.add(self.index.anchors(_heard(self.cues, 1.0, -20.0, lo, lo + 5, seed=lo))) == "jumped":
                caught = lo + 5 - 200
                break
        self.assertIsNotNone(caught)
        self.assertLessEqual(caught, 15)        # seconds of dialogue after the scene
        self.assertAlmostEqual(a.current.offset, -20.0, delta=0.3)

    def test_off_track_shows_live_captions(self):
        import tempfile
        from pathlib import Path

        from live_subtitles.media.controller import MediaController, Session
        from live_subtitles.media.store import Store, StoredTrack
        from live_subtitles.settings import Settings

        c = MediaController(Settings(), Store(Path(tempfile.mkdtemp()) / "m.db"))
        s = Session(1, "t")
        s.track = StoredTrack("k", "k", {}, "x", self.cues)
        s.aligner.segments.append(align.Segment(0.0, 1.0, 0.0))
        wanted = []
        c.bridge_wanted.connect(wanted.append)
        on = [(w, t, t + 0.3) for w, t in _heard(self.cues, 1.0, 0.0, 100, 105)]
        c._check_on_track(s, on)
        self.assertFalse(s.off_track)
        # An extra scene: words that are nowhere near this point of the track.
        scene = [(w, 110 + i * 0.4, 110.3 + i * 0.4) for i, w in
                 enumerate("hospital sandwich printer basketball dinosaur committee".split())]
        c._check_on_track(s, scene)
        self.assertTrue(s.off_track)
        self.assertEqual(wanted, [True])
        c.session = s
        self.assertIsNone(c.lines_now())          # live captions, not lines from elsewhere
        c._check_on_track(s, [(w, t, t + 0.3) for w, t in _heard(self.cues, 1.0, 0.0, 120, 125)])
        self.assertFalse(s.off_track)

    def test_page_without_a_title_identified_from_its_player(self):
        import tempfile
        from pathlib import Path

        from live_subtitles.media.controller import MediaController, Session
        from live_subtitles.settings import Settings

        class FakeTmdb:
            key = "k"

            def lookup(self, kind, tmdb_id, season=None, episode=None):
                return {1375161: Match("movie", 1375161, "Don't Move", 2026),
                        1058424: Match("movie", 1058424, "Something Else", 2023)}[tmdb_id]

        c = MediaController(Settings(), Store(Path(tempfile.mkdtemp()) / "m.db"))
        report = {"title": "CineWave", "tab": "https://watch.cinewave.qzz.io/play/DgYYDBJMR1ZUXF9TRg",
                  "page": "https://vsembed.ru/embed/movie/1375161",
                  "names": ["Don't Move", "CineWave"], "urls": []}
        title, names, ids, named = c._page_title(report)
        self.assertEqual((title, names, named), ("Don't Move", [], True))
        s = Session(1, title, parse(title))
        s.names, s.ids = names, ids
        self.assertEqual(c._match_from_ids(s, FakeTmdb()).label(), "Don't Move (2026)")
        # Only the site's name and the player's address: the embed id is trusted.
        title, names, ids, named = c._page_title(dict(report, names=[]))
        self.assertFalse(named)
        s = Session(1, title, None)
        s.ids = ids
        self.assertEqual(c._match_from_ids(s, FakeTmdb()).tmdb_id, 1375161)
        # A site's own page number that names another film than the page shows: not used.
        s = Session(1, "Hope (2026)", parse("Hope (2026)"))
        s.ids = c._page_title({"title": "Hope (2026)", "tab": "https://cinejoy.pk/watch/movie/1058424"})[2]
        self.assertIsNone(c._match_from_ids(s, FakeTmdb()))

    def test_notice_explains_live_captions(self):
        import tempfile
        from pathlib import Path

        from live_subtitles.media.controller import MediaController, Session
        from live_subtitles.media.store import Store, StoredTrack
        from live_subtitles.settings import Settings

        c = MediaController(Settings(), Store(Path(tempfile.mkdtemp()) / "m.db"))
        self.assertEqual(c.notice(), "")
        s = c.session = Session(1, "t")
        s.state = "searching"
        self.assertIn("getting subtitles", c.notice())
        s.track, s.state = StoredTrack("k", "k", {}, "x", self.cues), "aligning"
        self.assertIn("Subtitles found", c.notice())     # live captions bridge until the lock
        s.aligner.segments.append(align.Segment(0.0, 1.0, 0.0))
        s.state = "synced"
        self.assertEqual(c.notice(), "")                 # the subtitles themselves show
        s.off_track = True
        self.assertIn("finding the place", c.notice())

    def test_match_score_tells_tracks_apart(self):
        heard = _heard(self.cues, 1.0, 0.0, 50, 250)
        other = align.TrackIndex(_script(seed=99))
        self.assertGreater(align.match_score(self.index, heard), 0.3)
        self.assertLess(align.match_score(other, heard), align.MISMATCH_SCORE if hasattr(align, "MISMATCH_SCORE")
                        else 0.06)

    def test_speech_alignment(self):
        mask = align.cue_mask(self.cues)
        offset = 42.0
        spans = [((c.start - offset), (c.end - offset)) for c in self.cues if 200 < c.start < 380]
        m = align.speech_align(mask, spans, 200 - offset, 380 - offset)
        self.assertIsNotNone(m)
        self.assertAlmostEqual(m.offset, offset, delta=0.2)
        self.assertGreater(m.z, 7)



class ClockTests(unittest.TestCase):
    def test_counts_only_playing_audio(self):
        c = AudioClock()
        self.assertTrue(c.feed(np.full(1600, 0.01, np.float32), now=10.0))
        self.assertFalse(c.feed(np.zeros(1600, np.float32), now=10.1))
        self.assertAlmostEqual(c.tau, 0.1)
        self.assertTrue(c.playing(10.2))
        self.assertAlmostEqual(c.now(10.15), 0.25)
        self.assertFalse(c.playing(12.0))
        self.assertAlmostEqual(c.now(12.0), 0.1)

    def test_extension_clock(self):
        e = ExtensionClock()
        e.report(100.0, False, 1.0, 3600, now=5.0)
        self.assertAlmostEqual(e.now(5.5), 100.5)
        e.report(100.0, True, now=6.0)
        self.assertAlmostEqual(e.now(9.0), 100.0)

    def test_ring_maps_media_time(self):
        r = AudioRing()
        r.append(np.zeros(16000, np.float32), 0.0, 500.0)
        r.append(np.zeros(16000, np.float32), 1.0, 1200.0)       # a seek
        self.assertAlmostEqual(r.to_media(0.5), 500.5)
        self.assertAlmostEqual(r.to_media(1.25), 1200.25)
        audio, t0 = r.get(0.5, 1.5)
        self.assertEqual(audio.size, 16000)


class StoreTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            st = Store(Path(d) / "m.db")
            m = Match("episode", 1, "Show", 2020, 1, 2)
            t = StoredTrack(m.key, m.label(), m.to_dict(), "gestdown:x", [Cue(1, 2, ["hi"])])
            st.put(t)
            st.update(m.key, nudge=-0.5, segments=[[0, 1.0, 2.0]])
            got = st.get(m.key)
            self.assertEqual((got.cues[0].lines, got.nudge, got.segments), (["hi"], -0.5, [[0, 1.0, 2.0]]))
            st.reject(m.key, "gestdown:x")
            self.assertEqual(st.rejected(m.key), {"gestdown:x"})
            st.set_alias("k", m.to_dict())
            self.assertEqual(Match.from_dict(st.alias("k")).key, m.key)
            st.close()


class MiscTests(unittest.TestCase):
    def test_scoring_prefers_hi_when_wanted(self):
        hi = Candidate("opensubtitles", "1", hearing_impaired=True, downloads=100)
        plain = Candidate("opensubtitles", "2", downloads=100)
        self.assertGreater(score(hi, True), score(plain, True))
        self.assertGreater(score(plain, False), score(hi, False))
        self.assertLess(score(Candidate("x", "3", machine=True, downloads=10 ** 6), True), score(plain, True))

    def test_cue_lookup_and_gaps(self):
        cues = [Cue(1, 2, ["a"]), Cue(5, 6, ["b"])]
        self.assertEqual(_cue_at(cues, 1.5).lines, ["a"])
        self.assertIsNone(_cue_at(cues, 3.0))
        self.assertAlmostEqual(_gap_around(cues, 3.0), 3.0)
        self.assertEqual(_gap_around(cues, 5.5), 0.0)

    def test_text_cues_split_long_segments(self):
        cues = text_cues([(0.0, 10.0, " ".join(["word"] * 40))])
        self.assertGreater(len(cues), 1)
        self.assertAlmostEqual(cues[-1].end, 10.0)
        self.assertTrue(all(len(line) <= 42 for c in cues for line in c.lines))

    def test_tagger_features_shape(self):
        f = features(np.random.default_rng(0).standard_normal(16000 * 5).astype(np.float32) * 0.1)
        self.assertEqual(f.shape, (1024, 128))

    def test_merge_events(self):
        ev = merge_events([("[music]", 0, 10, 0.6), ("[music]", 5, 15, 0.7), ("[dog barking]", 3, 4, 0.5)])
        self.assertEqual(ev, [("[music]", 0, 15, 0.7), ("[dog barking]", 3, 4, 0.5)])

    def test_season_packs_lose_and_editions_win(self):
        one = Candidate("opensubtitles", "1", "The.Office.US.S02E03.1080p.WEB-DL", downloads=100)
        pack = Candidate("opensubtitles", "2", "Season 2 All Episodes", downloads=100)
        full = Candidate("opensubtitles", "3", "The.Office.US.S02.Complete.DVDRip", downloads=100)
        ext = Candidate("opensubtitles", "4", "The.Office.US.S02E03.EXTENDED.PCOK.WEB-DL", downloads=10)
        self.assertLess(score(pack, True), score(one, True))
        self.assertLess(score(full, True), score(one, True))
        self.assertGreater(score(ext, True, "extended"), score(one, True, "extended"))
        self.assertLess(score(ext, True), score(one, True))

    def test_gestdown_matches_country_tagged_show(self):
        from unittest import mock

        from live_subtitles.media import sources
        from live_subtitles.media.tmdb import Match

        shows = {"shows": [{"id": "uk", "name": "The Office (UK)"}, {"id": "us", "name": "The Office (US)"}]}
        asked = []

        def fake(url, *a, **k):
            asked.append(url)
            return shows if "/shows/search/" in url else {"matchingSubtitles": []}

        with mock.patch.object(sources.net, "get_json", fake):
            sources.Gestdown().search(Match("episode", 0, "The Office", 2005, 6, 9, show_imdb_id="tt0386676"))
        self.assertTrue(asked[-1].endswith("/subtitles/get/us/6/9/English"), asked[-1])

    def test_player_subtitles_keep_english_files(self):
        from live_subtitles.media import sources
        from live_subtitles.media.tmdb import Match

        page = sources.PageSubtitles([{"url": "https://cdn.x/subs/English.vtt"}, {"url": "https://cdn.x/subs/Arabic.vtt"},
                                      {"url": "https://cdn.x/s/ko.vtt"}, {"url": "https://cdn.x/s/a8f3e1.srt"},
                                      {"url": "https://proxy.x/sub/aHR0", "name": "Hope (2026) ZOOM.es.srt"},
                                      {"url": "https://proxy.x/sub/aHR1", "name": "Hope.2026.AMZN.WEB-DL.srt"}])
        names = [c.id for c in page.search(Match("movie", 0, "Hope", 2026))]
        self.assertEqual(names, ["https://cdn.x/subs/English.vtt", "https://cdn.x/s/a8f3e1.srt", "https://proxy.x/sub/aHR1"])

    def test_looks_english(self):
        en = [Cue(i, i + 1, ["What are you doing here? I told you to wait for me."]) for i in range(10)]
        es = [Cue(i, i + 1, ["¿Qué haces aquí? Te dije que me esperaras en la casa."]) for i in range(10)]
        self.assertTrue(subfile.looks_english(en))
        self.assertFalse(subfile.looks_english(es))

    def test_follows_the_tab_with_sound(self):
        from live_subtitles.media.controller import MediaController

        c = MediaController.__new__(MediaController)
        c._tabs, c._active_tab = {}, None
        now = 100.0
        office = {"paused": False}
        hope = {"paused": False}
        c._tabs = {1: (now, office, 10.0, 0.0), 2: (now, hope, 90.0, now)}
        self.assertEqual(c._pick_tab(now), 2)           # only Hope makes sound
        # Both heard: the one started most recently, steadily (no flip-flopping).
        c._tabs[1] = (now, office, 10.0, now)
        self.assertEqual(c._pick_tab(now), 2)
        # A quiet scene in Hope (no sound for a few seconds) doesn't hand over to the other tab.
        c._tabs[2] = (now, hope, 90.0, now - 5)
        self.assertEqual(c._pick_tab(now), 2)
        # A video left playing in a background tab loses to the one on screen.
        c._tabs[2] = (now, dict(hope, visible=False), 95.0, now)
        self.assertEqual(c._pick_tab(now), 1)
        # The watched tab goes quiet for a moment (an ad, buffering): a background tab waits.
        c._active_tab = 1
        c._tabs = {1: (now - 5, office, 10.0, now - 5), 2: (now, dict(hope, visible=False), 95.0, now)}
        self.assertEqual(c._pick_tab(now), 1)
        # ... but takes over once the watched one has been gone a while.
        c._tabs[1] = (now - 40, office, 10.0, now - 40)
        self.assertEqual(c._pick_tab(now), 2)
        # A tab on screen takes over straight away.
        c._tabs = {1: (now - 5, office, 10.0, now - 5), 2: (now, hope, 95.0, now)}
        self.assertEqual(c._pick_tab(now), 2)

    def test_saved_broadcast_track_is_not_reused_for_extended_cut(self):
        from live_subtitles.media.controller import _fits_cut
        from live_subtitles.media.store import StoredTrack
        from live_subtitles.media.tmdb import Match

        ext = Match("episode", 0, "The Office", 2005, 6, 9, edition="extended")
        broadcast = StoredTrack("k", "k", {"release": "2HD"}, "gestdown:1", [])
        pcok = StoredTrack("k", "k", {"release": "The.Office.US.S06E09.EXTENDED.1080p.PCOK"}, "opensubtitles:2", [])
        self.assertFalse(_fits_cut(broadcast, ext))
        self.assertTrue(_fits_cut(pcok, ext))
        self.assertTrue(_fits_cut(broadcast, Match("episode", 0, "The Office", 2005, 6, 9)))

    def test_episode_title_from_release(self):
        from live_subtitles.media.controller import _release_episode_title as named

        self.assertEqual(named("The.Office.US.S06E09.Double.Date.EXTENDED.1080p.PCOK.WEB-DL"), "Double Date")
        self.assertEqual(named("The Office (US) - S06E09 - Double Date"), "Double Date")
        self.assertEqual(named("The.Office.US.S06E09.720p.HDTV.X264-DIMENSION"), "")

    def test_srt_writer_round_trip(self):
        cues = subfile.load_bytes(to_srt([Cue(1.0, 2.0, ["a", "b"])]).encode())
        self.assertEqual(cues[0].lines, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
