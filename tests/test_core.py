"""Unit tests for the pure-Python pieces: caption lines, cues, writers, settings, hotkeys.

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from live_subtitles import settings as S
from live_subtitles.captions import CaptionBuilder
from live_subtitles.hotkeys import MOD_ALT, MOD_CONTROL, MOD_SHIFT, parse
from live_subtitles.models import live_model
from live_subtitles.subtitles import Word, ass_color, build_cues, split_lines, to_ass, to_srt, to_vtt
from live_subtitles.unified import UnifiedStreamer


class CaptionBuilderTests(unittest.TestCase):
    def test_wraps_at_max_chars(self):
        b = CaptionBuilder(max_lines=2, max_chars=20, sentence_breaks=False)
        b.add("one two three four five six seven".split(), "", 0)
        self.assertTrue(all(len(line) <= 20 for line in b.display_lines()))
        self.assertEqual(" ".join(b.display_lines()), "one two three four five six seven")

    def test_rollup_drops_oldest_line(self):
        b = CaptionBuilder(max_lines=2, max_chars=10, sentence_breaks=False)
        b.add("aaaa bbbb cccc dddd eeee ffff".split(), "", 0)
        self.assertEqual(b.display_lines(), ["cccc dddd", "eeee ffff"])

    def test_pop_mode_starts_a_fresh_block(self):
        b = CaptionBuilder(max_lines=2, max_chars=10, mode="pop", sentence_breaks=False)
        b.add("aaaa bbbb cccc dddd eeee".split(), "", 0)
        self.assertEqual(b.display_lines(), ["eeee"])

    def test_sentence_break_starts_new_line(self):
        b = CaptionBuilder(max_lines=2, max_chars=42)
        b.add(["Fine!", "I'm", "freaked", "out."], "", 0)
        self.assertEqual(b.display_lines(), ["Fine!", "I'm freaked out."])

    def test_pending_word_is_shown_but_not_committed(self):
        b = CaptionBuilder()
        b.add(["hello"], "wor", 0)
        self.assertEqual(b.display_lines(), ["hello wor"])
        b.add(["world"], "", 0)
        self.assertEqual(b.display_lines(), ["hello world"])

    def test_fillers_hidden(self):
        b = CaptionBuilder(hide_fillers=True)
        b.add(["so", "um,", "yes", "Uh"], "", 0)
        self.assertEqual(b.display_lines(), ["so yes"])

    def test_clears_after_quiet(self):
        b = CaptionBuilder()
        b.add(["hi"], "", 10.0)
        self.assertFalse(b.tick(12.0, 4.0))
        self.assertTrue(b.tick(14.5, 4.0))
        self.assertEqual(b.display_lines(), [])

    def test_punctuation_revises_recent_words(self):
        def fake(words):   # capitalise the first word, end every input with a period
            out = [w.lower().strip(".,") for w in words]
            out[0] = out[0].capitalize()
            out[-1] += "."
            return out

        b = CaptionBuilder(sentence_breaks=False, punctuate=fake)
        b.add(["hello", "there", "friend"], "", 0.0)
        # The newest word stays bare while speech continues...
        self.assertEqual(b.display_lines(), ["Hello there friend"])
        # ...and takes its punctuation once speech pauses.
        self.assertTrue(b.tick(1.0, 4.0))
        self.assertEqual(b.display_lines(), ["Hello there friend."])

    def test_punctuation_window_start_is_not_recapitalised(self):
        from live_subtitles.captions import PUNCT_EDITABLE, PUNCT_WINDOW

        def fake(words):
            return [words[0].capitalize(), *words[1:]]

        b = CaptionBuilder(max_lines=50, max_chars=90, sentence_breaks=False, punctuate=fake)
        words = [f"w{i}" for i in range(PUNCT_WINDOW + 10)]
        for w in words:
            b.add([w], "", 0.0)
        # Only w0 was ever a true start; later window starts are context, not sentence starts.
        self.assertEqual(b.shown[0], "W0")
        self.assertTrue(all(w.islower() for w in b.shown[1:]))
        self.assertLess(PUNCT_EDITABLE, PUNCT_WINDOW)

    def test_only_newest_words_take_punctuation_edits(self):
        from live_subtitles.captions import PUNCT_EDITABLE

        calls = []

        def shout(words):   # upper-case everything it is given
            calls.append(len(words))
            return [w.upper() for w in words]

        b = CaptionBuilder(max_lines=50, max_chars=200, sentence_breaks=False, punctuate=shout)
        for i in range(10):
            b.add([f"w{i}"], "", 0.0)
        # Each word was editable only while it was among the newest few, so all ended up
        # upper-cased exactly once, while they were new; nothing older is touched later.
        self.assertEqual(b.shown, [f"W{i}" for i in range(10)])
        snapshot = list(b.shown[:-PUNCT_EDITABLE])
        b.punctuate = lambda words: [w.lower() for w in words]
        b.add(["w10"], "", 0.0)
        self.assertEqual(b.shown[:len(snapshot)], snapshot)

    def test_placed_words_never_change_line(self):
        def late_period(words):   # puts a period after "two" once "four" has arrived
            out = list(words)
            if "four" in words:
                out[words.index("two")] = "two."
            return out

        b = CaptionBuilder(max_lines=3, max_chars=40, punctuate=late_period)
        b.add(["one", "two", "three"], "", 0.0)
        self.assertEqual(b.display_lines(), ["one two three"])
        b.add(["four"], "", 0.0)
        # "two." now ends a sentence, but "three" was already on screen on that line.
        self.assertEqual(b.display_lines(), ["one two. three four"])
        b.add(["five"], "", 0.0)
        self.assertEqual(b.line_of, [0, 0, 0, 0, 0])

    def test_pending_words_flow_after_committed(self):
        b = CaptionBuilder(max_lines=2, max_chars=12, sentence_breaks=False)
        b.add(["alpha", "beta"], ["gamma", "delta"], 0.0)
        self.assertEqual(b.display_lines(), ["alpha beta", "gamma delta"])
        self.assertEqual(b.line_of, [0, 0])

    def test_resize_relays_out(self):
        b = CaptionBuilder(max_lines=5, max_chars=40, sentence_breaks=False)
        b.add("aaaa bbbb cccc dddd".split(), "", 0.0)
        self.assertEqual(b.display_lines(), ["aaaa bbbb cccc dddd"])
        b.configure(max_chars=10)
        self.assertEqual(b.display_lines(), ["aaaa bbbb", "cccc dddd"])

    def test_sentence_case_fix(self):
        from live_subtitles.punctuate import fix_sentence_case, plain

        self.assertEqual(fix_sentence_case(["one.", "you're", "a", "jerk?", "tom"]),
                         ["one.", "You're", "a", "jerk?", "Tom"])
        self.assertEqual(plain("Don't,"), "don't")
        self.assertEqual(plain("follow-up."), "follow-up")

    def test_punctuation_mismatch_keeps_words(self):
        b = CaptionBuilder(punctuate=lambda words: None)
        b.add(["keep", "these"], "", 0.0)
        self.assertEqual(b.display_lines(), ["keep these"])

    def test_long_speech_is_trimmed_without_changing_view(self):
        from live_subtitles.captions import KEEP_WORDS

        b = CaptionBuilder(max_lines=2, max_chars=20, sentence_breaks=False)
        for i in range(KEEP_WORDS + 150):
            b.add([f"word{i}"], "", 0.0)
        self.assertLessEqual(len(b.raw), KEEP_WORDS + 1)
        self.assertTrue(b.display_lines()[-1].endswith(f"word{KEEP_WORDS + 149}"))

    def test_pixel_fit_callback_respected(self):
        b = CaptionBuilder(max_chars=80, sentence_breaks=False, fits=lambda t: len(t) <= 9)
        b.add("abc def ghi jkl".split(), "", 0)
        self.assertEqual(b.display_lines(), ["abc def", "ghi jkl"])


def words(text, start=0.0, step=0.3):
    return [Word(w, start + i * step, start + i * step + 0.25) for i, w in enumerate(text.split())]


class CueTests(unittest.TestCase):
    def test_split_lines_balances_and_respects_limits(self):
        lines = split_lines("Why don't you just admit that you're freaked out by my robot hand?".split(), 42, 2)
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(len(l) <= 42 for l in lines))
        self.assertLess(abs(len(lines[0]) - len(lines[1])), 15)

    def test_split_lines_prefers_punctuation(self):
        lines = split_lines("I told you, the robots are coming".split(), 30, 2)
        self.assertEqual(lines, ["I told you,", "the robots are coming"])
        # ...but not at the cost of a lopsided pyramid.
        lines = split_lines("Look Celia, we have to follow our dreams".split(), 30, 2)
        self.assertEqual(lines, ["Look Celia, we have", "to follow our dreams"])

    def test_cue_breaks_on_pause(self):
        w = words("one two three") + words("four five", start=5.0)
        cues = build_cues(w)
        self.assertEqual([c.text for c in cues], ["one two three", "four five"])

    def test_cue_never_overlaps_next(self):
        w = words("a sentence that ends here.") + words("Next one starts quickly", start=1.6)
        cues = build_cues(w)
        for a, b in zip(cues, cues[1:]):
            self.assertLessEqual(a.end, b.start)

    def test_cue_minimum_duration(self):
        cues = build_cues([Word("Hi.", 3.0, 3.2)])
        self.assertGreaterEqual(cues[0].end - cues[0].start, 1.0)

    def test_long_speech_splits_into_readable_cues(self):
        w = words(" ".join(["word"] * 60), step=0.25)
        cues = build_cues(w)
        for c in cues:
            self.assertLessEqual(len(c.lines), 2)
            self.assertTrue(all(len(l) <= 42 for l in c.lines))
            self.assertLessEqual(c.end - c.start, 7.0 + 1e-6)


class WriterTests(unittest.TestCase):
    def setUp(self):
        self.cues = build_cues(words("Hello there.") + words("General greeting", start=4.0))

    def test_srt(self):
        out = to_srt(self.cues)
        self.assertTrue(out.startswith("1\n00:00:00,000 --> 00:00:0"))
        self.assertIn("\n\n2\n00:00:04,000 --> ", out)

    def test_vtt(self):
        out = to_vtt(self.cues)
        self.assertTrue(out.startswith("WEBVTT\n\n00:00:00.000 --> "))

    def test_ass_color_inverts_alpha_and_swaps_order(self):
        self.assertEqual(ass_color("#FFFF0000"), "&H000000FF")   # opaque red
        self.assertEqual(ass_color("#80112233"), "&H7F332211")

    def test_ass_has_style_and_events(self):
        ap = S.preset_appearance("Boxed")
        out = to_ass(self.cues, ap)
        style = next(l for l in out.splitlines() if l.startswith("Style: Default"))
        self.assertEqual(style.split(",")[15], "3")   # BorderStyle 3 = opaque box
        self.assertEqual(sum(l.startswith("Dialogue:") for l in out.splitlines()), 2)

    def test_ass_escapes_override_braces(self):
        cues = build_cues([Word("{weird}", 0, 0.5)])
        self.assertNotIn("{weird}", to_ass(cues, S.Appearance()))


class SettingsTests(unittest.TestCase):
    def test_round_trip(self):
        s = S.Settings()
        s.appearance.font_size = 55
        s.layout.x = 100
        s.custom_presets["Mine"] = {"text_color": "#FF00FF00"}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "settings.json"
            S.save(s, p)
            self.assertEqual(S.load(p), s)

    def test_tolerates_missing_unknown_and_bad_keys(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "settings.json"
            p.write_text(json.dumps({"appearance": {"font_size": "huge", "bold": False, "nope": 1},
                                     "mystery": {}}))
            s = S.load(p)
            self.assertEqual(s.appearance.font_size, S.Appearance().font_size)
            self.assertFalse(s.appearance.bold)

    def test_presets_resolve_and_match(self):
        for name in S.PRESETS:
            ap = S.preset_appearance(name)
            self.assertEqual(S.matching_preset(ap), name)
        custom = {"Mine": {"font_size": 77}}
        self.assertEqual(S.matching_preset(S.preset_appearance("Mine", custom), custom), "Mine")


class HotkeyTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse("Ctrl+Alt+C"), (MOD_CONTROL | MOD_ALT, ord("C")))
        self.assertEqual(parse("Ctrl+Shift+F9"), (MOD_CONTROL | MOD_SHIFT, 0x78))
        self.assertEqual(parse("Alt+Space"), (MOD_ALT, 0x20))
        self.assertEqual(parse("Ctrl++"), (MOD_CONTROL, 0xBB))
        self.assertIsNone(parse(""))
        self.assertIsNone(parse("Ctrl+Alt"))
        self.assertIsNone(parse("Ctrl+Bogus"))


class ModelCatalogTests(unittest.TestCase):
    def test_english_uses_english_model(self):
        m = live_model("en", 560)
        self.assertIn("nemotron-speech-streaming-en", m.name)
        self.assertIsNone(m.language)

    def test_other_languages_use_multilingual(self):
        m = live_model("de-DE", 1120)
        self.assertIn("nemotron-3.5-asr-streaming", m.name)
        self.assertEqual(m.language, "de-DE")
        self.assertIn("1120ms", m.name)

    def test_default_engine_settings(self):
        e = S.Engine()
        self.assertEqual((e.recognizer, e.latency_ms), ("auto", 560))

    def test_unknown_latency_falls_back(self):
        self.assertEqual(live_model("en", 999).latency_ms, 560)


def bare_streamer() -> UnifiedStreamer:
    """A UnifiedStreamer with its feature and word state, but no ONNX sessions."""
    from importlib.resources import files

    import numpy as np

    u = UnifiedStreamer.__new__(UnifiedStreamer)
    u.left = 560
    with files("onnx_asr.preprocessors").joinpath("data").joinpath("fbanks.npz").open("rb") as f:
        u.fbank = np.load(f)["nemo128"].astype(np.float32)
    u.window = np.pad(np.hanning(400), (56, 56)).astype(np.float32)
    u.pre, u.pre_at, u.last_sample = np.zeros(256, np.float32), -256, 0.0
    u.feats, u.feat_at, u.n_feats, u.center = np.zeros((0, 128), np.float32), 0, 0, 0
    u.word = u.shown_word = ""
    u.last_token_t, u.phrase, u.phrase_words = 0.0, 0, 0
    return u


class UnifiedStreamerTests(unittest.TestCase):
    def test_streamed_features_match_whole_file(self):
        import numpy as np

        rng = np.random.default_rng(1)
        audio = (rng.standard_normal(16000 * 3) * 0.1).astype(np.float32)
        # NeMo's preprocessing over the whole clip at once: pre-emphasis, centred frames.
        x = audio - 0.97 * np.concatenate([[0], audio[:-1]])
        x = np.pad(x, (256, 256))
        u = bare_streamer()
        fr = np.lib.stride_tricks.sliding_window_view(x, 512)[::160] * u.window
        whole = np.log((np.abs(np.fft.rfft(fr, 512)) ** 2).astype(np.float32) @ u.fbank + 2 ** -24)
        i = 0
        for n in rng.integers(50, 2000, size=200):        # uneven WASAPI-like packets
            u._features(audio[i:i + n])
            i += n
            if i >= audio.size:
                break
        n = u.n_feats
        self.assertGreater(n, 290)
        np.testing.assert_allclose(u.feats[:n], whole[:n], rtol=1e-4, atol=1e-3)

    def test_tokens_become_words(self):
        u = bare_streamer()
        out = []
        for tok, t in (("▁he", 0.1), ("llo", 0.2), ("▁wor", 0.3), ("ld", 0.4), ("▁again", 0.5)):
            out += u._token(tok, t)
        self.assertEqual(out, ["hello", "world"])
        self.assertEqual(u.word, "again")

    def test_late_period_attaches_to_the_open_word(self):
        # The model emits a sentence's period only when the next sentence starts.
        u = bare_streamer()
        out = u._token("▁error", 7.9) + u._token(".", 10.6) + u._token("▁I", 10.8)
        self.assertEqual(out, ["error."])
        self.assertEqual(u.word, "I")
        self.assertEqual(u.phrase, 1)          # the 2.7 s gap started a new phrase

    def test_stray_punctuation_without_a_word_is_dropped(self):
        u = bare_streamer()
        self.assertEqual(u._token("▁,", 1.0), [])
        self.assertEqual(u._token("▁so", 1.1), [])
        self.assertEqual(u.word, "so")

    def test_flush_commits_the_open_word(self):
        u = bare_streamer()
        u._token("▁bye", 1.0)
        self.assertEqual(u._flush_pending(force=True), [(["bye"], [], 0)])
        self.assertEqual(u._flush_pending(force=True), [])


if __name__ == "__main__":
    unittest.main()
