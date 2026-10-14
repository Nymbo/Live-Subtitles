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
        from live_subtitles.captions import PUNCT_CONTEXT, PUNCT_WINDOW

        def fake(words):
            return [words[0].capitalize(), *words[1:]]

        b = CaptionBuilder(max_lines=50, max_chars=90, sentence_breaks=False, punctuate=fake)
        words = [f"w{i}" for i in range(PUNCT_WINDOW + 10)]
        for w in words:
            b.add([w], "", 0.0)
        # Only w0 was ever a true start; later window starts are context, not sentence starts.
        self.assertEqual(b.shown[0], "W0")
        self.assertTrue(all(w.islower() for w in b.shown[1:]))
        self.assertGreater(PUNCT_CONTEXT, 0)

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

    def test_unknown_latency_falls_back(self):
        self.assertEqual(live_model("en", 999).latency_ms, 560)


if __name__ == "__main__":
    unittest.main()
