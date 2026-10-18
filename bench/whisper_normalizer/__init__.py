"""Whisper's English text normaliser, for scoring WER the way the Open ASR Leaderboard does.

Vendored from openai/whisper (MIT, see LICENSE) at commit 86098128c0b4f24f0e2aa2994de830614b474227,
whisper/normalizers/{english.py,basic.py,english.json}. Local changes are marked
"Live-Subtitles change": no more_itertools or regex dependency, and the spelling map is read as UTF-8.
"""

from .english import EnglishTextNormalizer

__all__ = ["EnglishTextNormalizer"]
